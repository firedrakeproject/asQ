
import firedrake as fd
from firedrake.petsc import PETSc
import asQ

from utils import mg
from utils import units
from utils.planets import earth
import utils.shallow_water as swe
import utils.shallow_water.williamson1992.case2 as case2

Print = PETSc.Sys.Print

PETSc.Sys.popErrorHandler()

# get command arguments
import argparse
parser = argparse.ArgumentParser(
    description='Williamson 2 testcase for ParaDiag solver using fully implicit SWE solver.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--base_level', type=int, default=1, help='Base refinement level of icosahedral grid for MG solve.')
parser.add_argument('--ref_level', type=int, default=2, help='Refinement level of icosahedral grid.')
parser.add_argument('--nwindows', type=int, default=1, help='Number of time-windows.')
parser.add_argument('--nslices', type=int, default=2, help='Number of time-slices per time-window.')
parser.add_argument('--slice_length', type=int, default=2, help='Number of timesteps per time-slice.')
parser.add_argument('--alpha', type=float, default=0.0001, help='Circulant coefficient.')
parser.add_argument('--dt', type=float, default=0.5, help='Timestep in hours.')
parser.add_argument('--coords_degree', type=int, default=1, help='Degree of polynomials for sphere mesh approximation.')
parser.add_argument('--filename', type=str, default='williamson2', help='Name of output vtk files')
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space).')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    Print(args)

Print('')
Print('### === --- Setting up --- === ###')
Print('')

# time steps

time_partition = tuple((args.slice_length for _ in range(args.nslices)))
window_length = sum(time_partition)
nsteps = args.nwindows*window_length

dt = args.dt*units.hour
# multigrid mesh set up

ensemble = asQ.create_ensemble(time_partition)

# mesh set up
mesh = swe.create_mg_globe_mesh(comm=ensemble.comm,
                                base_level=args.base_level,
                                ref_level=args.ref_level,
                                coords_degree=1)

x = fd.SpatialCoordinate(mesh)

# Mixed function space for velocity and depth
W = swe.default_function_space(mesh, degree=args.degree)
V1, V2 = W.subfunctions[:]

# initial conditions
w0 = fd.Function(W)
un, hn = w0.subfunctions[:]

f = case2.coriolis_expression(*x)
b = case2.topography_function(*x, V2, name="Topography")
H = case2.H0

un.project(case2.velocity_expression(*x))
etan = case2.elevation_function(*x, V2, name="Elevation")
hn.assign(H + etan - b)


# nonlinear swe forms

def form_function(u, h, v, q, t):
    return swe.nonlinear.form_function(mesh, earth.Gravity, b, f, u, h, v, q, t)


def form_mass(u, h, v, q):
    return swe.nonlinear.form_mass(mesh, u, h, v, q)


def linear_function(u, h, v, q, t):
    return swe.linear.form_function(mesh, earth.Gravity, H, f, u, h, v, q, t)


def linear_mass(u, h, v, q):
    return swe.linear.form_mass(mesh, u, h, v, q)


# parameters for the implicit diagonal solve in step-(b)
patch_parameters = {
    'pc_patch': {
        'save_operators': True,
        'partition_of_unity': True,
        'sub_mat_type': 'seqdense',
        'construct_dim': 0,
        'construct_type': 'vanka',
        'local_type': 'additive',
        'precompute_element_tensors': True,
        'symmetrise_sweep': False
    },
    'sub': {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_shift_type': 'nonzero',
    }
}

from utils.mg import ManifoldTransferManager  # noqa: F401
mg_parameters = {
    'transfer_manager': f'{__name__}.ManifoldTransferManager',
    'levels': {
        'ksp_type': 'gmres',
        'ksp_max_it': 5,
        'pc_type': 'python',
        'pc_python_type': 'firedrake.PatchPC',
        'patch': patch_parameters
    },
    'coarse': {
        'pc_type': 'python',
        'pc_python_type': 'firedrake.AssembledPC',
        'assembled_pc_type': 'lu',
        'assembled_pc_factor_mat_solver_type': 'mumps'
    }
}

sparameters = {
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp': {
        'atol': 1e-5,
        'rtol': 1e-5,
        'max_it': 50,
    },
    'pc_type': 'mg',
    'pc_mg_cycle_type': 'w',
    'pc_mg_type': 'multiplicative',
    'mg': mg_parameters
}

atol = 1e0
sparameters_diag = {
    'snes': {
        'linesearch_type': 'basic',
        'monitor': None,
        'converged_reason': None,
        'atol': atol,
        'rtol': 1e-10,
        'stol': 1e-12,
        'ksp_ew': None,
        'ksp_ew_version': 1,
        'ksp_ew_threshold': 1e-2,
    },
    'mat_type': 'matfree',
    'ksp_type': 'preonly',
    'ksp': {
        'monitor': None,
        'converged_reason': None,
        'rtol': 1e-5,
        'atol': atol,
    },
    'pc_type': 'python',
    'pc_python_type': 'asQ.DiagFFTPC',
    'diagfft_alpha': args.alpha,
    'diagfft_state': 'window',
    'diagfft_linearisation': 'consistent',
    'aaos_jacobian_state': 'current',
    'aaos_jacobian_linearisation': 'consistent',
}

# reference conditions
wref = w0.copy(deepcopy=True)
uref, href = wref.subfunctions[:]
# uref.assign(0)
# href.assign(H)

Print('### === --- Calculating parallel solution --- === ###')
# Print('')

for i in range(window_length):
    sparameters_diag['diagfft_block_'+str(i)+'_'] = sparameters

# non-petsc information for block solve
appctx = {}

# mesh transfer operators
transfer_managers = []
nlocal_timesteps = time_partition[ensemble.ensemble_comm.rank]
for _ in range(nlocal_timesteps):
    transfer_managers.append(mg.ManifoldTransferManager())

appctx['diagfft_transfer_managers'] = transfer_managers

pdg = asQ.Paradiag(ensemble=ensemble,
                   form_function=form_function,
                   form_mass=form_mass,
                   jacobian_function=linear_function,
                   jacobian_mass=linear_mass,
                   ics=w0, reference_state=wref,
                   dt=dt, theta=0.5,
                   time_partition=time_partition,
                   solver_parameters=sparameters_diag,
                   appctx=appctx)


def window_preproc(pdg, wndw, rhs):
    Print('')
    Print(f'### === --- Calculating time-window {wndw} --- === ###')
    Print('')


# check against initial conditions
wcheck = w0.copy(deepcopy=True)
ucheck = wcheck.subfunctions[0]
hcheck = wcheck.subfunctions[1]
hcheck.assign(hcheck - H + b)


def steady_state_test(w):
    up = w.subfunctions[0]
    hp = w.subfunctions[1]
    hp.assign(hp - H + b)

    uerr = fd.errornorm(ucheck, up)/fd.norm(ucheck)
    herr = fd.errornorm(hcheck, hp)/fd.norm(hcheck)

    return uerr, herr


# check each timestep against steady state


def window_postproc(pdg, wndw, rhs):
    uerrors = asQ.SharedArray(time_partition, comm=ensemble.ensemble_comm)
    herrors = asQ.SharedArray(time_partition, comm=ensemble.ensemble_comm)
    for i in range(pdg.nlocal_timesteps):
        uerr, herr = steady_state_test(pdg.aaofunc[i])
        uerrors.dlocal[i] = uerr
        herrors.dlocal[i] = herr
    uerrors.synchronise()
    herrors.synchronise()

    for i in range(pdg.ntimesteps):
        timestep = wndw*window_length + i
        uerr = uerrors.dglobal[i]
        herr = herrors.dglobal[i]
        Print(f"timestep={timestep}, uerr={uerr}, herr={herr}")


pdg.solve(nwindows=args.nwindows,
          preproc=window_preproc,
          postproc=window_postproc)
