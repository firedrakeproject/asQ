
import firedrake as fd
from firedrake.petsc import PETSc
import asQ

from math import sqrt
from utils import mg
from utils import units
from utils.planets import earth
from utils.diagnostics import convective_cfl_calculator
import utils.shallow_water as swe
from utils.shallow_water import galewsky

Print = PETSc.Sys.Print

PETSc.Sys.popErrorHandler()

# get command arguments
import argparse
parser = argparse.ArgumentParser(
    description='Galewsky test case using a sliding window',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--base_level', type=int, default=1, help='Base refinement level of icosahedral grid for MG solve.')
parser.add_argument('--ref_level', type=int, default=2, help='Refinement level of icosahedral grid.')
parser.add_argument('--nwindows', type=int, default=1, help='Number of time-windows.')
parser.add_argument('--nslices', type=int, default=2, help='Number of time-slices per time-window.')
parser.add_argument('--slice_length', type=int, default=2, help='Number of timesteps per time-slice.')
parser.add_argument('--alpha', type=float, default=1e-4, help='Circulant coefficient.')
parser.add_argument('--dt', type=float, default=0.5, help='Timestep in hours.')
parser.add_argument('--theta', type=float, default=0.5, help='Implicit parameter.')
parser.add_argument('--atol', type=float, default=1e0, help='Average absolute tolerance for each timestep.')
parser.add_argument('--filename', type=str, default='galewsky', help='Name of output vtk files')
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
V1, V2 = W.subfunctions

# initial conditions
ics = fd.Function(W)
un, hn = ics.subfunctions

f = galewsky.coriolis_expression(*x)
b = galewsky.topography_expression(*x)
H = galewsky.H0

un.project(galewsky.velocity_expression(*x))
hn.project(galewsky.depth_expression(*x))


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

mg_parameters = {
    'levels': {
        'ksp_type': 'gmres',
        'ksp_max_it': 4,
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

block_sparameters = {
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp': {
        'atol': 1e-10,
        'rtol': 0.1*args.alpha,
        'max_it': 50,
    },
    'pc_type': 'mg',
    'pc_mg_cycle_type': 'v',
    'pc_mg_type': 'multiplicative',
    'mg': mg_parameters
}

atol = args.atol
patol = sqrt(sum(time_partition))*atol
sparameters = {
    'snes': {
        'linesearch_type': 'basic',
        # 'monitor': None,
        'converged_reason': None,
        'atol': patol,
        'rtol': 1e-12,
        'stol': 1e-12,
        'ksp_ew': None,
        'ksp_ew_version': 1,
        'ksp_ew_threshold': 1e-2,
    },
    'mat_type': 'matfree',
    'ksp_type': 'preonly',
    'ksp': {
        # 'monitor': None,
        # 'converged_reason': None,
        'rtol': 1e-5,
        'atol': patol,
    },
    'pc_type': 'python',
    'pc_python_type': 'asQ.DiagFFTPC',
    'diagfft_alpha': args.alpha,
    'diagfft_state': 'window',
}

Print('### === --- Calculating parallel solution --- === ###')
Print('')

for i in range(window_length):
    sparameters['diagfft_block_'+str(i)+'_'] = block_sparameters

# non-petsc information for block solve
appctx = {}

# mesh transfer operators
transfer_managers = []
nlocal_timesteps = time_partition[ensemble.ensemble_comm.rank]
for _ in range(nlocal_timesteps):
    tm = mg.manifold_transfer_manager(W)
    transfer_managers.append(tm)

appctx['diagfft_transfer_managers'] = transfer_managers

# set up all-at-once objects

aaofunc = asQ.AllAtOnceFunction(ensemble, time_partition, W)
aaofunc.assign(ics)

# timestep residual evaluations
res_array = asQ.SharedArray(time_partition, comm=ensemble.ensemble_comm)


def post_function_callback(aaosolver, X, F):
    res = aaosolver.aaoform.F
    for i in range(res.nlocal_timesteps):
        with res[i].dat.vec_ro as rvec:
            res_array.dlocal[i] = rvec.norm()
    res_array.synchronise()
    # Print('')
    PETSc.Sys.Print([f"{r:.4e}" for r in res_array.data()])
    # Print('')


aaoform = asQ.AllAtOnceForm(aaofunc, dt, args.theta,
                            form_mass, form_function)

aaosolver = asQ.AllAtOnceSolver(
    aaoform, aaofunc,
    solver_parameters=sparameters,
    appctx=appctx,
    post_function_callback=post_function_callback)

is_last_slice = aaofunc.layout.is_local(-1)

if is_last_slice:
    cfl_calc = convective_cfl_calculator(mesh)

    def max_cfl(u, dt):
        with cfl_calc(u, dt).dat.vec_ro as v:
            return v.max()[1]

    uout = fd.Function(V1, name="velocity")
    hout = fd.Function(V2, name="depth")
    ofile = fd.File(f"output/vtk/{args.filename}.pvd", comm=ensemble.comm)

    def write_file(w, t):
        uout.assign(w.subfunctions[0])
        hout.assign(w.subfunctions[1])
        ofile.write(uout, hout, t=0)

    write_file(ics, 0)


def window_preproc(wndw):
    Print(f'### === --- Calculating time-window {wndw} --- === ###')
    Print('')


def window_postproc(wndw):
    if is_last_slice:
        cfl = max_cfl(aaofunc[-1].subfunctions[0], dt)
        time = aaoform.time[-1]
        hours = float(time/units.hour)
        Print('', comm=ensemble.comm)
        PETSc.Sys.Print(f'Maximum CFL = {cfl}', comm=ensemble.comm)
        PETSc.Sys.Print(f'Hours = {hours}', comm=ensemble.comm)
        Print('', comm=ensemble.comm)
        write_file(aaofunc[-1], t=hours)


for wndw in range(args.nwindows):
    window_preproc(wndw)
    aaosolver.solve()
    window_postproc(wndw)

    if wndw != args.nwindows-1:
        aaofunc.bcast_field(-1, aaofunc.initial_condition)
        aaofunc.assign(aaofunc.initial_condition)
        aaoform.time_update()
        aaosolver.jacobian_form.time_update()
