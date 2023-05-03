
import numpy as np
import firedrake as fd
from petsc4py import PETSc
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
parser.add_argument('--filename', type=str, default='w5diag', help='Name of output vtk files')
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

def form_function(u, h, v, q):
    return swe.nonlinear.form_function(mesh, earth.Gravity, b, f, u, h, v, q)


def form_mass(u, h, v, q):
    return swe.nonlinear.form_mass(mesh, u, h, v, q)


def linearised_function(u, h, v, q):
    return swe.linear.form_function(mesh, earth.Gravity, H, f, u, h, v, q)


def linearised_mass(u, h, v, q):
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
        'pc_type': 'fieldsplit',
        'pc_fieldsplit_type': 'schur',
        'pc_fieldsplit_detect_saddle_point': None,
        'pc_fieldsplit_schur_fact_type': 'full',
        'pc_fieldsplit_schur_precondition': 'full',
        'fieldsplit_ksp_type': 'preonly',
        'fieldsplit_pc_type': 'lu',
    }
}

mg_parameters = {
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
        'max_it': 60
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
    },
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp': {
        'monitor': None,
        'converged_reason': None,
        'rtol': 1e-5,
        'atol': atol,
    },
    'pc_type': 'python',
    'pc_python_type': 'asQ.DiagFFTPC',
    'diagfft_state': 'reference',
    'diagfft_linearisation': 'consistent',
    'aaos_jacobian_state': 'reference',
    'aaos_jacobian_linearisation': 'consistent',
}

# reference conditions
wref = w0.copy(deepcopy=True)
uref, href = wref.subfunctions[:]
# uref.assign(0)
# href.assign(H)

Print('### === --- Calculating parallel solution --- === ###')
# Print('')

sparameters_diag['diagfft_block'] = sparameters

# non-petsc information for block solve
block_ctx = {}

# mesh transfer operators
transfer_managers = []
nlocal_timesteps = time_partition[ensemble.ensemble_comm.rank]
for _ in range(nlocal_timesteps):
    tm = mg.manifold_transfer_manager(W)
    transfer_managers.append(tm)

block_ctx['diagfft_transfer_managers'] = transfer_managers

PD = asQ.paradiag(ensemble=ensemble,
                  form_function=form_function,
                  form_mass=form_mass,
                  linearised_function=linearised_function,
                  linearised_mass=linearised_mass,
                  w0=w0, reference_state=wref,
                  dt=dt, theta=0.5,
                  alpha=args.alpha,
                  time_partition=time_partition,
                  solver_parameters=sparameters_diag,
                  circ=None, block_ctx=block_ctx)


def window_preproc(pdg, wndw):
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
def window_postproc(pdg, wndw):
    errors = np.zeros((window_length, 2))
    local_errors = np.zeros_like(errors)

    # collect errors for this slice
    def for_each_callback(window_index, slice_index, w):
        nonlocal local_errors
        local_errors[window_index, :] = steady_state_test(w)

    pdg.aaos.for_each_timestep(for_each_callback)

    # collect and print errors for full window
    ensemble.ensemble_comm.Reduce(local_errors, errors, root=0)
    if pdg.time_rank == 0:
        for window_index in range(window_length):
            timestep = wndw*window_length + window_index
            uerr = errors[window_index, 0]
            herr = errors[window_index, 1]
            Print(f"timestep={timestep}, uerr={uerr}, herr={herr}",
                  comm=ensemble.comm)


PD.solve(nwindows=args.nwindows,
         preproc=window_preproc,
         postproc=window_postproc)
