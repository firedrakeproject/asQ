
import firedrake as fd
from firedrake.petsc import PETSc

import asQ

from utils import units
from utils import mg
from utils.planets import earth
import utils.shallow_water as swe
from utils.shallow_water import galewsky

from utils.serial import ComparisonMiniapp

PETSc.Sys.popErrorHandler()

# get command arguments
import argparse
parser = argparse.ArgumentParser(
    description='Compare the serial and parallel solutions to the Galewsky testcase.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--ref_level', type=int, default=2, help='Refinement level of icosahedral grid.')
parser.add_argument('--nwindows', type=int, default=1, help='Number of time-windows.')
parser.add_argument('--nslices', type=int, default=2, help='Number of time-slices per time-window.')
parser.add_argument('--slice_length', type=int, default=2, help='Number of timesteps per time-slice.')
parser.add_argument('--alpha', type=float, default=0.0001, help='Circulant coefficient.')
parser.add_argument('--dt', type=float, default=0.5, help='Timestep in hours.')
parser.add_argument('--degree', type=float, default=swe.default_degree(), help='Degree of the depth function space.')
parser.add_argument('--theta', type=float, default=0.5, help='Parameter for implicit theta method. 0.5 for trapezium rule, 1 for backwards Euler.')
parser.add_argument('--filename', type=str, default='galewsky', help='Name of output vtk files')
parser.add_argument('--print_norms', type=bool, default=False, help='Print the norm of each timestep')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Setting up --- === ###')
PETSc.Sys.Print('')

# time steps

time_partition = [args.slice_length for _ in range(args.nslices)]

dt = args.dt*units.hour

ensemble = asQ.create_ensemble(time_partition)

# icosahedral mg mesh
mesh = swe.create_mg_globe_mesh(ref_level=args.ref_level,
                                comm=ensemble.comm,
                                coords_degree=1)
x = fd.SpatialCoordinate(mesh)

# shallow water equation function spaces (velocity and depth)
W = swe.default_function_space(mesh, degree=args.degree)

# parameters
gravity = earth.Gravity

topography = galewsky.topography_expression(*x)
coriolis = swe.earth_coriolis_expression(*x)

# initial conditions
w_initial = fd.Function(W)
u_initial = w_initial.subfunctions[0]
h_initial = w_initial.subfunctions[1]

u_initial.project(galewsky.velocity_expression(*x))
h_initial.project(galewsky.depth_expression(*x))


# shallow water equation forms
def form_function(u, h, v, q, t):
    return swe.nonlinear.form_function(mesh,
                                       gravity,
                                       topography,
                                       coriolis,
                                       u, h, v, q, t)


def form_mass(u, h, v, q):
    return swe.nonlinear.form_mass(mesh, u, h, v, q)


# solver parameters for the implicit solve
from utils.mg import ManifoldTransferManager  # noqa: F401
atol = 1e4
serial_sparameters = {
    'snes': {
        'monitor': None,
        'converged_reason': None,
        'atol': atol,
        'rtol': 1e-12,
        'stol': 1e-12,
    },
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp': {
        'atol': atol,
        'rtol': 1e-8,
    },
    'pc_type': 'mg',
    'pc_mg_cycle_type': 'w',
    'pc_mg_type': 'multiplicative',
    'mg': {
        'transfer_manager': f'{__name__}.ManifoldTransferManager',
        'levels': {
            'ksp_type': 'gmres',
            'ksp_max_it': 5,
            'pc_type': 'python',
            'pc_python_type': 'firedrake.PatchPC',
            'patch': {
                'pc_patch_save_operators': True,
                'pc_patch_partition_of_unity': True,
                'pc_patch_sub_mat_type': 'seqdense',
                'pc_patch_construct_dim': 0,
                'pc_patch_construct_type': 'vanka',
                'pc_patch_local_type': 'additive',
                'pc_patch_precompute_element_tensors': True,
                'pc_patch_symmetrise_sweep': False,
                'sub_ksp_type': 'preonly',
                'sub_pc_type': 'lu',
                'sub_pc_factor_shift_type': 'nonzero',
            },
        },
        'coarse': {
            'pc_type': 'python',
            'pc_python_type': 'firedrake.AssembledPC',
            'assembled_pc_type': 'lu',
            'assembled_pc_factor_mat_solver_type': 'mumps',
        },
    }
}

if ensemble.ensemble_comm.rank == 0:
    serial_sparameters['snes']['monitor'] = None
    serial_sparameters['snes']['converged_reason'] = None
    serial_sparameters['ksp']['monitor'] = None
    serial_sparameters['ksp']['converged_reason'] = None

# parameters for the implicit diagonal solve in step-(b)
block_sparameters = {
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp': {
        'atol': 1e-8,
        'rtol': 1e-4,
    },
    'pc_type': 'mg',
    'pc_mg_cycle_type': 'v',
    'pc_mg_type': 'multiplicative',
    'mg': {
        'transfer_manager': f'{__name__}.ManifoldTransferManager',
        'levels': {
            'ksp_type': 'gmres',
            'ksp_max_it': 5,
            'pc_type': 'python',
            'pc_python_type': 'firedrake.PatchPC',
            'patch': {
                'pc_patch_save_operators': True,
                'pc_patch_partition_of_unity': True,
                'pc_patch_sub_mat_type': 'seqdense',
                'pc_patch_construct_codim': 0,
                'pc_patch_construct_type': 'vanka',
                'pc_patch_local_type': 'additive',
                'pc_patch_precompute_element_tensors': True,
                'pc_patch_symmetrise_sweep': False,
                'sub_ksp_type': 'preonly',
                'sub_pc_type': 'lu',
                'sub_pc_factor_shift_type': 'nonzero',
            },
        },
        'coarse': {
            'pc_type': 'python',
            'pc_python_type': 'firedrake.AssembledPC',
            'assembled_pc_type': 'lu',
            'assembled_pc_factor_mat_solver_type': 'mumps',
        },
    }
}

parallel_sparameters = {
    'snes': {
        'linesearch_type': 'basic',
        'monitor': None,
        'converged_reason': None,
        'atol': atol,
        'rtol': 1e-12,
        'stol': 1e-12,
        'ksp_ew': None,
        'ksp_ew_version': 1,
        'ksp_ew_threshold': 1e-2,
    },
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp': {
        'monitor': None,
        'converged_reason': None,
        'atol': atol,
    },
    'pc_type': 'python',
    'pc_python_type': 'asQ.CirculantPC',
    'diagfft_alpha': args.alpha
}

parallel_sparameters['diagfft_block_'] = block_sparameters

appctx = {}
transfer_managers = []
for _ in range(time_partition[ensemble.ensemble_comm.rank]):
    transfer_managers.append(mg.ManifoldTransferManager())

appctx['diagfft_transfer_managers'] = transfer_managers

miniapp = ComparisonMiniapp(ensemble, time_partition,
                            form_mass=form_mass,
                            form_function=form_function,
                            w_initial=w_initial,
                            dt=dt, theta=args.theta,
                            serial_sparameters=serial_sparameters,
                            parallel_sparameters=parallel_sparameters,
                            appctx=appctx)

miniapp.serial_app.nlsolver.set_transfer_manager(
    mg.ManifoldTransferManager())

rank = ensemble.ensemble_comm.rank
norm0 = fd.norm(w_initial)


def preproc(serial_app, paradiag, wndw):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'### === --- Time window {wndw} --- === ###')
    PETSc.Sys.Print('')
    PETSc.Sys.Print('=== --- Parallel solve --- ===')
    PETSc.Sys.Print('')


def serial_postproc(app, it, t):
    if not args.print_norms:
        return
    if rank == 0:
        PETSc.Sys.Print(f'Rank {rank}: Serial timestep {it} norm {fd.norm(app.w0)/norm0}', comm=ensemble.comm)
    ensemble.global_comm.Barrier()
    return


def parallel_postproc(pdg, wndw, rhs):
    if args.print_norms:
        aaofunc = miniapp.paradiag.aaofunc
        for step in range(aaofunc.nlocal_timesteps):
            it = aaofunc.transform_index(step, from_range='slice', to_range='window')
            w = aaofunc[step]
            PETSc.Sys.Print(f'Rank {rank}: Parallel timestep {it} norm {fd.norm(w)/norm0}', comm=ensemble.comm)
    PETSc.Sys.Print('')
    PETSc.Sys.Print('=== --- Serial solve --- ===')
    PETSc.Sys.Print('')
    return


PETSc.Sys.Print('### === --- Timestepping loop --- === ###')

errors = miniapp.solve(nwindows=args.nwindows,
                       preproc=preproc,
                       serial_postproc=serial_postproc,
                       parallel_postproc=parallel_postproc)

PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Errors --- === ###')

for it, err in enumerate(errors):
    PETSc.Sys.Print(f'Timestep {it} error: {err/norm0}')
