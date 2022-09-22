
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
                                comm=ensemble.comm)
x = fd.SpatialCoordinate(mesh)

# shallow water equation function spaces (velocity and depth)
W = swe.default_function_space(mesh, degree=args.degree)

# parameters
gravity = earth.Gravity

topography = galewsky.topography_expression(*x)
coriolis = swe.earth_coriolis_expression(*x)

# initial conditions
w_initial = fd.Function(W)
u_initial, h_initial = w_initial.split()

u_initial.project(galewsky.velocity_expression(*x))
h_initial.project(galewsky.depth_expression(*x))

# current and next timestep
w0 = fd.Function(W).assign(w_initial)
w1 = fd.Function(W).assign(w_initial)


# shallow water equation forms
def form_function(u, h, v, q):
    return swe.nonlinear.form_function(mesh,
                                       gravity,
                                       topography,
                                       coriolis,
                                       u, h, v, q)


def form_mass(u, h, v, q):
    return swe.nonlinear.form_mass(mesh, u, h, v, q)


# solver parameters for the implicit solve
serial_sparameters = {
    'snes': {
        # 'monitor': None,
        # 'converged_reason': None,
        'atol': 1e-0,
        'rtol': 1e-12,
        'stol': 1e-12,
    },
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp': {
        'atol': 1e-8,
        'rtol': 1e-8,
        # 'monitor': None,
        # 'converged_reason': None
    },
    'pc_type': 'mg',
    'pc_mg_cycle_type': 'w',
    'pc_mg_type': 'multiplicative',
    'mg': {
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

# parameters for the implicit diagonal solve in step-(b)
block_sparameters = {
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp': {
        'atol': 1e-8,
        'rtol': 1e-8,
    },
    'pc_type': 'mg',
    'pc_mg_cycle_type': 'v',
    'pc_mg_type': 'multiplicative',
    'mg': {
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
        # 'monitor': None,
        # 'converged_reason': None,
        'atol': 1e-0,
        'rtol': 1e-12,
        'stol': 1e-12,
    },
    'mat_type': 'matfree',
    'ksp_type': 'preonly',
    'ksp': {
        # 'monitor': None,
        # 'converged_reason': None,
    },
    'pc_type': 'python',
    'pc_python_type': 'asQ.DiagFFTPC'
}

for i in range(sum(time_partition)):
    parallel_sparameters['diagfft_'+str(i)+'_'] = block_sparameters

block_ctx = {}
transfer_managers = []
for _ in range(time_partition[ensemble.ensemble_comm.rank]):
    tm = mg.manifold_transfer_manager(W)
    transfer_managers.append(tm)
block_ctx['diag_transfer_managers'] = transfer_managers

miniapp = ComparisonMiniapp(ensemble, time_partition,
                            form_mass,
                            form_function,
                            w_initial,
                            dt, args.theta, args.alpha,
                            serial_sparameters,
                            parallel_sparameters,
                            circ=None, block_ctx=block_ctx)

miniapp.serial_app.nlsolver.set_transfer_manager(
    mg.manifold_transfer_manager(W))

rank = ensemble.ensemble_comm.rank
norm0 = fd.norm(w_initial)


def preproc(serial_app, paradiag, wndw):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'=== --- Time window {wndw} --- ===')
    PETSc.Sys.Print('')


def serial_postproc(app, it, t):
    if not args.print_norms:
        return
    if rank == 0:
        PETSc.Sys.Print(f'Rank {rank}: Serial timestep {it} norm {fd.norm(app.w0)/norm0}', comm=ensemble.comm)
    ensemble.global_comm.Barrier()
    return


def parallel_postproc(pdg, wndw):
    if not args.print_norms:
        return
    aaos = miniapp.paradiag.aaos
    for step in range(aaos.nlocal_timesteps):
        it = aaos.shift_index(step, from_range='slice', to_range='window')
        w = aaos.get_timestep(step)
        PETSc.Sys.Print(f'Rank {rank}: Parallel timestep {it} norm {fd.norm(w)/norm0}', comm=ensemble.comm)
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
