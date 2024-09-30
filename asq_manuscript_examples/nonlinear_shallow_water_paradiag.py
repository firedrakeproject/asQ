from firedrake.petsc import PETSc

from utils.timing import SolverTimer
from utils.planets import earth
from utils import units

import utils.shallow_water as swe
from utils.shallow_water import galewsky

from math import sqrt
from functools import partial

import argparse
from argparse_formatter import DefaultsAndRawTextFormatter


# get command arguments
parser = argparse.ArgumentParser(
    description='Galewsky testcase for ParaDiag solver using fully implicit SWE solver.',
    epilog="""\
Optional PETSc command line arguments:

   -circulant_alpha :float: The circulant parameter to use in the preconditioner. Default 1e-4.
""",
    formatter_class=DefaultsAndRawTextFormatter
)

parser.add_argument('--ref_level', type=int, default=3, help='Refinement level of icosahedral grid. Total number of cells is 20*4^ref_level.')
parser.add_argument('--base_level', type=int, default=2, help='Base refinement level for multigrid.')
parser.add_argument('--dt', type=float, default=0.5, help='Timestep in hours.')
parser.add_argument('--theta', type=float, default=0.5, help='Parameter for implicit theta method.')
parser.add_argument('--nwindows', type=int, default=1, help='Number of time-windows to solve.')
parser.add_argument('--nslices', type=int, default=1, help='Number of time-slices in the all-at-once system. Must divide the number of MPI ranks exactly.')
parser.add_argument('--slice_length', type=int, default=4, help='Number of timesteps per time-slice. Total number of timesteps in the all-at-once system is nslices*slice_length.')
parser.add_argument('--vtkfile', type=str, default='vtk/galewsky_paradiag', help='Name of output vtk files for the last timestep of each window.')
parser.add_argument('--metrics_dir', type=str, default='metrics/nonlinear_shallow_water', help='Directory to save paradiag output metrics to.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Setting up --- === ###')
PETSc.Sys.Print('')

# This script broadly follows the same structure as the
# linear_shallow_water_paradiag.py script, with the main
# differences being:
#
# - a hierarchy of meshes is required for the multigrid method
# - the galewsky test case is used instead of the gravity waves case:
#   Galewsky et al, 2004, "An initial-value problem for testing
#   numerical models  of the global shallow-water equations"
# - the options parameters specify a multigrid scheme not hybridisation

# time steps

time_partition = tuple((args.slice_length for _ in range(args.nslices)))
window_length = sum(time_partition)
nsteps = args.nwindows*window_length

dt = args.dt*units.hour
H = galewsky.H0
g = earth.Gravity

# parameters for the implicit diagonal solve in step-(b)

from utils.mg import ManifoldTransferManager  # noqa: F401

block_parameters = {
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp': {
        'rtol': 1e-3,
        'max_it': 30,
    },
    'pc_type': 'mg',
    'pc_mg_cycle_type': 'v',
    'pc_mg_type': 'multiplicative',
    'mg': {
        'transfer_manager': f'{__name__}.ManifoldTransferManager',
        'levels': {
            'ksp_type': 'gmres',
            'ksp_max_it': 3,
            'pc_type': 'python',
            'pc_python_type': 'firedrake.PatchPC',
            'patch': {
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
        },
        'coarse': {
            'pc_type': 'python',
            'pc_python_type': 'firedrake.AssembledPC',
            'assembled': {
                'ksp_type': 'preonly',
                'pc_type': 'lu',
                'pc_factor_mat_solver_type': 'mumps',
            },
        },
    }
}

# atol is the same for Newton and (right-preconditioned) Krylov
atol = 1e4
patol = sqrt(window_length)*atol
paradiag_parameters = {
    'snes': {
        'linesearch_type': 'basic',
        'monitor': None,
        'converged_reason': None,
        'atol': patol,
        'rtol': 1e-10,
        'stol': 1e-12,
        'ksp_ew': None,
        'ksp_ew_version': 1,
        'ksp_ew_rtol0': 1e-1,
        'ksp_ew_threshold': 1e-2,
    },
    'ksp_type': 'fgmres',
    'ksp': {
        'monitor': None,
        'converged_rate': None,
        'rtol': 1e-5,
        'atol': patol,
    },
    'pc_type': 'python',
    'pc_python_type': 'asQ.CirculantPC',
    'circulant_block': block_parameters,
    'circulant_alpha': 1e-4,
}

create_mesh = partial(
    swe.create_mg_globe_mesh,
    ref_level=args.ref_level,
    base_level=args.base_level,
    coords_degree=1)

miniapp = swe.ShallowWaterMiniApp(gravity=earth.Gravity,
                                  topography_expression=galewsky.topography_expression,
                                  velocity_expression=galewsky.velocity_expression,
                                  depth_expression=galewsky.depth_expression,
                                  reference_depth=galewsky.H0,
                                  reference_state=True,
                                  create_mesh=create_mesh,
                                  dt=dt, theta=args.theta,
                                  time_partition=time_partition,
                                  paradiag_sparameters=paradiag_parameters,
                                  file_name=args.vtkfile,
                                  record_diagnostics={'cfl': True, 'file': True})

timer = SolverTimer()

fround = lambda x: round(float(x), 2)


def window_preproc(swe_app, pdg, wndw):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'### === --- Calculating time-window {wndw} --- === ###')
    PETSc.Sys.Print('')
    with PETSc.Log.Event("window_preproc.Coll_Barrier"):
        pdg.ensemble.ensemble_comm.Barrier()
    timer.start_timing()


def window_postproc(swe_app, pdg, wndw):
    timer.stop_timing()
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'Window solution time: {timer.times[-1]}')
    PETSc.Sys.Print('')

    if miniapp.layout.is_local(miniapp.save_step):
        nt = (pdg.total_windows - 1)*pdg.ntimesteps + (miniapp.save_step + 1)
        time = float(nt*pdg.aaoform.dt)
        comm = miniapp.ensemble.comm
        PETSc.Sys.Print(f'Maximum CFL = {fround(swe_app.cfl_series[wndw])}', comm=comm)
        PETSc.Sys.Print(f'Hours = {fround(time/units.hour)}', comm=comm)
        PETSc.Sys.Print(f'Days = {fround(time/earth.day)}', comm=comm)


paradiag = miniapp.paradiag
ics = paradiag.solver.aaofunc.initial_condition.copy(deepcopy=True)

# Setup all solver objects. The firedrake DM and options management
# makes it difficult to setup some preconditioners without actually
# calling `solve`, so we just run once to set everything up.
PETSc.Sys.Print('### === --- Setting up solver and prefactoring --- === ###')

with PETSc.Log.Event("warmup_solve"):
    miniapp.solve(nwindows=1,
                  preproc=window_preproc,
                  postproc=window_postproc)
PETSc.Sys.Print('')

# reset
timer.times.clear()
paradiag.reset_diagnostics()
paradiag.aaofunc.assign(ics)

PETSc.Sys.Print('### === --- Calculating parallel solution --- === ###')

with PETSc.Log.Event("timed_solves"):
    miniapp.solve(nwindows=args.nwindows,
                  preproc=window_preproc,
                  postproc=window_postproc)

PETSc.Sys.Print('### === --- Iteration counts --- === ###')
PETSc.Sys.Print('')

from asQ import write_paradiag_metrics
write_paradiag_metrics(paradiag, directory=args.metrics_dir)

nw = paradiag.total_windows
nt = paradiag.total_timesteps
PETSc.Sys.Print(f'windows: {nw}')
PETSc.Sys.Print(f'timesteps: {nt}')
PETSc.Sys.Print('')

# Show the parallel partition sizes.
ensemble = paradiag.ensemble
mesh = miniapp.mesh
W = miniapp.W

PETSc.Sys.Print(f'Total DoFs per window: {W.dim()*window_length}')
PETSc.Sys.Print(f'DoFs per timestep: {W.dim()}')
PETSc.Sys.Print(f'Total number of MPI ranks: {ensemble.global_comm.size}')
PETSc.Sys.Print(f'Number of MPI ranks per timestep: {mesh.comm.size}')
PETSc.Sys.Print(f'DoFs/rank: {W.dim()/mesh.comm.size}')
PETSc.Sys.Print(f'Complex block DoFs/rank: {2*W.dim()/mesh.comm.size}')
PETSc.Sys.Print('')

# paradiag collects a few iteration counts for us
lits = paradiag.linear_iterations
nlits = paradiag.nonlinear_iterations
blits = paradiag.block_iterations.data()

PETSc.Sys.Print(f'Nonlinear iterations: {nlits} | Iterations per window: {nlits/nw}')
PETSc.Sys.Print(f'Linear iterations: {lits} | Iterations per window: {lits/nw}')
PETSc.Sys.Print(f'Total block linear iterations: {blits}')
PETSc.Sys.Print(f'Iterations per block solve: {blits/lits}')
PETSc.Sys.Print(f'Minimum block iterations per solve: {min(blits)/lits}')
PETSc.Sys.Print(f'Maximum block iterations per solve: {max(blits)/lits}')
PETSc.Sys.Print('')

ensemble.global_comm.Barrier()
if miniapp.layout.is_local(miniapp.save_step):
    PETSc.Sys.Print(f'Maximum CFL = {max(miniapp.cfl_series)}')
    PETSc.Sys.Print(f'Minimum CFL = {min(miniapp.cfl_series)}')
    PETSc.Sys.Print('')
ensemble.global_comm.Barrier()

PETSc.Sys.Print(timer.string(timesteps_per_solve=window_length,
                             total_iterations=lits, ndigits=5))
PETSc.Sys.Print('')
