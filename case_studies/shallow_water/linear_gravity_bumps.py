import firedrake as fd
from firedrake.petsc import PETSc

from math import sqrt
from utils.timing import SolverTimer
from utils import units
from utils.planets import earth
import utils.shallow_water as swe
import utils.shallow_water.gravity_bumps as case

from functools import partial

PETSc.Sys.popErrorHandler()

# get command arguments
import argparse
parser = argparse.ArgumentParser(
    description='Gravity wave testcase for ParaDiag solver using fully implicit linear SWE solver.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--ref_level', type=int, default=2, help='Refinement level of icosahedral grid.')
parser.add_argument('--base_level', type=int, default=1, help='Refinement level of the coarsest grid.')
parser.add_argument('--nwindows', type=int, default=1, help='Number of time-windows.')
parser.add_argument('--nslices', type=int, default=2, help='Number of time-slices per time-window.')
parser.add_argument('--slice_length', type=int, default=2, help='Number of timesteps per time-slice.')
parser.add_argument('--alpha', type=float, default=1e-4, help='Circulant coefficient.')
parser.add_argument('--dt', type=float, default=0.05, help='Timestep in hours.')
parser.add_argument('--filename', type=str, default='gravity_waves', help='Name of output vtk files.')
parser.add_argument('--metrics_dir', type=str, default='metrics', help='Directory to save paradiag metrics to.')
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
window_length = sum(time_partition)
nsteps = args.nwindows*window_length

dt = args.dt*units.hour

# parameters for the implicit diagonal solve in step-(b)

from utils.hybridisation import HybridisedSCPC  # noqa: F401
linear_snes_params = {
    'lag_jacobian': -2,
    'lag_jacobian_persists': None,
    'lag_preconditioner': -2,
    'lag_preconditioner_persists': None,
}

hybridscpc_parameters = {
    "snes": linear_snes_params,
    "ksp_type": 'preonly',
    "mat_type": "matfree",
    "pc_type": "python",
    "pc_python_type": f"{__name__}.HybridisedSCPC",
    "hybridscpc_snes": linear_snes_params,
    "hybridscpc_condensed_field": {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
        'snes': linear_snes_params  # reuse factorisation
    }
}

rtol = 1e-11
atol = 1e-10
patol = sqrt(window_length)*atol
sparameters_diag = {
    'snes_type': 'ksponly',
    'snes': linear_snes_params,
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp': {
        'monitor': None,
        'converged_rate': None,
        'rtol': rtol,
        'atol': patol,
    },
    'pc_type': 'python',
    'pc_python_type': 'asQ.CirculantPC',
    'circulant_alpha': args.alpha,
    'circulant_state': 'linear',
    'aaos_jacobian_state': 'linear',
}

comm_size = fd.COMM_WORLD.size // args.nslices
block_id = fd.COMM_WORLD.rank // comm_size
sparameters_diag[f'circulant_block_{block_id}'] = hybridscpc_parameters
# for i in range(window_length):
#     sparameters_diag['circulant_block_'+str(i)+'_'] = hybridscpc_parameters

create_mesh = partial(
    swe.create_mg_globe_mesh,
    ref_level=args.ref_level,
    base_level=args.base_level,
    coords_degree=1)  # remove coords degree once UFL issue with gradient of cell normals fixed

PETSc.Sys.Print('### === --- Calculating parallel solution --- === ###')

miniapp = swe.ShallowWaterMiniApp(gravity=earth.Gravity,
                                  topography_expression=case.topography_expression,
                                  velocity_expression=case.velocity_expression,
                                  depth_expression=case.depth_expression,
                                  reference_depth=case.H,
                                  create_mesh=create_mesh,
                                  linear=True,
                                  dt=dt, theta=0.5,
                                  time_partition=time_partition,
                                  paradiag_sparameters=sparameters_diag,
                                  record_diagnostics={'cfl': False, 'file': False})

paradiag = miniapp.paradiag

timer = SolverTimer()
solver_times = []
options_times = []


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

    solver_times.append(pdg.solver._timing_solve)
    options_times.append(pdg.solver._timing_options)

    if pdg.layout.is_local(miniapp.save_step):
        nt = (pdg.total_windows - 1)*pdg.ntimesteps + (miniapp.save_step + 1)
        time = nt*pdg.aaoform.dt
        comm = miniapp.ensemble.comm
        PETSc.Sys.Print('', comm=comm)
        PETSc.Sys.Print(f'Hours = {float(time/units.hour)}', comm=comm)
        PETSc.Sys.Print(f'Days = {float(time/earth.day)}', comm=comm)
        PETSc.Sys.Print('', comm=comm)


ics = paradiag.solver.aaofunc.initial_condition.copy(deepcopy=True)

with PETSc.Log.Event("warmup_solve"):
    miniapp.solve(nwindows=1,
                  preproc=window_preproc,
                  postproc=window_postproc)

# paradiag.solver.aaofunc.assign(ics)
paradiag.reset_diagnostics()
aaofunc = paradiag.solver.aaofunc
aaofunc.bcast_field(-1, aaofunc.initial_condition)
aaofunc.assign(aaofunc.initial_condition)

with PETSc.Log.Event("timed_solves"):
    miniapp.solve(nwindows=args.nwindows,
                  preproc=window_preproc,
                  postproc=window_postproc)

PETSc.Sys.Print('### === --- Iteration counts --- === ###')

from asQ import write_paradiag_metrics
write_paradiag_metrics(paradiag, directory=args.metrics_dir)

PETSc.Sys.Print('')

nw = paradiag.total_windows
nt = paradiag.total_timesteps
PETSc.Sys.Print(f'windows: {nw}')
PETSc.Sys.Print(f'timesteps: {nt}')
PETSc.Sys.Print('')

lits = paradiag.linear_iterations
nlits = paradiag.nonlinear_iterations
blits = paradiag.block_iterations.data(deepcopy=False)

PETSc.Sys.Print(f'linear iterations: {lits} | iterations per window: {lits/nw}')
PETSc.Sys.Print(f'nonlinear iterations: {nlits} | iterations per window: {nlits/nw}')
PETSc.Sys.Print(f'block linear iterations: {blits} | iterations per block solve: {blits/lits}')
PETSc.Sys.Print('')

mesh = miniapp.mesh
W = miniapp.W
PETSc.Sys.Print(f'DoFs per timestep: {W.dim()}')
PETSc.Sys.Print(f'Number of MPI ranks per timestep: {mesh.comm.size}')
PETSc.Sys.Print(f'DoFs/rank: {W.dim()/mesh.comm.size}')
PETSc.Sys.Print(f'Block DoFs/rank: {2*W.dim()/mesh.comm.size}')
PETSc.Sys.Print('')

if timer.ntimes() > 1:
    timer.times = timer.times[1:]

PETSc.Sys.Print(timer.string(timesteps_per_solve=window_length,
                             total_iterations=lits, ndigits=5))
PETSc.Sys.Print('')
PETSc.Sys.Print(f'solver_times = {solver_times}')
PETSc.Sys.Print(f'options_times = {options_times}')
