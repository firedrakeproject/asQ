from firedrake.petsc import PETSc

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
parser.add_argument('--nwindows', type=int, default=1, help='Number of time-windows.')
parser.add_argument('--nslices', type=int, default=2, help='Number of time-slices per time-window.')
parser.add_argument('--slice_length', type=int, default=2, help='Number of timesteps per time-slice.')
parser.add_argument('--alpha', type=float, default=0.0001, help='Circulant coefficient.')
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
        'max_it': 60
    },
    'pc_type': 'mg',
    'pc_mg_cycle_type': 'w',
    'pc_mg_type': 'multiplicative',
    'mg': mg_parameters
}

rtol = 1e-10
atol = 1e0
sparameters_diag = {
    'snes_type': 'ksponly',
    'snes': {
        'linesearch_type': 'basic',
        'monitor': None,
        'converged_reason': None,
        'rtol': rtol,
        'atol': atol,
    },
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp': {
        'monitor': None,
        'converged_reason': None,
        'rtol': rtol,
        'atol': atol,
    },
    'pc_type': 'python',
    'pc_python_type': 'asQ.CirculantPC',
    'diagfft_alpha': args.alpha,
    'diagfft_state': 'linear',
    'aaos_jacobian_state': 'linear',
}

for i in range(window_length):
    sparameters_diag['diagfft_block_'+str(i)+'_'] = sparameters

create_mesh = partial(
    swe.create_mg_globe_mesh,
    ref_level=args.ref_level,
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
                                  file_name='output/'+args.filename)

paradiag = miniapp.paradiag

timer = SolverTimer()


def window_preproc(swe_app, pdg, wndw):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'### === --- Calculating time-window {wndw} --- === ###')
    PETSc.Sys.Print('')
    timer.start_timing()


def window_postproc(swe_app, pdg, wndw):
    timer.stop_timing()
    if pdg.layout.is_local(miniapp.save_step):
        nt = (pdg.total_windows - 1)*pdg.ntimesteps + (miniapp.save_step + 1)
        time = nt*pdg.aaoform.dt
        comm = miniapp.ensemble.comm
        PETSc.Sys.Print('', comm=comm)
        PETSc.Sys.Print(f'Hours = {time/units.hour}', comm=comm)
        PETSc.Sys.Print(f'Days = {time/earth.day}', comm=comm)
        PETSc.Sys.Print('', comm=comm)


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

PETSc.Sys.Print(f'Maximum CFL = {max(miniapp.cfl_series)}')
PETSc.Sys.Print(f'Minimum CFL = {min(miniapp.cfl_series)}')
PETSc.Sys.Print('')

if timer.ntimes() > 1:
    timer.times[0] = timer.times[1]

PETSc.Sys.Print(timer.string(timesteps_per_solve=window_length, ndigits=5))
PETSc.Sys.Print('')
