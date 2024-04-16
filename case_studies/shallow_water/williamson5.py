from firedrake.petsc import PETSc
from pyop2.mpi import MPI

from math import sqrt
from utils import units
from utils.planets import earth
import utils.shallow_water as swe
from utils.shallow_water.williamson1992 import case5

from functools import partial

PETSc.Sys.popErrorHandler()

# get command arguments
import argparse
parser = argparse.ArgumentParser(
    description='Williamson 5 testcase for ParaDiag solver using fully implicit SWE solver.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--ref_level', type=int, default=2, help='Refinement level of icosahedral grid.')
parser.add_argument('--nwindows', type=int, default=1, help='Number of time-windows.')
parser.add_argument('--nslices', type=int, default=2, help='Number of time-slices per time-window.')
parser.add_argument('--slice_length', type=int, default=2, help='Number of timesteps per time-slice.')
parser.add_argument('--alpha', type=float, default=0.0001, help='Circulant coefficient.')
parser.add_argument('--dt', type=float, default=0.5, help='Timestep in hours.')
parser.add_argument('--filename', type=str, default='williamson5', help='Name of output vtk files')
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

time_partition = tuple(args.slice_length for _ in range(args.nslices))
window_length = sum(time_partition)

dt = args.dt*units.hour
# parameters for the implicit diagonal solve in step-(b)
patch_parameters = {
    'pc_patch': {
        'save_operators': True,
        'partition_of_unity': True,
        'sub_mat_type': 'seqdense',
        'construct_dim': 0,
        'construct_type': 'star',
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
        'max_it': 50
    },
    'pc_type': 'mg',
    'pc_mg_cycle_type': 'v',
    'pc_mg_type': 'full',
    'mg': mg_parameters
}

atol = 1e4
patol = sqrt(window_length)*atol
sparameters_diag = {
    'snes': {
        'linesearch_type': 'basic',
        'monitor': None,
        'converged_reason': None,
        'atol': patol,
        'rtol': 1e-10,
        'stol': 1e-12,
        'ksp_ew': None,
        'ksp_ew_version': 1,
    },
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp': {
        'monitor': None,
        'converged_rate': None,
        'rtol': 1e-2,
        'atol': patol,
    },
    'pc_type': 'python',
    'pc_python_type': 'asQ.CirculantPC',
    'diagfft_alpha': args.alpha,
}

for i in range(window_length):
    sparameters_diag['diagfft_block_'+str(i)+'_'] = sparameters

create_mesh = partial(
    swe.create_mg_globe_mesh,
    ref_level=args.ref_level,
    coords_degree=1)  # remove coords degree once UFL issue with gradient of cell normals fixed

# initial conditions
b_exp = case5.topography_expression
u_exp = case5.velocity_expression


def h_exp(x, y, z):
    b_exp = case5.topography_expression
    eta_exp = case5.elevation_expression
    return case5.H0 + eta_exp(x, y, z) - b_exp(x, y, z)


PETSc.Sys.Print('### === --- Calculating parallel solution --- === ###')

miniapp = swe.ShallowWaterMiniApp(gravity=earth.Gravity,
                                  topography_expression=b_exp,
                                  velocity_expression=u_exp,
                                  depth_expression=h_exp,
                                  reference_depth=case5.H0,
                                  create_mesh=create_mesh,
                                  reference_state=True,
                                  dt=dt, theta=0.5,
                                  time_partition=time_partition,
                                  paradiag_sparameters=sparameters_diag,
                                  record_diagnostics={'cfl': True, 'file': False},
                                  file_name='output/'+args.filename)

solver_time = []


def window_preproc(swe_app, pdg, wndw):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'### === --- Calculating time-window {wndw} --- === ###')
    PETSc.Sys.Print('')
    stime = MPI.Wtime()
    solver_time.append(stime)


def window_postproc(swe_app, pdg, wndw):
    etime = MPI.Wtime()
    stime = solver_time[-1]
    duration = etime - stime
    solver_time[-1] = duration
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'Window solution time: {duration}')
    PETSc.Sys.Print('')

    nt = (pdg.total_windows - 1)*pdg.ntimesteps + (miniapp.save_step + 1)
    time = nt*pdg.aaoform.dt
    comm = miniapp.ensemble.comm
    PETSc.Sys.Print(f'Hours = {round(float(time/units.hour), 2)}')
    PETSc.Sys.Print(f'Days = {round(float(time/earth.day), 2)}')
    PETSc.Sys.Print('')

    if miniapp.layout.is_local(miniapp.save_step):
        PETSc.Sys.Print(f'Maximum CFL = {round(swe_app.cfl_series[wndw], 4)}', comm=comm)
        PETSc.Sys.Print('', comm=comm)


# solve for each window
miniapp.solve(nwindows=args.nwindows,
              preproc=window_preproc,
              postproc=window_postproc)


PETSc.Sys.Print('### === --- Iteration counts --- === ###')
PETSc.Sys.Print('')

from asQ import write_paradiag_metrics
write_paradiag_metrics(miniapp.paradiag, directory=args.metrics_dir)

nw = miniapp.paradiag.total_windows
nt = miniapp.paradiag.total_timesteps
PETSc.Sys.Print(f'windows: {nw}')
PETSc.Sys.Print(f'timesteps: {nt}')
PETSc.Sys.Print('')

lits = miniapp.paradiag.linear_iterations
nlits = miniapp.paradiag.nonlinear_iterations
blits = miniapp.paradiag.block_iterations.data()

PETSc.Sys.Print(f'linear iterations: {lits} | iterations per window: {lits/nw}')
PETSc.Sys.Print(f'nonlinear iterations: {nlits} | iterations per window: {nlits/nw}')
PETSc.Sys.Print(f'block linear iterations: {blits} | iterations per block solve: {blits/lits}')
PETSc.Sys.Print('')

PETSc.Sys.Print(f'Maximum CFL = {max(miniapp.cfl_series)}')
PETSc.Sys.Print(f'Minimum CFL = {min(miniapp.cfl_series)}')
PETSc.Sys.Print('')

W = miniapp.W
mesh = W.mesh()
PETSc.Sys.Print(f'DoFs per timestep: {W.dim()}')
PETSc.Sys.Print(f'Number of MPI ranks per timestep: {mesh.comm.size} ')
PETSc.Sys.Print(f'DoFs/rank: {W.dim()/mesh.comm.size}')
PETSc.Sys.Print(f'Block DoFs/rank: {2*W.dim()/mesh.comm.size}')
PETSc.Sys.Print('')

if len(solver_time) > 1:
    solver_time[0] = solver_time[1]

PETSc.Sys.Print(f'Total solution time: {sum(solver_time)}')
PETSc.Sys.Print(f'Average window solution time: {sum(solver_time)/len(solver_time)}')
PETSc.Sys.Print(f'Average timestep solution time: {sum(solver_time)/(window_length*len(solver_time))}')
PETSc.Sys.Print('')
