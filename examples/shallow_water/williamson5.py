
import firedrake as fd
from petsc4py import PETSc

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
parser.add_argument('--filename', type=str, default='w5diag', help='Name of output vtk files')
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
sparameters = {
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp_atol': 1e-8,
    'ksp_rtol': 1e-8,
    'ksp_max_it': 400,
    'pc_type': 'mg',
    'pc_mg_cycle_type': 'v',
    'pc_mg_type': 'multiplicative',
    'mg_levels_ksp_type': 'gmres',
    'mg_levels_ksp_max_it': 5,
    'mg_levels_pc_type': 'python',
    'mg_levels_pc_python_type': 'firedrake.PatchPC',
    'mg_levels_patch_pc_patch_save_operators': True,
    'mg_levels_patch_pc_patch_partition_of_unity': True,
    'mg_levels_patch_pc_patch_sub_mat_type': 'seqdense',
    'mg_levels_patch_pc_patch_construct_dim': 0,
    'mg_levels_patch_pc_patch_construct_type': 'vanka',
    'mg_levels_patch_pc_patch_local_type': 'additive',
    'mg_levels_patch_pc_patch_precompute_element_tensors': True,
    'mg_levels_patch_pc_patch_symmetrise_sweep': False,
    'mg_levels_patch_sub_ksp_type': 'preonly',
    'mg_levels_patch_sub_pc_type': 'lu',
    'mg_levels_patch_sub_pc_factor_shift_type': 'nonzero',
    'mg_coarse_pc_type': 'python',
    'mg_coarse_pc_python_type': 'firedrake.AssembledPC',
    'mg_coarse_assembled_pc_type': 'lu',
    'mg_coarse_assembled_pc_factor_mat_solver_type': 'mumps',
}

sparameters_diag = {
    'snes_linesearch_type': 'basic',
    'snes_monitor': None,
    'snes_converged_reason': None,
    'snes_atol': 1e-0,
    'snes_rtol': 1e-12,
    'snes_stol': 1e-12,
    'snes_max_it': 100,
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp_monitor': None,
    'ksp_converged_reason': None,
    'pc_type': 'python',
    'pc_python_type': 'asQ.DiagFFTPC'}

sparameters_diag['diagfft_block_'] = sparameters

create_mesh = partial(
    swe.create_mg_globe_mesh,
    ref_level=args.ref_level)

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
                                  dt=dt, theta=0.5,
                                  alpha=args.alpha, time_partition=time_partition,
                                  paradiag_sparameters=sparameters_diag)

ensemble = miniapp.ensemble
time_rank = miniapp.paradiag.time_rank

# only last slice does diagnostics/output
is_io_rank = (time_rank == len(time_partition)-1)
if is_io_rank:
    cfl_series = []
    linear_its = 0
    nonlinear_its = 0

    ofile = fd.File('output/'+args.filename+'.pvd',
                    comm=ensemble.comm)

    uout = fd.Function(miniapp.velocity_function_space(), name='velocity')
    hout = fd.Function(miniapp.depth_function_space(), name='depth')

    def time_at_last_step(w):
        return dt*(w + 1)*window_length


def window_preproc(swe_app, pdg, wndw):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'### === --- Calculating time-window {wndw} --- === ###')
    PETSc.Sys.Print('')


def window_postproc(swe_app, pdg, wndw):
    # make sure variables are properly captured
    global linear_its
    global nonlinear_its
    global cfl_series

    # postprocess this timeslice
    if is_io_rank:
        linear_its += pdg.snes.getLinearSolveIterations()
        nonlinear_its += pdg.snes.getIterationNumber()

        swe_app.get_velocity(-1, uout=uout)
        swe_app.get_elevation(-1, hout=hout)

        time = time_at_last_step(wndw)

        ofile.write(uout, hout,
                    swe_app.potential_vorticity(uout),
                    time=time/earth.day)

        cfl = swe_app.max_cfl(dt, -1)
        cfl_series.append(cfl)

        PETSc.Sys.Print('', comm=ensemble.comm)
        PETSc.Sys.Print(f'Maximum CFL = {cfl}', comm=ensemble.comm)
        PETSc.Sys.Print(f'Hours = {time/units.hour}', comm=ensemble.comm)
        PETSc.Sys.Print(f'Days = {time/earth.day}', comm=ensemble.comm)
        PETSc.Sys.Print('', comm=ensemble.comm)
    PETSc.Sys.Print('')


# solve for each window
miniapp.solve(nwindows=args.nwindows,
              preproc=window_preproc,
              postproc=window_postproc)


PETSc.Sys.Print('### === --- Iteration counts --- === ###')
PETSc.Sys.Print('')

if is_io_rank:
    PETSc.Sys.Print(f'Maximum CFL = {max(cfl_series)}', comm=ensemble.comm)
    PETSc.Sys.Print(f'Minimum CFL = {min(cfl_series)}', comm=ensemble.comm)
    PETSc.Sys.Print('', comm=ensemble.comm)

    PETSc.Sys.Print(f'windows: {(args.nwindows)}', comm=ensemble.comm)
    PETSc.Sys.Print(f'timesteps: {(args.nwindows)*window_length}', comm=ensemble.comm)
    PETSc.Sys.Print('', comm=ensemble.comm)

    PETSc.Sys.Print(f'linear iterations: {linear_its} | iterations per window: {linear_its/(args.nwindows)}', comm=ensemble.comm)
    PETSc.Sys.Print(f'nonlinear iterations: {nonlinear_its} | iterations per window: {nonlinear_its/(args.nwindows)}', comm=ensemble.comm)
    PETSc.Sys.Print('', comm=ensemble.comm)
