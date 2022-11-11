
import firedrake as fd
from petsc4py import PETSc

from utils import units
from utils.planets import earth
import utils.shallow_water as swe
from utils.shallow_water import galewsky

from functools import partial

PETSc.Sys.popErrorHandler()

# get command arguments
import argparse
parser = argparse.ArgumentParser(
    description='Galewsky testcase for ParaDiag solver using fully implicit SWE solver.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--ref_level', type=int, default=2, help='Refinement level of icosahedral grid.')
parser.add_argument('--nwindows', type=int, default=1, help='Number of time-windows.')
parser.add_argument('--nslices', type=int, default=2, help='Number of time-slices per time-window.')
parser.add_argument('--slice_length', type=int, default=2, help='Number of timesteps per time-slice.')
parser.add_argument('--alpha', type=float, default=0.0001, help='Circulant coefficient.')
parser.add_argument('--dt', type=float, default=0.5, help='Timestep in hours.')
parser.add_argument('--filename', type=str, default='galewsky', help='Name of output vtk files')
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
    'ksp': {
        'atol': 1e-8,
        'rtol': 1e-8,
        'max_it': 400,
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

sparameters_diag = {
    'snes': {
        'linesearch_type': 'basic',
        'monitor': None,
        'converged_reason': None,
        'atol': 1e-0,
        'rtol': 1e-12,
        'stol': 1e-12,
        'max_its': 1
    },
    'mat_type': 'matfree',
    'ksp_type': 'preonly',
    'ksp': {
        'monitor': None,
        'converged_reason': None,
    },
    'pc_type': 'python',
    'pc_python_type': 'asQ.DiagFFTPC'
}

for i in range(sum(time_partition)):
    sparameters_diag['diagfft_'+str(i)+'_'] = sparameters

create_mesh = partial(
    swe.create_mg_globe_mesh,
    ref_level=args.ref_level,
    coords_degree=1)

PETSc.Sys.Print('### === --- Calculating parallel solution --- === ###')

miniapp = swe.ShallowWaterMiniApp(gravity=earth.Gravity,
                                  topography_expression=galewsky.topography_expression,
                                  velocity_expression=galewsky.velocity_expression,
                                  depth_expression=galewsky.depth_expression,
                                  reference_depth=galewsky.H0,
                                  create_mesh=create_mesh,
                                  dt=dt, theta=0.5,
                                  alpha=args.alpha, time_partition=time_partition,
                                  paradiag_sparameters=sparameters_diag)

ensemble = miniapp.ensemble
time_rank = miniapp.paradiag.time_rank

# only last slice does diagnostics/output
if time_rank == len(time_partition)-1:
    cfl_series = []
    linear_its = 0
    nonlinear_its = 0

    ofile = fd.File('output/'+args.filename+'.pvd',
                    comm=ensemble.comm)

    uout = fd.Function(miniapp.velocity_function_space(), name='velocity')
    hout = fd.Function(miniapp.depth_function_space(), name='depth')

    miniapp.get_velocity(-1, uout=uout)
    miniapp.get_elevation(-1, hout=hout)

    ofile.write(uout, hout,
                miniapp.potential_vorticity(uout),
                time=0)

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
    if time_rank == len(time_partition)-1:
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
if args.nwindows == 0:
    quit()

miniapp.solve(nwindows=args.nwindows,
              preproc=window_preproc,
              postproc=window_postproc)


PETSc.Sys.Print('### === --- Iteration counts --- === ###')
PETSc.Sys.Print('')

if time_rank == len(time_partition)-1:
    PETSc.Sys.Print(f'Maximum CFL = {max(cfl_series)}', comm=ensemble.comm)
    PETSc.Sys.Print(f'Minimum CFL = {min(cfl_series)}', comm=ensemble.comm)
    PETSc.Sys.Print('', comm=ensemble.comm)

    PETSc.Sys.Print(f'windows: {(args.nwindows)}', comm=ensemble.comm)
    PETSc.Sys.Print(f'timesteps: {(args.nwindows)*window_length}', comm=ensemble.comm)
    PETSc.Sys.Print('', comm=ensemble.comm)

    PETSc.Sys.Print(f'linear iterations: {linear_its} | iterations per window: {linear_its/(args.nwindows)}', comm=ensemble.comm)
    PETSc.Sys.Print(f'nonlinear iterations: {nonlinear_its} | iterations per window: {nonlinear_its/(args.nwindows)}', comm=ensemble.comm)
    PETSc.Sys.Print('', comm=ensemble.comm)
