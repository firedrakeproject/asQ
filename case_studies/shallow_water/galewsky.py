from firedrake.petsc import PETSc

import asQ
import firedrake as fd  # noqa: F401
from utils import units
from utils.misc import function_mean
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

time_partition = tuple((args.slice_length for _ in range(args.nslices)))
window_length = sum(time_partition)
nsteps = args.nwindows*window_length

dt = args.dt*units.hour

# alternative operator to precondition blocks


def aux_form_function(u, h, v, q, t):
    mesh = v.ufl_domain()
    coords = fd.SpatialCoordinate(mesh)

    gravity = earth.Gravity
    topography = galewsky.topography_expression(*coords)  # noqa: F841
    coriolis = swe.earth_coriolis_expression(*coords)

    depth = fd.Function(swe.default_depth_function_space(mesh))
    depth.assign(galewsky.depth_expression(*coords))
    H = function_mean(depth)

    # Ku = swe.nonlinear.form_function_velocity(
    #     mesh, gravity, topography, coriolis, u, h, v, t, perp=fd.cross)

    # Ku = swe.linear.form_function_u(
    #     mesh, gravity, coriolis, u, h, v, t)

    # Kh = swe.linear.form_function_h(
    #     mesh, H, u, h, q, t)

    # return Ku + Kh

    return swe.linear.form_function(mesh, gravity, H, coriolis,
                                    u, h, v, q, t)


block_appctx = {
    'aux_form_function': aux_form_function
}

# parameters for the implicit diagonal solve in step-(b)
factorisation_params = {
    'ksp_type': 'preonly',
    'pc_factor_mat_ordering_type': 'rcm',
    'pc_factor_reuse_ordering': None,
    'pc_factor_reuse_fill': None,
}

lu_params = {'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}
lu_params.update(factorisation_params)

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
        'ksp_max_it': 5,
        'pc_type': 'python',
        'pc_python_type': 'firedrake.PatchPC',
        'patch': patch_parameters
    },
    'coarse': {
        'pc_type': 'python',
        'pc_python_type': 'firedrake.AssembledPC',
        'assembled': lu_params
    }
}

mg_pc = {
    'pc_type': 'mg',
    'pc_mg_cycle_type': 'v',
    'pc_mg_type': 'multiplicative',
    'mg': mg_parameters
}

aux_pc = {
    'snes_lag_preconditioner': -2,
    'snes_lag_preconditioner_persists': None,
    'pc_type': 'python',
    'pc_python_type': 'asQ.AuxiliaryBlockPC',
    'aux': lu_params,
}

block_sparams = {
    'mat_type': 'matfree',
    'ksp_type': 'gmres',
    'ksp': {
        'atol': 1e-100,
        'rtol': 1e-5,
        'max_it': 60,
    },
}

# block_sparams = lu_params
# block_sparams.update(mg_pc)
block_sparams.update(aux_pc)

atol = 1e4
sparameters_diag = {
    'snes': {
        'linesearch_type': 'basic',
        # 'monitor': None,
        'converged_reason': None,
        'atol': atol,
        'rtol': 1e-10,
        'stol': 1e-12,
        'ksp_ew': None,
        'ksp_ew_version': 1,
        'ksp_ew_threshold': 1e-2,
    },
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp': {
        # 'monitor': None,
        # 'converged_rate': None,
        'rtol': 1e-5,
        'atol': atol,
    },
    'pc_type': 'python',
    'pc_python_type': 'asQ.DiagFFTPC',
    'diagfft_alpha': args.alpha,
    'diagfft_state': 'window',
    'aaos_jacobian_state': 'current'
}

for i in range(window_length):
    sparameters_diag['diagfft_block_'+str(i)+'_'] = block_sparams

create_mesh = partial(
    swe.create_mg_globe_mesh,
    ref_level=args.ref_level,
    coords_degree=1)  # remove coords degree once UFL issue with gradient of cell normals fixed

PETSc.Sys.Print('### === --- Calculating parallel solution --- === ###')

appctx = {'block_appctx': block_appctx}

miniapp = swe.ShallowWaterMiniApp(gravity=earth.Gravity,
                                  topography_expression=galewsky.topography_expression,
                                  velocity_expression=galewsky.velocity_expression,
                                  depth_expression=galewsky.depth_expression,
                                  reference_depth=galewsky.H0,
                                  reference_state=True,
                                  create_mesh=create_mesh,
                                  dt=dt, theta=0.5,
                                  time_partition=time_partition,
                                  appctx=appctx,
                                  paradiag_sparameters=sparameters_diag,
                                  file_name='output/'+args.filename,
                                  record_diagnostics={'cfl': True, 'file': False})

ics = miniapp.aaofunc.initial_condition
reference_state = miniapp.solver.jacobian.reference_state
reference_state.assign(ics)
reference_state.subfunctions[0].assign(0)
reference_state.subfunctions[1].assign(galewsky.H0)


def window_preproc(swe_app, pdg, wndw):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'### === --- Calculating time-window {wndw} --- === ###')
    PETSc.Sys.Print('')


def window_postproc(swe_app, pdg, wndw):
    if miniapp.layout.is_local(miniapp.save_step):
        nt = (pdg.total_windows - 1)*pdg.ntimesteps + (miniapp.save_step + 1)
        time = float(nt*pdg.aaoform.dt)
        comm = miniapp.ensemble.comm
        PETSc.Sys.Print('', comm=comm)
        PETSc.Sys.Print(f'Maximum CFL = {swe_app.cfl_series[wndw]}', comm=comm)
        PETSc.Sys.Print(f'Hours = {time/units.hour}', comm=comm)
        PETSc.Sys.Print(f'Days = {time/earth.day}', comm=comm)
        PETSc.Sys.Print('', comm=comm)


miniapp.solve(nwindows=args.nwindows,
              preproc=window_preproc,
              postproc=window_postproc)

PETSc.Sys.Print('### === --- Iteration counts --- === ###')

asQ.write_paradiag_metrics(miniapp.paradiag, directory=args.metrics_dir)

PETSc.Sys.Print('')

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
