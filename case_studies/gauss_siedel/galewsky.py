from firedrake.petsc import PETSc

import asQ
import firedrake as fd  # noqa: F401
from utils import units
from utils.planets import earth
import utils.shallow_water as swe
from utils.shallow_water import galewsky

from functools import partial
from math import sqrt

PETSc.Sys.popErrorHandler()

# get command arguments
import argparse
parser = argparse.ArgumentParser(
    description='Galewsky testcase for ParaDiag solver using fully implicit SWE solver.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--ref_level', type=int, default=2, help='Refinement level of icosahedral grid.')
parser.add_argument('--nchunks', type=int, default=4, help='Number of chunks to solve simultaneously.')
parser.add_argument('--nsweeps', type=int, default=4, help='Number of nonlinear sweeps.')
parser.add_argument('--nslices', type=int, default=2, help='Number of time-slices per time-window.')
parser.add_argument('--slice_length', type=int, default=2, help='Number of timesteps per time-slice.')
parser.add_argument('--alpha', type=float, default=0.0001, help='Circulant coefficient.')
parser.add_argument('--dt', type=float, default=0.5, help='Timestep in hours.')
parser.add_argument('--filename', type=str, default='galewsky', help='Name of output vtk files')
parser.add_argument('--metrics_dir', type=str, default='metrics', help='Directory to save paradiag metrics to.')
parser.add_argument('--print_res', action='store_true', help='Print the residuals of each timestep at each iteration.')
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
chunk_length = sum(time_partition)

dt = args.dt*units.hour


# alternative operator to precondition blocks

def aux_form_function(u, h, v, q, t):
    mesh = v.ufl_domain()
    coords = fd.SpatialCoordinate(mesh)

    gravity = earth.Gravity
    coriolis = swe.earth_coriolis_expression(*coords)

    H = galewsky.H0

    return swe.linear.form_function(mesh, gravity, H, coriolis,
                                    u, h, v, q, t)


block_appctx = {
    'aux_form_function': aux_form_function
}

# parameters for the implicit diagonal solve in step-(b)
factorisation_params = {
    'ksp_type': 'preonly',
    # 'pc_factor_mat_ordering_type': 'rcm',
    'pc_factor_reuse_ordering': None,
    'pc_factor_reuse_fill': None,
}

lu_params = {'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}
lu_params.update(factorisation_params)

aux_pc = {
    'snes_lag_preconditioner': -2,
    'snes_lag_preconditioner_persists': None,
    'pc_type': 'python',
    'pc_python_type': 'asQ.AuxiliaryBlockPC',
    'aux': lu_params,
}


sparameters = {
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp': {
        'atol': 1e-5,
        'rtol': 1e-5,
        'max_it': 40,
        'converged_maxits': None,
    },
}
sparameters.update(aux_pc)

atol = 1e4
patol = sqrt(chunk_length)*atol
sparameters_diag = {
    'snes': {
        'monitor': None,
        # 'converged_reason': None,
        'atol': 1e-0,
        'rtol': 1e-10,
        'stol': 1e-12,
        # 'ksp_ew': None,
        # 'ksp_ew_version': 1,
        # 'ksp_ew_threshold': 1e-2,
        'max_it': 1,
        'convergence_test': 'skip',
    },
    'mat_type': 'matfree',
    'ksp_type': 'preonly',
    'ksp': {
        # 'monitor': None,
        # 'converged_reason': None,
        'rtol': 1e-5,
        'atol': 1e-0,
    },
    'pc_type': 'python',
    'pc_python_type': 'asQ.DiagFFTPC',
    'diagfft_alpha': args.alpha,
    'diagfft_state': 'window',
    'aaos_jacobian_state': 'current'
}

for i in range(chunk_length):
    sparameters_diag['diagfft_block_'+str(i)+'_'] = sparameters

create_mesh = partial(
    swe.create_mg_globe_mesh,
    ref_level=args.ref_level,
    coords_degree=1)  # remove coords degree once UFL issue with gradient of cell normals fixed

# check convergence of each timestep


def post_function_callback(aaosolver, X, F):
    if args.print_res:
        residuals = asQ.SharedArray(time_partition,
                                    comm=aaosolver.ensemble.ensemble_comm)
        # all-at-once residual
        res = aaosolver.aaoform.F
        for i in range(res.nlocal_timesteps):
            with res[i].dat.vec_ro as vec:
                residuals.dlocal[i] = vec.norm()
        residuals.synchronise()
        PETSc.Sys.Print('')
        PETSc.Sys.Print([f"{r:.4e}" for r in residuals.data()])
        PETSc.Sys.Print('')


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
                                  post_function_callback=post_function_callback,
                                  record_diagnostics={'cfl': True, 'file': False})

paradiag = miniapp.paradiag
aaosolver = paradiag.solver
aaofunc = aaosolver.aaofunc
aaoform = aaosolver.aaoform

chunks = tuple(aaofunc.copy() for _ in range(args.nchunks))

chunk_ids = [i for i in range(args.nchunks)]

nconverged = 0
for j in range(args.nsweeps):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'### === --- Calculating nonlinear sweep {j} --- === ###')
    PETSc.Sys.Print('')

    for i in range(args.nchunks):
        PETSc.Sys.Print(f'        --- Calculating chunk {i} ---        ')

        # 1) load chunk i solution
        aaofunc.assign(chunks[i])

        # 2) set ic from previous chunk
        if i > 0:
            chunks[i-1].bcast_field(-1, aaofunc.initial_condition)

        # 3) one iteration of chunk i
        aaosolver.solve()

        # 4) save chunk i solution
        chunks[i].assign(aaofunc)

        PETSc.Sys.Print("")

    # check if first chunk is converged
    aaoform.assemble(chunks[0])
    with aaoform.F.global_vec_ro() as rvec:
        res = rvec.norm()
    if res < patol:
        PETSc.Sys.Print(">>> First chunk has converged. Rotating chunks <<<")
        nconverged += 1

        # shuffle chunks down
        for i in range(args.nchunks-1):
            chunks[i].assign(chunks[i+1])

        # reset last chunk to start from end of series
        chunks[-1].bcast_field(-1, chunks[-1].initial_condition)
        chunks[-1].assign(chunks[-1].initial_condition)

PETSc.Sys.Print(f"Number of chunks: {args.nchunks}")
PETSc.Sys.Print(f"Number of sweeps: {args.nsweeps}")
PETSc.Sys.Print(f"Number of chunks converged: {nconverged}")
PETSc.Sys.Print(f"Number of chunks converged per sweep: {nconverged/args.nsweeps}")
