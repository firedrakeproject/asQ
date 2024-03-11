from firedrake.petsc import PETSc

import asQ
import firedrake as fd
from utils import units
from utils.planets import earth
import utils.shallow_water as swe
from utils.shallow_water import galewsky

from time import sleep  # noqa: F401
import numpy as np
from math import sqrt

PETSc.Sys.popErrorHandler()

# get command arguments
import argparse
parser = argparse.ArgumentParser(
    description='Galewsky testcase for ParaDiag solver using fully implicit SWE solver.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--ref_level', type=int, default=2, help='Refinement level of icosahedral grid.')
parser.add_argument('--base_level', type=int, default=2, help='Refinement level of coarse grid.')
parser.add_argument('--nwindows', type=int, default=1, help='Total number of time-windows.')
parser.add_argument('--nlmethod', type=str, default='gs', choices=['gs', 'jac'], help='Nonlinear method. "gs" for Gauss-Siedel or "jac" for Jacobi.')
parser.add_argument('--nchunks', type=int, default=4, help='Number of chunks to solve simultaneously.')
parser.add_argument('--nsweeps', type=int, default=4, help='Number of nonlinear sweeps.')
parser.add_argument('--nsmooth', type=int, default=1, help='Number of nonlinear iterations per chunk at each sweep.')
parser.add_argument('--nchecks', type=int, default=1, help='Maximum number of chunks allowed to converge after each sweep.')
parser.add_argument('--ninitialise', type=int, default=0, help='Number of sweeps before checking convergence.')
parser.add_argument('--atol', type=float, default=1e5, help='Average atol of each timestep.')
parser.add_argument('--nslices', type=int, default=2, help='Number of time-slices per time-window.')
parser.add_argument('--slice_length', type=int, default=2, help='Number of timesteps per time-slice.')
parser.add_argument('--alpha', type=float, default=1e-4, help='Circulant coefficient.')
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

chunk_partition = tuple((args.slice_length for _ in range(args.nslices)))
chunk_length = sum(chunk_partition)
total_timesteps = chunk_length*args.nchunks
total_slices = args.nslices*args.nchunks

global_comm = fd.COMM_WORLD
global_time_partition = tuple((args.slice_length for _ in range(total_slices)))
global_ensemble = asQ.create_ensemble(global_time_partition, global_comm)
chunk_ensemble = asQ.split_ensemble(global_ensemble, args.nslices)

# which chunk are we?
chunk_id = global_ensemble.ensemble_comm.rank // args.nslices

dt = args.dt*units.hour

mesh = swe.create_mg_globe_mesh(
    ref_level=args.ref_level, base_level=args.base_level,
    coords_degree=1, comm=chunk_ensemble.comm)
coords = fd.SpatialCoordinate(mesh)

# alternative operator to precondition blocks
g = earth.Gravity
f = swe.earth_coriolis_expression(*coords)
H = galewsky.H0
b = galewsky.topography_expression(*coords)


def form_mass(u, h, v, q):
    return swe.nonlinear.form_mass(mesh, u, h, v, q)


def form_function(u, h, v, q, t=None):
    return swe.nonlinear.form_function(mesh, g, b, f,
                                       u, h, v, q, t)


def aux_form_function(u, h, v, q, t=None):
    return swe.linear.form_function(mesh, g, H, f,
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
        'max_it': 30,
        'converged_maxits': None,
    },
}
sparameters.update(aux_pc)

atol = args.atol
patol = sqrt(chunk_length)*atol
sparameters_diag = {
    'snes': {
        'linesearch_type': 'basic',
        'monitor': None,
        'converged_reason': None,
        'atol': patol,
        'rtol': 1e-10,
        'stol': 1e-12,
        # 'ksp_ew': None,
        # 'ksp_ew_version': 1,
        # 'ksp_ew_threshold': 1e-2,
        'max_it': args.nsmooth,
        'convergence_test': 'skip',
    },
    'mat_type': 'matfree',
    'ksp_type': 'preonly',
    'ksp': {
        # 'monitor': None,
        # 'converged_reason': None,
        # 'max_it': 2,
        # 'converged_maxits': None,
        'rtol': 1e-2,
        'atol': patol,
    },
    'pc_type': 'python',
    'pc_python_type': 'asQ.DiagFFTPC',
    'diagfft_alpha': args.alpha,
    'diagfft_state': 'window',
    'aaos_jacobian_state': 'current'
}

for i in range(chunk_length):
    sparameters_diag['diagfft_block_'+str(i)+'_'] = sparameters

appctx = {'block_appctx': block_appctx}

# function spaces and initial conditions

W = swe.default_function_space(mesh)

winitial = fd.Function(W)
uinitial, hinitial = winitial.subfunctions
uinitial.project(galewsky.velocity_expression(*coords))
hinitial.interpolate(galewsky.depth_expression(*coords))

# all at once solver

chunk_aaofunc = asQ.AllAtOnceFunction(chunk_ensemble, chunk_partition, W)
chunk_aaofunc.assign(winitial)

theta = 0.5
chunk_aaoform = asQ.AllAtOnceForm(chunk_aaofunc, dt, theta,
                                  form_mass, form_function)

chunk_solver = asQ.AllAtOnceSolver(chunk_aaoform, chunk_aaofunc,
                                   solver_parameters=sparameters_diag,
                                   appctx=appctx)

# which chunk is holding which part of the total timeseries?
chunk_indexes = np.array([i for i in range(args.nchunks)], dtype=int)

# we need to make this an array so we can send it via mpi
convergence_flag = np.array([False], dtype=bool)

# first mpi rank on each chunk (assumes all chunks are equal size):
chunk_root = lambda c: c*chunk_ensemble.global_comm.size

# which chunks are currently at the beginning/end of the sweep?
first_chunk = 0
last_chunk = args.nchunks - 1

# update chunk ics from previous chunk
uprev = fd.Function(W)


def update_chunk_halos(uhalo):
    chunk_begin = chunk_aaofunc.layout.is_local(0)
    chunk_end = chunk_aaofunc.layout.is_local(-1)

    global_rank = global_ensemble.ensemble_comm.rank
    global_size = global_ensemble.ensemble_comm.size

    # ring communication so the first chunk can
    # pick up after last chunk after convergence

    # send forward
    if chunk_end:
        dst = (global_rank + 1) % global_size
        global_ensemble.send(chunk_aaofunc[-1], dest=dst, tag=dst)

    # recv previous
    if chunk_begin:
        src = (global_rank - 1) % global_size
        global_ensemble.recv(uhalo, source=src, tag=global_rank)

    # broadcast new ics to all ranks
    chunk_ensemble.bcast(uhalo)


PETSc.Sys.Print('### === --- Calculating parallel solution --- === ###')
PETSc.Sys.Print('')

nconverged = 0

PETSc.Sys.Print('### === --- Initialising all chunks --- === ###')
PETSc.Sys.Print('')
sleep_time = 0.1
for j in range(args.nchunks):
    PETSc.Sys.Print(f'    === --- Initial nonlinear sweep {j} --- ===    ')
    PETSc.Sys.Print('')

    # only smooth chunks that the first sweep has reached
    # for i in range(j+1):
    #     PETSc.Sys.Print(f'        --- Calculating chunk {i} ---        ')

    #     global_comm.Barrier(); sleep(sleep_time)
    #     if chunk_id == i:
    #         chunk_solver.solve()
    #     global_comm.Barrier(); sleep(sleep_time)

    #     PETSc.Sys.Print("")

    if chunk_id < j+1:
        chunk_solver.solve()

    # propogate solution
    update_chunk_halos(uprev)

    # update initial conditions
    if chunk_id != 0:
        chunk_aaofunc.initial_condition.assign(uprev)

    # initial guess before first sweep is persistence forecast
    if chunk_id > j:
        chunk_aaofunc.assign(chunk_aaofunc.initial_condition)

PETSc.Sys.Print('### === --- All chunks initialised  --- === ###')
PETSc.Sys.Print('')

for j in range(args.nsweeps):
    PETSc.Sys.Print(f'    === --- Calculating nonlinear sweep {j} --- ===    ')
    PETSc.Sys.Print('')

    # 1) one smoothing application on each chunk
    # for i in range(args.nchunks):
    #     PETSc.Sys.Print(f'        --- Calculating chunk {i} on solver {chunk_indexes[i]} ---        ')

    #     global_comm.Barrier(); sleep(sleep_time)
    #     if chunk_id == chunk_indexes[i]:
    #         chunk_solver.solve()
    #     global_comm.Barrier(); sleep(sleep_time)

    #     PETSc.Sys.Print("")
    chunk_solver.solve()

    # 2) update ics of each chunk from previous chunk
    update_chunk_halos(uprev)

    earliest_chunk = (chunk_id == chunk_indexes[0])

    # everyone uses latest ic guesses except chunk 0 (uses 'exact' ic)
    if not earliest_chunk:
        chunk_aaofunc.initial_condition.assign(uprev)

    # 3) check convergence of earliest chunk

    if earliest_chunk:
        chunk_aaoform.assemble()
        with chunk_aaoform.F.global_vec_ro() as rvec:
            res = rvec.norm()
        convergence_flag[0] = (res < patol)

    # rank 0 on the earliest chunk tells everyone if they've converged
    global_ensemble.global_comm.Bcast(convergence_flag,
                                      root=chunk_root(chunk_indexes[0]))

    # 4) shuffle if converged
    if convergence_flag[0]:
        # earliest chunk becomes last chunk
        if earliest_chunk:
            chunk_aaofunc.assign(uprev)

        # update record of which chunk is in which position
        for i in range(args.nchunks):
            chunk_indexes[i] = (chunk_indexes[i] + 1) % args.nchunks

        nconverged += 1

    PETSc.Sys.Print('')
    PETSc.Sys.Print(f">>> Converged chunks: {int(nconverged)}.")
    converged_time = nconverged*chunk_length*args.dt
    PETSc.Sys.Print(f">>> Converged time: {converged_time} hours.")
    PETSc.Sys.Print('')

    if nconverged >= args.nwindows:
        PETSc.Sys.Print(f"Finished iterating to {args.nwindows} windows.")
        PETSc.Sys.Print('')
        break

nsweeps = j

niterations = nsweeps*args.nsmooth

PETSc.Sys.Print(f"Number of chunks: {args.nchunks}")
PETSc.Sys.Print(f"Maximum number of sweeps: {args.nsweeps}")
PETSc.Sys.Print(f"Actual number of sweeps: {nsweeps}")
PETSc.Sys.Print(f"Number of chunks converged: {int(nconverged)}")
PETSc.Sys.Print(f"Number of chunks converged per sweep: {nconverged/nsweeps}")
PETSc.Sys.Print(f"Number of sweeps per converged chunk: {nsweeps/nconverged if nconverged else 'n/a'}")
PETSc.Sys.Print(f"Number of iterations per converged chunk: {niterations/nconverged if nconverged else 'n/a'}")
PETSc.Sys.Print(f"Number of timesteps per iteration: {nconverged*chunk_length/niterations}")
