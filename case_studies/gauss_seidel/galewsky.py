from firedrake.petsc import PETSc

from pyop2.mpi import MPI
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
    description='Galewsky testcase for ParaDiag solver using a pipelined nonlinear Gauss-Seidel method.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--ref_level', type=int, default=2, help='Refinement level of icosahedral grid.')
parser.add_argument('--base_level', type=int, default=2, help='Refinement level of coarse grid.')
parser.add_argument('--nwindows', type=int, default=1, help='Total number of time-windows.')
parser.add_argument('--nchunks', type=int, default=4, help='Number of chunks to solve simultaneously.')
parser.add_argument('--nsweeps', type=int, default=4, help='Number of nonlinear sweeps.')
parser.add_argument('--nsmooth', type=int, default=1, help='Number of nonlinear iterations per chunk at each sweep.')
parser.add_argument('--atol', type=float, default=1e5, help='Average atol of each timestep.')
parser.add_argument('--nslices', type=int, default=2, help='Number of time-slices per time-window.')
parser.add_argument('--slice_length', type=int, default=2, help='Number of timesteps per time-slice.')
parser.add_argument('--alpha', type=float, default=1e-4, help='Circulant coefficient.')
parser.add_argument('--dt', type=float, default=0.5, help='Timestep in hours.')
parser.add_argument('--serial', action='store_true', help='Calculate each chunk in serial.')
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

PETSc.Sys.Print('### === --- Create mesh --- === ###')
PETSc.Sys.Print('')

mesh = swe.create_mg_globe_mesh(
    ref_level=args.ref_level, base_level=args.base_level,
    coords_degree=1, comm=chunk_ensemble.comm)
coords = fd.SpatialCoordinate(mesh)

PETSc.Sys.Print('### === --- Forms --- === ###')
PETSc.Sys.Print('')

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


PETSc.Sys.Print('### === --- Parameters --- === ###')
PETSc.Sys.Print('')

block_appctx = {
    'aux_form_function': aux_form_function
}

snes_linear_params = {
    'type': 'ksponly',
    'lag_jacobian': -2,
    'lag_jacobian_persists': None,
    'lag_preconditioner': -2,
    'lag_preconditioner_persists': None,
}

# parameters for the implicit diagonal solve in step-(b)
factorisation_params = {
    'ksp_type': 'preonly',
    # 'pc_factor_mat_ordering_type': 'rcm',
    'pc_factor_reuse_ordering': None,
    'pc_factor_reuse_fill': None,
}

lu_params = {
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
    'pc_factor_shift_type': 'nonzero',
}
lu_params.update(factorisation_params)

aux_pc = {
    'pc_type': 'python',
    'pc_python_type': 'asQ.AuxiliaryBlockPC',
    'aux_snes': snes_linear_params,
    'aux': lu_params,
    'ksp_max_it': 50,
}

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
        'ksp_max_it': 4,
        'pc_type': 'python',
        'pc_python_type': 'firedrake.PatchPC',
        'patch': patch_parameters
    },
    'coarse': {
        'pc_type': 'python',
        'pc_python_type': 'firedrake.AssembledPC',
        'assembled': lu_params,
    }
}

mg_pc = {
    'pc_type': 'mg',
    'pc_mg_cycle_type': 'v',
    'pc_mg_type': 'multiplicative',
    'mg': mg_parameters,
    'ksp_max_it': 10,
}

sparameters = {
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp': {
        'atol': 1e-5,
        'rtol': 1e-5,
        'converged_maxits': None,
    },
}
# sparameters.update(aux_pc)
sparameters.update(mg_pc)

atol = args.atol
patol = sqrt(chunk_length)*atol
sparameters_diag = {
    'snes': {
        'linesearch_type': 'basic',
        # 'monitor': None,
        # 'converged_reason': None,
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
    'pc_python_type': 'asQ.CirculantPC',
    'circulant_alpha': args.alpha,
    'circulant_state': 'window',
    'circulant_block': sparameters,
    'aaos_jacobian_state': 'current'
}

appctx = {'block_appctx': block_appctx}

# function spaces and initial conditions

PETSc.Sys.Print('### === --- Initial conditions --- === ###')
PETSc.Sys.Print('')

PETSc.Sys.Print(1)
W = swe.default_function_space(mesh)

PETSc.Sys.Print(2)
winitial = fd.Function(W)
PETSc.Sys.Print(3)
uinitial, hinitial = winitial.subfunctions
PETSc.Sys.Print(4)
expr = galewsky.velocity_expression(*coords)
PETSc.Sys.Print(4.5)
uinitial.project(expr)
PETSc.Sys.Print(5)
hinitial.project(galewsky.depth_expression(*coords))

# all at once solver

PETSc.Sys.Print('### === --- AAOFunction --- === ###')
PETSc.Sys.Print('')

chunk_aaofunc = asQ.AllAtOnceFunction(chunk_ensemble, chunk_partition, W)
chunk_aaofunc.assign(winitial)

PETSc.Sys.Print('### === --- AAOForm --- === ###')
PETSc.Sys.Print('')

theta = 0.5
chunk_aaoform = asQ.AllAtOnceForm(chunk_aaofunc, dt, theta,
                                  form_mass, form_function)

PETSc.Sys.Print('### === --- AAOSolver --- === ###')
PETSc.Sys.Print('')

chunk_solver = asQ.AllAtOnceSolver(chunk_aaoform, chunk_aaofunc,
                                   solver_parameters=sparameters_diag,
                                   appctx=appctx)

# which chunk_id is holding which part of the total timeseries?
chunk_indexes = np.array([*range(args.nchunks)], dtype=int)

# we need to make this an array so we can send it via mpi
convergence_residual = np.array([0.])

# first mpi rank on each chunk (assumes all chunks are equal size):
chunk_root = lambda c: c*chunk_ensemble.global_comm.size

# am I the chunk with the earliest timesteps?
earliest = lambda: (chunk_id == chunk_indexes[0])

# update chunk ics from previous chunk
uprev = fd.Function(W)


def update_chunk_halos(uhalo):
    chunk_begin = chunk_aaofunc.layout.is_local(0)
    chunk_end = chunk_aaofunc.layout.is_local(-1)

    global_rank = global_ensemble.ensemble_comm.rank
    global_size = global_ensemble.ensemble_comm.size

    # ring communication so the first chunk can
    # pick up after last chunk after convergence

    # send forward last step of chunk
    if chunk_end:
        dest = (global_rank + 1) % global_size
        global_ensemble.send(chunk_aaofunc[-1], dest=dest, tag=dest)

    # recv updated ics from previous chunk
    if chunk_begin:
        source = (global_rank - 1) % global_size
        global_ensemble.recv(uhalo, source=source, tag=global_rank)

    # broadcast new ics to all ranks
    chunk_ensemble.bcast(uhalo)


PETSc.Sys.Print('### === --- Calculating parallel solution --- === ###')
PETSc.Sys.Print('')

nconverged = 0

PETSc.Sys.Print('### === --- Initialising all chunks --- === ###')
PETSc.Sys.Print('')
sleep_time = 0.01
for j in range(args.nchunks):
    PETSc.Sys.Print(f'=== --- Initial nonlinear sweep {j} --- ===    ')
    PETSc.Sys.Print('')

    # only smooth chunks that the first sweep has reached

    if args.serial:
        for i in range(j+1):
            PETSc.Sys.Print(f'--- Calculating chunk {i} ---        ')

            global_comm.Barrier()
            sleep(sleep_time)
            if chunk_id == i:
                chunk_solver.solve()
            global_comm.Barrier()
            sleep(sleep_time)

            PETSc.Sys.Print("")
    else:
         chunk_solver.solve()

    # propogate solution
    update_chunk_halos(uprev)

    # update initial condition guess for later chunks
    if chunk_id != 0:
        chunk_aaofunc.initial_condition.assign(uprev)

    # initial guess in front of first sweep is persistence forecast
    if chunk_id > j:
        chunk_aaofunc.assign(chunk_aaofunc.initial_condition)

PETSc.Sys.Print('### === --- All chunks initialised  --- === ###')
PETSc.Sys.Print('')

solver_time = []

for j in range(args.nsweeps):
    stime = MPI.Wtime()
    solver_time.append(stime)

    PETSc.Sys.Print(f'=== --- Calculating nonlinear sweep {j} --- ===    ')
    PETSc.Sys.Print('')

    # 1) one smoothing application on each chunk
    if args.serial:
        for i in range(args.nchunks):
            PETSc.Sys.Print(f'--- Calculating chunk {i} on solver {chunk_indexes[i]} ---        ')

            global_comm.Barrier()
            sleep(sleep_time)
            if chunk_id == chunk_indexes[i]:
                chunk_solver.solve()
            global_comm.Barrier()
            sleep(sleep_time)

            PETSc.Sys.Print("")

    else:
        chunk_solver.solve()

    # 2) update ics of each chunk from previous chunk
    update_chunk_halos(uprev)

    # everyone uses latest ic guesses, except chunk
    # with earliest timesteps (already has 'exact' ic)
    if not earliest():
        chunk_aaofunc.initial_condition.assign(uprev)

    # 3) check convergence of earliest chunk
    if earliest():
        chunk_aaoform.assemble()
        with chunk_aaoform.F.global_vec_ro() as rvec:
            res = rvec.norm()
        convergence_residual[0] = res

    # rank 0 on the earliest chunk tells everyone if they've converged
    global_ensemble.global_comm.Bcast(convergence_residual,
                                      root=chunk_root(chunk_indexes[0]))

    # update and report
    iteration_converged = convergence_residual[0] < patol
    if iteration_converged:
        nconverged += 1
        PETSc.Sys.Print(f">>> Chunk {nconverged-1} converged with residual {convergence_residual[0]:.3e}")
    else:
        PETSc.Sys.Print(f">>> Chunk {nconverged-1} did not converge with residual {convergence_residual[0]:.3e}")

    converged_time = nconverged*chunk_length*args.dt
    PETSc.Sys.Print(f">>> Converged time: {converged_time} hours")
    PETSc.Sys.Print('')

    etime = MPI.Wtime()
    stime = solver_time[-1]
    duration = etime - stime
    solver_time[-1] = duration
    PETSc.Sys.Print(f'Sweep solution time: {duration}')
    PETSc.Sys.Print('')

    # 4) stop iterating if we've reached the end
    if nconverged >= args.nwindows:
        PETSc.Sys.Print(f"Finished iterating to {args.nwindows} windows.")
        PETSc.Sys.Print('')
        break

    # 5) shuffle and restart if we haven't reached the end
    if iteration_converged:
        # earliest chunk_id becomes last chunk
        if earliest():
            chunk_aaofunc.assign(uprev)

        # update record of which chunk_id is in which position
        for i in range(args.nchunks):
            chunk_indexes[i] = (chunk_indexes[i] + 1) % args.nchunks

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

PETSc.Sys.Print(f'Total solution time: {sum(solver_time)}')
PETSc.Sys.Print(f'Average sweep solution time: {sum(solver_time)/len(solver_time)}')
PETSc.Sys.Print(f'Average chunk solution time: {sum(solver_time)/(nconverged)}')
PETSc.Sys.Print(f'Average timestep solution time: {sum(solver_time)/(nconverged*chunk_length)}')
PETSc.Sys.Print('')
