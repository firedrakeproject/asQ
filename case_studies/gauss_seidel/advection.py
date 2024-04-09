from firedrake.petsc import PETSc

import asQ
import firedrake as fd

from time import sleep  # noqa: F401
import numpy as np
from math import sqrt, pi, cos, sin

PETSc.Sys.popErrorHandler()

# get command arguments
import argparse
parser = argparse.ArgumentParser(
    description='DG scalar advection testcase for ParaDiag solver using a pipelined nonlinear Gauss-Seidel method.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--nx', type=int, default=16, help='Number of cells along each square side.')
parser.add_argument('--cfl', type=float, default=0.8, help='Convective CFL number.')
parser.add_argument('--angle', type=float, default=pi/6, help='Angle of the convective velocity.')
parser.add_argument('--degree', type=int, default=1, help='Degree of the scalar spaces.')
parser.add_argument('--width', type=float, default=0.2, help='Width of the Gaussian bump.')
parser.add_argument('--nwindows', type=int, default=1, help='Total number of time-windows.')
parser.add_argument('--nchunks', type=int, default=4, help='Number of chunks to solve simultaneously.')
parser.add_argument('--nsweeps', type=int, default=4, help='Number of nonlinear sweeps.')
parser.add_argument('--nsmooth', type=int, default=1, help='Number of nonlinear iterations per chunk at each sweep.')
parser.add_argument('--atol', type=float, default=1e-6, help='Average atol of each timestep.')
parser.add_argument('--nslices', type=int, default=2, help='Number of time-slices per time-window.')
parser.add_argument('--slice_length', type=int, default=2, help='Number of timesteps per time-slice.')
parser.add_argument('--alpha', type=float, default=1e-1, help='Circulant coefficient.')
parser.add_argument('--theta', type=float, default=0.5, help='Parameter for the implicit theta timestepping method.')
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

# Calculate the timestep from the CFL number
umax = 1.
dx = 1./args.nx
dt = args.cfl*dx/umax

# # # === --- domain and FE spaces --- === # # #

mesh = fd.PeriodicUnitSquareMesh(args.nx, args.nx, quadrilateral=True, comm=chunk_ensemble.comm)

V = fd.FunctionSpace(mesh, "DQ", args.degree)

# # # === --- initial conditions --- === # # #

x, y = fd.SpatialCoordinate(mesh)


def radius(x, y):
    return fd.sqrt(pow(x-0.5, 2) + pow(y-0.5, 2))


def gaussian(x, y):
    return fd.exp(-0.5*pow(radius(x, y)/args.width, 2))


# Gaussian bump centred at (0.5, 0.5)
q0 = fd.Function(V, name="scalar_initial")
q0.interpolate(1 + gaussian(x, y))

# The advecting velocity at angle to the x-axis
u = fd.Constant(fd.as_vector((umax*cos(args.angle), umax*sin(args.angle))))

# # # === --- finite element forms --- === # # #


# The time-derivative mass form for the scalar advection equation.
# asQ assumes that the mass form is linear so here
# q is a TrialFunction and phi is a TestFunction
def form_mass(q, phi):
    return phi*q*fd.dx


# The DG advection form for the scalar advection equation.
# asQ assumes that the function form is nonlinear so here
# q is a Function and phi is a TestFunction
def form_function(q, phi, t):
    # upwind switch
    n = fd.FacetNormal(mesh)
    un = fd.Constant(0.5)*(fd.dot(u, n) + abs(fd.dot(u, n)))

    # integration over element volume
    int_cell = q*fd.div(phi*u)*fd.dx

    # integration over internal facets
    int_facet = (phi('+')-phi('-'))*(un('+')*q('+')-un('-')*q('-'))*fd.dS

    return int_facet - int_cell


# # # === --- PETSc solver parameters --- === # # #

snes_linear_params = {
    'type': 'ksponly',
    'lag_jacobian': -2,
    'lag_jacobian_persists': None,
    'lag_preconditioner': -2,
    'lag_preconditioner_persists': None,
}

# parameters for the implicit diagonal solve in step-(b)
block_parameters = {
    'snes': snes_linear_params,
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps'
}

atol = args.atol
patol = sqrt(chunk_length)*atol
sparameters_diag = {
    'snes': {
        'monitor': None,
        'converged_reason': None,
        'atol': patol,
        'rtol': 1e-10,
        'stol': 1e-12,
        'max_it': args.nsmooth,
        'convergence_test': 'skip',
    },
    'mat_type': 'matfree',
    'ksp_type': 'preonly',
    'ksp': {
        'rtol': 1e-2,
        'atol': patol,
    },
    'pc_type': 'python',
    'pc_python_type': 'asQ.CirculantPC',
    'diagfft_alpha': args.alpha,
    'diagfft_state': 'linear',
    'aaos_jacobian_state': 'linear'
}
sparameters_diag['snes'].update(snes_linear_params)

for i in range(chunk_length):
    sparameters_diag['diagfft_block_'+str(i)+'_'] = block_parameters

# function spaces and initial conditions

# all at once solver

chunk_aaofunc = asQ.AllAtOnceFunction(chunk_ensemble, chunk_partition, V)
chunk_aaofunc.assign(q0)

theta = 0.5
chunk_aaoform = asQ.AllAtOnceForm(chunk_aaofunc, dt, theta,
                                  form_mass, form_function)

chunk_solver = asQ.AllAtOnceSolver(chunk_aaoform, chunk_aaofunc,
                                   solver_parameters=sparameters_diag)

# which chunk_id is holding which part of the total timeseries?
chunk_indexes = np.array([*range(args.nchunks)], dtype=int)

# we need to make this an array so we can send it via mpi
convergence_flag = np.array([False], dtype=bool)

# first mpi rank on each chunk (assumes all chunks are equal size):
chunk_root = lambda c: c*chunk_ensemble.global_comm.size

# am I the chunk with the earliest timesteps?
earliest = lambda: (chunk_id == chunk_indexes[0])

# update chunk ics from previous chunk
uprev = fd.Function(V)


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
    PETSc.Sys.Print(f'    === --- Initial nonlinear sweep {j} --- ===    ')
    PETSc.Sys.Print('')

    # only smooth chunks that the first sweep has reached

    if args.serial:
        for i in range(j+1):
            PETSc.Sys.Print(f'        --- Calculating chunk {i} ---        ')

            global_comm.Barrier()
            sleep(sleep_time)
            if chunk_id == i:
                chunk_solver.solve()
            global_comm.Barrier()
            sleep(sleep_time)

            PETSc.Sys.Print("")
    else:
        if chunk_id < j+1:
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

for j in range(args.nsweeps):
    PETSc.Sys.Print(f'    === --- Calculating nonlinear sweep {j} --- ===    ')
    PETSc.Sys.Print('')

    # 1) one smoothing application on each chunk
    if args.serial:
        for i in range(args.nchunks):
            PETSc.Sys.Print(f'        --- Calculating chunk {i} on solver {chunk_indexes[i]} ---        ')

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
        convergence_flag[0] = (res < patol)

    # rank 0 on the earliest chunk tells everyone if they've converged
    global_ensemble.global_comm.Bcast(convergence_flag,
                                      root=chunk_root(chunk_indexes[0]))

    # update and report
    if convergence_flag[0]:
        nconverged += 1

    converged_time = nconverged*chunk_length*dt
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f">>> Converged chunks: {nconverged}.")
    PETSc.Sys.Print(f">>> Converged time: {converged_time} hours.")
    PETSc.Sys.Print('')

    # 4) stop iterating if we've reached the end
    if nconverged >= args.nwindows:
        PETSc.Sys.Print(f"Finished iterating to {args.nwindows} windows.")
        PETSc.Sys.Print('')
        break

    # 5) shuffle and restart if we haven't reached the end
    if convergence_flag[0]:
        # earliest chunk_id becomes last chunk
        if earliest():
            chunk_aaofunc.assign(uprev)

        # update record of which chunk_id is in which position
        for i in range(args.nchunks):
            chunk_indexes[i] = (chunk_indexes[i] + 1) % args.nchunks

global_comm.Barrier()
sleep(sleep_time)
if earliest() and chunk_aaofunc.layout.is_local(-1):
    from utils.serial import SerialMiniApp
    serialapp = SerialMiniApp(dt, theta, q0, form_mass, form_function, block_parameters)
    serialapp.solve(nt=nconverged*chunk_length)
    PETSc.Sys.Print(f"serial error: {fd.errornorm(serialapp.w0, chunk_aaofunc[-1])}", comm=chunk_ensemble.comm)
    PETSc.Sys.Print('')
global_comm.Barrier()
sleep(sleep_time)

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
