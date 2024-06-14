from firedrake.petsc import PETSc

from pyop2.mpi import MPI
import asQ
import firedrake as fd

from time import sleep  # noqa: F401
import numpy as np
from math import sqrt, pi, cos, sin

PETSc.Sys.popErrorHandler()

# get command arguments
import argparse
parser = argparse.ArgumentParser(
    description='Galewsky testcase for ParaDiag solver using a pipelined nonlinear Gauss-Seidel method.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--nx', type=int, default=16, help='Number of cells along each square side.')
parser.add_argument('--dt', type=float, default=0.5, help='Timestep in hours.')
parser.add_argument('--angle', type=float, default=pi/6, help='Angle of the convective velocity.')
parser.add_argument('--degree', type=int, default=1, help='Degree of the scalar and velocity spaces.')
parser.add_argument('--theta', type=float, default=0.5, help='Parameter for the implicit theta timestepping method.')
parser.add_argument('--width', type=float, default=0.2, help='Width of the Gaussian bump.')
parser.add_argument('--modes', type=int, default=4, help='Number of time-varying modes. 0 for constant-coefficient')
parser.add_argument('--nwindows', type=int, default=1, help='Total number of time-windows.')
parser.add_argument('--nchunks', type=int, default=4, help='Number of chunks to solve simultaneously.')
parser.add_argument('--nsweeps', type=int, default=4, help='Number of linear sweeps.')
parser.add_argument('--nsmooth', type=int, default=1, help='Number of linear iterations per chunk at each sweep.')
parser.add_argument('--atol', type=float, default=1e-4, help='Average atol of each timestep.')
parser.add_argument('--nslices', type=int, default=2, help='Number of time-slices per time-window.')
parser.add_argument('--slice_length', type=int, default=2, help='Number of timesteps per time-slice.')
parser.add_argument('--alpha', type=float, default=1e-4, help='Circulant coefficient.')
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

# Calculate the CFL from the timestep number
ubar = 1.
dx = 1./args.nx
dt = args.dt
cfl = ubar*dt/dx
T = dt*chunk_length
pi2 = fd.Constant(2*pi)

# frequency of velocity oscillations in time
omegas = [1.43, 2.27, 4.83, 6.94]

# how many points per wavelength for each spatial and temporal frequency
PETSc.Sys.Print(f"{cfl = }, {T = }, {chunk_length = }")
PETSc.Sys.Print(f"Periods: {[round(T*o,3) for o in omegas[:args.modes]]}")

PETSc.Sys.Print(f"Temporal resolution (ppm): {[round(1/(o*dt),3) for o in omegas[:args.modes]]}")
PETSc.Sys.Print(f"Spatial resolution (ppm): {round(1/dx,3)}, {round((1/2)/dx,3)}, {round((1/4)/dx,3)}, {round((1/6)/dx,3)}")

PETSc.Sys.Print('### === --- Create mesh --- === ###')
PETSc.Sys.Print('')

mesh = fd.PeriodicUnitSquareMesh(args.nx, args.nx,
                                 quadrilateral=True,
                                 comm=chunk_ensemble.comm)

PETSc.Sys.Print('### === --- Forms --- === ###')
PETSc.Sys.Print('')

# We use a discontinuous Galerkin space for the advected scalar
V = fd.FunctionSpace(mesh, "DQ", args.degree)

# # # === --- initial conditions --- === # # #

x, y = fd.SpatialCoordinate(mesh)


def radius(x, y):
    return fd.sqrt(pow(x-0.5, 2) + pow(y-0.5, 2))


def gaussian(x, y):
    return fd.exp(-0.5*pow(radius(x, y)/args.width, 2))


# The scalar initial conditions are a Gaussian bump centred at (0.5, 0.5)
q0 = fd.Function(V, name="scalar_initial")
q0.interpolate(1 + gaussian(x, y))


# The advecting velocity field oscillates around a mean value in time and space
def velocity(t):
    c = fd.as_vector((fd.Constant(ubar*cos(args.angle)), fd.Constant(ubar*sin(args.angle))))
    if args.modes > 0:
        c += fd.sin(pi2*omegas[0]*t-0.0)*fd.as_vector([+0.25*fd.sin(1*x*pi2+0.3), +0.20*fd.cos(1*y*pi2-0.9)])
    if args.modes > 1:
        c -= fd.cos(pi2*omegas[1]*t-0.7)*fd.as_vector([-0.05*fd.cos(2*x*pi2-0.8), +0.10*fd.sin(2*x*pi2+0.1)])
    if args.modes > 2:
        c += fd.sin(pi2*omegas[2]*t+0.6)*fd.as_vector([+0.15*fd.cos(4*x*pi2+0.0), -0.05*fd.sin(4*x*pi2-0.4)])
    if args.modes > 3:
        c -= fd.cos(pi2*omegas[3]*t-0.3)*fd.as_vector([-0.03*fd.cos(6*x*pi2-0.9), +0.08*fd.sin(6*x*pi2+0.2)])
    return c


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
    n = fd.FacetNormal(mesh)
    u = velocity(t)

    # upwind switch
    un = fd.Constant(0.5)*(fd.dot(u, n) + abs(fd.dot(u, n)))

    # integration over element volume
    int_cell = q*fd.div(phi*u)*fd.dx

    # integration over internal facets
    int_facet = (phi('+')-phi('-'))*(un('+')*q('+')-un('-')*q('-'))*fd.dS

    return int_facet - int_cell


PETSc.Sys.Print('### === --- Parameters --- === ###')
PETSc.Sys.Print('')

sparameters = {
    'ksp_type': 'preonly',
    'pc_type': 'lu',
}

atol = args.atol
patol = sqrt(chunk_length)*atol
sparameters_diag = {
    'snes_type': 'newtonls',
    'snes': {
        'linesearch_type': 'basic',
        'monitor': None,
        'converged_reason': None,
        'max_it': 1,
        'convergence_test': 'skip',
    },
    'mat_type': 'matfree',
    'ksp_type': 'gmres',
    'ksp': {
        # 'monitor_true_residual': None,
        'monitor': None,
        'converged_rate': None,
        'max_it': args.nsmooth,
        'converged_maxits': None,
        # 'atol': patol,
    },
    'pc_type': 'python',
    'pc_python_type': 'asQ.CirculantPC',
    'circulant_alpha': args.alpha,
    'circulant_block': sparameters,
    'circulant_state': 'window',
    'aaos_jacobian_state': 'current'
}

# all at once solver

PETSc.Sys.Print('### === --- AAOFunction --- === ###')
PETSc.Sys.Print('')

chunk_aaofunc = asQ.AllAtOnceFunction(chunk_ensemble, chunk_partition, V)
chunk_aaofunc.assign(q0)

PETSc.Sys.Print('### === --- AAOForm --- === ###')
PETSc.Sys.Print('')

theta = 0.5
chunk_aaoform = asQ.AllAtOnceForm(chunk_aaofunc, dt, theta,
                                  form_mass, form_function)
for i in range(chunk_id):
    chunk_aaoform.time_update()

PETSc.Sys.Print('### === --- AAOSolver --- === ###')
PETSc.Sys.Print('')

chunk_solver = asQ.AllAtOnceSolver(chunk_aaoform, chunk_aaofunc,
                                   solver_parameters=sparameters_diag)

# which chunk_id is holding which part of the total timeseries?
chunk_indexes = np.array([*range(args.nchunks)], dtype=int)

# we need to make this an array so we can send it via mpi
convergence_residual = np.array([0.])

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
    t0 = chunk_aaoform.t0.values()[0]
    if args.serial:
        for i in range(args.nchunks):
            PETSc.Sys.Print(f'--- Calculating chunk {i} on solver {chunk_indexes[i]} ---        ')
            global_comm.Barrier()
            sleep(sleep_time)

            if chunk_id == chunk_indexes[i]:

                # chunk_aaoform.assemble()
                # with chunk_aaoform.F.global_vec_ro() as rvec:
                #     chunk_residual = rvec.norm()
                #     PETSc.Sys.Print(f">>> Solver {chunk_id} residual = {chunk_residual}",
                #                     comm=chunk_ensemble.global_comm)

                chunk_solver.solve()

                # chunk_aaoform.assemble()
                # with chunk_aaoform.F.global_vec_ro() as rvec:
                #     chunk_residual = rvec.norm()
                #     PETSc.Sys.Print(f">>> Solver {chunk_id} residual = {chunk_residual}",
                #                     comm=chunk_ensemble.global_comm)

            global_comm.Barrier()
            sleep(sleep_time)
            PETSc.Sys.Print("")

    else:
        chunk_solver.solve()
    assert chunk_aaoform.t0.values()[0] == t0, f'{chunk_id = }'

    # 2) update ics of each chunk from previous chunk
    update_chunk_halos(uprev)

    # everyone uses latest ic guesses, except chunk
    # with earliest timesteps (already has 'exact' ic)
    if not earliest():
        chunk_aaofunc.initial_condition.assign(uprev)
    else:
        PETSc.Sys.Print(f">>> Solver {chunk_id} is earliest",
                        comm=chunk_ensemble.global_comm)
    assert chunk_aaoform.t0.values()[0] == t0, f'{chunk_id = }'


    # 3) check convergence of earliest chunk
    if earliest():
        chunk_aaoform.assemble()
        with chunk_aaoform.F.global_vec_ro() as rvec:
            res = rvec.norm()
        convergence_residual[0] = res
        # convergence_residual[0] = chunk_solver.snes.ksp.getResidualNorm()
    assert chunk_aaoform.t0.values()[0] == t0, f'{chunk_id = }'

    # rank 0 on the earliest chunk tells everyone if they've converged
    global_ensemble.global_comm.Bcast(convergence_residual,
                                      root=chunk_root(chunk_indexes[0]))

    # update and report
    iteration_converged = convergence_residual[0] < patol
    if iteration_converged:
        PETSc.Sys.Print(f">>> Chunk {nconverged} converged with residual {convergence_residual[0]:.3e}")
        nconverged += 1
    else:
        PETSc.Sys.Print(f">>> Chunk {nconverged} did not converge with residual {convergence_residual[0]:.3e}")

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
            # shift time to the end of the series
            for i in range(args.nchunks):
                chunk_aaoform.time_update()

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
