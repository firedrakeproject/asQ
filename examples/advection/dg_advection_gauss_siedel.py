from math import pi, cos, sin, sqrt

import firedrake as fd
from firedrake.petsc import PETSc
import asQ

import argparse

parser = argparse.ArgumentParser(
    description='ParaDiag timestepping for scalar advection of a Gaussian bump in a periodic square with DG in space and implicit-theta in time. Based on the Firedrake DG advection example https://www.firedrakeproject.org/demos/DG_advection.py.html',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--nx', type=int, default=16, help='Number of cells along each square side.')
parser.add_argument('--cfl', type=float, default=0.8, help='Convective CFL number.')
parser.add_argument('--angle', type=float, default=pi/6, help='Angle of the convective velocity.')
parser.add_argument('--degree', type=int, default=1, help='Degree of the scalar spaces.')
parser.add_argument('--theta', type=float, default=0.5, help='Parameter for the implicit theta timestepping method.')
parser.add_argument('--width', type=float, default=0.2, help='Width of the Gaussian bump.')
parser.add_argument('--nwindows', type=int, default=1, help='Total number of time-windows.')
parser.add_argument('--nchunks', type=int, default=4, help='Number of chunks to solve simultaneously.')
parser.add_argument('--nsweeps', type=int, default=4, help='Number of nonlinear sweeps.')
parser.add_argument('--nsmooth', type=int, default=1, help='Number of nonlinear iterations per chunk at each sweep.')
parser.add_argument('--nchecks', type=int, default=1, help='Maximum number of chunks allowed to converge after each sweep.')
parser.add_argument('--ninitialise', type=int, default=0, help='Number of sweeps before checking convergence.')
parser.add_argument('--insurance_freq', type=int, default=0, help='Frequency of sweeps where no convergence is allowed.')
parser.add_argument('--atol', type=float, default=1e-8, help='Average atol of each timestep.')
parser.add_argument('--nslices', type=int, default=2, help='Number of time-slices per time-window.')
parser.add_argument('--slice_length', type=int, default=2, help='Number of timesteps per time-slice.')
parser.add_argument('--alpha', type=float, default=0.0001, help='Circulant coefficient.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

# The time partition describes how many timesteps are included on each time-slice of the ensemble
# Here we use the same number of timesteps on each slice, but they can be different

time_partition = tuple(args.slice_length for _ in range(args.nslices))
chunk_length = sum(time_partition)

# Calculate the timestep from the CFL number
umax = 1.
dx = 1./args.nx
dt = args.cfl*dx/umax

# The Ensemble with the spatial and time communicators
global_comm = fd.COMM_WORLD
ensemble = asQ.create_ensemble(time_partition, comm=global_comm)

# # # === --- domain --- === # # #

# The mesh needs to be created with the spatial communicator
mesh = fd.PeriodicUnitSquareMesh(args.nx, args.nx, quadrilateral=True, comm=ensemble.comm)

# We use a discontinuous Galerkin space for the advected scalar
# and a continuous Galerkin space for the advecting velocity field
V = fd.FunctionSpace(mesh, "DQ", args.degree)
W = fd.VectorFunctionSpace(mesh, "CG", args.degree+1)

# # # === --- initial conditions --- === # # #

x, y = fd.SpatialCoordinate(mesh)


def radius(x, y):
    return fd.sqrt(pow(x-0.5, 2) + pow(y-0.5, 2))


def gaussian(x, y):
    return fd.exp(-0.5*pow(radius(x, y)/args.width, 2))


# The scalar initial conditions are a Gaussian bump centred at (0.5, 0.5)
q0 = fd.Function(V, name="scalar_initial")
q0.interpolate(1 + gaussian(x, y))

# The advecting velocity field is constant and directed at an angle to the x-axis
u = fd.Function(W, name='velocity')
u.interpolate(fd.as_vector((umax*cos(args.angle), umax*sin(args.angle))))

# We create an all-at-once function representing the timeseries solution of the scalar

aaofunc = asQ.AllAtOnceFunction(ensemble, time_partition, V)
aaofunc.assign(q0)


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


# Construct the all-at-once form representing the coupled equations for
# the implicit-theta method at every timestep of the timeseries.

aaoform = asQ.AllAtOnceForm(aaofunc, dt, args.theta,
                            form_mass, form_function)

# # # === --- PETSc solver parameters --- === # # #


# The PETSc solver parameters used to solve the
# blocks in step (b) of inverting the ParaDiag matrix.
block_parameters = {
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps'
}

# The PETSc solver parameters for solving the all-at-once system.
# The python preconditioner 'asQ.DiagFFTPC' applies the ParaDiag matrix.
#
# The equation is linear so we can use 'snes_type': 'ksponly' and
# use your favourite Krylov method (if a Krylov method is used on
# the blocks then the outer Krylov method must be either flexible
# or Richardson).

patol = sqrt(chunk_length)*args.atol
solver_parameters = {
    'snes_type': 'ksponly',
    'snes_convergence_test': 'skip',
    'mat_type': 'matfree',
    'ksp_type': 'preonly',
    'ksp': {
        'monitor': None,
        'converged_rate': None,
        'rtol': 1e-100,
        'atol': patol,
        'stol': 1e-12,
    },
    'pc_type': 'python',
    'pc_python_type': 'asQ.DiagFFTPC',
    'diagfft_alpha': args.alpha,
    'diagfft_state': 'linear',
    'aaos_jacobian_state': 'linear',
}

# We need to add a block solver parameters dictionary for each block.
# Here they are all the same but they could be different.
for i in range(aaofunc.ntimesteps):
    solver_parameters['diagfft_block_'+str(i)+'_'] = block_parameters

# Create a solver object to set up and solve the (possibly nonlinear) problem
# for the timeseries in the all-at-once function.
aaosolver = asQ.AllAtOnceSolver(aaoform, aaofunc,
                                solver_parameters=solver_parameters)

chunks = tuple(aaofunc.copy() for _ in range(args.nchunks))

nconverged = 0
PETSc.Sys.Print('')

for j in range(args.nsweeps):
    PETSc.Sys.Print(f'### === --- Calculating nonlinear sweep {j} --- === ###')
    PETSc.Sys.Print('')

    # only iterate chunks that the wavefront has reached
    active_chunks = min(j+1, args.nchunks)

    for i in range(active_chunks):
        PETSc.Sys.Print(f'        --- Calculating chunk {i} ---        ')

        # 1) load chunk i solution
        aaofunc.assign(chunks[i])

        # 2) one iteration of chunk i
        aaosolver.solve()
        PETSc.Sys.Print("")

        # 3) save chunk i solution
        chunks[i].assign(aaofunc)

    # 4) pass forward ic to next chunk
    for i in range(active_chunks):
        if i < args.nchunks-1:
            chunks[i].bcast_field(-1, chunks[i+1].initial_condition)

    # skip convergence test during the ramp up and periodically after
    ramp_up = j < (args.nchunks + args.ninitialise)
    insurance_step = (j % args.insurance_freq) == 0 if args.insurance_freq > 1 else False

    if ramp_up or insurance_step:
        continue

    # check if first chunk has converged
    aaoform.assemble(chunks[0])
    with aaoform.F.global_vec_ro() as rvec:
        res = rvec.norm()

    if res < patol:
        PETSc.Sys.Print(f">>> Chunk 0 converged with function norm = {res:.6e} <<<")
        nconverged += 1

        # pass on the most up to date initial condition for the next chunk
        chunks[0].bcast_field(-1, chunks[1].initial_condition)

        # shuffle chunks down
        for i in range(args.nchunks-1):
            chunks[i].assign(chunks[i+1])

        # reset last chunk to start from end of series
        chunks[-1].bcast_field(-1, chunks[-1].initial_condition)
        chunks[-1].assign(chunks[-1].initial_condition)

    PETSc.Sys.Print('')
    PETSc.Sys.Print(f">>> Converged chunks: {int(nconverged)}.")
    converged_time = nconverged*chunk_length*dt
    PETSc.Sys.Print(f">>> Converged time: {converged_time} hours.")
    PETSc.Sys.Print('')

    if nconverged >= args.nwindows:
        PETSc.Sys.Print(f"Finished iterating to {args.nwindows}.")
        break

nsweeps = j

pc_block_its = aaosolver.jacobian.pc.block_iterations
pc_block_its.synchronise()
pc_block_its = pc_block_its.data(deepcopy=True)
pc_block_its = pc_block_its/args.nchunks

niterations = nsweeps*args.nsmooth

PETSc.Sys.Print(f"Number of chunks: {args.nchunks}")
PETSc.Sys.Print(f"Maximum number of sweeps: {args.nsweeps}")
PETSc.Sys.Print(f"Actual number of sweeps: {nsweeps}")
PETSc.Sys.Print(f"Number of chunks converged: {int(nconverged)}")
PETSc.Sys.Print(f"Number of chunks converged per sweep: {nconverged/nsweeps}")
PETSc.Sys.Print(f"Number of sweeps per converged chunk: {nsweeps/nconverged if nconverged else 'n/a'}")
PETSc.Sys.Print(f"Number of iterations per converged chunk: {niterations/nconverged if nconverged else 'n/a'}")
PETSc.Sys.Print(f"Number of timesteps per iteration: {nconverged*chunk_length/niterations}")
PETSc.Sys.Print(f'Block iterations: {pc_block_its}')
PETSc.Sys.Print(f'Block iterations per block solve: {pc_block_its/niterations}')
