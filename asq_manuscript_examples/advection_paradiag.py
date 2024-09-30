from math import pi, cos, sin
from utils.timing import SolverTimer
import firedrake as fd
from firedrake.petsc import PETSc
import asQ
from argparse import ArgumentParser
from argparse_formatter import DefaultsAndRawTextFormatter

parser = ArgumentParser(
    description='ParaDiag timestepping for scalar advection of a Gaussian bump in a periodic square with DG in space and implicit-theta in time.',
    epilog="""\
Optional PETSc command line arguments:

   -circulant_alpha :float: The circulant parameter to use in the preconditioner. Default 1e-4.
   -ksp_rtol :float: The relative residual drop required for convergence. Default 1e-11.
                     See https://petsc.org/release/manualpages/KSP/KSPSetTolerances/
   -ksp_type :str: The Krylov method to use for the all-at-once iterations. Default 'richardson'.
                   Alternatives include gmres or fgmres.
                   See https://petsc.org/release/manualpages/KSP/KSPSetType/
""",
    formatter_class=DefaultsAndRawTextFormatter
)
parser.add_argument('--nx', type=int, default=16, help='Number of cells along each side of the square.')
parser.add_argument('--cfl', type=float, default=0.8, help='Convective CFL number.')
parser.add_argument('--angle', type=float, default=pi/6, help='Angle of the convective velocity to the horizontal.')
parser.add_argument('--degree', type=int, default=1, help='Degree of the scalar space.')
parser.add_argument('--theta', type=float, default=0.5, help='Parameter for the implicit theta timestepping method.')
parser.add_argument('--width', type=float, default=0.2, help='Width of the Gaussian bump.')
parser.add_argument('--nwindows', type=int, default=1, help='Number of time-windows to solve.')
parser.add_argument('--nslices', type=int, default=1, help='Number of time-slices in the all-at-once system. Must divide the number of MPI ranks exactly.')
parser.add_argument('--slice_length', type=int, default=4, help='Number of timesteps per time-slice. Total number of timesteps in the all-at-once system is nslices*slice_length.')
parser.add_argument('--metrics_dir', type=str, default='metrics/advection', help='Directory to save paradiag output metrics to.')
parser.add_argument('--show_args', action='store_true', help='Print all the arguments when the script starts.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

# The time partition describes how many timesteps
# are included on each time-slice of the ensemble

time_partition = tuple(args.slice_length for _ in range(args.nslices))
window_length = sum(time_partition)
nsteps = args.nwindows*window_length

# Calculate the timestep size dt from the CFL number
umax = 1.
dx = 1./args.nx
dt = args.cfl*dx/umax

# The Ensemble with the spatial and time communicators
ensemble = asQ.create_ensemble(time_partition)

# # # === --- domain --- === # # #

# The mesh needs to be created with the spatial communicator
mesh = fd.PeriodicUnitSquareMesh(args.nx, args.nx, quadrilateral=True, comm=ensemble.comm)

# We use a discontinuous Galerkin space for the advected scalar
V = fd.FunctionSpace(mesh, "DQ", args.degree)

# # # === --- initial conditions --- === # # #

x, y = fd.SpatialCoordinate(mesh)


# The scalar initial condition is a Gaussian bump centred at (0.5, 0.5)
def radius(x, y):
    return fd.sqrt(pow(x-0.5, 2) + pow(y-0.5, 2))


def gaussian(x, y):
    return fd.exp(-0.5*pow(radius(x, y)/args.width, 2))


q0 = fd.Function(V, name="scalar_initial")
q0.interpolate(1 + gaussian(x, y))

# The advecting velocity field is constant and directed at an angle to the x-axis
u = fd.Constant(fd.as_vector((umax*cos(args.angle), umax*sin(args.angle))))

# # # === --- finite element forms --- === # # #


# The time-derivative mass form for the scalar advection equation.
# asQ assumes that the mass form is linear so here
# q is a TrialFunction and phi is a TestFunction
def form_mass(q, phi):
    return phi*q*fd.dx


# The DG advection form for the scalar advection equation.
# asQ assumes that the function form may be nonlinear so
# here q is a Function and phi is a TestFunction
def form_function(q, phi, t):
    # upwind switch
    n = fd.FacetNormal(mesh)
    un = fd.Constant(0.5)*(fd.dot(u, n) + abs(fd.dot(u, n)))

    # integration over element volume
    int_cell = q*fd.div(phi*u)*fd.dx

    # integration over internal facets
    int_facet = (phi('+') - phi('-'))*(un('+')*q('+') - un('-')*q('-'))*fd.dS

    return int_facet - int_cell


# # # === --- PETSc solver parameters --- === # # #

# The PETSc solver parameters used to solve the
# blocks in step (b) of inverting the ParaDiag matrix.
# MUMPS is a parallel direct solver so spatial parallelism can be used
block_parameters = {
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
}

# The PETSc solver parameters for solving the all-at-once system.
# The python preconditioner 'asQ.CirculantPC' applies the ParaDiag matrix.
#
# The equation is linear so we can either:
# a) Solve it using a preconditioned Krylov method:
#    P^{-1}Au = P^{-1}b
#    The solver option for this is:
#    -ksp_type gmres
# b) Solve it with stationary iterations:
#    Pu_{k+1} = (P - A)u_{k} + b
#    The solver option for this is:
#    -ksp_type richardson

paradiag_parameters = {
    'snes_type': 'ksponly',      # only solve 1 "Newton iteration" per window (i.e. a linear problem)
    'ksp_type': 'richardson',    # stationary iterations
    'ksp': {
        'monitor': None,         # show the residual at every iteration
        'converged_rate': None,  # show the contraction rate once the linear solve has converged
        'rtol': 1e-11,           # relative residual tolerance
    },
    'pc_type': 'python',
    'pc_python_type': 'asQ.CirculantPC',  # the alpha-circulant preconditioner
    'circulant_alpha': 1e-4,              # set other values from command line using: -circulant_alpha <value>
    'circulant_block': block_parameters,  # options dictionary for the inner solve
    'circulant_state': 'linear',          # system is linear so don't update the preconditioner reference state
    'aaos_jacobian_state': 'linear',      # system is linear so don't update the jacobian reference state
}


# # # === --- Setup ParaDiag --- === # # #

# Give everything to the Paradiag object which will build the all-at-once system.
paradiag = asQ.Paradiag(ensemble=ensemble,
                        form_function=form_function,
                        form_mass=form_mass,
                        ics=q0, dt=dt, theta=args.theta,
                        time_partition=time_partition,
                        solver_parameters=paradiag_parameters)

# create a timer to profile the calculations
timer = SolverTimer()


# This function will be called before paradiag solves each time-window. We can use
# this to make the output a bit easier to read, and to time the window calculation
def window_preproc(paradiag, wndw, rhs):
    PETSc.Sys.Print(f'### === --- Calculating time-window {wndw} --- === ###')
    PETSc.Sys.Print('')
    # for now we are interested in timing only the solve, this
    # makes sure we don't time any synchronisation after prints.
    with PETSc.Log.Event("window_preproc.Coll_Barrier"):
        paradiag.ensemble.ensemble_comm.Barrier()
    timer.start_timing()


# This function will be called after paradiag solves each time-window. We can use
# this to finish the window calculation timing and print the result.
def window_postproc(paradiag, wndw, rhs):
    timer.stop_timing()
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'Window solution time: {round(timer.times[-1], 5)}')
    PETSc.Sys.Print('')


# Setup all solver objects. The firedrake DM and options management
# makes it difficult to setup some preconditioners without actually
# calling `solve`, so we just run once to set everything up.
PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Setting up solver and prefactoring --- === ###')
PETSc.Sys.Print('')
with PETSc.Log.Event("warmup_solve"):
    paradiag.solve(1)
PETSc.Sys.Print('')

# reset solution and iteration counts for timed solved
paradiag.reset_diagnostics()
aaofunc = paradiag.solver.aaofunc
aaofunc.bcast_field(-1, aaofunc.initial_condition)
aaofunc.assign(aaofunc.initial_condition)

PETSc.Sys.Print('### === --- Solving timeseries --- === ###')
PETSc.Sys.Print('')

# Solve nwindows of the all-at-once system
with PETSc.Log.Event("timed_solves"):
    paradiag.solve(args.nwindows,
                   preproc=window_preproc,
                   postproc=window_postproc)

# # # === --- Solver diagnostics --- === # # #

PETSc.Sys.Print('### === --- Iteration and timing results --- === ###')
PETSc.Sys.Print('')

asQ.write_paradiag_metrics(paradiag, directory=args.metrics_dir)

nw = paradiag.total_windows
nt = paradiag.total_timesteps
PETSc.Sys.Print(f'Total windows: {nw}')
PETSc.Sys.Print(f'Total timesteps: {nt}')
PETSc.Sys.Print('')

# Show the parallel partition sizes.
PETSc.Sys.Print(f'Total DoFs per window: {V.dim()*window_length}')
PETSc.Sys.Print(f'DoFs per timestep: {V.dim()}')
PETSc.Sys.Print(f'Total number of MPI ranks: {ensemble.global_comm.size}')
PETSc.Sys.Print(f'Number of MPI ranks per timestep: {mesh.comm.size}')
PETSc.Sys.Print(f'DoFs/rank: {V.dim()/mesh.comm.size}')
PETSc.Sys.Print(f'Complex block DoFs/rank: {2*V.dim()/mesh.comm.size}')
PETSc.Sys.Print('')

# paradiag collects a few iteration counts for us
lits = paradiag.linear_iterations
nlits = paradiag.nonlinear_iterations
blits = paradiag.block_iterations.data()

# Number of nonlinear iterations will be 1 per window for linear problems
PETSc.Sys.Print(f'Nonlinear iterations: {str(nlits).rjust(5)}  |  Iterations per window: {str(nlits/nw).rjust(5)}')

# Number of linear iterations of the all-at-once system, total and per window.
PETSc.Sys.Print(f'Linear iterations:    {str(lits).rjust(5)}  |  Iterations per window: {str(lits/nw).rjust(5)}')

# Number of iterations needed for each block in step-(b), total and per block solve
PETSc.Sys.Print(f'Total block linear iterations: {blits}')
PETSc.Sys.Print(f'Iterations per block solve: {blits/lits}')
PETSc.Sys.Print('')

# Timing measurements
PETSc.Sys.Print(timer.string(timesteps_per_solve=window_length,
                             total_iterations=paradiag.linear_iterations, ndigits=5))
PETSc.Sys.Print('')
