from math import pi, cos, sin
from utils.timing import SolverTimer
import firedrake as fd
from firedrake.petsc import PETSc
from utils.serial import SerialMiniApp
from argparse import ArgumentParser
from argparse_formatter import DefaultsAndRawTextFormatter

parser = ArgumentParser(
    description='Serial timestepping for scalar advection of a Gaussian bump in a periodic square with DG in space and implicit-theta in time.',
    formatter_class=DefaultsAndRawTextFormatter
)
parser.add_argument('--nx', type=int, default=16, help='Number of cells along each side of the square.')
parser.add_argument('--cfl', type=float, default=0.8, help='Convective CFL number.')
parser.add_argument('--angle', type=float, default=pi/6, help='Angle of the convective velocity to the horizontal.')
parser.add_argument('--degree', type=int, default=1, help='Degree of the scalar space.')
parser.add_argument('--theta', type=float, default=0.5, help='Parameter for the implicit theta timestepping method.')
parser.add_argument('--width', type=float, default=0.2, help='Width of the Gaussian bump.')
parser.add_argument('--nt', type=int, default=4, help='Number of timesteps to solve.')
parser.add_argument('--show_args', action='store_true', help='Print all the arguments when the script starts.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

# Calculate the timestep from the CFL number
umax = 1.
dx = 1./args.nx
dt = args.cfl*dx/umax

# # # === --- domain --- === # # #

# Quadrilateral square mesh
mesh = fd.PeriodicUnitSquareMesh(args.nx, args.nx, quadrilateral=True)

# We use a discontinuous Galerkin space for the advected scalar
V = fd.FunctionSpace(mesh, "DQ", args.degree)

# # # === --- initial conditions --- === # # #

x, y = fd.SpatialCoordinate(mesh)


# The scalar initial conditions are a Gaussian bump centred at (0.5, 0.5)
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


# The PETSc solver parameters used to solve the
# serial-in-time blocks
block_parameters = {
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
}

serial_parameters = {
    'snes': {  # 'ksponly' means we are solving a linear problem, and lagging prevents the block from being refactorised every timestep.
        'type': 'ksponly',
        'lag_jacobian': -2,
        'lag_jacobian_persists': None,
        'lag_preconditioner': -2,
        'lag_preconditioner_persists': None,
    },
    'ksp': {
        'monitor': None,         # show the residual at every iteration
        'converged_rate': None,  # show the contraction rate once the linear solve has converged
        'rtol': 1e-11,
    },
}
serial_parameters.update(block_parameters)


# # # === --- Setup the solver --- === # # #


# The SerialMiniApp class will set up the implicit-theta system
# for the serial-in-time method.
miniapp = SerialMiniApp(dt, args.theta, q0,
                        form_mass, form_function,
                        serial_parameters)

# create a timer to profile the calculations
timer = SolverTimer()

PETSc.Sys.Print('### === --- Timestepping loop --- === ###')
PETSc.Sys.Print('')
linear_its = 0
nonlinear_its = 0


# This function will be called before solving each timestep. We can use
# this to make the output a bit easier to read, and to time the calculation
def preproc(app, step, t):
    PETSc.Sys.Print(f'### === --- Calculating timestep {step} --- === ###')
    PETSc.Sys.Print('')
    # for now we are interested in timing only the solve, this
    # makes sure we don't time any synchronisation after prints.
    with PETSc.Log.Event("timestep_preproc.Coll_Barrier"):
        mesh.comm.Barrier()
    timer.start_timing()


# This function will be called after solving each timestep. We can use
# this to finish the timestep calculation timing and print the result,
# and to record the number of iterations.
def postproc(app, step, t):
    timer.stop_timing()
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'Timestep solution time: {round(timer.times[-1], 5)}')
    PETSc.Sys.Print('')

    global linear_its
    global nonlinear_its
    linear_its += app.nlsolver.snes.getLinearSolveIterations()
    nonlinear_its += app.nlsolver.snes.getIterationNumber()


# run one timestep to get all solver objects set up e.g. factorisations

# Setup all solver objects. The firedrake DM and options management
# makes it difficult to setup some preconditioners without actually
# calling `solve`, so we just run once to set everything up.
PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Setting up solver and prefactoring --- === ###')
PETSc.Sys.Print('')
with PETSc.Log.Event("warmup_solve"):
    miniapp.solve(1)
PETSc.Sys.Print('')

# reset solution
miniapp.w0.assign(q0)
miniapp.w1.assign(q0)

PETSc.Sys.Print('### === --- Solving timeseries --- === ###')
PETSc.Sys.Print('')

# Solve nt timesteps
with PETSc.Log.Event("timed_solves"):
    miniapp.solve(args.nt,
                  preproc=preproc,
                  postproc=postproc)

# # # === --- Solver diagnostics --- === # # #

PETSc.Sys.Print('### === --- Iteration and timing results --- === ###')
PETSc.Sys.Print('')

# parallelism
PETSc.Sys.Print(f'DoFs per timestep: {V.dim()}')
PETSc.Sys.Print(f'Number of MPI ranks per timestep: {mesh.comm.size}')
PETSc.Sys.Print(f'DoFs/rank: {V.dim()/mesh.comm.size}')
PETSc.Sys.Print('')

# Number of nonlinear iterations will be 1 per timestep for linear problems
PETSc.Sys.Print(f'Nonlinear iterations: {str(nonlinear_its).rjust(5)}  |  Iterations per window: {str(nonlinear_its/args.nt).rjust(5)}')

# Number of linear iterations of the all-at-once system, total and per window.
PETSc.Sys.Print(f'Linear iterations:    {str(linear_its).rjust(5)}  |  Iterations per window: {str(linear_its/args.nt).rjust(5)}')
PETSc.Sys.Print('')

# Timing measurements
PETSc.Sys.Print(timer.string(timesteps_per_solve=1,
                             total_iterations=linear_its, ndigits=5))
PETSc.Sys.Print('')
