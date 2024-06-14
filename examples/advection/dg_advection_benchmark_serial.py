from math import pi, cos, sin

from utils.timing import SolverTimer
import firedrake as fd
from firedrake.petsc import PETSc
from utils.serial import SerialMiniApp

import argparse

parser = argparse.ArgumentParser(
    description='ParaDiag timestepping for scalar advection of a Gaussian bump in a periodic square with DG in space and implicit-theta in time. Based on the Firedrake DG advection example https://www.firedrakeproject.org/demos/DG_advection.py.html',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--nx', type=int, default=16, help='Number of cells along each square side.')
parser.add_argument('--cfl', type=float, default=0.8, help='Convective CFL number.')
parser.add_argument('--angle', type=float, default=pi/6, help='Angle of the convective velocity.')
parser.add_argument('--degree', type=int, default=1, help='Degree of the scalar and velocity spaces.')
parser.add_argument('--theta', type=float, default=0.5, help='Parameter for the implicit theta timestepping method.')
parser.add_argument('--width', type=float, default=0.2, help='Width of the Gaussian bump.')
parser.add_argument('--nt', type=int, default=8, help='Number of timesteps.')
parser.add_argument('--lu_solver', type=str, default='petsc', help='Direct solver for the blocks')
parser.add_argument('--show_args', action='store_true', default=True, help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

# Calculate the timestep from the CFL number
umax = 1.
dx = 1./args.nx
dt = args.cfl*dx/umax

# # # === --- domain --- === # # #

# The mesh needs to be created with the spatial communicator
mesh = fd.PeriodicUnitSquareMesh(args.nx, args.nx, quadrilateral=True)

# We use a discontinuous Galerkin space for the advected scalar
# and a continuous Galerkin space for the advecting velocity field
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

atol = 1e-10
rtol = 1e-10
block_parameters = {
    'snes_type': 'ksponly',  # for a linear system
    'snes': {
        'rtol': rtol,
        'atol': atol,
        'stol': 1e-12,
        'lag_jacobian': -2,
        'lag_jacobian_persists': None,
        'lag_preconditioner': -2,
        'lag_preconditioner_persists': None,
    },
    'ksp': {
        'monitor': None,
        'converged_rate': None,
        'rtol': rtol,
        'atol': atol,
        'stol': 1e-12,
    },
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': args.lu_solver,
}


# # # === --- Setup ParaDiag --- === # # #


# Give everything to the miniapp to create a serial timestepper
miniapp = SerialMiniApp(dt, args.theta, q0,
                        form_mass, form_function,
                        block_parameters)

# time the calculations
timer = SolverTimer()


PETSc.Sys.Print('### === --- Timestepping loop --- === ###')
linear_its = 0
nonlinear_its = 0

# This is a callback which will be called before solving each timestep
# We can use this to make the output a bit easier to read and time the calculation
def preproc(app, step, t):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'### === --- Calculating timestep {step} --- === ###')
    PETSc.Sys.Print('')
    timer.start_timing()

# This is a callback which will be called after solving each timestep
# We can use this to time the calculation
def postproc(app, step, t):
    timer.stop_timing()
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'Timestep solution time: {round(timer.times[-1], 5)}')
    PETSc.Sys.Print('')

    global linear_its
    global nonlinear_its
    linear_its += app.nlsolver.snes.getLinearSolveIterations()
    nonlinear_its += app.nlsolver.snes.getIterationNumber()


# Solve nwindows of the all-at-once system
miniapp.solve(args.nt,
              preproc=preproc,
              postproc=postproc)

# # # === --- Postprocessing --- === # # #

# parallelism
PETSc.Sys.Print(f'DoFs per timestep: {V.dim()}')
PETSc.Sys.Print(f'Number of MPI ranks per timestep: {mesh.comm.size}')
PETSc.Sys.Print(f'DoFs/rank: {V.dim()/mesh.comm.size}')
PETSc.Sys.Print('')

# iteration counts
PETSc.Sys.Print(f'linear iterations: {linear_its} | iterations per timestep: {linear_its/args.nt}')
PETSc.Sys.Print(f'nonlinear iterations: {nonlinear_its} | iterations per timestep: {nonlinear_its/args.nt}')
PETSc.Sys.Print('')

# timing
if timer.ntimes() > 1:
    timer.times[0] = timer.times[1]

PETSc.Sys.Print(timer.string(timesteps_per_solve=1,
                             total_iterations=linear_its, ndigits=5))
PETSc.Sys.Print('')
