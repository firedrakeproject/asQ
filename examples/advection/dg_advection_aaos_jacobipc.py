
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import pi, cos, sin

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
parser.add_argument('--nwindows', type=int, default=1, help='Number of time-windows.')
parser.add_argument('--nslices', type=int, default=2, help='Number of time-slices per time-window.')
parser.add_argument('--slice_length', type=int, default=2, help='Number of timesteps per time-slice.')
parser.add_argument('--alpha', type=float, default=0.0001, help='Circulant coefficient.')
parser.add_argument('--nsample', type=int, default=32, help='Number of sample points for plotting.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')
parser.add_argument('--mpeg', action='store_true', help='Create mp4 of timeseries')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

# The time partition describes how many timesteps are included on each time-slice of the ensemble
# Here we use the same number of timesteps on each slice, but they can be different

time_partition = tuple(args.slice_length for _ in range(args.nslices))

# Calculate the timestep from the CFL number
umax = 1.
dx = 1./args.nx
dt = args.cfl*dx/umax

# The Ensemble with the spatial and time communicators
ensemble = asQ.create_ensemble(time_partition)

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
#
# We have a selection of all-at-once preconditioners:
#   'asQ.CirculantPC'
#       - applies the block circulant ParaDiag matrix.
#   'asQ.JacobiPC'
#       - applies a block Jacobi preconditioner where each block
#         is a single timestep.
#   'asQ.SliceJacobiPC'
#       - applies a block Jacobi preconditioner where each block
#         is a set of 'slice_jacobi_nsteps' timesteps. Each block
#         can in turn be preconditioned with either CirculantPC or
#         some other method.
#
# The equation is linear so we can use 'snes_type': 'ksponly' and
# use your favourite Krylov method (if a Krylov method is used on
# the blocks then the outer Krylov method must be either flexible
# or Richardson).

circulant_parameters = {
    'pc_type': 'python',
    'pc_python_type': 'asQ.CirculantPC',
    'diagfft_state': 'linear',
    'diagfft_alpha': 1e-4,
    'diagfft_block': block_parameters,
}

jacobi_parameters = {
    'pc_type': 'python',
    'pc_python_type': 'asQ.JacobiPC',
    'aaojacobi_state': 'linear',
    'aaojacobi_block': block_parameters,
}

slice_jacobi_parameters = {
    'pc_type': 'python',
    'pc_python_type': 'asQ.SliceJacobiPC',
    'slice_jacobi_nsteps': 4,
    'slice_jacobi_slice': {
        'aaos_jacobian_state': 'linear',
        'ksp_type': 'preonly',
    }
}

atol = 1e-100
rtol = 1e-8
solver_parameters = {
    'snes_type': 'ksponly',  # for a linear system
    'snes': {
        'monitor': None,
        'converged_reason': None,
        'rtol': rtol,
        'atol': atol,
        'stol': 1e-12,
    },
    'mat_type': 'matfree',
    'ksp_type': 'gmres',
    'ksp': {
        'monitor': None,
        'converged_rate': None,
        'rtol': rtol,
        'atol': atol,
        'stol': 1e-12,
    },
    'aaos_jacobian_state': 'linear',
}

# pick which preconditioning method we want here
# solver_parameters.update(jacobi_parameters)
# solver_parameters.update(circulant_parameters)
solver_parameters.update(slice_jacobi_parameters)

slice_jacobi_parameters['slice_jacobi_slice'].update(circulant_parameters)


# Create a solver object to set up and solve the (possibly nonlinear) problem
# for the timeseries in the all-at-once function.
aaosolver = asQ.AllAtOnceSolver(aaoform, aaofunc,
                                solver_parameters=solver_parameters)

# set up diagnostic recording

linear_iterations = 0
nonlinear_iterations = 0
total_timesteps = 0
total_windows = 0

w = fd.Function(V)

# The last time-slice will be saving snapshots to create an animation.
# The layout member describes the time_partition.
# layout.is_local(i) returns True/False if the timestep index i is on the
# current time-slice. Here we use -1 to mean the last timestep in the window.
is_last_slice = aaofunc.layout.is_local(-1)

if is_last_slice:
    timeseries = [q0.copy(deepcopy=True)]


# record some diagnostics from the solve
def record_diagnostics():
    global linear_iterations, nonlinear_iterations, total_timesteps, total_windows
    linear_iterations += aaosolver.snes.getLinearSolveIterations()
    nonlinear_iterations += aaosolver.snes.getIterationNumber()
    total_timesteps += sum(aaosolver.time_partition)
    total_windows += 1
    if is_last_slice:
        w.assign(aaofunc[-1])
        timeseries.append(w.copy(deepcopy=True))


for i in range(args.nwindows):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'### === --- Calculating time-window {i} --- === ###')
    PETSc.Sys.Print('')

    aaosolver.solve()
    record_diagnostics()

    # restart timeseries using final timestep as new
    # initial conditions and the initial guess.
    aaofunc.assign(aaofunc.bcast_field(-1, aaofunc.initial_condition))

# Print out some iteration counts

# Number of nonlinear iterations, total and per window.
# (Will be 1 per window for this linear problem)
PETSc.Sys.Print(f'nonlinear iterations: {nonlinear_iterations}  |  iterations per window: {nonlinear_iterations/total_windows}')

# Number of linear iterations, total and per window.
PETSc.Sys.Print(f'linear iterations: {linear_iterations}  |  iterations per window: {linear_iterations/total_windows}')

# Number of iterations needed for each block in step-(b), total and per block solve
# The number of iterations for each block will usually be different because of the different eigenvalues
if hasattr(aaosolver.jacobian.pc, "block_iterations"):
    block_iterations = aaosolver.jacobian.pc.block_iterations
    block_iterations.synchronise()
    PETSc.Sys.Print(f'block linear iterations: {block_iterations.data()}  |  iterations per block solve: {block_iterations.data()/linear_iterations}')

# Make an animation from the snapshots we collected and save it to periodic.mp4.
if is_last_slice and args.mpeg:
    fn_plotter = fd.FunctionPlotter(mesh, num_sample_points=args.nsample)

    fig, axes = plt.subplots()
    axes.set_aspect('equal')
    colors = fd.tripcolor(w, num_sample_points=args.nsample, vmin=1, vmax=2, axes=axes)
    fig.colorbar(colors)

    def animate(q):
        colors.set_array(fn_plotter(q))

    interval = 1e2
    animation = FuncAnimation(fig, animate, frames=timeseries, interval=interval)

    animation.save("periodic.mp4", writer="ffmpeg")
