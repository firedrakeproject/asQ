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
parser.add_argument('--nx', type=int, default=64, help='Number of cells along each square side.')
parser.add_argument('--cfl', type=float, default=0.8, help='Convective CFL number.')
parser.add_argument('--angle', type=float, default=pi/6, help='Angle of the convective velocity.')
parser.add_argument('--degree', type=int, default=1, help='Degree of the scalar and velocity spaces.')
parser.add_argument('--theta', type=float, default=0.5, help='Parameter for the implicit theta timestepping method.')
parser.add_argument('--width', type=float, default=0.1, help='Width of the Gaussian bump.')
parser.add_argument('--nwindows', type=int, default=1, help='Number of time-windows.')
parser.add_argument('--nslices', type=int, default=2, help='Number of time-slices per time-window.')
parser.add_argument('--slice_length', type=int, default=2, help='Number of timesteps per time-slice.')
parser.add_argument('--alpha', type=float, default=0.0001, help='Circulant coefficient.')
parser.add_argument('--nsample', type=int, default=32, help='Number of sample points for plotting.')
parser.add_argument('--mpeg', action='store_true', help='Output an mpeg of the timeseries.')
parser.add_argument('--write_metrics', action='store_true', help='Write various solver metrics to file.')
parser.add_argument('--show_args', action='store_true', default=True, help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

# The time partition describes how many timesteps are included on each time-slice of the ensemble
# Here we use the same number of timesteps on each slice, but they can be different

time_partition = tuple(args.slice_length for _ in range(args.nslices))
window_length = sum(time_partition)
nsteps = args.nwindows*window_length

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
W = fd.VectorFunctionSpace(mesh, "CG", args.degree)

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
    un = 0.5*(fd.dot(u, n) + abs(fd.dot(u, n)))

    # integration over element volume
    int_cell = q*fd.div(phi*u)*fd.dx

    # integration over internal facets
    int_facet = (phi('+')-phi('-'))*(un('+')*q('+')-un('-')*q('-'))*fd.dS

    return int_facet - int_cell


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
# The equation is linear so we can either:
# a) Solve it in one shot using a preconditioned Krylov method:
#    P^{-1}Au = P^{-1}b
#    The solver options for this are:
#    'ksp_type': 'fgmres'
#    We need fgmres here because gmres is used on the blocks.
# b) Solve it with Picard iterations:
#    Pu_{k+1} = (P - A)u_{k} + b
#    The solver options for this are:
#    'ksp_type': 'preonly'

atol = 1e-10
rtol = 1e-8
paradiag_parameters = {
    'snes_type': 'ksponly',
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
        'converged_reason': None,
        'rtol': rtol,
        'atol': atol,
        'stol': 1e-12,
    },
    'pc_type': 'python',
    'pc_python_type': 'asQ.DiagFFTPC'
}

# We need to add a block solver parameters dictionary for each block.
# Here they are all the same but they could be different.
for i in range(window_length):
    paradiag_parameters['diagfft_block_'+str(i)+'_'] = block_parameters


# # # === --- Setup ParaDiag --- === # # #


# Give everything to asQ to create the paradiag object.
# the circ parameter determines where the alpha-circulant
# approximation is introduced. None means only in the preconditioner.
pdg = asQ.paradiag(ensemble=ensemble,
                   form_function=form_function,
                   form_mass=form_mass,
                   w0=q0, dt=dt, theta=args.theta,
                   alpha=args.alpha, time_partition=time_partition,
                   solver_parameters=paradiag_parameters,
                   circ=None)


# This is a callback which will be called before pdg solves each time-window
# We can use this to make the output a bit easier to read
def window_preproc(pdg, wndw):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'### === --- Calculating time-window {wndw} --- === ###')
    PETSc.Sys.Print('')


# The last time-slice will be saving snapshots to create an animation.
# The layout member describes the time_partition.
# layout.is_local(i) returns True/False if the timestep index i is on the
# current time-slice. Here we use -1 to mean the last timestep in the window.
is_last_slice = pdg.layout.is_local(-1)

# Make an output Function on the last time-slice and start a snapshot list
if is_last_slice:
    qout = fd.Function(V)
    timeseries = [q0.copy(deepcopy=True)]


# This is a callback which will be called after pdg solves each time-window
# We can use this to save the last timestep of each window for plotting.
def window_postproc(pdg, wndw):
    if is_last_slice:
        # The aaos is the AllAtOnceSystem which represents the time-dependent problem.
        # get_field extracts one timestep of the window. -1 is again used to get the last
        # timestep and place it in qout.
        pdg.aaos.get_field(-1, index_range='window', wout=qout)
        timeseries.append(qout.copy(deepcopy=True))


# Solve nwindows of the all-at-once system
pdg.solve(args.nwindows,
          preproc=window_preproc,
          postproc=window_postproc)


# # # === --- Postprocessing --- === # # #

# paradiag collects a few solver diagnostics for us to inspect
nw = args.nwindows

# Number of nonlinear iterations, total and per window.
# (1 for fgmres and # picard iterations for preonly)
PETSc.Sys.Print(f'nonlinear iterations: {pdg.nonlinear_iterations}  |  iterations per window: {pdg.nonlinear_iterations/nw}')

# Number of linear iterations, total and per window.
# (# of gmres iterations for fgmres and # picard iterations for preonly)
PETSc.Sys.Print(f'linear iterations: {pdg.linear_iterations}  |  iterations per window: {pdg.linear_iterations/nw}')

# Number of iterations needed for each block in step-(b), total and per block solve
# The number of iterations for each block will usually be different because of the different eigenvalues
PETSc.Sys.Print(f'block linear iterations: {pdg.block_iterations._data}  |  iterations per block solve: {pdg.block_iterations._data/pdg.linear_iterations}')

# We can write these diagnostics to file, along with some other useful information.
# Files written are: aaos_metrics.txt, block_metrics.txt, paradiag_setup.txt, solver_parameters.txt
if args.write_metrics:
    asQ.write_paradiag_metrics(pdg)

# Make an animation from the snapshots we collected and save it to periodic.mp4.
if is_last_slice and args.mpeg:

    fn_plotter = fd.FunctionPlotter(mesh, num_sample_points=args.nsample)

    fig, axes = plt.subplots()
    axes.set_aspect('equal')
    colors = fd.tripcolor(qout, num_sample_points=args.nsample, vmin=1, vmax=2, axes=axes)
    fig.colorbar(colors)

    def animate(q):
        colors.set_array(fn_plotter(q))

    interval = 1e2
    animation = FuncAnimation(fig, animate, frames=timeseries, interval=interval)

    animation.save("periodic.mp4", writer="ffmpeg")
