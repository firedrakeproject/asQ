import matplotlib.pyplot as plt
from math import pi
from matplotlib.animation import FuncAnimation
import firedrake as fd
from firedrake.petsc import PETSc
import asQ

import argparse

parser = argparse.ArgumentParser(
    description='Paradiag for Stratigraphic model that simulate formation of sedimentary rock over geological time.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--nx', type=int, default=200, help='Number of cells along each square side.')
parser.add_argument('--degree', type=int, default=1, help='Degree of the scalar and velocity spaces.')
parser.add_argument('--theta', type=float, default=0.5, help='Parameter for the implicit theta timestepping method.')
parser.add_argument('--nwindows', type=int, default=1, help='Number of time-windows.')
parser.add_argument('--nslices', type=int, default=2, help='Number of time-slices per time-window.')
parser.add_argument('--slice_length', type=int, default=2, help='Number of timesteps per time-slice.')
parser.add_argument('--alpha', type=float, default=0.0001, help='Circulant coefficient.')
parser.add_argument('--nsample', type=int, default=16, help='Number of sample points for plotting.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)
# The time partition describes how many timesteps are included on each time-slice of the ensemble
# Here we use the same number of timesteps on each slice, but they can be different

time_partition = tuple(args.slice_length for _ in range(args.nslices))
window_length = sum(time_partition)
nsteps = args.nwindows*window_length

dt = 1000

# The Ensemble with the spatial and time communicators
ensemble = asQ.create_ensemble(time_partition)

# # # === --- domain --- === # # #

# The mesh needs to be created with the spatial communicator
mesh = fd.SquareMesh(args.nx, args.nx, 100, quadrilateral=False, comm=ensemble.comm)

V = fd.FunctionSpace(mesh, "CG", args.degree)

# # # === --- initial conditions --- === # # #

x, y = fd.SpatialCoordinate(mesh)

s0 = fd.Function(V, name="scalar_initial")
s0.interpolate(fd.Constant(0.0))


# The sediment movement D
def D(D_c, d):
    return D_c*2/fd.Constant(fd.sqrt(2*pi))*fd.exp(-1/2*((d-5)/10)**2)


# The carbonate growth L.
def L(G_0, d):
    return G_0*fd.conditional(d > 0, fd.exp(-d/10)/(1 + fd.exp(-50*d)), fd.exp((50-1/10)*d)/(fd.exp(50*d) + 1))


# # # === --- finite element forms --- === # # #


def form_mass(s, q):
    return s*q*fd.dx


D_c = fd.Constant(.002)
G_0 = fd.Constant(.004)
A = fd.Constant(50)
b = 100*fd.tanh(1/20*(x-50))


def form_function(s, q, t):
    return D(D_c, A*fd.sin(2*pi*t/500000)-b-s)*fd.inner(fd.grad(s), fd.grad(q))*fd.dx-L(G_0, A*fd.sin(2*pi*t/500000)-b-s)*q*fd.dx


# # # === --- PETSc solver parameters --- === # # #


# The PETSc solver parameters used to solve the
# blocks in step (b) of inverting the ParaDiag matrix.
block_parameters = {
    "ksp_type": "preonly",
    "pc_type": "lu",
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

paradiag_parameters = {
    'snes': {
        'linesearch_type': 'basic',
        'monitor': None,
        'converged_reason': None,
        'rtol': 1e-10,
        'atol': 1e-12,
        'stol': 1e-12,
    },
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp': {
        'monitor': None,
        'converged_reason': None,
        'rtol': 1e-10,
        'atol': 1e-12,
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
                   w0=s0, dt=dt, theta=args.theta,
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
    timeseries = [s0.copy(deepcopy=True)]


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
asQ.write_paradiag_metrics(pdg)

# Make an animation from the snapshots we collected and save it to periodic.mp4.
if is_last_slice:

    fn_plotter = fd.FunctionPlotter(mesh, num_sample_points=args.nsample)

    fig, axes = plt.subplots()
    axes.set_aspect('equal')
    colors = fd.tripcolor(qout, num_sample_points=args.nsample, axes=axes)
    fig.colorbar(colors)

    def animate(q):
        colors.set_array(fn_plotter(q))

    animation = FuncAnimation(fig, animate, frames=timeseries)

    animation.save("periodic.mp4", writer="ffmpeg")
