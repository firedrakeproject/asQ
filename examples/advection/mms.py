
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import pi, cos, sin

import firedrake as fd
from firedrake.petsc import PETSc
import numpy as np
import asQ

import argparse

parser = argparse.ArgumentParser(
    description='ParaDiag timestepping for the Heat equation with time coefficient.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--nx', type=int, default=16, help='Number of cells along each square side.')
parser.add_argument('--cfl', type=float, default=1.0, help='Convective CFL number.')
parser.add_argument('--angle', type=float, default=pi/6, help='Angle of the convective velocity.')
parser.add_argument('--degree', type=int, default=1, help='Degree of the scalar and velocity spaces.')
parser.add_argument('--theta', type=float, default=0.5, help='Parameter for the implicit theta timestepping method.')
parser.add_argument('--width', type=float, default=0.1, help='Width of the Gaussian bump.')
parser.add_argument('--nwindows', type=int, default=1, help='Number of time-windows.')
parser.add_argument('--nslices', type=int, default=2, help='Number of time-slices per time-window.')
parser.add_argument('--slice_length', type=int, default=2, help='Number of timesteps per time-slice.')
parser.add_argument('--alpha', type=float, default=0.0001, help='Circulant coefficient.')
parser.add_argument('--nsample', type=int, default=32, help='Number of sample points for plotting.')
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

# Calculate the timestep from the CFL number
dx = 1./args.nx
dt = args.cfl*dx
# The Ensemble with the spatial and time communicators
ensemble = asQ.create_ensemble(time_partition)

# # # === --- domain --- === # # #

# The mesh needs to be created with the spatial communicator
mesh = fd.UnitSquareMesh(args.nx, args.nx, quadrilateral=False, comm=ensemble.comm)

V = fd.FunctionSpace(mesh, "CG", args.degree)

# # # === --- initial conditions --- === # # #

x, y = fd.SpatialCoordinate(mesh)

# Solving u_t - (1+2t) \Delta u = f

u_exact = fd.sin(pi*x)*fd.cos(pi*y)

#f = (u_exact)_t - (1+2*t)\Delta u_exact


# The scalar initial conditions are a Gaussian bump centred at (0.5, 0.5)
w0 = fd.Function(V, name="scalar_initial")
w0.interpolate(fd.sin(pi*x)*fd.cos(pi*y))


# # # === --- finite element forms --- === # # #


# asQ assumes that the mass form is linear so here
# q is a TrialFunction and phi is a TestFunction
def form_mass(q, phi):
    return phi*q*fd.dx

DBc = [fd.DirichletBC(V, u_exact, 'on_boundary')]


# q is a Function and phi is a TestFunction
def form_function(q, phi, t):

    return (1+2*t)*fd.inner(fd.grad(q), fd.grad(phi))*fd.dx - fd.inner(2*pi**2*(1+2*t)*u_exact, phi)*fd.dx


# # # === --- PETSc solver parameters --- === # # #


# The PETSc solver parameters used to solve the
# blocks in step (b) of inverting the ParaDiag matrix.
block_parameters = {
    'ksp_type': 'preonly',
    'pc_type': 'lu',
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
    'ksp_type': 'preonly',
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
    paradiag_parameters['diagfft_'+str(i)+'_'] = block_parameters


# # # === --- Setup ParaDiag --- === # # #


# Give everything to asQ to create the paradiag object.
# the circ parameter determines where the alpha-circulant
# approximation is introduced. None means only in the preconditioner.
PD = asQ.paradiag(ensemble=ensemble,
                   form_function=form_function,
                   form_mass=form_mass,
                   w0=w0, dt=dt, theta=args.theta,
                   alpha=args.alpha, time_partition=time_partition, bcs = DBc,
                   solver_parameters=paradiag_parameters,
                   circ=None)


# This is a callback which will be called before pdg solves each time-window
# We can use this to make the output a bit easier to read
def window_preproc(pdg, wndw):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'### === --- Calculating time-window {wndw} --- === ###')
    PETSc.Sys.Print('')

qcheck = w0.copy(deepcopy=True)
#timeseries = [q0.copy(deepcopy=True)]

def Exact(u_exact):
    q_err = fd.errornorm(u_exact, qcheck)
    return q_err

# The last time-slice will be saving snapshots to create an animation.
# The layout member describes the time_partition.
# layout.is_local(i) returns True/False if the timestep index i is on the
# current time-slice. Here we use -1 to mean the last timestep in the window.
#is_last_slice = pdg.layout.is_local(-1)

# Make an output Function on the last time-slice and start a snapshot list
#if is_last_slice:
#    qout = fd.Function(V)
#    timeseries = [q0.copy(deepcopy=True)]


# This is a callback which will be called after pdg solves each time-window
# We can use this to save the last timestep of each window for plotting.
def window_postproc(pdg, wndw):
    errors = np.zeros((window_length, 1))
    local_errors = np.zeros_like(errors)
    # collect errors for this slice
    def for_each_callback(window_index, slice_index, w):
        nonlocal local_errors
        local_errors[window_index] = Exact(w)
    pdg.aaos.for_each_timestep(for_each_callback)

    def for_each_callback(window_index, slice_index, w):
        nonlocal local_errors
        local_errors[window_index, :] = Exact(w)

    pdg.aaos.for_each_timestep(for_each_callback)

    # collect and print errors for full window
    ensemble.ensemble_comm.Reduce(local_errors, errors, root=0)
    if pdg.time_rank == 0:
        for window_index in range(window_length):
            timestep = wndw*window_length + window_index
            q_err = errors[window_index,0]
            PETSc.Sys.Print(f"timestep={timestep}, q_err={q_err}",
                            comm=ensemble.comm)








# Solve nwindows of the all-at-once system
PD.solve(args.nwindows,
          preproc=window_preproc,
          postproc=window_postproc)


