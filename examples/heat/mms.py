from math import pi
import firedrake as fd
from firedrake.petsc import PETSc
import asQ

import argparse

parser = argparse.ArgumentParser(
    description='ParaDiag timestepping for a linear equation with time-dpendent coefficient. Here, we use the MMS to solve u_t - (1+2sin(pi x) sin(pi y)) Delta u = f over the domain, Omega = [0,1]^2 with Dirichlet BCs $',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--nx', type=int, default=64, help='Number of cells along each square side.')
parser.add_argument('--degree', type=int, default=1, help='Degree of the scalar and velocity spaces.')
parser.add_argument('--theta', type=float, default=0.5, help='Parameter for the implicit theta timestepping method.')
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

# Set the timesstep
dx = 1./args.nx
dt = dx

# The Ensemble with the spatial and time communicators
ensemble = asQ.create_ensemble(time_partition)

# # # === --- domain --- === # # #

# The mesh needs to be created with the spatial communicator
mesh = fd.UnitSquareMesh(args.nx, args.nx, quadrilateral=False, comm=ensemble.comm)

V = fd.FunctionSpace(mesh, "CG", args.degree)

# # # === --- initial conditions --- === # # #

x, y = fd.SpatialCoordinate(mesh)

# We use the method of manufactured solutions, prescribing a Rhs, initial and boundary data to exactly match those of a known solution.
u_exact = fd.sin(pi*x)*fd.cos(pi*y)


def time_coef(t):
    return 2 + fd.sin(2*pi*t)


def Rhs(t):
    # As our exact solution is independent of t, the Rhs is just $-(2 + \sin(2*pi*t)) \Delta u$
    return -time_coef(t)*fd.div(fd.grad(u_exact))


# Initial conditions.
w0 = fd.Function(V, name="scalar_initial")
w0.interpolate(u_exact)
# Dirichlet BCs
bcs = [fd.DirichletBC(V, u_exact, 'on_boundary')]


# # # === --- finite element forms --- === # # #


# asQ assumes that the mass form is linear so here
# q is a TrialFunction and phi is a TestFunction
def form_mass(q, phi):
    return phi*q*fd.dx


# q is a Function and phi is a TestFunction
def form_function(q, phi, t):
    return time_coef(t)*fd.inner(fd.grad(q), fd.grad(phi))*fd.dx - fd.inner(Rhs(t), phi)*fd.dx


# # # === --- PETSc solver parameters --- === # # #


# The PETSc solver parameters used to solve the
# blocks in step (b) of inverting the ParaDiag matrix.
block_parameters = {
    'ksp_type': 'gmres',
    'pc_type': 'bjacobi',
}

# The PETSc solver parameters for solving the all-at-once system.
# The python preconditioner 'asQ.CirculantPC' applies the ParaDiag matrix.
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
    'snes_type': 'ksponly',
    'snes': {
        'monitor': None,
        'converged_reason': None,
        'rtol': 1e-10,
        'atol': 1e-8,
        'stol': 1e-12,
    },
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp': {
        'monitor': None,
        'converged_reason': None,
        'rtol': 1e-10,
        'atol': 1e-8,
        'stol': 1e-12,
    },
    'pc_type': 'python',
    'pc_python_type': 'asQ.CirculantPC',
    'diagfft_alpha': args.alpha,
}

# We need to add a block solver parameters dictionary for each block.
# Here they are all the same but they could be different.
for i in range(window_length):
    paradiag_parameters['diagfft_block_'+str(i)+'_'] = block_parameters


# # # === --- Setup ParaDiag --- === # # #


# Give everything to asQ to create the paradiag object.
# the circ parameter determines where the alpha-circulant
# approximation is introduced. None means only in the preconditioner.
pdg = asQ.Paradiag(ensemble=ensemble,
                   form_function=form_function,
                   form_mass=form_mass,
                   ics=w0, dt=dt, theta=args.theta,
                   time_partition=time_partition, bcs=bcs,
                   solver_parameters=paradiag_parameters)


# This is a callback which will be called before pdg solves each time-window
# We can use this to make the output a bit easier to read
def window_preproc(pdg, wndw, rhs):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'### === --- Calculating time-window {wndw} --- === ###')
    PETSc.Sys.Print('')


qcheck = w0.copy(deepcopy=True)


def exact(w):
    q_err = fd.errornorm(w, qcheck)
    return q_err


def window_postproc(pdg, wndw, rhs):
    uerrors = asQ.SharedArray(time_partition, comm=ensemble.ensemble_comm)
    for i in range(pdg.aaofunc.nlocal_timesteps):
        uerr = exact(pdg.aaofunc[i])
        uerrors.dlocal[i] = uerr
    uerrors.synchronise()

    for i in range(pdg.ntimesteps):
        timestep = wndw*window_length + i
        uerr = uerrors.dglobal[i]
        PETSc.Sys.Print(f"timestep={timestep}, uerr={uerr}")


# Solve nwindows of the all-at-once system
pdg.solve(args.nwindows,
          preproc=window_preproc,
          postproc=window_postproc)
