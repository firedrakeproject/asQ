from math import pi
import firedrake as fd
from firedrake.petsc import PETSc
import asQ
print = lambda x: PETSc.Sys.Print(x)

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


def time_coef(k, t):
    return fd.exp(k*t)


def Rhs(k, t):
    # As our exact solution is independent of t, the Rhs is just $-(2 + \sin(2*pi*t)) \Delta u$
    return -(time_coef(k, t))*fd.div(fd.grad(u_exact)) - u_exact**2


# Initial conditions.
w0 = fd.Function(V, name="scalar_initial")
w0.interpolate(u_exact)
# Dirichlet BCs
bcs = [fd.DirichletBC(V, u_exact, 'on_boundary')]

aaofunc = asQ.AllAtOnceFunction(ensemble, time_partition, V)
aaofunc.assign(w0)


# # # === --- finite element forms --- === # # #


# asQ assumes that the mass form is linear so here
# q is a TrialFunction and phi is a TestFunction
def form_mass(q, phi):
    return phi*q*fd.dx


k = 3


# q is a Function and phi is a TestFunction
def form_function(q, phi, t):
    return time_coef(k, t)*fd.inner(fd.grad(q), fd.grad(phi))*fd.dx-fd.inner(Rhs(k, t), phi)*fd.dx - fd.inner(fd.cos(q**2),phi)*fd.dx




aaoform = asQ.AllAtOnceForm(aaofunc, dt, args.theta,
                            form_mass, form_function, bcs=bcs)


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

solver_parameters = {
#    'snes_type': 'newtonls',
    'snes': {
        'monitor': None,
        'converged_reason': None,
        'rtol': 1e-9,
        'atol': 1e-9,
        'stol': 1e-9,    },
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp': {
        'monitor': None,
        'converged_reason': None,
        'rtol': 1e-9,
        'atol': 1e-9,
        'stol': 1e-9,    },
    'pc_type': 'python',
    'pc_python_type': 'asQ.DiagFFTPC',
    'diagfft_alpha': args.alpha,
#    'diagfft_state': 'linear',
#    'aaos_jacobian_state': 'linear',
    'diagfft_state': 'window',
    'diagfft_linearisation': 'consistent',
    'aaos_jacobian_state': 'current',
    'aaos_jacobian_linearisation': 'consistent',

}

# We need to add a block solver parameters dictionary for each block.
# Here they are all the same but they could be different.
for i in range(window_length):
    solver_parameters['diagfft_block_'+str(i)+'_'] = block_parameters

aaosolver = asQ.AllAtOnceSolver(aaoform, aaofunc,
                                solver_parameters=solver_parameters)

# set up diagnostic recording

linear_iterations = 0
nonlinear_iterations = 0
total_timesteps = 0
total_windows = 0

w = fd.Function(V)


# record some diagnostics from the solve
def record_diagnostics():
    global linear_iterations, nonlinear_iterations, total_timesteps, total_windows
    linear_iterations += aaosolver.snes.getLinearSolveIterations()
    nonlinear_iterations += aaosolver.snes.getIterationNumber()
    total_timesteps += sum(aaosolver.time_partition)
    total_windows += 1


for i in range(args.nwindows):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'### === --- Calculating time-window {i} --- === ###')
    PETSc.Sys.Print('')

    aaosolver.solve()
    record_diagnostics()

    # restart timeseries using final timestep as new
    # initial conditions and the initial guess.
    aaofunc.assign(aaofunc.bcast_field(-1, aaofunc.initial_condition))
    for n in range(args.slice_length):
        aaoform.time[n].assign(aaoform.time[n] + dt*aaofunc.ntimesteps)
    aaofunc.t0.assign(aaofunc.t0 + dt*aaofunc.ntimesteps)
# Number of nonlinear iterations, total and per window.
PETSc.Sys.Print(f'nonlinear iterations: {nonlinear_iterations}  |  iterations per window: {nonlinear_iterations/total_windows}')

# Number of linear iterations, total and per window.
PETSc.Sys.Print(f'linear iterations: {linear_iterations}  |  iterations per window: {linear_iterations/total_windows}')

# Number of iterations needed for each block in step-(b), total and per block solve
# The number of iterations for each block will usually be different because of the different eigenvalues
block_iterations = aaosolver.jacobian.pc.block_iterations
block_iterations.synchronise()
PETSc.Sys.Print(f'block linear iterations: {block_iterations.data()}  |  iterations per block solve: {block_iterations.data()/linear_iterations}')

# Make an animation from the snapshots we collected and save it to periodic.mp4.
