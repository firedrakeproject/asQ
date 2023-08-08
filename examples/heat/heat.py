import firedrake as fd
from firedrake.petsc import PETSc
import asQ

import argparse

parser = argparse.ArgumentParser(
    description='While we try to figure out how to implement time-dependent Dirichlet BCs, one can use Nitsche-type penalty method. Here, we consider Heat equatiion(u_t = Delta u) with boundary conditions match those of exp(1.25t + 0.5x + y)',
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
nx = args.nx
dt = 1./nx

# The Ensemble with the spatial and time communicators
ensemble = asQ.create_ensemble(time_partition)

# # # === --- domain --- === # # #

# The mesh needs to be created with the spatial communicator
mesh = fd.UnitSquareMesh(args.nx, args.nx, quadrilateral=False, comm=ensemble.comm)

V = fd.FunctionSpace(mesh, "CG", args.degree)

# # # === --- initial conditions --- === # # #
# q_exact = exp(0.5x + y + 1.25t), u_t-\Deltau = 0.

x, y = fd.SpatialCoordinate(mesh)
n = fd.FacetNormal(mesh)
# Initial conditions.
w0 = fd.Function(V, name="scalar_initial")
w0.interpolate(fd.exp(0.5*x + y))

# # # === --- finite element forms --- === # # #


# asQ assumes that the mass form is linear so here
# q is a TrialFunction and phi is a TestFunction
def form_mass(q, phi):
    return phi*q*fd.dx


# q is a Function and phi is a TestFunction
def form_function(q, phi, t):
    return fd.inner(fd.grad(q), fd.grad(phi))*fd.dx - fd.inner(phi, fd.inner(fd.grad(q), n))*fd.ds - fd.inner(q-fd.exp(0.5*x + y + 1.25*t), fd.inner(fd.grad(phi), n))*fd.ds + 20*nx*fd.inner(q-fd.exp(0.5*x + y + 1.25*t), phi)*fd.ds


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
    'snes_type': 'ksponly',
    'snes': {
        'monitor': None,
        'converged_reason': None,
        'atol': 1e-8,
    },
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp': {
        'monitor': None,
        'converged_reason': None,
        'atol': 1e-8,
    },
    'pc_type': 'python',
    'pc_python_type': 'asQ.DiagFFTPC',
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
                   time_partition=time_partition,
                   solver_parameters=paradiag_parameters)


# This is a callback which will be called before pdg solves each time-window
# We can use this to make the output a bit easier to read
def window_preproc(pdg, wndw, rhs):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'### === --- Calculating time-window {wndw} --- === ###')
    PETSc.Sys.Print('')


# We find the L2-error at each timestep
q_exact = fd.Function(V)
qp = fd.Function(V)
errors = asQ.SharedArray(time_partition, comm=ensemble.ensemble_comm)
times = asQ.SharedArray(time_partition, comm=ensemble.ensemble_comm)


def window_postproc(pdg, wndw, rhs):
    for step in range(pdg.aaofunc.ntimesteps):
        if pdg.aaoform.layout.is_local(step):
            local_step = pdg.aaofunc.transform_index(step, from_range='window')
            t = pdg.aaoform.time[local_step]
            q_exact.interpolate(fd.exp(.5*x + y + 1.25*t))
            pdg.aaofunc.get_field(local_step, uout=qp)

            errors.dlocal[local_step] = fd.errornorm(qp, q_exact)
            times.dlocal[local_step] = t

    errors.synchronise()
    times.synchronise()

    for step in range(pdg.aaofunc.ntimesteps):
        PETSc.Sys.Print(f"Time={str(times.dglobal[step]).ljust(8, ' ')}, qerr={errors.dglobal[step]}")


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
