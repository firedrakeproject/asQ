from math import pi, cos, sin

from utils.timing import SolverTimer
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
parser.add_argument('--degree', type=int, default=1, help='Degree of the scalar and velocity spaces.')
parser.add_argument('--theta', type=float, default=0.5, help='Parameter for the implicit theta timestepping method.')
parser.add_argument('--width', type=float, default=0.2, help='Width of the Gaussian bump.')
parser.add_argument('--nwindows', type=int, default=1, help='Number of time-windows.')
parser.add_argument('--nslices', type=int, default=2, help='Number of time-slices per time-window.')
parser.add_argument('--slice_length', type=int, default=2, help='Number of timesteps per time-slice.')
parser.add_argument('--alpha', type=float, default=1e-4, help='Circulant coefficient.')
parser.add_argument('--lu_solver', type=str, default='petsc', help='Direct solver for the blocks')
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
# blocks in step (b) of inverting the ParaDiag matrix.
block_parameters = {
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': args.lu_solver,
    'snes': {
        'lag_jacobian': -2,
        'lag_jacobian_persists': None,
        'lag_preconditioner': -2,
        'lag_preconditioner_persists': None,
    },
}

# The PETSc solver parameters for solving the all-at-once system.
# The python preconditioner 'asQ.CirculantPC' applies the ParaDiag matrix.
#
# The equation is linear so we can either:
# a) Solve it in one shot using a preconditioned Krylov method:
#    P^{-1}Au = P^{-1}b
#    The solver options for this are:
#    'ksp_type': 'gmres'
# b) Solve it with stationary iterations:
#    Pu_{k+1} = (P - A)u_{k} + b
#    The solver options for this are:
#    'ksp_type': 'richardson'

atol = 1e-100
rtol = 1e-11
paradiag_parameters = {
    'snes_type': 'ksponly',  # for a linear system
    'snes': {
        # 'monitor': None,
        # 'converged_reason': None,
        'lag_jacobian': -2,
        'lag_jacobian_persists': None,
        'lag_preconditioner': -2,
        'lag_preconditioner_persists': None,
    },
    'mat_type': 'matfree',
    'ksp_type': 'richardson',
    'ksp': {
        'monitor': None,
        'converged_rate': None,
        'rtol': rtol,
        'atol': atol,
        'stol': 1e-12,
    },
    'pc_type': 'python',
    'pc_python_type': 'asQ.CirculantPC',
    'circulant_alpha': args.alpha,
    'circulant_state': 'linear',
    'aaos_jacobian_state': 'linear',
}

# We need to add a block solver parameters dictionary for each block.
# Here they are all the same but they could be different.
comm_size = ensemble.global_comm.size // args.nslices
block_id = ensemble.global_comm.rank // comm_size
paradiag_parameters[f'circulant_block_{block_id}'] = block_parameters
# for i in range(window_length):
#     paradiag_parameters['circulant_block_'+str(i)+'_'] = block_parameters


# # # === --- Setup ParaDiag --- === # # #


# Give everything to asQ to create the paradiag object.
pdg = asQ.Paradiag(ensemble=ensemble,
                   form_function=form_function,
                   form_mass=form_mass,
                   ics=q0, dt=dt, theta=args.theta,
                   time_partition=time_partition,
                   solver_parameters=paradiag_parameters)

# time the calculations
timer = SolverTimer()

# This is a callback which will be called before pdg solves each time-window
# We can use this to make the output a bit easier to read and time the window calculation
def window_preproc(pdg, wndw, rhs):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'### === --- Calculating time-window {wndw} --- === ###')
    PETSc.Sys.Print('')
    with PETSc.Log.Event("window_preproc.Coll_Barrier"):
        pdg.ensemble.ensemble_comm.Barrier()
    timer.start_timing()

# This is a callback which will be called after pdg solves each time-window
# We can use this to time the window calculation
def window_postproc(pdg, wndw, rhs):
    timer.stop_timing()
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'Window solution time: {round(timer.times[-1], 5)}')
    PETSc.Sys.Print('')

# run one window to get all solver objects set up
with PETSc.Log.Event("warmup_solve"):
   pdg.solve(1, window_preproc, window_postproc)

# reset solution and iteration counts
pdg.solver.aaofunc.assign(q0)
pdg.reset_diagnostics()

# Solve nwindows of the all-at-once system
with PETSc.Log.Event("timed_solves"):
   pdg.solve(args.nwindows,
             preproc=window_preproc,
             postproc=window_postproc)

# # # === --- Postprocessing --- === # # #

PETSc.Sys.Print(f'DoFs per timestep: {V.dim()}')
PETSc.Sys.Print(f'Number of MPI ranks per timestep: {mesh.comm.size}')
PETSc.Sys.Print(f'DoFs/rank: {V.dim()/mesh.comm.size}')
PETSc.Sys.Print(f'Block DoFs/rank: {2*V.dim()/mesh.comm.size}')
PETSc.Sys.Print('')

# paradiag collects a few solver diagnostics for us to inspect
nw = args.nwindows

# Number of nonlinear iterations, total and per window.
# (Will be 1 per window for this linear problem)
PETSc.Sys.Print(f'nonlinear iterations: {pdg.nonlinear_iterations}  |  iterations per window: {pdg.nonlinear_iterations/nw}')

# Number of linear iterations of the all-at-once system, total and per window.
PETSc.Sys.Print(f'linear iterations: {pdg.linear_iterations}  |  iterations per window: {pdg.linear_iterations/nw}')

# Number of iterations needed for each block in step-(b), total and per block solve
# The number of iterations for each block will usually be different because of the different eigenvalues
block_iterations = pdg.solver.jacobian.pc.block_iterations
PETSc.Sys.Print(f'block linear iterations: {block_iterations.data()}  |  iterations per block solve: {block_iterations.data()/pdg.linear_iterations}')

if timer.ntimes() > 1:
    timer.times = timer.times[1:]

PETSc.Sys.Print(timer.string(timesteps_per_solve=window_length,
                             total_iterations=pdg.linear_iterations, ndigits=5))
PETSc.Sys.Print('')
