import firedrake as fd
from firedrake.petsc import PETSc
from firedrake.output import VTKFile

from utils.serial import SerialMiniApp
from utils.timing import SolverTimer
from utils.planets import earth
from utils import units

from argparse import ArgumentParser
from argparse_formatter import DefaultsAndRawTextFormatter


# get command arguments
parser = ArgumentParser(
    description='Gravity wave testcase for serial-in-time solver using fully implicit linear SWE.',
    formatter_class=DefaultsAndRawTextFormatter
)

parser.add_argument('--ref_level', type=int, default=3, help='Refinement level of icosahedral grid. Total number of cells is 20*4^ref_level.')
parser.add_argument('--nt', type=int, default=20, help='Number of time steps.')
parser.add_argument('--dt', type=float, default=0.25, help='Timestep in hours.')
parser.add_argument('--theta', type=float, default=0.5, help='Parameter for implicit theta method.')
parser.add_argument('--vtkfile', type=str, default='vtk/gravity_waves_serial', help='Name of output vtk files')
parser.add_argument('--write_freq', type=int, default=1, help='How often to write the solution to file.')
parser.add_argument('--show_args', action='store_true', help='Print all the arguments when the script starts.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Setting up --- === ###')
PETSc.Sys.Print('')

# Icosahedral sphere mesh
distribution_parameters = {
    "partition": True,
    "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)
}
mesh = fd.IcosahedralSphereMesh(
    radius=earth.radius, refinement_level=args.ref_level,
    distribution_parameters=distribution_parameters)
x = fd.SpatialCoordinate(mesh)
mesh.init_cell_orientations(x)

# time step
dt = args.dt*units.hour

# We have a variety of utilities for the shallow water equations
# in the utils module. The two used here are:
#
# - The function spaces, and compatible finite element forms
#   for the linearisation around a state of rest.
#
# - The gravity_bumps submodule has expressions for the initial
#   conditions for the test case of:
#   Schreiber & Loft, 2019, "A Parallel Time-Integrator for Solving
#   the Linearized Shallow Water Equations on the Rotating Sphere"

import utils.shallow_water as swe
import utils.shallow_water.gravity_bumps as gwcase

# shallow water equation function spaces (velocity and depth)
W = swe.default_function_space(mesh)

# parameters
g = earth.Gravity
H = gwcase.H
f = gwcase.coriolis_expression(*x)

# initial conditions
w_initial = fd.Function(W)
u_initial, h_initial = w_initial.subfunctions

u_initial.project(gwcase.velocity_expression(*x))
h_initial.project(gwcase.depth_expression(*x))


# shallow water equation forms
def form_mass(u, h, v, q):
    return swe.linear.form_mass(mesh, u, h, v, q)


def form_function(u, h, v, q, t):
    return swe.linear.form_function(mesh, g, H, f,
                                    u, h, v, q, t)


# solver parameters for the implicit solve

from utils.hybridisation import HybridisedSCPC  # noqa: F401
block_parameters = {
    "mat_type": "matfree",
    'ksp_type': 'preonly',
    "pc_type": "python",
    "pc_python_type": f"{__name__}.HybridisedSCPC",
    "hybridscpc_condensed_field": {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
    }
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
        'monitor': None,
        'converged_rate': None,
        'rtol': 1e-11,
    },
}
serial_parameters.update(block_parameters)

# The SerialMiniApp class will set up the implicit-theta system
# for the serial-in-time method.
miniapp = SerialMiniApp(dt, args.theta, w_initial,
                        form_mass, form_function,
                        serial_parameters)

PETSc.Sys.Print('### === --- Timestepping loop --- === ###')
linear_its = 0
nonlinear_its = 0

ofile = VTKFile(args.vtkfile+'.pvd')
uout = fd.Function(u_initial.function_space(), name='velocity')
hout = fd.Function(h_initial.function_space(), name='depth')

uout.assign(u_initial)
hout.assign(h_initial - gwcase.H)
ofile.write(uout, hout, time=0)

timer = SolverTimer()


# This function will be called before solving each timestep. We can use
# this to make the output a bit easier to read, and to time the calculation
def preproc(app, step, t):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'=== --- Timestep {step} --- ===')
    PETSc.Sys.Print('')
    # for now we are interested in timing only the solve, this
    # makes sure we don't time any synchronisation after prints.
    with PETSc.Log.Event("timestep_preproc.Coll_Barrier"):
        mesh.comm.Barrier()
    timer.start_timing()


# This function will be called after solving each timestep. We can use
# this to finish the timestep calculation timing and print the result,
# to record the number of iterations, and to write the solution to file.
def postproc(app, step, t):
    global linear_its, nonlinear_its
    timer.stop_timing()
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'Timestep solution time: {timer.times[-1]}')
    PETSc.Sys.Print('')

    linear_its += app.nlsolver.snes.getLinearSolveIterations()
    nonlinear_its += app.nlsolver.snes.getIterationNumber()

    if ((step + 1) % args.write_freq) == 0:
        u, h = app.w0.subfunctions
        uout.assign(u)
        hout.assign(h-gwcase.H)
        ofile.write(uout, hout, time=t/units.hour)


# Setup all solver objects. The firedrake DM and options management
# makes it difficult to setup some preconditioners without actually
# calling `solve`, so we just run once to set everything up.
PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Setting up solver and prefactoring --- === ###')
PETSc.Sys.Print('')
with PETSc.Log.Event("warmup_solve"):
    miniapp.solve(nt=1)
PETSc.Sys.Print('')

PETSc.Sys.Print('### === --- Solving timeseries --- === ###')
PETSc.Sys.Print('')

# Solve nt timesteps
with PETSc.Log.Event("timed_solves"):
    miniapp.solve(nt=args.nt,
                  preproc=preproc,
                  postproc=postproc)

PETSc.Sys.Print('### === --- Iteration counts --- === ###')
PETSc.Sys.Print('')

# parallelism
PETSc.Sys.Print(f'DoFs per timestep: {W.dim()}')
PETSc.Sys.Print(f'Number of MPI ranks per timestep: {mesh.comm.size}')
PETSc.Sys.Print(f'DoFs/rank: {W.dim()/mesh.comm.size}')
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
