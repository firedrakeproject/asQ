import firedrake as fd
from firedrake.petsc import PETSc
from firedrake.output import VTKFile

from utils.serial import SerialMiniApp
from utils.timing import SolverTimer
from utils.planets import earth
from utils import units

from argparse import ArgumentParser
from argparse_formatter import DefaultsAndRawTextFormatter

import utils.shallow_water as swe
from utils.shallow_water import galewsky
from utils import diagnostics


# get command arguments
parser = ArgumentParser(
    description='Galewsky testcase for serial-in-time solver using fully implicit nonlinear SWE.',
    formatter_class=DefaultsAndRawTextFormatter
)

parser.add_argument('--ref_level', type=int, default=3, help='Refinement level of icosahedral grid. Total number of cells is 20*4^ref_level.')
parser.add_argument('--base_level', type=int, default=2, help='Base refinement level for multigrid.')
parser.add_argument('--nt', type=int, default=10, help='Number of time steps.')
parser.add_argument('--dt', type=float, default=0.5, help='Timestep in hours.')
parser.add_argument('--theta', type=float, default=0.5, help='Parameter for implicit theta method. 0.5 for trapezium rule, 1 for backwards Euler.')
parser.add_argument('--degree', type=float, default=swe.default_degree(), help='Degree of the depth function space.')
parser.add_argument('--vtkfile', type=str, default='vtk/galewsky_serial', help='Name of output vtk files')
parser.add_argument('--write_freq', type=int, default=1, help='How often to write the solution to file.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Setting up --- === ###')
PETSc.Sys.Print('')

# This script broadly follows the same structure as the
# linear_shallow_water_serial.py script, with the main
# differences being:
#
# - a hierarchy of meshes is required for the multigrid method
# - the galewsky test case is used instead of the gravity waves case:
#   Galewsky et al, 2004, "An initial-value problem for testing
#   numerical models  of the global shallow-water equations"
# - the options parameters specify a multigrid scheme not hybridisation

# hierarchy of icosahedral sphere meshes for multigrid
mesh = swe.create_mg_globe_mesh(ref_level=args.ref_level,
                                base_level=args.base_level,
                                coords_degree=1)
x = fd.SpatialCoordinate(mesh)

# time step
dt = args.dt*units.hour

# shallow water equation function spaces (velocity and depth)
W = swe.default_function_space(mesh, degree=args.degree)

# parameters
g = earth.Gravity

b = galewsky.topography_expression(*x)
f = swe.earth_coriolis_expression(*x)

# initial conditions
w_initial = fd.Function(W)
u_initial = w_initial.subfunctions[0]
h_initial = w_initial.subfunctions[1]

u_initial.interpolate(galewsky.velocity_expression(*x))
h_initial.interpolate(galewsky.depth_expression(*x))

# current and next timestep
w0 = fd.Function(W).assign(w_initial)
w1 = fd.Function(W).assign(w_initial)

H = galewsky.H0


# shallow water equation forms
def form_function(u, h, v, q, t):
    return swe.nonlinear.form_function(mesh, g, b, f,
                                       u, h, v, q, t)


def form_mass(u, h, v, q):
    return swe.nonlinear.form_mass(mesh, u, h, v, q)


# grid transfers for non-nested manifolds.
from utils.mg import ManifoldTransferManager  # noqa: F401

# solver parameters for the implicit solve
mg_sparameters = {
    'mat_type': 'matfree',
    'pc_type': 'mg',
    'pc_mg_cycle_type': 'v',
    'pc_mg_type': 'multiplicative',
    'mg': {
        'transfer_manager': f'{__name__}.ManifoldTransferManager',
        'levels': {
            'ksp_type': 'gmres',
            'ksp_max_it': 3,
            'pc_type': 'python',
            'pc_python_type': 'firedrake.PatchPC',
            'patch': {
                'pc_patch': {
                    'save_operators': True,
                    'partition_of_unity': True,
                    'sub_mat_type': 'seqdense',
                    'construct_dim': 0,
                    'construct_type': 'vanka',
                    'local_type': 'additive',
                    'precompute_element_tensors': True,
                    'symmetrise_sweep': False
                },
                'sub': {
                    'ksp_type': 'preonly',
                    'pc_type': 'lu',
                    'pc_factor_shift_type': 'nonzero',
                }
            }
        },
        'coarse': {
            'pc_type': 'python',
            'pc_python_type': 'firedrake.AssembledPC',
            'assembled': {
                'ksp_type': 'preonly',
                'pc_type': 'lu',
                'pc_factor_mat_solver_type': 'mumps',
            },
        },
    }
}

# atol is the same for Newton and (right-preconditioned) Krylov
atol = 1e4
serial_parameters = {
    'snes': {
        'monitor': None,
        'converged_reason': None,
        'rtol': 1e-12,
        'atol': atol,
        'ksp_ew': None,
        'ksp_ew_version': 1,
    },
    'ksp_type': 'fgmres',
    'ksp': {
        'monitor': None,
        'converged_rate': None,
        'atol': atol,
        'rtol': 1e-5,
    },
}
serial_parameters.update(mg_sparameters)

# The SerialMiniApp class will set up the implicit-theta system
# for the serial-in-time method.
miniapp = SerialMiniApp(dt, args.theta, w_initial,
                        form_mass, form_function,
                        serial_parameters)

PETSc.Sys.Print('### === --- Timestepping loop --- === ###')
linear_its = 0
nonlinear_its = 0

ofile = VTKFile(f"{args.vtkfile}.pvd")
uout = fd.Function(u_initial.function_space(), name='velocity')
hout = fd.Function(h_initial.function_space(), name='elevation')

potential_vorticity = diagnostics.potential_vorticity_calculator(
    u_initial.function_space(), name='vorticity')

uout.assign(u_initial)
hout.assign(h_initial)
ofile.write(uout, hout, potential_vorticity(uout), time=0)

timer = SolverTimer()


def preproc(app, step, t):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'=== --- Timestep {step} --- ===')
    PETSc.Sys.Print('')
    with PETSc.Log.Event("timestep_preproc.Coll_Barrier"):
        mesh.comm.Barrier()
    timer.start_timing()


def postproc(app, step, t):
    global linear_its, nonlinear_its
    timer.stop_timing()
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'Timestep solution time: {timer.times[-1]}')
    PETSc.Sys.Print('')

    linear_its += app.nlsolver.snes.getLinearSolveIterations()
    nonlinear_its += app.nlsolver.snes.getIterationNumber()

    if ((step + 1) % args.write_freq) == 0:
        uout.assign(miniapp.w0.subfunctions[0])
        hout.assign(miniapp.w0.subfunctions[1])
        ofile.write(uout, hout, potential_vorticity(uout), time=t)


# Setup all solver objects. The firedrake DM and options management
# makes it difficult to setup some preconditioners without actually
# calling `solve`, so we just run once to set everything up.
PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Setting up solver and prefactoring --- === ###')
PETSc.Sys.Print('')
with PETSc.Log.Event("warmup_solve"):
    miniapp.solve(nt=1)
PETSc.Sys.Print('')

miniapp.w0.assign(w_initial)
miniapp.w1.assign(w_initial)
linear_its -= linear_its
nonlinear_its -= nonlinear_its

PETSc.Sys.Print('### === --- Solving timeseries --- === ###')
PETSc.Sys.Print('')

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

PETSc.Sys.Print(f'Nonlinear iterations: {str(nonlinear_its).rjust(5)}  |  Iterations per window: {str(nonlinear_its/args.nt).rjust(5)}')
PETSc.Sys.Print(f'Linear iterations:    {str(linear_its).rjust(5)}  |  Iterations per window: {str(linear_its/args.nt).rjust(5)}')
PETSc.Sys.Print('')

# Timing measurements
PETSc.Sys.Print(timer.string(timesteps_per_solve=1,
                             total_iterations=linear_its, ndigits=5))
PETSc.Sys.Print('')
