import firedrake as fd
from firedrake.petsc import PETSc
from firedrake.output import VTKFile

from utils.timing import SolverTimer
from utils.serial import SerialMiniApp
from utils.diagnostics import convective_cfl_calculator
from utils import compressible_flow as euler
from math import pi

from argparse import ArgumentParser
from argparse_formatter import DefaultsAndRawTextFormatter


def initial_conditions(mesh, W, Vv, gas, H, perturbation=True, hydrostatic=False):
    if hydrostatic:
        return NotImplementedError

    x, z = fd.SpatialCoordinate(mesh)
    V2 = W.subfunctions[1]
    up = fd.as_vector([fd.Constant(0.0), fd.Constant(1.0)])  # up direction

    Un = fd.Function(W)
    un, rhon, thetan = Un.subfunctions

    Tsurf = fd.Constant(300.)
    thetab = Tsurf*fd.exp(gas.N**2*z/gas.g)

    thetan.interpolate(thetab)

    Pi = fd.Function(V2)

    # background state is in hydrostatic balance
    euler.hydrostatic_rho(Vv, V2, mesh, thetan, rhon,
                          pi_boundary=fd.Constant(1.0),
                          gas=gas, Up=up, top=True, Pi=Pi)

    un.project(fd.as_vector([20.0, 0.0]))

    Uback = Un.copy(deepcopy=True)

    # temperature perturbation to initiate gravity wave
    if perturbation:
        a = fd.Constant(5e3)
        dtheta = fd.Constant(1e-2)

        theta_pert = dtheta*fd.sin(pi*z/H)/(1 + x**2/a**2)
        thetan.interpolate(thetan + theta_pert)

    return Un, Uback


parser = ArgumentParser(
    description='Nonhydrostatic gravity wave testcase for serial-in-time solver using fully implicit vertical slice.',
    formatter_class=DefaultsAndRawTextFormatter
)
parser.add_argument('--nlayers', type=int, default=10, help='Number of layers in the vertical direction.')
parser.add_argument('--ncolumns', type=int, default=300, help='Number of columns in the horizontal direction.')
parser.add_argument('--nt', type=int, default=10, help='Number of timesteps to solve.')
parser.add_argument('--dt', type=float, default=12, help='Timestep in seconds.')
parser.add_argument('--theta', type=float, default=0.5, help='Parameter for implicit theta method. 0.5 for trapezium rule, 1 for backwards Euler.')
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space).')
parser.add_argument('--vtkfile', type=str, default='vtk/gravity_wave_serial', help='Name of output vtk files')
parser.add_argument('--write_freq', type=int, default=1, help='How often to write the solution to file.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Setting up --- === ###')
PETSc.Sys.Print('')

# This script broadly follows the same structure
# as the advection_serial.py script, with the main
# differences being:
#
# - more complicated finite element spaces and forms are required,
#   which are provided by the utils.compressible_flow module.
# - more complicated initial conditions (the function at the top of
#   the file), from the test case of
#   Skamarock & Klemp, 1994, "Efficiency and accuracy of the
#   Klemp-Wilhelmson time-splitting technique".
# - options parameters specifying more involved block preconditioning.

comm = fd.COMM_WORLD

# parameters

nt = args.nt
dt = args.dt

# set up the mesh
L = 300e3
H = 10e3

distribution_parameters = {
    "partition": True,
    "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)
}

base_mesh = fd.PeriodicIntervalMesh(
    args.ncolumns, float(L),
    distribution_parameters=distribution_parameters,
    comm=comm)
base_mesh.coordinates.dat.data[:] -= float(L)/2

mesh = fd.ExtrudedMesh(base_mesh,
                       layers=args.nlayers,
                       layer_height=float(H/args.nlayers))
n = fd.FacetNormal(mesh)

# compatible function spaces

W, Vv = euler.function_space(mesh, horizontal_degree=args.degree,
                             vertical_degree=args.degree,
                             vertical_velocity_space=True)
V1, V2, Vt = W.subfunctions  # velocity, density, temperature

PETSc.Sys.Print(f"DoFs: {W.dim()}")
PETSc.Sys.Print(f"DoFs/core: {W.dim()/comm.size}")
PETSc.Sys.Print("")

PETSc.Sys.Print("Calculating initial condiions")

# ideal gas properties
gas = euler.StandardAtmosphere(N=0.01)

# initial conditions and background state
Un, Uback = initial_conditions(mesh, W, Vv, gas, H)
uback, rho_back, theta_back = Uback.subfunctions

U0 = Uback.copy(deepcopy=True)  # background state at rest
U0.subfunctions[0].assign(0)

# finite element forms

form_mass = euler.get_form_mass()

up = fd.as_vector([fd.Constant(0.0), fd.Constant(1.0)])  # up direction

form_function = euler.get_form_function(
    n, up, c_pen=fd.Constant(2.0**(-7./2)), gas=gas, mu=None)

# tangential flow boundary conditions at ground and lid
zv = fd.as_vector([fd.Constant(0.), fd.Constant(0.)])
bcs = [fd.DirichletBC(W.sub(0), zv, "bottom"),
       fd.DirichletBC(W.sub(0), zv, "top")]
for bc in bcs:
    bc.apply(Un)

# Parameters for the newton iterations
patch_parameters = {
    'pc_type': 'python',
    'pc_python_type': 'firedrake.AssembledPC',
    'assembled': {
        'pc_type': 'python',
        'pc_python_type': 'firedrake.ASMStarPC',
        'pc_star': {
            'construct_dim': 0,
            'sub_sub_pc_type': 'lu',
            'sub_sub_pc_factor_mat_solver_type': 'mumps',
        },
    },
}

aux_parameters = {
    'pc_type': 'python',
    'pc_python_type': 'asQ.AuxiliaryRealBlockPC',
    'aux': {
        'frozen': None,  # never update the factorisation
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
    }
}

block_parameters = {
    'pc_type': 'composite',
    'pc_composite_type': 'multiplicative',
    'pc_composite_pcs': 'ksp,ksp',
    'sub_0_ksp': aux_parameters,
    'sub_0_ksp_ksp': {
        'type': 'gmres',
        'max_it': 2,
        'convergence_test': 'skip',
        'converged_maxits': None,
    },
    'sub_1_ksp': patch_parameters,
    'sub_1_ksp_ksp': {
        'type': 'gmres',
        'max_it': 2,
        'convergence_test': 'skip',
        'converged_maxits': None,
    },
}

# atol is the same for Newton and (right-preconditioned) Krylov
atol = 1e-4
solver_parameters = {
    'snes': {
        'linesearch_type': 'basic',
        'monitor': None,
        'converged_reason': None,
        'stol': 1e-100,
        'atol': atol,
        'ksp_ew': None,
        'ksp_ew_version': 1,
        'ksp_ew_rtol0': 1e-1,
        'ksp_ew_threshold': 1e-10,
        'lag_preconditioner': -2,  # preconditioner is constant-in-time
        'lag_preconditioner_persists': None,
    },
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp': {
        'monitor': None,
        'converged_rate': None,
        'atol': atol,
        'max_it': 30,
        'converged_maxits': None,
        "min_it": 1,
    },
}
solver_parameters.update(block_parameters)

# reference state at rest for the wave preconditioner
appctx = {'aux_uref': U0}

miniapp = SerialMiniApp(dt=dt, theta=args.theta, w_initial=Un,
                        form_mass=form_mass,
                        form_function=form_function,
                        solver_parameters=solver_parameters,
                        bcs=bcs, appctx=appctx)

PETSc.Sys.Print('### === --- Timestepping loop --- === ###')
linear_its = 0
nonlinear_its = 0

uout = fd.Function(V1, name='velocity')
thetaout = fd.Function(Vt, name='temperature')
rhoout = fd.Function(V2, name='density')

ofile = VTKFile(f'{args.vtkfile}.pvd', comm=comm)


def assign_out_functions():
    uout.assign(miniapp.w0.subfunctions[0])
    rhoout.assign(miniapp.w0.subfunctions[1])
    thetaout.assign(miniapp.w0.subfunctions[2])

    # output density and temperature variations, not absolute values
    rhoout.assign(rhoout - rho_back)
    thetaout.assign(thetaout - theta_back)


def write_to_file(time):
    assign_out_functions()
    ofile.write(uout, rhoout, thetaout, t=time)


# Setup all solver objects. The firedrake DM and options management
# makes it difficult to setup some preconditioners without actually
# calling `solve`, so we just run once to set everything up.
PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Setting up solver and prefactoring --- === ###')
PETSc.Sys.Print('')
with PETSc.Log.Event("warmup_solve"):
    miniapp.solve(nt=1)
miniapp.w0.assign(Un)
miniapp.w1.assign(Un)
PETSc.Sys.Print('')

linear_its = 0
nonlinear_its = 0

cfl_calc = convective_cfl_calculator(mesh)
cfl_series = []

timer = SolverTimer()


def max_cfl(u, dt):
    with cfl_calc(u, dt).dat.vec_ro as v:
        return v.max()[1]


def preproc(app, step, time):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'### === --- Calculating time-step {step} --- === ###')
    PETSc.Sys.Print('')
    with PETSc.Log.Event("timestep_preproc.Coll_Barrier"):
        comm.Barrier()
    timer.start_timing()


def postproc(app, step, time):
    global linear_its
    global nonlinear_its
    timer.stop_timing()
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'Timestep solution time: {timer.times[-1]}')
    PETSc.Sys.Print('')

    linear_its += miniapp.nlsolver.snes.getLinearSolveIterations()
    nonlinear_its += miniapp.nlsolver.snes.getIterationNumber()

    if ((step + 1) % args.write_freq) == 0:
        write_to_file(time=time)

    cfl = max_cfl(uout, dt)
    cfl_series.append(cfl)
    PETSc.Sys.Print(f'Time = {time}')
    PETSc.Sys.Print(f'Maximum CFL = {cfl}')


# solve for each window
PETSc.Sys.Print('### === --- Solving timeseries --- === ###')
PETSc.Sys.Print('')
with PETSc.Log.Event("timed_solves"):
    miniapp.solve(nt=nt,
                  preproc=preproc,
                  postproc=postproc)
PETSc.Sys.Print('')

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
