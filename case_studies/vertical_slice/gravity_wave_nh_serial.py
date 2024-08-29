import firedrake as fd
from firedrake.petsc import PETSc
from firedrake.output import VTKFile
from pyop2.mpi import MPI
from utils.diagnostics import convective_cfl_calculator
from utils import compressible_flow as euler
from utils.serial import SerialMiniApp
import numpy as np


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

    euler.hydrostatic_rho(Vv, V2, mesh, thetan, rhon,
                          pi_boundary=fd.Constant(1.0),
                          gas=gas, Up=up, top=True, Pi=Pi)

    un.project(fd.as_vector([20.0, 0.0]))

    Uback = Un.copy(deepcopy=True)

    if perturbation:
        a = fd.Constant(5e3)
        dtheta = fd.Constant(1e-2)

        theta_pert = dtheta*fd.sin(np.pi*z/H)/(1 + x**2/a**2)
        thetan.interpolate(thetan + theta_pert)

    return Un, Uback


import argparse
parser = argparse.ArgumentParser(description='Nonhydrostatic gravity wave.')
parser.add_argument('--nlayers', type=int, default=5, help='Number of layers.')
parser.add_argument('--ncolumns', type=int, default=150, help='Number of columns.')
parser.add_argument('--nt', type=int, default=10, help='Number of timesteps to solve.')
parser.add_argument('--dt', type=float, default=12, help='Timestep in seconds.')
parser.add_argument('--atol', type=float, default=1e-4, help='Average absolute tolerance for each timestep')
parser.add_argument('--rtol', type=float, default=1e-8, help='Relative absolute tolerance for each timestep')
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space). Default 1.')
parser.add_argument('--filename', type=str, default='gravity_wave_nh', help='Name of vtk file.')
parser.add_argument('--write_file', action='store_true', help='Write vtk file at end of each timestep.')
parser.add_argument('--output_freq', type=int, default=10, help='How often to write to file.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

PETSc.Sys.Print("Setting up problem")

comm = fd.COMM_WORLD

# parameters

output_freq = args.output_freq
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

# function spaces

W, Vv = euler.function_space(mesh, horizontal_degree=args.degree,
                             vertical_degree=args.degree,
                             vertical_velocity_space=True)
V1, V2, Vt = W.subfunctions  # velocity, density, temperature

PETSc.Sys.Print(f"DoFs: {W.dim()}")
PETSc.Sys.Print(f"DoFs/core: {W.dim()/comm.size}")
PETSc.Sys.Print("")

PETSc.Sys.Print("Calculating initial condiions")

gas = euler.StandardAtmosphere(N=0.01)

Un, Uback = initial_conditions(mesh, W, Vv, gas, H)
uback, rho_back, theta_back = Uback.subfunctions

U0 = Uback.copy(deepcopy=True)  # background state at rest
U0.subfunctions[0].assign(0)

# finite element forms

form_mass = euler.get_form_mass()

up = fd.as_vector([fd.Constant(0.0), fd.Constant(1.0)])  # up direction

form_function = euler.get_form_function(
    n, up, c_pen=fd.Constant(2.0**(-7./2)), gas=gas, mu=None)

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
            'sub_sub_pc_factor_mat_solver_type': 'petsc',
        },
    },
}

aux_parameters = {
    'pc_type': 'python',
    'pc_python_type': 'asQ.AuxiliaryRealBlockPC',
    'aux': {
        'frozen': None,
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
    }
}

composite_parameters = {
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

solver_parameters = {
    'snes': {
        'monitor': None,
        'converged_reason': None,
        'atol': args.atol,
        "rtol": args.rtol,
        'ksp_ew': None,
        'ksp_ew_version': 2,
        'ksp_ew_rtol0': 1e-2,
        'ksp_ew_threshold': 1e-5,
        'lag_preconditioner': -2,
        'lag_preconditioner_persists': None,
        'lag_jacobian': -2,
        'lag_jacobian_persists': None,
    },
    'ksp_type': 'fgmres',
    # 'ksp_pc_side': 'right',
    # 'ksp_norm_type': 'unpreconditioned',
    'ksp': {
        'monitor': None,
        'converged_rate': None,
        'atol': 1.0*args.atol,
        'max_it': 50,
        "min_it": 1,
        # 'view': None,
    },
}

solver_parameters.update(composite_parameters)

appctx = {'aux_uref': U0}

theta = 0.5

miniapp = SerialMiniApp(dt=dt, theta=theta, w_initial=Un,
                        form_mass=form_mass,
                        form_function=form_function,
                        solver_parameters=solver_parameters,
                        bcs=bcs, appctx=appctx)

PETSc.Sys.Print("Solving problem")

uout = fd.Function(V1, name='velocity')
thetaout = fd.Function(Vt, name='temperature')
rhoout = fd.Function(V2, name='density')

ofile = VTKFile(f'output/{args.filename}.pvd', comm=comm)


def assign_out_functions():
    uout.assign(miniapp.w0.subfunctions[0])
    rhoout.assign(miniapp.w0.subfunctions[1])
    thetaout.assign(miniapp.w0.subfunctions[2])

    rhoout.assign(rhoout - rho_back)
    thetaout.assign(thetaout - theta_back)


def write_to_file(time):
    ofile.write(uout, rhoout, thetaout, time=time)


PETSc.Sys.Print('### === --- Timestepping loop --- === ###')
PETSc.Sys.Print('')
linear_its = 0
nonlinear_its = 0
solver_time = []

cfl_calc = convective_cfl_calculator(mesh)
cfl_series = []


def max_cfl(u, dt):
    with cfl_calc(u, dt).dat.vec_ro as v:
        return v.max()[1]


assign_out_functions()
cfl = max_cfl(uout, dt)
cfl_series.append(cfl)
PETSc.Sys.Print(f'Maximum initial CFL = {round(cfl, 4)}')
PETSc.Sys.Print('')


def preproc(app, it, time):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'### === --- Calculating time-step {it} --- === ###')
    PETSc.Sys.Print('')
    comm.Barrier()
    stime = MPI.Wtime()
    solver_time.append(stime)


def postproc(app, it, time):
    global linear_its
    global nonlinear_its

    etime = MPI.Wtime()
    stime = solver_time[-1]
    duration = etime - stime
    solver_time[-1] = duration
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'Timestep solution time: {duration}\n')
    PETSc.Sys.Print('')

    linear_its += miniapp.nlsolver.snes.getLinearSolveIterations()
    nonlinear_its += miniapp.nlsolver.snes.getIterationNumber()

    if (it % output_freq) == 0:
        assign_out_functions()
        if args.write_file:
            write_to_file(time=time)

    cfl = max_cfl(uout, dt)
    cfl_series.append(cfl)
    PETSc.Sys.Print(f'Time = {time}')
    PETSc.Sys.Print(f'Maximum CFL = {round(cfl, 4)}')


# solve for each window
miniapp.solve(nt=nt,
              preproc=preproc,
              postproc=postproc)

PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Iteration counts --- === ###')
PETSc.Sys.Print('')

PETSc.Sys.Print(f'linear iterations: {linear_its} | iterations per timestep: {linear_its/nt}')
PETSc.Sys.Print(f'nonlinear iterations: {nonlinear_its} | iterations per timestep: {nonlinear_its/nt}')
PETSc.Sys.Print('')

PETSc.Sys.Print(f'Total DoFs: {W.dim()}')
PETSc.Sys.Print(f'Number of MPI ranks: {mesh.comm.size} ')
PETSc.Sys.Print(f'DoFs/rank: {W.dim()/mesh.comm.size}')
PETSc.Sys.Print('')

if len(solver_time) > 1:
    # solver_time = solver_time[1:]
    solver_time[0] = solver_time[1]

PETSc.Sys.Print(f'Total solution time: {sum(solver_time)}')
PETSc.Sys.Print(f'Average timestep solution time: {sum(solver_time)/len(solver_time)}')
PETSc.Sys.Print('')
