import firedrake as fd
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
from utils.diagnostics import convective_cfl_calculator
from utils.serial import SerialMiniApp
from utils import compressible_flow as euler
from utils.vertical_slice import mount_agnesi as mountain

import argparse
parser = argparse.ArgumentParser(description='Mountain testcase.')
parser.add_argument('--nlayers', type=int, default=35, help='Number of layers, default 10.')
parser.add_argument('--ncolumns', type=int, default=90, help='Number of columns, default 10.')
parser.add_argument('--nt', type=int, default=1, help='Number of timesteps to solve.')
parser.add_argument('--dt', type=float, default=5, help='Timestep in seconds. Default 1.')
parser.add_argument('--atol', type=float, default=1e-3, help='Average absolute tolerance for each timestep')
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space). Default 1.')
parser.add_argument('--filename', type=str, default='slice_mountain', help='Name of vtk file.')
parser.add_argument('--write_file', action='store_true', help='Write vtk file at end of each window.')
parser.add_argument('--output_freq', type=int, default=10, help='How often to write to file.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

PETSc.Sys.Print("Setting up problem")

comm = fd.COMM_WORLD

# set up the mesh

output_freq = args.output_freq
nt = args.nt
dt = args.dt

mesh = mountain.mesh(comm, ncolumns=args.ncolumns,
                     nlayers=args.nlayers,
                     hydrostatic=False)
n = fd.FacetNormal(mesh)
x, z = fd.SpatialCoordinate(mesh)

W, Vv = euler.function_space(mesh, horizontal_degree=args.degree,
                             vertical_degree=args.degree,
                             vertical_velocity_space=True)
V1, V2, Vt = W.subfunctions  # velocity, density, temperature

PETSc.Sys.Print(f"DoFs: {W.dim()}")
PETSc.Sys.Print(f"DoFs/core: {W.dim()/comm.size}")

gas = euler.StandardAtmosphere(N=0.01)

dT = fd.Constant(dt)

Un = fd.Function(W)

PETSc.Sys.Print("Calculating hydrostatic state")

Un = mountain.initial_conditions(mesh, W, Vv, gas)
rho_back = fd.Function(V2).assign(Un.subfunctions[1])
theta_back = fd.Function(Vt).assign(Un.subfunctions[2])

mu = mountain.sponge_layer(mesh, V2, dt)

form_mass = euler.get_form_mass()

up = fd.as_vector([fd.Constant(0.0), fd.Constant(1.0)])  # up direction
form_function = euler.get_form_function(
    n, up, c_pen=fd.Constant(2.0**(-7./2)), gas=gas, mu=mu)

bcs = mountain.boundary_conditions(W)
for bc in bcs:
    bc.apply(Un)

# Parameters for the newton iterations
solver_parameters = {
    "snes": {
        "monitor": None,
        "converged_reason": None,
        "stol": 1e-12,
        "atol": args.atol,
        "rtol": 1e-8,
        "ksp_ew": None,
        "ksp_ew_version": 1,
        "ksp_ew_threshold": 1e-5,
        "ksp_ew_rtol0": 1e-3,
    },
    "ksp_type": "fgmres",
    "ksp": {
        "monitor": None,
        "converged_reason": None,
        "atol": args.atol,
    },
    "pc_type": "python",
    "pc_python_type": "firedrake.AssembledPC",
    "assembled": {
        "pc_type": "python",
        "pc_python_type": "firedrake.ASMVankaPC",
        "pc_vanka": {
            "construct_dim": 0,
            "sub_sub_pc_type": "lu",
            "sub_sub_pc_factor_mat_solver_type": 'mumps',
        },
    },
}

theta = 0.5

miniapp = SerialMiniApp(dt=dt, theta=theta, w_initial=Un,
                        form_mass=form_mass,
                        form_function=form_function,
                        solver_parameters=solver_parameters,
                        bcs=bcs)

PETSc.Sys.Print("Solving problem")

uout = fd.Function(V1, name='velocity')
thetaout = fd.Function(Vt, name='temperature')
rhoout = fd.Function(V2, name='density')

ofile = fd.File('output/slice_mountain.pvd',
                comm=comm)


def assign_out_functions():
    uout.assign(miniapp.w0.subfunctions[0])
    rhoout.assign(miniapp.w0.subfunctions[1])
    thetaout.assign(miniapp.w0.subfunctions[2])

    rhoout.assign(rhoout - rho_back)
    thetaout.assign(thetaout - theta_back)


def write_to_file(time):
    ofile.write(uout, rhoout, thetaout, t=time)


# assign_out_functions()
# write_to_file(time=0)

PETSc.Sys.Print('### === --- Timestepping loop --- === ###')
linear_its = 0
nonlinear_its = 0
solver_time = []

cfl_calc = convective_cfl_calculator(mesh)
cfl_series = []


def max_cfl(u, dt):
    with cfl_calc(u, dt).dat.vec_ro as v:
        return v.max()[1]


def preproc(app, it, time):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'### === --- Calculating time-step {it} --- === ###')
    PETSc.Sys.Print('')
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
        write_to_file(time=time)

    cfl = max_cfl(uout, dt)
    cfl_series.append(cfl)
    PETSc.Sys.Print(f'Time = {time}')
    PETSc.Sys.Print(f'Maximum CFL = {cfl}')


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
