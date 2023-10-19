import firedrake as fd
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
from math import pi
from utils.misc import function_maximum
from utils.diagnostics import convective_cfl_calculator
from utils.serial import SerialMiniApp
from utils import compressible_flow as euler

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

nlayers = args.nlayers  # horizontal layers
base_columns = args.ncolumns  # number of columns
L = 144e3
H = 35e3  # Height position of the model top

distribution_parameters = {
    "partition": True,
    "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)
}

# surface mesh of ground
base_mesh = fd.PeriodicIntervalMesh(base_columns, L,
                                    distribution_parameters=distribution_parameters,
                                    comm=comm)

# volume mesh of the slice
mesh = fd.ExtrudedMesh(base_mesh,
                       layers=nlayers,
                       layer_height=H/nlayers)
n = fd.FacetNormal(mesh)
x, z = fd.SpatialCoordinate(mesh)

gas = euler.StandardAtmosphere(N=0.01)

dT = fd.Constant(dt)

# making a mountain out of a molehill
a = 10000.
xc = L/2.
hm = 1.
zs = hm*a**2/((x-xc)**2 + a**2)

smooth_z = True
name = "mountain_nh"
if smooth_z:
    name += '_smootherz'
    zh = 5000.
    xexpr = fd.as_vector([x, fd.conditional(z < zh, z + fd.cos(0.5*pi*z/zh)**6*zs, z)])
else:
    xexpr = fd.as_vector([x, z + ((H-z)/H)*zs])
mesh.coordinates.interpolate(xexpr)

W, Vv = euler.function_space(mesh, horizontal_degree=args.degree,
                             vertical_degree=args.degree,
                             vertical_velocity_space=True)
V1, V2, Vt = W.subfunctions  # velocity, density, temperature

PETSc.Sys.Print(f"DoFs: {W.dim()}")
PETSc.Sys.Print(f"DoFs/core: {W.dim()/comm.size}")

Un = fd.Function(W)

# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
Tsurf = fd.Constant(300.)
thetab = Tsurf*fd.exp(gas.N**2*z/gas.g)

Up = fd.as_vector([fd.Constant(0.0), fd.Constant(1.0)])  # up direction

un, rhon, thetan = Un.subfunctions
un.project(fd.as_vector([10.0, 0.0]))
thetan.interpolate(thetab)
theta_back = fd.Function(Vt).assign(thetan)
rhon.assign(1.0e-5)

PETSc.Sys.Print("Calculating hydrostatic state")

Pi = fd.Function(V2)

euler.hydrostatic_rho(Vv, V2, mesh, thetan, rhon,
                      pi_boundary=fd.Constant(0.02),
                      gas=gas, Up=Up, top=True, Pi=Pi)
p0 = function_maximum(Pi)

euler.hydrostatic_rho(Vv, V2, mesh, thetan, rhon,
                      pi_boundary=fd.Constant(0.05),
                      gas=gas, Up=Up, top=True, Pi=Pi)
p1 = function_maximum(Pi)
alpha = 2.*(p1-p0)
beta = p1-alpha
pi_top = (1.-beta)/alpha

euler.hydrostatic_rho(Vv, V2, mesh, thetan, rhon,
                      pi_boundary=fd.Constant(pi_top),
                      gas=gas, Up=Up, top=True)

rho_back = fd.Function(V2).assign(rhon)

zc = fd.Constant(H-10000.)
mubar = fd.Constant(0.15/dt)
mu_top = fd.conditional(z <= zc, 0.0,
                        mubar*fd.sin(fd.Constant(pi/2.)*(z-zc)/(fd.Constant(H)-zc))**2)
mu = fd.Function(V2).interpolate(mu_top)

form_mass = euler.get_form_mass()

form_function = euler.get_form_function(
    n, Up, c_pen=fd.Constant(2.0**(-7./2)), gas=gas, mu=mu)

zv = fd.as_vector([fd.Constant(0.), fd.Constant(0.)])
bcs = [fd.DirichletBC(W.sub(0), zv, "bottom"),
       fd.DirichletBC(W.sub(0), zv, "top")]

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
