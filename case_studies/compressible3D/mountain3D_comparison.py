import firedrake as fd
from math import pi, sqrt
from utils.serial import ComparisonMiniapp
from firedrake.petsc import PETSc
from utils.misc import function_maximum
from utils import compressible_flow as euler
import asQ

import argparse
parser = argparse.ArgumentParser(
    description='3D mountain testcase.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--nx', type=int, default=16, help='Number of columns in the streamwise direction.')
parser.add_argument('--ny', type=int, default=16, help='Number of columns in the spanwise direction.')
parser.add_argument('--nz', type=int, default=16, help='Number of layers.')
parser.add_argument('--Lx', type=float, default=32e3, help='Streamwise length of domain.')
parser.add_argument('--Ly', type=float, default=32e3, help='Spanwise length of domain.')
parser.add_argument('--Lz', type=float, default=32e3, help='Vertical length of domain.')
parser.add_argument('--nwindows', type=int, default=1, help='Number of windows to solve.')
parser.add_argument('--nslices', type=int, default=2, help='Number of slices in the all-at-once system.')
parser.add_argument('--slice_length', type=int, default=2, help='Number of timesteps in each slice of the all-at-once system.')
parser.add_argument('--atol', type=float, default=1e-1, help='Average absolute tolerance for each timestep.')
parser.add_argument('--dt', type=float, default=5, help='Timestep in seconds. Default 1.')
parser.add_argument('--alpha', type=float, default=1e-3, help='Circulant parameter')
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space). Default 1.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

PETSc.Sys.Print("Setting up problem")

# set up the ensemble communicator for space-time parallelism
time_partition = tuple((args.slice_length for _ in range(args.nslices)))

ensemble = asQ.create_ensemble(time_partition, comm=fd.COMM_WORLD)

comm = ensemble.comm

# set up the mesh
dt = args.dt

nx = args.nx  # number streamwise of columns
ny = args.ny  # number spanwise of columns
nz = args.nz  # horizontal layers

Lx = args.Lx
Ly = args.Ly
Lz = args.Lz  # Height position of the model top

distribution_parameters = {
    "partition": True,
    "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)
}

# surface mesh of ground
base_mesh = fd.PeriodicRectangleMesh(nx, ny, Lx, Ly,
                                     direction='both', quadrilateral=True,
                                     distribution_parameters=distribution_parameters,
                                     comm=comm)

# volume mesh of the slice
mesh = fd.ExtrudedMesh(base_mesh, layers=nz, layer_height=Lz/nz)
n = fd.FacetNormal(mesh)
x, y, z = fd.SpatialCoordinate(mesh)

gas = euler.StandardAtmosphere(N=0.01)

dT = fd.Constant(dt)

# making a mountain out of a molehill
a = 1000.
xc = Lx/2.
yc = Ly/2.
x, y, z = fd.SpatialCoordinate(mesh)
hm = 1.
r2 = ((x - xc)/a)**2 + ((y - yc)/(2*a))**2
zs = hm*fd.exp(-r2)

smooth_z = True
name = "mountain_nh"
if smooth_z:
    name += '_smootherz'
    zh = 5000.
    xexpr = fd.as_vector([x, y, fd.conditional(z < zh, z + fd.cos(0.5*pi*z/zh)**6*zs, z)])
else:
    xexpr = fd.as_vector([x, y, z + ((Lz-z)/Lz)*zs])
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

Up = fd.as_vector([fd.Constant(0.0), fd.Constant(0.0), fd.Constant(1.0)])  # up direction

un, rhon, thetan = Un.subfunctions
un.project(fd.as_vector([10.0, 0.0, 0.0]))
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

zc = fd.Constant(Lz-10000.)
mubar = fd.Constant(0.15/dt)
mu_top = fd.conditional(z <= zc, 0.0,
                        mubar*fd.sin(fd.Constant(pi/2.)*(z-zc)/(Lz-zc))**2)
mu = fd.Function(V2).interpolate(mu_top)

form_mass = euler.get_form_mass()

form_function = euler.get_form_function(
    n, Up, c_pen=fd.Constant(2.0**(-7./2)),
    gas=gas, mu=mu)

zv = fd.as_vector([fd.Constant(0.), fd.Constant(0.), fd.Constant(0.)])
bcs = [fd.DirichletBC(W.sub(0), zv, "bottom"),
       fd.DirichletBC(W.sub(0), zv, "top")]

for bc in bcs:
    bc.apply(Un)

# Parameters for the newton iterations
atol = args.atol
rtol = 1e-8

lines_parameters = {
    "ksp_type": "gmres",
    "ksp_rtol": 1e-4,
    "pc_type": "python",
    "pc_python_type": "firedrake.AssembledPC",
    "assembled": {
        "pc_type": "python",
        "pc_python_type": "firedrake.ASMVankaPC",
        "pc_vanka": {
            "construct_dim": 0,
            "sub_sub": {
                "pc_type": "lu",
                "pc_factor_mat_ordering_type": "rcm",
                "pc_factor_reuse_ordering": None,
                "pc_factor_reuse_fill": None,
            }
        },
    },
}

serial_parameters = {
    "snes": {
        "atol": atol,
        "rtol": rtol,
        "stol": 1e-12,
        "ksp_ew": None,
        "ksp_ew_version": 1,
        "ksp_ew_threshold": 1e-5,
        "ksp_ew_rtol0": 1e-3,
    },
    "ksp": {
        "atol": atol,
    },
}
serial_parameters.update(lines_parameters)
serial_parameters["ksp_type"] = "fgmres"

if ensemble.ensemble_comm.rank == 0:
    serial_parameters['snes']['monitor'] = None
    serial_parameters['snes']['converged_reason'] = None
    serial_parameters['ksp']['monitor'] = None
    serial_parameters['ksp']['converged_reason'] = None

patol = sqrt(sum(time_partition))*atol
parallel_parameters = {
    "snes": {
        "monitor": None,
        "converged_reason": None,
        "atol": patol,
        "rtol": rtol,
        "stol": 1e-12,
        "ksp_ew": None,
        "ksp_ew_version": 1,
        "ksp_ew_threshold": 1e-5,
        "ksp_ew_rtol0": 1e-3,
        "lag_preconditioner": 100
    },
    "ksp_type": "fgmres",
    "mat_type": "matfree",
    "ksp": {
        "monitor": None,
        "converged_reason": None,
        "atol": patol,
    },
    "pc_type": "python",
    "pc_python_type": "asQ.CirculantPC",
    "diagfft_alpha": args.alpha,
}

for i in range(sum(time_partition)):
    parallel_parameters["diagfft_block_"+str(i)+"_"] = lines_parameters
theta = 0.5

miniapp = ComparisonMiniapp(ensemble, time_partition,
                            form_mass=form_mass,
                            form_function=form_function,
                            w_initial=Un, dt=dt, theta=theta,
                            boundary_conditions=bcs,
                            serial_sparameters=serial_parameters,
                            parallel_sparameters=parallel_parameters)
pdg = miniapp.paradiag
aaofunc = pdg.aaofunc
is_last_slice = pdg.layout.is_local(-1)

PETSc.Sys.Print("Solving problem")

rank = ensemble.ensemble_comm.rank
norm0 = fd.norm(Un)


def preproc(serial_app, paradiag, wndw):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'### === --- Time window {wndw} --- === ###')
    PETSc.Sys.Print('')
    PETSc.Sys.Print('=== --- Parallel solve --- ===')
    PETSc.Sys.Print('')


def serial_postproc(app, it, t):
    return


def parallel_postproc(pdg, wndw, rhs):
    PETSc.Sys.Print('')
    PETSc.Sys.Print('=== --- Serial solve --- ===')
    PETSc.Sys.Print('')
    return


PETSc.Sys.Print('### === --- Timestepping loop --- === ###')

errors = miniapp.solve(nwindows=args.nwindows,
                       preproc=preproc,
                       serial_postproc=serial_postproc,
                       parallel_postproc=parallel_postproc)

PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Errors --- === ###')

for it, err in enumerate(errors):
    PETSc.Sys.Print(f'Timestep {it} error: {err/norm0}')

if is_last_slice:
    ofile = fd.File("output/compressible3D/mountain.pvd",
                    comm=ensemble.comm)
    Un.assign(aaofunc[-1])
    ofile.write(un, rhon, thetan)
