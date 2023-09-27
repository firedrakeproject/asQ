import firedrake as fd
from math import sqrt
from utils.serial import ComparisonMiniapp
from utils.vertical_slice import hydrostatic_rho, pi_formula, \
    get_form_mass, get_form_function
from firedrake.petsc import PETSc
import asQ

import argparse
parser = argparse.ArgumentParser(description='Straka testcase.')
parser.add_argument('--nlayers', type=int, default=16, help='Number of layers, default 10.')
parser.add_argument('--ncolumns', type=int, default=128, help='Number of columns, default 10.')
parser.add_argument('--nwindows', type=int, default=1, help='Number of ParaDiag windows.')
parser.add_argument('--output_freq', type=int, default=1, help='Output frequency in timesteps.')
parser.add_argument('--dt', type=float, default=2, help='Timestep in seconds. Default 1.')
parser.add_argument('--filename', type=str, default='straka')
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space). Default 1.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

PETSc.Sys.Print("Setting up problem")

time_partition = tuple((2 for _ in range(2)))

ensemble = asQ.create_ensemble(time_partition, comm=fd.COMM_WORLD)

comm = ensemble.comm

# set up the mesh

dt = args.dt

nlayers = args.nlayers  # horizontal layers
base_columns = args.ncolumns  # number of columns
L = 512e3
H = 6.4e3  # Height position of the model top

distribution_parameters = {
    "partition": True,
    "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)
}

# surface mesh of ground
base_mesh = fd.PeriodicIntervalMesh(base_columns, L,
                                    distribution_parameters=distribution_parameters,
                                    comm=comm)
base_mesh.coordinates.dat.data[:] -= 25600

# volume mesh of the slice
mesh = fd.ExtrudedMesh(base_mesh,
                       layers=nlayers,
                       layer_height=H/nlayers)
n = fd.FacetNormal(mesh)
x, z = fd.SpatialCoordinate(mesh)

g = fd.Constant(9.810616)
N = fd.Constant(0.01)  # Brunt-Vaisala frequency (1/s)
cp = fd.Constant(1004.5)  # SHC of dry air at const. pressure (J/kg/K)
R_d = fd.Constant(287.)  # Gas constant for dry air (J/kg/K)
kappa = fd.Constant(2.0/7.0)  # R_d/c_p
p_0 = fd.Constant(1000.0*100.0)  # reference pressure (Pa, not hPa)
cv = fd.Constant(717.)  # SHC of dry air at const. volume (J/kg/K)
T_0 = fd.Constant(273.15)  # ref. temperature

horizontal_degree = args.degree
vertical_degree = args.degree

S1 = fd.FiniteElement("CG", fd.interval, horizontal_degree+1)
S2 = fd.FiniteElement("DG", fd.interval, horizontal_degree)

# vertical base spaces
T0 = fd.FiniteElement("CG", fd.interval, vertical_degree+1)
T1 = fd.FiniteElement("DG", fd.interval, vertical_degree)

# build spaces V2, V3, Vt
V2h_elt = fd.HDiv(fd.TensorProductElement(S1, T1))
V2t_elt = fd.TensorProductElement(S2, T0)
V3_elt = fd.TensorProductElement(S2, T1)
V2v_elt = fd.HDiv(V2t_elt)
V2_elt = V2h_elt + V2v_elt

V1 = fd.FunctionSpace(mesh, V2_elt, name="Velocity")
V2 = fd.FunctionSpace(mesh, V3_elt, name="Pressure")
Vt = fd.FunctionSpace(mesh, V2t_elt, name="Temperature")
Vv = fd.FunctionSpace(mesh, V2v_elt, name="Vv")

W = V1 * V2 * Vt  # velocity, density, temperature

PETSc.Sys.Print(f"DoFs: {W.dim()}")
PETSc.Sys.Print(f"DoFs/core: {W.dim()/comm.size}")

Un = fd.Function(W)

# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
Tsurf = fd.Constant(300.)

thetab = Tsurf

cp = fd.Constant(1004.5)  # SHC of dry air at const. pressure (J/kg/K)
Up = fd.as_vector([fd.Constant(0.0), fd.Constant(1.0)])  # up direction

un, rhon, thetan = Un.subfunctions
thetan.interpolate(thetab)
theta_back = fd.Function(Vt).assign(thetan)
rhon.assign(1.0e-5)

hydrostatic_rho(Vv, V2, mesh, thetan, rhon, pi_boundary=fd.Constant(1.0),
                cp=cp, R_d=R_d, p_0=p_0, kappa=kappa, g=g, Up=Up,
                top=False)

x = fd.SpatialCoordinate(mesh)
xc = 0.
xr = 4000.
zc = 3000.
zr = 2000.
r = fd.sqrt(((x[0]-xc)/xr)**2 + ((x[1]-zc)/zr)**2)
T_pert = fd.conditional(r > 1., 0., -7.5*(1.+fd.cos(fd.pi*r)))
# T = theta*Pi so Delta theta = Delta T/Pi assuming Pi fixed

Pi_back = pi_formula(rhon, thetan, R_d, p_0, kappa)
# this keeps perturbation at zero away from bubble
thetan.project(theta_back + T_pert/Pi_back)
# save the background stratification for rho
rho_back = fd.Function(V2).assign(rhon)
# Compute the new rho
# using rho*theta = Pi which should be held fixed
rhon.project(rhon*thetan/theta_back)

# The timestepping forms

viscosity = fd.Constant(75.)

form_mass = get_form_mass()

form_function = get_form_function(n=n, Up=Up, c_pen=fd.Constant(2.0**(-7./2)),
                                  cp=cp, g=g, R_d=R_d, p_0=p_0, kappa=kappa, mu=None,
                                  viscosity=viscosity, diffusivity=viscosity)

zv = fd.as_vector([fd.Constant(0.), fd.Constant(0.)])
bcs = [fd.DirichletBC(W.sub(0), zv, "bottom"),
       fd.DirichletBC(W.sub(0), zv, "top")]

for bc in bcs:
    bc.apply(Un)

# Parameters for the newton iterations
atol = 1e4
stol = 1e-100

lines_parameters = {
    "ksp_type": "fgmres",
    "ksp_rtol": 1e-4,
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

serial_parameters = {
    "snes": {
        "atol": atol,
        "stol": stol,
        "rtol": 1e-8,
        "ksp_ew": None,
        "ksp_ew_version": 1,
        "ksp_ew_threshold": 1e-10,
        "ksp_ew_rtol0": 1e-1,
    },
    "ksp": {
        "atol": atol,
        "stol": stol,
    },
}
serial_parameters.update(lines_parameters)

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
        "stol": stol,
        "rtol": 1e-8,
        "ksp_ew": None,
        "ksp_ew_version": 1,
        "ksp_ew_threshold": 1e-10,
        "ksp_ew_rtol0": 1e-1,
    },
    "ksp_type": "fgmres",
    "mat_type": "matfree",
    "ksp": {
        "monitor": None,
        "converged_reason": None,
        "atol": atol,
    },
    "pc_type": "python",
    "pc_python_type": "asQ.DiagFFTPC"
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
