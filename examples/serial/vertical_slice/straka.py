import firedrake as fd
from firedrake.petsc import PETSc
from utils.diagnostics import convective_cfl_calculator
from utils.serial import SerialMiniApp
from utils.vertical_slice import hydrostatic_rho, pi_formula,\
    get_form_mass, get_form_function, maximum, minimum
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='Straka testcase.')
parser.add_argument('--nlayers', type=int, default=16, help='Number of layers, default 10.')
parser.add_argument('--ncolumns', type=int, default=128, help='Number of columns, default 10.')
parser.add_argument('--tmax', type=float, default=15, help='Final time in minutes. Default 15.')
parser.add_argument('--dumpt', type=float, default=0.5, help='Dump time in minutes. Default 1.')
parser.add_argument('--dt', type=float, default=2, help='Timestep in seconds. Default 1.')
parser.add_argument('--filename', type=str, default='straka')
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space). Default 1.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

PETSc.Sys.Print("Setting up problem")

comm = fd.COMM_WORLD

nlayers = args.nlayers
base_columns = args.ncolumns
dt = args.dt # timestep
L = 51200.
distribution_parameters = {"partition": True, "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)}
m = fd.PeriodicIntervalMesh(base_columns, L, distribution_parameters =
                            distribution_parameters, comm=comm)
#translate the mesh to the left by 51200/25600
m.coordinates.dat.data[:] -= 25600

# build volume mesh
H = 6400.  # Height position of the model top
mesh = fd.ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
n = fd.FacetNormal(mesh)

dT = fd.Constant(1)
g = fd.Constant(9.810616)
N = fd.Constant(0.01)  # Brunt-Vaisala frequency (1/s)
cp = fd.Constant(1004.5)  # SHC of dry air at const. pressure (J/kg/K)
R_d = fd.Constant(287.)  # Gas constant for dry air (J/kg/K)
kappa = fd.Constant(2.0/7.0)  # R_d/c_p
p_0 = fd.Constant(1000.0*100.0)  # reference pressure (Pa, not hPa)
cv = fd.Constant(717.)  # SHC of dry air at const. volume (J/kg/K)
T_0 = fd.Constant(273.15)  # ref. temperature

name = args.filename    
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

W = V1 * V2 * Vt #velocity, density, temperature

PETSc.Sys.Print(f"DoFs: {W.dim()}")
PETSc.Sys.Print(f"DoFs/core: {W.dim()/mesh.comm.size}")

Un = fd.Function(W)
Unp1 = fd.Function(W)

x, z = fd.SpatialCoordinate(mesh)

# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
Tsurf = fd.Constant(300.)
thetab = Tsurf

cp = fd.Constant(1004.5)  # SHC of dry air at const. pressure (J/kg/K)
Up = fd.as_vector([fd.Constant(0.0), fd.Constant(1.0)]) # up direction

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
T_pert =fd.conditional(r > 1., 0., -7.5*(1.+fd.cos(fd.pi*r)))
# T = theta*Pi so Delta theta = Delta T/Pi assuming Pi fixed

Pi_back = pi_formula(rhon, thetan, R_d, p_0, kappa)
# this keeps perturbation at zero away from bubble
thetan.project(theta_back + T_pert/Pi_back)
# save the background stratification for rho
rho_back = fd.Function(V2).assign(rhon)
# Compute the new rho
# using rho*theta = Pi which should be held fixed
rhon.project(rhon*thetan/theta_back)

#The timestepping solver
un, rhon, thetan = fd.split(Un)
unp1, rhonp1, thetanp1 = fd.split(Unp1)

du, drho, dtheta = fd.TestFunctions(W)

eqn = slice_imr_form(un, unp1, rhon, rhonp1, thetan, thetanp1,
                     du, drho, dtheta,
                     dT=dT, n=n, Up=Up, c_pen=fd.Constant(2.0**(-7./2)),
                     cp=cp, g=g, R_d=R_d, p_0=p_0,
                     kappa=kappa, mu=None,
                     viscosity=fd.Constant(75.),
                     diffusivity=fd.Constant(75.))

form_mass = get_form_mass()

bcs = [fd.DirichletBC(W.sub(0), 0., "bottom"),
       fd.DirichletBC(W.sub(0), 0., "top")]

for bc in bcs:
    bc.apply(Un)

sparameters = {
    "snes": {
        "monitor": None,
        "converged_reason": None,
        "stol": 1e-12,
        "atol": 1e-6,
        "rtol": 1e-6,
        "ksp_ew": None,
        "ksp_ew_version": 1,
        "ksp_ew_threshold": 1e-5,
        "ksp_ew_rtol0": 1e-3,
    },
    "ksp_type": "gmres",
    "ksp": {
        "monitor": None,
        "converged_reason": None,
        "atol": 1e-7,
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

# miniapp = SerialMiniApp(dt=dt, theta=theta, w_initial=Un,
#                         form_mass=form_mass,
#                         form_function=form_function,
#                         solver_parameters=solver_parameters,
#                         bcs=bcs)
    
# file_gw = fd.File(f'output/{name}.pvd')
# un, rhon, thetan = Un.subfunctions
# delta_theta = fd.Function(Vt, name="delta theta").assign(thetan-theta_back)
# delta_rho = fd.Function(V2, name="delta rho").assign(rhon-rho_back)
# 
# un.rename("un")
# rhon.rename("rhon")
# thetan.rename("thetan")
# 
# dT.assign(dt)
# 
# DG0 = fd.FunctionSpace(mesh, "DG", 0)
# One = fd.Function(DG0).assign(1.0)
# unn = 0.5*(fd.inner(-un, n) + abs(fd.inner(-un, n))) # gives fluxes *into* cell only
# v = fd.TestFunction(DG0)
# Courant_num = fd.Function(DG0, name="Courant numerator")
# Courant_num_form = dT*(
#     both(unn*v)*(fd.dS_v + fd.dS_h)
#     + unn*v*fd.ds_tb
# )
# Courant_denom = fd.Function(DG0, name="Courant denominator")
# fd.assemble(One*v*fd.dx, tensor=Courant_denom)
# Courant = fd.Function(DG0, name="Courant")
# 
# fd.assemble(Courant_num_form, tensor=Courant_num)
# Courant.interpolate(Courant_num/Courant_denom)
# 
# DG = fd.FunctionSpace(mesh, "DG", 0)
# frontdetector = fd.Function(DG, name="frontdetector")
# 
# front_expr = fd.conditional(thetan < fd.Constant(Tsurf*0.999), x[0], 0)
# 
# frontdetector.interpolate(front_expr)
# 
# file_gw.write(un, rhon, thetan, delta_rho, delta_theta, Courant, frontdetector, t=0)
# Unp1.assign(Un)
# 
# t = 0.
# tmax = args.tmax*60.
# dumpt = args.dumpt*60.
# tdump = 0.
# 
# 
# delta_theta.assign(thetan-theta_back)
# PETSc.Sys.Print("maxes and mins of Delta theta",
#                 maximum(delta_theta),
#                 minimum(delta_theta))
# 
# PETSc.Sys.Print('tmax', tmax, 'dt', dt)
# itcount = 0
# stepcount = 0
# 
# while t < tmax - 0.5*dt:
#     t += dt
#     tdump += dt
# 
#     with timed_stage("Time Solver"):
#         nsolver.solve()
#     Un.assign(Unp1)
#     frontdetector.interpolate(front_expr)
#     delta_theta.assign(thetan-theta_back)
#     PETSc.Sys.Print("")
#     PETSc.Sys.Print("Time: ", t)
#     PETSc.Sys.Print("Front detector", maximum(frontdetector))
#     PETSc.Sys.Print("maxes and mins of Delta theta",
#                     maximum(delta_theta),
#                     minimum(delta_theta))
#     PETSc.Sys.Print("")
#     
#     if tdump > dumpt - dt*0.5:
#         delta_theta.assign(thetan-theta_back)
#         delta_rho.assign(rhon-rho_back)
# 
#         fd.assemble(Courant_num_form, tensor=Courant_num)
#         Courant.interpolate(Courant_num/Courant_denom)
#         file_gw.write(un, rhon, thetan, delta_rho, delta_theta,
#                       Courant, frontdetector, t=t)
#         tdump -= dumpt
#     stepcount += 1
#     itcount += nsolver.snes.getLinearSolveIterations()
# PETSc.Sys.Print("Iterations", itcount, "its per step", itcount/stepcount)
