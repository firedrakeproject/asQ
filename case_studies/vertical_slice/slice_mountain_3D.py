import firedrake as fd
from math import pi, sqrt
from pyop2.mpi import MPI
from utils.diagnostics import convective_cfl_calculator
from utils.vertical_slice import hydrostatic_rho, \
    get_form_mass, get_form_function, maximum
from firedrake.petsc import PETSc
import asQ

import argparse
parser = argparse.ArgumentParser(description='3D mountain testcase.')
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
parser.add_argument('--filename', type=str, default='slice_mountain3D', help='Name of vtk file.')
parser.add_argument('--write_file', action='store_true', help='Write vtk file at end of each window.')
parser.add_argument('--metrics_dir', type=str, default='output', help='Directory to save paradiag metrics and vtk to.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

PETSc.Sys.Print("Setting up problem")

# set up the ensemble communicator for space-time parallelism
time_partition = tuple((args.slice_length for _ in range(args.nslices)))
window_length = sum(time_partition)

global_comm = fd.COMM_WORLD

ensemble = asQ.create_ensemble(time_partition, comm=global_comm)

comm = ensemble.comm

# set up the mesh

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
                                     comm=ensemble.comm)

# volume mesh of the slice
mesh = fd.ExtrudedMesh(base_mesh, layers=nz, layer_height=Lz/nz)
n = fd.FacetNormal(mesh)

g = fd.Constant(9.810616)
N = fd.Constant(0.01)  # Brunt-Vaisala frequency (1/s)
cp = fd.Constant(1004.5)  # SHC of dry air at const. pressure (J/kg/K)
R_d = fd.Constant(287.)  # Gas constant for dry air (J/kg/K)
kappa = fd.Constant(2.0/7.0)  # R_d/c_p
p_0 = fd.Constant(1000.0*100.0)  # reference pressure (Pa, not hPa)
cv = fd.Constant(717.)  # SHC of dry air at const. volume (J/kg/K)
T_0 = fd.Constant(273.15)  # ref. temperature

dt = args.dt

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

horizontal_degree = args.degree
vertical_degree = args.degree

S1 = fd.FiniteElement("RTCF", fd.quadrilateral, horizontal_degree+1)
S2 = fd.FiniteElement("DQ", fd.quadrilateral, horizontal_degree)

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
PETSc.Sys.Print(f"DoFs/core: {W.dim()/ensemble.comm.size}")

Un = fd.Function(W)

# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
Tsurf = fd.Constant(300.)
thetab = Tsurf*fd.exp(N**2*z/g)

cp = fd.Constant(1004.5)  # SHC of dry air at const. pressure (J/kg/K)
Up = fd.as_vector([fd.Constant(0.0), fd.Constant(0.0), fd.Constant(1.0)])  # up direction

un = Un.subfunctions[0]
rhon = Un.subfunctions[1]
thetan = Un.subfunctions[2]
un.project(fd.as_vector([10.0, 0.0, 0.0]))
thetan.interpolate(thetab)
theta_back = fd.Function(Vt).assign(thetan)
rhon.assign(1.0e-5)

PETSc.Sys.Print("Calculating hydrostatic state")

Pi = fd.Function(V2)

hydrostatic_rho(Vv, V2, mesh, thetan, rhon, pi_boundary=fd.Constant(0.02),
                cp=cp, R_d=R_d, p_0=p_0, kappa=kappa, g=g, Up=Up,
                top=True, Pi=Pi)
p0 = maximum(Pi)

hydrostatic_rho(Vv, V2, mesh, thetan, rhon, pi_boundary=fd.Constant(0.05),
                cp=cp, R_d=R_d, p_0=p_0, kappa=kappa, g=g, Up=Up,
                top=True, Pi=Pi)
p1 = maximum(Pi)
alpha = 2.*(p1-p0)
beta = p1-alpha
pi_top = (1.-beta)/alpha

hydrostatic_rho(Vv, V2, mesh, thetan, rhon, pi_boundary=fd.Constant(pi_top),
                cp=cp, R_d=R_d, p_0=p_0, kappa=kappa, g=g, Up=Up,
                top=True)

rho_back = fd.Function(V2).assign(rhon)

zc = fd.Constant(Lz-10000.)
mubar = fd.Constant(0.15/dt)
mu_top = fd.conditional(z <= zc, 0.0,
                        mubar*fd.sin(fd.Constant(pi/2.)*(z-zc)/(Lz-zc))**2)
mu = fd.Function(V2).interpolate(mu_top)

form_function = get_form_function(n, Up, c_pen=2.0**(-7./2),
                                  cp=cp, g=g, R_d=R_d,
                                  p_0=p_0, kappa=kappa, mu=mu)

form_mass = get_form_mass()

zv = fd.as_vector([fd.Constant(0.), fd.Constant(0.), fd.Constant(0.)])
bcs = [fd.DirichletBC(W.sub(0), zv, "bottom"),
       fd.DirichletBC(W.sub(0), zv, "top")]

for bc in bcs:
    bc.apply(Un)

# Parameters for the diag

block_parameters = {
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
                "pc_factor_mat_solver_type": 'mumps',
            }
        },
    },
}

atol = sqrt(window_length)*args.atol
solver_parameters_diag = {
    "snes": {
        "monitor": None,
        "converged_reason": None,
        "rtol": 1e-8,
        "atol": atol,
        "ksp_ew": None,
        "ksp_ew_version": 1,
        "ksp_ew_threshold": 1e-5,
        "ksp_ew_rtol0": 1e-3
    },
    "ksp_type": "fgmres",
    "mat_type": "matfree",
    "ksp": {
        "monitor": None,
        "converged_reason": None,
        "atol": atol,
    },
    "pc_type": "python",
    "pc_python_type": "asQ.DiagFFTPC",
    "diagfft_alpha": args.alpha,
}

for i in range(sum(time_partition)):
    solver_parameters_diag["diagfft_block_"+str(i)+"_"] = block_parameters

theta = 0.5

pdg = asQ.Paradiag(ensemble=ensemble,
                   time_partition=time_partition,
                   form_mass=form_mass,
                   form_function=form_function,
                   ics=Un, dt=dt, theta=theta, bcs=bcs,
                   solver_parameters=solver_parameters_diag)

aaofunc = pdg.aaofunc
is_last_slice = pdg.layout.is_local(-1)

PETSc.Sys.Print("Solving problem")

# only last slice does diagnostics/output
if is_last_slice:
    uout = fd.Function(V1, name='velocity')
    if args.write_file:
        thetaout = fd.Function(Vt, name='temperature')
        rhoout = fd.Function(V2, name='density')

        ofile = fd.File(f'output/{args.filename}.pvd',
                        comm=ensemble.comm)

    def assign_out_functions():
        aaofunc.get_component(-1, 0, uout=uout)
        if args.write_file:
            aaofunc.get_component(-1, 1, uout=rhoout)
            aaofunc.get_component(-1, 2, uout=thetaout)

            rhoout.assign(rhoout - rho_back)
            thetaout.assign(thetaout - theta_back)

    def write_to_file():
        if args.write_file:
            ofile.write(uout, rhoout, thetaout)

    cfl_calc = convective_cfl_calculator(mesh)
    cfl_series = []

    def max_cfl(u, dt):
        with cfl_calc(u, dt).dat.vec_ro as v:
            return v.max()[1]

solver_time = []


def window_preproc(pdg, wndw, rhs):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'### === --- Calculating time-window {wndw} --- === ###')
    PETSc.Sys.Print('')
    stime = MPI.Wtime()
    solver_time.append(stime)


def window_postproc(pdg, wndw, rhs):
    # postprocess this timeslice
    etime = MPI.Wtime()
    stime = solver_time[-1]
    duration = etime - stime
    solver_time[-1] = duration
    PETSc.Sys.Print('', comm=global_comm)
    PETSc.Sys.Print(f'Window solution time: {duration}', comm=global_comm)
    PETSc.Sys.Print('', comm=global_comm)

    if is_last_slice:
        assign_out_functions()
        if args.write_file:
            write_to_file()
        PETSc.Sys.Print('', comm=ensemble.comm)

        cfl = max_cfl(uout, dt)
        cfl_series.append(cfl)
        PETSc.Sys.Print(f'Maximum CFL = {cfl}', comm=ensemble.comm)


# solve for each window
pdg.solve(nwindows=args.nwindows,
          preproc=window_preproc,
          postproc=window_postproc)

PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Iteration counts --- === ###')
PETSc.Sys.Print('')

from asQ import write_paradiag_metrics
write_paradiag_metrics(pdg, directory=args.metrics_dir)

nw = pdg.total_windows
nt = pdg.total_timesteps
PETSc.Sys.Print(f'windows: {nw}')
PETSc.Sys.Print(f'timesteps: {nt}')
PETSc.Sys.Print('')

lits = pdg.linear_iterations
nlits = pdg.nonlinear_iterations
blits = pdg.block_iterations.data()

PETSc.Sys.Print(f'linear iterations: {lits} | iterations per window: {lits/nw}')
PETSc.Sys.Print(f'nonlinear iterations: {nlits} | iterations per window: {nlits/nw}')
PETSc.Sys.Print(f'block linear iterations: {blits} | iterations per block solve: {blits/lits}')
PETSc.Sys.Print('')

ensemble.global_comm.Barrier()
if is_last_slice:
    PETSc.Sys.Print(f'Maximum CFL = {max(cfl_series)}', comm=ensemble.comm)
    PETSc.Sys.Print(f'Minimum CFL = {min(cfl_series)}', comm=ensemble.comm)
    PETSc.Sys.Print('', comm=ensemble.comm)
ensemble.global_comm.Barrier()

PETSc.Sys.Print(f'DoFs per timestep: {W.dim()}', comm=global_comm)
PETSc.Sys.Print(f'Number of MPI ranks per timestep: {mesh.comm.size} ', comm=global_comm)
PETSc.Sys.Print(f'DoFs/rank: {W.dim()/mesh.comm.size}', comm=global_comm)
PETSc.Sys.Print(f'Block DoFs/rank: {2*W.dim()/mesh.comm.size}', comm=global_comm)
PETSc.Sys.Print('')

if len(solver_time) > 1:
    solver_time[0] = solver_time[1]

PETSc.Sys.Print(f'Total solution time: {sum(solver_time)}', comm=global_comm)
PETSc.Sys.Print(f'Average window solution time: {sum(solver_time)/len(solver_time)}', comm=global_comm)
PETSc.Sys.Print(f'Average timestep solution time: {sum(solver_time)/(window_length*len(solver_time))}', comm=global_comm)
PETSc.Sys.Print('', comm=global_comm)
