import firedrake as fd
import asQ
from pyop2.mpi import MPI
from math import sqrt
from utils.diagnostics import convective_cfl_calculator
from utils.vertical_slice import hydrostatic_rho, pi_formula, \
    get_form_mass, get_form_function
from firedrake.petsc import PETSc

import argparse
parser = argparse.ArgumentParser(description='Straka testcase.')
parser.add_argument('--nlayers', type=int, default=16, help='Number of layers, default 10.')
parser.add_argument('--ncolumns', type=int, default=128, help='Number of columns, default 10.')
parser.add_argument('--nwindows', type=int, default=1, help='Number of windows to solve.')
parser.add_argument('--nslices', type=int, default=2, help='Number of slices in the all-at-once system.')
parser.add_argument('--slice_length', type=int, default=2, help='Number of timesteps in each slice of the all-at-once system.')
parser.add_argument('--dt', type=float, default=2, help='Timestep in seconds. Default 1.')
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space). Default 1.')
parser.add_argument('--filename', type=str, default='straka', help='Name of vtk file.')
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

# set up the mesh

dt = args.dt

nlayers = args.nlayers  # horizontal layers
base_columns = args.ncolumns  # number of columns
L = 51.2e3
H = 6.4e3  # Height position of the model top

distribution_parameters = {
    "partition": True,
    "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)
}

# surface mesh of ground
base_mesh = fd.PeriodicIntervalMesh(base_columns, L,
                                    distribution_parameters=distribution_parameters,
                                    comm=ensemble.comm)
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
cp = fd.Constant(1004.5)  # SHC of dry air at const. pressure (J/kg/K)

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
PETSc.Sys.Print(f"DoFs/core: {W.dim()/mesh.comm.size}")

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
            "sub_sub_pc_type": "lu",
            "sub_sub_pc_factor_mat_ordering_type": "rcm",
            "sub_sub_pc_factor_reuse_ordering": None,
            "sub_sub_pc_factor_reuse_fill": None,
        },
    },
}

atol = 1e4
rtol = 1e-10
stol = 1e-100
patol = sqrt(sum(time_partition))*atol

solver_parameters_diag = {
    "snes": {
        "monitor": None,
        "converged_reason": None,
        "atol": patol,
        "stol": stol,
        "rtol": rtol,
        "ksp_ew": None,
        "ksp_ew_version": 1,
        "ksp_ew_threshold": 1e-10,
        "ksp_ew_rtol0": 1e-3,
    },
    "ksp_type": "fgmres",
    "mat_type": "matfree",
    "ksp": {
        "monitor": None,
        "converged_reason": None,
        "atol": patol,
    },
    "pc_type": "python",
    "pc_python_type": "asQ.DiagFFTPC",
    "diagfft_alpha": 1e-3,
}

for i in range(sum(time_partition)):
    solver_parameters_diag["diagfft_block_"+str(i)+"_"] = lines_parameters

theta = 0.5

pdg = asQ.Paradiag(ensemble=ensemble,
                   form_mass=form_mass,
                   form_function=form_function,
                   ics=Un, dt=dt, theta=theta,
                   time_partition=time_partition, bcs=bcs,
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

        ofile = fd.File(f'{args.metrics_dir}/vtk/{args.filename}.pvd',
                        comm=ensemble.comm)

        def write_to_file(time):
            ofile.write(uout, rhoout, thetaout, t=time)

    def assign_out_functions():
        uout.assign(aaofunc[-1].subfunctions[0])
        if args.write_file:
            rhoout.assign(aaofunc[-1].subfunctions[1])
            thetaout.assign(aaofunc[-1].subfunctions[2])

            rhoout.assign(rhoout - rho_back)
            thetaout.assign(thetaout - theta_back)

    if args.write_file:
        assign_out_functions()
        write_to_file(time=0)

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
    etime = MPI.Wtime()
    stime = solver_time[-1]
    duration = etime - stime
    solver_time[-1] = duration
    PETSc.Sys.Print('', comm=global_comm)
    PETSc.Sys.Print(f'Window solution time: {duration}', comm=global_comm)
    PETSc.Sys.Print('', comm=global_comm)

    # postprocess this timeslice
    if is_last_slice:
        assign_out_functions()

        if args.write_file:
            write_to_file(time=pdg.aaoform.time[-1])

        cfl = max_cfl(uout, dt)
        cfl_series.append(cfl)

        PETSc.Sys.Print('', comm=ensemble.comm)
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
