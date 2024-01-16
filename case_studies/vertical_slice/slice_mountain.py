import firedrake as fd
import asQ
from math import sqrt
from pyop2.mpi import MPI
from utils.diagnostics import convective_cfl_calculator
from utils import compressible_flow as euler
from utils.vertical_slice import mount_agnesi as mountain
from firedrake.petsc import PETSc

import argparse
parser = argparse.ArgumentParser(description='Mountain testcase.')
parser.add_argument('--nlayers', type=int, default=35, help='Number of layers, default 10.')
parser.add_argument('--ncolumns', type=int, default=90, help='Number of columns, default 10.')
parser.add_argument('--nwindows', type=int, default=1, help='Number of windows to solve.')
parser.add_argument('--nslices', type=int, default=2, help='Number of slices in the all-at-once system.')
parser.add_argument('--slice_length', type=int, default=2, help='Number of timesteps in each slice of the all-at-once system.')
parser.add_argument('--dt', type=float, default=5, help='Timestep in seconds. Default 1.')
parser.add_argument('--atol', type=float, default=1e-3, help='Average absolute tolerance for each timestep')
parser.add_argument('--alpha', type=float, default=1e-3, help='Circulant parameter')
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space). Default 1.')
parser.add_argument('--filename', type=str, default='slice_mountain', help='Name of vtk file.')
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
mesh = mountain.mesh(ensemble.comm,
                     ncolumns=args.ncolumns,
                     nlayers=args.nlayers,
                     hydrostatic=False)
n = fd.FacetNormal(mesh)

dt = args.dt

gas = euler.StandardAtmosphere(N=0.01)

W, Vv = euler.function_space(mesh, horizontal_degree=args.degree,
                             vertical_degree=args.degree,
                             vertical_velocity_space=True)
V1, V2, Vt = W.subfunctions  # velocity, density, temperature

PETSc.Sys.Print(f"DoFs: {W.dim()}")
PETSc.Sys.Print(f"DoFs/core: {W.dim()/ensemble.comm.size}")

PETSc.Sys.Print("Calculating hydrostatic state")

Un = mountain.initial_conditions(mesh, W, Vv, gas)
rho_back = fd.Function(V2).assign(Un.subfunctions[1])
theta_back = fd.Function(Vt).assign(Un.subfunctions[2])

mu = mountain.sponge_layer(mesh, V2, dt)

form_mass = euler.get_form_mass()

up = fd.as_vector([fd.Constant(0.0), fd.Constant(1.0)])  # up direction
form_function = euler.get_form_function(
    n, up, c_pen=fd.Constant(2.0**(-7./2)),
    gas=gas, mu=mu)

bcs = mountain.boundary_conditions(W)
for bc in bcs:
    bc.apply(Un)

# Parameters for the diag
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
            "sub_sub_pc_factor_mat_solver_type": 'mumps',
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
        "ksp_ew_rtol0": 1e-3,
    },
    "ksp_type": "fgmres",
    "mat_type": "matfree",
    "ksp": {
        "monitor": None,
        "converged_reason": None,
        "atol": atol,
    },
    "pc_type": "python",
    "pc_python_type": "asQ.CirculantPC",
    "diagfft_alpha": args.alpha,
}

for i in range(sum(time_partition)):
    solver_parameters_diag["diagfft_block_"+str(i)+"_"] = lines_parameters

theta = 0.5

pdg = asQ.Paradiag(ensemble=ensemble,
                   time_partition=time_partition,
                   form_function=form_function,
                   form_mass=form_mass,
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
        uout.assign(aaofunc[-1].subfunctions[0])
        if args.write_file:
            rhoout.assign(aaofunc[-1].subfunctions[1])
            thetaout.assign(aaofunc[-1].subfunctions[2])

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
write_paradiag_metrics(pdg, directory=f'{args.metrics_dir}')

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
