import firedrake as fd
from firedrake.petsc import PETSc
from firedrake.output import VTKFile
from pyop2.mpi import MPI
from utils.diagnostics import convective_cfl_calculator
from utils import compressible_flow as euler
from math import sqrt
import numpy as np
import asQ


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
parser.add_argument('--dt', type=float, default=12, help='Timestep in seconds.')
parser.add_argument('--atol', type=float, default=1e-4, help='Average absolute tolerance for each timestep')
parser.add_argument('--nwindows', type=int, default=1, help='Number of windows to solve.')
parser.add_argument('--nslices', type=int, default=2, help='Number of slices in the all-at-once system.')
parser.add_argument('--slice_length', type=int, default=2, help='Number of timesteps in each slice of the all-at-once system.')
parser.add_argument('--alpha', type=float, default=1e-4, help='Circulant parameter')
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space).')
parser.add_argument('--filename', type=str, default='gravity_wave_nh', help='Name of vtk file.')
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
    comm=ensemble.comm)
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

PETSc.Sys.Print(f"DoFs/timestep: {W.dim()}")
PETSc.Sys.Print(f"DoFs/core: {W.dim()*window_length/global_comm.size}")
PETSc.Sys.Print(f"Block DoFs/core: {2*W.dim()/ensemble.comm.size}")
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

# Parameters for the blocks
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
    'pc_python_type': 'asQ.AuxiliaryComplexBlockPC',
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

block_parameters = {
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp': {
        'rtol': args.alpha,
        'max_it': 50,
        'converged_maxits': None,
        'convergence_test': 'skip',
    },
}

block_parameters.update(composite_parameters)

patol = sqrt(window_length)*args.atol
solver_parameters_diag = {
    "snes": {
        "monitor": None,
        "converged_reason": None,
        "rtol": 1e-8,
        "atol": patol,
        "ksp_ew": None,
        "ksp_ew_version": 1,
        "ksp_ew_rtol0": 1e-1,
        "ksp_ew_threshold": 1e-1,
        'lag_preconditioner': -2,
        'lag_preconditioner_persists': None,
    },
    "mat_type": "matfree",
    "ksp_type": "fgmres",
    "ksp": {
        "monitor": None,
        "converged_rate": None,
        "atol": patol,
    },
    "pc_type": "python",
    "pc_python_type": "asQ.CirculantPC",
    "circulant_state": "reference",
    "circulant_alpha": args.alpha,
    "circulant_block": block_parameters
}

appctx = {'block_appctx': {'aux_uref': U0}}

theta = 0.5

pdg = asQ.Paradiag(ensemble=ensemble,
                   time_partition=time_partition,
                   form_function=form_function,
                   form_mass=form_mass,
                   ics=Un, dt=dt, theta=theta,
                   reference_state=Uback,
                   bcs=bcs, appctx=appctx,
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

        ofile = VTKFile(f'output/{args.filename}.pvd',
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
    ensemble.global_comm.Barrier()
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
