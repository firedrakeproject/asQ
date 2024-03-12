import firedrake as fd
from math import sqrt
from utils.serial import ComparisonMiniapp
from utils.vertical_slice import straka
from utils import compressible_flow as euler
from firedrake.petsc import PETSc
import asQ

import argparse
parser = argparse.ArgumentParser(description='Straka testcase.')
parser.add_argument('--nlayers', type=int, default=16, help='Number of layers, default 10.')
parser.add_argument('--ncolumns', type=int, default=128, help='Number of columns, default 10.')
parser.add_argument('--nwindows', type=int, default=1, help='Number of windows to solve.')
parser.add_argument('--nslices', type=int, default=2, help='Number of slices in the all-at-once system.')
parser.add_argument('--slice_length', type=int, default=2, help='Number of timesteps in each slice of the all-at-once system.')
parser.add_argument('--dt', type=float, default=2, help='Timestep in seconds. Default 1.')
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space). Default 1.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

PETSc.Sys.Print("Setting up problem")

time_partition = tuple((args.slice_length for _ in range(args.nslices)))

ensemble = asQ.create_ensemble(time_partition, comm=fd.COMM_WORLD)

comm = ensemble.comm

# set up the mesh
mesh = straka.mesh(ensemble.comm,
                   ncolumns=args.ncolumns,
                   nlayers=args.nlayers)
n = fd.FacetNormal(mesh)

dt = args.dt

gas = euler.StandardAtmosphere(N=0.01)

W, Vv = euler.function_space(mesh, horizontal_degree=args.degree,
                             vertical_degree=args.degree,
                             vertical_velocity_space=True)
V1, V2, Vt = W.subfunctions  # velocity, density, temperature

PETSc.Sys.Print(f"DoFs: {W.dim()}")
PETSc.Sys.Print(f"DoFs/core: {W.dim()/comm.size}")

Un, rho_back, theta_back = straka.initial_conditions(mesh, W, Vv, gas)

# The timestepping forms

viscosity = fd.Constant(75.)

form_mass = euler.get_form_mass()

up = fd.as_vector([fd.Constant(0.0), fd.Constant(1.0)])  # up direction
form_function = euler.get_form_function(
    n=n, Up=up, c_pen=fd.Constant(2.0**(-7./2)),
    gas=gas, mu=None, viscosity=viscosity, diffusivity=viscosity)

bcs = straka.boundary_conditions(W)
for bc in bcs:
    bc.apply(Un)

# Parameters for the newton iterations
atol = 1e2
rtol = 1e-10
stol = 1e-100

lines_parameters = {
    "ksp_type": "gmres",
    "ksp_rtol": 1e-5,
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

serial_parameters = {
    "snes": {
        "atol": atol,
        "rtol": rtol,
        "stol": stol,
        "ksp_ew": None,
        "ksp_ew_version": 1,
        "ksp_ew_threshold": 1e-10,
        "ksp_ew_rtol0": 1e-3,
    },
    "ksp": {
        "atol": atol,
        "stol": stol,
    },
}
serial_parameters.update(lines_parameters)
serial_parameters['ksp_type'] = 'fgmres'

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
        "stol": stol,
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
        "rtol": 1e-5,
        "stol": stol,
    },
    "pc_type": "python",
    "pc_python_type": "asQ.CirculantPC",
    "diagfft_alpha": 1e-4,
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
