
import firedrake as fd
from petsc4py import PETSc
import asQ
import numpy as np

from utils import mg
from utils import units
from utils.planets import earth
import utils.shallow_water.nonlinear as swe
import utils.shallow_water.williamson1992.case2 as case2

PETSc.Sys.popErrorHandler()
# get command arguments
import argparse
parser = argparse.ArgumentParser(description='Williamson 5 testcase for approximate Schur complement solver.')
parser.add_argument('--base_level', type=int, default=1, help='Base refinement level of icosahedral grid for MG solve. Default 1.')
parser.add_argument('--ref_level', type=int, default=2, help='Refinement level of icosahedral grid. Default 3.')
# parser.add_argument('--nsteps', type=int, default=10, help='Number of timesteps. Default 10.')
parser.add_argument('--alpha', type=float, default=0.0001, help='Circulant coefficient. Default 0.0001.')
parser.add_argument('--dt', type=float, default=0.05, help='Timestep in hours. Default 0.05.')
parser.add_argument('--filename', type=str, default='w5diag')
parser.add_argument('--coords_degree', type=int, default=1, help='Degree of polynomials for sphere mesh approximation.')
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space).')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)


M = [2]
nspatial_domains = 2

# set up the ensemble communicator for space-time parallelism

# some domain, parameters and FS setup
distribution_parameters = {"partition": True, "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)}

ensemble = fd.Ensemble(fd.COMM_WORLD, nspatial_domains)

# mesh set up
mesh = mg.icosahedral_mesh(R0=earth.radius,
                           base_level=args.base_level,
                           degree=args.coords_degree,
                           distribution_parameters=distribution_parameters,
                           nrefs=args.ref_level-args.base_level,
                           comm=ensemble.comm)

x = fd.SpatialCoordinate(mesh)

degree = 1
V1 = fd.FunctionSpace(mesh, "BDM", degree+1)
V2 = fd.FunctionSpace(mesh, "DG", degree)
W = fd.MixedFunctionSpace((V1, V2))

# initial conditions
f = case2.coriolis_expression(*x)

g = earth.Gravity
H = case2.H0
b = fd.Constant(0)

# W = V1 * V2
w0 = fd.Function(W)
un, hn = w0.split()
un.project(case2.velocity_expression(*x))
hn.project(H - b + case2.elevation_expression(*x))

# finite element forms


def form_function(u, h, v, q):
    return swe.form_function(mesh, g, b, f, h, u, q, v)


def form_mass(u, h, v, q):
    return swe.form_mass(mesh, h, u, q, v)


# Parameters for the diag block solve
sparameters = {
    # "snes_monitor": None,
    "mat_type": "matfree",
    "ksp_type": "preonly",
    # "ksp_monitor": None,
    # "ksp_monitor_true_residual": None,
    # "ksp_converged_reason": None,
    "ksp_atol": 1e-8,
    "ksp_rtol": 1e-8,
    "ksp_max_it": 400,
    "pc_type": "mg",
    "pc_mg_cycle_type": "v",
    "pc_mg_type": "multiplicative",
    "mg_levels_ksp_type": "gmres",
    "mg_levels_ksp_max_it": 5,
    # "mg_levels_ksp_convergence_test": "skip",
    "mg_levels_pc_type": "python",
    "mg_levels_pc_python_type": "firedrake.PatchPC",
    "mg_levels_patch_pc_patch_save_operators": True,
    "mg_levels_patch_pc_patch_partition_of_unity": True,
    "mg_levels_patch_pc_patch_sub_mat_type": "seqdense",
    "mg_levels_patch_pc_patch_construct_codim": 0,
    "mg_levels_patch_pc_patch_construct_type": "vanka",
    "mg_levels_patch_pc_patch_local_type": "additive",
    "mg_levels_patch_pc_patch_precompute_element_tensors": True,
    "mg_levels_patch_pc_patch_symmetrise_sweep": False,
    "mg_levels_patch_sub_ksp_type": "preonly",
    "mg_levels_patch_sub_pc_type": "lu",
    "mg_levels_patch_sub_pc_factor_shift_type": "nonzero",
    "mg_coarse_pc_type": "python",
    "mg_coarse_pc_python_type": "firedrake.AssembledPC",
    "mg_coarse_assembled_pc_type": "lu",
    "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
}

solver_parameters_diag = {
    "snes_linesearch_type": "basic",
    'snes_atol': 1e3,
    # 'snes_monitor': None,
    # 'snes_converged_reason': None,
    'ksp_rtol': 1e-3,
    # 'ksp_monitor': None,
    # 'ksp_converged_reason': None,
    'mat_type': 'matfree',
    'ksp_type': 'gmres',
    'pc_type': 'python',
    'pc_python_type': 'asQ.DiagFFTPC'}

M = [2, 2]
for i in range(np.sum(M)):
    solver_parameters_diag["diagfft_"+str(i)+"_"] = sparameters

# non-petsc information for block solve
block_ctx = {}

# mesh transfer operators
transfer_managers = []
for _ in range(sum(M)):
    tm = mg.manifold_transfer_manager(W)
    transfer_managers.append(tm)

block_ctx['diag_transfer_managers'] = transfer_managers

dt = 0.2*units.hour

alpha = 1.0e-3
theta = 0.5

PD = asQ.paradiag(ensemble=ensemble,
                  form_function=form_function,
                  form_mass=form_mass, W=W, w0=w0,
                  dt=dt, theta=theta,
                  alpha=alpha,
                  M=M, solver_parameters=solver_parameters_diag,
                  circ=None, tol=1.0e-6, maxits=None,
                  ctx={}, block_ctx=block_ctx, block_mat_type="aij")
PD.solve()

# check against initial conditions
walls = PD.w_all.split()
hn.assign(hn-H+b)

hmag = fd.sqrt(fd.assemble(hn*hn*fd.dx))
umag = fd.sqrt(fd.assemble(fd.inner(un, un)*fd.dx))

for step in range(M[PD.rT]):

    up = walls[2*step]
    hp = walls[2*step+1]
    hp.assign(hp-H+b)

    dh = hp - hn
    du = up - un

    herr = fd.sqrt(fd.assemble(dh*dh*fd.dx))
    uerr = fd.sqrt(fd.assemble(fd.inner(du, du)*fd.dx))

    herr /= hmag
    uerr /= umag

    timestep = sum(M[:PD.rT]) + step
    PETSc.Sys.Print(f"timestep={timestep}, herr={herr}, uerr={uerr}")
