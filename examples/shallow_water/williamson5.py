import firedrake as fd
from petsc4py import PETSc
import asQ

from utils import mg
from utils import units
from utils.planets import earth
import utils.shallow_water.nonlinear as swe
from utils.shallow_water.williamson1992 import case5

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
nspatial_domains = 1

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

degree = args.degree
V1 = fd.FunctionSpace(mesh, "BDM", degree+1)
V2 = fd.FunctionSpace(mesh, "DG", degree)
W = fd.MixedFunctionSpace((V1, V2))

H = case5.H0
f = case5.coriolis_expression(*x)
g = earth.Gravity
b = case5.topography_function(*x, V2, name="Topography")

# initial conditions
w0 = fd.Function(W)
un, hn = w0.split()

un.project(case5.velocity_expression(*x))
etan = case5.elevation_function(*x, V2, name="Elevation")
hn.assign(etan + H - b)
# D = eta + b


# nonlinear swe forms

def form_function(u, h, v, q):
    return swe.form_function(mesh, g, b, f, u, h, v, q)


def form_mass(u, h, v, q):
    return swe.form_mass(mesh, u, h, v, q)


dt = args.dt*units.hour

# parameters for the implicit diagonal solve in step-(b)
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
    'snes_monitor': None,
    'snes_converged_reason': None,
    "snes_atol": 1e-8,
    # "snes_rtol": 1e-8,
    'mat_type': 'matfree',
    'ksp_type': 'gmres',
    # 'ksp_type': 'preonly',
    'ksp_max_it': 10,
    "ksp_atol": 1e-8,
    # "ksp_rtol": 1e-8,
    'ksp_monitor': None,
    # "ksp_monitor_true_residual": None,
    "ksp_converged_reason": None,
    'pc_type': 'python',
    'pc_python_type': 'asQ.DiagFFTPC'}

for i in range(sum(M)):  # should this be sum(M) or max(M)?
    solver_parameters_diag["diagfft_"+str(i)+"_"] = sparameters

alpha = args.alpha
theta = 0.5

# non-petsc information for block solve
block_ctx = {}

# mesh transfer operators
transfer_managers = []
for _ in range(sum(M)):
    tm = mg.manifold_transfer_manager(W)
    transfer_managers.append(tm)

block_ctx['diag_transfer_managers'] = transfer_managers

PD = asQ.paradiag(ensemble=ensemble,
                  form_function=form_function,
                  form_mass=form_mass, W=W, w0=w0,
                  dt=dt, theta=theta,
                  alpha=alpha,
                  M=M, solver_parameters=solver_parameters_diag,
                  circ=None, tol=1.0e-6, maxits=None,
                  ctx={}, block_ctx=block_ctx, block_mat_type="aij")

PD.solve()

asQ.post.write_timeseries(PD,
                          file_name='output/'+args.filename,
                          function_names=['Velocity', 'Depth'],
                          frequency=1,
                          time_scale=1./units.hour)
