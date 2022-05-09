import firedrake as fd
from petsc4py import PETSc
import asQ

import post

# multigrid transfer manager for diagonal block solve
from firedrake_utils import mg

PETSc.Sys.popErrorHandler()

# get command arguments
import argparse
parser = argparse.ArgumentParser(description='Williamson 5 testcase for approximate Schur complement solver.')
parser.add_argument('--base_level', type=int, default=1, help='Base refinement level of icosahedral grid for MG solve. Default 1.')
parser.add_argument('--ref_level', type=int, default=2, help='Refinement level of icosahedral grid. Default 2.')
# parser.add_argument('--nsteps', type=int, default=10, help='Number of timesteps. Default 10.')
parser.add_argument('--alpha', type=float, default=0.0001, help='Circulant coefficient. Default 0.0001.')
parser.add_argument('--dt', type=float, default=0.05, help='Timestep in hours. Default 0.05.')
parser.add_argument('--filename', type=str, default='w5diag')
parser.add_argument('--coords_degree', type=int, default=3, help='Degree of polynomials for sphere mesh approximation.')
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space).')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

# some domain, parameters and FS setup
R0 = 6371220.
H = fd.Constant(5960.)
base_level = args.base_level
nrefs = args.ref_level - base_level
deg = args.coords_degree
distribution_parameters = {"partition": True, "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)}

# mesh set up
nspatial_domains = 2
ensemble = fd.Ensemble(fd.COMM_WORLD, nspatial_domains)

mesh = mg.icosahedral_mesh(R0=R0,
                           base_level=args.base_level,
                           degree=args.coords_degree,
                           distribution_parameters=distribution_parameters,
                           nrefs=args.ref_level-args.base_level,
                           comm=ensemble.comm)

R0 = fd.Constant(R0)
x, y, z = fd.SpatialCoordinate(mesh)

outward_normals = fd.CellNormal(mesh)

degree = args.degree
V1 = fd.FunctionSpace(mesh, "BDM", degree+1)
V2 = fd.FunctionSpace(mesh, "DG", degree)
V0 = fd.FunctionSpace(mesh, "CG", degree+2)
W = fd.MixedFunctionSpace((V1, V2))

Omega = fd.Constant(7.292e-5)  # rotation rate
f = 2*Omega*z/fd.Constant(R0)  # Coriolis parameter
g = fd.Constant(9.8)  # Gravitational constant
b = fd.Function(V2, name="Topography")
# D = eta + b

# nonlinear swe forms


def perp(u):
    return fd.cross(outward_normals, u)


def both(u):
    return 2*fd.avg(u)


def form_function(u, h, v, q):
    K = 0.5*fd.inner(u, u)
    n = fd.FacetNormal(mesh)
    uup = 0.5 * (fd.dot(u, n) + abs(fd.dot(u, n)))
    Upwind = 0.5 * (fd.sign(fd.dot(u, n)) + 1)

    eqn = (
        fd.inner(v, f*perp(u))*fd.dx
        - fd.inner(perp(fd.grad(fd.inner(v, perp(u)))), u)*fd.dx
        + fd.inner(both(perp(n)*fd.inner(v, perp(u))),
                   both(Upwind*u))*fd.dS
        - fd.div(v)*(g*(h + b) + K)*fd.dx
        - fd.inner(fd.grad(q), u)*h*fd.dx
        + fd.jump(q)*(uup('+')*h('+')
                      - uup('-')*h('-'))*fd.dS
    )
    return eqn


def form_mass(u, h, v, q):
    return fd.inner(u, v)*fd.dx + h*q*fd.dx


dt = 60*60*args.dt
t = 0.

# initial conditions

u_0 = 20.0  # maximum amplitude of the zonal wind [m/s]
u_max = fd.Constant(u_0)
u_expr = fd.as_vector([-u_max*y/R0, u_max*x/R0, 0.0])
eta_expr = - ((R0 * Omega * u_max + u_max*u_max/2.0)*(z*z/(R0*R0)))/g
# W = V1 * V2
w0 = fd.Function(W)
un, etan = w0.split()
un.project(u_expr)
etan.project(eta_expr)

# Topography.
rl = fd.pi/9.0
lambda_x = fd.atan_2(y/R0, x/R0)
lambda_c = -fd.pi/2.0
phi_x = fd.asin(z/R0)
phi_c = fd.pi/6.0
minarg = fd.Min(pow(rl, 2),
                pow(phi_x - phi_c, 2) + pow(lambda_x - lambda_c, 2))
bexpr = 2000.0*(1 - fd.sqrt(minarg)/rl)
b.interpolate(bexpr)

# parameters for the implicit diagonal solve in step-(b)
sparameters_orig = {
    # "ksp_converged_reason": None,
    "ksp_type": "preonly",
    'pc_python_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps'}

sparameters_new = {
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

sparameters = sparameters_new

solver_parameters_diag = {
    "snes_linesearch_type": "basic",
    'snes_monitor': None,
    'snes_converged_reason': None,
    'mat_type': 'matfree',
    'ksp_type': 'gmres',
    # 'ksp_type': 'preonly',
    'ksp_max_it': 10,
    'ksp_monitor': None,
    # "ksp_monitor_true_residual": None,
    "ksp_converged_reason": None,
    'pc_type': 'python',
    'pc_python_type': 'asQ.DiagFFTPC'}

# M = [1, 1, 1, 1, 1, 1, 1, 1]
# M = [2, 2, 2, 2]
# M = [4, 4]
M = [3]

for i in range(sum(M)):  # should this be sum(M) or max(M)?
    solver_parameters_diag["diagfft_"+str(i)+"_"] = sparameters

alpha = args.alpha
theta = 0.5

# non-petsc information for block solve
block_ctx = {}

# mesh transfer operators
transfer_managers = []
for _ in range(sum(M)):
    #vtransfer = mg.ManifoldTransfer()
    #transfers = {
    #    V1.ufl_element(): (vtransfer.prolong, vtransfer.restrict,
    #                       vtransfer.inject),
    #    V2.ufl_element(): (vtransfer.prolong, vtransfer.restrict,
    #                       vtransfer.inject)
    #}
    #transfer_managers += [fd.TransferManager(native_transfers=transfers)]
    tm = mg.manifold_transfer_manager(W)
    transfer_managers += [tm]

block_ctx['diag_transfer_managers'] = transfer_managers

PD = asQ.paradiag(ensemble=ensemble,
                  form_function=form_function,
                  form_mass=form_mass, W=W, w0=w0,
                  dt=dt, theta=theta,
                  alpha=alpha,
                  M=M, solver_parameters=solver_parameters_diag,
                  circ=None,
                  jac_average="newton", tol=1.0e-6, maxits=None,
                  ctx={}, block_ctx=block_ctx, block_mat_type="aij")

PD.solve()

filename = 'output/'+args.filename
funcnames = ['velocity','elevation']
post.write_timesteps(PD,
                     file_name=filename,
                     function_names=funcnames)

