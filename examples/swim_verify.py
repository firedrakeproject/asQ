import firedrake as fd
from petsc4py import PETSc
import asQ

# multigrid transfer manager for diagonal block solve
from utils import units
from utils import mg
from utils.planets import earth
import utils.shallow_water.nonlinear as swe
from utils.shallow_water.williamson1992 import case5

# serial solution for verification
from swim_serial import swim_serial

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

nt = 4

# M = [1, 1, 1, 1]
# M = [2, 2]
M = [nt]

nspatial_domains = 4

# list of serial timesteps
PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Calculating serial solution --- === ###')
PETSc.Sys.Print('')

wserial = swim_serial(base_level=args.base_level,
                      ref_level=args.ref_level,
                      tmax=nt,
                      dumpt=1,
                      dt=args.dt,
                      coords_degree=args.coords_degree,
                      degree=args.degree)
PETSc.Sys.Print('')

PETSc.Sys.Print('### === --- Setting up parallel solution --- === ###')
PETSc.Sys.Print('')

# some domain, parameters and FS setup
H = case5.H0
distribution_parameters = {"partition": True, "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)}

# mesh set up
ensemble = fd.Ensemble(fd.COMM_WORLD, nspatial_domains)

mesh = mg.icosahedral_mesh(R0=earth.radius,
                           base_level=args.base_level,
                           degree=args.coords_degree,
                           distribution_parameters=distribution_parameters,
                           nrefs=args.ref_level-args.base_level,
                           comm=ensemble.comm)

x, y, z = fd.SpatialCoordinate(mesh)

V1 = fd.FunctionSpace(mesh, "BDM", args.degree+1)
V2 = fd.FunctionSpace(mesh, "DG", args.degree)
V0 = fd.FunctionSpace(mesh, "CG", args.degree+2)
W = fd.MixedFunctionSpace((V1, V2))

g = earth.Gravity
f = case5.coriolis_expression(x, y, z)
b = case5.topography_function(x, y, z, V2, name="Topography")
# D = eta + b

# initial conditions

# W = V1 * V2
w0 = fd.Function(W)
un, hn = w0.split()
un.project(case5.velocity_expression(x, y, z))
etan = case5.elevation_function(x, y, z, V2, name="Elevation")
hn.assign(etan + H - b)

# nonlinear swe forms


def form_function(u, h, v, q):
    return swe.form_function(mesh, g, b, f, h, u, q, v)


def form_mass(u, h, v, q):
    return swe.form_mass(mesh, h, u, q, v)


dt = args.dt*units.hour

# parameters for the implicit diagonal solve in step-(b)
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
    transfer_managers += [tm]

block_ctx['diag_transfer_managers'] = transfer_managers

PD = asQ.paradiag(ensemble=ensemble,
                  form_function=form_function,
                  form_mass=form_mass, W=W, w0=w0,
                  dt=dt, theta=theta,
                  alpha=alpha,
                  M=M, solver_parameters=solver_parameters_diag,
                  circ=None, tol=1.0e-8, maxits=None,
                  ctx={}, block_ctx=block_ctx, block_mat_type="aij")

PETSc.Sys.Print('### === --- Calculating parallel solution --- === ###')
PETSc.Sys.Print('')

PD.solve()

PETSc.Sys.Print('### === --- Comparing solutions --- === ###')
PETSc.Sys.Print('')

ws = fd.Function(W)
wp = fd.Function(W)

us, hs = ws.split()
up, hp = wp.split()

for i in range(nt):

    us.assign(wserial[i].split()[0])
    hs.assign(wserial[i].split()[1])

    up.assign(PD.w_all.split()[2*i])
    hp.assign(PD.w_all.split()[2*i+1])

    hmag = fd.sqrt(fd.assemble(hs*hs*fd.dx))
    umag = fd.sqrt(fd.assemble(fd.inner(us, us)*fd.dx))

    herror = fd.sqrt(fd.assemble((hp - hs)*(hp - hs)*fd.dx))
    uerror = fd.sqrt(fd.assemble(fd.inner(up - us, up - us)*fd.dx))

    PETSc.Sys.Print('timestep:', i, 'uerror:', uerror/umag, '|', 'herror: ', herror/hmag)

PETSc.Sys.Print('')
