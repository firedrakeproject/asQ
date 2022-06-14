
import firedrake as fd
from petsc4py import PETSc
import asQ

from utils import mg
from utils import units
from utils.planets import earth
import utils.shallow_water.nonlinear as swe
import utils.shallow_water.williamson1992.case2 as case2

PETSc.Sys.popErrorHandler()

# get command arguments
import argparse
parser = argparse.ArgumentParser(description='Williamson 2 testcase for ParaDiag solver using fully implicit SWE solver.')
parser.add_argument('--base_level', type=int, default=1, help='Base refinement level of icosahedral grid for MG solve. Default 1.')
parser.add_argument('--ref_level', type=int, default=2, help='Refinement level of icosahedral grid. Default 2.')
parser.add_argument('--nwindows', type=int, default=1, help='Number of time-windows. Default 1.')
parser.add_argument('--nslices', type=int, default=2, help='Number of time-slices per time-window. Default 2.')
parser.add_argument('--slice_length', type=int, default=2, help='Number of timesteps per time-slice. Default 2.')
parser.add_argument('--nspatial_domains', type=int, default=2, help='Size of spatial partition. Default 2.')
parser.add_argument('--alpha', type=float, default=0.0001, help='Circulant coefficient. Default 0.0001.')
parser.add_argument('--dt', type=float, default=0.5, help='Timestep in hours. Default 0.5.')
parser.add_argument('--filename', type=str, default='w5diag')
parser.add_argument('--coords_degree', type=int, default=1, help='Degree of polynomials for sphere mesh approximation. Default 1.')
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space).')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Setting up --- === ###')
PETSc.Sys.Print('')

# time steps

M = [args.slice_length for _ in range(args.nslices)]
window_length = sum(M)
nsteps = args.nwindows*window_length

dt = args.dt*units.hour

# multigrid mesh set up

ensemble = fd.Ensemble(fd.COMM_WORLD, args.nspatial_domains)

distribution_parameters = {"partition": True, "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)}

# mesh set up
mesh = mg.icosahedral_mesh(R0=earth.radius,
                           base_level=args.base_level,
                           degree=args.coords_degree,
                           distribution_parameters=distribution_parameters,
                           nrefs=args.ref_level-args.base_level,
                           comm=ensemble.comm)
x = fd.SpatialCoordinate(mesh)

# Mixed function space for velocity and depth
degree = args.degree
V1 = fd.FunctionSpace(mesh, "BDM", degree+1)
V2 = fd.FunctionSpace(mesh, "DG", degree)
W = fd.MixedFunctionSpace((V1, V2))

# initial conditions
w0 = fd.Function(W)
un, hn = w0.split()

f = case2.coriolis_expression(*x)
b = fd.Constant(0)
H = case2.H0

un.project(case2.velocity_expression(*x))
etan = case2.elevation_function(*x, V2, name="Elevation")
hn.assign(H + etan - b)


# nonlinear swe forms

def form_function(u, h, v, q):
    return swe.form_function(mesh, earth.Gravity, b, f, u, h, v, q)


def form_mass(u, h, v, q):
    return swe.form_mass(mesh, u, h, v, q)


# parameters for the implicit diagonal solve in step-(b)
sparameters = {
    # 'snes_monitor': None,
    'mat_type': 'matfree',
    # 'ksp_type': 'preonly',
    'ksp_type': 'fgmres',
    # 'ksp_monitor': None,
    # 'ksp_monitor_true_residual': None,
    # 'ksp_converged_reason': None,
    'ksp_atol': 1e-8,
    'ksp_rtol': 1e-8,
    'ksp_max_it': 400,
    'pc_type': 'mg',
    'pc_mg_cycle_type': 'v',
    'pc_mg_type': 'multiplicative',
    'mg_levels_ksp_type': 'gmres',
    'mg_levels_ksp_max_it': 5,
    # 'mg_levels_ksp_convergence_test': 'skip',
    'mg_levels_pc_type': 'python',
    'mg_levels_pc_python_type': 'firedrake.PatchPC',
    'mg_levels_patch_pc_patch_save_operators': True,
    'mg_levels_patch_pc_patch_partition_of_unity': True,
    'mg_levels_patch_pc_patch_sub_mat_type': 'seqdense',
    'mg_levels_patch_pc_patch_construct_codim': 0,
    'mg_levels_patch_pc_patch_construct_type': 'vanka',
    'mg_levels_patch_pc_patch_local_type': 'additive',
    'mg_levels_patch_pc_patch_precompute_element_tensors': True,
    'mg_levels_patch_pc_patch_symmetrise_sweep': False,
    'mg_levels_patch_sub_ksp_type': 'preonly',
    'mg_levels_patch_sub_pc_type': 'lu',
    'mg_levels_patch_sub_pc_factor_shift_type': 'nonzero',
    'mg_coarse_pc_type': 'python',
    'mg_coarse_pc_python_type': 'firedrake.AssembledPC',
    'mg_coarse_assembled_pc_type': 'lu',
    'mg_coarse_assembled_pc_factor_mat_solver_type': 'mumps',
}

sparameters_diag = {
    'snes_linesearch_type': 'basic',
    # 'snes_linesearch_damping': 1.0,
    'snes_monitor': None,
    'snes_converged_reason': None,
    'snes_atol': 1e-0,
    'snes_rtol': 1e-12,
    'snes_stol': 1e-12,
    # 'snes_divergence_tolerance': 1e6,
    'snes_max_it': 100,
    'mat_type': 'matfree',
    'ksp_type': 'gmres',
    # 'ksp_type': 'preonly',
    # 'ksp_atol': 1e-8,
    # 'ksp_rtol': 1e-8,
    # 'ksp_stol': 1e-8,
    # 'ksp_max_it': 2000,
    # 'ksp_gmres_restart': 100,
    # 'ksp_gmres_modifiedgramschmidt': None,
    # 'ksp_max_it': 300,
    # 'ksp_convergence_test': 'skip',
    # 'snes_max_linear_solver_fails': 5,
    'ksp_monitor': None,
    # 'ksp_monitor_true_residual': None,
    'ksp_converged_reason': None,
    'pc_type': 'python',
    'pc_python_type': 'asQ.DiagFFTPC'}

PETSc.Sys.Print('### === --- Calculating parallel solution --- === ###')
PETSc.Sys.Print('')

for i in range(sum(M)):
    sparameters_diag['diagfft_'+str(i)+'_'] = sparameters

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
                  dt=dt, theta=0.5,
                  alpha=args.alpha,
                  M=M, solver_parameters=sparameters_diag,
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

    herr = fd.errornorm(hn, hp)/fd.norm(hn)
    uerr = fd.errornorm(un, up)/fd.norm(un)

    timestep = sum(M[:PD.rT]) + step
    PETSc.Sys.Print(f"timestep={timestep}, herr={herr}, uerr={uerr}", comm=ensemble.comm)
