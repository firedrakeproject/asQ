import numpy as np
import firedrake as fd
from petsc4py import PETSc
import asQ

from utils import mg
from utils import units
from utils.planets import earth
import utils.shallow_water.nonlinear as swe
from utils.shallow_water.williamson1992 import case5
from utils.shallow_water.verifications.williamson5 import serial_solve

PETSc.Sys.popErrorHandler()

# get command arguments
import argparse
parser = argparse.ArgumentParser(description='Williamson 5 testcase for approximate Schur complement solver.')
parser.add_argument('--base_level', type=int, default=1, help='Base refinement level of icosahedral grid for MG solve. Default 1.')
parser.add_argument('--ref_level', type=int, default=2, help='Refinement level of icosahedral grid. Default 3.')
parser.add_argument('--nwindows', type=int, default=1, help='Number of time-windows. Default 1.')
parser.add_argument('--nslices', type=int, default=2, help='Number of time-slices. Default 2.')
parser.add_argument('--slice_length', type=int, default=2, help='Number of timesteps per time-slice. Default 2.')
parser.add_argument('--nspatial_domains', type=int, default=2, help='Size of spatial partition. Default 2.')
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

PETSc.Sys.Print('### === --- Setting up --- === ###')
PETSc.Sys.Print('')

M = [args.slice_length for _ in range(args.nslices)]
nsteps = args.nwindows*sum(M)

# some domain, parameters and FS setup
distribution_parameters = {"partition": True, "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)}

ensemble = fd.Ensemble(fd.COMM_WORLD, args.nspatial_domains)

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
    # 'snes_monitor': None,
    'mat_type': 'matfree',
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
    'snes_monitor': None,
    'snes_converged_reason': None,
    'snes_atol': 1e-0,
    'snes_rtol': 1e-16,
    'snes_stol': 1e-100,
    'mat_type': 'matfree',
    'ksp_type': 'gmres',
    # 'ksp_type': 'preonly',
    'ksp_max_it': 10,
    # 'ksp_atol': 1e-8,
    # 'ksp_rtol': 1e-8,
    # 'ksp_stol': 1e-8,
    'ksp_monitor': None,
    # 'ksp_monitor_true_residual': None,
    'ksp_converged_reason': None,
    'pc_type': 'python',
    'pc_python_type': 'asQ.DiagFFTPC'}

PETSc.Sys.Print('### === --- Calculating serial solution --- === ###')
PETSc.Sys.Print('')
wserial = serial_solve(base_level=args.base_level,
                       ref_level=args.ref_level,
                       tmax=nsteps,
                       dumpt=1,
                       dt=args.dt,
                       coords_degree=args.coords_degree,
                       degree=args.degree,
                       sparameters=sparameters,
                       comm=ensemble.comm,
                       verbose=False)

sparameters['ksp_type'] = 'preonly'

for i in range(sum(M)):  # should this be sum(M) or max(M)?
    sparameters_diag['diagfft_'+str(i)+'_'] = sparameters

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
                  M=M, solver_parameters=sparameters_diag,
                  circ=None, tol=1.0e-6, maxits=None,
                  ctx={}, block_ctx=block_ctx, block_mat_type="aij")

time_series = []

ws = fd.Function(W)
wp = fd.Function(W)

us, hs = ws.split()
up, hp = wp.split()

r = PD.rT
# solve for each window
PETSc.Sys.Print('### === --- Calculating parallel solution --- === ###')
linear_its = 0
nonlinear_its = 0
for w in range(args.nwindows):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'### === --- Calculating time-window {w} --- === ###')
    PETSc.Sys.Print('')

    PD.solve()
    linear_its += PD.snes.getLinearSolveIterations()
    nonlinear_its += PD.snes.getIterationNumber()

    # build time series
    window_series = []

    for i in range(M[r]):

        up.assign(PD.w_all.split()[2*i])
        hp.assign(PD.w_all.split()[2*i+1])

        window_series.append(wp.copy(deepcopy=True))

    time_series.append(window_series)

    PD.next_window()

# compare against serial solution
window_length = args.nslices*args.slice_length

local_errors = np.zeros((2, nsteps))
errors = np.zeros((2, nsteps))

PETSc.Sys.Print('### === --- Calculate parallel-serial differences errors --- === ###')
PETSc.Sys.Print('')

for w in range(args.nwindows):

    window_step0 = w*window_length
    slice_step0 = window_step0 + sum(M[:r])

    for i in range(M[r]):

        tstep = slice_step0 + i

        up.assign(time_series[w][i].split()[0])
        hp.assign(time_series[w][i].split()[1])

        us.assign(wserial[tstep].split()[0])
        hs.assign(wserial[tstep].split()[1])

        uerror = fd.errornorm(us, up)/fd.norm(us)
        herror = fd.errornorm(hs, hp)/fd.norm(hs)

        local_errors[0][tstep] = uerror
        local_errors[1][tstep] = herror

ensemble.ensemble_comm.Allreduce(local_errors, errors)

if r == 0:
    for tstep in range(nsteps):
        uerror = errors[0][tstep]
        herror = errors[1][tstep]

        PETSc.Sys.Print('timestep:', tstep, '|', 'uerror:', uerror, '|', 'herror: ', herror, comm=ensemble.comm)

    PETSc.Sys.Print('')
    PETSc.Sys.Print('nonlinear iterations:', nonlinear_its, '|', 'linear iterations:', linear_its, comm=ensemble.comm)
    PETSc.Sys.Print('')
