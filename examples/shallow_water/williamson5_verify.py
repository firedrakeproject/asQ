import numpy as np
import firedrake as fd
from petsc4py import PETSc

from utils.shallow_water.williamson1992 import case5
from utils.shallow_water.verifications.williamson5 import serial_solve, parallel_solve

PETSc.Sys.popErrorHandler()

# get command arguments
import argparse
parser = argparse.ArgumentParser(description='Williamson 5 testcase for ParaDiag solver using fully implicit SWE solver.')
parser.add_argument('--base_level', type=int, default=1, help='Base refinement level of icosahedral grid for MG solve. Default 1.')
parser.add_argument('--ref_level', type=int, default=2, help='Refinement level of icosahedral grid. Default 2.')
parser.add_argument('--nwindows', type=int, default=1, help='Number of time-windows. Default 1.')
parser.add_argument('--nslices', type=int, default=2, help='Number of time-slices. Default 2.')
parser.add_argument('--slice_length', type=int, default=2, help='Number of timesteps per time-slice. Default 2.')
parser.add_argument('--nspatial_domains', type=int, default=2, help='Size of spatial partition. Default 2.')
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

M = [args.slice_length for _ in range(args.nslices)]

nsteps = args.nwindows*sum(M)

# mesh set up
ensemble = fd.Ensemble(fd.COMM_WORLD, args.nspatial_domains)

trank = ensemble.ensemble_comm.rank

# block solver options
sparameters = {
    "snes_atol": 1e-8,
    "mat_type": "matfree",
    "ksp_type": "fgmres",
    "ksp_atol": 1e-8,
    "ksp_max_it": 400,
    "pc_type": "mg",
    "pc_mg_cycle_type": "v",
    "pc_mg_type": "multiplicative",
    "mg_levels_ksp_type": "gmres",
    "mg_levels_ksp_max_it": 3,
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

# paradiag solver options
sparameters_diag = {
    'snes_monitor': None,
    'snes_converged_reason': None,
    'ksp_monitor': None,
    'ksp_converged_reason': None,
    'snes_linesearch_type': 'basic',
    'snes_atol': 1e-0,
    'snes_rtol': 1e-16,
    'snes_stol': 1e-100,
    'mat_type': 'matfree',
    'ksp_type': 'gmres',
    'ksp_max_it': 10,
    'pc_type': 'python',
    'pc_python_type': 'asQ.DiagFFTPC'}


# list of serial timesteps
PETSc.Sys.Print('')
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

# only keep the timesteps on the current time-slice
timestep_start = sum(M[:trank])
timestep_end = timestep_start + M[trank]

PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Calculating parallel solution --- === ###')
PETSc.Sys.Print('')

# block solve is linear for parallel solution
sparameters['ksp_type'] = 'preonly'

wparallel = parallel_solve(base_level=args.base_level,
                           ref_level=args.ref_level,
                           M=M,
                           nwindows=args.nwindows,
                           nspatial_domains=args.nspatial_domains,
                           dumpt=1,
                           dt=args.dt,
                           coords_degree=args.coords_degree,
                           degree=args.degree,
                           sparameters=sparameters,
                           sparameters_diag=sparameters_diag,
                           ensemble=ensemble,
                           alpha=args.alpha,
                           verbose=True)

PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Comparing solutions --- === ###')
PETSc.Sys.Print('')

window_length = sum(M)

W = wserial[0].function_space()

ws = fd.Function(W)
wp = fd.Function(W)

us, hs = ws.split()
up, hp = wp.split()

# calculate error against serial
local_errors = np.zeros((2, nsteps))

b = case5.topography_function(*fd.SpatialCoordinate(W.mesh()),
                              fd.FunctionSpace(W.mesh(), "DG", args.degree),
                              name="Topography")

for w in range(args.nwindows):

    # time step at beginning this window, and the local slice
    window_step0 = w*window_length
    slice_step0 = window_step0 + sum(M[:trank])

    # each rank calculates error for own slice
    for i in range(M[trank]):

        tstep = slice_step0 + i

        up.assign(wparallel[w][i].split()[0])
        hp.assign(wparallel[w][i].split()[1])
        hp.assign(hp + b - case5.H0)

        us.assign(wserial[tstep].split()[0])
        hs.assign(wserial[tstep].split()[1])
        hs.assign(hs + b - case5.H0)

        uerror = fd.errornorm(us, up)/fd.norm(us)
        herror = fd.errornorm(hs, hp)/fd.norm(hs)

        local_errors[0][tstep] = uerror
        local_errors[1][tstep] = herror

# errors over whole series
errors = np.zeros((2, nsteps))
ensemble.ensemble_comm.Allreduce(local_errors, errors)

if trank == 0:
    for tstep in range(nsteps):
        uerror = errors[0][tstep]
        herror = errors[1][tstep]
        PETSc.Sys.Print('timestep:', f'{tstep:>3}', '|', 'uerror:', f'{uerror:<.4e}', '|', 'herror: ', f'{herror:<.4e}', comm=ensemble.comm)
    PETSc.Sys.Print('', comm=ensemble.comm)
