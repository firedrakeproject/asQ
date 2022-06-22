
import firedrake as fd
from petsc4py import PETSc
import asQ

from utils import mg
from utils import units
from utils import diagnostics
from utils.planets import earth
import utils.shallow_water as swe
from utils.shallow_water.williamson1992 import case5

PETSc.Sys.popErrorHandler()

# get command arguments
import argparse
parser = argparse.ArgumentParser(description='Williamson 5 testcase for ParaDiag solver using fully implicit SWE solver.')
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
V1 = swe.default_velocity_function_space(mesh, degree=args.degree)
V2 = swe.default_depth_function_space(mesh, degree=args.degree)
W = fd.MixedFunctionSpace((V1, V2))

# initial conditions
w0 = fd.Function(W)
un, hn = w0.split()

f = case5.coriolis_expression(*x)
b = case5.topography_function(*x, V2, name="Topography")
H = case5.H0

un.project(case5.velocity_expression(*x))
etan = case5.elevation_function(*x, V2, name="Elevation")
hn.assign(H + etan - b)


# nonlinear swe forms

def form_function(u, h, v, q):
    return swe.nonlinear.form_function(mesh, earth.Gravity, b, f, u, h, v, q)


def form_mass(u, h, v, q):
    return swe.nonlinear.form_mass(mesh, u, h, v, q)


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
    'ksp_type': 'fgmres',
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


# only last slice does diagnostics/output
if PD.rT == len(M)-1:
    cfl_series = []
    linear_its = 0
    nonlinear_its = 0

    ofile = fd.File('output/'+args.filename+'.pvd',
                    comm=ensemble.comm)

    uout = fd.Function(V1, name='velocity')
    hout = fd.Function(V2, name='depth')

    cfl_calc = diagnostics.convective_cfl_calculator(mesh)

    pvcalc = diagnostics.potential_vorticity_calculator(
        V1, name='vorticity')

    def assign_out_functions():
        uout.assign(PD.w_all.split()[-2])
        hout.assign(PD.w_all.split()[-1])
        hout.assign(hout + b - H)

    def time_at_last_step(w):
        return dt*(w + 1)*window_length

    def write_to_file(t):
        ofile.write(uout, hout, pvcalc(uout), time=t/earth.day)

    def max_cfl():
        with cfl_calc(uout, dt).dat.vec_ro as v:
            return v.max()[1]


def window_preproc(pdg, wndw):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'### === --- Calculating time-window {wndw} --- === ###')
    PETSc.Sys.Print('')


def window_postproc(pdg, wndw):
    # make sure variables are properly captured
    global linear_its
    global nonlinear_its
    global cfl_series

    # postprocess this timeslice
    if PD.rT == len(M)-1:
        linear_its += pdg.snes.getLinearSolveIterations()
        nonlinear_its += pdg.snes.getIterationNumber()

        assign_out_functions()

        time = time_at_last_step(wndw)

        # write to file at the end of each day
        for day in range(51):
            midnight = day*earth.day
            if midnight-0.5*dt < time < midnight+0.5*dt:
                write_to_file(time)

        cfl = max_cfl()
        cfl_series += [cfl]
        PETSc.Sys.Print('', comm=ensemble.comm)
        PETSc.Sys.Print(f'Maximum CFL = {cfl}', comm=ensemble.comm)
        PETSc.Sys.Print(f'Hours = {time/units.hour}', comm=ensemble.comm)
        PETSc.Sys.Print(f'Days = {time/earth.day}', comm=ensemble.comm)
        PETSc.Sys.Print('', comm=ensemble.comm)

    PETSc.Sys.Print('')


# solve for each window
PD.solve(nwindows=args.nwindows,
         preproc=window_preproc,
         postproc=window_postproc)


PETSc.Sys.Print('### === --- Iteration counts --- === ###')
PETSc.Sys.Print('')

if PD.rT == len(M)-1:
    PETSc.Sys.Print(f'Maximum CFL = {max(cfl_series)}', comm=ensemble.comm)
    PETSc.Sys.Print(f'Minimum CFL = {min(cfl_series)}', comm=ensemble.comm)
    PETSc.Sys.Print('', comm=ensemble.comm)

    PETSc.Sys.Print(f'windows: {(args.nwindows)}', comm=ensemble.comm)
    PETSc.Sys.Print(f'timesteps: {(args.nwindows)*window_length}', comm=ensemble.comm)
    PETSc.Sys.Print('', comm=ensemble.comm)

    PETSc.Sys.Print(f'linear iterations: {linear_its} | iterations per window: {linear_its/(args.nwindows)}', comm=ensemble.comm)
    PETSc.Sys.Print(f'nonlinear iterations: {nonlinear_its} | iterations per window: {nonlinear_its/(args.nwindows)}', comm=ensemble.comm)
    PETSc.Sys.Print('', comm=ensemble.comm)
