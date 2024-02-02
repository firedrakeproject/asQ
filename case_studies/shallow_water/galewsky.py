from firedrake.petsc import PETSc

import asQ
import firedrake as fd
from utils.timing import SolverTimer
from utils import units
from utils.planets import earth
import utils.shallow_water as swe
from utils.shallow_water import galewsky

from math import sqrt
from functools import partial

PETSc.Sys.popErrorHandler()

# get command arguments
import argparse
parser = argparse.ArgumentParser(
    description='Galewsky testcase for ParaDiag solver using fully implicit SWE solver.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--ref_level', type=int, default=2, help='Refinement level of icosahedral grid.')
parser.add_argument('--base_level', type=int, default=2, help='Base refinement level for multigrid.')
parser.add_argument('--nwindows', type=int, default=1, help='Number of time-windows.')
parser.add_argument('--nslices', type=int, default=2, help='Number of time-slices per time-window.')
parser.add_argument('--slice_length', type=int, default=2, help='Number of timesteps per time-slice.')
parser.add_argument('--slice_jacobi_nsteps', type=int, default=4, help='Number of timesteps per SliceJacobiPC.')
parser.add_argument('--aaopc', type=str, default='CirculantPC', choices=['CirculantPC', 'JacobiPC', 'SliceJacobiPC'], help='Preconditioner for the AllAtOnceSystem')
parser.add_argument('--dt', type=float, default=0.5, help='Timestep in hours.')

parser.add_argument('--alpha', type=float, default=1e-4, help='Circulant coefficient.')

parser.add_argument('--block_method', type=str, default='aux', help='PC method for the blocks. aux or lu or mg or hybr.')
parser.add_argument('--atol', type=float, default=1e0, help='Average atol of each timestep.')
parser.add_argument('--block_rtol', type=float, default=1e-5, help='rtol for the block solves.')
parser.add_argument('--block_max_it', type=int, default=25, help='Maximum iterations for the blocks.')

parser.add_argument('--filename', type=str, default='galewsky', help='Name of output vtk files')
parser.add_argument('--metrics_dir', type=str, default='metrics', help='Directory to save paradiag metrics to.')
parser.add_argument('--record_cfl', action='store_true', help='Calculate and output CFL at each window.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Setting up --- === ###')
PETSc.Sys.Print('')

# time steps

time_partition = tuple((args.slice_length for _ in range(args.nslices)))
window_length = sum(time_partition)
nsteps = args.nwindows*window_length

dt = args.dt*units.hour

# alternative operator to precondition blocks
H = galewsky.H0
g = earth.Gravity


def aux_form_function(u, h, v, q, t):
    mesh = u.ufl_domain()
    coords = fd.SpatialCoordinate(mesh)
    f = swe.earth_coriolis_expression(*coords)
    return swe.linear.form_function(mesh, g, H, f,
                                    u, h, v, q, t)


def form_mass_tr(u, h, tr, v, q, s):
    mesh = v.ufl_domain()
    return swe.linear.form_mass(mesh, u, h, v, q)


def form_function_tr(u, h, tr, v, q, dtr, t=None):
    mesh = v.ufl_domain()
    coords = fd.SpatialCoordinate(mesh)
    f = swe.earth_coriolis_expression(*coords)
    K = swe.linear.form_function(mesh, g, H, f,
                                 u, h, v, q, t)
    n = fd.FacetNormal(mesh)
    Khybr = (
        g*fd.jump(v, n)*tr('+')
        + fd.jump(u, n)*dtr('+')
    )*fd.dS

    return K + Khybr


# PC forming approximate hybridisable system (without advection)
# solve it using hybridisation and then return the DG part
# (for use in a Schur compement setup)
class ApproxHybridPC(fd.PCBase):
    def initialize(self, pc):
        if pc.getType() != "python":
            raise ValueError("Expecting PC type python")

        cpx = appctx['cpx']

        u0 = appctx['u0']
        d1 = appctx['d1']
        d2 = appctx['d2']

        mesh = u0.ufl_domain()

        W = u0.function_space()
        Wu, Wh = W.subfunctions

        Wub = fd.FunctionSpace(mesh, fd.BrokenElement(Wu.ufl_element()))
        Tr = fd.FunctionSpace(mesh, "HDivT", Wu.ufl_element().degree())
        Wtr = Wub*Wh*Tr

        # input and output functions
        self.xfstar = fd.Cofunction(Wh.dual())
        self.xf = fd.Function(Wh)  # result of riesz map of the above
        self.yf = fd.Function(Wh)  # the preconditioned residual

        Mtr = cpx.BilinearForm(Wtr, d1, form_mass_tr)
        Ktr = cpx.BilinearForm(Wtr, d2, form_function_tr)

        Atr = Mtr + Ktr

        self.wtr = fd.Function(Wtr)
        _, self.htr, _ = self.wtr.subfunctions
        _, q, _ = fd.TestFunctions(Wtr)

        Ltr = fd.inner(q, self.xf)*fd.dx

        condensed_params = {'ksp_type': 'preonly',
                            'pc_type': 'lu',
                            "pc_factor_mat_solver_type": "mumps"}

        hbps = {
            "mat_type": "matfree",
            "ksp_type": "preonly",
            "pc_type": "python",
            "pc_python_type": "firedrake.SCPC",
            'pc_sc_eliminate_fields': '0, 1',
            'condensed_field': condensed_params
        }

        problem = fd.LinearVariationalProblem(Atr, Ltr, self.wtr)
        self.solver = fd.LinearVariationalSolver(
            problem, solver_parameters=hbps)

    def update(self, pc):
        pass

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError

    def apply(self, pc, x, y):
        # copy petsc vec into Function
        with self.xfstar.dat.vec_wo as v:
            x.copy(v)
        self.xf.assign(self.xfstar.riesz_representation())
        self.wtr.assign(0)
        self.solver.solve()
        self.yf.assign(self.htr)

        # copy petsc vec into Function
        with self.yf.dat.vec_ro as v:
            v.copy(y)


block_appctx = {'aux_form_function': aux_form_function}

# parameters for the implicit diagonal solve in step-(b)
factorisation_params = {
    'ksp_type': 'preonly',
    # 'pc_factor_mat_ordering_type': 'rcm',
    'pc_factor_reuse_ordering': None,
    'pc_factor_reuse_fill': None,
}

lu_pc = {'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}
lu_pc.update(factorisation_params)

patch_parameters = {
    'pc_patch': {
        'save_operators': True,
        'partition_of_unity': True,
        'sub_mat_type': 'seqdense',
        'construct_dim': 0,
        'construct_type': 'vanka',
        'local_type': 'additive',
        'precompute_element_tensors': True,
        'symmetrise_sweep': False
    },
    'sub': {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_shift_type': 'nonzero',
    }
}

mg_parameters = {
    'levels': {
        'ksp_type': 'gmres',
        'ksp_max_it': 4,
        'pc_type': 'python',
        'pc_python_type': 'firedrake.PatchPC',
        'patch': patch_parameters
    },
    'coarse': {
        'pc_type': 'python',
        'pc_python_type': 'firedrake.AssembledPC',
        'assembled': lu_pc,
    }
}

mg_pc = {
    'pc_type': 'mg',
    'pc_mg_cycle_type': 'v',
    'pc_mg_type': 'multiplicative',
    'mg': mg_parameters
}

aux_pc = {
    'snes_lag_preconditioner': -2,
    'snes_lag_preconditioner_persists': None,
    'pc_type': 'python',
    'pc_python_type': 'asQ.AuxiliaryBlockPC',
    'aux': lu_pc,
}

hybr_pc = {
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'schur',
    'pc_fieldsplit_schur_fact_type': 'full',
    'fieldsplit_0': {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
    },
    'fieldsplit_1': {
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'pc_python_type': f'{__name__}.ApproxHybridPC',
    },
}

block_sparams = {
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp': {
        'atol': 1e-100,
        'rtol': args.block_rtol,
        'max_it': args.block_max_it,
        'converged_maxits': None,
    },
}

if args.block_method == 'lu':
   block_sparams = lu_pc
else:
   block_sparams.update({
      'mg': mg_pc,
      'aux': aux_pc,
      'hybr': hybr_pc,
      }[args.block_method])

patol = sqrt(window_length)*args.atol

none_parameters = {
    'pc_type': 'none'
}

circulant_parameters = {
    'pc_type': 'python',
    'pc_python_type': 'asQ.CirculantPC',
    'diagfft_block': block_sparams,
    'diagfft_alpha': args.alpha,
}

jacobi_parameters = {
    'pc_type': 'python',
    'pc_python_type': 'asQ.JacobiPC',
    'aaojacobi_block': block_sparams,
}

slice_jacobi_parameters = {
    'pc_type': 'python',
    'pc_python_type': 'asQ.SliceJacobiPC',
    'slice_jacobi_nsteps': args.slice_jacobi_nsteps,
    'slice_jacobi_slice_ksp_type': 'preonly',
    # 'slice_jacobi_slice': jacobi_parameters,
    'slice_jacobi_slice': circulant_parameters,
}

# JacobiPC and SliceJacobiPC have disconnected blocks.
# Because the all-at-once jacobian is triangular it
# will take `nblocks` iterations for information to
# travel # the entire way across the solution vector.
# This means # we should allow at least as many gmres
# iterations as the number of blocks before restarting.
default_gmres_restart = 30  # PETSc default
if args.aaopc == "CirculantPC":
    nblocks = 1
elif args.aaopc == "JacobiPC":
    nblocks = window_length
elif args.aaopc == "SliceJacobiPC":
    nblocks = window_length//args.slice_jacobi_nsteps
gmres_restart = max(nblocks + 1, default_gmres_restart)

sparameters_diag = {
    'snes': {
        'linesearch_type': 'basic',
        'monitor': None,
        'converged_reason': None,
        'atol': patol,
        'rtol': 1e-10,
        'stol': 1e-12,
        'ksp_ew': None,
        'ksp_ew_version': 1,
        'ksp_ew_threshold': 1e-2,
    },
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp': {
        'monitor': None,
        'converged_rate': None,
        'rtol': 1e-5,
        'atol': patol,
        'gmres_restart': gmres_restart,
    },
}

if args.aaopc == 'JacobiPC':
   sparameters_diag.update(jacobi_parameters)
elif args.aaopc == 'CirculantPC':
   sparameters_diag.update(circulant_parameters)
elif args.aaopc == 'SliceJacobiPC':
   sparameters_diag.update(slice_jacobi_parameters)

create_mesh = partial(
    swe.create_mg_globe_mesh,
    ref_level=args.ref_level,
    base_level=args.base_level,
    coords_degree=1)  # remove coords degree once UFL issue with gradient of cell normals fixed

PETSc.Sys.Print('### === --- Calculating parallel solution --- === ###')

appctx = {'block_appctx': block_appctx}

miniapp = swe.ShallowWaterMiniApp(gravity=earth.Gravity,
                                  topography_expression=galewsky.topography_expression,
                                  velocity_expression=galewsky.velocity_expression,
                                  depth_expression=galewsky.depth_expression,
                                  reference_depth=galewsky.H0,
                                  reference_state=True,
                                  create_mesh=create_mesh,
                                  dt=dt, theta=0.5,
                                  time_partition=time_partition,
                                  appctx=appctx,
                                  paradiag_sparameters=sparameters_diag,
                                  file_name='output/'+args.filename,
                                  record_diagnostics={'cfl': args.record_cfl, 'file': False})

timer = SolverTimer()

fround = lambda x: round(float(x), 2)


def window_preproc(swe_app, pdg, wndw):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'### === --- Calculating time-window {wndw} --- === ###')
    PETSc.Sys.Print('')
    timer.start_timing()


def window_postproc(swe_app, pdg, wndw):
    timer.stop_timing()
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'Window solution time: {timer.times[-1]}')
    PETSc.Sys.Print('')

    if miniapp.layout.is_local(miniapp.save_step):
        nt = (pdg.total_windows - 1)*pdg.ntimesteps + (miniapp.save_step + 1)
        time = float(nt*pdg.aaoform.dt)
        comm = miniapp.ensemble.comm
        if args.record_cfl:
            PETSc.Sys.Print(f'Maximum CFL = {fround(swe_app.cfl_series[wndw])}', comm=comm)
        PETSc.Sys.Print(f'Hours = {fround(time/units.hour)}', comm=comm)
        PETSc.Sys.Print(f'Days = {fround(time/earth.day)}', comm=comm)
        PETSc.Sys.Print('', comm=comm)


miniapp.solve(nwindows=args.nwindows,
              preproc=window_preproc,
              postproc=window_postproc)

PETSc.Sys.Print('### === --- Iteration counts --- === ###')

paradiag = miniapp.paradiag
asQ.write_paradiag_metrics(paradiag, directory=args.metrics_dir)

PETSc.Sys.Print('')

nw = paradiag.total_windows
nt = paradiag.total_timesteps
PETSc.Sys.Print(f'windows: {nw}')
PETSc.Sys.Print(f'timesteps: {nt}')
PETSc.Sys.Print('')

lits = paradiag.linear_iterations
nlits = paradiag.nonlinear_iterations

if sparameters_diag['pc_python_type'] != 'asQ.SliceJacobiPC':
    blits = paradiag.block_iterations.data()
else:
    slice_solver = paradiag.solver.jacobian.pc.slice_solver
    blits = slice_solver.jacobian.pc.block_iterations
    blits.synchronise()
    blits = blits.data()

PETSc.Sys.Print(f'linear iterations: {lits} | iterations per window: {lits/nw}')
PETSc.Sys.Print(f'nonlinear iterations: {nlits} | iterations per window: {nlits/nw}')
PETSc.Sys.Print(f'block linear iterations: {blits} | iterations per block solve: {blits/lits}')
PETSc.Sys.Print('')

if args.record_cfl:
   PETSc.Sys.Print(f'Maximum CFL = {max(miniapp.cfl_series)}')
   PETSc.Sys.Print(f'Minimum CFL = {min(miniapp.cfl_series)}')
   PETSc.Sys.Print('')

mesh = miniapp.mesh
W = miniapp.W
PETSc.Sys.Print(f'DoFs per timestep: {W.dim()}')
PETSc.Sys.Print(f'Number of MPI ranks per timestep: {mesh.comm.size}')
PETSc.Sys.Print(f'DoFs/rank: {W.dim()/mesh.comm.size}')
PETSc.Sys.Print(f'Block DoFs/rank: {2*W.dim()/mesh.comm.size}')
PETSc.Sys.Print('')

if timer.ntimes() > 1:
    timer.times[0] = timer.times[1]

PETSc.Sys.Print(timer.string(timesteps_per_solve=window_length,
                             total_iterations=lits, ndigits=5))
PETSc.Sys.Print('')
