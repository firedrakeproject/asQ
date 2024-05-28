
import firedrake as fd
from firedrake.petsc import PETSc

from firedrake.output import VTKFile

from utils import units
from utils.planets import earth
from utils.timing import SolverTimer
import utils.shallow_water as swe
import utils.shallow_water.gravity_bumps as gcase
from utils.hybridisation import HybridisedSCPC  # noqa: F401

from utils.serial import SerialMiniApp

PETSc.Sys.popErrorHandler()

# get command arguments
import argparse
parser = argparse.ArgumentParser(
    description='Schreiber & Loft testcase using fully implicit linear SWE solver.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--ref_level', type=int, default=3, help='Refinement level of icosahedral grid.')
parser.add_argument('--nt', type=int, default=20, help='Number of time steps.')
parser.add_argument('--dt', type=float, default=0.05, help='Timestep in hours.')
parser.add_argument('--degree', type=float, default=swe.default_degree(), help='Degree of the depth function space.')
parser.add_argument('--theta', type=float, default=0.5, help='Parameter for implicit theta method. 0.5 for trapezium rule, 1 for backwards Euler.')
parser.add_argument('--method', type=str, default='mg', choices=['lu', 'mg', 'hybr', 'scpc'], help='Preconditioning method to use.')
parser.add_argument('--filename', type=str, default='swe', help='Name of output vtk files')
parser.add_argument('--write_file', action='store_true', help='Write each timestep to file.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Setting up --- === ###')
PETSc.Sys.Print('')

# icosahedral mg mesh
mesh = swe.create_mg_globe_mesh(ref_level=args.ref_level, coords_degree=1)
x = fd.SpatialCoordinate(mesh)

# time step
dt = args.dt*units.hour

# shallow water equation function spaces (velocity and depth)
W = swe.default_function_space(mesh, degree=args.degree)

# parameters
g = earth.Gravity
H = gcase.H
f = gcase.coriolis_expression(*x)

# initial conditions
w_initial = fd.Function(W)
u_initial, h_initial = w_initial.subfunctions

u_initial.project(gcase.velocity_expression(*x))
h_initial.project(gcase.depth_expression(*x))


# shallow water equation forms
def form_mass(u, h, v, q):
    return swe.linear.form_mass(mesh, u, h, v, q)


def form_function(u, h, v, q, t):
    return swe.linear.form_function(mesh, g, H, f,
                                    u, h, v, q, t)


# solver parameters for the implicit solve

linear_snes_params = {
    'lag_jacobian': -2,
    'lag_jacobian_persists': None,
    'lag_preconditioner': -2,
    'lag_preconditioner_persists': None,
}

lu_params = {
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
    'pc_factor_reuse_ordering': None,
    'pc_factor_reuse_fill': None,
}

from utils.mg import ManifoldTransferManager  # noqa: F401
mg_sparameters = {
    'mat_type': 'matfree',
    'pc_type': 'mg',
    'pc_mg_cycle_type': 'v',
    'pc_mg_type': 'multiplicative',
    'mg': {
        'transfer_manager': f'{__name__}.ManifoldTransferManager',
        'levels': {
            'ksp_type': 'gmres',
            'ksp_max_it': 3,
            'pc_type': 'python',
            'pc_python_type': 'firedrake.PatchPC',
            'patch': {
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
        },
        'coarse': {
            'pc_type': 'python',
            'pc_python_type': 'firedrake.AssembledPC',
            'assembled': lu_params
        },
    }
}

trace_params = lu_params

hybridization_sparams = {
    "mat_type": "matfree",
    "pc_type": "python",
    "pc_python_type": "firedrake.HybridizationPC",
    "hybridization": trace_params
}

scpc_sparams = {
    "mat_type": "matfree",
    "pc_type": "python",
    "pc_python_type": f"{__name__}.HybridisedSCPC",
    "hybridscpc_condensed_field": lu_params,
}

atol = 1e0
sparameters = {
    'snes': linear_snes_params,
    'snes_type': 'ksponly',
    'ksp_type': 'fgmres',
    'ksp': {
        'atol': atol,
        'rtol': 1e-10,
        'stol': 1e-12,
        'monitor': None,
        'converged_rate': None
    },
}
if args.method == 'lu':
    sparameters.update(lu_params)
elif args.method == 'mg':
    sparameters.update(mg_sparameters)
elif args.method == 'hybr':
    sparameters.update(hybridization_sparams)
elif args.method == 'scpc':
    sparameters.update(hybridization_sparams)

# set up nonlinear solver
miniapp = SerialMiniApp(dt, args.theta, w_initial,
                        form_mass, form_function,
                        sparameters)

PETSc.Sys.Print('### === --- Timestepping loop --- === ###')
linear_its = 0
nonlinear_its = 0

if args.write_file:
    ofile = VTKFile('output/'+args.filename+'.pvd')
    uout = fd.Function(u_initial.function_space(), name='velocity')
    hout = fd.Function(h_initial.function_space(), name='depth')

    uout.assign(u_initial)
    hout.assign(h_initial - gcase.H)
    ofile.write(uout, hout, time=0)

timer = SolverTimer()


def preproc(app, step, t):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'=== --- Timestep {step} --- ===')
    PETSc.Sys.Print('')
    timer.start_timing()


def postproc(app, step, t):
    global linear_its, nonlinear_its
    timer.stop_timing()
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'Timestep solution time: {timer.times[-1]}')
    PETSc.Sys.Print('')

    linear_its += app.nlsolver.snes.getLinearSolveIterations()
    nonlinear_its += app.nlsolver.snes.getIterationNumber()

    if args.write_file:
        u, h = app.w0.subfunctions
        uout.assign(u)
        hout.assign(h-gcase.H)
        ofile.write(uout, hout, time=t/units.hour)


with PETSc.Log.Event("warmup_solve"):
    miniapp.solve(nt=1,
                  preproc=preproc,
                  postproc=postproc)

miniapp.w0.assign(w_initial)
miniapp.w1.assign(w_initial)
linear_its -= linear_its
nonlinear_its -= nonlinear_its

with PETSc.Log.Event("timed_solves"):
    miniapp.solve(nt=args.nt,
                  preproc=preproc,
                  postproc=postproc)

PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Iteration counts --- === ###')
PETSc.Sys.Print('')

PETSc.Sys.Print(f'linear iterations: {linear_its} | iterations per timestep: {linear_its/args.nt}')
PETSc.Sys.Print(f'nonlinear iterations: {nonlinear_its} | iterations per timestep: {nonlinear_its/args.nt}')
PETSc.Sys.Print('')

PETSc.Sys.Print(f'DoFs per timestep: {W.dim()}')
PETSc.Sys.Print(f'Number of MPI ranks per timestep: {mesh.comm.size}')
PETSc.Sys.Print(f'DoFs/rank: {W.dim()/mesh.comm.size}')
PETSc.Sys.Print('')

if timer.ntimes() > 1:
    timer.times = timer.times[1:]

PETSc.Sys.Print(timer.string(timesteps_per_solve=1,
                             total_iterations=linear_its, ndigits=5))
PETSc.Sys.Print('')
