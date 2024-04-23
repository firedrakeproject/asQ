
import firedrake as fd
from firedrake.petsc import PETSc

from utils import units
from utils import mg
from utils.planets import earth
import utils.shallow_water as swe
from utils.shallow_water import galewsky

from utils.serial import SerialMiniApp

PETSc.Sys.popErrorHandler()

# get command arguments
import argparse
parser = argparse.ArgumentParser(
    description='Galewsky testcase using fully implicit SWE solver.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--ref_level', type=int, default=2, help='Refinement level of icosahedral grid.')
parser.add_argument('--nt', type=int, default=48, help='Number of time steps.')
parser.add_argument('--dt', type=float, default=0.5, help='Timestep in hours.')
parser.add_argument('--degree', type=float, default=swe.default_degree(), help='Degree of the depth function space.')
parser.add_argument('--theta', type=float, default=0.5, help='Parameter for implicit theta method. 0.5 for trapezium rule, 1 for backwards Euler.')
parser.add_argument('--atol', type=float, default=1e0, help='Absolute tolerance for solution of each timestep.')
parser.add_argument('--filename', type=str, default='hdf5/galewsky_series', help='Name of checkpoint file.')
parser.add_argument('--save_freq', type=int, default=12, help='How many timesteps between each checkpoint.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')
parser.add_argument('--verbose', '-v', action='store_true', help='Print SNES and KSP outputs.')

args = parser.parse_known_args()
args = args[0]

nt = args.nt
degree = args.degree

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
Vu, Vh = W.subfunctions

# parameters
g = earth.Gravity

topography_expr = galewsky.topography_expression(*x)
coriolis_expr = swe.earth_coriolis_expression(*x)
b = fd.Function(Vh, name="topography").project(topography_expr)
f = fd.Function(Vh, name="coriolis").project(coriolis_expr)

# initial conditions
w_initial = fd.Function(W)
u_initial, h_initial = w_initial.subfunctions

u_initial.project(galewsky.velocity_expression(*x))
h_initial.project(galewsky.depth_expression(*x))

# current and next timestep
w0 = fd.Function(W).assign(w_initial)
w1 = fd.Function(W).assign(w_initial)

# mean height
H = galewsky.H0


# shallow water equation forms
def form_mass(u, h, v, q):
    return swe.nonlinear.form_mass(mesh, u, h, v, q)


def form_function(u, h, v, q, t):
    return swe.nonlinear.form_function(mesh, g, b, f,
                                       u, h, v, q, t)


def aux_form_function(u, h, v, q, t):
    return swe.linear.form_function(mesh, g, H, f,
                                    u, h, v, q, t)


appctx = {'aux_form_function': aux_form_function}


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
    'pc_factor': {
        'mat_solver_type': 'mumps',
        'reuse_ordering': None,
        'reuse_fill': None,
    }
}

hybridization_sparams = {
    "mat_type": "matfree",
    "pc_type": "python",
    "pc_python_type": "firedrake.HybridizationPC",
    "hybridization": lu_params,
    "hybridization_snes": linear_snes_params
}

aux_sparams = {
    "mat_type": "matfree",
    "pc_type": "python",
    "pc_python_type": "asQ.AuxiliaryRealBlockPC",
    "aux": hybridization_sparams,
    "aux_snes": linear_snes_params
}


sparameters = {
    'snes': {
        'rtol': 1e-12,
        'atol': args.atol,
        'ksp_ew': None,
        'ksp_ew_version': 1,
        'ksp_ew_threshold': 1e-5,
        'ksp_ew_rtol0': 1e-2,
        'lag_preconditioner': -2,
        'lag_preconditioner_persists': None,
    },
    'ksp_type': 'fgmres',
    'ksp': {
        'atol': args.atol,
        'rtol': 1e-5,
    },
}
sparameters.update(aux_sparams)

if args.verbose:
    sparameters['snes_monitor'] = None
    sparameters['snes_converged_reason'] = None
    sparameters['ksp_monitor'] = None
    sparameters['ksp_converged_rate'] = None

# set up nonlinear solver
miniapp = SerialMiniApp(dt, args.theta,
                        w_initial,
                        form_mass,
                        form_function,
                        sparameters,
                        appctx=appctx)

miniapp.nlsolver.set_transfer_manager(
    mg.ManifoldTransferManager())

# save initial conditions

PETSc.Sys.Print('### === --- Timestepping loop --- === ###')


def preproc(app, step, t):
    if args.verbose:
        PETSc.Sys.Print('')
    PETSc.Sys.Print(f'=== --- Timestep {step} --- ===')


wout = fd.Function(W, name="swe").assign(w_initial)
checkpoint = fd.CheckpointFile(f"{args.filename}.h5", 'w')
checkpoint.save_mesh(mesh)
checkpoint.save_function(b)
checkpoint.save_function(f)
idx = 0
checkpoint.save_function(wout, idx=idx)
idx += 1


def postproc(app, step, t):
    global idx
    if ((step+1) % args.save_freq) == 0:
        wout.assign(app.w1)
        checkpoint.save_function(wout, idx=idx)
        idx += 1


miniapp.solve(args.nt,
              preproc=preproc,
              postproc=postproc)

checkpoint.close()
