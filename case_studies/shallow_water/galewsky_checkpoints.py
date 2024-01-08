
import firedrake as fd
from firedrake.petsc import PETSc

from utils import units
from utils import mg
from utils.misc import function_mean
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
parser.add_argument('--nt', type=int, default=10, help='Number of time steps.')
parser.add_argument('--dt', type=float, default=0.5, help='Timestep in hours.')
parser.add_argument('--degree', type=float, default=swe.default_degree(), help='Degree of the depth function space.')
parser.add_argument('--theta', type=float, default=0.5, help='Parameter for implicit theta method. 0.5 for trapezium rule, 1 for backwards Euler.')
parser.add_argument('--filename', type=str, default='galewsky_series', help='Name of checkpoint file.')
parser.add_argument('--save_freq', type=int, default=12, help='How many timesteps between each checkpoint.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

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
gravity = earth.Gravity

topography_expr = galewsky.topography_expression(*x)
coriolis_expr = swe.earth_coriolis_expression(*x)
topography = fd.Function(Vh, name="topography").project(topography_expr)
coriolis = fd.Function(Vh, name="coriolis").project(coriolis_expr)

# initial conditions
w_initial = fd.Function(W)
u_initial = w_initial.subfunctions[0]
h_initial = w_initial.subfunctions[1]

u_initial.project(galewsky.velocity_expression(*x))
h_initial.project(galewsky.depth_expression(*x))

# current and next timestep
w0 = fd.Function(W).assign(w_initial)
w1 = fd.Function(W).assign(w_initial)


# mean height
H = function_mean(h_initial)


# shallow water equation forms
def form_mass(u, h, v, q):
    return swe.nonlinear.form_mass(mesh, u, h, v, q)


def form_function(u, h, v, q, t):
    return swe.nonlinear.form_function(mesh,
                                       gravity,
                                       topography,
                                       coriolis,
                                       u, h, v, q, t)


def aux_form_function(u, h, v, q, t):
    return swe.linear.form_function(mesh,
                                    gravity, H,
                                    coriolis,
                                    u, h, v, q, t)


appctx = {'aux_form_function': aux_form_function}


# solver parameters for the implicit solve
factorisation_params = {
    'ksp_type': 'preonly',
    'pc_factor_mat_ordering_type': 'rcm',
    'pc_factor_reuse_ordering': None,
    'pc_factor_reuse_fill': None,
}

lu_params = {'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}
lu_params.update(factorisation_params)

icc_params = {'pc_type': 'icc'}
icc_params.update(factorisation_params)

gamg_sparams = {
    'ksp_type': 'preonly',
    'pc_type': 'gamg',
    'pc_gamg_sym_graph': None,
    'pc_mg_type': 'multiplicative',
    'pc_mg_cycle_type': 'v',
    'mg': {
        'levels': {
            'ksp_type': 'cg',
            'ksp_max_it': 3,
            'pc_type': 'bjacobi',
            'sub': icc_params,
        },
        'coarse': lu_params
    }
}

hybridization_sparams = {
    "mat_type": "matfree",
    "pc_type": "python",
    "pc_python_type": "firedrake.HybridizationPC",
    "hybridization": gamg_sparams
}

aux_sparams = {
    "mat_type": "matfree",
    "pc_type": "python",
    "pc_python_type": "utils.serial.AuxiliarySerialPC",
    # "aux": hybridization_sparams
    "aux": lu_params
}


atol = 1e0
sparameters = {
    'snes': {
        # 'monitor': None,
        'converged_reason': None,
        'rtol': 1e-12,
        'atol': atol,
        'ksp_ew': None,
        'ksp_ew_version': 1,
        'ksp_ew_threshold': 1e-5,
        'ksp_ew_rtol0': 1e-3,
        'lag_preconditioner': -2,
        'lag_preconditioner_persists': None,
    },
    'ksp_type': 'fgmres',
    'ksp': {
        # 'monitor': None,
        # 'converged_rate': None,
        'atol': atol,
        'rtol': 1e-5,
    },
}
sparameters.update(aux_sparams)

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
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'=== --- Timestep {step} --- ===')


wout = fd.Function(W, name="swe").assign(w_initial)
checkpoint = fd.CheckpointFile(f"{args.filename}.h5", 'w')
checkpoint.save_mesh(mesh)
checkpoint.save_function(topography)
checkpoint.save_function(coriolis)
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
