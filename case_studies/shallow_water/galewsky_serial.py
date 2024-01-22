
import firedrake as fd
from firedrake.petsc import PETSc

from utils import units
from utils import mg
from utils.misc import function_mean
from utils.planets import earth
import utils.shallow_water as swe
from utils.shallow_water import galewsky
from utils import diagnostics

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
parser.add_argument('--filename', type=str, default='galewsky', help='Name of output vtk files')
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

# parameters
gravity = earth.Gravity

topography = galewsky.topography_expression(*x)
coriolis = swe.earth_coriolis_expression(*x)

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
def form_function(u, h, v, q, t):
    return swe.nonlinear.form_function(mesh, gravity,
                                       topography, coriolis,
                                       u, h, v, q, t)


def form_mass(u, h, v, q):
    return swe.nonlinear.form_mass(mesh, u, h, v, q)


def aux_form_function(u, h, v, q, t):
    # Ku = swe.nonlinear.form_function_velocity(
    #     mesh, gravity, topography, coriolis, u, h, v, t, perp=fd.cross)

    Ku = swe.linear.form_function_u(
        mesh, gravity, coriolis, u, h, v, t)

    Kh = swe.linear.form_function_h(
        mesh, H, u, h, q, t)

    return Ku + Kh


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

ilu_params = {'pc_type': 'ilu'}
ilu_params.update(factorisation_params)

icc_params = {'pc_type': 'icc'}
icc_params.update(factorisation_params)

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
        'ksp_max_it': 5,
        'pc_type': 'python',
        'pc_python_type': 'firedrake.PatchPC',
        'patch': patch_parameters
    },
    'coarse': {
        'pc_type': 'python',
        'pc_python_type': 'firedrake.AssembledPC',
        'assembled': lu_params
    },
}

mg_sparams = {
    "mat_type": "matfree",
    'pc_type': 'mg',
    'pc_mg_cycle_type': 'w',
    'pc_mg_type': 'multiplicative',
    'mg': mg_parameters
}

gamg_sparams = {
    'ksp_type': 'preonly',
    # 'ksp_rtol': 1e-4,
    'pc_type': 'gamg',
    'pc_gamg_sym_graph': None,
    'pc_mg_type': 'full',
    'pc_mg_cycle_type': 'v',
    'mg': {
        'levels': {
            'ksp_type': 'cg',
            'ksp_max_it': 2,
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
    # "hybridization": lu_params
    "hybridization": gamg_sparams
}

aux_sparams = {
    "mat_type": "matfree",
    "pc_type": "python",
    "pc_python_type": "utils.serial.AuxiliarySerialPC",
    # "aux": lu_params
    # "aux": mg_sparams
    "aux": hybridization_sparams
}


atol = 1e2
sparameters = {
    'snes': {
        'monitor': None,
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
        'monitor': None,
        'converged_rate': None,
        'atol': atol,
        'rtol': 1e-5,
    },
}

# sparameters.update(lu_params)
# sparameters.update(mg_sparams)
# sparameters.update(hybridization_sparams)
sparameters.update(aux_sparams)

# from json import dumps as dump_json
# PETSc.Sys.Print("sparameters =")
# PETSc.Sys.Print(dump_json(sparameters, indent=4))

# set up nonlinear solver
miniapp = SerialMiniApp(dt, args.theta,
                        w_initial,
                        form_mass,
                        form_function,
                        sparameters,
                        appctx=appctx)

miniapp.nlsolver.set_transfer_manager(
    mg.ManifoldTransferManager())

potential_vorticity = diagnostics.potential_vorticity_calculator(
    u_initial.function_space(), name='vorticity')

uout = fd.Function(u_initial.function_space(), name='velocity')
hout = fd.Function(h_initial.function_space(), name='elevation')
ofile = fd.File(f"output/{args.filename}.pvd")
# save initial conditions
uout.assign(u_initial)
hout.assign(h_initial)
ofile.write(uout, hout, potential_vorticity(uout), time=0)

PETSc.Sys.Print('### === --- Timestepping loop --- === ###')
linear_its = 0
nonlinear_its = 0


def preproc(app, step, t):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'=== --- Timestep {step} --- ===')
    PETSc.Sys.Print('')


def postproc(app, step, t):
    global linear_its
    global nonlinear_its

    linear_its += app.nlsolver.snes.getLinearSolveIterations()
    nonlinear_its += app.nlsolver.snes.getIterationNumber()

    uout.assign(miniapp.w0.subfunctions[0])
    hout.assign(miniapp.w0.subfunctions[1])
    ofile.write(uout, hout, potential_vorticity(uout), time=t)


miniapp.solve(args.nt,
              preproc=preproc,
              postproc=postproc)

PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Iteration counts --- === ###')
PETSc.Sys.Print('')

PETSc.Sys.Print(f'linear iterations: {linear_its} | iterations per timestep: {linear_its/args.nt}')
PETSc.Sys.Print(f'nonlinear iterations: {nonlinear_its} | iterations per timestep: {nonlinear_its/args.nt}')
PETSc.Sys.Print('')
