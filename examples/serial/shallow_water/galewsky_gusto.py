
import firedrake as fd
from firedrake.petsc import PETSc
import gusto

from utils import units
from utils import mg
from utils.planets import earth
import utils.shallow_water as swe
from utils.shallow_water import galewsky
from utils import diagnostics

from utils.serial import SerialMiniApp

from functools import partial

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


# shallow water equation forms
def form_function(u, h, v, q, t):
    return swe.nonlinear.form_function(mesh,
                                       gravity,
                                       topography,
                                       coriolis,
                                       u, h, v, q, t)


def form_mass(u, h, v, q):
    return swe.nonlinear.form_mass(mesh, u, h, v, q)

# gusto forms

swe_parameters = gusto.ShallowWaterParameters(H=galewsky.H0,
                                              g=earth.Gravity,
                                              Omega=earth.Omega)

domain = gusto.Domain(mesh, dt, 'BDM', degree=args.degree)

eqn = gusto.ShallowWaterEquations(domain,
                                  swe_parameters,
                                  fexpr=galewsky.coriolis_expression(*x))

from gusto.labels import replace_subject, replace_test_function, time_derivative, prognostic
from gusto.fml.form_manipulation_labelling import all_terms, drop
from gusto import Term
from firedrake.formmanipulation import split_form


def extract_form_mass(u, h, v, q, residual=None):
    M = residual.label_map(lambda t: t.has_label(time_derivative),
                           map_if_false=drop)

    Mu = M.label_map(lambda t: t.get(prognostic) == "u",
                     lambda t: Term(split_form(t.form)[0].form, t.labels),
                     drop)

    Mh = M.label_map(lambda t: t.get(prognostic) == "D",
                     lambda t: Term(split_form(t.form)[1].form, t.labels),
                     drop)

    Mu = Mu.label_map(all_terms, replace_test_function(v))
    Mu = Mu.label_map(all_terms, replace_subject(u, idx=0))

    Mh = Mh.label_map(all_terms, replace_test_function(q))
    Mh = Mh.label_map(all_terms, replace_subject(h, idx=1))

    M = Mu + Mh
    print("form_mass: ", M.form)
    return M.form


def extract_form_function(u, h, v, q, t, residual=None):
    K = residual.label_map(lambda t: t.has_label(time_derivative),
                           map_if_true=drop)

    Ku = K.label_map(lambda t: t.get(prognostic) == "u",
                     lambda t: Term(split_form(t.form)[0].form, t.labels),
                     drop)

    Kh = K.label_map(lambda t: t.get(prognostic) == "D",
                     lambda t: Term(split_form(t.form)[1].form, t.labels),
                     drop)

    Ku = Ku.label_map(all_terms, replace_test_function(v))
    Ku = Ku.label_map(all_terms, replace_subject(u, idx=0))

    Kh = Kh.label_map(all_terms, replace_test_function(q))
    Kh = Kh.label_map(all_terms, replace_subject(h, idx=1))

    K = Ku + Kh
    print("form_function: ", K.form)
    return K.form


form_mass_gusto = partial(extract_form_mass, residual=eqn.residual)
form_function_gusto = partial(extract_form_function, residual=eqn.residual)


# solver parameters for the implicit solve
atol = 1e-12
sparameters = {
    'snes': {
        'monitor': None,
        'converged_reason': None,
        'rtol': 1e-12,
        'atol': atol,
        'ksp_ew': None,
        'ksp_ew_version': 1,
    },
    #'mat_type': 'matfree',
    #'ksp_type': 'fgmres',
    'ksp': {
        'monitor': None,
        'converged_reason': None,
        'atol': atol,
        'rtol': 1e-5,
    },
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
#    'pc_type': 'mg',
#    'pc_mg_cycle_type': 'w',
#    'pc_mg_type': 'multiplicative',
#    'mg': {
#        'levels': {
#            'ksp_type': 'gmres',
#            'ksp_max_it': 5,
#            'pc_type': 'python',
#            'pc_python_type': 'firedrake.PatchPC',
#            'patch': {
#                'pc_patch_save_operators': True,
#                'pc_patch_partition_of_unity': True,
#                'pc_patch_sub_mat_type': 'seqdense',
#                'pc_patch_construct_dim': 0,
#                'pc_patch_construct_type': 'vanka',
#                'pc_patch_local_type': 'additive',
#                'pc_patch_precompute_element_tensors': True,
#                'pc_patch_symmetrise_sweep': False,
#                'sub_ksp_type': 'preonly',
#                'sub_pc_type': 'lu',
#                'sub_pc_factor_shift_type': 'nonzero',
#            },
#        },
#        'coarse': {
#            'pc_type': 'python',
#            'pc_python_type': 'firedrake.AssembledPC',
#            'assembled_pc_type': 'lu',
#            'assembled_pc_factor_mat_solver_type': 'mumps',
#        },
#    }
}

# set up nonlinear solver
miniapp = SerialMiniApp(dt, args.theta,
                        w_initial,
                        form_mass,
                        form_function_gusto,
                        sparameters)

miniapp.nlsolver.set_transfer_manager(
    mg.manifold_transfer_manager(W))

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
    ofile.write(uout, hout, potential_vorticity(uout), time=float(t))


miniapp.solve(args.nt,
              preproc=preproc,
              postproc=postproc)

PETSc.Sys.Print('### === --- Iteration counts --- === ###')
PETSc.Sys.Print('')

PETSc.Sys.Print(f'linear iterations: {linear_its} | iterations per timestep: {linear_its/args.nt}')
PETSc.Sys.Print(f'nonlinear iterations: {nonlinear_its} | iterations per timestep: {nonlinear_its/args.nt}')
PETSc.Sys.Print('')
