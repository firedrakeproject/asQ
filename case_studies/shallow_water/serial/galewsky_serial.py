
import firedrake as fd
from firedrake.petsc import PETSc

from utils.timing import SolverTimer
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

parser.add_argument('--ref_level', type=int, default=3, help='Refinement level of icosahedral grid.')
parser.add_argument('--base_level', type=int, default=1, help='Refinement level of icosahedral grid.')
parser.add_argument('--nt', type=int, default=10, help='Number of time steps.')
parser.add_argument('--dt', type=float, default=0.5, help='Timestep in hours.')
parser.add_argument('--degree', type=float, default=swe.default_degree(), help='Degree of the depth function space.')
parser.add_argument('--theta', type=float, default=0.5, help='Parameter for implicit theta method. 0.5 for trapezium rule, 1 for backwards Euler.')
parser.add_argument('--atol', type=float, default=1e0, help='atol of each timestep.')
parser.add_argument('--write_file', action='store_true', help='Write solution to VTK file.')
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
mesh = swe.create_mg_globe_mesh(ref_level=args.ref_level,
                                base_level=args.base_level,
                                coords_degree=1)
x = fd.SpatialCoordinate(mesh)

# time step
dt = args.dt*units.hour

# shallow water equation function spaces (velocity and depth)
W = swe.default_function_space(mesh, degree=args.degree)

# parameters
g = earth.Gravity

b = galewsky.topography_expression(*x)
f = swe.earth_coriolis_expression(*x)

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
H = galewsky.H0


# shallow water equation forms
def form_function(u, h, v, q, t):
    return swe.nonlinear.form_function(mesh, g, b, f,
                                       u, h, v, q, t)


def form_mass(u, h, v, q):
    return swe.nonlinear.form_mass(mesh, u, h, v, q)


def aux_form_function(u, h, v, q, t):
    return swe.linear.form_function(mesh, g, H, f,
                                    u, h, v, q, t)


appctx = {'aux_form_function': aux_form_function}


# solver parameters for the implicit solve
factorisation_params = {
    'ksp_type': 'preonly',
    # 'pc_factor_mat_ordering_type': 'rcm',
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
    "hybridization": lu_params
    # "hybridization": gamg_sparams
}

aux_sparams = {
    "mat_type": "matfree",
    "pc_type": "python",
    "pc_python_type": "utils.serial.AuxiliarySerialPC",
    "aux": lu_params
    # "aux": mg_sparams
    # "aux": hybridization_sparams
}


sparameters = {
    'snes': {
        'monitor': None,
        'converged_reason': None,
        'rtol': 1e-12,
        'atol': args.atol,
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
        'atol': args.atol,
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

if args.write_file:
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

timer = SolverTimer()


def preproc(app, step, t):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'=== --- Timestep {step} --- ===')
    PETSc.Sys.Print('')
    timer.start_timing()


def postproc(app, step, t):
    timer.stop_timing()
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'Timestep solution time: {timer.times[-1]}')
    PETSc.Sys.Print('')

    global linear_its
    global nonlinear_its

    linear_its += app.nlsolver.snes.getLinearSolveIterations()
    nonlinear_its += app.nlsolver.snes.getIterationNumber()

    if args.write_file:
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

W = miniapp.function_space
PETSc.Sys.Print(f'DoFs per timestep: {W.dim()}')
PETSc.Sys.Print(f'Number of MPI ranks per timestep: {mesh.comm.size}')
PETSc.Sys.Print(f'DoFs/rank: {W.dim()/mesh.comm.size}')
PETSc.Sys.Print('')

if timer.ntimes() > 1:
    timer.times[0] = timer.times[1]

PETSc.Sys.Print(timer.string(timesteps_per_solve=1,
                             total_iterations=linear_its, ndigits=5))
PETSc.Sys.Print('')
