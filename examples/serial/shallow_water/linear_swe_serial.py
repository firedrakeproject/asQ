
import firedrake as fd
from firedrake.petsc import PETSc

from utils import units
from utils import mg
from utils.planets import earth
import utils.shallow_water as swe
import utils.shallow_water.gravity_bumps as case

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
parser.add_argument('--filename', type=str, default='swe', help='Name of output vtk files')
parser.add_argument('--show_args', action='store_true', default=True, help='Output all the arguments.')

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
coriolis = case.coriolis_expression(*x)


# initial conditions
w_initial = fd.Function(W)
u_initial, h_initial = w_initial.split()

u_initial.project(case.velocity_expression(*x))
h_initial.project(case.depth_expression(*x))


# shallow water equation forms
def form_function(u, h, v, q):
    return swe.linear.form_function(mesh,
                                    earth.Gravity, case.H,
                                    coriolis,
                                    u, h, v, q)


def form_mass(u, h, v, q):
    return swe.linear.form_mass(mesh, u, h, v, q)


patch_parameters = {
    'pc_patch_save_operators': True,
    'pc_patch_partition_of_unity': True,
    'pc_patch_sub_mat_type': 'seqdense',
    'pc_patch_construct_dim': 0,
    'pc_patch_construct_type': 'vanka',
    'pc_patch_local_type': 'additive',
    'pc_patch_precompute_element_tensors': True,
    'pc_patch_symmetrise_sweep': False,
    'sub_ksp_type': 'preonly',
    'sub_pc_type': 'lu',
    'sub_pc_factor_shift_type': 'nonzero',
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
        'assembled_pc_type': 'lu',
        'assembled_pc_factor_mat_solver_type': 'mumps',
    },
}

# solver parameters for the implicit solve
sparameters = {
    'snes_type': 'ksponly',
    'snes': {
        'monitor': None,
        'converged_reason': None,
        'atol': 1e-0,
        'rtol': 1e-10
    },
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp_rtol': 1e-10,
    'ksp': {
        'atol': 1e-0,
        'monitor': None,
        'converged_reason': None
    },
    'pc_type': 'mg',
    'pc_mg_cycle_type': 'v',
    'pc_mg_type': 'multiplicative',
    'mg': mg_parameters
}

# set up nonlinear solver
miniapp = SerialMiniApp(dt, args.theta,
                        w_initial,
                        form_mass,
                        form_function,
                        sparameters)

miniapp.nlsolver.set_transfer_manager(
    mg.manifold_transfer_manager(W))

PETSc.Sys.Print('### === --- Timestepping loop --- === ###')
linear_its = 0
nonlinear_its = 0

ofile = fd.File('output/'+args.filename+'.pvd')
uout = fd.Function(u_initial.function_space(), name='velocity')
hout = fd.Function(h_initial.function_space(), name='depth')

uout.assign(u_initial)
hout.assign(h_initial - case.H)
ofile.write(uout, hout, time=0)


def preproc(app, step, t):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'=== --- Timestep {step} --- ===')
    PETSc.Sys.Print('')


def postproc(app, step, t):
    global linear_its
    global nonlinear_its

    linear_its += app.nlsolver.snes.getLinearSolveIterations()
    nonlinear_its += app.nlsolver.snes.getIterationNumber()

    u, h = app.w0.split()
    uout.assign(u)
    hout.assign(h-case.H)
    ofile.write(uout, hout, time=t/units.hour)


miniapp.solve(args.nt,
              preproc=preproc,
              postproc=postproc)

PETSc.Sys.Print('### === --- Iteration counts --- === ###')
PETSc.Sys.Print('')

PETSc.Sys.Print(f'linear iterations: {linear_its} | iterations per timestep: {linear_its/args.nt}')
PETSc.Sys.Print(f'nonlinear iterations: {nonlinear_its} | iterations per timestep: {nonlinear_its/args.nt}')
PETSc.Sys.Print('')
