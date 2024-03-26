
import firedrake as fd
from firedrake.petsc import PETSc

from firedrake.output import VTKFile

from utils import units
from utils.planets import earth
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
g = earth.Gravity
H = gcase.H
f = gcase.coriolis_expression(*x)

# initial conditions
w_initial = fd.Function(W)
u_initial, h_initial = w_initial.subfunctions

u_initial.project(gcase.velocity_expression(*x))
h_initial.project(gcase.depth_expression(*x))

# broken space
Vu, Vh = W.subfunctions

Vub = fd.FunctionSpace(mesh, fd.BrokenElement(Vu.ufl_element()))
Tr = fd.FunctionSpace(mesh, "HDivT", Vu.ufl_element().degree())
Wtr = Vub*Vh*Tr


# shallow water equation forms
def form_function(u, h, v, q, t):
    return swe.linear.form_function(mesh, g, H, f,
                                    u, h, v, q, t)


def form_mass(u, h, v, q):
    return swe.linear.form_mass(mesh, u, h, v, q)


# solver parameters for the implicit solve

lu_params = {
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps'
}


scpc_sparams = {
    "mat_type": "matfree",
    "pc_type": "python",
    "pc_python_type": f"{__name__}.HybridisedSCPC",
    "hybridscpc_condensed_field": lu_params
}

atol = 1e2
sparameters = {
    'snes': {
        'atol': atol,
        'rtol': 1e-10,
        'stol': 1e-12,
    },
    'ksp_type': 'preonly',
    'ksp': {
        'atol': atol,
        'rtol': 1e-10,
        'stol': 1e-12,
        'monitor': None,
        'converged_rate': None
    },
}
sparameters.update(scpc_sparams)

# appctx = {'broken_space': Vub}

# set up nonlinear solver
miniapp = SerialMiniApp(dt, args.theta,
                        w_initial,
                        form_mass,
                        form_function,
                        sparameters)

PETSc.Sys.Print('### === --- Timestepping loop --- === ###')
linear_its = 0
nonlinear_its = 0

ofile = VTKFile('output/'+args.filename+'.pvd')
uout = fd.Function(u_initial.function_space(), name='velocity')
hout = fd.Function(h_initial.function_space(), name='depth')

uout.assign(u_initial)
hout.assign(h_initial - gcase.H)
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

    u, h = app.w0.subfunctions
    uout.assign(u)
    hout.assign(h-gcase.H)
    ofile.write(uout, hout, time=t/units.hour)


miniapp.solve(args.nt,
              preproc=preproc,
              postproc=postproc)

PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Iteration counts --- === ###')
PETSc.Sys.Print('')

PETSc.Sys.Print(f'linear iterations: {linear_its} | iterations per timestep: {linear_its/args.nt}')
PETSc.Sys.Print(f'nonlinear iterations: {nonlinear_its} | iterations per timestep: {nonlinear_its/args.nt}')
PETSc.Sys.Print('')
