from firedrake import *
from firedrake.petsc import PETSc
import asQ

Print = PETSc.Sys.Print

time_partition = [4]
ensemble = asQ.create_ensemble(time_partition)

nx = 32
nu = 1

dy = 1/32

cfl = 1
dt = cfl*dy*dy/nu

mesh = UnitSquareMesh(nx=nx, ny=nx,
                      comm=ensemble.comm)
x, y = SpatialCoordinate(mesh)

V = FunctionSpace(mesh, "CG", 1)
uinitial = Function(V)
uinitial.project(sin(x) + cos(y))

bcs = [DirichletBC(V, 0, 1)]

# space-dependent diffusivity
mu = Function(V).interpolate(nu*(1 - 0.2*cos(2*y + 1) + 0.3*sin(1.5*x - 0.4)))


def form_mass(u, v):
    return u*v*dx


# the problem we're solving uses the variable diffusivity
def form_function(u, v, t):
    return inner(mu*grad(u), grad(v))*dx


# the blocks are preconditioned with constant diffusivity
def aux_form_function(u, v, t):
    return inner(nu*grad(u), grad(v))*dx


lu_parameters = {
    'ksp_type': 'preonly',
    'pc_type': 'lu'
}

aux_parameters = {
    'ksp_type': 'gmres',
    'pc_type': 'python',
    'pc_python_type': 'asQ.AuxiliaryComplexBlockPC',
    'aux': {
        'pc_type': 'ilu'
    }
}

# give the alternative form to the auxiliary pc through the appctx
block_appctx = {
    'aux_form_function': aux_form_function
}

block_parameters = aux_parameters

solver_parameters = {
    'snes_type': 'ksponly',
    'ksp': {
        'monitor': None,
        'converged_rate': None,
        'rtol': 1e-8,
    },
    'mat_type': 'matfree',
    'ksp_type': 'richardson',
    'ksp_norm_type': 'unpreconditioned',
    'pc_type': 'python',
    'pc_python_type': 'asQ.CirculantPC',
    'circulant_alpha': 1e-3,
    'circulant_block': block_parameters
}

appctx = {'block_appctx': block_appctx}

paradiag = asQ.Paradiag(
    ensemble=ensemble,
    form_mass=form_mass,
    form_function=form_function,
    ics=uinitial,
    dt=dt, theta=1.0,
    time_partition=time_partition,
    appctx=appctx,
    solver_parameters=solver_parameters)


def preproc(pdg, window, rhs):
    Print(f"\n=== --- Solving window {window} --- ===")


paradiag.solve(nwindows=2,
               preproc=preproc)

block_iterations = paradiag.block_iterations
PETSc.Sys.Print('')
PETSc.Sys.Print(f'block linear iterations: {block_iterations.data()}')
PETSc.Sys.Print(f'iterations per block solve: {block_iterations.data()/paradiag.linear_iterations}')
