
import firedrake as fd
from firedrake.petsc import PETSc
Print = PETSc.Sys.Print

from utils.serial import SerialMiniApp

T = 0.5
nt = 64
nx = 16
cfl = 1.
theta = 0.5
nu = 1.

dx = 1./nx
dt = T/nt

cfl = nu*dt/(dx*dx)
Print(cfl, dt)
verbose = False

mesh = fd.UnitSquareMesh(nx, nx, quadrilateral=True)
x, y = fd.SpatialCoordinate(mesh)

V = fd.FunctionSpace(mesh, "CG", 1)

Nu = fd.Constant(nu)


def form_mass(u, v):
    return fd.inner(u, v)*fd.dx


def form_function(u, v):
    return Nu*fd.inner(fd.grad(u), fd.grad(v))*fd.dx


initial_expr = (1+8*fd.pi*fd.pi)*fd.cos(x*fd.pi*2)*fd.cos(y*fd.pi*2)
uinitial = fd.Function(V).interpolate(initial_expr)

sparameters = {
    'snes_type': 'ksponly',
    'snes': {
        # 'monitor': None,
        # 'converged_reason': None,
        'rtol': 1e-8,
    },
    'ksp': {
        # 'monitor': None,
        # 'converged_reason': None,
        'rtol': 1e-8,
    },
    'ksp_type': 'preonly',
    'pc_type': 'lu',
}

if verbose:
    sparameters['snes']['monitor'] = None

miniapp = SerialMiniApp(dt, theta,
                        uinitial,
                        form_mass, form_function,
                        sparameters)

Print('### === --- Timestepping loop --- === ###')
linear_its = 0
nonlinear_its = 0

ofile = fd.File("output/heat.pvd")
ofile.write(miniapp.w0, time=0)


def preproc(app, step, t):
    if verbose:
        Print('')
        Print(f'=== --- Timestep {step} --- ===')
        Print('')


def postproc(app, step, t):
    global linear_its
    global nonlinear_its

    linear_its += app.nlsolver.snes.getLinearSolveIterations()
    nonlinear_its += app.nlsolver.snes.getIterationNumber()

    ofile.write(app.w0, time=t)


miniapp.solve(nt,
              preproc=preproc,
              postproc=postproc)

Print('')
Print('### === --- Iteration counts --- === ###')
Print('')

Print(f'linear iterations: {linear_its} | iterations per timestep: {linear_its/nt}')
Print(f'nonlinear iterations: {nonlinear_its} | iterations per timestep: {nonlinear_its/nt}')
Print('')
