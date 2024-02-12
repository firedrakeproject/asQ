from math import sqrt
import firedrake as fd
from firedrake.petsc import PETSc
Print = PETSc.Sys.Print

from utils.serial import SerialMiniApp

T = 0.5
nt = 128
nx = 16
cfl = 1.5
theta = 0.5
nu = 0.1

dx = 1./nx
dtf = T/nt

cfl = nu*dtf/(dx*dx)
Print(cfl, dtf)
verbose = False

ntf = 16
ntc = nt//ntf
nits = ntc

dtc = ntf*dtf

mesh = fd.UnitSquareMesh(nx, nx, quadrilateral=True)
x, y = fd.SpatialCoordinate(mesh)

V = fd.FunctionSpace(mesh, "CG", 1)

Nu = fd.Constant(nu)


def form_mass(u, v):
    return fd.inner(u, v)*fd.dx


def form_function(u, v, t=None):
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

miniapp = SerialMiniApp(dtf, theta,
                        uinitial,
                        form_mass, form_function,
                        sparameters)

# ## define F and G


def G(u, uout, **kwargs):
    miniapp.dt.assign(dtc)
    miniapp.w0.assign(u)
    miniapp.solve(1, **kwargs)
    uout.assign(miniapp.w0)


def F(u, uout, **kwargs):
    miniapp.dt.assign(dtf)
    miniapp.w0.assign(u)
    miniapp.solve(ntf, **kwargs)
    uout.assign(miniapp.w0)


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


def coarse_series():
    return [fd.Function(V) for _ in range(ntc+1)]


def copy_series(dst, src):
    for d, s in zip(dst, src):
        d.assign(s)


def series_norm(series):
    norm = 0.
    for s in series:
        sn = fd.norm(s)
        norm += sn*sn
    return sqrt(norm)


def series_error(exact, series):
    norm = 0.
    for e, s in zip(exact, series):
        en = fd.errornorm(e, s)
        norm += en*en
    return sqrt(norm)


# ## find "exact" fine solution at coarse points in serial
userial = coarse_series()
userial[0].assign(uinitial)

for i in range(ntc):
    F(userial[i], userial[i+1])

# ## initialise coarse points

Gk = coarse_series()
Uk = coarse_series()
Fk = coarse_series()

Gk1 = coarse_series()
Uk1 = coarse_series()

Gk[0].assign(uinitial)
Uk[0].assign(uinitial)
Fk[0].assign(uinitial)

Gk1[0].assign(uinitial)
Uk1[0].assign(uinitial)

for i in range(ntc):
    G(Gk1[i], Gk1[i+1])

copy_series(Uk1, Gk1)


# ## parareal iterations

for it in range(nits):
    copy_series(Uk, Uk1)
    copy_series(Gk, Gk1)

    for i in range(ntc):
        F(Uk[i], Fk[i+1])

    for i in range(ntc):
        G(Uk1[i], Gk1[i+1])
        Uk1[i+1].assign(Fk[i+1] + Gk1[i+1] - Gk[i+1])

    res = series_error(Uk, Uk1)
    err = series_error(userial, Uk1)
    Print(f"{str(it).ljust(3)} | {err:.5e} | {res:.5e}")
