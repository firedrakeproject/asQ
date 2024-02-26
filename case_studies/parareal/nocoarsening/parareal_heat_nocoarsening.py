from math import sqrt
import firedrake as fd
from firedrake.petsc import PETSc
Print = PETSc.Sys.Print

from utils.serial import SerialMiniApp
import asQ

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

# one member ensemble
time_partition = tuple((ntf,))

ensemble = asQ.create_ensemble(time_partition, fd.COMM_WORLD)
comm = ensemble.comm

mesh = fd.UnitSquareMesh(nx, nx, quadrilateral=True, comm=comm)
x, y = fd.SpatialCoordinate(mesh)

V = fd.FunctionSpace(mesh, "CG", 1)

Nu = fd.Constant(nu)


def form_mass(u, v):
    return fd.inner(u, v)*fd.dx


def form_function(u, v, t=None):
    return Nu*fd.inner(fd.grad(u), fd.grad(v))*fd.dx


initial_expr = (1+8*fd.pi*fd.pi)*fd.cos(x*fd.pi*2)*fd.cos(y*fd.pi*2)
uinitial = fd.Function(V).interpolate(initial_expr)

cheap_type = 'paradiag'
cheap_rtol = 1e-1

Print(f"cheap_type = {cheap_type}")
Print(f"cheap_rtol = {cheap_rtol}")

cheap_sparams = {
    'snes_type': 'ksponly',
    'snes': {
        # 'monitor': None,
        # 'converged_reason': None,
        'rtol': 9e-1,
    },
    'ksp': {
        # 'monitor': None,
        # 'converged_reason': None,
        'rtol': cheap_rtol,
    },
    'ksp_type': 'gmres',
    'pc_type': 'sor',
}

coarse_sparams = {
    'snes_type': 'ksponly',
    'snes': {
        # 'monitor': None,
        # 'converged_reason': None,
        'rtol': 1e-1,
    },
    'ksp': {
        # 'monitor': None,
        # 'converged_reason': None,
        'rtol': 1e-10,
    },
    'ksp_type': 'preonly',
    'pc_type': 'lu',
}

fine_sparams = {
    'snes_type': 'ksponly',
    'snes': {
        # 'monitor': None,
        # 'converged_reason': None,
        'rtol': 1e-1,
    },
    'ksp': {
        # 'monitor': None,
        # 'converged_reason': None,
        'rtol': 1e-10,
    },
    'ksp_type': 'preonly',
    'pc_type': 'lu',
}

paradiag_sparams = {
    'snes_type': 'ksponly',
    'snes': {
        # 'monitor': None,
        # 'converged_reason': None,
        'rtol': 9e-1,
    },
    'ksp': {
        # 'monitor': None,
        # 'converged_reason': None,
        'rtol': 1e-1,
    },
    'mat_type': 'matfree',
    'ksp_type': 'preonly',
    'pc_type': 'python',
    'pc_python_type': 'asQ.DiagFFTPC',
    'diagfft_alpha': cheap_rtol,
    'diagfft_block': {
        'ksp_type': 'preonly',
        'pc_type': 'lu'
    }
}

cheap_stepper = SerialMiniApp(dtf, theta,
                              uinitial,
                              form_mass, form_function,
                              cheap_sparams)

coarse_stepper = SerialMiniApp(dtc, theta,
                               uinitial,
                               form_mass, form_function,
                               coarse_sparams)

fine_stepper = SerialMiniApp(dtf, theta,
                             uinitial,
                             form_mass, form_function,
                             fine_sparams)

paradiag = asQ.Paradiag(ensemble=ensemble,
                        form_function=form_function,
                        form_mass=form_mass,
                        ics=uinitial, dt=dtf, theta=theta,
                        time_partition=time_partition,
                        solver_parameters=paradiag_sparams)
aaostepper = paradiag.solver

# ## define F and G
if cheap_type == 'coarsen':
    def G(u, uout, **kwargs):
        coarse_stepper.w0.assign(u)
        coarse_stepper.solve(nt=1, **kwargs)
        uout.assign(coarse_stepper.w0)

elif cheap_type == 'cheap':
    def G(u, uout, **kwargs):
        cheap_stepper.w0.assign(u)
        cheap_stepper.solve(nt=ntf, **kwargs)
        uout.assign(cheap_stepper.w0)

elif cheap_type == 'paradiag':
    def G(u, uout, **kwargs):
        aaostepper.aaofunc.assign(u)
        aaostepper.solve()
        uout.assign(aaostepper.aaofunc[-1])

else:
    raise ValueError(f"invalid cheap_type {cheap_type}")


def F(u, uout, **kwargs):
    fine_stepper.w0.assign(u)
    fine_stepper.solve(ntf, **kwargs)
    uout.assign(fine_stepper.w0)


Print('### === --- Timestepping loop --- === ###')
linear_its = 0
nonlinear_its = 0


def preproc(app, step, t):
    if verbose:
        Print('')
        Print(f'=== --- Timestep {step} --- ===')
        Print('')


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
