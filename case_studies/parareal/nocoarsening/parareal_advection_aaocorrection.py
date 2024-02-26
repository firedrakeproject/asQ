from math import sqrt
import firedrake as fd
from firedrake.petsc import PETSc
import asQ

Print = PETSc.Sys.Print

nt = 256
nx = 16
cfl = 1.5
theta = 0.5
angle = 1/3

umax = 1.
dx = 1./nx

dtf = cfl*dx/umax
T = nt*dtf

Print(T, dtf)
verbose = False

ntf = 16
ntc = nt//ntf
nits = ntc

dtc = ntf*dtf

# one member ensemble
time_partition = tuple((ntf,))

ensemble = asQ.create_ensemble(time_partition, fd.COMM_WORLD)
comm = ensemble.comm

mesh = fd.PeriodicUnitSquareMesh(nx, nx, quadrilateral=True, comm=comm)
x, y = fd.SpatialCoordinate(mesh)

V = fd.FunctionSpace(mesh, "DQ", 1)
c = fd.Constant(fd.as_vector((umax*fd.cos(angle), umax*fd.sin(angle))))


def form_mass(q, phi):
    return phi*q*fd.dx


def form_function(q, phi, t=None):
    # upwind switch
    n = fd.FacetNormal(mesh)
    un = fd.Constant(0.5)*(fd.dot(c, n) + abs(fd.dot(c, n)))

    # integration over element volume
    int_cell = q*fd.div(phi*c)*fd.dx

    # integration over internal facets
    int_facet = (phi('+')-phi('-'))*(un('+')*q('+')-un('-')*q('-'))*fd.dS

    return int_facet - int_cell


initial_expr = (1+8*fd.pi*fd.pi)*fd.cos(x*fd.pi*2)*fd.cos(y*fd.pi*2)
uinitial = fd.Function(V).interpolate(initial_expr)

correction_type = 'final'  # 'final' or 'full'
cheap_rtol = 1e-2
fine_rtol = 1e-10

Print(f"correction_type = {correction_type}")
Print(f"cheap_rtol = {cheap_rtol}")
Print(f"fine_rtol = {fine_rtol}")

block_sparams = {
    'ksp_type': 'preonly',
    'pc_type': 'lu',
}

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
    'mat_type': 'matfree',
    'ksp_type': 'preonly',
    'pc_type': 'python',
    'pc_python_type': 'asQ.DiagFFTPC',
    'diagfft_alpha': cheap_rtol,
    'diagfft_block': block_sparams
}

fine_sparams = {
    'snes_type': 'ksponly',
    'snes': {
        # 'monitor': None,
        # 'converged_reason': None,
        'rtol': 9e-1,
    },
    'ksp': {
        # 'monitor': None,
        # 'converged_reason': None,
        'rtol': fine_rtol,
    },
    'mat_type': 'matfree',
    'ksp_type': 'richardson',
    'pc_type': 'python',
    'pc_python_type': 'asQ.DiagFFTPC',
    'diagfft_alpha': 1e-4,
    'diagfft_block': block_sparams
}

fine_paradiag = asQ.Paradiag(ensemble=ensemble,
                             form_function=form_function,
                             form_mass=form_mass,
                             ics=uinitial, dt=dtf, theta=theta,
                             time_partition=time_partition,
                             solver_parameters=fine_sparams)
fine_stepper = fine_paradiag.solver

cheap_paradiag = asQ.Paradiag(ensemble=ensemble,
                              form_function=form_function,
                              form_mass=form_mass,
                              ics=uinitial, dt=dtf, theta=theta,
                              time_partition=time_partition,
                              solver_parameters=cheap_sparams)
cheap_stepper = cheap_paradiag.solver


# ## define F and G
def G(u, uout, **kwargs):
    cheap_stepper.aaofunc.assign(u)
    cheap_stepper.solve()
    uout.assign(cheap_stepper.aaofunc)


def F(u, uout, **kwargs):
    fine_stepper.aaofunc.assign(u)
    fine_stepper.solve()
    uout.assign(fine_stepper.aaofunc)


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

tol = 1e-10
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
    if err < tol:
        break
