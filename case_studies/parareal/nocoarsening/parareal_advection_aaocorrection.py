from math import sqrt
import firedrake as fd
from firedrake.petsc import PETSc
import asQ

Print = PETSc.Sys.Print

nt = 128
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

cheap_rtol = 1e-3
cheap_alpha = cheap_rtol
cheap_ksp = 'preonly'

fine_rtol = 1e-11
fine_alpha = 1e-4
fine_ksp = 'richardson'

exact_rtol = 1e-11
exact_alpha = 1e-4
exact_ksp = 'richardson'

Print(f"correction_type = {correction_type}")
Print(f"cheap_rtol = {cheap_rtol}")
Print(f"fine_rtol = {fine_rtol}")
Print('')

Print('### === --- Setup --- === ###')
Print('')


def paradiag_smoother(rtol, alpha, ksp_type='preonly'):
    sparams = {
        'snes_type': 'ksponly',
        'ksp': {
            # 'monitor': None,
            # 'converged_reason': None,
            'rtol': rtol,
            'atol': 1e-100,
        },
        'mat_type': 'matfree',
        'ksp_type': ksp_type,
        'pc_type': 'python',
        'pc_python_type': 'asQ.DiagFFTPC',
        'diagfft_alpha': alpha,
        'diagfft_block': {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
        }
    }
    paradiag = asQ.Paradiag(ensemble=ensemble,
                            form_function=form_function,
                            form_mass=form_mass,
                            ics=uinitial, dt=dtf, theta=theta,
                            time_partition=time_partition,
                            solver_parameters=sparams)
    solver = paradiag.solver

    def smoother(u, uout):
        solver.aaofunc.assign(u)
        solver.solve()
        uout.assign(solver.aaofunc)

    smoother.solver = solver
    return smoother


# ## define smoothers F and G and E

G = paradiag_smoother(cheap_rtol, cheap_rtol, cheap_ksp)
F = paradiag_smoother(fine_rtol, fine_alpha, fine_ksp)
E = paradiag_smoother(exact_rtol, exact_alpha, exact_ksp)

# ## set up buffers


def preproc(app, step, t):
    if verbose:
        Print('')
        Print(f'=== --- Timestep {step} --- ===')
        Print('')


def coarse_series():
    return [asQ.AllAtOnceFunction(ensemble, time_partition, V)
            for _ in range(ntc)]


def copy_series(dst, src):
    for d, s in zip(dst, src):
        d.assign(s)


def series_norm(series):
    norm = 0.
    for s in series:
        sn = fd.norm(s.function)
        norm += sn*sn
    return sqrt(norm)


def series_error(exact, series, point='all'):
    norm = 0.
    for e, s in zip(exact, series):
        if point == 'all':
            en = fd.errornorm(e.function, s.function)
        elif point == 'ic':
            en = fd.errornorm(e.initial_condition, s.initial_condition)
        else:
            en = fd.errornorm(e[point], s[point])
        norm += en*en
    return sqrt(norm)


Print('### === --- Calculate reference solution --- === ###')
Print('')


# ## find "exact" fine solution at coarse points in serial
userial = coarse_series()
userial[0].assign(uinitial)

for i in range(ntc):
    if i > 0:
        userial[i].assign(userial[i-1][-1])
    E(userial[i], userial[i])

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
    if i > 0:
        Gk1[i].assign(Gk1[i-1][-1])
    G(Gk1[i], Gk[i])

copy_series(Uk1, Gk1)


# ## parareal iterations

Print('### === --- Timestepping loop --- === ###')

tol = 1e-10
for it in range(nits):
    copy_series(Uk, Uk1)
    copy_series(Gk, Gk1)

    for i in range(ntc):
        Uk[i].assign(Uk[i].initial_condition)
        F(Uk[i], Fk[i])

    for i in range(ntc):
        G(Uk1[i], Gk1[i])
        if i < ntc-1:
            uk1 = Fk[i][-1] + Gk1[i][-1] - Gk[i][-1]
            Uk1[i+1].initial_condition.assign(uk1)

    res = series_error(Uk, Uk1, point='ic')
    err = series_error(userial, Uk1, point='ic')
    Print(f"{str(it).ljust(3)} | {err:.5e} | {res:.5e}")
    if err < tol:
        break
