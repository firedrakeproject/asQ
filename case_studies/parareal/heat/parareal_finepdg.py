from math import sqrt
import firedrake as fd
from firedrake.petsc import PETSc
from utils.serial import SerialMiniApp
import asQ

Print = PETSc.Sys.Print
global_comm = fd.COMM_WORLD

T = 0.5
nt = 128
nx = 16
cfl = 1.5
theta = 0.5
nu = 0.1
alpha = 1e-2

dx = 1./nx
dtf = T/nt

cfl = nu*dtf/(dx*dx)
Print(cfl, dtf)
verbose = False
serial = False
tol = 1e-16

ntf = 16
ntc = nt//ntf
nits = ntc

assert (nt % ntc) == 0

dtc = ntf*dtf

nslices = global_comm.size
slice_length = ntf//nslices

time_partition = tuple(slice_length for _ in range(nslices))
ensemble = asQ.create_ensemble(time_partition, global_comm)
rank = ensemble.ensemble_comm.rank
is_root = (rank == 0)

mesh = fd.UnitSquareMesh(nx, nx, quadrilateral=True, comm=ensemble.comm)
x, y = fd.SpatialCoordinate(mesh)

V = fd.FunctionSpace(mesh, "CG", 1)

Nu = fd.Constant(nu)


def form_mass(u, v):
    return fd.inner(u, v)*fd.dx


def form_function(u, v):
    return Nu*fd.inner(fd.grad(u), fd.grad(v))*fd.dx


initial_expr = (1+8*fd.pi*fd.pi)*fd.cos(x*fd.pi*2)*fd.cos(y*fd.pi*2)
uinitial = fd.Function(V).interpolate(initial_expr)

serial_sparams = {
    'snes_type': 'ksponly',
    'snes': {
        # 'monitor': None,
        # 'converged_reason': None,
        'rtol': 1e-12,
    },
    'ksp': {
        # 'monitor': None,
        # 'converged_reason': None,
        'rtol': 1e-12,
    },
    'ksp_type': 'preonly',
    'pc_type': 'lu',
}

block_sparams = {
    'ksp': {
        'rtol': 1e-12,
    },
    'ksp_type': 'preonly',
    'pc_type': 'lu',
}

pdg_sparams = {
    'snes_type': 'ksponly',
    'snes': {
        # 'monitor': None,
        # 'converged_reason': None,
        'rtol': 1e-14,
    },
    'ksp': {
        # 'monitor': None,
        # 'converged_reason': None,
        'rtol': 1e-14,
    },
    'mat_type': 'matfree',
    'ksp_type': 'gmres',
    'pc_type': 'python',
    'pc_python_type': 'asQ.ParaDiagPC',
    'diagfft_alpha': alpha,
}
for i in range(sum(time_partition)):
    pdg_sparams['diagfft_block_'+str(i)] = block_sparams

if verbose and is_root:
    serial_sparams['snes']['monitor'] = None
    pdg_sparams['snes']['monitor'] = None
    pdg_sparams['snes']['converged_reason'] = None
    pdg_sparams['ksp']['monitor'] = None
    pdg_sparams['ksp']['converged_reason'] = None

# serial coarse propogator
miniapp = SerialMiniApp(dtc, theta,
                        uinitial,
                        form_mass, form_function,
                        serial_sparams)

# parallel fine propogator
aaofunc = asQ.AllAtOnceFunction(ensemble, time_partition, V)
aaofunc.set_all_fields(uinitial)

aaoform = asQ.AllAtOnceForm(aaofunc, dtf, theta,
                            form_mass, form_function)

aaosolver = asQ.AllAtOnceSolver(aaoform, aaofunc,
                                solver_parameters=pdg_sparams)

# ## define F and G


def G(u, uout, **kwargs):
    miniapp.dt.assign(dtc)
    miniapp.solve(1, ics=u, **kwargs)
    uout.assign(miniapp.w0)


def F(u, uout, serial=serial, **kwargs):
    if serial:
        miniapp.dt.assign(dtf)
        miniapp.solve(ntf, ics=u, **kwargs)
        uout.assign(miniapp.w0)
    else:
        aaofunc.set_all_fields(u)
        aaosolver.solve()
        aaofunc.bcast_field(-1, uout)


Print('### === --- Timestepping loop --- === ###')
linear_its = 0
nonlinear_its = 0

ofile = fd.File("output/heat.pvd")
ofile.write(miniapp.w0, time=0)


def preproc(app, step, rhs=None):
    if verbose:
        Print('')
        Print(f'=== --- Timestep {step} --- ===')
        Print('')


def postproc(app, step, rhs=None):
    pass


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
    F(userial[i], userial[i+1], serial=True)

uparallel = coarse_series()
uparallel[0].assign(uinitial)

for i in range(ntc):
    F(uparallel[i], uparallel[i+1], serial=False)

err = series_error(userial, uparallel)
Print(rank, err)

# ## initialise coarse points

Gk = coarse_series()
Uk = coarse_series()
Fk = coarse_series()

Gk1 = coarse_series()
Uk1 = coarse_series()

Gk[0].assign(uinitial)
for i in range(ntc):
    G(Gk[i], Gk[i+1])

copy_series(Uk, Gk)
copy_series(Gk1, Gk)
copy_series(Uk1, Gk)


# ## parareal iterations

for it in range(nits):
    copy_series(Uk, Uk1)
    copy_series(Gk, Gk1)

    for i in range(ntc):
        F(Uk[i], Fk[i+1])

    Uk1[0].assign(uinitial)

    for i in range(ntc):
        G(Uk1[i], Gk1[i+1])

        Uk1[i+1].assign(Fk[i+1] + Gk1[i+1] - Gk[i+1])

    err = series_error(userial, Uk1)
    res = series_error(Uk, Uk1)
    if is_root:
        Print(f"{str(rank).ljust(3)} | {str(it).ljust(3)} | {err:.5e} | {res:.5e}",
              comm=ensemble.comm)
    if err < tol:
        break
