from math import sqrt
import numpy as np
import firedrake as fd
from firedrake.petsc import PETSc
from utils.serial import SerialMiniApp
from asQ import create_ensemble

Print = PETSc.Sys.Print
global_comm = fd.COMM_WORLD

# discretisation parameters

T = 0.5
nt = 128
nx = 16
cfl = 1.5
theta = 0.5
nu = 0.1

dx = 1./nx

# number of fine timesteps
dtf = T/nt

cfl = nu*dtf/(dx*dx)
Print(cfl, dtf)

verbose = False
verbose_ranks = False

# number of coarse time intervals and parareal iterations

ntc = 8
ntf = nt//ntc
nits = ntc

dtc = ntf*dtf

# set up parallelism

assert (ntc == global_comm.size)

time_partition = tuple(1 for _ in range(ntc))
ensemble = create_ensemble(time_partition, global_comm)
rank = ensemble.ensemble_comm.rank
is_root = (rank == 0)
is_last = (rank == ensemble.ensemble_comm.size - 1)

mesh = fd.UnitSquareMesh(nx, nx, quadrilateral=True, comm=ensemble.comm)
x, y = fd.SpatialCoordinate(mesh)

V = fd.FunctionSpace(mesh, "CG", 1)

Nu = fd.Constant(nu)

# FE forms for heat equation


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

# use serial miniapp for both coarse and fine propogators

miniapp = SerialMiniApp(dtf, theta,
                        uinitial,
                        form_mass, form_function,
                        sparameters)

# ## define F and G


def G(u, uout, **kwargs):
    miniapp.dt.assign(dtc)
    miniapp.solve(1, ics=u, **kwargs)
    uout.assign(miniapp.w0)


def F(u, uout, **kwargs):
    miniapp.dt.assign(dtf)
    miniapp.solve(ntf, ics=u, **kwargs)
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


# convenience functions


def coarse_series():
    return [fd.Function(V) for _ in range(2)]


def copy_series(dst, src):
    for d, s in zip(dst, src):
        d.assign(s)


def parallel_norm(u):
    norml = np.zeros(1)
    normg = np.zeros(1)

    nl = fd.norm(u)
    norml[0] = nl*nl
    ensemble.ensemble_comm.Allreduce(norml, normg)

    return nl, sqrt(normg[0])


def parallel_errnorm(ue, u):
    return parallel_norm(u-ue)


# ## find "exact" fine solution at coarse points at beginning/end of slice

userial = coarse_series()
userial[0].assign(uinitial)
userial[1].assign(uinitial)

for i in range(rank+1):
    F(userial[1], userial[1])

# ## initialise coarse points
# Uk[0] is point at beginning of local interval
# Uk[1] is point at end of local interval

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

# coarse point at beginning of slice
for i in range(rank):
    G(Gk1[0], Gk1[0])
# coarse point at end of slice
G(Gk1[0], Gk1[1])

copy_series(Uk1, Gk1)

# ## parareal iterations

# neighbouring ranks
dst = (rank + 1) % ensemble.ensemble_comm.size
src = (rank - 1) % ensemble.ensemble_comm.size


for it in range(nits):
    copy_series(Uk, Uk1)
    copy_series(Gk, Gk1)

    # ##  step 1: fine propogator

    # run fine propogator in parallel
    F(Uk[0], Fk[1])

    # send forward fine solution (be lazy and do a sendrecv then reset ics)
    ensemble.sendrecv(fsend=Fk[1], dest=dst, sendtag=dst,
                      frecv=Fk[0], source=src, recvtag=rank)
    if is_root:
        Fk[0].assign(uinitial)

    # ## step 2: coarse propogator

    for i in range(ntc):
        # propogate and correct
        if rank == i:
            G(Uk1[0], Gk1[1])
            Uk1[1].assign(Fk[1] + Gk1[1] - Gk[1])

        # send corrected solution on all but last interval
        last_interval = (i == (ntc-1))

        if rank == i and not last_interval:
            ensemble.send(Uk1[1], dest=dst, tag=dst)
        elif rank == (i+1) and not last_interval:
            ensemble.recv(Uk1[0], source=src, tag=rank)

    # ## step 3: check error vs serial solution and change in solution over this iteration

    resl, resg = parallel_errnorm(Uk[1], Uk1[1])
    errl, errg = parallel_errnorm(userial[1], Uk1[1])

    if verbose_ranks:
        for i in range(ntc):
            if rank == i:
                Print(f"{str(rank).ljust(3)} | {str(it).ljust(3)} | {errl:.5e} | {resl:.5e}",
                      comm=ensemble.comm)
            ensemble.global_comm.Barrier()

    Print(f"{str(it).ljust(3)} | {errg:.5e} | {resg:.5e}")
    ensemble.global_comm.Barrier()
