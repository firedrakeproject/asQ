import numpy as np
import firedrake as fd
from firedrake.petsc import PETSc
from utils.parareal import Parareal
from asQ import create_ensemble

Print = PETSc.Sys.Print
global_comm = fd.COMM_WORLD

# discretisation parameters

T = 0.5
nt = 128
nx = 16
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

ntc = 16
ntf = nt//ntc
nits = ntc
tol = 1e-8

dtc = ntf*dtf

# set up parallelism

assert (ntc == global_comm.size)

time_partition = tuple(1 for _ in range(ntc))
ensemble = create_ensemble(time_partition, global_comm)
rank = ensemble.ensemble_comm.rank

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
        'rtol': 1e-8,
    },
    'ksp': {
        'rtol': 1e-8,
    },
    'ksp_type': 'preonly',
    'pc_type': 'lu',
}

if verbose:
    sparameters['snes']['monitor'] = None

# use serial miniapp for both coarse and fine propogators

parareal = Parareal(ensemble, ntf, dtf, theta,
                    uinitial,
                    form_mass, form_function,
                    sparameters)

# ## find "exact" fine solution at coarse points at beginning/end of slice

userial = [fd.Function(V) for _ in range(2)]
userial[0].assign(uinitial)
userial[1].assign(uinitial)

for i in range(rank+1):
    parareal.F(userial[1], userial[1])


def parallel_norm(u):
    norml = np.zeros(1)
    normg = np.zeros(1)

    norml[0] = fd.norm(u)
    ensemble.ensemble_comm.Allreduce(norml, normg)

    return norml[0], normg[0]


def parallel_errnorm(ue, u):
    return parallel_norm(u-ue)


def preproc(prrl, it):
    if verbose:
        Print('')
        Print(f'=== --- Parareal iteration {it} --- ===')
        Print('')


def postproc(prrl, it):
    errl, errg = parallel_errnorm(userial[1], prrl.u1)
    resl, resg = parallel_errnorm(prrl.u1, prrl.u1)

    if verbose_ranks:
        for i in range(ntc):
            if rank == i:
                Print(f"{str(rank).ljust(3)} | {str(it).ljust(3)} | {errl:.5e} | {resl:.5e}",
                      comm=ensemble.comm)
            ensemble.global_comm.Barrier()

    Print(f"{str(it).ljust(3)} | {errg:.5e} | {resg:.5e}")


# ## solve timeseries using parareal

Print('### === --- Parareal loop --- === ###')

parareal.solve(max_its=nits, tol=tol,
               preproc=preproc,
               postproc=postproc)
