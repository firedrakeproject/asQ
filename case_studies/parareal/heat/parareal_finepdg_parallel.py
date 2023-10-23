from math import sqrt
import numpy as np
import firedrake as fd
from firedrake.petsc import PETSc
from utils.serial import SerialMiniApp
import asQ
from ensemble_square import ensemble_square

Print = PETSc.Sys.Print
global_comm = fd.COMM_WORLD

# discretisation parameters

T = 0.5
nt = 128
nx = 16
cfl = 1.5
theta = 0.5
nu = 0.1
alpha = 1e-3

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

slice_length = 2
nchunks = ntc
chunk_length = ntf//slice_length

assert slice_length*chunk_length == ntf

assert global_comm.size == nchunks*chunk_length

fine_ensemble, coarse_ensemble = ensemble_square(nchunks,
                                                 chunk_length,
                                                 global_comm)

time_partition = tuple(slice_length for _ in range(ntc))
fine_rank = fine_ensemble.ensemble_comm.rank
coarse_rank = coarse_ensemble.ensemble_comm.rank
is_root = (fine_rank == 0)

# serial coarse propogator

coarse_mesh = fd.UnitSquareMesh(nx, nx, quadrilateral=True, comm=coarse_ensemble.comm)
x, y = fd.SpatialCoordinate(coarse_mesh)

Vc = fd.FunctionSpace(coarse_mesh, "CG", 1)

Nu = fd.Constant(nu)

# FE forms for heat equation


def form_mass(u, v):
    return fd.inner(u, v)*fd.dx


def form_function(u, v):
    return Nu*fd.inner(fd.grad(u), fd.grad(v))*fd.dx


initial_expr = (1+8*fd.pi*fd.pi)*fd.cos(x*fd.pi*2)*fd.cos(y*fd.pi*2)
coarse_ics = fd.Function(Vc).interpolate(initial_expr)

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

# use serial miniapp for both coarse and fine propogators

miniapp = SerialMiniApp(dtf, theta,
                        coarse_ics,
                        form_mass, form_function,
                        sparameters)

# parallel fine propogator

fine_mesh = fd.UnitSquareMesh(nx, nx, quadrilateral=True, comm=fine_ensemble.comm)
Vf = fd.FunctionSpace(fine_mesh, "CG", 1)
fine_ics = fd.Function(Vf).assign(coarse_ics)

aaofunc = asQ.AllAtOnceFunction(fine_ensemble, time_partition, Vf)
aaofunc.set_all_fields(fine_ics)

aaoform = asQ.AllAtOnceForm(aaofunc, dtf, theta,
                            form_mass, form_function)

aaosolver = asQ.AllAtOnceSolver(aaoform, aaofunc,
                                solver_parameters=pdg_sparams)


# ## define F and G


def G(u, uout, **kwargs):
    miniapp.dt.assign(dtc)
    miniapp.solve(1, ics=u, **kwargs)
    uout.assign(miniapp.w0)


def F(u, uout, **kwargs):
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
    return [fd.Function(Vc) for _ in range(2)]


def copy_series(dst, src):
    for d, s in zip(dst, src):
        d.assign(s)


def parallel_norm(u, ensemble=coarse_ensemble):
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

for i in range(coarse_rank+1):
    F(userial[1], userial[1])

# ## initialise coarse points
# Uk[0] is point at beginning of local interval
# Uk[1] is point at end of local interval

Gk = coarse_series()
Uk = coarse_series()
Fk = coarse_series()

Gk1 = coarse_series()
Uk1 = coarse_series()

Gk[0].assign(coarse_ics)
Uk[0].assign(coarse_ics)
Fk[0].assign(coarse_ics)

Gk1[0].assign(coarse_ics)
Uk1[0].assign(coarse_ics)

# coarse point at beginning of slice
for i in range(coarse_rank):
    G(Gk1[0], Gk1[0])
# coarse point at end of slice
G(Gk1[0], Gk1[1])

copy_series(Uk1, Gk1)

# ## parareal iterations

# neighbouring ranks
dst = (rank + 1) % coarse_ensemble.ensemble_comm.size
src = (rank - 1) % coarse_ensemble.ensemble_comm.size


for it in range(nits):
    copy_series(Uk, Uk1)
    copy_series(Gk, Gk1)

    # ##  step 1: fine propogator

    # run fine propogator in parallel
    F(Uk[0], Fk[1])

    # send forward fine solution (be lazy and do a sendrecv then reset ics)
    coarse_ensemble.sendrecv(fsend=Fk[1], dest=dst, sendtag=dst,
                             frecv=Fk[0], source=src, recvtag=rank)
    if is_root:
        Fk[0].assign(coarse_ics)

    # ## step 2: coarse propogator

    for i in range(ntc):
        # propogate and correct
        if rank == i:
            G(Uk1[0], Gk1[1])
            Uk1[1].assign(Fk[1] + Gk1[1] - Gk[1])

        # send corrected solution on all but last interval
        last_interval = (i == (ntc-1))

        if rank == i and not last_interval:
            coarse_ensemble.send(Uk1[1], dest=dst, tag=dst)
        elif rank == (i+1) and not last_interval:
            coarse_ensemble.recv(Uk1[0], source=src, tag=rank)

    # ## step 3: check error vs serial solution and change in solution over this iteration

    resl, resg = parallel_errnorm(Uk[1], Uk1[1])
    errl, errg = parallel_errnorm(userial[1], Uk1[1])

    if verbose_ranks:
        for i in range(ntc):
            if rank == i:
                Print(f"{str(rank).ljust(3)} | {str(it).ljust(3)} | {errl:.5e} | {resl:.5e}",
                      comm=ensemble.comm)
            global_comm.Barrier()

    Print(f"{str(it).ljust(3)} | {errg:.5e} | {resg:.5e}")
    global_comm.Barrier()
