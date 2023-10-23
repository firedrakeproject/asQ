import firedrake as fd
from firedrake.petsc import PETSc
import asQ

Print = PETSc.Sys.Print
global_comm = fd.COMM_WORLD

T = 0.5
nt = 16
nx = 16
cfl = 1.5
theta = 0.5
nu = 0.1
alpha = 1.e-2

nwindows = 1

dx = 1./nx
dt = T/nt

cfl = nu*dt/(dx*dx)
Print(cfl, dt)
verbose = False

nslices = global_comm.size
assert (nt % nslices) == 0
slice_length = nt//nslices

time_partition = tuple(slice_length for _ in range(nslices))
ensemble = asQ.create_ensemble(time_partition, global_comm)

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

# parameters

block_sparams = {
    'ksp_type': 'preonly',
    'pc_type': 'lu',
}

sparameters = {
    'snes_type': 'ksponly',
    'snes': {
        'monitor': None,
        'converged_reason': None,
        'rtol': 1e-12,
    },
    'ksp': {
        'monitor': None,
        'converged_reason': None,
        'rtol': 1e-12,
    },
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'pc_type': 'python',
    'pc_python_type': 'asQ.ParaDiagPC',
    'diagfft_alpha': alpha,
}

for i in range(sum(time_partition)):
    sparameters['diagfft_block_'+str(i)] = block_sparams

# set up the all-at-once system

aaofunc = asQ.AllAtOnceFunction(ensemble, time_partition, V)
aaofunc.set_all_fields(uinitial)

aaoform = asQ.AllAtOnceForm(aaofunc, dt, theta,
                            form_mass, form_function)

aaosolver = asQ.AllAtOnceSolver(aaoform, aaofunc,
                                solver_parameters=sparameters)


def window_preproc(pdg, wndw, rhs=None):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'### === --- Calculating time-window {wndw} --- === ###')
    PETSc.Sys.Print('')


aaosolver.solve()
