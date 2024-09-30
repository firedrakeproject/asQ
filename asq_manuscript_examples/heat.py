from firedrake import *
import asQ

time_partition = [2, 2, 2, 2]
ensemble = asQ.create_ensemble(
    time_partition, comm=COMM_WORLD)

mesh = SquareMesh(nx=32, ny=32, L=1,
                  comm=ensemble.comm)
x, y = SpatialCoordinate(mesh)

V = FunctionSpace(mesh, "CG", 1)
u0 = Function(V)
u0.interpolate(sin(0.25*pi*x)*cos(2*pi*y))

aaofunc = asQ.AllAtOnceFunction(
    ensemble, time_partition, V)
aaofunc.initial_condition.assign(u0)

dt = 0.05
theta = 1

bcs = [DirichletBC(V, 0, sub_domain=1)]


def form_mass(u, v):
    return u*v*dx


def form_function(u, v, t):
    return inner(grad(u), grad(v))*dx


aaoform = asQ.AllAtOnceForm(
    aaofunc, dt, theta, form_mass,
    form_function, bcs=bcs)

solver_parameters = {
    'snes_type': 'ksponly',
    'mat_type': 'matfree',
    'ksp_type': 'richardson',
    'ksp_rtol': 1e-12,
    'ksp_monitor': None,
    'ksp_converged_rate': None,
    'pc_type': 'python',
    'pc_python_type': 'asQ.CirculantPC',
    'circulant_block': {'pc_type': 'lu'},
    'circulant_alpha': 1e-4}

aaosolver = asQ.AllAtOnceSolver(
    aaoform, aaofunc, solver_parameters)

aaofunc.assign(u0)
for i in range(6):
    aaosolver.solve()
    aaofunc.bcast_field(
        -1, aaofunc.initial_condition)
    aaofunc.assign(
        aaofunc.initial_condition)
