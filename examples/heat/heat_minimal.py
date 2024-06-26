from firedrake import *
import asQ

time_partition = [2, 2, 2, 2]
ensemble = asQ.create_ensemble(time_partition)

mesh = SquareMesh(nx=8, ny=8, L=1,
                  comm=ensemble.comm)
x, y = SpatialCoordinate(mesh)

V = FunctionSpace(mesh, "CG", 1)
uinitial = Function(V)
uinitial.project(sin(x) + cos(y))


def form_mass(u, v):
    return u*v*dx


def form_function(u, v, t):
    return inner(grad(u), grad(v))*dx


solver_parameters = {
    'ksp_monitor': None,
    'ksp_converged_rate': None,
    'snes_type': 'ksponly',
    'mat_type': 'matfree',
    'ksp_type': 'richardson',
    'ksp_rtol': 1e-10,
    'pc_type': 'python',
    'pc_python_type': 'asQ.CirculantPC',
    'circulant_alpha': 1e-4,
    'circulant_block': {
        'ksp_rtol': 1e-6,
        'ksp_type': 'gmres',
        'pc_type': 'ilu',
    },
}

paradiag = asQ.Paradiag(
    ensemble=ensemble,
    form_mass=form_mass,
    form_function=form_function,
    ics=uinitial, dt=0.1, theta=0.5,
    time_partition=time_partition,
    solver_parameters=solver_parameters)

paradiag.solve(nwindows=1)
