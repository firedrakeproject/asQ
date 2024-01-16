from firedrake import *
import asQ

time_partition = [2, 2]
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


block_parameters = {
    'ksp_type': 'preonly',
    'pc_type': 'lu'}

solver_parameters = {
    'ksp_monitor': None,
    'ksp_converged_rate': None,
    'snes_type': 'ksponly',
    'mat_type': 'matfree',
    'ksp_type': 'gmres',
    'pc_type': 'python',
    # 'pc_python_type': 'asQ.DiagFFTPC',
    # 'diagfft_block': block_parameters,
    # 'diagfft_state': 'linear',
    'pc_python_type': 'asQ.JacobiPC',
    'aaojacobi_block': block_parameters,
    'aaojacobi_state': 'linear',
    'aaos_jacobian_state': 'linear'}

paradiag = asQ.Paradiag(
    ensemble=ensemble,
    form_mass=form_mass,
    form_function=form_function,
    ics=uinitial, dt=0.5, theta=1.0,
    time_partition=time_partition,
    solver_parameters=solver_parameters)

paradiag.solve(nwindows=1)
