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
    'pc_type': 'ilu',
    'ksp_view': None,
    'pc_factor_mat_solver_type': 'petsc'}

solver_parameters = {
    'ksp_monitor': None,
    'ksp_converged_rate': None,
    'snes_type': 'ksponly',
    'mat_type': 'matfree',
    'ksp_type': 'preonly',
    'pc_type': 'python',
    # 'pc_python_type': 'asQ.JacobiPC',
    # 'aaojacobi_block': {
    #     'ksp_type': 'gmres',
    #     'pc_type': 'ilu',
    #     'ksp_view': None,
    #     'pc_factor_mat_solver_type': 'petsc'
    # },
    # 'aaojacobi_block_0': {
    #     'pc_type': 'jacobi',
    # },
    # 'aaojacobi_block_1': {
    #     'ksp_type': 'fgmres',
    # },
    'pc_type': 'python',
    'pc_python_type': 'asQ.SliceJacobiPC',
    'slice_jacobi_nsteps': 2,
    'slice_jacobi_slice': {
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'pc_python_type': 'asQ.JacobiPC',
        'aaojacobi_block': {
            'ksp_type': 'richardson',
            'pc_type': 'ilu',
            'ksp_view': None,
            'pc_factor_mat_solver_type': 'petsc'
        },
        'aaojacobi_block_0': {
            'pc_type': 'jacobi',
        },
        'aaojacobi_block_1': {
            'ksp_type': 'fgmres',
        },
    },
    'slice_jacobi_slice_1': {
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'pc_python_type': 'asQ.CirculantPC',
        'circulant_block': {
            'ksp_type': 'richardson',
            'pc_type': 'sor',
            'ksp_view': None,
            'pc_factor_mat_solver_type': 'petsc'
        },
        'circulant_block_0': {
            'ksp_type': 'fgmres',
        },
        'circulant_block_1': {
            'pc_type': 'jacobi',
        },
    }
}

paradiag = asQ.Paradiag(
    ensemble=ensemble,
    form_mass=form_mass,
    form_function=form_function,
    ics=uinitial, dt=0.1, theta=0.5,
    time_partition=time_partition,
    solver_parameters=solver_parameters)

paradiag.solve(nwindows=1)
