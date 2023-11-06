from firedrake import *
from firedrake.petsc import PETSc

PETSc.Sys.popErrorHandler()

basemesh = UnitSquareMesh(4, 4)
hierarchy = MeshHierarchy(basemesh, refinement_levels=2)
mesh = hierarchy[-1]
x, y = SpatialCoordinate(mesh)

V = FunctionSpace(mesh, "CG", 1)

u = TrialFunction(V)
v = TestFunction(V)

A = inner(grad(u), grad(v))*dx
f = Function(V).interpolate(sin(x)*cos(y))

L = f.riesz_representation()
# L = assemble(inner(f, v)*dx)
# L = inner(f, v)*dx

bcs = DirichletBC(V, zero(), 1)

lu_params = {
    'ksp_type': 'preonly',
    'pc_type': 'lu'
}

patch_params = {
    'pc_patch': {
        'sub_mat_type': 'seqdense',
        'construct_dim': 0,
        'construct_type': 'star',
        'local_type': 'additive',
    },
    'sub': {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
    },
}

mg_params = {
    'ksp_type': 'richardson',
    'ksp_rtol': 1e-5,
    'pc_type': 'mg',
    'pc_mg_cycle_type': 'w',
    'pc_mg_type': 'full',
    'mg': {
        'levels': {
            'ksp_type': 'richardson',
            'ksp_richardson_scale': 2/3,
            'pc_type': 'python',
            'pc_python_type': 'firedrake.PatchPC',
            'patch': patch_params
        },
        'coarse': lu_params
    }
}

params = {
    'ksp_monitor': None,
    'ksp_converged_rate': None,
    # 'ksp_view': None,
}

params.update(mg_params)
# params.update(lu_params)

u = Function(V).zero()

problem = LinearVariationalProblem(A, L, u, bcs=bcs)
solver = LinearVariationalSolver(problem, solver_parameters=params)

solver.solve()
