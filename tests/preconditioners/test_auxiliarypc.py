import firedrake as fd
import numpy as np
import pytest


def create_complex_solver(cpx, mesh, V, W, bcs, sparams):
    def form_mass(u, v):
        return u*v*fd.dx

    def form_function(u, v, t=None):
        return fd.inner(fd.grad(u), fd.grad(v))*fd.dx

    d1 = 1.5 + 0.4j
    d2 = 0.3 - 0.2j

    u = fd.Function(W)

    cpx_bcs = tuple((cb for bc in bcs
                     for cb in cpx.DirichletBC(W, V, bc, 0*bc.function_arg)))

    M = cpx.BilinearForm(W, d1, form_mass)
    K = cpx.derivative(d2, form_function, u)

    A = M + K

    L = fd.Cofunction(W.dual())
    np.random.seed(12345)
    for dat in L.dat:
        dat.data[:] = np.random.rand(*(dat.data.shape))

    appctx = {
        'cpx': cpx,
        'uref': u,
        'tref': None,
        'd1': d1,
        'd2': d2,
        'bcs': cpx_bcs,
        'form_mass': form_mass,
        'form_function': form_function
    }

    problem = fd.LinearVariationalProblem(A, L, u, bcs=cpx_bcs)
    solver = fd.LinearVariationalSolver(problem, appctx=appctx,
                                        solver_parameters=sparams)
    return solver, u


bcopts = ['none', 'dirichlet']
cpxopts = ['cpx_vector', 'cpx_mixed']


@pytest.mark.parametrize("bcopt", bcopts)
@pytest.mark.parametrize("cpxopt", cpxopts)
def test_complex_blockpc(bcopt, cpxopt):
    if cpxopt == 'cpx_vector':
        from asQ.complex_proxy import vector as cpx
    elif cpxopt == 'cpx_mixed':
        from asQ.complex_proxy import mixed as cpx
    else:
        assert False and "complex_proxy type not recognised"

    mesh = fd.UnitSquareMesh(4, 4)

    V = fd.FunctionSpace(mesh, "CG", 1)
    W = cpx.FunctionSpace(V)

    if bcopt == 'none':
        bcs = []
    elif bcopt == 'dirichlet':
        bcs = [fd.DirichletBC(V, 0, sub_domain=1)]
    else:
        assert False and "boundary condition option not recognised"

    pc_type = 'ilu'

    aux_sparams = {
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'pc_python_type': 'asQ.AuxiliaryComplexBlockPC',
        'aux_pc_type': pc_type
    }

    direct_sparams = {
        'ksp_type': 'preonly',
        'pc_type': pc_type,
    }

    direct_solver, udirect = create_complex_solver(cpx, mesh, V, W, bcs, direct_sparams)
    aux_solver, uaux = create_complex_solver(cpx, mesh, V, W, bcs, aux_sparams)

    direct_solver.solve()
    aux_solver.solve()

    assert fd.errornorm(udirect, uaux) < 1e-12
