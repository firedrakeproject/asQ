import firedrake as fd
from asQ.complex_proxy import vector as cpx

import numpy as np

mesh = fd.UnitSquareMesh(4, 4)

V = fd.FunctionSpace(mesh, "CG", 1)
W = cpx.FunctionSpace(V)


def form_mass(u, v):
    return u*v*fd.dx


def form_function(u, v, t=None):
    return fd.inner(fd.grad(u), fd.grad(v))*fd.dx


d1 = 1.5 + 0.4j
d2 = 0.3 - 0.2j

u = fd.Function(W)

M = cpx.BilinearForm(W, d1, form_mass)
K = cpx.derivative(d2, form_function, u)

A = M + K

L = fd.Cofunction(W.dual())
np.random.seed(12345)
for dat in L.dat:
    dat.data[:] = np.random.rand(*(dat.data.shape))

sparams = {
    'ksp': {
        'monitor': None,
        'converged_rate': None,
    },
    'ksp_type': 'preonly',
    'pc_type': 'python',
    'pc_python_type': 'asQ.AuxiliaryComplexBlockPC',
    'aux': {
        'pc_type': 'lu',
        'd1r': d1.real,
        'd1i': d1.imag,
    }
}

appctx = {
    'cpx': cpx,
    'uref': u,
    'tref': None,
    'd1': d1,
    'd2': d2,
    'bcs': [],
    'form_mass': form_mass,
    'form_function': form_function
}

problem = fd.LinearVariationalProblem(A, L, u)
solver = fd.LinearVariationalSolver(problem, appctx=appctx,
                                    solver_parameters=sparams)
solver.solve()
