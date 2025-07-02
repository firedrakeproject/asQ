import firedrake as fd
import utils.shallow_water.gravity_bumps as lcase
from utils import shallow_water as swe
from utils.planets import earth
from utils import units
from utils.hybridisation import HybridisedSCPC  # noqa: F401
import numpy as np


def test_real_hybridisation():
    theta = 0.5
    dt = units.hour

    d1 = fd.Constant(1/dt)
    d2 = fd.Constant(theta)

    mesh = swe.create_mg_globe_mesh(ref_level=3, coords_degree=1)
    x = fd.SpatialCoordinate(mesh)

    # case parameters
    g = earth.Gravity
    f = lcase.coriolis_expression(*x)
    H = lcase.H

    # shallow water equation forms
    def form_mass(u, h, v, q):
        return swe.linear.form_mass(mesh, u, h, v, q)

    def form_function(u, h, v, q, t=None):
        return swe.linear.form_function(mesh, g, H, f,
                                        u, h, v, q, t)

    V = swe.default_function_space(mesh, degree=0)

    # random rhs
    L = fd.Cofunction(V.dual())

    # PETSc solver parameters
    sparams = {
        'mat_type': 'matfree',
        'ksp_atol': 1e-100,
        'ksp_rtol': 1e-14,
        'ksp_type': 'richardson',
        'pc_type': 'python',
        'pc_python_type': f'{__name__}.HybridisedSCPC',
        'hybridscpc_condensed_field': {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
        }
    }

    # trace component should have zero rhs
    np.random.seed(12345)
    L.assign(0)
    for ldat in L.dat:
        ldat.data[:] = np.random.rand(*(ldat.data.shape))

    # block forms
    us = fd.TrialFunctions(V)
    vs = fd.TestFunctions(V)

    M = form_mass(*us, *vs)
    K = form_function(*us, *vs)

    A = d1*M + d2*K

    appctx = {
        'uref': fd.Function(V),
        'bcs': None,
        'tref': None,
        'form_mass': form_mass,
        'form_function': form_function,
        'dt': dt,
        'theta': theta,
    }

    wout = fd.Function(V)
    problem = fd.LinearVariationalProblem(A, L, wout)
    solver = fd.LinearVariationalSolver(problem, appctx=appctx,
                                        solver_parameters=sparams)
    solver.solve()

    niterations = solver.snes.getLinearSolveIterations()
    converged_reason = solver.snes.getKSP().getConvergedReason()

    assert niterations == 1, "hybridisation should be equivalent to direct solve"
    assert converged_reason == 2, "rtol should be almost machine precision for direct solve"


def test_complex_hybridisation():
    from scipy.fft import fft
    from asQ.complex_proxy import vector as cpx

    # eigenvalues
    nt = 16
    theta = 0.5
    alpha = 1e-3
    dt = units.hour

    gamma = alpha**(np.arange(nt)/nt)
    C1 = np.zeros(nt)
    C2 = np.zeros(nt)

    C1[:2] = [1/dt, -1/dt]
    C2[:2] = [theta, 1-theta]

    D1 = np.sqrt(nt)*fft(gamma*C1)
    D2 = np.sqrt(nt)*fft(gamma*C2)

    eigenvalue = 3
    d1 = D1[eigenvalue]
    d2 = D2[eigenvalue]

    d1c = cpx.ComplexConstant(d1)
    d2c = cpx.ComplexConstant(d2)

    mesh = swe.create_mg_globe_mesh(ref_level=3, coords_degree=1)
    x = fd.SpatialCoordinate(mesh)

    # case parameters
    g = earth.Gravity
    f = lcase.coriolis_expression(*x)
    H = lcase.H

    # function spaces
    V = swe.default_function_space(mesh)
    W = cpx.FunctionSpace(V)

    # shallow water equation forms
    def form_mass(u, h, v, q):
        return swe.linear.form_mass(mesh, u, h, v, q)

    def form_function(u, h, v, q, t=None):
        return swe.linear.form_function(mesh, g, H, f,
                                        u, h, v, q, t)

    # random rhs
    L = fd.Cofunction(W.dual())

    # PETSc solver parameters
    sparams = {
        'mat_type': 'matfree',
        'ksp_atol': 1e-100,
        'ksp_rtol': 1e-14,
        'ksp_type': 'richardson',
        'pc_type': 'python',
        'pc_python_type': f'{__name__}.HybridisedSCPC',
        'hybridscpc_condensed_field': {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
        }
    }

    # trace component should have zero rhs
    np.random.seed(12345)
    L.assign(0)
    for dat in L.dat:
        dat.data[:] = np.random.rand(*(dat.data.shape))

    # block forms
    M = cpx.BilinearForm(W, d1c, form_mass)
    K = cpx.BilinearForm(W, d2c, form_function)

    A = M + K

    appctx = {
        'cpx': cpx,
        'uref': fd.Function(W),
        'bcs': None,
        'tref': None,
        'form_mass': form_mass,
        'form_function': form_function,
        'd1': d1c,
        'd2': d2c,
    }

    wout = fd.Function(W).assign(0)
    problem = fd.LinearVariationalProblem(A, L, wout)
    solver = fd.LinearVariationalSolver(problem, appctx=appctx,
                                        solver_parameters=sparams)
    solver.solve()

    niterations = solver.snes.getLinearSolveIterations()
    converged_reason = solver.snes.getKSP().getConvergedReason()

    assert niterations == 1, "hybridisation should be equivalent to direct solve"
    assert converged_reason == 2, "rtol should be almost machine precision for direct solve"
