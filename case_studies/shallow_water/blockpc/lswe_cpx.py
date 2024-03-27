import firedrake as fd
from firedrake.petsc import PETSc
from asQ.complex_proxy import vector as cpx

import utils.shallow_water.gravity_bumps as lcase
from utils import shallow_water as swe
from utils.planets import earth
from utils import units
from utils.hybridisation import HybridisedSCPC  # noqa: F401

import numpy as np
from scipy.fft import fft, fftfreq

PETSc.Sys.popErrorHandler()
Print = PETSc.Sys.Print

# get command arguments
import argparse
parser = argparse.ArgumentParser(
    description='Test preconditioners for the complex linear shallow water equations.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--dt', type=float, default=1.0, help='Timestep in hours (used to calculate the circulant eigenvalues).')
parser.add_argument('--nt', type=int, default=16, help='Number of timesteps (used to calculate the circulant eigenvalues).')
parser.add_argument('--theta', type=float, default=0.5, help='Parameter for implicit theta method. 0.5 for trapezium rule, 1 for backwards Euler (used to calculate the circulant eigenvalues).')
parser.add_argument('--alpha', type=float, default=1e-3, help='Circulant parameter (used to calculate the circulant eigenvalues).')
parser.add_argument('--eigenvalue', type=int, default=0, help='Index of the circulant eigenvalues to use for the complex coefficients.')
parser.add_argument('--seed', type=int, default=12345, help='Seed for the random right hand side.')
parser.add_argument('--ref_level', type=int, default=3, help='Icosahedral sphere mesh refinement level with mesh hierarchy. 0 for no mesh hierarchy.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    Print(args)

# eigenvalues
nt, theta, alpha = args.nt, args.theta, args.alpha
dt = args.dt*units.hour

gamma = args.alpha**(np.arange(nt)/nt)
C1 = np.zeros(nt)
C2 = np.zeros(nt)

C1[:2] = [1/dt, -1/dt]
C2[:2] = [theta, 1-theta]

D1 = np.sqrt(nt)*fft(gamma*C1)
D2 = np.sqrt(nt)*fft(gamma*C2)
freqs = fftfreq(nt, dt)

d1 = D1[args.eigenvalue]
d2 = D2[args.eigenvalue]

d1c = cpx.ComplexConstant(d1)
d2c = cpx.ComplexConstant(d2)

dhat = (d1/d2) / (1/(theta*dt))

mesh = swe.create_mg_globe_mesh(ref_level=args.ref_level,
                                coords_degree=1)
x = fd.SpatialCoordinate(mesh)

# case parameters
g = earth.Gravity
f = lcase.coriolis_expression(*x)
H = lcase.H

# function spaces
V = swe.default_function_space(mesh)
W = cpx.FunctionSpace(V)

Wu, Wh = W.subfunctions

Vu, Vh = V.subfunctions
Vub = fd.FunctionSpace(mesh, fd.BrokenElement(Vu.ufl_element()))
Vt = fd.FunctionSpace(mesh, "HDivT", Vu.ufl_element().degree())
Vtr = Vub*Vh*Vt

Wtr = cpx.FunctionSpace(Vtr)

Wub, _, Wt = Wtr.subfunctions


# shallow water equation forms
def form_mass(u, h, v, q):
    return swe.linear.form_mass(mesh, u, h, v, q)


def form_function(u, h, v, q, t=None):
    return swe.linear.form_function(mesh, g, H, f,
                                    u, h, v, q, t)


# random rhs
L = fd.Cofunction(W.dual())

# PETSc solver parameters
lu_params = {
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps'
}

condensed_params = lu_params

scpc_params = {
    "ksp_type": 'preonly',
    "mat_type": "matfree",
    "pc_type": "python",
    "pc_python_type": f"{__name__}.HybridisedSCPC",
    "hybridscpc_condensed_field": lu_params
}

rtol = 1e-3
params = {
    'ksp': {
        'monitor': None,
        'converged_rate': None,
        'rtol': rtol,
    },
}
params.update(scpc_params)

# trace component should have zero rhs
np.random.seed(args.seed)
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
                                    solver_parameters=params)

Print(f"dhat = {np.round(dhat, 4)}")
solver.solve()
