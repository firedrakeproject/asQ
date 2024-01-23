import firedrake as fd
from firedrake.petsc import PETSc
from asQ.complex_proxy import vector as cpx

import utils.shallow_water.gravity_bumps as lcase
from utils import shallow_water as swe
from utils.planets import earth
from utils import units

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


# generate linear solver
def make_solver(W, form_mass, form_function,
                d1c, d2c, L, sparams):

    # block forms
    M = cpx.BilinearForm(W, d1c, form_mass)
    K = cpx.BilinearForm(W, d2c, form_function)

    A = M + K

    wout = fd.Function(W)
    problem = fd.LinearVariationalProblem(A, L, wout)
    solver = fd.LinearVariationalSolver(problem,
                                        solver_parameters=sparams)
    return wout, solver


# shallow water equation forms
def form_mass(u, h, v, q):
    return swe.linear.form_mass(mesh, u, h, v, q)


def form_function(u, h, v, q, t=None):
    return swe.linear.form_function(mesh, g, H, f,
                                    u, h, v, q, t)


# shallow water equation forms with trace variable
def form_mass_tr(u, h, tr, v, q, s):
    return swe.linear.form_mass(mesh, u, h, v, q)


def form_function_tr(u, h, tr, v, q, s, t=None):
    K = swe.linear.form_function(mesh, g, H, f,
                                 u, h, v, q, t)
    n = fd.FacetNormal(mesh)
    Khybr = (
        g*fd.jump(v, n)*tr('+')
        + fd.jump(u, n)*s('+')
    )*fd.dS

    return K + Khybr


V = swe.default_function_space(mesh)
W = cpx.FunctionSpace(V)

Vu, Vh = V.subfunctions
Tr = fd.FunctionSpace(mesh, "HDivT", Vu.ufl_element().degree())
Vtr = Vu*Vh*Tr
Wtr = cpx.FunctionSpace(Vtr)

# random rhs
L = fd.Cofunction(W.dual())
Ltr = fd.Cofunction(Wtr.dual())

# PETSc solver parameters
lu_params = {
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps'
}

rtol = 1e-5
sparams = {
    'ksp': {
        'monitor': None,
        'converged_reason': None,
        'rtol': rtol,
        # 'view': None
    },
}
sparams.update(lu_params)

sparams_tr = {
    'ksp': {
        'monitor': None,
        'converged_reason': None,
        'rtol': rtol,
        # 'view': None
    },
    'mat_type': 'matfree',
    'ksp_type': 'gmres',
    'pc_type': 'python',
    'pc_python_type': 'firedrake.SCPC',
    'pc_sc_eliminate_fields': '0, 1',
    'condensed_field': lu_params,
    # 'condensed_field_ksp': {
    #     'monitor': None,
    #     'converged_reason': None,
    # }
}

# trace component should have zero rhs
np.random.seed(args.seed)
L.assign(0)
Ltr.assign(0)
for dat in L.dat:
    dat.data[:] = np.random.rand(*(dat.data.shape))

Ltr.subfunctions[0].assign(L.subfunctions[0])
Ltr.subfunctions[1].assign(L.subfunctions[1])

w, solver = make_solver(W, form_mass, form_function,
                        d1c, d2c, L, sparams)
wtr, solver_tr = make_solver(Wtr, form_mass_tr, form_function_tr,
                             d1c, d2c, Ltr, sparams_tr)

PETSc.Sys.Print("")
PETSc.Sys.Print("Solving original system")
solver.solve()
PETSc.Sys.Print("")
PETSc.Sys.Print("Solving system with trace")
solver_tr.solve()

u, h = w.subfunctions
utr, htr, tr = wtr.subfunctions
uerr = fd.errornorm(u, utr)/fd.norm(u)
herr = fd.errornorm(h, htr)/fd.norm(h)
PETSc.Sys.Print("")
PETSc.Sys.Print(f"Velocity errornorm = {uerr}")
PETSc.Sys.Print(f"Depth errornorm    = {herr}")
