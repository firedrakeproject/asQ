from math import pi, cos, sin
import numpy as np

import firedrake as fd
from firedrake.petsc import PETSc
from asQ.complex_proxy import vector as cpx

import argparse

parser = argparse.ArgumentParser(
    description='ParaDiag timestepping for scalar advection of a Gaussian bump in a periodic square with DG in space and implicit-theta in time. Based on the Firedrake DG advection example https://www.firedrakeproject.org/demos/DG_advection.py.html',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--nx', type=int, default=16, help='Number of cells along each square side.')
parser.add_argument('--cfl', type=float, default=0.8, help='Convective CFL number.')
parser.add_argument('--angle', type=float, default=pi/6, help='Angle of the convective velocity.')
parser.add_argument('--degree', type=int, default=1, help='Degree of the scalar and velocity spaces.')
parser.add_argument('--theta', type=float, default=0.5, help='Parameter for the implicit theta timestepping method.')
parser.add_argument('--width', type=float, default=0.2, help='Width of the Gaussian bump.')
parser.add_argument('--show_args', action='store_true', default=True, help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

# Calculate the timestep from the CFL number
umax = 1.
dx = 1./args.nx
dt = args.cfl*dx/umax

# # # === --- domain --- === # # #

# The mesh needs to be created with the spatial communicator
mesh = fd.PeriodicUnitSquareMesh(args.nx, args.nx, quadrilateral=True)

# We use a discontinuous Galerkin space for the advected scalar
# and a continuous Galerkin space for the advecting velocity field
V = fd.FunctionSpace(mesh, "DQ", args.degree)
W = fd.VectorFunctionSpace(mesh, "CG", args.degree)

# The advecting velocity field is constant and directed at an angle to the x-axis
u = fd.Function(W, name='velocity')
u.interpolate(fd.as_vector((umax*cos(args.angle), umax*sin(args.angle))))


# # # === --- finite element forms --- === # # #


# The time-derivative mass form for the scalar advection equation.
# asQ assumes that the mass form is linear so here
# q is a TrialFunction and phi is a TestFunction
def form_mass(q, phi):
    return phi*q*fd.dx


# The DG advection form for the scalar advection equation.
# asQ assumes that the function form is nonlinear so here
# q is a Function and phi is a TestFunction
def form_function(q, phi, t=None):
    # upwind switch
    n = fd.FacetNormal(mesh)
    un = fd.Constant(0.5)*(fd.dot(u, n) + abs(fd.dot(u, n)))

    # integration over element volume
    int_cell = q*fd.div(phi*u)*fd.dx

    # integration over internal facets
    int_facet = (phi('+')-phi('-'))*(un('+')*q('+')-un('-')*q('-'))*fd.dS

    return int_facet - int_cell


# # # === --- Setup problem --- === # # #

C = cpx.FunctionSpace(V)
w = fd.Function(C)

# original problem
d1 = 0.5 + 2j
d2 = 1

D1 = d1/dt
D2 = d2*args.theta

# shift preconditioning
d1pc = d1
d1pc = complex(1, d1.imag)
d2pc = 1

D1pc = d1pc/dt
D2pc = d2pc*args.theta

# finite element forms

M = cpx.BilinearForm(C, D1, form_mass)
K = cpx.BilinearForm(C, D2, form_function)

A = M + K

L = fd.Cofunction(C.dual())
np.random.seed(12345)
for dat in L.dat:
    dat.data[:] = np.random.rand(*(dat.data.shape))


# # # === --- PETSc solver parameters --- === # # #


# The PETSc solver parameters used to solve the
# blocks in step (b) of inverting the ParaDiag matrix.

lu_pc = {
    'ksp_type': 'preonly',
    'pc_type': 'lu',
}

rtol = 1e-10
sparams = {
    'ksp': {
        'monitor': None,
        'converged_reason': None,
        'rtol': rtol,
    },
    'ksp_type': 'gmres',
    'pc_type': 'python',
    'pc_python_type': 'asQ.AuxiliaryBlockPC',
    'aux': {
        'pc_type': 'ilu'
    }
}

# All of these are given to the block solver by the diagpc
# but we need to do it manually here.
# We can change form_mass and form_function to precondition
# with a different operator.
# We can change d1 and d2 to use shift preconditioning.

appctx = {
    'cpx': cpx,
    'u0': w,
    't0': None,
    'd1': D1pc,
    'd2': D2pc,
    'bcs': [],
    'form_mass': form_mass,
    'form_function': form_function
}

# # # === --- Setup solver --- === # # #

problem = fd.LinearVariationalProblem(A, L, w)
solver = fd.LinearVariationalSolver(problem, appctx=appctx,
                                    solver_parameters=sparams)

solver.solve()
