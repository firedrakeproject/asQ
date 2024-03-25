import firedrake as fd
from firedrake.petsc import PETSc

import utils.shallow_water.gravity_bumps as lcase
from utils import shallow_water as swe
from utils.planets import earth
from utils import units

import numpy as np

PETSc.Sys.popErrorHandler()
Print = PETSc.Sys.Print


class HybridisedSCPC(fd.PCBase):
    def initialize(self, pc):
        if pc.getType() != "python":
            raise ValueError("Expecting PC type python")

        from ufl import replace
        from utils.broken_projections import BrokenHDivProjector

        appctx = self.get_appctx(pc)
        _, P = pc.getOperators()
        ctx = P.getPythonContext()
        test, trial = ctx.a.arguments()
        tests, trials = fd.split(test), fd.split(trial)

        V = test.function_space()
        mesh = V.mesh()

        Vu, Vh = V.subfunctions
        Vub = fd.FunctionSpace(mesh, fd.BrokenElement(Vu.ufl_element()))
        Tr = fd.FunctionSpace(mesh, "HDivT", Vu.ufl_element().degree())
        Vtr = Vub*Vh*Tr

        # index of hdiv space
        iu = 0

        self.projector = BrokenHDivProjector(Vu)

        self.x = fd.Cofunction(V.dual())
        self.y = fd.Function(V)

        self.xu, self.xh = self.x.subfunctions
        self.yu, self.yh = self.y.subfunctions

        self.xtr = fd.Cofunction(Vtr.dual())
        self.ytr = fd.Function(Vtr)

        self.xbu, self.xbh, _ = self.xtr.subfunctions
        self.ybu, self.ybh, _ = self.ytr.subfunctions

        utrs = fd.TrialFunctions(Vtr)
        vtrs = fd.TestFunctions(Vtr)

        # break the original form
        news = (*vtrs[:-1], *utrs[:-1])
        olds = (*tests, *trials)
        arg_map = {old: new for old, new in zip(olds, news)}
        A = replace(ctx.a, arg_map)

        # add the trace bit
        n = fd.FacetNormal(mesh)
        A += (
            fd.jump(vtrs[iu], n)*utrs[-1]('+')
            + fd.jump(utrs[iu], n)*vtrs[-1]('+')
        )*fd.dS

        Ltr = self.xtr

        scpc_params = {
            "mat_type": "matfree",
            "ksp_type": "preonly",
            "pc_type": "python",
            "pc_python_type": "firedrake.SCPC",
            "pc_sc_eliminate_fields": "0, 1",
        }

        problem = fd.LinearVariationalProblem(A, Ltr, self.ytr)
        self.solver = fd.LinearVariationalSolver(
            problem, appctx=appctx,
            solver_parameters=scpc_params,
            options_prefix=pc.getOptionsPrefix()+"hybridscpc")

    def apply(self, pc, x, y):
        # copy into unbroken vector
        with self.x.dat.vec_wo as v:
            x.copy(v)

        # break velocity
        self.projector.project(self.xu, self.xbu)

        # depth already broken
        self.xbh.assign(self.xh)

        self.ytr.assign(0)
        self.solver.solve()

        # mend velocity
        self.projector.project(self.ybu, self.yu)

        # depth already mended
        self.yh.assign(self.ybh)

        # copy out to petsc
        with self.y.dat.vec_ro as v:
            v.copy(y)

    def update(self, pc):
        pass

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError


# get command arguments
import argparse
parser = argparse.ArgumentParser(
    description='Test preconditioners for the complex linear shallow water equations.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--dt', type=float, default=1.0, help='Timestep in hours (used to calculate the circulant eigenvalues).')
parser.add_argument('--theta', type=float, default=0.5, help='Parameter for implicit theta method. 0.5 for trapezium rule, 1 for backwards Euler (used to calculate the circulant eigenvalues).')
parser.add_argument('--seed', type=int, default=12345, help='Seed for the random right hand side.')
parser.add_argument('--ref_level', type=int, default=3, help='Icosahedral sphere mesh refinement level with mesh hierarchy. 0 for no mesh hierarchy.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    Print(args)

# eigenvalues
theta = args.theta
dt = args.dt*units.hour

d1 = fd.Constant(1/dt)
d2 = fd.Constant(theta)

mesh = swe.create_mg_globe_mesh(ref_level=args.ref_level,
                                coords_degree=1)
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
lu_params = {
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'petsc'
}

sparams = {
    'ksp': {
        'monitor': None,
        'converged_rate': None,
    },
    'mat_type': 'matfree',
    'ksp_type': 'preonly',
    'pc_type': 'python',
    'pc_python_type': f'{__name__}.HybridisedSCPC',
    'hybridscpc_condensed_field': lu_params
}

# trace component should have zero rhs
np.random.seed(args.seed)
L.assign(0)
for ldat in L.dat:
    ldat.data[:] = np.random.rand(*(ldat.data.shape))

# block forms
us = fd.TrialFunctions(V)
vs = fd.TestFunctions(V)

M = form_mass(*us, *vs)
K = form_function(*us, *vs)

A = d1*M + d2*K

wout = fd.Function(V)
problem = fd.LinearVariationalProblem(A, L, wout)
solver = fd.LinearVariationalSolver(problem,
                                    solver_parameters=sparams)
solver.solve()
