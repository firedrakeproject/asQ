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

d1 = 1 + 1j
d2 = 1

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
Tr = fd.FunctionSpace(mesh, "HDivT", Vu.ufl_element().degree())
Vtr = Vub*Vh*Tr

Wtr = cpx.FunctionSpace(Vtr)

Wub, _, Wt = Wtr.subfunctions


# shallow water equation forms
def form_mass(u, h, v, q):
    return swe.linear.form_mass(mesh, u, h, v, q)


def form_function(u, h, v, q, t=None):
    return swe.linear.form_function(mesh, g, H, f,
                                    u, h, v, q, t)


# shallow water equation forms with trace variable
def form_mass_tr(u, h, tr, v, q, s):
    return form_mass(u, h, v, q)


def form_function_tr(u, h, tr, v, q, dtr, t=None):
    K = form_function(u, h, v, q, t)
    n = fd.FacetNormal(mesh)
    K += (
        g*fd.jump(v, n)*tr('+')
    )*fd.dS
    return K


def form_trace(u, h, tr, v, q, dtr, t=None):
    n = fd.FacetNormal(mesh)
    Ktr = (
        fd.jump(u, n)*dtr('+')
    )*fd.dS
    return Ktr


class HybridisedSCPC(fd.PCBase):
    def initialize(self, pc):
        if pc.getType() != "python":
            raise ValueError("Expecting PC type python")

        from utils.broken_projections import BrokenHDivProjector
        self.projectorV = BrokenHDivProjector(Vu)
        self.projectorW = BrokenHDivProjector(Wu, Wub)

        self.v = fd.Function(self.projectorV.V)
        self.vb = fd.Function(self.projectorV.Vb)

        self.vs = fd.Cofunction(self.projectorV.V.dual())
        self.vsb = fd.Cofunction(self.projectorV.Vb.dual())

        self.x = fd.Cofunction(W.dual()).assign(0)
        self.y = fd.Function(W).assign(0)

        self.xu, self.xh = self.x.subfunctions
        self.yu, self.yh = self.y.subfunctions

        self.xtr = fd.Cofunction(Wtr.dual()).assign(0)
        self.ytr = fd.Function(Wtr).assign(0)

        self.xbu, self.xbh, _ = self.xtr.subfunctions
        self.ybu, self.ybh, _ = self.ytr.subfunctions

        M = cpx.BilinearForm(Wtr, d1c, form_mass_tr)
        K = cpx.BilinearForm(Wtr, d2c, form_function_tr)
        Tr = cpx.BilinearForm(Wtr, 1, form_trace)

        A = M + K + Tr
        L = self.xtr

        condensed_params = lu_params
        scpc_params = {
            # 'ksp_monitor': None,
            # 'ksp_converged_reason': None,
            "mat_type": "matfree",
            "ksp_type": "preonly",
            "pc_type": "python",
            "pc_python_type": "firedrake.SCPC",
            "pc_sc_eliminate_fields": "0, 1",
            "condensed_field": condensed_params
        }

        problem = fd.LinearVariationalProblem(A, L, self.ytr)
        self.solver = fd.LinearVariationalSolver(
            problem, solver_parameters=scpc_params)

    def _break(self, w, wb):
        # self.projectorW.project(w, wb)
        self._project(w, wb, self.vs, self.vsb)

    def _mend(self, w, wb):
        # self.projectorW.project(wb, w)
        self._project(wb, w, self.vb, self.v)

    def _project(self, wsrc, wdst, vsrc, vdst):
        sr, si = wsrc.sub(0), wsrc.sub(1)
        dr, di = wdst.sub(0), wdst.sub(1)
        for src, dst in ((sr, dr), (si, di)):
            for vdat, sdat in zip(vsrc.dat, src.dat):
                vdat.data[:] = sdat.data[:]
            self.projectorV.project(vsrc, vdst)
            for vdat, sdat in zip(vdst.dat, dst.dat):
                sdat.data[:] = vdat.data[:]

    def apply(self, pc, x, y):
        # copy into unbroken vector
        with self.x.dat.vec_wo as v:
            x.copy(v)

        # break each component of velocity
        self._break(self.xu, self.xbu)

        # depth already broken
        self.xbh.assign(self.xh)

        self.ytr.assign(0)
        self.solver.solve()

        # mend each component of velocity
        self._mend(self.yu, self.ybu)

        # depth already mended
        self.yh.assign(self.ybh)

        # copy out to petsc
        with self.y.dat.vec_ro as v:
            v.copy(y)

    def update(self, pc):
        pass

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError


# random rhs
L = fd.Cofunction(W.dual())

# PETSc solver parameters
lu_params = {
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    # 'pc_factor_mat_ordering_type': 'rcm',
    'pc_factor_mat_solver_type': 'mumps'
}

scpc_sparams = {
    "ksp_type": 'gmres',
    "mat_type": "matfree",
    "pc_type": "python",
    "pc_python_type": f"{__name__}.HybridisedSCPC",
}

rtol = 1e-3
sparams = {
    'ksp': {
        'monitor': None,
        'converged_rate': None,
        'rtol': rtol,
        # 'view': None
    },
}
# sparams.update(lu_params)
sparams.update(scpc_sparams)

# trace component should have zero rhs
np.random.seed(args.seed)
L.assign(0)
for dat in L.dat:
    dat.data[:] = np.random.rand(*(dat.data.shape))

# block forms
M = cpx.BilinearForm(W, d1c, form_mass)
K = cpx.BilinearForm(W, d2c, form_function)

A = M + K

wout = fd.Function(W).assign(0)
problem = fd.LinearVariationalProblem(A, L, wout)
solver = fd.LinearVariationalSolver(problem,
                                    solver_parameters=sparams)

Print(f"dhat = {np.round(dhat, 4)}")
solver.solve()
