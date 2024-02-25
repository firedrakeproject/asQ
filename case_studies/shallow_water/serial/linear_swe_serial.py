
import firedrake as fd
from firedrake.petsc import PETSc

from utils import units
from utils import mg
from utils.planets import earth
import utils.shallow_water as swe
import utils.shallow_water.gravity_bumps as gcase

from utils.serial import SerialMiniApp

PETSc.Sys.popErrorHandler()

# get command arguments
import argparse
parser = argparse.ArgumentParser(
    description='Schreiber & Loft testcase using fully implicit linear SWE solver.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--ref_level', type=int, default=3, help='Refinement level of icosahedral grid.')
parser.add_argument('--nt', type=int, default=20, help='Number of time steps.')
parser.add_argument('--dt', type=float, default=0.05, help='Timestep in hours.')
parser.add_argument('--degree', type=float, default=swe.default_degree(), help='Degree of the depth function space.')
parser.add_argument('--theta', type=float, default=0.5, help='Parameter for implicit theta method. 0.5 for trapezium rule, 1 for backwards Euler.')
parser.add_argument('--filename', type=str, default='swe', help='Name of output vtk files')
parser.add_argument('--show_args', action='store_true', default=True, help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Setting up --- === ###')
PETSc.Sys.Print('')

# icosahedral mg mesh
mesh = swe.create_mg_globe_mesh(ref_level=args.ref_level, coords_degree=1)
x = fd.SpatialCoordinate(mesh)

# time step
dt = args.dt*units.hour

# shallow water equation function spaces (velocity and depth)
W = swe.default_function_space(mesh, degree=args.degree)

# parameters
g = earth.Gravity
H = gcase.H
f = gcase.coriolis_expression(*x)

# initial conditions
w_initial = fd.Function(W)
u_initial, h_initial = w_initial.subfunctions

u_initial.project(gcase.velocity_expression(*x))
h_initial.project(gcase.depth_expression(*x))

# broken space
Vu, Vh = W.subfunctions

Vub = fd.FunctionSpace(mesh, fd.BrokenElement(Vu.ufl_element()))
Tr = fd.FunctionSpace(mesh, "HDivT", Vu.ufl_element().degree())
Wtr = Vub*Vh*Tr


# shallow water equation forms
def form_function(u, h, v, q, t):
    return swe.linear.form_function(mesh, g, H, f,
                                    u, h, v, q, t)


def form_mass(u, h, v, q):
    return swe.linear.form_mass(mesh, u, h, v, q)


def form_mass_tr(u, h, tr, v, q, s):
    return form_mass(u, h, v, q)


def form_function_tr(u, h, tr, v, q, dtr, t=None):
    K = form_function(u, h, v, q, t)
    n = fd.FacetNormal(mesh)
    Khybr = (
        g*fd.jump(v, n)*tr('+')
        + fd.jump(u, n)*dtr('+')
    )*fd.dS
    return K + Khybr


class HybridisedSCPC(fd.PCBase):
    def initialize(self, pc):
        if pc.getType() != "python":
            raise ValueError("Expecting PC type python")

        from broken_projections import BrokenHDivProjector
        self.projector = BrokenHDivProjector(Vu)

        self.x = fd.Cofunction(W.dual())
        self.y = fd.Function(W)

        self.xu, self.xh = self.x.subfunctions
        self.yu, self.yh = self.y.subfunctions

        self.xtr = fd.Cofunction(Wtr.dual())
        self.ytr = fd.Function(Wtr)

        self.xbu, self.xbh, _ = self.xtr.subfunctions
        self.ybu, self.ybh, _ = self.ytr.subfunctions

        utr = fd.TrialFunction(Wtr)
        vtr = fd.TestFunction(Wtr)

        utrs = fd.split(utr)
        vtrs = fd.split(vtr)

        dt1 = fd.Constant(1/dt)
        tht = fd.Constant(args.theta)
        M = form_mass_tr(*utrs, *vtrs)
        K = form_function_tr(*utrs, *vtrs)

        A = dt1*M + tht*K
        L = self.xtr

        condensed_params = gamg_sparams
        scpc_params = {
            "mat_type": "matfree",
            "ksp_type": "preonly",
            "pc_type": "python",
            "pc_python_type": "firedrake.SCPC",
            'pc_sc_eliminate_fields': '0, 1',
            'condensed_field': condensed_params
        }

        problem = fd.LinearVariationalProblem(A, L, self.ytr)
        self.solver = fd.LinearVariationalSolver(
            problem, solver_parameters=scpc_params)

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


# solver parameters for the implicit solve
factorisation_params = {
    'ksp_type': 'preonly',
    'pc_factor_mat_ordering_type': 'rcm',
    # 'pc_factor_reuse_ordering': None,
    # 'pc_factor_reuse_fill': None,
}

lu_params = {'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}
lu_params.update(factorisation_params)

ilu_params = {'pc_type': 'ilu'}
ilu_params.update(factorisation_params)

icc_params = {'pc_type': 'icc'}
icc_params.update(factorisation_params)

patch_parameters = {
    'pc_patch': {
        'save_operators': True,
        'partition_of_unity': True,
        'sub_mat_type': 'seqdense',
        'construct_dim': 0,
        'construct_type': 'vanka',
        'local_type': 'additive',
        'precompute_element_tensors': True,
        'symmetrise_sweep': False
    },
    'sub': {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_shift_type': 'nonzero',
    }
}

mg_parameters = {
    'levels': {
        'ksp_type': 'gmres',
        'ksp_max_it': 5,
        'pc_type': 'python',
        'pc_python_type': 'firedrake.PatchPC',
        'patch': patch_parameters
    },
    'coarse': {
        'pc_type': 'python',
        'pc_python_type': 'firedrake.AssembledPC',
        'assembled': lu_params
    },
}

mg_sparams = {
    'mat_type': 'matfree',
    'pc_type': 'mg',
    'pc_mg_cycle_type': 'w',
    'pc_mg_type': 'multiplicative',
    'mg': mg_parameters
}

gamg_sparams = {
    'ksp_type': 'fgmres',
    'ksp_rtol': 1e-10,
    'ksp_converged_rate': None,
    'pc_type': 'gamg',
    'pc_mg_cycle_type': 'v',
    'pc_mg_type': 'full',
    'mg_levels': {
        'ksp_type': 'gmres',
        'ksp_max_it': 5,
        'pc_type': 'bjacobi',
        'sub': ilu_params,
    },
    'mg_coarse': lu_params,
}

hybridization_sparams = {
    "mat_type": "matfree",
    "pc_type": "python",
    "pc_python_type": "firedrake.HybridizationPC",
    # "hybridization": lu_params
    "hybridization": gamg_sparams
}

scpc_sparams = {
    "mat_type": "matfree",
    "pc_type": "python",
    "pc_python_type": f"{__name__}.HybridisedSCPC",
}

atol = 1e2
sparameters = {
    'snes_type': 'ksponly',
    'snes': {
        # 'monitor': None,
        # 'converged_reason': None,
        'atol': atol,
        'rtol': 1e-10,
        'stol': 1e-12,
        'lag_jacobian': -2,
        'lag_jacobian_persists': None,
        'lag_preconditioner': -2,
        'lag_preconditioner_persists': None,
    },
    'ksp_type': 'fgmres',
    'ksp': {
        'atol': atol,
        'rtol': 1e-10,
        'stol': 1e-12,
        'monitor': None,
        'converged_rate': None
    },
}

# sparameters.update(lu_params)
# sparameters.update(mg_sparams)
# sparameters.update(hybridization_sparams)
sparameters.update(scpc_sparams)

# set up nonlinear solver
miniapp = SerialMiniApp(dt, args.theta,
                        w_initial,
                        form_mass,
                        form_function,
                        sparameters)

miniapp.nlsolver.set_transfer_manager(mg.ManifoldTransferManager())

PETSc.Sys.Print('### === --- Timestepping loop --- === ###')
linear_its = 0
nonlinear_its = 0

ofile = fd.File('output/'+args.filename+'.pvd')
uout = fd.Function(u_initial.function_space(), name='velocity')
hout = fd.Function(h_initial.function_space(), name='depth')

uout.assign(u_initial)
hout.assign(h_initial - gcase.H)
ofile.write(uout, hout, time=0)


def preproc(app, step, t):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'=== --- Timestep {step} --- ===')
    PETSc.Sys.Print('')


def postproc(app, step, t):
    global linear_its
    global nonlinear_its

    linear_its += app.nlsolver.snes.getLinearSolveIterations()
    nonlinear_its += app.nlsolver.snes.getIterationNumber()

    u, h = app.w0.subfunctions
    uout.assign(u)
    hout.assign(h-gcase.H)
    ofile.write(uout, hout, time=t/units.hour)


miniapp.solve(args.nt,
              preproc=preproc,
              postproc=postproc)

PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Iteration counts --- === ###')
PETSc.Sys.Print('')

PETSc.Sys.Print(f'linear iterations: {linear_its} | iterations per timestep: {linear_its/args.nt}')
PETSc.Sys.Print(f'nonlinear iterations: {nonlinear_its} | iterations per timestep: {nonlinear_its/args.nt}')
PETSc.Sys.Print('')
