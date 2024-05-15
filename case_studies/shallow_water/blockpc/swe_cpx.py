import firedrake as fd
from firedrake.petsc import PETSc
from asQ.complex_proxy import vector as cpx

from utils import shallow_water as swe
from utils.planets import earth
from utils.misc import function_mean
from utils import units

import numpy as np
from scipy.fft import fft, fftfreq

PETSc.Sys.popErrorHandler()
Print = PETSc.Sys.Print


# PC forming approximate hybridisable system (without advection)
# solve it using hybridisation and then return the DG part
# (for use in a Schur compement setup)
class ApproxHybridPC(fd.PCBase):
    def initialize(self, pc):
        if pc.getType() != "python":
            raise ValueError("Expecting PC type python")

        appctx = self.get_appctx(pc)

        # get function space
        V = appctx['uref'].function_space()

        # input and output functions
        _, Vh = V.subfunctions
        self.xfstar = fd.Cofunction(Vh.dual())
        self.xf = fd.Function(Vh)  # result of riesz map of the above

        self.wf = fd.Function(V)  # solution of expanded problem
        _, self.yf = self.wf.subfunctions  # residual result

        w = fd.TrialFunction(V)
        v = fd.TestFunction(V)
        _, q = fd.split(v)

        # doesn't actually matter what the form is because
        # the HybridisedSCPC pulls form_{mass,function} off
        # the appctx and uses those.
        Af = fd.inner(w, v)*fd.dx
        Lf = fd.inner(q, self.xf)*fd.dx

        subpc_params = {
            'mat_type': 'matfree',
            'ksp_type': 'preonly',
            'pc_type': 'python',
            "pc_python_type": f"{__name__}.HybridisedSCPC",
            "hybridscpc_condensed_field": {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            }
        }

        problem = fd.LinearVariationalProblem(Af, Lf, self.wf)
        self.solver = fd.LinearVariationalSolver(
            problem,
            solver_parameters=subpc_params,
            appctx=appctx)

    def update(self, pc):
        pass

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError

    def apply(self, pc, x, y):
        # copy petsc vec into Function
        with self.xfstar.dat.vec_wo as v:
            x.copy(v)
        self.xf.assign(self.xfstar.riesz_representation())

        self.wf.assign(0)
        self.solver.solve()

        # copy petsc vec into Function
        with self.yf.dat.vec_ro as v:
            v.copy(y)


def read_checkpoint(checkpoint_name, funcname, index, ref_level=0):
    with fd.CheckpointFile(f"{checkpoint_name}.h5", "r") as checkpoint:
        mesh = checkpoint.load_mesh()
        coriolis = checkpoint.load_function(mesh, "coriolis")
        topography = checkpoint.load_function(mesh, "topography")

        if index < 0:
            u = checkpoint.load_function(mesh, funcname)
        else:
            u = checkpoint.load_function(mesh, funcname, idx=index)

    if ref_level == 0:
        return mesh, u, coriolis, topography
    else:
        mesh_new = swe.create_mg_globe_mesh(base_level=1,
                                            ref_level=ref_level,
                                            coords_degree=1)

        V = fd.FunctionSpace(mesh_new, u.function_space().ufl_element())
        Vu, Vh = V.subfunctions

        unew = fd.Function(V)
        coriolis_new = fd.Function(Vh)
        topography_new = fd.Function(Vh)

        pairs = (
            zip(u.dat, unew.dat),
            zip(coriolis.dat, coriolis_new.dat),
            zip(topography.dat, topography_new.dat)
        )

        for pair in pairs:
            for src, dst in pair:
                dst.data[:] = src.data[:]

        return mesh_new, unew, coriolis_new, topography_new


# get command arguments
import argparse
description = "Test preconditioners for the complex block for the nonlinear shallow water equations.\
    - Calculates the circulant eigenvalues from the given arguments, and tests either one\
      or all eigenvalue pairs with random right hand sides.\
    - Requires a CheckpointFile containing the state for the shallow water equation as\
      well as Functions for 'topography' and for 'coriolis'.\
    - If using the multigrid method, the ref_level of the mesh in the CheckpointFile must\
      be provided, and the script must be run serially. This is because CheckpointFile\
      can't save a MeshHierarchy."

parser = argparse.ArgumentParser(
    description=description,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--dt', type=float, default=1.0, help='Timestep in hours (used to calculate the circulant eigenvalues).')
parser.add_argument('--nt', type=int, default=16, help='Number of timesteps (used to calculate the circulant eigenvalues).')
parser.add_argument('--theta', type=float, default=0.5, help='Parameter for implicit theta method. 0.5 for trapezium rule, 1 for backwards Euler (used to calculate the circulant eigenvalues).')
parser.add_argument('--alpha', type=float, default=1e-3, help='Circulant parameter (used to calculate the circulant eigenvalues).')
parser.add_argument('--eigenvalue', type=int, default=-1, help='Index of the circulant eigenvalues to use for the complex coefficients. -1 for all')
parser.add_argument('--seed', type=int, default=12345, help='Seed for the random right hand side.')
parser.add_argument('--nrhs', type=int, default=1, help='Number of random right hand sides to solve for.')
parser.add_argument('--method', type=str, default='lswe', choices=['lu', 'mg', 'lswe', 'hybr', 'schur', 'composite'], help='Preconditioning method to use.')
parser.add_argument('--rtol', type=float, default=1e-5, help='Relative tolerance for solution of each block.')
parser.add_argument('--foutname', type=str, default='iterations', help='Name of output file to write iteration counts.')
parser.add_argument('--checkpoint', type=str, default='swe_series', help='Name of checkpoint file.')
parser.add_argument('--funcname', type=str, default='swe', help='Name of the Function in the checkpoint file.')
parser.add_argument('--index', type=int, default=0, help='Index of Function in checkpoint file.')
parser.add_argument('--ref_level', type=int, default=0, help='Icosahedral sphere mesh refinement level with mesh hierarchy. 0 for no mesh hierarchy.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')
parser.add_argument('-v', action='store_true', help='Print KSP convergence.')
parser.add_argument('-vv', action='store_true', help='Print KSP convergence and monitor.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    Print(args)

verbosity = 0
if args.v:
    verbosity = 1
if args.vv:
    verbosity = 2

mesh, u, f, b = read_checkpoint(args.checkpoint, args.funcname,
                                args.index, args.ref_level)

x = fd.SpatialCoordinate(mesh)
outward_normals = fd.CellNormal(mesh)
facet_normals = fd.FacetNormal(mesh)

# case parameters
g = earth.Gravity

uref, href = u.subfunctions
H = function_mean(href)


# shallow water equation forms
def form_mass(u, h, v, q):
    return swe.nonlinear.form_mass(mesh, u, h, v, q)


def form_function(u, h, v, q, t=None):
    return swe.nonlinear.form_function(mesh, g, b, f,
                                       u, h, v, q, t)


def aux_form_function(u, h, v, q, t=None):
    return swe.linear.form_function(mesh, g, H, f,
                                    u, h, v, q, t)


V = u.function_space()
W = cpx.FunctionSpace(V)

# reference state to linearise around
wref = fd.Function(W)
cpx.set_real(wref, u)
cpx.set_imag(wref, u)

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

# block forms
M, d1r, d1i = cpx.BilinearForm(W, d1c, form_mass, return_z=True)
K, d2r, d2i = cpx.derivative(d2c, form_function, wref, return_z=True)

A = M + K

# random rhs
# use one-form because Cofunction doesn't work with fieldsplit
l = fd.Function(W)
L = fd.inner(fd.TestFunction(W), l)*fd.dx

# PETSc solver parameters

lu_params = {
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
    'pc_factor_reuse_ordering': None,
    'pc_factor_reuse_fill': None,
    'pc_factor_shift_type': 'nonzero',
}

mg_sparams = {
    "mat_type": "matfree",
    'pc_type': 'mg',
    'pc_mg_cycle_type': 'v',
    'pc_mg_type': 'full',
    'mg': {
        'levels': {
            'ksp_type': 'gmres',
            'ksp_max_it': 4,
            'pc_type': 'python',
            'pc_python_type': 'firedrake.PatchPC',
            'patch': {
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
        },
        'coarse': {
            'pc_type': 'python',
            'pc_python_type': 'firedrake.AssembledPC',
            'assembled': lu_params
        },
    }
}

from utils.hybridisation import HybridisedSCPC  # noqa: F401
hybridization_sparams = {
    "mat_type": "matfree",
    "pc_type": "python",
    "pc_python_type": f"{__name__}.HybridisedSCPC",
    "hybridscpc_condensed_field": lu_params,
}

aux_sparams = {
    "mat_type": "matfree",
    "pc_type": "python",
    "pc_python_type": "asQ.AuxiliaryComplexBlockPC",
    "aux": lu_params,
}

# using fgmres on the schur complement results in an
# outer residual drop of approximate the same factor
# as the schur complement residual.
schurhybr_sparams = {
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'schur',
    'pc_fieldsplit_schur_fact_type': 'full',
    'fieldsplit_0': lu_params,
    'fieldsplit_1': {
        'ksp_type': 'fgmres',
        'ksp_converged_rate': None,
        'gmres_restart': 60,
        'ksp_rtol': 0.9*args.rtol,
        'pc_type': 'python',
        'pc_python_type': f'{__name__}.ApproxHybridPC',
    },
}

# linear operator is much more effective with gmres
# but this means we need fgmres on the patch smoother
composite_sparams = {
    "pc_type": "composite",
    "pc_composite_type": "multiplicative",
    "pc_composite_pcs": "ksp,ksp",
    "sub_0": {
        "ksp_ksp_type": "gmres",
        "ksp_ksp_max_it": 2,
        "ksp_ksp_convergence_test": 'skip',
        "ksp_ksp_converged_maxits": None,
        "ksp": aux_sparams,
    },
    "sub_1_ksp": {
        "ksp_type": "fgmres",
        "ksp_max_it": 1,
        "ksp_convergence_test": 'skip',
        "ksp_converged_maxits": None,
        "pc_type": "python",
        "pc_python_type": "firedrake.PatchPC",
        'patch': {
            'pc_patch': {
                'save_operators': True,
                'partition_of_unity': True,
                'sub_mat_type': 'seqdense',
                'construct_dim': 0,
                'construct_type': 'star',
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
    }
}

sparams = {
    'ksp_rtol': args.rtol,
    'ksp_type': 'fgmres',
}

if args.method == 'lu':
    sparams.update(lu_params)
elif args.method == 'mg':
    sparams.update(mg_sparams)
elif args.method == 'lswe':
    sparams.update(aux_sparams)
elif args.method == 'hybr':
    sparams.update(hybridization_sparams)
elif args.method == 'schur':
    sparams.update(schurhybr_sparams)
elif args.method == 'composite':
    sparams.update(composite_sparams)
else:
    raise ValueError(f"Unknown method {args.method}")

if verbosity > 0:
    sparams["ksp_converged_rate"] = None
if verbosity > 1:
    sparams["ksp_monitor"] = None

appctx = {
    'cpx': cpx,
    'uref': wref,
    'tref': None,
    'd1': d1,
    'd2': d2,
    'bcs': [],
    'form_mass': form_mass,
    'form_function': form_function,
    'aux_form_function': aux_form_function,
    'hybridscpc_form_function': aux_form_function,
}

wout = fd.Function(W)

problem = fd.LinearVariationalProblem(A, L, wout,
                                      constant_jacobian=True)

fstr = lambda x, n: str(round(x, n)).rjust(3+n)

neigs = args.nt//2+1 if args.eigenvalue < 0 else 1
eigs = [*range(neigs)] if args.eigenvalue < 0 else [args.eigenvalue,]
nits = np.zeros((neigs, args.nrhs), dtype=int)
for i in range(neigs):
    k = eigs[i]
    freq = freqs[k]
    d1 = D1[k]
    d2 = D2[k]
    dhat = (d1/d2) / (1/(theta*dt))

    appctx['d1'] = d1
    appctx['d2'] = d2

    d1r.assign(d1.real)
    d1i.assign(d1.imag)

    d2r.assign(d2.real)
    d2i.assign(d2.imag)

    solver = fd.LinearVariationalSolver(
        problem, appctx=appctx,
        options_prefix=f"block_{k}",
        solver_parameters=sparams)
    solver.invalidate_jacobian()

    if verbosity > 0:
        PETSc.Sys.Print(f"Eigenvalues {str(k).rjust(3)}:\n    d1 = {np.round(d1,4)}, d2 = {np.round(d2,4)}, dhat = {np.round(dhat,4)}")

    np.random.seed(args.seed)
    for j in range(args.nrhs):
        wout.assign(0)

        for dat in l.dat:
            dat.data[:] = np.random.random_sample(dat.data.shape)
        solver.solve()
        nits[i, j] = solver.snes.getLinearSolveIterations()

meanit = np.mean(nits, axis=1)
maxit = np.max(nits, axis=1)
minit = np.min(nits, axis=1)
stdit = np.std(nits, axis=1)

for i in range(neigs):
    PETSc.Sys.Print(f"Eigenvalue {str(i).rjust(2)} iterations: mean = {fstr(meanit[i], 1)} | max = {str(maxit[i]).rjust(2)} | min = {str(minit[i]).rjust(2)} | std = {fstr(stdit[i], 2)}")

PETSc.Sys.Print(f"min iterations: {round(min(meanit), 3)}, max iterations: {round(max(meanit), 3)}")
PETSc.Sys.Print([round(it, 3) for it in np.mean(nits, axis=1)])

if fd.COMM_WORLD.rank == 0:
    with open(f"{args.foutname}.dat", "w") as f:
        f.write("# " + "   ".join(["index", "freq", "mean", "max", "min", "std"]) + "\n")
        for i in range(neigs):
            f.write("   ".join(map(str, [i, freqs[i], meanit[i], maxit[i], minit[i], stdit[i]])))
            f.write("\n")
