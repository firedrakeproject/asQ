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
parser = argparse.ArgumentParser(
    description='Test preconditioners for the complex block for the Galewsky testcase.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--dt', type=float, default=1.0, help='Timestep in hours (used to calculate the circulant eigenvalues).')
parser.add_argument('--nt', type=int, default=16, help='Number of timesteps (used to calculate the circulant eigenvalues).')
parser.add_argument('--theta', type=float, default=0.5, help='Parameter for implicit theta method. 0.5 for trapezium rule, 1 for backwards Euler (used to calculate the circulant eigenvalues).')
parser.add_argument('--alpha', type=float, default=1e-3, help='Circulant parameter (used to calculate the circulant eigenvalues).')
parser.add_argument('--eigenvalue', type=int, default=0, help='Index of the circulant eigenvalues to use for the complex coefficients.')
parser.add_argument('--seed', type=int, default=12345, help='Seed for the random right hand side.')
parser.add_argument('--nrhs', type=int, default=1, help='Number of random right hand sides to solve for.')
parser.add_argument('--rtol', type=float, default=1e-5, help='Relative tolerance for solution of each block.')
parser.add_argument('--foutname', type=str, default='iterations', help='Name of output file to write iteration counts.')
parser.add_argument('--checkpoint', type=str, default='swe_series', help='Name of checkpoint file.')
parser.add_argument('--funcname', type=str, default='swe', help='Name of the Function in the checkpoint file.')
parser.add_argument('--index', type=int, default=0, help='Index of Function in checkpoint file.')
parser.add_argument('--ref_level', type=int, default=0, help='Icosahedral sphere mesh refinement level with mesh hierarchy. 0 for no mesh hierarchy.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')
parser.add_argument('--verbose', '-v', action='store_true', help='Print KSP outputs.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    Print(args)

mesh, u, f, b = read_checkpoint(args.checkpoint, args.funcname,
                                args.index, args.ref_level)

x = fd.SpatialCoordinate(mesh)

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


def aux_form_function(u, h, v, q, t):
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
# Print(f"D1 = {D1}")
# Print(f"D2 = {D2}")

# block forms
M, d1r, d1i = cpx.BilinearForm(W, d1c, form_mass, return_z=True)
K, d2r, d2i = cpx.derivative(d2c, form_function, wref, return_z=True)

A = M + K

# random rhs
L = fd.Cofunction(W.dual())

# PETSc solver parameters

linear_snes_params = {
    'lag_preconditioner': -2,
    'lag_preconditioner_persists': None,
}

factorisation_params = {
    'ksp_type': 'preonly',
    # 'pc_factor_mat_ordering_type': 'rcm',
    'pc_factor_reuse_ordering': None,
    'pc_factor_reuse_fill': None,
}

lu_params = {'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}
lu_params.update(factorisation_params)

ilu_params = {'pc_type': 'ilu'}
ilu_params.update(factorisation_params)

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
    "mat_type": "matfree",
    'pc_type': 'mg',
    'pc_mg_cycle_type': 'v',
    'pc_mg_type': 'full',
    'mg': mg_parameters
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

sparams = {
    'ksp_rtol': args.rtol,
    'ksp_type': 'gmres',
}
sparams.update(hybridization_sparams)
# sparams.update(aux_sparams)
# sparams.update(mg_sparams)

if args.verbose:
    sparams["ksp_monitor"] = None
    sparams["ksp_converged_rate"] = None

appctx = {
    'cpx': cpx,
    'uref': wref,
    'tref': None,
    'd1': d1c,
    'd2': d2c,
    'bcs': [],
    'form_mass': form_mass,
    'form_function': form_function,
    'aux_form_function': aux_form_function,
    'hybridscpc_form_function': aux_form_function,
}

wout = fd.Function(W)

problem = fd.LinearVariationalProblem(A, L, wout)
solver = fd.LinearVariationalSolver(problem, appctx=appctx,
                                    options_prefix="block",
                                    solver_parameters=sparams)

neigs = args.nt//2+1
nits = np.zeros((neigs, args.nrhs), dtype=int)
for i in range(neigs):
    freq = freqs[i]
    d1 = D1[i]
    d2 = D2[i]
    dhat = (d1/d2) / (1/(theta*dt))

    d1r.assign(d1.real)
    d1i.assign(d1.imag)

    d2r.assign(d2.real)
    d2i.assign(d2.imag)

    np.random.seed(args.seed)
    for j in range(args.nrhs):
        wout.assign(0)

        for dat in L.dat:
            dat.data[:] = np.random.random_sample(dat.data.shape)
        solver.solve()
        nits[i, j] = solver.snes.getLinearSolveIterations()

meanit = np.mean(nits, axis=1)
maxit = np.max(nits, axis=1)
minit = np.min(nits, axis=1)
stdit = np.std(nits, axis=1)

fstr = lambda x, n: str(round(x, n)).rjust(3+n)
for i in range(neigs):
    PETSc.Sys.Print(f"Eigenvalue {str(i).rjust(2)} iterations: mean = {fstr(meanit[i], 1)} | max = {str(maxit[i]).rjust(2)} | min = {str(minit[i]).rjust(2)} | std = {fstr(stdit[i], 2)}")

PETSc.Sys.Print(f"max iterations: {round(max(meanit), 3)}, min iterations: {round(min(meanit), 3)}")
PETSc.Sys.Print([round(it, 3) for it in np.mean(nits, axis=1)])

if fd.COMM_WORLD.rank == 0:
    with open(f"{args.foutname}.dat", "w") as f:
        f.write("# " + "   ".join(["index", "freq", "mean", "max", "min", "std"]) + "\n")
        for i in range(neigs):
            f.write("   ".join(map(str, [i, freqs[i], meanit[i], maxit[i], minit[i], stdit[i]])))
            f.write("\n")
