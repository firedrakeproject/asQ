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
parser.add_argument('--checkpoint', type=str, default='swe_series', help='Name of checkpoint file.')
parser.add_argument('--funcname', type=str, default='swe', help='Name of the Function in the checkpoint file.')
parser.add_argument('--index', type=int, default=0, help='Index of Function in checkpoint file.')
parser.add_argument('--ref_level', type=int, default=0, help='Icosahedral sphere mesh refinement level with mesh hierarchy. 0 for no mesh hierarchy.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    Print(args)

mesh, u, coriolis, topography = read_checkpoint(args.checkpoint, args.funcname,
                                                args.index, args.ref_level)

x = fd.SpatialCoordinate(mesh)

# case parameters
gravity = earth.Gravity

uref, href = u.subfunctions
H = function_mean(href)


# shallow water equation forms
def form_mass(u, h, v, q):
    return swe.nonlinear.form_mass(mesh, u, h, v, q)


def form_function(u, h, v, q, t=None):
    return swe.nonlinear.form_function(mesh,
                                       gravity,
                                       topography,
                                       coriolis,
                                       u, h, v, q, t)


def aux_form_function(u, h, v, q, t):
    # Ku = swe.nonlinear.form_function_velocity(
    #     mesh, gravity, topography, coriolis, u, h, v, t, perp=fd.cross)

    Ku = swe.linear.form_function_u(
        mesh, gravity, coriolis, u, h, v, t)

    Kh = swe.linear.form_function_h(
        mesh, H, u, h, q, t)

    return Ku + Kh


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

aux_sparams = {
    "mat_type": "matfree",
    "pc_type": "python",
    "pc_python_type": "asQ.AuxiliaryBlockPC",
    "aux": lu_params
}

rtol = 1e-5
sparams = {
    'ksp': {
        # 'monitor': None,
        'converged_rate': None,
        'rtol': rtol,
        # 'view': None
    },
    'ksp_type': 'fgmres',
}
sparams.update(aux_sparams)
# sparams.update(mg_sparams)

appctx = {
    'cpx': cpx,
    'u0': wref,
    't0': None,
    'd1': d1c,
    'd2': d2c,
    'bcs': [],
    'form_mass': form_mass,
    'form_function': form_function,
    'aux_form_function': aux_form_function,
}

wout = fd.Function(W)

problem = fd.LinearVariationalProblem(A, L, wout)
solver = fd.LinearVariationalSolver(problem, appctx=appctx,
                                    solver_parameters=sparams)

np.random.seed(args.seed)
for dat in L.dat:
    dat.data[:] = np.random.rand(*(dat.data.shape))

Print("")
for i in range(args.nt//2+1):
    wout.assign(0)
    freq = freqs[i]
    d1 = D1[i]
    d2 = D2[i]
    dhat = (d1/d2) / (1/(theta*dt))

    d1r.assign(d1.real)
    d1i.assign(d1.imag)

    d2r.assign(d2.real)
    d2i.assign(d2.imag)

    # Print(f"=== Eigenvalue {i} ===")
    # Print(f"freq = {np.round(freq, 5)}")
    # Print(f"d1 = {np.round(d1, 5)}")
    # Print(f"d2 = {np.round(d2, 5)}")
    # Print(f"(d1/d2)/(theta*dt) = {dhat}")
    # Print(f"abs((d1/d2)/(theta*dt)) = {abs(dhat)}")
    # Print("")
    solver.solve()
    # Print("")
