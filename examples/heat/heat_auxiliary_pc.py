import firedrake as fd
from firedrake.petsc import PETSc
import asQ
from math import pi
from numpy import random as rand
rand.seed(6)

import argparse

Print = PETSc.Sys.Print

parser = argparse.ArgumentParser(
    description='Solve the heat equation all-at-once system by preconditioning with an auxiliary operator with a different diffusion coefficient.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--nx', type=int, default=32, help='Number of cells along each square side.')
parser.add_argument('--degree', type=int, default=1, help='Degree of the finite element spaces.')
parser.add_argument('--theta', type=float, default=1.0, help='Parameter for the implicit theta timestepping method.')
parser.add_argument('--nwindows', type=int, default=1, help='Number of time-windows.')
parser.add_argument('--nslices', type=int, default=-1, help='Number of time-slices per time-window. If <1 then nslices=COMM_WORLD.size.')
parser.add_argument('--slice_length', type=int, default=2, help='Number of timesteps per time-slice.')
parser.add_argument('--dt', type=float, default=0.1, help='Timestep size.')
parser.add_argument('--alpha', type=float, default=1e-4, help='Circulant coefficient.')
parser.add_argument('--nu', type=float, default=1, help='Diffusion coefficient.')
parser.add_argument('--pnu', type=float, default=1, help='Diffusion coefficient in the auxiliary preconditioning operator.')
parser.add_argument('--print_params', action='store_true', help='Print the parameters dictionary.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    Print(args)

Print = Print

nslices = args.nslices if (args.nslices > 0) else fd.COMM_WORLD.size
time_partition = [args.slice_length for _ in range(nslices)]
ensemble = asQ.create_ensemble(time_partition, comm=fd.COMM_WORLD)

nt = sum(time_partition)
dt = args.dt
nx = args.nx
dy = 1/nx


nu = args.nu
pnu = args.pnu
dnu = pnu - nu

cfl = nu*dt/dy**2
cfl_aux = pnu*dt/dy**2

T = nt*dt
Print(f"{nt = } | {T = :.2e} | {cfl = :.2e} | {dnu = :.2e}")


def phi(z, theta):
    return (1 + (1-theta)*z)/(1 - theta*z)


def gamma(pz, n):
    return min(n, (1 - abs(pz)**n)/(1 - abs(pz)))


def eta(z, zhat, theta, alpha, n):
    pz = phi(z, theta)
    pzh = phi(zhat, theta)
    dp = abs(pz - pzh)
    gm = gamma(pzh, n)
    return (dp*gm + alpha)/(1 - alpha)


zmin = -4*cfl
zmax = -cfl*(pi/nx)**2
zaux_min = -4*cfl_aux
zaux_max = -cfl_aux*(pi/nx)**2

phi_min = phi(zmin, args.theta)
phi_max = phi(zmax, args.theta)
phi_aux_min = phi(zaux_min, args.theta)
phi_aux_max = phi(zaux_max, args.theta)

Print(f"{phi_min     = :.4e} | {phi_max     = :.4e}")
Print(f"{phi_aux_min = :.4e} | {phi_aux_max = :.4e}")

dphi_min = abs(phi_min - phi_aux_min)
dphi_max = abs(phi_max - phi_aux_max)
gamma_min = gamma(phi_aux_min, nt)
gamma_max = gamma(phi_aux_max, nt)

Print(f"{dphi_min    = :.4e} | {dphi_max    = :.4e}")
Print(f"{gamma_min   = :.4e} | {gamma_max   = :.4e}")

eta_min = eta(zmin, zaux_min, args.theta, args.alpha, nt)
eta_max = eta(zmax, zaux_max, args.theta, args.alpha, nt)

Print(f"{eta_min     = :.4e} | {eta_max     = :.4e}")

mesh = fd.UnitIntervalMesh(nx, comm=ensemble.comm)
V = fd.FunctionSpace(mesh, "CG", 1)
bcs = [fd.DirichletBC(V, 0, 1)]


def form_mass(u, v):
    return u*v*fd.dx


def form_heat(nu):
    def formfunc(u, v, t):
        return fd.inner(fd.Constant(nu)*fd.grad(u), fd.grad(v))*fd.dx
    return formfunc


aaofunc = asQ.AllAtOnceFunction(
    ensemble, time_partition, V)

# the problem we're solving uses the reference diffusivity
aaoform = asQ.AllAtOnceForm(
    aaofunc, dt, args.theta,
    form_mass, form_heat(nu),
    bcs=bcs)

# the preconditioner is built with the shifted diffusivity
aaoform_pc = asQ.AllAtOnceForm(
    aaofunc, dt, args.theta,
    form_mass, form_heat(pnu),
    bcs=bcs)


class AuxiliaryHeatPC(asQ.AuxiliaryOperatorPC):
    def get_jacobian(self, pc):
        return asQ.AllAtOnceJacobian(aaoform_pc)


def circulant_params(alpha=args.alpha, block_params=None):
    return {
        'pc_type': 'python',
        'pc_python_type': 'asQ.CirculantPC',
        'circulant_alpha': alpha,
        'circulant_block': {
            'ksp_type': 'preonly',
            'pc_type': 'ilu',
        } if block_params is None else block_params,
        'circulant_block_0_ksp_view': ':logs/block_0_ksp_view.log',
        **{f'circulant_block_{j}_ksp_converged_rate': f':logs/block_{j}_ksp_rate.log'
           for j in range(nt)},
    }


def aaoaux_params(
        ksp_type='richardson', ksp_rtol=1e-12,
        ksp_converged_rate=':logs/aux_ksp_rate.log',
        alpha=args.alpha, aux_params=None, block_params=None):
    return {
        'pc_type': 'python',
        'pc_python_type': f'{__name__}.AuxiliaryHeatPC',
        'aaoaux_ksp_converged_rate': ksp_converged_rate,
        'aaoaux_ksp_type': ksp_type,
        'aaoaux_ksp_rtol': ksp_rtol,
        'aaoaux': circulant_params(alpha, block_params) if aux_params is None else aux_params,
    }


solver_parameters = {
    'mat_type': 'matfree',
    'ksp': {
        'view': ':logs/ksp_view.log',
        'monitor': ':logs/ksp_monitor.log',
        'converged_rate': None,
        'rtol': 1e-8,
        'initial_guess_nonzero': None,
    },
    'ksp_type': 'richardson',
    # **circulant_params(),
    **aaoaux_params(),
}
if args.print_params:
    from json import dumps
    Print(dumps(solver_parameters, indent=3))

solver = asQ.LinearSolver(
    aaoform, aaoform_pc=aaoform_pc,
    solver_parameters=solver_parameters)

sol = aaofunc.copy(copy_values=False)
rhs = aaofunc.riesz_representation()

with rhs.global_vec_wo() as v:
    v.array[:] = rand.random_sample(v.array.shape)
with sol.global_vec_wo() as v:
    v.array[:] = rand.random_sample(v.array.shape)
solver.solve(rhs, sol)
