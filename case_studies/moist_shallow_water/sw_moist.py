import firedrake as fd
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
PETSc.Sys.popErrorHandler()

from utils import units
from utils.planets import earth
from utils import shallow_water as swe
from utils.shallow_water.williamson1992 import case5 as w5
from utils.mg import ManifoldTransferManager  # noqa: F401
from utils.diagnostics import convective_cfl_calculator

import asQ

import argparse
parser = argparse.ArgumentParser(
    description='Moist Williamson 5 testcase',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--base_level', type=int, default=1, help='Base refinement level of icosahedral grid for MG solve.')
parser.add_argument('--ref_level', type=int, default=5, help='Refinement level of icosahedral grid.')
parser.add_argument('--dt', type=float, default=1, help='Timestep in hours.')
parser.add_argument('--theta', type=float, default=0.5, help='Implicit parameter.')
parser.add_argument('--coords_degree', type=int, default=1, help='Degree of polynomials for sphere mesh approximation.')
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space).')
parser.add_argument('--nwindows', type=int, default=1, help='Number of time-windows.')
parser.add_argument('--nslices', type=int, default=2, help='Number of time-slices per time-window.')
parser.add_argument('--slice_length', type=int, default=2, help='Number of timesteps per time-slice.')
parser.add_argument('--alpha', type=float, default=0.0001, help='Circulant coefficient.')
parser.add_argument('--kspmg', type=int, default=3, help='Max number of KSP iterations in the MG levels.')
parser.add_argument('--filename', type=str, default='w5moist')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Setting up --- === ###')
PETSc.Sys.Print('')

time_partition = tuple((args.slice_length for _ in range(args.nslices)))
window_length = sum(time_partition)
nsteps = args.nwindows*window_length

global_comm = fd.COMM_WORLD
ensemble = asQ.create_ensemble(time_partition, comm=global_comm)

# some domain, parameters and FS setup
R0 = earth.Radius
H = w5.H0
base_level = args.base_level
nrefs = args.ref_level - base_level
name = args.filename

mesh = swe.create_mg_globe_mesh(base_level=args.base_level,
                                ref_level=args.ref_level,
                                coords_degree=1,
                                comm=ensemble.comm)

R0 = fd.Constant(R0)
x, y, z = fd.SpatialCoordinate(mesh)
outward_normals = fd.CellNormal(mesh)

Vdry = swe.default_function_space(mesh, degree=args.degree)
V1, V2 = Vdry.subfunctions
V0 = fd.FunctionSpace(mesh, "CG", args.degree+2)
W = fd.MixedFunctionSpace((V1, V2, V2, V2, V2, V2))
# velocity, depth, temperature, vapour, cloud, rain

PETSc.Sys.Print(f"DoFs: {W.dim()}")
PETSc.Sys.Print(f"DoFs/core: {W.dim()/ensemble.comm.size}")

Omega = earth.Omega  # rotation rate
f = w5.coriolis_expression(x, y, z)  # Coriolis parameter
g = earth.Gravity  # Gravitational constant
b = fd.Function(V2, name="Topography")

# h = D + b

# finite element forms

n = fd.FacetNormal(mesh)
outward_normals = fd.CellNormal(mesh)

# mass matrices


def form_mass_velocity(u, v):
    return fd.inner(u, v)*fd.dx


def form_mass_depth(h, p):
    return h*p*fd.dx


def form_mass_moisture(q, phi):
    return q*phi*fd.dx


def form_mass(u, h, B, qv, qc, qr, du, dh, dB, dqv, dqc, dqr, t=None):
    return (form_mass_velocity(u, du)
            + form_mass_depth(h, dh)
            + form_mass_moisture(B, dB)
            + form_mass_moisture(qv, dqv)
            + form_mass_moisture(qc, dqc)
            + form_mass_moisture(qr, dqr))


# velocity / depth forms

def perp(u):
    return fd.cross(outward_normals, u)


def form_advection(u, v):
    def both(u):
        return 2*fd.avg(u)
    upwind = 0.5 * (fd.sign(fd.dot(u, n)) + 1)
    k = 0.5*fd.inner(u, u)
    return (fd.inner(v, f*perp(u))*fd.dx
            - fd.inner(perp(fd.grad(fd.inner(v, perp(u)))), u)*fd.dx
            - fd.div(v)*k*fd.dx
            + fd.inner(both(perp(n)*fd.inner(v, perp(u))),
                       both(upwind*u))*fd.dS)


def form_transport(u, q, p, conservative=False):
    uup = 0.5 * (fd.dot(u, n) + abs(fd.dot(u, n)))
    flux = fd.jump(p)*(uup('+')*q('+')
                       - uup('-')*q('-'))*fd.dS
    if conservative:
        volume = -fd.inner(fd.grad(p), u)*q*fd.dx
    else:
        volume = -fd.inner(fd.div(p*u), q)*fd.dx
    return volume + flux


def form_function_velocity_dry(g, b, f, u, h, v):
    A = form_advection(u, v)
    P = -fd.div(v)*g*(h + b)*fd.dx
    return A + P


def form_function_velocity_moist(g, b, f, u, h, B, qv, qc, qr, v):
    A = form_advection(u, v)
    # Thermal terms from Nell
    # b grad (D+B) + D/2 grad b
    # integrate by parts
    # -<div(bv), (D+B)> + << jump(vb, n), {D+B} >>
    # -<div(Dv), b/2> + <<jump(Dv, n), {b/2} >>
    P = (- fd.div(B*v)*(h + b)*fd.dx
         + fd.jump(B*v, n)*fd.avg(h + b)*fd.dS
         - fd.div(h*v)*(B/2)*fd.dx
         + fd.jump(h*v, n)*fd.avg(B/2)*fd.dS)
    return A + P


def form_function_depth_moist(u, h, B, qv, qc, qr, p):
    return form_transport(u, h, p, conservative=True)


# moisture forms


dx0 = fd.dx('everywhere', metadata={'quadrature_degree': 6})

q0 = fd.Constant(135)
gamma_r = fd.Constant(1.0e-3)
q_precip = fd.Constant(1.0e-4)

dT = fd.Constant(0.)
L = fd.Constant(10)


def del_qv(qv, qsat, gamma_v, dT):
    return fd.max_value(0, gamma_v*(qv - qsat))/dT


def del_qc(qc, qv, qsat, gamma_v, dT):
    return fd.min_value(qc, fd.max_value(0, gamma_v*(qsat - qv)))/dT


def del_qr(qc, q_precip, gamma_r, dT):
    return fd.max_value(0, gamma_r*(qc - q_precip))/dT


def gamma_v_expr(h, B):
    return 1/(1 + L*(20*q0/g/(h + b))*fd.exp(20*(1-B/g)))


def qsat_expr(h, B):
    return q0/g/(h + b)*fd.exp(20*(1-B/g))


def form_function_bouyancy(u, h, B, qv, qc, qr, dB):
    gamma_v = gamma_v_expr(h, B)
    qsat = qsat_expr(h, B)
    delqv = del_qv(qv, qsat, gamma_v, dT)
    delqc = del_qc(qc, qv, qsat, gamma_v, dT)
    return (form_transport(u, B, dB, conservative=False)
            + g*L*dB*(delqv - delqc)*dx0)


def form_function_vapour(u, h, B, qv, qc, qr, dqv):
    gamma_v = gamma_v_expr(h, B)
    qsat = qsat_expr(h, B)
    delqv = del_qv(qv, qsat, gamma_v, dT)
    delqc = del_qc(qc, qv, qsat, gamma_v, dT)
    return (form_transport(u, qv, dqv, conservative=False)
            - dqv*(delqc - delqv)*dx0)


def form_function_cloud(u, h, B, qv, qc, qr, dqc):
    gamma_v = gamma_v_expr(h, B)
    qsat = qsat_expr(h, B)
    delqv = del_qv(qv, qsat, gamma_v, dT)
    delqc = del_qc(qc, qv, qsat, gamma_v, dT)
    delqr = del_qr(qc, q_precip, gamma_r, dT)
    return (form_transport(u, qc, dqc, conservative=False)
            - dqc*(delqv - delqc - delqr)*dx0)


def form_function_rain(u, h, B, qv, qc, qr, dqr):
    delqr = del_qr(qc, q_precip, gamma_r, dT)
    return (form_transport(u, qr, dqr, conservative=False)
            - dqr*delqr*dx0)


def form_function(*args, t=None):
    ncpts = 6
    trials, tests = args[:ncpts], args[ncpts:]
    return (form_function_velocity_moist(g, b, f, *trials, tests[0])
            + form_function_depth_moist(*trials, tests[1])
            + form_function_bouyancy(*trials, tests[2])
            + form_function_vapour(*trials, tests[3])
            + form_function_cloud(*trials, tests[4])
            + form_function_rain(*trials, tests[5]))


# monolithic solver options

lu_params = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "pc_factor_mat_ordering_type": "rcm",
    "pc_factor_reuse_ordering": None,
    "pc_factor_reuse_fill": None,
}

atol = 1e5
sparameters = {
    "mat_type": "matfree",
    "ksp_type": "fgmres",
    "ksp": {
        # "monitor": None,
        # "converged_rate": None,
        "atol": 1e-100,
        "rtol": 1e-5,
        "max_it": 30,
    },
    "pc_type": "mg",
    "pc_mg_cycle_type": "v",
    "pc_mg_type": "full",
    "mg": {
        "transfer_manager": f"{__name__}.ManifoldTransferManager",
        "levels": {
            "ksp_type": "gmres",
            "ksp_max_it": args.kspmg,
            "ksp_convergence_test": "skip",
            "pc_type": "python",
            "pc_python_type": "firedrake.PatchPC",
            "patch": {
                "pc_patch": {
                    "save_operators": True,
                    "partition_of_unity": True,
                    "sub_mat_type": "seqdense",
                    "construct_dim": 0,
                    "construct_type": "star",
                    "local_type": "additive",
                    "precompute_element_tensors": True,
                    "symmetrise_sweep": False,
                },
                "sub": {
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                    "pc_factor_shift_type": "nonzero",
                },
            },
        },
        "coarse": {
            "pc_type": "python",
            "pc_python_type": "firedrake.AssembledPC",
            "assembled": lu_params
        },
    },
}

sparameters_diag = {
    'snes': {
        'linesearch_type': 'basic',
        'monitor': None,
        'converged_reason': None,
        'atol': atol,
        'rtol': 1e-10,
        'stol': 1e-12,
        'ksp_ew': None,
        'ksp_ew_version': 1,
        'ksp_ew_threshold': 1e-2,
    },
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp': {
        'monitor': None,
        'converged_rate': None,
        'rtol': 1e-2,
        'atol': atol,
    },
    'pc_type': 'python',
    'pc_python_type': 'asQ.CirculantPC',
    'diagfft_alpha': args.alpha,
}

for i in range(sum(time_partition)):
    sparameters_diag["diagfft_block_"+str(i)+"_"] = lu_params

dt = units.hour*args.dt
dT.assign(dt)
t = 0.

w0 = fd.Function(W)

u_expr = w5.velocity_expression(x, y, z)
eta_expr = w5.elevation_expression(x, y, z)

un = fd.Function(V1, name="Velocity").project(u_expr)
etan = fd.Function(V2, name="Elevation").project(eta_expr)

# Topography.
bexpr = w5.topography_expression(x, y, z)

b.interpolate(bexpr)

u0, h0, B0, qv0, qc0, qr0 = w0.subfunctions
u0.assign(un)
h0.assign(etan + H - b)

# The below is from Nell Hartney
eps = fd.Constant(1.0/300)
EQ = 30*eps
SP = -40*eps
NP = -20*eps
mu1 = fd.Constant(0.05)
mu2 = fd.Constant(0.98)

# expression for initial buoyancy - note the bracket around 1-mu
lambda_x = fd.atan2(y/R0, x/R0)  # longitude
phi_x = fd.asin(z/R0)  # latitude

F = (2/(fd.pi**2))*(phi_x*(phi_x-fd.pi/2)*SP
                    - 2*(phi_x+fd.pi/2)*(phi_x-fd.pi/2)*(1-mu1)*EQ
                    + phi_x*(phi_x+fd.pi/2)*NP)
theta_expr = F + mu1*EQ*fd.cos(phi_x)*fd.sin(lambda_x)
buoyexpr = g * (1 - theta_expr)
B0.interpolate(buoyexpr)

# The below is from Nell Hartney
# expression for initial water vapour depends on initial saturation
initial_msat = q0/(g*h0 + g*bexpr) * fd.exp(20*theta_expr)
vexpr = mu2 * initial_msat
qv0.interpolate(vexpr)
# cloud and rain initially zero

pdg = asQ.Paradiag(ensemble=ensemble,
                   time_partition=time_partition,
                   form_function=form_function,
                   form_mass=form_mass,
                   ics=w0, dt=dt, theta=args.theta,
                   solver_parameters=sparameters_diag)

is_last_slice = pdg.layout.is_local(-1)

if is_last_slice:
    q = fd.TrialFunction(V0)
    p = fd.TestFunction(V0)

    qn = fd.Function(V0, name="Relative Vorticity")
    veqn = q*p*fd.dx + fd.inner(perp(fd.grad(p)), un)*fd.dx
    vprob = fd.LinearVariationalProblem(fd.lhs(veqn), fd.rhs(veqn), qn)
    qparams = {'ksp_type': 'cg'}
    qsolver = fd.LinearVariationalSolver(vprob,
                                         solver_parameters=qparams)

    uout, hout, Bout, qvout, qcout, qrout = pdg.aaofunc[-1].subfunctions
    file_sw = fd.File(f'output/{name}.pvd', comm=ensemble.comm)
    etan.assign(hout - H + b)
    un.assign(uout)
    qsolver.solve()
    Bn = fd.Function(V2, name="Buoyancy")
    qvn = fd.Function(V2, name="Water Vapour")
    qcn = fd.Function(V2, name="Cloud Vapour")
    qrn = fd.Function(V2, name="Rain")
    Bn.interpolate(Bout)
    qvn.interpolate(qvout)
    qcn.interpolate(qcout)
    qrn.interpolate(qrout)
    file_sw.write(un, etan, Bn, qvn, qcn, qrn)

    cfl_calc = convective_cfl_calculator(mesh)
    cfl_series = []

    def max_cfl(u, dt):
        with cfl_calc(u, dt).dat.vec_ro as v:
            return v.max()[1]


solver_time = []


def window_preproc(pdg, wndw, rhs):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'### === --- Calculating time-window {wndw} --- === ###')
    PETSc.Sys.Print('')
    stime = MPI.Wtime()
    solver_time.append(stime)


def window_postproc(pdg, wndw, rhs):
    etime = MPI.Wtime()
    stime = solver_time[-1]
    duration = etime - stime
    solver_time[-1] = duration
    PETSc.Sys.Print('', comm=global_comm)
    PETSc.Sys.Print(f'Window solution time: {duration}', comm=global_comm)
    PETSc.Sys.Print('', comm=global_comm)

    # postprocess this timeslice
    if is_last_slice:
        etan.assign(hout - H + b)
        un.assign(uout)
        Bn.assign(Bout)
        qvn.interpolate(qvout)
        qcn.interpolate(qcout)
        qrn.interpolate(qrout)
        file_sw.write(un, etan, Bn, qvn, qcn, qrn)

        cfl = max_cfl(uout, dt)
        cfl_series.append(cfl)
        PETSc.Sys.Print('', comm=ensemble.comm)
        PETSc.Sys.Print(f'Maximum CFL = {cfl}', comm=ensemble.comm)


# solve for each window
pdg.solve(nwindows=args.nwindows,
          preproc=window_preproc,
          postproc=window_postproc)

PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Iteration counts --- === ###')
PETSc.Sys.Print('')

nw = pdg.total_windows
nt = pdg.total_timesteps
PETSc.Sys.Print(f'windows: {nw}')
PETSc.Sys.Print(f'timesteps: {nt}')
PETSc.Sys.Print('')

lits = pdg.linear_iterations
nlits = pdg.nonlinear_iterations
blits = pdg.block_iterations.data()

PETSc.Sys.Print(f'linear iterations: {lits} | iterations per window: {lits/nw}')
PETSc.Sys.Print(f'nonlinear iterations: {nlits} | iterations per window: {nlits/nw}')
PETSc.Sys.Print(f'block linear iterations: {blits} | iterations per block solve: {blits/lits}')
PETSc.Sys.Print('')

ensemble.global_comm.Barrier()
if is_last_slice:
    PETSc.Sys.Print(f'Maximum CFL = {max(cfl_series)}', comm=ensemble.comm)
    PETSc.Sys.Print(f'Minimum CFL = {min(cfl_series)}', comm=ensemble.comm)
    PETSc.Sys.Print('', comm=ensemble.comm)
ensemble.global_comm.Barrier()

PETSc.Sys.Print(f'DoFs per timestep: {W.dim()}', comm=global_comm)
PETSc.Sys.Print(f'Number of MPI ranks per timestep: {mesh.comm.size} ', comm=global_comm)
PETSc.Sys.Print(f'DoFs/rank: {W.dim()/mesh.comm.size}', comm=global_comm)
PETSc.Sys.Print(f'Block DoFs/rank: {2*W.dim()/mesh.comm.size}', comm=global_comm)
PETSc.Sys.Print('')

if len(solver_time) > 1:
    solver_time[0] = solver_time[1]

PETSc.Sys.Print(f'Total solution time: {sum(solver_time)}', comm=global_comm)
PETSc.Sys.Print(f'Average window solution time: {sum(solver_time)/len(solver_time)}', comm=global_comm)
PETSc.Sys.Print(f'Average timestep solution time: {sum(solver_time)/(window_length*len(solver_time))}', comm=global_comm)
PETSc.Sys.Print('', comm=global_comm)
