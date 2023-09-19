
import firedrake as fd
from firedrake.petsc import PETSc

import asQ

from utils import units
from utils.diagnostics import convective_cfl_calculator
from utils.vertical_slice import hydrostatic_rho, \
    get_form_mass, get_form_function, maximum

from utils.serial import ComparisonMiniapp

PETSc.Sys.popErrorHandler()

# get command arguments
import argparse
parser = argparse.ArgumentParser(
    description='Compare the serial and parallel solutions to the vertical slice testcase.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--nlayers', type=int, default=35, help='Number of horizontal layers.')
parser.add_argument('--ncolumns', type=int, default=90, help='Number of vertical columns.')
parser.add_argument('--nwindows', type=int, default=1, help='Number of time-windows.')
parser.add_argument('--nslices', type=int, default=2, help='Number of time-slices per time-window.')
parser.add_argument('--slice_length', type=int, default=2, help='Number of timesteps per time-slice.')
parser.add_argument('--alpha', type=float, default=0.0001, help='Circulant coefficient.')
parser.add_argument('--dt', type=float, default=0.5, help='Timestep in hours.')
parser.add_argument('--horizontal_degree', type=int, default=1, help='Degree of the pressure function space in the horizontal direction.')
parser.add_argument('--vertical_degree', type=int, default=1, help='Degree of the pressure function space in the vertical direction.')
parser.add_argument('--theta', type=float, default=0.5, help='Parameter for implicit theta method. 0.5 for trapezium rule, 1 for backwards Euler.')
parser.add_argument('--filename', type=str, default='slice', help='Name of output vtk files')
parser.add_argument('--print_norms', type=bool, default=False, help='Print the norm of each timestep')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Setting up --- === ###')
PETSc.Sys.Print('')

# time steps

time_partition = [args.slice_length for _ in range(args.nslices)]

dt = args.dt*units.hour

ensemble = asQ.create_ensemble(time_partition)

# extruded slice mesh
L = 144e3
H = 35e3  # Height position of the model top

distribution_parameters = {
    "partition": True,
    "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)
}

# surface mesh of ground
base_mesh = fd.PeriodicIntervalMesh(args.ncolumns, L,
                                    distribution_parameters=distribution_parameters,
                                    comm=ensemble.comm)

# volume mesh of the slice
mesh = fd.ExtrudedMesh(base_mesh,
                       layers=args.nlayers,
                       layer_height=H/args.nlayers)
n = fd.FacetNormal(mesh)
x, z = fd.SpatialCoordinate(mesh)

# problem parameters
g = fd.Constant(9.810616)
N = fd.Constant(0.01)  # Brunt-Vaisala frequency (1/s)
cp = fd.Constant(1004.5)  # SHC of dry air at const. pressure (J/kg/K)
R_d = fd.Constant(287.)  # Gas constant for dry air (J/kg/K)
kappa = fd.Constant(2.0/7.0)  # R_d/c_p
p_0 = fd.Constant(1000.0*100.0)  # reference pressure (Pa, not hPa)
cv = fd.Constant(717.)  # SHC of dry air at const. volume (J/kg/K)
T_0 = fd.Constant(273.15)  # ref. temperature
cp = fd.Constant(1004.5)  # SHC of dry air at const. pressure (J/kg/K)

dt = 1.
dT = fd.Constant(dt)

# making a mountain out of a molehill
a = 10000.
xc = L/2.
hm = 1.
zs = hm*a**2/((x-xc)**2 + a**2)

smooth_z = True
name = "mountain_nh"
if smooth_z:
    name += '_smootherz'
    zh = 5000.
    xexpr = fd.as_vector([x, fd.conditional(z < zh, z + fd.cos(0.5*pi*z/zh)**6*zs, z)])
else:
    xexpr = fd.as_vector([x, z + ((H-z)/H)*zs])
mesh.coordinates.interpolate(xexpr)

# function spaces
S1 = fd.FiniteElement("CG", fd.interval, args.horizontal_degree+1)
S2 = fd.FiniteElement("DG", fd.interval, args.horizontal_degree)

# vertical base spaces
T0 = fd.FiniteElement("CG", fd.interval, args.vertical_degree+1)
T1 = fd.FiniteElement("DG", fd.interval, args.vertical_degree)

# build spaces V2, V3, Vt
V2h_elt = fd.HDiv(fd.TensorProductElement(S1, T1))
V2t_elt = fd.TensorProductElement(S2, T0)
V3_elt = fd.TensorProductElement(S2, T1)
V2v_elt = fd.HDiv(V2t_elt)
V2_elt = V2h_elt + V2v_elt

V1 = fd.FunctionSpace(mesh, V2_elt, name="Velocity")
V2 = fd.FunctionSpace(mesh, V3_elt, name="Pressure")
Vt = fd.FunctionSpace(mesh, V2t_elt, name="Temperature")
Vv = fd.FunctionSpace(mesh, V2v_elt, name="Vv")

W = V1 * V2 * Vt  # velocity, density, temperature

PETSc.Sys.Print(f"DoFs: {W.dim()}")
PETSc.Sys.Print(f"DoFs/core: {W.dim()/ensemble.comm.size}")

# initial conditions
Un = fd.Function(W)

# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
Tsurf = fd.Constant(300.)
thetab = Tsurf*fd.exp(N**2*z/g)

Up = fd.as_vector([fd.Constant(0.0), fd.Constant(1.0)])  # up direction

un = Un.subfunctions[0]
rhon = Un.subfunctions[1]
thetan = Un.subfunctions[2]
un.project(fd.as_vector([10.0, 0.0]))
thetan.interpolate(thetab)
theta_back = fd.Function(Vt).assign(thetan)
rhon.assign(1.0e-5)

# hydrostatic background state
PETSc.Sys.Print("Calculating hydrostatic state")

Pi = fd.Function(V2)

hydrostatic_rho(Vv, V2, mesh, thetan, rhon, pi_boundary=fd.Constant(0.02),
                cp=cp, R_d=R_d, p_0=p_0, kappa=kappa, g=g, Up=Up,
                top=True, Pi=Pi)
p0 = maximum(Pi)

hydrostatic_rho(Vv, V2, mesh, thetan, rhon, pi_boundary=fd.Constant(0.05),
                cp=cp, R_d=R_d, p_0=p_0, kappa=kappa, g=g, Up=Up,
                top=True, Pi=Pi)
p1 = maximum(Pi)
alpha = 2.*(p1-p0)
beta = p1-alpha
pi_top = (1.-beta)/alpha

hydrostatic_rho(Vv, V2, mesh, thetan, rhon, pi_boundary=fd.Constant(pi_top),
                cp=cp, R_d=R_d, p_0=p_0, kappa=kappa, g=g, Up=Up,
                top=True)

rho_back = fd.Function(V2).assign(rhon)

zc = H-10000.
mubar = 0.15/dt
mu_top = fd.conditional(z <= zc, 0.0, mubar*fd.sin((pi/2.)*(z-zc)/(H-zc))**2)
mu = fd.Function(V2).interpolate(mu_top/dT)


# vertical slice forms
form_function = get_form_function(n, Up, c_pen=2.0**(-7./2),
                                  cp=cp, g=g, R_d=R_d,
                                  p_0=p_0, kappa=kappa, mu=mu)

form_mass = get_form_mass()

zv = fd.as_vector([fd.Constant(0.), fd.Constant(0.)])
bcs = [fd.DirichletBC(W.sub(0), zv, "bottom"),
       fd.DirichletBC(W.sub(0), zv, "top")]

for bc in bcs:
    bc.apply(Un)



# solver parameters for the implicit solve
serial_sparameters = {
    'snes': {
        'atol': 1e-0,
        'rtol': 1e-12,
        'stol': 1e-12,
    },
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp': {
        'atol': 1e-8,
        'rtol': 1e-8,
    },
    'pc_type': 'mg',
    'pc_mg_cycle_type': 'w',
    'pc_mg_type': 'multiplicative',
    'mg': {
        'levels': {
            'ksp_type': 'gmres',
            'ksp_max_it': 5,
            'pc_type': 'python',
            'pc_python_type': 'firedrake.PatchPC',
            'patch': {
                'pc_patch_save_operators': True,
                'pc_patch_partition_of_unity': True,
                'pc_patch_sub_mat_type': 'seqdense',
                'pc_patch_construct_codim': 0,
                'pc_patch_construct_type': 'vanka',
                'pc_patch_local_type': 'additive',
                'pc_patch_precompute_element_tensors': True,
                'pc_patch_symmetrise_sweep': False,
                'sub_ksp_type': 'preonly',
                'sub_pc_type': 'lu',
                'sub_pc_factor_shift_type': 'nonzero',
            },
        },
        'coarse': {
            'pc_type': 'python',
            'pc_python_type': 'firedrake.AssembledPC',
            'assembled_pc_type': 'lu',
            'assembled_pc_factor_mat_solver_type': 'mumps',
        },
    }
}

if ensemble.ensemble_comm.rank == 0:
    serial_sparameters['snes']['monitor'] = None
    serial_sparameters['snes']['converged_reason'] = None
    serial_sparameters['ksp']['monitor'] = None
    serial_sparameters['ksp']['converged_reason'] = None

# parameters for the implicit diagonal solve in step-(b)
block_sparameters = {
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp': {
        'atol': 1e-8,
        'rtol': 1e-8,
    },
    'pc_type': 'mg',
    'pc_mg_cycle_type': 'v',
    'pc_mg_type': 'multiplicative',
    'mg': {
        'levels': {
            'ksp_type': 'gmres',
            'ksp_max_it': 5,
            'pc_type': 'python',
            'pc_python_type': 'firedrake.PatchPC',
            'patch': {
                'pc_patch_save_operators': True,
                'pc_patch_partition_of_unity': True,
                'pc_patch_sub_mat_type': 'seqdense',
                'pc_patch_construct_codim': 0,
                'pc_patch_construct_type': 'vanka',
                'pc_patch_local_type': 'additive',
                'pc_patch_precompute_element_tensors': True,
                'pc_patch_symmetrise_sweep': False,
                'sub_ksp_type': 'preonly',
                'sub_pc_type': 'lu',
                'sub_pc_factor_shift_type': 'nonzero',
            },
        },
        'coarse': {
            'pc_type': 'python',
            'pc_python_type': 'firedrake.AssembledPC',
            'assembled_pc_type': 'lu',
            'assembled_pc_factor_mat_solver_type': 'mumps',
        },
    }
}

parallel_sparameters = {
    'snes': {
        'linesearch_type': 'basic',
        'monitor': None,
        'converged_reason': None,
        'atol': 1e-0,
        'rtol': 1e-12,
        'stol': 1e-12,
        'ksp_ew': None,
        'ksp_ew_version': 1,
        'ksp_ew_threshold': 1e-2,
    },
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp': {
        'monitor': None,
        'converged_reason': None,
        'atol': 1e-0,
    },
    'pc_type': 'python',
    'pc_python_type': 'asQ.DiagFFTPC',
    'diagfft': {
        'alpha': args.alpha
    }
}

parallel_sparameters['diagfft_block_'] = block_sparameters

appctx = {}
transfer_managers = []
for _ in range(time_partition[ensemble.ensemble_comm.rank]):
    tm = mg.manifold_transfer_manager(W)
    transfer_managers.append(tm)
appctx['diagfft_transfer_managers'] = transfer_managers

miniapp = ComparisonMiniapp(ensemble, time_partition,
                            form_mass=form_mass,
                            form_function=form_function,
                            w_initial=w_initial,
                            dt=dt, theta=args.theta,
                            serial_sparameters=serial_sparameters,
                            parallel_sparameters=parallel_sparameters,
                            appctx=appctx)

miniapp.serial_app.nlsolver.set_transfer_manager(
    mg.manifold_transfer_manager(W))

rank = ensemble.ensemble_comm.rank
norm0 = fd.norm(w_initial)


def preproc(serial_app, paradiag, wndw):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'### === --- Time window {wndw} --- === ###')
    PETSc.Sys.Print('')
    PETSc.Sys.Print('=== --- Parallel solve --- ===')
    PETSc.Sys.Print('')


def serial_postproc(app, it, t):
    if not args.print_norms:
        return
    if rank == 0:
        PETSc.Sys.Print(f'Rank {rank}: Serial timestep {it} norm {fd.norm(app.w0)/norm0}', comm=ensemble.comm)
    ensemble.global_comm.Barrier()
    return


def parallel_postproc(pdg, wndw, rhs):
    if args.print_norms:
        aaos = miniapp.paradiag.aaos
        for step in range(aaos.nlocal_timesteps):
            it = aaos.transform_index(step, from_range='slice', to_range='window')
            w = aaos.get_field(step)
            PETSc.Sys.Print(f'Rank {rank}: Parallel timestep {it} norm {fd.norm(w)/norm0}', comm=ensemble.comm)
    PETSc.Sys.Print('')
    PETSc.Sys.Print('=== --- Serial solve --- ===')
    PETSc.Sys.Print('')
    return


PETSc.Sys.Print('### === --- Timestepping loop --- === ###')

errors = miniapp.solve(nwindows=args.nwindows,
                       preproc=preproc,
                       serial_postproc=serial_postproc,
                       parallel_postproc=parallel_postproc)

PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Errors --- === ###')

for it, err in enumerate(errors):
    PETSc.Sys.Print(f'Timestep {it} error: {err/norm0}')
