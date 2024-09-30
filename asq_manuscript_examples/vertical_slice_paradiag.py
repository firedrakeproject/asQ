import firedrake as fd
from firedrake.petsc import PETSc
from firedrake.output import VTKFile

from utils.timing import SolverTimer
from utils import compressible_flow as euler
from utils.diagnostics import convective_cfl_calculator
from math import sqrt, pi
import asQ

from argparse import ArgumentParser
from argparse_formatter import DefaultsAndRawTextFormatter

Print = PETSc.Sys.Print


def initial_conditions(mesh, W, Vv, gas, H, perturbation=True, hydrostatic=False):
    if hydrostatic:
        return NotImplementedError

    x, z = fd.SpatialCoordinate(mesh)
    V2 = W.subfunctions[1]
    up = fd.as_vector([fd.Constant(0.0), fd.Constant(1.0)])  # up direction

    Un = fd.Function(W)
    un, rhon, thetan = Un.subfunctions

    Tsurf = fd.Constant(300.)
    thetab = Tsurf*fd.exp(gas.N**2*z/gas.g)

    thetan.interpolate(thetab)

    Pi = fd.Function(V2)

    # background state is in hydrostatic balance
    euler.hydrostatic_rho(Vv, V2, mesh, thetan, rhon,
                          pi_boundary=fd.Constant(1.0),
                          gas=gas, Up=up, top=True, Pi=Pi)

    un.project(fd.as_vector([20.0, 0.0]))

    Uback = Un.copy(deepcopy=True)

    # temperature perturbation to initiate gravity wave
    if perturbation:
        a = fd.Constant(5e3)
        dtheta = fd.Constant(1e-2)

        theta_pert = dtheta*fd.sin(pi*z/H)/(1 + x**2/a**2)
        thetan.interpolate(thetan + theta_pert)

    return Un, Uback


parser = ArgumentParser(
    description='Nonhydrostatic gravity wave testcase for ParaDiag solver using fully implicit vertical slice.',
    epilog="""\
Optional PETSc command line arguments:

   -circulant_alpha :float: The circulant parameter to use in the preconditioner. Default 1e-5.
   -circulant_block_rtol :float: The relative tolerance to solve each block to. Default 1e-5.
""",
    formatter_class=DefaultsAndRawTextFormatter
)
parser.add_argument('--nlayers', type=int, default=10, help='Number of layers in the vertical direction.')
parser.add_argument('--ncolumns', type=int, default=300, help='Number of columns in the horizontal direction.')
parser.add_argument('--dt', type=float, default=12, help='Timestep in seconds.')
parser.add_argument('--theta', type=float, default=0.5, help='Parameter for implicit theta method.')
parser.add_argument('--nwindows', type=int, default=1, help='Number of time-windows to solve.')
parser.add_argument('--nslices', type=int, default=1, help='Number of time-slices in the all-at-once system. Must divide the number of MPI ranks exactly.')
parser.add_argument('--slice_length', type=int, default=4, help='Number of timesteps per time-slice. Total number of timesteps in the all-at-once system is nslices*slice_length.')
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space). Default 1.')
parser.add_argument('--vtkfile', type=str, default='vtk/gravity_wave_paradiag', help='Name of output vtk files')
parser.add_argument('--metrics_dir', type=str, default='metrics/vertical_slice', help='Directory to save paradiag output metrics to.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    Print(args)

Print('')
Print('### === --- Setting up --- === ###')
Print('')

# This script broadly follows the same structure
# as the advection_paradiag.py script, with the main
# differences being:
#
# - more complicated finite element spaces and forms are required,
#   which are provided by the utils.compressible_flow module.
# - more complicated initial conditions (the function at the top of
#   the file), from the test case of
#   Skamarock & Klemp, 1994, "Efficiency and accuracy of the
#   Klemp-Wilhelmson time-splitting technique".
# - options parameters specifying more involved block preconditioning.

# set up the ensemble communicator for space-time parallelism
time_partition = tuple((args.slice_length for _ in range(args.nslices)))
window_length = sum(time_partition)

global_comm = fd.COMM_WORLD
ensemble = asQ.create_ensemble(time_partition, comm=global_comm)

dt = args.dt

# set up the mesh
L = 300e3
H = 10e3

distribution_parameters = {
    "partition": True,
    "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)
}

base_mesh = fd.PeriodicIntervalMesh(
    args.ncolumns, float(L),
    distribution_parameters=distribution_parameters,
    comm=ensemble.comm)
base_mesh.coordinates.dat.data[:] -= float(L)/2

mesh = fd.ExtrudedMesh(base_mesh,
                       layers=args.nlayers,
                       layer_height=float(H/args.nlayers))
n = fd.FacetNormal(mesh)

# compatible function spaces

W, Vv = euler.function_space(mesh, horizontal_degree=args.degree,
                             vertical_degree=args.degree,
                             vertical_velocity_space=True)
V1, V2, Vt = W.subfunctions  # velocity, density, temperature

Print(f'DoFs per timestep: {W.dim()}', comm=global_comm)
Print(f'Number of MPI ranks per timestep: {mesh.comm.size}', comm=global_comm)
Print(f'DoFs/rank: {W.dim()/mesh.comm.size}', comm=global_comm)
Print(f'Block DoFs/rank: {2*W.dim()/mesh.comm.size}', comm=global_comm)
Print('')

Print("Calculating initial condiions")
Print('')

# ideal gas properties
gas = euler.StandardAtmosphere(N=0.01)

# initial conditions and background state
Un, Uback = initial_conditions(mesh, W, Vv, gas, H)
uback, rho_back, theta_back = Uback.subfunctions

U0 = Uback.copy(deepcopy=True)  # background state at rest
U0.subfunctions[0].assign(0)

# finite element forms

form_mass = euler.get_form_mass()

up = fd.as_vector([fd.Constant(0.0), fd.Constant(1.0)])  # up direction

form_function = euler.get_form_function(
    n, up, c_pen=fd.Constant(2.0**(-7./2)), gas=gas, mu=None)

# tangential flow boundary conditions at ground and lid
zv = fd.as_vector([fd.Constant(0.), fd.Constant(0.)])
bcs = [fd.DirichletBC(W.sub(0), zv, "bottom"),
       fd.DirichletBC(W.sub(0), zv, "top")]
for bc in bcs:
    bc.apply(Un)

# Parameters for the diag
patch_parameters = {
    "pc_type": "python",
    "pc_python_type": "firedrake.AssembledPC",
    "assembled": {
        "pc_type": "python",
        "pc_python_type": "firedrake.ASMStarPC",
        "pc_star": {
            "construct_dim": 0,
            "sub_sub_pc_type": "lu",
            "sub_sub_pc_factor_mat_solver_type": 'mumps',
        },
    },
}

aux_parameters = {
    'pc_type': 'python',
    'pc_python_type': 'asQ.AuxiliaryComplexBlockPC',
    'aux': {
        'frozen': None,  # never update the factorisation
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
    },
}

block_parameters = {
    'mat_type': 'matfree',
    "ksp_type": "fgmres",
    'ksp': {
        'rtol': 1e-5,
        'max_it': 200,
        'converged_maxits': None,
        'gmres_restart': 200,
    },
    'pc_type': 'composite',
    'pc_composite_type': 'multiplicative',
    'pc_composite_pcs': 'ksp,ksp',
    'sub_0_ksp': aux_parameters,
    'sub_0_ksp_ksp': {
        'type': 'gmres',
        'max_it': 2,
        'convergence_test': 'skip',
        'converged_maxits': None,
    },
    'sub_1_ksp': patch_parameters,
    'sub_1_ksp_ksp': {
        'type': 'gmres',
        'max_it': 2,
        'convergence_test': 'skip',
        'converged_maxits': None,
    },
}

# atol is the same for Newton and (right-preconditioned) Krylov
atol = 1e-4
patol = sqrt(window_length)*atol
solver_parameters = {
    "snes": {
        'linesearch_type': 'basic',
        "monitor": None,
        "converged_reason": None,
        'stol': 1e-100,
        "atol": patol,
        "ksp_ew": None,
        "ksp_ew_version": 1,
        "ksp_ew_rtol0": 1e-1,
        "ksp_ew_threshold": 1e-10,
        "lag_preconditioner": -2,  # preconditioner is constant-in-time
        "lag_preconditioner_persists": None,
    },
    "mat_type": "matfree",
    "ksp_type": "fgmres",
    "ksp": {
        "monitor": None,
        "converged_rate": None,
        "atol": patol,
        'max_it': 10,
        "min_it": 1,
        'converged_maxits': None,
    },
    "pc_type": "python",
    "pc_python_type": "asQ.CirculantPC",
    "circulant_state": 'reference',  # use the Jacobian reference state
    "circulant_alpha": 1e-5,
    "circulant_block": block_parameters,
}


# the reference state for the linearisation around a state of rest
appctx = {'block_appctx': {'aux_uref': U0}}

paradiag = asQ.Paradiag(ensemble=ensemble,
                        time_partition=time_partition,
                        form_function=form_function,
                        form_mass=form_mass,
                        ics=Un, dt=dt, theta=args.theta,
                        reference_state=Uback,
                        bcs=bcs, appctx=appctx,
                        solver_parameters=solver_parameters)

aaofunc = paradiag.aaofunc
# are we on the comm with timestep -1? If so, we are in charge of output.
is_last_slice = paradiag.layout.is_local(-1)

# only last slice does diagnostics/output
if is_last_slice:
    uout = fd.Function(V1, name='velocity')
    thetaout = fd.Function(Vt, name='temperature')
    rhoout = fd.Function(V2, name='density')

    ofile = VTKFile(f'{args.vtkfile}.pvd',
                    comm=ensemble.comm)

    def assign_out_functions():
        uout.assign(aaofunc[-1].subfunctions[0])
        rhoout.assign(aaofunc[-1].subfunctions[1])
        thetaout.assign(aaofunc[-1].subfunctions[2])

        # output density and temperature variations, not absolute values
        rhoout.assign(rhoout - rho_back)
        thetaout.assign(thetaout - theta_back)

    def write_to_file():
        ofile.write(uout, rhoout, thetaout)

    cfl_calc = convective_cfl_calculator(mesh)
    cfl_series = []

    def max_cfl(u, dt):
        with cfl_calc(u, dt).dat.vec_ro as v:
            return v.max()[1]


timer = SolverTimer()

fround = lambda x: round(float(x), 2)


def window_preproc(pdg, wndw, rhs):
    Print('')
    Print(f'### === --- Calculating time-window {wndw} --- === ###')
    Print('')
    with PETSc.Log.Event("window_preproc.Coll_Barrier"):
        pdg.ensemble.ensemble_comm.Barrier()
    timer.start_timing()


def window_postproc(pdg, wndw, rhs):
    timer.stop_timing()
    Print('', comm=global_comm)
    Print(f'Window solution time: {timer.times[-1]}', comm=global_comm)
    Print('', comm=global_comm)

    # postprocess this timeslice
    if is_last_slice:
        assign_out_functions()
        write_to_file()

        cfl = max_cfl(uout, dt)
        cfl_series.append(cfl)
        Print(f'Maximum CFL = {cfl}', comm=ensemble.comm)

        nt = pdg.total_windows*pdg.ntimesteps - 1
        time = float(nt*pdg.aaoform.dt)
        Print(f'Time = {fround(time)}', comm=ensemble.comm)


# Setup all solver objects. The firedrake DM and options management
# makes it difficult to setup some preconditioners without actually
# calling `solve`, so we just run once to set everything up.
Print('### === --- Setting up solver and prefactoring --- === ###')

with PETSc.Log.Event("warmup_solve"):
    paradiag.solve(nwindows=1,
                   preproc=window_preproc,
                   postproc=window_postproc)
Print('')

# reset
timer.times.clear()
paradiag.solver.aaofunc.assign(Un)
paradiag.reset_diagnostics()

Print('### === --- Calculating parallel solution --- === ###')

# solve for each window
with PETSc.Log.Event("timed_solves"):
    paradiag.solve(nwindows=args.nwindows,
                   preproc=window_preproc,
                   postproc=window_postproc)

Print('### === --- Iteration counts --- === ###')
Print('')

asQ.write_paradiag_metrics(paradiag, directory=args.metrics_dir)

nw = paradiag.total_windows
nt = paradiag.total_timesteps
Print(f'windows: {nw}')
Print(f'timesteps: {nt}')
Print('')

# Show the parallel partition sizes.
Print(f'Total DoFs per window: {W.dim()*window_length}')
Print(f'DoFs per timestep: {W.dim()}')
Print(f'Total number of MPI ranks: {ensemble.global_comm.size}')
Print(f'Number of MPI ranks per timestep: {mesh.comm.size}')
Print(f'DoFs/rank: {W.dim()/mesh.comm.size}')
Print(f'Complex block DoFs/rank: {2*W.dim()/mesh.comm.size}')
Print('')

lits = paradiag.linear_iterations
nlits = paradiag.nonlinear_iterations
blits = paradiag.block_iterations.data()

Print(f'Nonlinear iterations: {nlits} | Iterations per window: {nlits/nw}')
Print(f'Linear iterations: {lits} | Iterations per window: {lits/nw}')
Print(f'Total block linear iterations: {blits}')
Print(f'Iterations per block solve: {blits/lits}')
Print(f'Minimum block iterations per solve: {min(blits)/lits}')
Print(f'Maximum block iterations per solve: {max(blits)/lits}')
Print('')

ensemble.global_comm.Barrier()
if is_last_slice:
    Print(f'Maximum CFL = {max(cfl_series)}', comm=ensemble.comm)
    Print(f'Minimum CFL = {min(cfl_series)}', comm=ensemble.comm)
    Print('', comm=ensemble.comm)
ensemble.global_comm.Barrier()

Print(timer.string(timesteps_per_solve=window_length,
                   total_iterations=lits, ndigits=5))
Print('')
