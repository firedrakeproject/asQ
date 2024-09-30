import firedrake as fd
from firedrake.petsc import PETSc

from utils.timing import SolverTimer
from utils.planets import earth
from utils import units

from argparse import ArgumentParser
from argparse_formatter import DefaultsAndRawTextFormatter


# get command arguments
parser = ArgumentParser(
    description='Gravity wave testcase for ParaDiag solver using fully implicit linear SWE solver.',
    epilog="""\
Optional PETSc command line arguments:

   -circulant_alpha :float: The circulant parameter to use in the preconditioner. Default 1e-4.
   -ksp_rtol :float: The relative residual drop required for convergence. Default 1e-11.
                     See https://petsc.org/release/manualpages/KSP/KSPSetTolerances/
   -ksp_type :str: The Krylov method to use for the all-at-once iterations. Default 'fgmres'.
                   Alternatives include richardson or gmres.
                   See https://petsc.org/release/manualpages/KSP/KSPSetType/
""",
    formatter_class=DefaultsAndRawTextFormatter
)

parser.add_argument('--ref_level', type=int, default=3, help='Refinement level of icosahedral grid. Total number of cells is 20*4^ref_level.')
parser.add_argument('--dt', type=float, default=0.25, help='Timestep in hours.')
parser.add_argument('--theta', type=float, default=0.5, help='Parameter for implicit theta method.')
parser.add_argument('--nwindows', type=int, default=1, help='Number of time-windows to solve.')
parser.add_argument('--nslices', type=int, default=1, help='Number of time-slices in the all-at-once system. Must divide the number of MPI ranks exactly.')
parser.add_argument('--slice_length', type=int, default=4, help='Number of timesteps per time-slice. Total number of timesteps in the all-at-once system is nslices*slice_length.')
parser.add_argument('--vtkfile', type=str, default='vtk/gravity_waves_paradiag', help='Name of output vtk files for the last timestep of each window.')
parser.add_argument('--metrics_dir', type=str, default='metrics/linear_shallow_water', help='Directory to save paradiag output metrics to.')
parser.add_argument('--show_args', action='store_true', help='Print all the arguments when the script starts.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Setting up --- === ###')
PETSc.Sys.Print('')

# The time partition describes how many timesteps
# are included on each time-slice of the ensemble

time_partition = [args.slice_length for _ in range(args.nslices)]
window_length = sum(time_partition)
nsteps = args.nwindows*window_length

dt = args.dt*units.hour

# Parameters for the implicit diagonal solve in step-(b).
# We use hybridisation to reduce the block to a smaller
# trace finite element space defined only on the element
# facets. The 'hybridscpc_condensed_field' options define
# how this trace system is solved.

from utils.hybridisation import HybridisedSCPC  # noqa: F401

block_parameters = {
    "mat_type": "matfree",
    "ksp_type": 'preonly',
    "pc_type": "python",
    "pc_python_type": f"{__name__}.HybridisedSCPC",
    "hybridscpc_condensed_field": {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
    }
}

paradiag_parameters = {
    'snes_type': 'ksponly',      # only solve 1 "Newton iteration" per window (i.e. a linear problem)
    'ksp_type': 'fgmres',        # fgmres requires one less preconditioner application than gmres or richardson
    'ksp': {
        'monitor': None,         # show the residual at every iteration
        'converged_rate': None,  # show the contraction rate once the linear solve has converged
        'rtol': 1e-11,           # relative residual tolerance
    },
    'pc_type': 'python',
    'pc_python_type': 'asQ.CirculantPC',       # the alpha-circulant preconditioner
    'circulant_alpha': 1e-4,                   # set other values from command line using: -circulant_alpha <value>
    'circulant_block': block_parameters,  # options dictionary for the inner solve
    'circulant_state': 'linear',               # system is linear so don't update the preconditioner reference state
    'aaos_jacobian_state': 'linear',           # system is linear so don't update the jacobian reference state
}


# In this script we use the ShallowWaterMiniApp class to set
# everything up for us. The miniapp creates the ensemble, so
# at this stage we don't have a spatial communicator to define
# a mesh over. Instead we provide a function to the miniapp
# to create the mesh given a communicator.
def create_mesh(comm):
    distribution_parameters = {
        "partition": True,
        "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)
    }
    mesh = fd.IcosahedralSphereMesh(
        radius=earth.radius, refinement_level=args.ref_level,
        distribution_parameters=distribution_parameters,
        comm=comm)
    x = fd.SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)
    return mesh


# We have a variety of utilities for the shallow water equations
# in the utils module. The two used here are:
#
# - The ShallowWaterMiniApp builds the compatible finite element
#   forms for the shallow water equations and sets up the Paradiag
#   object for us. It will also record some diagnostics, for
#   example the advective Courant number (not used for this linear
#   example), and writing the solution at the last timestep of each
#   window to a VTK file.
#
# - The gravity_bumps submodule has expressions for the initial
#   conditions for the test case of:
#   Schreiber & Loft, 2019, "A Parallel Time-Integrator for Solving
#   the Linearized Shallow Water Equations on the Rotating Sphere"

from utils.shallow_water import (ShallowWaterMiniApp,
                                 gravity_bumps as gwcase)

miniapp = ShallowWaterMiniApp(
    gravity=earth.Gravity,
    topography_expression=gwcase.topography_expression,
    velocity_expression=gwcase.velocity_expression,
    depth_expression=gwcase.depth_expression,
    reference_depth=gwcase.H,
    create_mesh=create_mesh,
    linear=True,
    dt=dt, theta=args.theta,
    time_partition=time_partition,
    paradiag_sparameters=paradiag_parameters,
    record_diagnostics={'cfl': False, 'file': True},
    file_name=args.vtkfile)


timer = SolverTimer()


# This function will be called before paradiag solves each time-window. We can use
# this to make the output a bit easier to read, and to time the window calculation
def window_preproc(swe_app, pdg, wndw):
    PETSc.Sys.Print(f'### === --- Calculating time-window {wndw} --- === ###')
    PETSc.Sys.Print('')
    # for now we are interested in timing only the solve, this
    # makes sure we don't time any synchronisation after prints.
    with PETSc.Log.Event("window_preproc.Coll_Barrier"):
        pdg.ensemble.ensemble_comm.Barrier()
    timer.start_timing()


# This function will be called after paradiag solves each time-window. We can use
# this to finish the window calculation timing and print the result.
# The time at the last timestep of the window is also printed.
def window_postproc(swe_app, pdg, wndw):
    timer.stop_timing()
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'Window solution time: {timer.times[-1]}')
    PETSc.Sys.Print('')

    if pdg.layout.is_local(miniapp.save_step):
        nt = (pdg.total_windows - 1)*pdg.ntimesteps + (miniapp.save_step + 1)
        time = nt*pdg.aaoform.dt
        comm = miniapp.ensemble.comm
        PETSc.Sys.Print(f'Hours = {float(time/units.hour)}', comm=comm)
        PETSc.Sys.Print(f'Days = {float(time/earth.day)}', comm=comm)
        PETSc.Sys.Print('', comm=comm)


paradiag = miniapp.paradiag
ics = paradiag.solver.aaofunc.initial_condition.copy(deepcopy=True)

# Setup all solver objects. The firedrake DM and options management
# makes it difficult to setup some preconditioners without actually
# calling `solve`, so we just run once to set everything up.
PETSc.Sys.Print('### === --- Setting up solver and prefactoring --- === ###')
PETSc.Sys.Print('')

with PETSc.Log.Event("warmup_solve"):
    miniapp.solve(nwindows=1)

# reset solution and iteration counts for timed solved
timer.times.clear()
paradiag.reset_diagnostics()
aaofunc = paradiag.solver.aaofunc
aaofunc.bcast_field(-1, aaofunc.initial_condition)
aaofunc.assign(aaofunc.initial_condition)

PETSc.Sys.Print('### === --- Calculating parallel solution --- === ###')
PETSc.Sys.Print('')

with PETSc.Log.Event("timed_solves"):
    miniapp.solve(nwindows=args.nwindows,
                  preproc=window_preproc,
                  postproc=window_postproc)

PETSc.Sys.Print('### === --- Iteration and timing results --- === ###')
PETSc.Sys.Print('')

from asQ import write_paradiag_metrics
write_paradiag_metrics(paradiag, directory=args.metrics_dir)

nw = paradiag.total_windows
nt = paradiag.total_timesteps
PETSc.Sys.Print(f'windows: {nw}')
PETSc.Sys.Print(f'timesteps: {nt}')
PETSc.Sys.Print('')

# Show the parallel partition sizes.
ensemble = paradiag.ensemble
mesh = miniapp.mesh
W = miniapp.W

PETSc.Sys.Print(f'Total DoFs per window: {W.dim()*window_length}')
PETSc.Sys.Print(f'DoFs per timestep: {W.dim()}')
PETSc.Sys.Print(f'Total number of MPI ranks: {ensemble.global_comm.size}')
PETSc.Sys.Print(f'Number of MPI ranks per timestep: {mesh.comm.size}')
PETSc.Sys.Print(f'DoFs/rank: {W.dim()/mesh.comm.size}')
PETSc.Sys.Print(f'Complex block DoFs/rank: {2*W.dim()/mesh.comm.size}')
PETSc.Sys.Print('')

# paradiag collects a few iteration counts for us
lits = paradiag.linear_iterations
nlits = paradiag.nonlinear_iterations
blits = paradiag.block_iterations.data()

PETSc.Sys.Print(f'Nonlinear iterations: {nlits} | Iterations per window: {nlits/nw}')
PETSc.Sys.Print(f'Linear iterations: {lits} | Iterations per window: {lits/nw}')
PETSc.Sys.Print(f'Total block linear iterations: {blits}')
PETSc.Sys.Print(f'Iterations per block solve: {blits/lits}')
PETSc.Sys.Print('')

PETSc.Sys.Print(timer.string(timesteps_per_solve=window_length,
                             total_iterations=lits, ndigits=5))
PETSc.Sys.Print('')
