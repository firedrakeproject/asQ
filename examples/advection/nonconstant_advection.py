import matplotlib.pyplot as plt
from math import pi, cos, sin

import firedrake as fd
from firedrake.petsc import PETSc
import asQ

import argparse

parser = argparse.ArgumentParser(
    description='ParaDiag timestepping for scalar advection of a Gaussian bump in a periodic square with DG in space and implicit-theta in time. Based on the Firedrake DG advection example https://www.firedrakeproject.org/demos/DG_advection.py.html',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--nx', type=int, default=16, help='Number of cells along each square side.')
parser.add_argument('--dt', type=float, default=0.02, help='Convective CFL number.')
parser.add_argument('--angle', type=float, default=pi/6, help='Angle of the convective velocity.')
parser.add_argument('--degree', type=int, default=1, help='Degree of the scalar and velocity spaces.')
parser.add_argument('--theta', type=float, default=0.5, help='Parameter for the implicit theta timestepping method.')
parser.add_argument('--width', type=float, default=0.2, help='Width of the Gaussian bump.')
parser.add_argument('--nwindows', type=int, default=1, help='Number of time-windows.')
parser.add_argument('--nslices', type=int, default=2, help='Number of time-slices per time-window.')
parser.add_argument('--slice_length', type=int, default=2, help='Number of timesteps per time-slice.')
parser.add_argument('--alpha', type=float, default=0.0001, help='Circulant coefficient.')
parser.add_argument('--modes', type=int, default=4, help='Number of time-varying modes. 0 for constant-coefficient')
parser.add_argument('--cycle_type', type=str, default='fv', help='Cycle type. "v" for v-cycle, "d" for down cycle, "fv" for full cycle with v sweeps, "fd" for full cycle with down sweeps.')
parser.add_argument('--last_cycle', type=str, default='v', help='Type of the final sweep for full cycles. "v" for v cycle, "d" for down cycle.')
parser.add_argument('--coarsen', type=int, default=2, help='Coarsening ratio between SliceJacobiPC levels.')
parser.add_argument('--levels', type=int, default=1, help='Number of coarser levels. 0 for only the CirculantPC.')
parser.add_argument('--nsample', type=int, default=32, help='Number of sample points for plotting.')
parser.add_argument('--mpeg', action='store_true', help='Output an mpeg of the timeseries.')
parser.add_argument('--write_metrics', action='store_true', help='Write various solver metrics to file.')
parser.add_argument('--show_args', action='store_true', default=True, help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

# The time partition describes how many timesteps are included on each time-slice of the ensemble
# Here we use the same number of timesteps on each slice, but they can be different

time_partition = tuple(args.slice_length for _ in range(args.nslices))
window_length = sum(time_partition)
nsteps = args.nwindows*window_length
nt = window_length

# Calculate the CFL from the timestep number
ubar = 1.
dx = 1./args.nx
dt = args.dt
cfl = ubar*dt/dx
T = dt*nt
pi2 = fd.Constant(2*pi)

# frequency of velocity oscillations in time
omegas = [1.43, 2.27, 4.83, 6.94]

# how many points per wavelength for each spatial and temporal frequency
PETSc.Sys.Print(f"{cfl = }, {T = }, {nt = }")
PETSc.Sys.Print(f"Periods: {[round(T*o,3) for o in omegas[:args.modes]]}")

PETSc.Sys.Print(f"Temporal resolution (ppm): {[round(1/(o*dt),3) for o in omegas[:args.modes]]}")
PETSc.Sys.Print(f"Spatial resolution (ppm): {round(1/dx,3)}, {round((1/2)/dx,3)}, {round((1/4)/dx,3)}, {round((1/6)/dx,3)}")

# The Ensemble with the spatial and time communicators
ensemble = asQ.create_ensemble(time_partition)

# # # === --- domain --- === # # #

# The mesh needs to be created with the spatial communicator
mesh = fd.PeriodicUnitSquareMesh(args.nx, args.nx, quadrilateral=True, comm=ensemble.comm)

# We use a discontinuous Galerkin space for the advected scalar
V = fd.FunctionSpace(mesh, "DQ", args.degree)

# # # === --- initial conditions --- === # # #

x, y = fd.SpatialCoordinate(mesh)


def radius(x, y):
    return fd.sqrt(pow(x-0.5, 2) + pow(y-0.5, 2))


def gaussian(x, y):
    return fd.exp(-0.5*pow(radius(x, y)/args.width, 2))


# The scalar initial conditions are a Gaussian bump centred at (0.5, 0.5)
q0 = fd.Function(V, name="scalar_initial")
q0.interpolate(1 + gaussian(x, y))


# The advecting velocity field oscillates around a mean value in time and space
def velocity(t):
    c = fd.as_vector((fd.Constant(ubar*cos(args.angle)), fd.Constant(ubar*sin(args.angle))))
    if args.modes > 0:
        c += fd.sin(pi2*omegas[0]*t-0.0)*fd.as_vector([+0.25*fd.sin(1*x*pi2+0.3), +0.20*fd.cos(1*y*pi2-0.9)])
    if args.modes > 1:
        c -= fd.cos(pi2*omegas[1]*t-0.7)*fd.as_vector([-0.05*fd.cos(2*x*pi2-0.8), +0.10*fd.sin(2*x*pi2+0.1)])
    if args.modes > 2:
        c += fd.sin(pi2*omegas[2]*t+0.6)*fd.as_vector([+0.15*fd.cos(4*x*pi2+0.0), -0.05*fd.sin(4*x*pi2-0.4)])
    if args.modes > 3:
        c -= fd.cos(pi2*omegas[3]*t-0.3)*fd.as_vector([-0.03*fd.cos(6*x*pi2-0.9), +0.08*fd.sin(6*x*pi2+0.2)])
    return c


# # # === --- finite element forms --- === # # #


# The time-derivative mass form for the scalar advection equation.
# asQ assumes that the mass form is linear so here
# q is a TrialFunction and phi is a TestFunction
def form_mass(q, phi):
    return phi*q*fd.dx


# The DG advection form for the scalar advection equation.
# asQ assumes that the function form is nonlinear so here
# q is a Function and phi is a TestFunction
def form_function(q, phi, t):
    n = fd.FacetNormal(mesh)
    u = velocity(t)

    # upwind switch
    un = fd.Constant(0.5)*(fd.dot(u, n) + abs(fd.dot(u, n)))

    # integration over element volume
    int_cell = q*fd.div(phi*u)*fd.dx

    # integration over internal facets
    int_facet = (phi('+')-phi('-'))*(un('+')*q('+')-un('-')*q('-'))*fd.dS

    return int_facet - int_cell


# # # === --- PETSc solver parameters --- === # # #


# The PETSc solver parameters used to solve the
# blocks in step (b) of inverting the ParaDiag matrix.
# fixed number of iteration so we can use gmres outside.
block_parameters = {
    # 'ksp_type': 'preonly',
    # 'pc_type': 'lu',
    # 'pc_factor_mat_solver_type': 'mumps'
    'pc_type': 'ilu',
    'ksp_type': 'richardson',
    'ksp_richardson_scale': 0.8,
    # 'ksp_rtol': 1e-5,
    'ksp_max_it': 50,
    'ksp_convergence_test': 'skip',
    'converged_maxits': None,
}

jacobi_parameters = {
    'pc_type': 'python',
    'pc_python_type': 'asQ.JacobiPC',
    'aaojacobi_block': block_parameters,
}

circulant_parameters = {
    'pc_type': 'python',
    'pc_python_type': 'asQ.CirculantPC',
    'circulant_alpha': args.alpha,
    'circulant_block': block_parameters,
}


def slice_parameters(nsteps):
    return {
        'pc_type': 'python',
        'pc_python_type': 'asQ.SliceJacobiPC',
        'slice_jacobi_nsteps': nsteps,
        'slice_jacobi_slice': circulant_parameters,
        'slice_jacobi_slice_ksp': {
            'type': 'gmres',
            'richardson_scale': 0.8,
            'max_it': 1 if nsteps > 0 else 1,
            'convergence_test': 'skip',
            'converged_maxits': None,
        },
        # 'slice_jacobi_slice_0_ksp_converged_rate': None,
    } if nsteps > 1 else jacobi_parameters


def composite_parameters(*nsteps):
    params = {
        'pc_type': 'composite',
        'pc_composite_type': 'multiplicative',
        'pc_composite_pcs': ','.join('ksp' for _ in range(len(nsteps))),
    }
    for i, s in enumerate(nsteps):
        params[f'sub_{i}_ksp'] = slice_parameters(s)
        params[f'sub_{i}_ksp'].update({
            'ksp_type': 'richardson',
            'ksp_richardson_scale': 0.9,
            'ksp_max_it': 4 if s != nt else 1,
            'ksp_convergence_test': 'skip',
            'ksp_converged_maxits': None,
            'ksp_converged_rate': None,
            # 'ksp_monitor_true_residual': None,
        })
    return params


def composite_composite_parameters(nsteps_list):
    return {
        'pc_type': 'composite',
        'pc_composite_type': 'multiplicative',
        'pc_composite_pcs': ','.join('composite' for _ in range(len(nsteps_list))),
    } | {f'sub_{i}': composite_parameters(*nsteps)
         for i, nsteps in enumerate(nsteps_list)}


paradiag_parameters = {
    'snes_type': 'ksponly',  # for a linear system
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp_gmres_restart': nt,
    'ksp_richardson_scale': 0.9,
    'ksp': {
        # 'monitor_true_residual': None,
        'monitor': None,
        'converged_rate': None,
        'rtol': 1e-4,
        'atol': 1e-12,
        'stol': 1e-12,
        # 'view': None,
    },
}


# calculate the number of timesteps for a sequence of levels given a 'coarsening' ratio
# level 0 has all timesteps, level 1 has nt/ratio per slice, level 2 has nt/ratio^2 etc
def level_steps(n, ratio, levels):
    from math import log
    max_level = int(log(n)/log(ratio))
    return tuple(n//pow(ratio, l if l >= 0 else max_level+1+l)
                 for l in levels)


# PCComposite can only have max 8 pcs so we nest them with n=8
def unflatten(x, n):
    from math import ceil
    npacks = int(ceil(len(x)/n))
    return tuple(tuple(x[i*n:min((i+1)*n, len(x))])
                 for i in range(npacks))


def dcycle(n, *args, **kwargs):
    return tuple(range(n+1))


def vcycle(n, *args, **kwargs):
    return tuple((*dcycle(n), *range(n-1, -1, -1)))


def fcycle(n, ctype='v', last_cycle='v', initial=True):
    if ctype == 'v':
        return tuple((*fcycle(n-1, ctype, initial=False), *cycle(n, last_cycle)[1:]) if n > 0 else (0,))
    elif ctype == 'd':
        post = cycle(n, last_cycle) if initial else dcycle(n)
        return tuple((*fcycle(n-1, ctype, initial=False), *post) if n > 0 else ())


def cycle(n, ctype, **kwargs):
    if n == 0:
        return tuple((0,))
    if ctype == 'd':
        return dcycle(n, **kwargs)
    elif ctype == 'v':
        return vcycle(n, **kwargs)
    elif ctype[0] == 'f':
        return fcycle(n, ctype=ctype[1], **kwargs)
    else:
        raise ValueError(f"unknown cycle type {ctype = }")


composite_steps = level_steps(nt, args.coarsen,
                              cycle(args.levels, args.cycle_type, last_cycle=args.last_cycle))
PETSc.Sys.Print(f"{len(composite_steps)} sweeps: {composite_steps}")
composite_steps = unflatten(composite_steps, 8)

paradiag_parameters.update(composite_composite_parameters(composite_steps))
# paradiag_parameters.update(circulant_parameters)

# from json import dumps
# PETSc.Sys.Print(dumps(paradiag_parameters, indent=3))
# PETSc.Sys.Print(composite_steps)
# from sys import exit; exit()


# # # === --- Setup ParaDiag --- === # # #


# Give everything to asQ to create the paradiag object.
pdg = asQ.Paradiag(ensemble=ensemble,
                   form_function=form_function,
                   form_mass=form_mass,
                   ics=q0, dt=dt, theta=args.theta,
                   time_partition=time_partition,
                   solver_parameters=paradiag_parameters)


# This is a callback which will be called before pdg solves each time-window
# We can use this to make the output a bit easier to read
def window_preproc(pdg, wndw, rhs):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'### === --- Calculating time-window {wndw} --- === ###')
    PETSc.Sys.Print('')


# The last time-slice will be saving snapshots to create an animation.
# The layout member describes the time_partition.
# layout.is_local(i) returns True/False if the timestep index i is on the
# current time-slice. Here we use -1 to mean the last timestep in the window.
is_last_slice = pdg.layout.is_local(-1)

# Make an output Function on the last time-slice and start a snapshot list
if is_last_slice and args.mpeg:
    qout = fd.Function(V)
    timeseries = [q0.copy(deepcopy=True)]


# This is a callback which will be called after pdg solves each time-window
# We can use this to save the last timestep of each window for plotting.
def window_postproc(pdg, wndw, rhs):
    if is_last_slice and args.mpeg:
        # The aaofunc is the AllAtOnceFunction which represents the time-series.
        # indexing the AllAtOnceFunction accesses one timestep on the local slice.
        # -1 is again used to get the last timestep and place it in qout.
        qout.assign(pdg.aaofunc[-1])
        timeseries.append(qout)


# Solve nwindows of the all-at-once system
pdg.solve(args.nwindows,
          preproc=window_preproc,
          postproc=window_postproc)


# # # === --- Postprocessing --- === # # #

# paradiag collects a few solver diagnostics for us to inspect
nw = args.nwindows

# Number of linear iterations of the all-at-once system, total and per window.
PETSc.Sys.Print(f'linear iterations: {pdg.linear_iterations}  |  iterations per window: {pdg.linear_iterations/nw}')

# Number of iterations needed for each block in step-(b), total and per block solve
# The number of iterations for each block will usually be different because of the different eigenvalues
# block_iterations = pdg.solver.jacobian.pc.block_iterations
# PETSc.Sys.Print(f'block linear iterations: {block_iterations.data()}  |  iterations per block solve: {block_iterations.data()/pdg.linear_iterations}')

# We can write these diagnostics to file, along with some other useful information.
# Files written are: aaos_metrics.txt, block_metrics.txt, paradiag_setup.txt, solver_parameters.txt
if args.write_metrics:
    asQ.write_paradiag_metrics(pdg)

# Make an animation from the snapshots we collected and save it to periodic.mp4.
if is_last_slice and args.mpeg:
    from matplotlib.animation import FuncAnimation

    fn_plotter = fd.FunctionPlotter(mesh, num_sample_points=args.nsample)

    fig, axes = plt.subplots()
    axes.set_aspect('equal')
    colors = fd.tripcolor(qout, num_sample_points=args.nsample, vmin=1, vmax=2, axes=axes)
    fig.colorbar(colors)

    def animate(q):
        colors.set_array(fn_plotter(q))

    interval = 1e2
    animation = FuncAnimation(fig, animate, frames=timeseries, interval=interval)

    animation.save("periodic.mp4", writer="ffmpeg")
