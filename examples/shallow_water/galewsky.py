
import firedrake as fd
from petsc4py import PETSc

from utils import units
from utils.planets import earth
import utils.shallow_water as swe

from functools import partial
from math import pi
import numpy as np

PETSc.Sys.popErrorHandler()

# get command arguments
import argparse
parser = argparse.ArgumentParser(description='Galewsky testcase for ParaDiag solver using fully implicit SWE solver.')
parser.add_argument('--ref_level', type=int, default=2, help='Refinement level of icosahedral grid. Default 2.')
parser.add_argument('--nwindows', type=int, default=1, help='Number of time-windows. Default 1.')
parser.add_argument('--nslices', type=int, default=2, help='Number of time-slices per time-window. Default 2.')
parser.add_argument('--slice_length', type=int, default=2, help='Number of timesteps per time-slice. Default 2.')
parser.add_argument('--alpha', type=float, default=0.0001, help='Circulant coefficient. Default 0.0001.')
parser.add_argument('--dt', type=float, default=0.5, help='Timestep in hours. Default 0.5.')
parser.add_argument('--filename', type=str, default='galewsky')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Setting up --- === ###')
PETSc.Sys.Print('')

# time steps

M = [args.slice_length for _ in range(args.nslices)]
window_length = sum(M)
nsteps = args.nwindows*window_length

dt = args.dt*units.hour

# parameters for the implicit diagonal solve in step-(b)
sparameters = {
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp': {
        'atol': 1e-8,
        'rtol': 1e-8,
        'max_it': 400,
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

sparameters_diag = {
    'snes': {
        'linesearch_type': 'basic',
        'monitor': None,
        'converged_reason': None,
        'atol': 1e-0,
        'rtol': 1e-12,
        'stol': 1e-12,
    },
    'mat_type': 'matfree',
    'ksp_type': 'preonly',
    'ksp': {
        'monitor': None,
        'converged_reason': None,
    },
    'pc_type': 'python',
    'pc_python_type': 'asQ.DiagFFTPC'
}

for i in range(sum(M)):
    sparameters_diag['diagfft_'+str(i)+'_'] = sparameters

create_mesh = partial(
    swe.create_mg_globe_mesh,
    ref_level=args.ref_level)

# initial conditions


def b_exp(x, y, z):
    return fd.Constant(0)


umax = 80.
H = 10e3
theta0 = pi/7.
theta1 = pi/2. - theta0
en = np.exp(-4./((theta1-theta0)**2))


def u_exp(x, y, z):
    theta, lamda = earth.cart_to_sphere_coords(x, y, z)
    uzonal_expr = (umax/en)*fd.exp(1./((theta - theta0)*(theta - theta1)))
    uzonal = fd.conditional(fd.ge(theta, theta0),
                            fd.conditional(fd.le(theta, theta1),
                                           uzonal_expr, 0.), 0.)
    umerid = 0.
    return earth.sphere_to_cart_vector(x, y, z, uzonal, umerid)


def h_integrand(theta):
    # Initial D field is calculated by integrating D_integrand w.r.t. theta
    # Assumes the input is between theta0 and theta1.
    # Note that this function operates on vectorized input.
    f = 2.0*earth.omega*np.sin(theta)
    uzonal = (umax/en)*np.exp(1.0/((theta - theta0)*(theta - theta1)))
    return uzonal*(f + np.tan(theta)*uzonal/earth.radius)


def h_calculation(x):
    # Function to return value of D at X
    from scipy import integrate

    # Preallocate output array
    xdat = x.dat.data_ro

    val = np.zeros(len(xdat))

    angles = np.zeros(len(xdat))

    # Minimize work by only calculating integrals for points with
    # theta between theta_0 and theta_1.
    # For theta <= theta_0, the integral is 0
    # For theta >= theta_1, the integral is constant.

    # Precalculate this constant:
    poledepth, _ = integrate.fixed_quad(h_integrand, theta0, theta1, n=64)
    poledepth *= -earth.radius/earth.gravity

    angles[:] = np.arcsin(xdat[:, 2]/earth.radius)

    for ii in range(len(xdat)):
        if angles[ii] <= theta0:
            val[ii] = 0.0
        elif angles[ii] >= theta1:
            val[ii] = poledepth
        else:
            # Fixed quadrature with 64 points gives absolute errors below 1e-13
            # for a quantity of order 1e-3.
            v, _ = integrate.fixed_quad(h_integrand, theta0, angles[ii], n=64)
            val[ii] = -(earth.radius/earth.gravity)*v
    return val


def h_perturbation(theta, lamda, V):
    alpha = fd.Constant(1/3.)
    beta = fd.Constant(1/15.)
    hhat = fd.Constant(120)
    theta2 = fd.Constant(pi/4.)
    return fd.Function(V).interpolate(hhat*fd.cos(theta)*fd.exp(-(lamda/alpha)**2)*fd.exp(-((theta2 - theta)/beta)**2))


def h_exp(x, y, z):
    mesh = x.ufl_domain()

    V = swe.default_depth_function_space(mesh)
    W = fd.VectorFunctionSpace(mesh, V.ufl_element())
    coords = fd.interpolate(mesh.coordinates, W)
    h = fd.Function(V)
    h.dat.data[:] = h_calculation(coords)
    cells = fd.Function(V).assign(fd.Constant(1))
    area = fd.assemble(cells*fd.dx)
    hmean = fd.assemble(h*fd.dx)/area
    hpert = h_perturbation(*earth.cart_to_sphere_coords(x, y, z), V)
    h += H - hmean + hpert
    return h


PETSc.Sys.Print('### === --- Calculating parallel solution --- === ###')

miniapp = swe.ShallowWaterMiniApp(gravity=earth.Gravity,
                                  topography_expression=b_exp,
                                  velocity_expression=u_exp,
                                  depth_expression=h_exp,
                                  reference_depth=H,
                                  create_mesh=create_mesh,
                                  dt=dt, theta=0.5,
                                  alpha=args.alpha, slice_partition=M,
                                  paradiag_sparameters=sparameters_diag)

ensemble = miniapp.ensemble
time_rank = miniapp.paradiag.rT

# only last slice does diagnostics/output
if time_rank == len(M)-1:
    cfl_series = []
    linear_its = 0
    nonlinear_its = 0

    ofile = fd.File('output/'+args.filename+'.pvd',
                    comm=ensemble.comm)

    uout = fd.Function(miniapp.velocity_function_space(), name='velocity')
    hout = fd.Function(miniapp.depth_function_space(), name='depth')

    miniapp.get_velocity(-1, uout=uout)
    miniapp.get_elevation(-1, hout=hout)

    ofile.write(uout, hout,
                miniapp.potential_vorticity(uout),
                time=0)

    def time_at_last_step(w):
        return dt*(w + 1)*window_length


def window_preproc(swe_app, pdg, wndw):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'### === --- Calculating time-window {wndw} --- === ###')
    PETSc.Sys.Print('')


def window_postproc(swe_app, pdg, wndw):
    # make sure variables are properly captured
    global linear_its
    global nonlinear_its
    global cfl_series

    # postprocess this timeslice
    if time_rank == len(M)-1:
        linear_its += pdg.snes.getLinearSolveIterations()
        nonlinear_its += pdg.snes.getIterationNumber()

        swe_app.get_velocity(-1, uout=uout)
        swe_app.get_elevation(-1, hout=hout)

        time = time_at_last_step(wndw)

        ofile.write(uout, hout,
                    swe_app.potential_vorticity(uout),
                    time=time/earth.day)

        cfl = swe_app.max_cfl(dt, -1)
        cfl_series.append(cfl)

        PETSc.Sys.Print('', comm=ensemble.comm)
        PETSc.Sys.Print(f'Maximum CFL = {cfl}', comm=ensemble.comm)
        PETSc.Sys.Print(f'Hours = {time/units.hour}', comm=ensemble.comm)
        PETSc.Sys.Print(f'Days = {time/earth.day}', comm=ensemble.comm)
        PETSc.Sys.Print('', comm=ensemble.comm)
    PETSc.Sys.Print('')


# solve for each window
if args.nwindows == 0:
    quit()

miniapp.solve(nwindows=args.nwindows,
              preproc=window_preproc,
              postproc=window_postproc)


PETSc.Sys.Print('### === --- Iteration counts --- === ###')
PETSc.Sys.Print('')

if time_rank == len(M)-1:
    PETSc.Sys.Print(f'Maximum CFL = {max(cfl_series)}', comm=ensemble.comm)
    PETSc.Sys.Print(f'Minimum CFL = {min(cfl_series)}', comm=ensemble.comm)
    PETSc.Sys.Print('', comm=ensemble.comm)

    PETSc.Sys.Print(f'windows: {(args.nwindows)}', comm=ensemble.comm)
    PETSc.Sys.Print(f'timesteps: {(args.nwindows)*window_length}', comm=ensemble.comm)
    PETSc.Sys.Print('', comm=ensemble.comm)

    PETSc.Sys.Print(f'linear iterations: {linear_its} | iterations per window: {linear_its/(args.nwindows)}', comm=ensemble.comm)
    PETSc.Sys.Print(f'nonlinear iterations: {nonlinear_its} | iterations per window: {nonlinear_its/(args.nwindows)}', comm=ensemble.comm)
    PETSc.Sys.Print('', comm=ensemble.comm)
