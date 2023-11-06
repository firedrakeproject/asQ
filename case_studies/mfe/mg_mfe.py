
import firedrake as fd
from firedrake.petsc import PETSc

from utils import mg
from utils.planets import earth
import utils.shallow_water as swe
import utils.shallow_water.gravity_bumps as case

from utils.serial import SerialMiniApp

PETSc.Sys.popErrorHandler()

PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Setting up --- === ###')
PETSc.Sys.Print('')

# icosahedral mg mesh
mesh = swe.create_mg_globe_mesh(ref_level=3, coords_degree=1)
x = fd.SpatialCoordinate(mesh)

# time step
dt = 1

# shallow water equation function spaces (velocity and depth)
W = swe.default_function_space(mesh, degree=1)

# parameters
coriolis = case.coriolis_expression(*x)


# initial conditions
w_initial = fd.Function(W)
u_initial, h_initial = w_initial.subfunctions

u_initial.project(case.velocity_expression(*x))
h_initial.project(case.depth_expression(*x))


# shallow water equation forms
def form_function(u, h, v, q, t):
    return swe.linear.form_function(mesh,
                                    earth.Gravity, case.H,
                                    coriolis,
                                    u, h, v, q, t)


def form_mass(u, h, v, q):
    return swe.linear.form_mass(mesh, u, h, v, q)


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
        'pc_type': 'jacobi',
        # 'pc_type': 'python',
        # 'pc_python_type': 'firedrake.PatchPC',
        # 'patch': patch_parameters
    },
    'coarse': {
        'pc_type': 'python',
        'pc_python_type': 'firedrake.AssembledPC',
        'assembled_pc_type': 'lu',
        'assembled_pc_factor_mat_solver_type': 'mumps',
    },
}

# solver parameters for the implicit solve
sparameters = {
    'snes_type': 'ksponly',
    'snes': {
        'monitor': None,
        'converged_reason': None,
        'atol': 1e-0,
        'rtol': 1e-10,
        'stol': 1e-12,
    },
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp': {
        'atol': 1e-0,
        'rtol': 1e-10,
        'stol': 1e-12,
        'monitor': None,
        'converged_reason': None
    },
    'pc_type': 'mg',
    'pc_mg_cycle_type': 'v',
    'pc_mg_type': 'multiplicative',
    'mg': mg_parameters
}

# set up nonlinear solver
miniapp = SerialMiniApp(dt, 0.5,
                        w_initial,
                        form_mass,
                        form_function,
                        sparameters)

miniapp.nlsolver.set_transfer_manager(
    mg.manifold_transfer_manager(W))

PETSc.Sys.Print('### === --- Timestepping loop --- === ###')


def preproc(app, step, t):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'=== --- Timestep {step} --- ===')
    PETSc.Sys.Print('')


miniapp.solve(nt=1, preproc=preproc)
