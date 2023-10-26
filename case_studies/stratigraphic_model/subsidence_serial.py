import firedrake as fd
from math import pi
from utils.serial import SerialMiniApp

N = 250
mesh = fd.SquareMesh(N, N, 100000, quadrilateral=False)
hierarchy = fd.MeshHierarchy(mesh, 2)
mesh = hierarchy[-1]

V = fd.FunctionSpace(mesh, "CG", 1)

# # # === --- initial conditions --- === # # #

x, y = fd.SpatialCoordinate(mesh)

s0 = fd.Function(V, name="scalar_initial")
s0.interpolate(fd.Constant(0.0))


# The sediment movement D
def D(D_c, d):
    return D_c*2*1e6/fd.Constant(fd.sqrt(2*pi))*fd.exp(-1/2*((d-5)/10)**2)


# The carbonate growth L.
def L(G_0, d):
    return G_0*fd.conditional(d > 0, fd.exp(-d/10)/(1 + fd.exp(-50*d)), fd.exp((50-1/10)*d)/(fd.exp(50*d) + 1))


def Subsidence(t):
    return -((100000-x)/10000)*(t/100000)


def sea_level(A, t):
    return A*fd.sin(2*pi*t/500000)


# # # === --- finite element forms --- === # # #
def form_mass(s, q):
    return (s*q*fd.dx)


D_c = fd.Constant(.002)
G_0 = fd.Constant(.004)
A = fd.Constant(50)
b = 100*fd.tanh(1/20000*(x-50000))


def form_function(s, q, t):
    return (D(D_c, sea_level(A, t)-b-Subsidence(t)-s)*fd.inner(fd.grad(s), fd.grad(q))*fd.dx-L(G_0, sea_level(A, t)-b-Subsidence(t)-s)*q*fd.dx)


sp = {
    "mat_type": "aij",
    "snes_type": "newtonls",
    "snes_atol": 1.0e-8,
    "snes_rtol": 1.0e-8,
    "snes_stol": 1.0e-100,
    "snes_monitor": None,
    "ksp_type": "fgmres",
    "ksp_norm_type": "unpreconditioned",
    "ksp_monitor": None,
    "snes_linesearch_type": "l2",
    "snes_linesearch_monitor": None,
    "ksp_converged_reason": None,
    "snes_ksp_ew": None,
    "ksp_max_it": 100,
    "ksp_gmres_restart": 50,
    "pc_type": "mg",
    "pc_mg_cycle_type": "v",
    "pc_mg_type": "multiplicative",
    "mg_levels_ksp_type": "gmres",
    "mg_levels_ksp_max_it": 3,
    "mg_levels_pc_type": "python",
    "mg_levels_pc_python_type": "firedrake.PatchPC",
    "mg_levels_patch_pc_patch_save_operators": True,
    "mg_levels_patch_pc_patch_partition_of_unity": False,
    "mg_levels_patch_pc_patch_sub_mat_type": "seqdense",
    "mg_levels_patch_pc_patch_construct_codim": 0,
    "mg_levels_patch_pc_patch_construct_type": "vanka",
    "mg_levels_patch_pc_patch_exclude_subspaces": 1,
    "mg_levels_patch_pc_patch_precompute_element_tensors": True,
    "mg_levels_patch_sub_ksp_type": "preonly",
    "mg_levels_patch_sub_pc_type": "lu",
    "mg_levels_patch_pc_factor_mat_solver_type": "mumps",
    "mg_coarse_pc_type": "python",
    "mg_coarse_pc_python_type": "firedrake.AssembledPC",
    "mg_coarse_assembled_pc_type": "lu",
    "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
}


dt = 1000
theta = 0.5

miniapp = SerialMiniApp(dt, theta,
                        s0,
                        form_mass,
                        form_function,
                        sp)
# Solve of the required number of timesteps j
j = 128
miniapp.solve(j)
