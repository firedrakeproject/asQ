import firedrake as fd

from utils import units
from utils import mg
from utils.planets import earth
import utils.shallow_water.nonlinear as swe
from utils.shallow_water.williamson1992 import case5

from petsc4py import PETSc


def williamson5_serial(base_level=1,
                       ref_level=2,
                       tmax=4,
                       dumpt=1,
                       dt=1,
                       coords_degree=3,
                       degree=1):

    # some domain, parameters and FS setup
    R0 = earth.radius
    H = case5.H0

    distribution_parameters = {"partition": True, "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)}

    mesh = mg.icosahedral_mesh(R0=R0,
                               base_level=base_level,
                               degree=coords_degree,
                               distribution_parameters=distribution_parameters,
                               nrefs=ref_level-base_level)

    R0 = fd.Constant(R0)
    x, y, z = fd.SpatialCoordinate(mesh)

    V1 = fd.FunctionSpace(mesh, "BDM", degree+1)
    V2 = fd.FunctionSpace(mesh, "DG", degree)
    W = fd.MixedFunctionSpace((V1, V2))

    # U_t + N(U) = 0
    #
    # TRAPEZOIDAL RULE
    # U^{n+1} - U^n + dt*( N(U^{n+1}) + N(U^n) )/2 = 0.

    # Newton's method
    # f(x) = 0, f:R^M -> R^M
    # [Df(x)]_{i,j} = df_i/dx_j
    # x^0, x^1, ...
    # Df(x^k).xp = -f(x^k)
    # x^{k+1} = x^k + xp.

    f = case5.coriolis_expression(x, y, z)
    b = case5.topography_function(x, y, z, V2, name="Topography")
    g = earth.Gravity

    Un = fd.Function(W)
    Unp1 = fd.Function(W)

    v, phi = fd.TestFunctions(W)

    u0, h0 = fd.split(Un)
    u1, h1 = fd.split(Unp1)

    half = fd.Constant(0.5)
    dT = fd.Constant(0.)

    equation = (
        swe.form_mass(mesh, h1-h0, u1-u0, phi, v)
        + half*dT*swe.form_function(mesh, g, b, f, h0, u0, phi, v)
        + half*dT*swe.form_function(mesh, g, b, f, h1, u1, phi, v))

    # monolithic solver options

    sparameters = {
        "snes_monitor": None,
        "snes_atol": 1e-8,
        # "snes_rtol": 1e-8,
        "mat_type": "matfree",
        "ksp_type": "fgmres",
        # "ksp_monitor_true_residual": None,
        "ksp_converged_reason": None,
        "ksp_atol": 1e-8,
        # "ksp_rtol": 1e-8,
        "ksp_max_it": 400,
        "pc_type": "mg",
        "pc_mg_cycle_type": "v",
        "pc_mg_type": "multiplicative",
        "mg_levels_ksp_type": "gmres",
        "mg_levels_ksp_max_it": 3,
        # "mg_levels_ksp_convergence_test": "skip",
        "mg_levels_pc_type": "python",
        "mg_levels_pc_python_type": "firedrake.PatchPC",
        "mg_levels_patch_pc_patch_save_operators": True,
        "mg_levels_patch_pc_patch_partition_of_unity": True,
        "mg_levels_patch_pc_patch_sub_mat_type": "seqdense",
        "mg_levels_patch_pc_patch_construct_codim": 0,
        "mg_levels_patch_pc_patch_construct_type": "vanka",
        "mg_levels_patch_pc_patch_local_type": "additive",
        "mg_levels_patch_pc_patch_precompute_element_tensors": True,
        "mg_levels_patch_pc_patch_symmetrise_sweep": False,
        "mg_levels_patch_sub_ksp_type": "preonly",
        "mg_levels_patch_sub_pc_type": "lu",
        "mg_levels_patch_sub_pc_factor_shift_type": "nonzero",
        "mg_coarse_pc_type": "python",
        "mg_coarse_pc_python_type": "firedrake.AssembledPC",
        "mg_coarse_assembled_pc_type": "lu",
        "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
    }

    dt = dt*units.hour
    dT.assign(dt)

    nprob = fd.NonlinearVariationalProblem(equation, Unp1)
    nsolver = fd.NonlinearVariationalSolver(nprob,
                                            solver_parameters=sparameters,
                                            appctx={})

    transfermanager = mg.manifold_transfer_manager(W)
    nsolver.set_transfer_manager(transfermanager)

    un = case5.velocity_function(x, y, z, V1, name="Velocity")
    etan = case5.elevation_function(x, y, z, V2, name="Elevation")

    u0, h0 = Un.split()
    u0.assign(un)
    h0.assign(etan + H - b)

    Unp1.assign(Un)

    t = 0.
    time_series = [Un.copy(deepcopy=True)]
    for tstep in range(0, tmax):
        t += dt

        nsolver.solve()
        Un.assign(Unp1)

        if tstep % dumpt == 0:
            time_series.append(Un.copy(deepcopy=True))
            PETSc.Sys.Print('===---', 'iteration:', tstep, '|', 'time:', t/(60*60), '---===')

    return time_series
