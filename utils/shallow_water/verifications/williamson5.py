import firedrake as fd
import asQ

from utils import units
from utils import mg
from utils.planets import earth
import utils.shallow_water.nonlinear as swe
from utils.shallow_water.williamson1992 import case5

from petsc4py import PETSc


def serial_solve(base_level=1,
                 ref_level=2,
                 tmax=4,
                 dumpt=1,
                 dt=1,
                 coords_degree=3,
                 degree=1,
                 sparameters=None,
                 comm=fd.COMM_WORLD,
                 verbose=False):

    # some domain, parameters and FS setup
    R0 = earth.radius
    H = case5.H0

    distribution_parameters = {"partition": True, "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)}

    mesh = mg.icosahedral_mesh(R0=R0,
                               base_level=base_level,
                               degree=coords_degree,
                               distribution_parameters=distribution_parameters,
                               nrefs=ref_level-base_level,
                               comm=comm)

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

    wn = fd.Function(W)
    wn1 = fd.Function(W)

    v, phi = fd.TestFunctions(W)

    u0, h0 = fd.split(wn)
    u1, h1 = fd.split(wn1)

    half = fd.Constant(0.5)
    dT = fd.Constant(0.)

    equation = (
        swe.form_mass(mesh, u1-u0, h1-h0, v, phi)
        + half*dT*swe.form_function(mesh, g, b, f, u0, h0, v, phi)
        + half*dT*swe.form_function(mesh, g, b, f, u1, h1, v, phi))

    # monolithic solver options

    if sparameters is None:
        sparameters = {
            "snes_atol": 1e-8,
            "mat_type": "matfree",
            "ksp_type": "fgmres",
            "ksp_atol": 1e-8,
            "ksp_max_it": 400,
            "pc_type": "mg",
            "pc_mg_cycle_type": "v",
            "pc_mg_type": "multiplicative",
            "mg_levels_ksp_type": "gmres",
            "mg_levels_ksp_max_it": 3,
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

    if verbose is True:
        sparameters['snes_monitor'] = None
        sparameters['ksp_monitor'] = None

    dT.assign(dt*units.hour)

    nprob = fd.NonlinearVariationalProblem(equation, wn1)
    nsolver = fd.NonlinearVariationalSolver(nprob,
                                            solver_parameters=sparameters,
                                            appctx={})

    transfermanager = mg.manifold_transfer_manager(W)
    nsolver.set_transfer_manager(transfermanager)

    un = case5.velocity_function(x, y, z, V1, name="Velocity")
    etan = case5.elevation_function(x, y, z, V2, name="Elevation")

    u0, h0 = wn.split()
    u0.assign(un)
    h0.assign(etan + H - b)

    wn1.assign(wn)

    t = 0.
    time_series = []
    time_series.append(wn.copy(deepcopy=True))
    for tstep in range(0, tmax):
        t += dt

        nsolver.solve()
        wn.assign(wn1)

        if tstep % dumpt == 0:
            time_series.append(wn.copy(deepcopy=True))
            if verbose is True:
                PETSc.Sys.Print('===---', 'iteration:', tstep, '|', 'time:', t/(60*60), '---===')

    return time_series


def parallel_solve(base_level=1,
                   ref_level=2,
                   M=[2, 2],
                   nwindows=1,
                   nspatial_domains=2,
                   dumpt=1,
                   dt=1,
                   coords_degree=3,
                   degree=1,
                   sparameters=None,
                   sparameters_diag=None,
                   ensemble=None,
                   alpha=0.0001,
                   verbose=False):

    # mesh set up
    if ensemble is None:
        ensemble = fd.Ensemble(fd.COMM_WORLD, nspatial_domains)

    # some domain, parameters and FS setup
    distribution_parameters = {"partition": True, "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)}

    mesh = mg.icosahedral_mesh(R0=earth.radius,
                               base_level=base_level,
                               degree=coords_degree,
                               distribution_parameters=distribution_parameters,
                               nrefs=ref_level-base_level,
                               comm=ensemble.comm)

    x = fd.SpatialCoordinate(mesh)

    V1 = fd.FunctionSpace(mesh, "BDM", degree+1)
    V2 = fd.FunctionSpace(mesh, "DG", degree)
    W = fd.MixedFunctionSpace((V1, V2))

    H = case5.H0
    g = earth.Gravity
    f = case5.coriolis_expression(*x)
    b = case5.topography_function(*x, V2, name="Topography")
    # D = eta + b

    # initial conditions

    # W = V1 * V2
    w0 = fd.Function(W)
    un, hn = w0.split()
    un.project(case5.velocity_expression(*x))
    etan = case5.elevation_function(*x, V2, name="Elevation")
    hn.assign(etan + H - b)

    # nonlinear swe forms

    def form_function(u, h, v, q):
        return swe.form_function(mesh, g, b, f, u, h, v, q)

    def form_mass(u, h, v, q):
        return swe.form_mass(mesh, u, h, v, q)

    dT = dt*units.hour

    # parameters for the implicit diagonal solve in step-(b)
    if sparameters is None:
        sparameters = {
            "mat_type": "matfree",
            "ksp_type": "preonly",
            "ksp_atol": 1e-8,
            "ksp_rtol": 1e-8,
            "ksp_max_it": 400,
            "pc_type": "mg",
            "pc_mg_cycle_type": "v",
            "pc_mg_type": "multiplicative",
            "mg_levels_ksp_type": "gmres",
            "mg_levels_ksp_max_it": 5,
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

    if sparameters_diag is None:
        sparameters_diag = {
            'snes_linesearch_type': 'basic',
            'mat_type': 'matfree',
            'ksp_type': 'gmres',
            'ksp_max_it': 10,
            'pc_type': 'python',
            'pc_python_type': 'asQ.DiagFFTPC'
        }

    if verbose:
        sparameters_diag['snes_monitor'] = None
        sparameters_diag['snes_converged_reason'] = None
        sparameters_diag['ksp_monitor'] = None
        sparameters_diag['ksp_converged_reason'] = None

    for i in range(sum(M)):  # should this be sum(M) or max(M)?
        sparameters_diag["diagfft_"+str(i)+"_"] = sparameters

    theta = 0.5

    # non-petsc information for block solve
    block_ctx = {}

    # mesh transfer operators
    transfer_managers = []
    for _ in range(sum(M)):
        tm = mg.manifold_transfer_manager(W)
        transfer_managers += [tm]

    block_ctx['diag_transfer_managers'] = transfer_managers

    PETSc.Sys.Print("setting up paradiag")
    PD = asQ.paradiag(ensemble=ensemble,
                      form_function=form_function,
                      form_mass=form_mass, w0=w0,
                      dt=dT, theta=theta,
                      alpha=alpha,
                      time_partition=M, solver_parameters=sparameters_diag,
                      circ=None, tol=1.0e-8, maxits=None,
                      ctx={}, block_ctx=block_ctx, block_mat_type="aij")

    trank = PD.time_rank

    # list of snapshots: time_series[window][slice-local-index]
    time_series = []
    wp = fd.Function(W)
    up, hp = wp.split()

    PETSc.Sys.Print("paradiag solve:")
    for w in range(nwindows):
        if verbose:
            PETSc.Sys.Print('')
            PETSc.Sys.Print(f'### === --- Calculating time-window {w} --- === ###')
            PETSc.Sys.Print('')

        PD.solve()

        # build time series for this window
        window_series = []

        for i in range(M[trank]):

            up.assign(PD.aaos.w_all.split()[2*i])
            hp.assign(PD.aaos.w_all.split()[2*i+1])

            window_series.append(wp.copy(deepcopy=True))

        time_series.append(window_series)

        PD.aaos.next_window()

    return time_series
