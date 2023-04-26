import asQ
import firedrake as fd
import numpy as np
import pytest
from petsc4py import PETSc
from functools import reduce
from operator import mul


@pytest.mark.parallel(nprocs=4)
def test_galewsky_timeseries():
    from utils import units
    from utils import mg
    from utils.planets import earth
    import utils.shallow_water as swe
    from utils.shallow_water import galewsky
    from utils.serial import ComparisonMiniapp
    from copy import deepcopy

    ref_level = 2
    nwindows = 1
    nslices = 2
    slice_length = 2
    alpha = 0.0001
    dt = 0.5
    theta = 0.5
    degree = swe.default_degree()

    time_partition = [slice_length for _ in range(nslices)]

    dt = dt*units.hour

    ensemble = asQ.create_ensemble(time_partition)

    # icosahedral mg mesh
    mesh = swe.create_mg_globe_mesh(ref_level=ref_level,
                                    comm=ensemble.comm,
                                    coords_degree=1)
    x = fd.SpatialCoordinate(mesh)

    # shallow water equation function spaces (velocity and depth)
    W = swe.default_function_space(mesh, degree=degree)

    # parameters
    gravity = earth.Gravity

    topography = galewsky.topography_expression(*x)
    coriolis = swe.earth_coriolis_expression(*x)

    # initial conditions
    w_initial = fd.Function(W)
    u_initial = w_initial.subfunctions[0]
    h_initial = w_initial.subfunctions[1]

    u_initial.project(galewsky.velocity_expression(*x))
    h_initial.project(galewsky.depth_expression(*x))

    # shallow water equation forms
    def form_function(u, h, v, q):
        return swe.nonlinear.form_function(mesh,
                                           gravity,
                                           topography,
                                           coriolis,
                                           u, h, v, q)

    def form_mass(u, h, v, q):
        return swe.nonlinear.form_mass(mesh, u, h, v, q)

    # vanka patch smoother
    patch_parameters = {
        'pc_patch': {
            'save_operators': True,
            'partition_of_unity': True,
            'sub_mat_type': 'seqdense',
            'construct_dim': 0,
            'construct_type': 'vanka',
            'local_type': 'additive',
            'precompute_element_tensors': True,
            'symmetrise_sweep': False,
        },
        'sub': {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_shift_type': 'nonzero',
        }
    }

    # mg with patch smoother
    mg_parameters = {
        'levels': {
            'ksp_type': 'gmres',
            'ksp_max_it': 5,
            'pc_type': 'python',
            'pc_python_type': 'firedrake.PatchPC',
            'patch': patch_parameters
        },
        'coarse': {
            'pc_type': 'python',
            'pc_python_type': 'firedrake.AssembledPC',
            'assembled_pc_type': 'lu',
            'assembled_pc_factor_mat_solver_type': 'mumps',
        }
    }

    # parameters for the implicit solves at:
    #   each Newton iteration of serial method
    #   each diagonal block solve in step-(b) of parallel method
    block_sparameters = {
        'mat_type': 'matfree',
        'ksp_type': 'fgmres',
        'ksp': {
            'atol': 1e-5,
            'rtol': 1e-5,
        },
        'pc_type': 'mg',
        'pc_mg_cycle_type': 'v',
        'pc_mg_type': 'multiplicative',
        'mg': mg_parameters
    }

    # nonlinear solver options
    snes_sparameters = {
        'monitor': None,
        'converged_reason': None,
        'atol': 1e-0,
        'rtol': 1e-10,
        'stol': 1e-12,
        'ksp_ew': None,
        'ksp_ew_version': 1,
    }

    # solver parameters for serial method
    serial_sparameters = {
        'snes': snes_sparameters
    }
    serial_sparameters.update(deepcopy(block_sparameters))
    serial_sparameters['ksp']['monitor'] = None
    serial_sparameters['ksp']['converged_reason'] = None

    # solver parameters for parallel method
    parallel_sparameters = {
        'snes': snes_sparameters,
        'mat_type': 'matfree',
        'ksp_type': 'fgmres',
        'ksp': {
            'monitor': None,
            'converged_reason': None,
        },
        'pc_type': 'python',
        'pc_python_type': 'asQ.DiagFFTPC',
    }

    for i in range(sum(time_partition)):
        parallel_sparameters['diagfft_block_'+str(i)] = block_sparameters

    block_ctx = {}
    transfer_managers = []
    for _ in range(time_partition[ensemble.ensemble_comm.rank]):
        tm = mg.manifold_transfer_manager(W)
        transfer_managers.append(tm)
    block_ctx['diag_transfer_managers'] = transfer_managers

    miniapp = ComparisonMiniapp(ensemble, time_partition,
                                form_mass,
                                form_function,
                                w_initial,
                                dt, theta, alpha,
                                serial_sparameters,
                                parallel_sparameters,
                                circ=None, block_ctx=block_ctx)

    miniapp.serial_app.nlsolver.set_transfer_manager(
        mg.manifold_transfer_manager(W))

    norm0 = fd.norm(w_initial)

    def preproc(serial_app, paradiag, wndw):
        PETSc.Sys.Print('')
        PETSc.Sys.Print(f'### === --- Time window {wndw} --- === ###')
        PETSc.Sys.Print('')
        PETSc.Sys.Print('=== --- Parallel solve --- ===')
        PETSc.Sys.Print('')

    def parallel_postproc(pdg, wndw):
        PETSc.Sys.Print('')
        PETSc.Sys.Print('=== --- Serial solve --- ===')
        PETSc.Sys.Print('')
        return

    PETSc.Sys.Print('')
    PETSc.Sys.Print('### === --- Timestepping loop --- === ###')

    errors = miniapp.solve(nwindows=nwindows,
                           preproc=preproc,
                           parallel_postproc=parallel_postproc)

    PETSc.Sys.Print('')
    PETSc.Sys.Print('### === --- Errors --- === ###')

    for it, err in enumerate(errors):
        PETSc.Sys.Print(f'Timestep {it} error: {err/norm0}')

    for err in errors:
        assert err/norm0 < 1e-5


@pytest.mark.parallel(nprocs=4)
def test_steady_swe():
    # test that steady-state is maintained for shallow water eqs
    import utils.units as units
    import utils.planets.earth as earth
    import utils.shallow_water.nonlinear as swe
    import utils.shallow_water.williamson1992.case2 as case2

    # set up the ensemble communicator for space-time parallelism
    nspatial_domains = 2
    ref_level = 2
    degree = 1

    ensemble = fd.Ensemble(fd.COMM_WORLD, nspatial_domains)
    mesh = fd.IcosahedralSphereMesh(radius=earth.radius,
                                    refinement_level=ref_level,
                                    degree=degree,
                                    comm=ensemble.comm)
    x = fd.SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)

    V1 = fd.FunctionSpace(mesh, "BDM", degree+1)
    V2 = fd.FunctionSpace(mesh, "DG", degree)
    W = fd.MixedFunctionSpace((V1, V2))

    # initial conditions
    f = case2.coriolis_expression(*x)

    g = earth.Gravity
    H = case2.H0
    b = fd.Constant(0)

    # W = V1 * V2
    w0 = fd.Function(W)
    un = w0.subfunctions[0]
    hn = w0.subfunctions[1]
    un.project(case2.velocity_expression(*x))
    hn.project(H - b + case2.elevation_expression(*x))

    # finite element forms

    def form_function(u, h, v, q):
        return swe.form_function(mesh, g, b, f, u, h, v, q)

    def form_mass(u, h, v, q):
        return swe.form_mass(mesh, u, h, v, q)

    # Parameters for the diag
    sparameters = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps'}

    solver_parameters_diag = {
        "snes_linesearch_type": "basic",
        'snes_atol': 1e3,
        # 'snes_monitor': None,
        # 'snes_converged_reason': None,
        'ksp_rtol': 1e-3,
        # 'ksp_monitor': None,
        # 'ksp_converged_reason': None,
        'mat_type': 'matfree',
        'ksp_type': 'gmres',
        'pc_type': 'python',
        'pc_python_type': 'asQ.DiagFFTPC',
        'diagfft_state': 'initial',
        'aaos_jacobian_state': 'initial',
    }

    M = [2, 2]
    solver_parameters_diag["diagfft_block_"] = sparameters
    solver_parameters_diag["diagfft_block_0_"] = sparameters

    dt = 0.2*units.hour

    alpha = 1.0e-3
    theta = 0.5

    PD = asQ.paradiag(ensemble=ensemble,
                      form_function=form_function,
                      form_mass=form_mass, w0=w0,
                      dt=dt, theta=theta,
                      alpha=alpha,
                      time_partition=M, solver_parameters=solver_parameters_diag,
                      circ=None, tol=1.0e-6, maxits=None,
                      ctx={}, block_mat_type="aij")
    PD.solve()

    # check against initial conditions
    walls = PD.aaos.w_all.subfunctions
    hn.assign(hn - H + b)

    hmag = fd.norm(hn)
    umag = fd.norm(un)

    for step in range(M[PD.time_rank]):

        up = walls[2*step]
        hp = walls[2*step+1]
        hp.assign(hp-H+b)

        herr = fd.errornorm(hn, hp)/hmag
        uerr = fd.errornorm(un, up)/umag

        htol = pow(10, -ref_level)
        utol = pow(10, -ref_level)

        assert (abs(herr) < htol)
        assert (abs(uerr) < utol)


@pytest.mark.parallel(nprocs=6)
def test_jacobian_heat_equation():
    # tests the basic snes setup
    # using the heat equation
    # solves using unpreconditioned GMRES

    # only one spatial domain
    ensemble = fd.Ensemble(fd.COMM_WORLD, 2)

    mesh = fd.UnitSquareMesh(4, 4, comm=ensemble.comm)
    V = fd.FunctionSpace(mesh, "CG", 1)

    x, y = fd.SpatialCoordinate(mesh)
    u0 = fd.Function(V).interpolate(fd.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.5 ** 2))
    dt = 0.01
    theta = 0.5
    alpha = 0.001
    M = [2, 2, 2]
    solver_parameters = {'ksp_type': 'gmres', 'pc_type': 'none',
                         'ksp_rtol': 1.0e-10, 'ksp_atol': 1.0e-10,
                         'ksp_monitor': None,
                         'ksp_max_it': 45,
                         'ksp_converged_reason': None,
                         # 'snes_test_jacobian': None,
                         # 'snes_test_jacobian_view': None,
                         'snes_converged_reason': None,
                         'snes_max_it': 2,
                         # 'snes_view': None
                         }

    def form_function(u, v):
        return fd.inner(fd.grad(u), fd.grad(v)) * fd.dx

    def form_mass(u, v):
        return u * v * fd.dx

    PD = asQ.paradiag(ensemble=ensemble,
                      form_function=form_function,
                      form_mass=form_mass, w0=u0,
                      dt=dt, theta=theta,
                      alpha=alpha,
                      time_partition=M, solver_parameters=solver_parameters,
                      circ="none",
                      tol=1.0e-6, maxits=None)

    PD.solve()

    assert (1 < PD.snes.getConvergedReason() < 5)


@pytest.mark.parallel(nprocs=6)
def test_set_para_form():
    # checks that the all-at-once system is the same as solving
    # timesteps sequentially using the heat equation as an example by
    # substituting the sequential solution and evaluating the residual

    # only one spatial domain
    ensemble = fd.Ensemble(fd.COMM_WORLD, 2)

    mesh = fd.UnitSquareMesh(4, 4, comm=ensemble.comm)
    V = fd.FunctionSpace(mesh, "CG", 1)

    x, y = fd.SpatialCoordinate(mesh)
    u0 = fd.Function(V).interpolate(fd.exp(-((x-0.5)**2 + (y-0.5)**2)/0.5**2))
    dt = 0.01
    theta = 0.5
    alpha = 0.001
    M = [2, 2, 2]
    solver_parameters = {'ksp_type': 'gmres', 'pc_type': 'none',
                         'ksp_rtol': 1.0e-8, 'ksp_atol': 1.0e-8,
                         'ksp_monitor': None}

    def form_function(u, v):
        return fd.inner(fd.grad(u), fd.grad(v))*fd.dx

    def form_mass(u, v):
        return u*v*fd.dx

    PD = asQ.paradiag(ensemble=ensemble,
                      form_function=form_function,
                      form_mass=form_mass, w0=u0,
                      dt=dt, theta=theta,
                      alpha=alpha,
                      time_partition=M, solver_parameters=solver_parameters,
                      circ="none",
                      tol=1.0e-6, maxits=None,
                      ctx={}, block_mat_type="aij")

    # sequential assembly
    WFull = V * V * V * V * V * V * V * V
    ufull = fd.Function(WFull)
    np.random.seed(132574)
    ufull_list = ufull.subfunctions
    for i in range(8):
        ufull_list[i].dat.data[:] = np.random.randn(*(ufull_list[i].dat.data.shape))

    rT = ensemble.ensemble_comm.rank
    # copy the data from the full list into the time slice for this rank in PD.w_all
    w_alls = PD.aaos.w_all.subfunctions
    w_alls[0].assign(ufull_list[rT*2])
    w_alls[1].assign(ufull_list[rT*2+1])
    # copy from w_all into the PETSc vec PD.X
    with PD.aaos.w_all.dat.vec_ro as v:
        v.copy(PD.X)

    # make a form for all of the time slices
    vfull = fd.TestFunction(WFull)
    ufulls = fd.split(ufull)
    vfulls = fd.split(vfull)
    for i in range(8):
        if i == 0:
            un = u0
        else:
            un = ufulls[i-1]
        unp1 = ufulls[i]
        v = vfulls[i]
        tform = form_mass(unp1 - un, v/dt) + form_function((unp1+un)/2, v)
        if i == 0:
            fullform = tform
        else:
            fullform += tform

    Ffull = fd.assemble(fullform)

    PD.aaos._assemble_function(PD.snes, PD.X, PD.F)
    PD_Ff = fd.Function(PD.aaos.function_space_all)

    with PD_Ff.dat.vec_wo as v:
        v.array[:] = PD.F.array_r

    assert (fd.errornorm(Ffull.sub(rT * 2), PD_Ff.sub(0)) < 1.0e-12)
    assert (fd.errornorm(Ffull.sub(rT * 2 + 1), PD_Ff.sub(1)) < 1.0e-12)


@pytest.mark.parallel(nprocs=6)
def test_set_para_form_mixed_parallel():
    # checks that the all-at-once system is the same as solving
    # timesteps sequentially using the heat equation as an example by
    # substituting the sequential solution and evaluating the residual

    # only one spatial domain
    ensemble = fd.Ensemble(fd.COMM_WORLD, 2)

    mesh = fd.UnitSquareMesh(4, 4, comm=ensemble.comm)
    V = fd.FunctionSpace(mesh, "BDM", 1)
    Q = fd.FunctionSpace(mesh, "DG", 0)
    W = V * Q

    x, y = fd.SpatialCoordinate(mesh)
    # u0 = fd.Function(V).interpolate(fd.exp(-((x-0.5)**2 + (y-0.5)**2)/0.5**2))
    w0 = fd.Function(W)
    u0 = w0.subfunctions[0]
    p0 = w0.subfunctions[1]
    p0.interpolate(fd.exp(-((x-0.5)**2 + (y-0.5)**2)/0.5**2))
    dt = 0.01
    theta = 0.5
    alpha = 0.001
    M = [2, 2, 2]
    solver_parameters = {'ksp_type': 'gmres', 'pc_type': 'none',
                         'ksp_rtol': 1.0e-8, 'ksp_atol': 1.0e-8,
                         'ksp_monitor': None}

    def form_function(uu, up, vu, vp):
        return (fd.div(vu)*up - fd.div(uu)*vp)*fd.dx

    def form_mass(uu, up, vu, vp):
        return (fd.inner(uu, vu) + up*vp)*fd.dx

    PD = asQ.paradiag(ensemble=ensemble,
                      form_function=form_function,
                      form_mass=form_mass, w0=w0,
                      dt=dt, theta=theta,
                      alpha=alpha,
                      time_partition=M, solver_parameters=solver_parameters,
                      circ="none",
                      tol=1.0e-6, maxits=None,
                      ctx={}, block_mat_type="aij")

    # sequential assembly
    WFull = W * W * W * W * W * W * W * W
    ufull = fd.Function(WFull)
    np.random.seed(132574)
    ufull_list = ufull.subfunctions
    for i in range((2*8)):
        ufull_list[i].dat.data[:] = np.random.randn(*(ufull_list[i].dat.data.shape))

    rT = ensemble.ensemble_comm.rank
    # copy the data from the full list into the time slice for this rank in PD.w_all
    w_alls = PD.aaos.w_all.subfunctions
    w_alls[0].assign(ufull_list[4 * rT])   # 1st time slice V
    w_alls[1].assign(ufull_list[4 * rT + 1])  # 1st time slice Q
    w_alls[2].assign(ufull_list[4 * rT + 2])  # 2nd time slice V
    w_alls[3].assign(ufull_list[4 * rT + 3])  # 2nd time slice Q

    # copy from w_all into the PETSc vec PD.X
    with PD.aaos.w_all.dat.vec_ro as v:
        v.copy(PD.X)

    # make a form for all of the time slices
    vfull = fd.TestFunction(WFull)
    ufulls = fd.split(ufull)
    vfulls = fd.split(vfull)

    for i in range(8):
        if i == 0:
            un = u0
            pn = p0
        else:
            un = ufulls[2 * i - 2]
            pn = ufulls[2 * i - 1]
        unp1 = ufulls[2 * i]
        pnp1 = ufulls[2 * i + 1]
        vu = vfulls[2 * i]
        vp = vfulls[2 * i + 1]
        # forms have 2 components and 2 test functions: (u, h, w, phi)
        tform = form_mass(unp1 - un, pnp1 - pn, vu / dt, vp / dt) \
            + form_function((unp1 + un) / 2, (pnp1 + pn) / 2, vu, vp)
        if i == 0:
            fullform = tform
        else:
            fullform += tform

    Ffull = fd.assemble(fullform)

    PD.aaos._assemble_function(PD.snes, PD.X, PD.F)
    PD_F = fd.Function(PD.aaos.function_space_all)

    with PD_F.dat.vec_wo as v:
        v.array[:] = PD.F.array_r

    assert (fd.errornorm(Ffull.sub(rT*4), PD_F.sub(0)) < 1.0e-12)
    assert (fd.errornorm(Ffull.sub(rT*4+1), PD_F.sub(1)) < 1.0e-12)
    assert (fd.errornorm(Ffull.sub(rT*4+2), PD_F.sub(2)) < 1.0e-12)
    assert (fd.errornorm(Ffull.sub(rT*4+3), PD_F.sub(3)) < 1.0e-12)


@pytest.mark.parallel(nprocs=6)
def test_jacobian_mixed_parallel():
    ensemble = fd.Ensemble(fd.COMM_WORLD, 2)

    mesh = fd.UnitSquareMesh(6, 6, comm=ensemble.comm)
    V = fd.FunctionSpace(mesh, "BDM", 1)
    Q = fd.FunctionSpace(mesh, "DG", 0)
    W = V * Q

    x, y = fd.SpatialCoordinate(mesh)
    w0 = fd.Function(W)
    u0 = w0.subfunctions[0]
    p0 = w0.subfunctions[1]
    p0.interpolate(fd.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.5 ** 2))
    dt = 0.01
    theta = 0.5
    alpha = 0.001
    M = [2, 2, 2]
    Ml = np.sum(M)
    c = fd.Constant(0.1)
    eps = fd.Constant(0.001)

    solver_parameters = {'ksp_type': 'gmres', 'pc_type': 'none',
                         'ksp_rtol': 1.0e-8, 'ksp_atol': 1.0e-8,
                         'ksp_monitor': None}

    def form_function(uu, up, vu, vp):
        return (fd.div(vu) * up
                + c * fd.sqrt(fd.inner(uu, uu) + eps) * fd.inner(uu, vu)
                - fd.div(uu) * vp) * fd.dx

    def form_mass(uu, up, vu, vp):
        return (fd.inner(uu, vu) + up * vp) * fd.dx

    PD = asQ.paradiag(ensemble=ensemble,
                      form_function=form_function,
                      form_mass=form_mass, w0=w0,
                      dt=dt, theta=theta,
                      alpha=alpha,
                      time_partition=M, solver_parameters=solver_parameters,
                      circ="none",
                      tol=1.0e-6, maxits=None,
                      ctx={}, block_mat_type="aij")

    # sequential assembly
    WFull = reduce(mul, (W for _ in range(Ml)))
    ufull = fd.Function(WFull)
    np.random.seed(132574)
    ufull_list = ufull.subfunctions
    for i in range((2 * Ml)):
        ufull_list[i].dat.data[:] = np.random.randn(*(ufull_list[i].dat.data.shape))

    # make another function v_alls:
    vfull = fd.Function(WFull)
    vfull_list = vfull.subfunctions
    for i in range((2 * Ml)):
        vfull_list[i].dat.data[:] = np.random.randn(*(vfull_list[i].dat.data.shape))

    rT = ensemble.ensemble_comm.rank

    # copy the data from the full list into the time slice for this rank in PD.w_all
    w_alls = PD.aaos.w_all.subfunctions
    v_all = fd.Function(PD.aaos.function_space_all)
    v_alls = v_all.subfunctions

    nM = M[rT]
    for i in range(nM):
        # sum over the entries of M until rT determines left position left
        left = np.sum(M[:rT], dtype=int)
        ind1 = 2*left + 2*i
        ind2 = 2*left + 2*i + 1
        w_alls[2*i].assign(ufull_list[ind1])  # ith time slice V
        w_alls[2*i + 1].assign(ufull_list[ind2])  # ith time slice Q
        v_alls[2*i].assign(vfull_list[ind1])  # ith time slice V
        v_alls[2*i + 1].assign(vfull_list[ind2])  # ith time slice Q

    # Parallel PARADIAG: calculate Jac1 with PD
    # copy from w_all into the PETSc vec PD.X
    with PD.aaos.w_all.dat.vec_ro as v:
        v.copy(PD.X)
    PD.aaos.update(PD.X)

    # use PD to calculate the Jacobian
    Jac1 = PD.jacobian
    Jac1.update()  # updates Jacobian state from aaos

    # construct Petsc vector X1, Y1:
    nlocal = M[rT]*W.node_set.size  # local times x local space
    nglobal = np.prod(M)*W.dim()  # global times x global space
    X1 = PETSc.Vec().create(comm=fd.COMM_WORLD)
    X1.setSizes((nlocal, nglobal))
    X1.setFromOptions()
    Y1 = PETSc.Vec().create(comm=fd.COMM_WORLD)
    Y1.setSizes((nlocal, nglobal))
    Y1.setFromOptions()

    # copy from v_all into the PETSc vec X1
    with v_all.dat.vec_ro as v:
        v.copy(X1)

    # do the matrix multiplication of Jac1 with X1, write output into Y1
    Jac1.mult(None, X1, Y1)

    # SERIAL: make a form for all of the time slices
    tfull = fd.TestFunction(WFull)
    ufulls = fd.split(ufull)
    tfulls = fd.split(tfull)

    for i in range(Ml):
        if i == 0:
            un = u0
            pn = p0
        else:
            un, pn = ufulls[2*i - 2: 2*i]
        unp1, pnp1 = ufulls[2*i: 2*i + 2]
        vu, vp = tfulls[2*i: 2*i + 2]

        tform = form_mass(unp1 - un, pnp1 - pn, vu / dt, vp / dt) \
            + 0.5*form_function(unp1, pnp1, vu, vp) \
            + 0.5*form_function(un, pn, vu, vp)
        if i == 0:
            fullform = tform
        else:
            fullform += tform

    # calculate derivative of Jac2 directly from serial fullform wrt ufull:
    Jac2 = fd.derivative(fullform, ufull)
    # do the matrix multiplication with vfull:
    jacout = fd.assemble(fd.action(Jac2, vfull))

    # generalization of the error evaluation to nM time slices
    PD_J = fd.Function(PD.aaos.function_space_all)
    with PD_J.dat.vec_wo as v:
        v.array[:] = Y1.array_r

    for i in range(nM):
        left = np.sum(M[:rT], dtype=int)
        ind1 = 2*left + 2*i
        ind2 = 2*left + 2*i + 1
        assert (fd.errornorm(jacout.sub(ind1), PD_J.sub(2*i)) < 1.0e-11)
        assert (fd.errornorm(jacout.sub(ind2), PD_J.sub(2*i+1)) < 1.0e-11)


bc_opts = ["no_bcs", "homogeneous_bcs", "inhomogeneous_bcs"]

extruded_mixed = [pytest.param(False, id="standard_mesh"),
                  pytest.param(True, id="extruded_mesh",
                               marks=pytest.mark.xfail(reason="fd.split for TensorProductElements in unmixed spaces broken by ufl PR#122."))]


@pytest.mark.parallel(nprocs=6)
@pytest.mark.parametrize("bc_opt", bc_opts)
@pytest.mark.parametrize("extruded", extruded_mixed)
def test_solve_para_form(bc_opt, extruded):
    # checks that the all-at-once system is the same as solving
    # timesteps sequentially using the NONLINEAR heat equation as an example by
    # solving the all-at-once system and comparing with the sequential
    # solution

    # set up the ensemble communicator for space-time parallelism
    nspatial_domains = 2
    ensemble = fd.Ensemble(fd.COMM_WORLD, nspatial_domains)

    if extruded:
        mesh1D = fd.UnitIntervalMesh(4, comm=ensemble.comm)
        mesh = fd.ExtrudedMesh(mesh1D, 4, layer_height=0.25)
    else:
        mesh = fd.UnitSquareMesh(4, 4, quadrilateral=True, comm=ensemble.comm)

    V = fd.FunctionSpace(mesh, "CG", 1)

    x, y = fd.SpatialCoordinate(mesh)
    u0 = fd.Function(V).interpolate(fd.exp(-((x-0.5)**2 + (y-0.5)**2)/0.5**2))
    dt = 0.01
    theta = 0.5
    alpha = 0.001
    c = fd.Constant(1)
    M = [2, 2, 2]
    Ml = np.sum(M)

    # Parameters for the diag
    sparameters = {
        "ksp_type": "preonly",
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps'
    }

    solver_parameters_diag = {
        "snes_linesearch_type": "basic",
        'snes_monitor': None,
        'snes_stol': 1.0e-100,
        'snes_converged_reason': None,
        'mat_type': 'matfree',
        'ksp_type': 'gmres',
        'ksp_monitor': None,
        'pc_type': 'python',
        'pc_python_type': 'asQ.DiagFFTPC',
    }

    for i in range(sum(M)):
        solver_parameters_diag[f"diagfft_block_{i}_"] = sparameters

    def form_function(u, v):
        return fd.inner((1.+c*fd.inner(u, u))*fd.grad(u), fd.grad(v))*fd.dx

    def form_mass(u, v):
        return u*v*fd.dx

    if bc_opt == "inhomogeneous_bcs":
        bcs = [fd.DirichletBC(V, fd.sin(2*fd.pi*x), "on_boundary")]
    elif bc_opt == "homogeneous_bcs":
        bcs = [fd.DirichletBC(V, 0., "on_boundary")]
    else:
        bcs = []

    PD = asQ.paradiag(ensemble=ensemble,
                      form_function=form_function,
                      form_mass=form_mass, w0=u0,
                      dt=dt, theta=theta,
                      alpha=alpha,
                      time_partition=M, bcs=bcs,
                      solver_parameters=solver_parameters_diag,
                      circ="quasi", tol=1.0e-6, maxits=None,
                      ctx={}, block_mat_type="aij")
    PD.solve()

    # sequential solver
    un = fd.Function(V)
    unp1 = fd.Function(V)

    un.assign(u0)
    v = fd.TestFunction(V)

    eqn = (unp1 - un)*v*fd.dx
    eqn += fd.Constant(dt*(1-theta))*form_function(un, v)
    eqn += fd.Constant(dt*theta)*form_function(unp1, v)

    sprob = fd.NonlinearVariationalProblem(eqn, unp1, bcs=bcs)
    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu'}
    ssolver = fd.NonlinearVariationalSolver(sprob,
                                            solver_parameters=solver_parameters)

    # Calculation of time slices in serial:
    VFull = reduce(mul, (V for _ in range(Ml)))
    vfull = fd.Function(VFull)
    vfull_list = vfull.subfunctions
    rT = ensemble.ensemble_comm.rank

    for i in range(Ml):
        ssolver.solve()
        vfull_list[i].assign(unp1)
        un.assign(unp1)

    nM = M[rT]
    for i in range(nM):
        # sum over the entries of M until rT determines left position left
        left = np.sum(M[:rT], dtype=int)
        ind1 = left + i
        assert (fd.errornorm(vfull.sub(ind1), PD.aaos.w_all.sub(i)) < 1.0e-9)


extruded_primal = [pytest.param(False, id="standard_mesh"),
                   pytest.param(True, id="extruded_mesh")]


@pytest.mark.parallel(nprocs=6)
@pytest.mark.parametrize("extruded", extruded_primal)
def test_solve_para_form_mixed(extruded):
    # checks that the all-at-once system is the same as solving
    # timesteps sequentially using the NONLINEAR mixed wave equation as an
    # example by substituting the sequential solution and evaluating
    # the residual

    # set up the ensemble communicator for space-time parallelism
    nspatial_domains = 2
    ensemble = fd.Ensemble(fd.COMM_WORLD, nspatial_domains)

    if extruded:
        mesh1D = fd.UnitIntervalMesh(4, comm=ensemble.comm)
        mesh = fd.ExtrudedMesh(mesh1D, 4, layer_height=0.25)

        horizontal_degree = 1
        vertical_degree = 1
        S1 = fd.FiniteElement("CG", fd.interval, horizontal_degree+1)
        S2 = fd.FiniteElement("DG", fd.interval, horizontal_degree)

        # vertical base spaces
        T0 = fd.FiniteElement("CG", fd.interval, vertical_degree+1)
        T1 = fd.FiniteElement("DG", fd.interval, vertical_degree)

        # build spaces V2, V3
        V2h_elt = fd.HDiv(fd.TensorProductElement(S1, T1))
        V2t_elt = fd.TensorProductElement(S2, T0)
        V3_elt = fd.TensorProductElement(S2, T1)
        V2v_elt = fd.HDiv(V2t_elt)
        V2_elt = V2h_elt + V2v_elt

        V = fd.FunctionSpace(mesh, V2_elt, name="HDiv")
        Q = fd.FunctionSpace(mesh, V3_elt, name="DG")
    else:
        mesh = fd.UnitSquareMesh(4, 4, comm=ensemble.comm)
        V = fd.FunctionSpace(mesh, "BDM", 1)
        Q = fd.FunctionSpace(mesh, "DG", 0)

    # mesh = fd.PeriodicUnitSquareMesh(4, 4, comm=ensemble.comm)

    W = V * Q

    x, y = fd.SpatialCoordinate(mesh)
    w0 = fd.Function(W)
    u0, p0 = w0.subfunctions
    p0.interpolate(fd.exp(-((x-0.5)**2 + (y-0.5)**2)/0.5**2))
    dt = 0.01
    theta = 0.5
    alpha = 0.001
    c = fd.Constant(10)
    eps = fd.Constant(0.001)

    M = [2, 2, 2]
    Ml = np.sum(M)

    # Parameters for the diag
    sparameters = {
        "ksp_type": "preonly",
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps'
    }

    solver_parameters_diag = {
        "snes_linesearch_type": "basic",
        'snes_monitor': None,
        'snes_converged_reason': None,
        'mat_type': 'matfree',
        'ksp_type': 'gmres',
        'ksp_monitor': None,
        'pc_type': 'python',
        'pc_python_type': 'asQ.DiagFFTPC',
    }

    solver_parameters_diag["diagfft_block_"] = sparameters

    def form_function(uu, up, vu, vp):
        return (fd.div(vu) * up + c * fd.sqrt(fd.inner(uu, uu) + eps) * fd.inner(uu, vu)
                - fd.div(uu) * vp) * fd.dx

    def form_mass(uu, up, vu, vp):
        return (fd.inner(uu, vu) + up * vp) * fd.dx

    PD = asQ.paradiag(ensemble=ensemble,
                      form_function=form_function,
                      form_mass=form_mass, w0=w0, dt=dt,
                      theta=theta, alpha=alpha, time_partition=M,
                      solver_parameters=solver_parameters_diag,
                      circ="quasi",
                      tol=1.0e-6, maxits=None,
                      ctx={}, block_mat_type="aij")
    PD.solve()

    # sequential solver
    un = fd.Function(W)
    unp1 = fd.Function(W)

    un.assign(w0)
    v = fd.TestFunction(W)

    eqn = form_mass(*(fd.split(unp1)), *(fd.split(v)))
    eqn -= form_mass(*(fd.split(un)), *(fd.split(v)))
    eqn += fd.Constant(dt*(1-theta))*form_function(*(fd.split(un)),
                                                   *(fd.split(v)))
    eqn += fd.Constant(dt*theta)*form_function(*(fd.split(unp1)),
                                               *(fd.split(v)))

    sprob = fd.NonlinearVariationalProblem(eqn, unp1)
    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu',
                         'pc_factor_mat_solver_type': 'mumps',
                         'mat_type': 'aij'}
    ssolver = fd.NonlinearVariationalSolver(sprob,
                                            solver_parameters=solver_parameters)
    ssolver.solve()

    # Calculation of time slices in serial:
    VFull = reduce(mul, (W for _ in range(Ml)))
    vfull = fd.Function(VFull)
    vfull_list = vfull.subfunctions

    rT = ensemble.ensemble_comm.rank

    for i in range(Ml):
        ssolver.solve()
        for k in range(2):
            vfull.sub(2*i+k).assign(unp1.sub(k))
        un.assign(unp1)

    nM = M[rT]
    for i in range(nM):
        # sum over the entries of M until rT determines left position left
        left = np.sum(M[:rT], dtype=int)
        ind1 = 2*left + 2*i
        ind2 = 2*left + 2*i + 1
        assert (fd.errornorm(vfull_list[ind1], PD.aaos.w_all.sub(2*i)) < 1.0e-9)
        assert (fd.errornorm(vfull_list[ind2], PD.aaos.w_all.sub(2*i+1)) < 1.0e-9)


@pytest.mark.parallel(nprocs=6)
def test_diagnostics():
    # tests that the diagnostics recording is accurate
    ensemble = fd.Ensemble(fd.COMM_WORLD, 2)

    mesh = fd.UnitSquareMesh(4, 4, comm=ensemble.comm)
    V = fd.FunctionSpace(mesh, "CG", 1)

    x, y = fd.SpatialCoordinate(mesh)
    u0 = fd.Function(V).interpolate(fd.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.5 ** 2))
    dt = 0.01
    theta = 0.5
    alpha = 0.001
    M = [2, 2, 2]

    block_sparameters = {
        "ksp_type": "preonly",
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps'
    }

    diag_sparameters = {
        'snes_converged_reason': None,
        'ksp_converged_reason': None,
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'pc_python_type': 'asQ.DiagFFTPC'
    }

    for i in range(np.sum(M)):
        diag_sparameters["diagfft_block_" + str(i) + "_"] = block_sparameters

    def form_function(u, v):
        return fd.inner(fd.grad(u), fd.grad(v))*fd.dx

    def form_mass(u, v):
        return u*v*fd.dx

    pdg = asQ.paradiag(ensemble=ensemble,
                       form_function=form_function,
                       form_mass=form_mass, w0=u0,
                       dt=dt, theta=theta,
                       alpha=alpha,
                       time_partition=M,
                       solver_parameters=diag_sparameters,
                       circ=None)

    pdg.solve(nwindows=1)

    pdg.sync_diagnostics()

    assert pdg.total_timesteps == sum(M)
    assert pdg.total_windows == 1
    assert pdg.linear_iterations == pdg.snes.getLinearSolveIterations()
    assert pdg.nonlinear_iterations == pdg.snes.getIterationNumber()

    # direct block solve
    for i in range(sum(M)):
        assert pdg.block_iterations.dglobal[i] == pdg.linear_iterations

    linear_iterations0 = pdg.linear_iterations
    nonlinear_iterations0 = pdg.nonlinear_iterations

    pdg.solve(nwindows=1)

    assert pdg.total_timesteps == 2*sum(M)
    assert pdg.total_windows == 2
    assert pdg.linear_iterations == linear_iterations0 + pdg.snes.getLinearSolveIterations()
    assert pdg.nonlinear_iterations == nonlinear_iterations0 + pdg.snes.getIterationNumber()

    for i in range(sum(M)):
        assert pdg.block_iterations.dglobal[i] == pdg.linear_iterations
