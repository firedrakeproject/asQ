import asQ
import firedrake as fd
import pytest


@pytest.mark.parallel(nprocs=6)
def test_solve_heat_equation():
    """
    Tests the basic solver setup using the heat equation.
    Solves using unpreconditioned GMRES and checks that the
    residual of the all-at-once-form is below tolerance.
    """

    # set up space-time parallelism

    nslices = fd.COMM_WORLD.size//2
    slice_length = 2

    time_partition = tuple((slice_length for _ in range(nslices)))
    ensemble = asQ.create_ensemble(time_partition, comm=fd.COMM_WORLD)

    mesh = fd.UnitSquareMesh(6, 6, comm=ensemble.comm)
    V = fd.FunctionSpace(mesh, "CG", 1)

    # all-at-once function and initial conditions

    x, y = fd.SpatialCoordinate(mesh)
    ics = fd.Function(V).interpolate(fd.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.5**2))

    aaofunc = asQ.AllAtOnceFunction(ensemble, time_partition, V)
    aaofunc.assign(ics)

    # all-at-once form

    dt = 0.01
    theta = 1.0

    def form_function(u, v):
        return fd.inner(fd.grad(u), fd.grad(v))*fd.dx

    def form_mass(u, v):
        return fd.inner(u, v)*fd.dx

    aaoform = asQ.AllAtOnceForm(aaofunc, dt, theta,
                                form_mass, form_function)

    # solver and options

    atol = 1.0e-10
    solver_parameters = {
        'snes_type': 'ksponly',
        'snes': {
            'monitor': None,
            'converged_reason': None,
        },
        'pc_type': 'none',
        'ksp_type': 'gmres',
        'mat_type': 'matfree',
        'ksp': {
            'monitor': None,
            'converged_reason': None,
            'atol': atol,
            'rtol': 1.0e-100,
            'stol': 1.0e-100,
        }
    }

    aaosolver = asQ.AllAtOnceSolver(aaoform, aaofunc,
                                    solver_parameters=solver_parameters)

    aaosolver.solve()

    # check residual

    aaoform.assemble(func=aaofunc)
    residual = fd.norm(aaoform.F.function)

    assert residual < atol


extruded = [pytest.param(False, id="standard_mesh"),
            pytest.param(True, id="extruded_mesh")]


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize("extrude", extruded)
def test_solve_mixed_wave_equation(extrude):
    """
    Tests the solver setup using a nonlinear wave equation.
    Solves using GMRES preconditioned with the circulant matrix and
    checks that the residual of the all-at-once-form is below tolerance.
    """

    # space-time parallelism
    nslices = fd.COMM_WORLD.size//2
    slice_length = 2

    time_partition = tuple((slice_length for _ in range(nslices)))
    ensemble = asQ.create_ensemble(time_partition, comm=fd.COMM_WORLD)

    # mesh and function spaces
    if extrude:
        mesh1D = fd.UnitIntervalMesh(6, comm=ensemble.comm)
        mesh = fd.ExtrudedMesh(mesh1D, 6, layer_height=0.25)

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
        mesh = fd.UnitSquareMesh(6, 6, comm=ensemble.comm)
        V = fd.FunctionSpace(mesh, "BDM", 1)
        Q = fd.FunctionSpace(mesh, "DG", 0)

    W = V * Q

    # all-at-once function and ics
    x, y = fd.SpatialCoordinate(mesh)
    ics = fd.Function(W)
    u0, p0 = ics.subfunctions
    p0.interpolate(fd.exp(-((x-0.5)**2 + (y-0.5)**2)/0.5**2))

    aaofunc = asQ.AllAtOnceFunction(ensemble, time_partition, W)
    aaofunc.assign(ics)

    # all-at-once form

    dt = 0.01
    theta = 0.5
    c = fd.Constant(10)
    eps = fd.Constant(0.001)

    def form_function(uu, up, vu, vp):
        return (fd.div(vu) * up + c * fd.sqrt(fd.inner(uu, uu) + eps) * fd.inner(uu, vu)
                - fd.div(uu) * vp) * fd.dx

    def form_mass(uu, up, vu, vp):
        return (fd.inner(uu, vu) + up * vp) * fd.dx

    aaoform = asQ.AllAtOnceForm(aaofunc, dt, theta,
                                form_mass, form_function)

    # solver and options

    block_parameters = {
        "ksp_type": "preonly",
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps'
    }

    atol = 1e-8
    solver_parameters = {
        'snes': {
            'linesearch_type': 'basic',
            'monitor': None,
            'converged_reason': None,
            'atol': atol,
            'rtol': 1e-100,
            'stol': 1e-100,
        },
        'mat_type': 'matfree',
        'ksp_type': 'gmres',
        'ksp': {
            'monitor': None,
            'converged_reason': None,
        },
        'pc_type': 'python',
        'pc_python_type': 'asQ.DiagFFTPC',
        'diagfft_alpha': 1e-3,
    }

    for i in range(aaofunc.ntimesteps):
        solver_parameters[f"diagfft_block_{i}"] = block_parameters

    aaosolver = asQ.AllAtOnceSolver(aaoform, aaofunc,
                                    solver_parameters=solver_parameters)

    aaosolver.solve()

    # check residual
    aaoform.assemble(func=aaofunc)
    residual = fd.norm(aaoform.F.function)

    assert residual < atol