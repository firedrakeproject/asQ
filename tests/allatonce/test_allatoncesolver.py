import asQ
import firedrake as fd
import pytest


@pytest.mark.parallel(nprocs=6)
def test_solve_heat_equation_nopc():
    """
    Tests the basic solver setup using the heat equation.
    Solves using unpreconditioned GMRES and checks that the
    residual of the all-at-once-form is below tolerance.
    """

    # set up space-time parallelism

    nspace_ranks = 2
    nslices = fd.COMM_WORLD.size//nspace_ranks
    slice_length = 2

    time_partition = tuple((slice_length for _ in range(nslices)))
    ensemble = asQ.create_ensemble(time_partition, comm=fd.COMM_WORLD)

    mesh = fd.UnitSquareMesh(
        6, 6, comm=ensemble.comm,
        distribution_parameters={'partitioner_type': 'simple'})

    V = fd.FunctionSpace(mesh, "CG", 1)

    # all-at-once function and initial conditions

    x, y = fd.SpatialCoordinate(mesh)
    ics = fd.Function(V).interpolate(fd.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.5**2))

    aaofunc = asQ.AllAtOnceFunction(ensemble, time_partition, V)
    aaofunc.assign(ics)

    # all-at-once form

    dt = 0.01
    theta = 1.0

    def form_function(u, v, t):
        return fd.inner(fd.grad(u), fd.grad(v))*fd.dx

    def form_mass(u, v):
        return fd.inner(u, v)*fd.dx

    aaoform = asQ.AllAtOnceForm(aaofunc, dt, theta,
                                form_mass, form_function)

    # solver and options

    atol = 1e-10
    solver_parameters = {
        'snes_type': 'ksponly',
        'pc_type': 'none',
        'ksp_type': 'gmres',
        'mat_type': 'matfree',
        'ksp': {
            'converged_rate': None,
            'atol': atol,
            'rtol': 0,
            'stol': 0,
        }
    }

    aaosolver = asQ.AllAtOnceSolver(aaoform, aaofunc,
                                    solver_parameters=solver_parameters)

    aaosolver.solve()

    # check residual
    aaoform.assemble(func=aaofunc)

    residual = aaoform.F.cofunction.riesz_representation(
        'l2', solver_options={'function_space': aaofunc.function.function_space()})

    assert fd.norm(residual) < atol, "GMRES should converge to prescribed tolerance even without preconditioning"


def test_solve_heat_equation_circulantpc():
    """
    Tests the basic solver setup using the heat equation.
    Solves using GMRES preconditioned with the CirculantPC
    and checks that the residual of the all-at-once-form
    is below tolerance.
    """

    # set up space-time parallelism

    window_length = 4

    time_partition = window_length
    ensemble = asQ.create_ensemble(time_partition, comm=fd.COMM_WORLD)

    mesh = fd.UnitSquareMesh(
        6, 6, comm=ensemble.comm,
        distribution_parameters={'partitioner_type': 'simple'})

    V = fd.FunctionSpace(mesh, "CG", 1)

    # all-at-once function and initial conditions

    x, y = fd.SpatialCoordinate(mesh)
    ics = fd.Function(V).interpolate(fd.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.5**2))

    aaofunc = asQ.AllAtOnceFunction(ensemble, time_partition, V)
    aaofunc.assign(ics)

    # all-at-once form

    dt = 0.01
    theta = 1.0

    def form_function(u, v, t):
        return fd.inner(fd.grad(u), fd.grad(v))*fd.dx

    def form_mass(u, v):
        return fd.inner(u, v)*fd.dx

    aaoform = asQ.AllAtOnceForm(aaofunc, dt, theta,
                                form_mass, form_function)

    # solver and options

    atol = 1.0e-8
    solver_parameters = {
        'snes_type': 'ksponly',
        'ksp_type': 'gmres',
        'mat_type': 'matfree',
        'ksp': {
            'monitor': None,
            'converged_rate': None,
            'atol': atol,
            'rtol': 0,
            'stol': 0,
        },
        'pc_type': 'python',
        'pc_python_type': 'asQ.CirculantPC',
    }

    aaosolver = asQ.AllAtOnceSolver(aaoform, aaofunc,
                                    solver_parameters=solver_parameters)

    aaosolver.solve()

    # check residual
    aaoform.assemble(func=aaofunc)

    residual = aaoform.F.cofunction.riesz_representation(
        'l2', solver_options={'function_space': aaofunc.function.function_space()})

    assert fd.norm(residual) < atol, "GMRES should converge to prescribed tolerance with CirculantPC"


extruded = [pytest.param(False, id="standard_mesh"),
            pytest.param(True, id="extruded_mesh")]

cpx_types = [pytest.param('vector', id="vector_cpx"),
             pytest.param('mixed', id="mixed_cpx")]


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize("extrude", extruded)
@pytest.mark.parametrize("cpx_type", cpx_types)
def test_solve_mixed_wave_equation(extrude, cpx_type):
    """
    Tests the solver setup using a nonlinear wave equation.
    Solves using GMRES preconditioned with CirculantPC and checks
    that the residual of the all-at-once-form is below tolerance.
    """

    # space-time parallelism
    nspace_ranks = 2
    nslices = fd.COMM_WORLD.size//nspace_ranks
    slice_length = 2

    time_partition = tuple((slice_length for _ in range(nslices)))
    ensemble = asQ.create_ensemble(time_partition, comm=fd.COMM_WORLD)

    # mesh and function spaces
    nx = 6
    if extrude:
        mesh1D = fd.UnitIntervalMesh(
            nx, comm=ensemble.comm,
            distribution_parameters={'partitioner_type': 'simple'})
        mesh = fd.ExtrudedMesh(mesh1D, nx, layer_height=1./nx)

        horizontal_degree = 0
        vertical_degree = 0
        S1 = fd.FiniteElement("CG", fd.interval, horizontal_degree+1)
        S2 = fd.FiniteElement("DG", fd.interval, horizontal_degree)

        # vertical base spaces
        T0 = fd.FiniteElement("CG", fd.interval, vertical_degree+1)
        T1 = fd.FiniteElement("DG", fd.interval, vertical_degree)

        # build spaces V2, V3
        V2h_elt = fd.HDiv(fd.TensorProductElement(S1, T1))
        V2v_elt = fd.HDiv(fd.TensorProductElement(S2, T0))
        V3_elt = fd.TensorProductElement(S2, T1)
        V2_elt = V2h_elt + V2v_elt

        V = fd.FunctionSpace(mesh, V2_elt, name="HDiv")
        Q = fd.FunctionSpace(mesh, V3_elt, name="DG")
    else:
        mesh = fd.UnitSquareMesh(
            nx, nx, comm=ensemble.comm,
            distribution_parameters={'partitioner_type': 'simple'})

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

    def form_function(uu, up, vu, vp, t):
        return (fd.div(vu) * up + c * fd.sqrt(fd.inner(uu, uu) + eps) * fd.inner(uu, vu)
                - fd.div(uu) * vp) * fd.dx(degree=4)

    def form_mass(uu, up, vu, vp):
        return (fd.inner(uu, vu) + up * vp) * fd.dx

    aaoform = asQ.AllAtOnceForm(aaofunc, dt, theta,
                                form_mass, form_function)

    # solver and options

    atol = 1e-8
    solver_parameters = {
        'snes': {
            'linesearch_type': 'basic',
            'monitor': None,
            'converged_reason': None,
            'atol': atol,
            'rtol': 0,
            'stol': 0,
        },
        'mat_type': 'matfree',
        'ksp_type': 'preonly',
        'ksp': {
            'monitor': None,
            'converged_rate': None,
        },
        'pc_type': 'python',
        'pc_python_type': 'asQ.CirculantPC',
        'circulant_alpha': 1e-3,
        'circulant_block': {
            'ksp_type': "preonly",
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps'
        },
        'circulant_complex_proxy': cpx_type
    }

    aaosolver = asQ.AllAtOnceSolver(aaoform, aaofunc,
                                    solver_parameters=solver_parameters)

    aaosolver.solve()

    # check residual
    residual = aaoform.F.cofunction.riesz_representation(
        'l2', solver_options={'function_space': aaofunc.function.function_space()})

    assert fd.norm(residual) < atol, "GMRES should converge to prescribed tolerance with CirculantPC"
