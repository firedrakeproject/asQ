import asQ
import firedrake as fd
import pytest
from pytest_mpi.parallel_assert import parallel_assert

form_params = [
    pytest.param(None, id="no_form_params"),
    *[pytest.param({"form_construct_type": ftype},
                   id=ftype.replace("-", "_"))
      for ftype in ("monolithic", "stepwise", "single_step")]]


@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize('partition', ['time_serial', 'time_parallel'])
@pytest.mark.parametrize("form_parameters", form_params)
def test_solve_heat_equation_nopc(partition, form_parameters):
    """
    Tests the basic solver setup using the heat equation.
    Solves using unpreconditioned GMRES and checks that the
    residual of the all-at-once-form is below tolerance.
    """

    # set up space-time parallelism

    window_length = 2

    nprocs = fd.COMM_WORLD.size
    assert window_length % nprocs == 0, "test setup incorrectly"

    if partition == 'time_serial':
        nslices = 1
    elif partition == 'time_parallel':
        nslices = nprocs
    else:
        assert False, "Unrecognised partition type"

    slice_length = window_length//nslices
    time_partition = [slice_length for _ in range(nslices)]
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

    aaoform = asQ.AllAtOnceForm(
        aaofunc, dt, theta,
        form_mass, form_function,
        form_parameters=form_parameters)

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
            'converged_rate': None,
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
    with aaoform.F.global_vec_ro() as fvec:
        residual = fvec.norm()

    for n in reversed(range(aaoform.F.nlocal_timesteps)):
        with aaoform.F[n].dat.vec_ro as fvec:
            residual = fvec.norm()
        parallel_assert(
            lambda: residual < atol,
            msg=f"Residual at step {n} is not converged")


extruded = [pytest.param(False, id="standard_mesh"),
            pytest.param(True, id="extruded_mesh")]


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize("extrude", extruded)
@pytest.mark.parametrize("partition", ["time_serial", "time_parallel"])
@pytest.mark.parametrize("form_parameters", form_params)
def test_solve_mixed_wave_equation_nopc(extrude, partition, form_parameters):
    """
    Tests the solver setup using a nonlinear wave equation.
    Solves using GMRES preconditioned with CirculantPC and checks
    that the residual of the all-at-once-form is below tolerance.
    """

    # space-time parallelism
    ntotal_steps = 4
    nslices = 1 if (partition == "time_serial") else 2
    slice_length = ntotal_steps // nslices

    time_partition = tuple((slice_length for _ in range(nslices)))
    ensemble = asQ.create_ensemble(time_partition, comm=fd.COMM_WORLD)
    rank = fd.COMM_WORLD.rank

    # mesh and function spaces
    nx = 8
    if extrude:
        mesh1D = fd.UnitIntervalMesh(
            nx, comm=ensemble.comm,
            distribution_parameters={'partitioner_type': 'simple'})
        mesh = fd.ExtrudedMesh(mesh1D, nx, layer_height=1./nx)

        horizontal_degree = 1
        vertical_degree = 1
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

    def form_function(u, p, v, q, t):
        return (
            (fd.div(v)*p - fd.div(u)*q)*fd.dx
            + c*fd.sqrt(fd.inner(u, u) + eps)*fd.inner(u, v)*fd.dx(degree=4)
        )

    def form_mass(u, p, v, q):
        return (fd.inner(u, v) + p*q) * fd.dx

    aaoform = asQ.AllAtOnceForm(
        aaofunc, dt, theta,
        form_mass, form_function,
        form_parameters=form_parameters)

    # solver and options

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
        'pc_type': 'none',
        'ksp_type': 'gmres',
        'ksp': {
            'converged_rate': None,
            'rtol': 1e-3,
        },
    }

    aaosolver = asQ.AllAtOnceSolver(aaoform, aaofunc,
                                    solver_parameters=solver_parameters)

    print(f"\nBefore solve: {rank = }")
    aaosolver.solve()
    print(f"\nAfter solve: {rank = }")

    # check residual
    aaoform.assemble(func=aaofunc)
    with aaoform.F.global_vec_ro() as fvec:
        residual = fvec.norm()

    parallel_assert(
        lambda: residual < atol,
        msg="GMRES should converge to prescribed tolerance with CirculantPC")


if __name__ == "__main__":
    test_solve_heat_equation_nopc(
        partition="time_parallel",
        form_parameters=None)
