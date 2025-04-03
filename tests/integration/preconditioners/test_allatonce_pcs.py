import firedrake as fd
import asQ
import pytest
from functools import partial
from pytest_mpi.parallel_assert import parallel_assert


def form_mass(u, v):
    return u*v*fd.dx


def variable_form_function(u, v, t, variable_coefficients=False):
    vnu = fd.Constant(0.05)
    one = fd.Constant(1)
    nu = (one + t*vnu) if variable_coefficients else one
    return fd.inner(nu*fd.grad(u), fd.grad(v))*fd.dx


def make_paradiag(time_partition, parameters,
                  variable_coefficients=False,
                  form_parameters=None):
    ensemble = asQ.create_ensemble(time_partition)
    mesh = fd.UnitSquareMesh(
        nx=8, ny=8, comm=ensemble.comm,
        distribution_parameters={'partitioner_type': 'simple'})

    x, y = fd.SpatialCoordinate(mesh)

    V = fd.FunctionSpace(mesh, "CG", 1)
    uinitial = fd.Function(V)
    uinitial.project(fd.sin(x) + fd.cos(y))

    form_function = partial(
        variable_form_function,
        variable_coefficients=variable_coefficients)

    return asQ.Paradiag(
        ensemble=ensemble,
        form_mass=form_mass,
        form_function=form_function,
        ics=uinitial, dt=0.1, theta=0.5,
        time_partition=time_partition,
        solver_parameters=parameters,
        form_parameters=form_parameters)


form_params = [
    pytest.param(None, id="no_form_params"),
    *[pytest.param({"form_construct_type": ftype},
                   id=ftype.replace("-", "_"))
      for ftype in ("monolithic", "stepwise", "single_step")]]

nts = [pytest.param(n, id=f"nt{n}") for n in (4, 8, 16)]

vcoeffs = [
    pytest.param(False, id="constant_coeffs"),
    pytest.param(True, id="variable_coeffs"),
]


@pytest.mark.parallel(nprocs=[1, 4])
@pytest.mark.parametrize("nt", nts)
@pytest.mark.parametrize("variable_coefficients", vcoeffs)
@pytest.mark.parametrize("form_parameters", form_params)
def test_jacobipc(nt, variable_coefficients, form_parameters):
    nslices = fd.COMM_WORLD.size
    slice_length = nt // nslices
    time_partition = [slice_length for _ in range(nslices)]

    solver_parameters = {
        'snes_type': 'ksponly',
        'mat_type': 'matfree',
        'ksp_type': 'richardson',
        'ksp_rtol': 1e-14,
        'pc_type': 'python',
        'pc_python_type': 'asQ.JacobiPC',
        'aaojacobi_block': {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
        },
    }

    paradiag = make_paradiag(
        time_partition, solver_parameters,
        variable_coefficients=variable_coefficients,
        form_parameters=form_parameters)

    paradiag.solve(nwindows=1)

    niterations = paradiag.solver.snes.getLinearSolveIterations()

    parallel_assert(
        lambda: niterations == nt,
        msg=f"JacobiPC should solve exactly after nt iterations, not {niterations}"
    )


@pytest.mark.parallel(nprocs=[1, 4])
@pytest.mark.parametrize("nt", nts)
@pytest.mark.parametrize("variable_coefficients", vcoeffs)
@pytest.mark.parametrize("form_parameters", form_params)
def test_gauss_seidelpc(nt, variable_coefficients, form_parameters):
    nslices = fd.COMM_WORLD.size
    slice_length = nt // nslices
    time_partition = [slice_length for _ in range(nslices)]

    solver_parameters = {
        'snes_type': 'ksponly',
        'mat_type': 'matfree',
        'ksp_type': 'richardson',
        'ksp_rtol': 1e-14,
        'pc_type': 'python',
        'pc_python_type': 'asQ.GaussSeidelPC',
        'aaogs_block': {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
        },
    }

    paradiag = make_paradiag(
        time_partition, solver_parameters,
        variable_coefficients=variable_coefficients,
        form_parameters=form_parameters)

    paradiag.solve(nwindows=1)

    niterations = paradiag.solver.snes.getLinearSolveIterations()

    parallel_assert(
        lambda: niterations == 1,
        msg=f"GaussSeidelPC should solve exactly after 1 iteration, not {niterations}"
    )


@pytest.mark.parallel(nprocs=[1, 4])
@pytest.mark.parametrize("alpha", [1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
def test_circulantpc(alpha):
    nslices = fd.COMM_WORLD.size
    nt = 16
    slice_length = nt // nslices
    time_partition = [slice_length for _ in range(nslices)]

    solver_parameters = {
        'snes_type': 'ksponly',
        'mat_type': 'matfree',
        'ksp_type': 'richardson',
        'ksp_rtol': alpha**3,
        'pc_type': 'python',
        'pc_python_type': 'asQ.CirculantPC',
        'circulant_alpha': alpha,
        'circulant_block': {
            'ksp_type': 'richardson',
            'ksp_rtol': alpha**2,
            'pc_type': 'ilu',
        },
    }

    paradiag = make_paradiag(time_partition, solver_parameters)

    paradiag.solve(nwindows=1)

    niterations = paradiag.solver.snes.getLinearSolveIterations()

    parallel_assert(
        lambda: niterations == 3,
        msg=f"CirculantPC should converge at rate {alpha} in 3 iterations not {niterations}"
    )


nstep1to8 = [pytest.param(n, id=f"interval{n}") for n in (1, 2, 4, 8)]
nstep2to16 = [pytest.param(n, id=f"interval{n}") for n in (2, 4, 8, 16)]


@pytest.mark.parallel(nprocs=8)
@pytest.mark.parametrize("interval_length", nstep1to8)
@pytest.mark.parametrize("variable_coefficients", vcoeffs)
def test_intervaljacobipc_jacobi(interval_length, variable_coefficients):
    slice_length = 1
    time_partition = [slice_length for _ in range(fd.COMM_WORLD.size)]
    ensemble = asQ.create_ensemble(time_partition)

    mesh = fd.UnitSquareMesh(
        nx=16, ny=16, comm=ensemble.comm,
        distribution_parameters={'partitioner_type': 'simple'})

    x, y = fd.SpatialCoordinate(mesh)

    V = fd.FunctionSpace(mesh, "CG", 1)
    uinitial = fd.Function(V)
    uinitial.project(fd.sin(x) + fd.cos(y))

    form_function = partial(
        variable_form_function,
        variable_coefficients=variable_coefficients)

    jacobi_parameters = {
        'ksp_monitor': None,
        'ksp_converged_rate': None,
        'snes_type': 'ksponly',
        'mat_type': 'matfree',
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'pc_python_type': 'asQ.JacobiPC',
    }

    interval_parameters = {
        'ksp_monitor': None,
        'ksp_converged_rate': None,
        'snes_type': 'ksponly',
        'mat_type': 'matfree',
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'pc_python_type': 'asQ.IntervalJacobiPC',
        'pc_ijacobi_interval_length': interval_length,
        'ijacobi': {
            'pc_type': 'python',
            'pc_python_type': 'asQ.JacobiPC',
        }
    }

    paradiag_jacobi = asQ.Paradiag(
        ensemble=ensemble,
        form_mass=form_mass,
        form_function=form_function,
        ics=uinitial, dt=0.1, theta=0.5,
        time_partition=time_partition,
        solver_parameters=jacobi_parameters)

    paradiag_interval = asQ.Paradiag(
        ensemble=ensemble,
        form_mass=form_mass,
        form_function=form_function,
        ics=uinitial, dt=0.1, theta=0.5,
        time_partition=time_partition,
        solver_parameters=interval_parameters)

    paradiag_jacobi.solve(nwindows=1)
    paradiag_interval.solve(nwindows=1)

    jfunc = paradiag_jacobi.aaofunc
    sfunc = paradiag_interval.aaofunc

    with jfunc.global_vec_ro() as jvec, sfunc.global_vec_ro() as svec:
        errvec = jvec - svec
        err = errvec.norm()

    parallel_assert(
        lambda: err < 1e-15,
        msg="IntervalJacobiPC with JacobiPC should be exactly JacobiPC for any interval size"
    )


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize("variable_coefficients", vcoeffs)
def test_intervaljacobipc_circulant(variable_coefficients):
    slice_length = 2
    time_partition = [slice_length for _ in range(4)]
    ensemble = asQ.create_ensemble(time_partition)

    mesh = fd.UnitSquareMesh(
        nx=16, ny=16, comm=ensemble.comm,
        distribution_parameters={'partitioner_type': 'simple'})
    x, y = fd.SpatialCoordinate(mesh)

    V = fd.FunctionSpace(mesh, "CG", 1)
    uinitial = fd.Function(V)
    uinitial.project(fd.sin(x) + fd.cos(y))

    form_function = partial(
        variable_form_function,
        variable_coefficients=variable_coefficients)

    circulant_parameters = {
        'ksp_monitor': None,
        'ksp_converged_rate': None,
        'snes_type': 'ksponly',
        'mat_type': 'matfree',
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'pc_python_type': 'asQ.CirculantPC',
    }

    interval_parameters = {
        'ksp_monitor': None,
        'ksp_converged_rate': None,
        'snes_type': 'ksponly',
        'mat_type': 'matfree',
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'pc_python_type': 'asQ.IntervalJacobiPC',
        'pc_ijacobi_interval_length': sum(time_partition),
        'ijacobi': {
            'pc_type': 'python',
            'pc_python_type': 'asQ.CirculantPC',
        }
    }

    paradiag_circulant = asQ.Paradiag(
        ensemble=ensemble,
        form_mass=form_mass,
        form_function=form_function,
        ics=uinitial, dt=0.1, theta=0.5,
        time_partition=time_partition,
        solver_parameters=circulant_parameters)

    paradiag_interval = asQ.Paradiag(
        ensemble=ensemble,
        form_mass=form_mass,
        form_function=form_function,
        ics=uinitial, dt=0.1, theta=0.5,
        time_partition=time_partition,
        solver_parameters=interval_parameters)

    paradiag_circulant.solve(nwindows=1)
    paradiag_interval.solve(nwindows=1)

    cfunc = paradiag_circulant.aaofunc
    sfunc = paradiag_interval.aaofunc

    with cfunc.global_vec_ro() as cvec, sfunc.global_vec_ro() as svec:
        errvec = cvec - svec
        err = errvec.norm()

    assert err < 1e-15, "IntervalJacobiPC with CirculantPC should be exactly CirculantPC if nstep=nt"


@pytest.mark.parallel(nprocs=8)
@pytest.mark.parametrize("interval_length", nstep2to16)
@pytest.mark.parametrize("variable_coefficients", vcoeffs)
@pytest.mark.parametrize("form_parameters", form_params)
def test_intervaljacobipc(interval_length, variable_coefficients, form_parameters):
    slice_length = 2
    time_partition = [slice_length for _ in range(fd.COMM_WORLD.size)]
    nintervals = sum(time_partition)//interval_length

    solver_parameters = {
        # 'ksp_monitor': ':ksp_monitor.log',
        'ksp_monitor': None,
        'ksp_converged_rate': None,
        'snes_type': 'ksponly',
        'mat_type': 'matfree',
        'ksp_type': 'richardson',
        'ksp_rtol': 1e-14,
        'pc_type': 'python',
        'pc_python_type': 'asQ.IntervalJacobiPC',
        'pc_ijacobi_interval_length': interval_length,
        'ijacobi': {
            'ksp_type': 'preonly',
            'pc_type': 'python',
            'pc_python_type': 'asQ.GaussSeidelPC',
            'aaogs_block': {
                'ksp_type': 'preonly',
                'pc_type': 'lu',
            },
        },
    }

    paradiag = make_paradiag(
        time_partition,
        solver_parameters,
        variable_coefficients,
        form_parameters=form_parameters)

    paradiag.solve(nwindows=1)

    niterations = paradiag.solver.snes.getLinearSolveIterations()
    assert niterations == nintervals, "IntervalJacobiPC with exact interval solves should solve exactly after nt iterations"


@pytest.mark.parallel(nprocs=8)
@pytest.mark.parametrize("interval_length", nstep2to16)
@pytest.mark.parametrize("variable_coefficients", vcoeffs)
@pytest.mark.parametrize("form_parameters", form_params)
def test_intervalgaussseidelpc(interval_length, variable_coefficients, form_parameters):
    slice_length = 2
    time_partition = [slice_length for _ in range(fd.COMM_WORLD.size)]

    solver_parameters = {
        'snes_type': 'ksponly',
        # 'ksp_monitor': None,
        # 'ksp_converged_rate': None,
        'mat_type': 'matfree',
        'ksp_type': 'preonly',
        'ksp_rtol': 1e-14,
        'pc_type': 'python',
        'pc_python_type': 'asQ.IntervalGaussSeidelPC',
        'pc_igs_interval_length': interval_length,
        'igs': {
            'ksp_type': 'richardson',
            'pc_type': 'python',
            'pc_python_type': 'asQ.GaussSeidelPC',
            'aaogs_block': {
                'ksp_type': 'preonly',
                'pc_type': 'lu',
            },
        },
    }

    paradiag = make_paradiag(
        time_partition,
        solver_parameters,
        variable_coefficients,
        form_parameters=form_parameters)

    paradiag.solve(nwindows=1)

    niterations = paradiag.solver.snes.getLinearSolveIterations()
    assert niterations == 1, f"IntervalGaussSeidelPC with exact interval solves should solve exactly after 1 iteration not {niterations}"
