import firedrake as fd
import asQ
import pytest


def make_paradiag(time_partition, parameters):
    ensemble = asQ.create_ensemble(time_partition)
    mesh = fd.UnitSquareMesh(
        nx=16, ny=16, comm=ensemble.comm,
        distribution_parameters={'partitioner_type': 'simple'})

    x, y = fd.SpatialCoordinate(mesh)

    V = fd.FunctionSpace(mesh, "CG", 1)
    uinitial = fd.Function(V)
    uinitial.project(fd.sin(x) + fd.cos(y))

    def form_mass(u, v):
        return u*v*fd.dx

    def form_function(u, v, t):
        return fd.inner(fd.grad(u), fd.grad(v))*fd.dx

    return asQ.Paradiag(
        ensemble=ensemble,
        form_mass=form_mass,
        form_function=form_function,
        ics=uinitial, dt=0.1, theta=0.5,
        time_partition=time_partition,
        solver_parameters=parameters)


nts = [pytest.param(n, id=f"nt{n}") for n in (4, 8, 16, 32)]


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize("nt", nts)
def test_jacobipc(nt):
    slice_length = nt//4
    time_partition = [slice_length for _ in range(4)]

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

    paradiag = make_paradiag(time_partition, solver_parameters)

    paradiag.solve(nwindows=1)

    niterations = paradiag.solver.snes.getLinearSolveIterations()
    assert niterations == nt, "JacobiPC should solve exactly after nt iterations"


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize("alpha", [1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
def test_circulantpc(alpha):
    slice_length = 4
    time_partition = [slice_length for _ in range(4)]

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
    assert niterations == 3, "CirculantPC should have a convergence rate alpha"


nstep1to8 = [pytest.param(n, id=f"nstep{n}") for n in (1, 2, 4, 8)]
nstep2to16 = [pytest.param(n, id=f"nstep{n}") for n in (2, 4, 8, 16)]


@pytest.mark.parallel(nprocs=8)
@pytest.mark.parametrize("nsteps", nstep1to8)
def test_slicejacobipc_jacobi(nsteps):
    slice_length = 1
    time_partition = [slice_length for _ in range(8)]
    ensemble = asQ.create_ensemble(time_partition)

    mesh = fd.UnitSquareMesh(
        nx=16, ny=16, comm=ensemble.comm,
        distribution_parameters={'partitioner_type': 'simple'})

    x, y = fd.SpatialCoordinate(mesh)

    V = fd.FunctionSpace(mesh, "CG", 1)
    uinitial = fd.Function(V)
    uinitial.project(fd.sin(x) + fd.cos(y))

    def form_mass(u, v):
        return u*v*fd.dx

    def form_function(u, v, t):
        return fd.inner(fd.grad(u), fd.grad(v))*fd.dx

    jacobi_parameters = {
        'ksp_monitor': None,
        'ksp_converged_rate': None,
        'snes_type': 'ksponly',
        'mat_type': 'matfree',
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'pc_python_type': 'asQ.JacobiPC',
    }

    slice_parameters = {
        'ksp_monitor': None,
        'ksp_converged_rate': None,
        'snes_type': 'ksponly',
        'mat_type': 'matfree',
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'pc_python_type': 'asQ.SliceJacobiPC',
        'slice_jacobi_nsteps': nsteps,
        'slice_jacobi_slice': {
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

    paradiag_slice = asQ.Paradiag(
        ensemble=ensemble,
        form_mass=form_mass,
        form_function=form_function,
        ics=uinitial, dt=0.1, theta=0.5,
        time_partition=time_partition,
        solver_parameters=slice_parameters)

    paradiag_jacobi.solve(nwindows=1)
    paradiag_slice.solve(nwindows=1)

    jfunc = paradiag_jacobi.aaofunc
    sfunc = paradiag_slice.aaofunc

    with jfunc.global_vec_ro() as jvec, sfunc.global_vec_ro() as svec:
        errvec = jvec - svec
        err = errvec.norm()

    assert err < 1e-15, "SliceJacobiPC with JacobiPC should be exactly JacobiPC for any slice size"


@pytest.mark.parallel(nprocs=4)
def test_slicejacobipc_circulant():
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

    def form_mass(u, v):
        return u*v*fd.dx

    def form_function(u, v, t):
        return fd.inner(fd.grad(u), fd.grad(v))*fd.dx

    circulant_parameters = {
        'ksp_monitor': None,
        'ksp_converged_rate': None,
        'snes_type': 'ksponly',
        'mat_type': 'matfree',
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'pc_python_type': 'asQ.CirculantPC',
    }

    slice_parameters = {
        'ksp_monitor': None,
        'ksp_converged_rate': None,
        'snes_type': 'ksponly',
        'mat_type': 'matfree',
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'pc_python_type': 'asQ.SliceJacobiPC',
        'slice_jacobi_nsteps': sum(time_partition),
        'slice_jacobi_slice': {
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

    paradiag_slice = asQ.Paradiag(
        ensemble=ensemble,
        form_mass=form_mass,
        form_function=form_function,
        ics=uinitial, dt=0.1, theta=0.5,
        time_partition=time_partition,
        solver_parameters=slice_parameters)

    paradiag_circulant.solve(nwindows=1)
    paradiag_slice.solve(nwindows=1)

    cfunc = paradiag_circulant.aaofunc
    sfunc = paradiag_slice.aaofunc

    with cfunc.global_vec_ro() as cvec, sfunc.global_vec_ro() as svec:
        errvec = cvec - svec
        err = errvec.norm()

    assert err < 1e-15, "SliceJacobiPC with CirculantPC should be exactly CirculantPC if nstep=nt"


@pytest.mark.parallel(nprocs=8)
@pytest.mark.parametrize("nsteps", nstep2to16)
def test_slicejacobipc_slice(nsteps):
    slice_length = 2
    time_partition = [slice_length for _ in range(8)]
    nslices = sum(time_partition)//nsteps

    solver_parameters = {
        'ksp_monitor': None,
        'ksp_converged_rate': None,
        'snes_type': 'ksponly',
        'mat_type': 'matfree',
        'ksp_type': 'richardson',
        'ksp_rtol': 1e-15,
        'pc_type': 'python',
        'pc_python_type': 'asQ.SliceJacobiPC',
        'slice_jacobi_nsteps': nsteps,
        'slice_jacobi_state': 'linear',
        'slice_jacobi_slice': {
            'ksp_converged_rate': None,
            'ksp_type': 'richardson',
            'ksp_rtol': 1e-15,
            'pc_type': 'python',
            'pc_python_type': 'asQ.CirculantPC',
            'circulant_alpha': 1e-8,
            'circulant_block': {
                'ksp_type': 'preonly',
                'pc_type': 'lu',
            },
            'circulant_state': 'linear',
        }
    }

    paradiag = make_paradiag(time_partition, solver_parameters)

    paradiag.solve(nwindows=1)

    niterations = paradiag.solver.snes.getLinearSolveIterations()
    assert niterations == nslices, "SliceJacobiPC with exactly solved slices should solve exactly after nt iterations"
