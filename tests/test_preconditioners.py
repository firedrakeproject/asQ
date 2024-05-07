import firedrake as fd
import asQ
import pytest

nts = [pytest.param(n, id=f"nt{n}") for n in (4, 8, 16, 32)]


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize("nt", nts)
def test_jacobipc(nt):
    slice_length = nt//4
    time_partition = [slice_length for _ in range(4)]
    ensemble = asQ.create_ensemble(time_partition)

    mesh = fd.UnitSquareMesh(nx=16, ny=16,
                             comm=ensemble.comm)
    x, y = fd.SpatialCoordinate(mesh)

    V = fd.FunctionSpace(mesh, "CG", 1)
    uinitial = fd.Function(V)
    uinitial.project(fd.sin(x) + fd.cos(y))

    def form_mass(u, v):
        return u*v*fd.dx

    def form_function(u, v, t):
        return fd.inner(fd.grad(u), fd.grad(v))*fd.dx

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

    paradiag = asQ.Paradiag(
        ensemble=ensemble,
        form_mass=form_mass,
        form_function=form_function,
        ics=uinitial, dt=0.1, theta=0.5,
        time_partition=time_partition,
        solver_parameters=solver_parameters)

    paradiag.solve(nwindows=1)

    niterations = paradiag.solver.snes.getLinearSolveIterations()
    assert niterations == nt, "JacobiPC should solve exactly after nt iterations"


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize("alpha", [1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
def test_circulantpc(alpha):
    slice_length = 4
    time_partition = [slice_length for _ in range(4)]
    ensemble = asQ.create_ensemble(time_partition)

    mesh = fd.UnitSquareMesh(nx=16, ny=16,
                             comm=ensemble.comm)
    x, y = fd. SpatialCoordinate(mesh)

    V = fd.FunctionSpace(mesh, "CG", 1)
    uinitial = fd.Function(V)
    uinitial.project(fd.sin(x) + fd.cos(y))

    def form_mass(u, v):
        return u*v*fd.dx

    def form_function(u, v, t):
        return fd.inner(fd.grad(u), fd.grad(v))*fd.dx

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

    paradiag = asQ.Paradiag(
        ensemble=ensemble,
        form_mass=form_mass,
        form_function=form_function,
        ics=uinitial, dt=0.1, theta=0.5,
        time_partition=time_partition,
        solver_parameters=solver_parameters)

    paradiag.solve(nwindows=1)

    niterations = paradiag.solver.snes.getLinearSolveIterations()
    assert niterations == 3, "CirculantPC should have a convergence rate alpha"
