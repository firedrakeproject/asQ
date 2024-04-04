
import firedrake as fd
import numpy as np


def test_convective_cfl():
    '''
    test that the convective cfl calculation is the correct
    '''
    from utils import diagnostics

    n = 5
    dx = 1./5

    mesh = fd.UnitSquareMesh(n, n, quadrilateral=True)

    V = fd.VectorFunctionSpace(mesh, "DG", 0)

    u = fd.Function(V, name="velocity")

    zero = fd.Constant(0)
    one = fd.Constant(1)

    dt = fd.Constant(0.5)

    # test zero velocity case
    vel = fd.Constant(fd.as_vector([zero, zero]))
    cfl_check = fd.Constant(zero*dt/dx)

    u.assign(vel)
    cfl = diagnostics.convective_cfl(u, dt)

    assert (fd.errornorm(cfl_check, cfl) < 1e-12)

    # test unit velocity x case
    vel = fd.Constant(fd.as_vector([one, zero]))
    cfl_check = fd.Constant(one*dt/dx)

    u.assign(vel)
    cfl = diagnostics.convective_cfl(u, dt)

    assert (fd.errornorm(cfl_check, cfl) < 1e-12)

    # test unit velocity y case
    vel = fd.Constant(fd.as_vector([zero, one]))
    cfl_check = fd.Constant(one*dt/dx)

    u.assign(vel)
    cfl = diagnostics.convective_cfl(u, dt)

    assert (fd.errornorm(cfl_check, cfl) < 1e-12)

    # test unit velocity xy case
    vel = fd.Constant(fd.as_vector([one, one]))
    cfl_check = fd.Constant(2*dt/dx)

    u.assign(vel)
    cfl = diagnostics.convective_cfl(u, dt)

    assert (fd.errornorm(cfl_check, cfl) < 1e-12)

    # rng
    np.random.seed(23767)

    nrng = 10
    for i in range(nrng):
        dt.assign(fd.Constant(np.random.rand()))
        vx = np.random.rand()
        vy = np.random.rand()
        vel = fd.Constant(fd.as_vector([vx, vy]))
        cfl_check = fd.Constant((vx+vy)*dt/dx)

        u.assign(vel)
        cfl = diagnostics.convective_cfl(u, dt)

        assert (fd.errornorm(cfl_check, cfl) < 1e-12)


def test_get_cfl_calculator():
    '''
    test that the convective cfl calculator is the correct
    '''
    from utils import diagnostics

    n = 5
    dx = 1./5

    mesh = fd.UnitSquareMesh(n, n, quadrilateral=True)

    V = fd.VectorFunctionSpace(mesh, "DG", 0)

    u = fd.Function(V, name="velocity")

    zero = fd.Constant(0)
    one = fd.Constant(1)

    dt = fd.Constant(0.5)

    cfl_calculator = diagnostics.convective_cfl_calculator(mesh)

    # test zero velocity case
    vel = fd.Constant(fd.as_vector([zero, zero]))
    cfl_check = fd.Constant(zero*dt/dx)

    u.assign(vel)
    cfl = cfl_calculator(u, dt)

    assert (fd.errornorm(cfl_check, cfl) < 1e-12)

    # test unit velocity x case
    vel = fd.Constant(fd.as_vector([one, zero]))
    cfl_check = fd.Constant(one*dt/dx)

    u.assign(vel)
    cfl = cfl_calculator(u, dt)

    assert (fd.errornorm(cfl_check, cfl) < 1e-12)

    # test unit velocity y case
    vel = fd.Constant(fd.as_vector([zero, one]))
    cfl_check = fd.Constant(one*dt/dx)

    u.assign(vel)
    cfl = cfl_calculator(u, dt)

    assert (fd.errornorm(cfl_check, cfl) < 1e-12)

    # test unit velocity xy case
    vel = fd.Constant(fd.as_vector([one, one]))
    cfl_check = fd.Constant(2*dt/dx)

    u.assign(vel)
    cfl = cfl_calculator(u, dt)

    assert (fd.errornorm(cfl_check, cfl) < 1e-12)

    # rng
    np.random.seed(23767)

    nrng = 10
    for i in range(nrng):
        dt.assign(fd.Constant(np.random.rand()))
        vx = np.random.rand()
        vy = np.random.rand()
        vel = fd.Constant(fd.as_vector([vx, vy]))
        cfl_check = fd.Constant((vx+vy)*dt/dx)

        u.assign(vel)
        cfl = cfl_calculator(u, dt)

        assert (fd.errornorm(cfl_check, cfl) < 1e-12)


def test_get_cfl_calculator_extruded():
    '''
    test that the convective cfl calculator is the correct for extruded meshes
    '''
    from utils import diagnostics

    n = 5
    dx = 1./5

    base_mesh = fd.UnitIntervalMesh(n)
    mesh = fd.ExtrudedMesh(base_mesh, n, layer_height=dx)

    V = fd.VectorFunctionSpace(mesh, "DG", 0)

    u = fd.Function(V, name="velocity")

    zero = fd.Constant(0)
    one = fd.Constant(1)

    dt = fd.Constant(0.5)

    cfl_calculator = diagnostics.convective_cfl_calculator(mesh)

    # test zero velocity case
    vel = fd.Constant(fd.as_vector([zero, zero]))
    cfl_check = fd.Constant(zero*dt/dx)

    u.assign(vel)
    cfl = cfl_calculator(u, dt)

    assert (fd.errornorm(cfl_check, cfl) < 1e-12)

    # test unit velocity x case
    vel = fd.Constant(fd.as_vector([one, zero]))
    cfl_check = fd.Constant(one*dt/dx)

    u.assign(vel)
    cfl = cfl_calculator(u, dt)

    assert (fd.errornorm(cfl_check, cfl) < 1e-12)

    # test unit velocity y case
    vel = fd.Constant(fd.as_vector([zero, one]))
    cfl_check = fd.Constant(one*dt/dx)

    u.assign(vel)
    cfl = cfl_calculator(u, dt)

    assert (fd.errornorm(cfl_check, cfl) < 1e-12)

    # test unit velocity xy case
    vel = fd.Constant(fd.as_vector([one, one]))
    cfl_check = fd.Constant(2*dt/dx)

    u.assign(vel)
    cfl = cfl_calculator(u, dt)

    assert (fd.errornorm(cfl_check, cfl) < 1e-12)

    # rng
    np.random.seed(23767)

    nrng = 10
    for i in range(nrng):
        dt.assign(fd.Constant(np.random.rand()))
        vx = np.random.rand()
        vy = np.random.rand()
        vel = fd.Constant(fd.as_vector([vx, vy]))
        cfl_check = fd.Constant((vx+vy)*dt/dx)

        u.assign(vel)
        cfl = cfl_calculator(u, dt)

        assert (fd.errornorm(cfl_check, cfl) < 1e-12)
