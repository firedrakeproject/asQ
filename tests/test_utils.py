
from math import pi

import firedrake as fd
import utils
import numpy as np


def test_williamson2_elevation():
    '''
    Test the initial elevation field for williamson1992 case 2
    '''
    case2 = utils.shallow_water.williamson1992.case2
    earth = utils.planets.earth

    r = earth.radius
    w = earth.omega
    g = earth.gravity

    # test at nsamples random points
    np.random.seed(824395)
    nsamples = 100

    uref = np.random.rand()
    href = np.random.rand()

    # expression from williamson1992
    longitude = 2*pi*np.random.rand(nsamples)
    latitude = 2*pi*np.random.rand(nsamples)

    hcheck = -(r*w*uref + 0.5*uref*uref)*pow(np.sin(latitude), 2)/g

    # cartesian form
    x = r*np.cos(latitude)*np.cos(longitude)
    y = r*np.cos(latitude)*np.sin(longitude)
    z = r*np.sin(latitude)

    h = case2.elevation_expression(x, y, z, href=href, uref=uref)

    # get value of fd constant/expression
    def evalc(c):
        return c.evaluate(None, None, None, None)

    for i in range(nsamples):
        assert (abs(evalc(h[i]) - hcheck[i]) < 1e-12)


def test_williamson2_velocity():
    '''
    Test the initial velocity field for williamson1992 case 2
    '''
    case2 = utils.shallow_water.williamson1992.case2
    earth = utils.planets.earth

    r = earth.radius

    # test at nsamples random points
    np.random.seed(927456)
    nsamples = 100

    uref = np.random.rand()

    # expression from williamson1992
    longitude = 2*pi*np.random.rand(nsamples)
    latitude = 2*pi*np.random.rand(nsamples)

    umag = uref*np.cos(latitude)

    # project to cartesian coordinates
    ucheck = np.array([-umag*np.sin(longitude),
                       umag*np.cos(longitude),
                       np.zeros(nsamples)])
    ucheck = ucheck.transpose()

    # cartesian form
    x = r*np.cos(latitude)*np.cos(longitude)
    y = r*np.cos(latitude)*np.sin(longitude)
    z = r*np.sin(latitude)

    # get value of fd constant/expression
    def evalc(c):
        return c.evaluate(None, None, None, None)

    for i in range(nsamples):

        u = case2.velocity_expression(x[i], y[i], z[i], uref=uref)

        assert (abs(evalc(u[0]) - ucheck[i][0]) < 1e-12)
        assert (abs(evalc(u[1]) - ucheck[i][1]) < 1e-12)
        assert (abs(evalc(u[2]) - ucheck[i][2]) < 1e-12)


def test_williamson5_elevation():
    '''
    Test the initial elevation field for williamson1992 case 5
    Should be identical to case2
    '''
    case2 = utils.shallow_water.williamson1992.case2
    case5 = utils.shallow_water.williamson1992.case5
    earth = utils.planets.earth

    r = earth.radius

    # test at nsamples random points
    np.random.seed(824395)
    nsamples = 100

    uref = np.random.rand()
    href = np.random.rand()

    # expression from williamson1992
    longitude = 2*pi*np.random.rand(nsamples)
    latitude = 2*pi*np.random.rand(nsamples)

    # cartesian form
    x = r*np.cos(latitude)*np.cos(longitude)
    y = r*np.cos(latitude)*np.sin(longitude)
    z = r*np.sin(latitude)

    hcheck = case2.elevation_expression(x, y, z, href=href, uref=uref)
    h = case5.elevation_expression(x, y, z, href=href, uref=uref)

    # get value of fd constant/expression
    def evalc(c):
        return c.evaluate(None, None, None, None)

    for i in range(nsamples):
        assert (abs(evalc(h[i]) - evalc(hcheck[i])) < 1e-12)


def test_williamson5_velocity():
    '''
    Test the initial velocity field for williamson1992 case 5
    Should be identical to case2
    '''
    case2 = utils.shallow_water.williamson1992.case2
    case5 = utils.shallow_water.williamson1992.case5
    earth = utils.planets.earth

    r = earth.radius

    # test at nsamples random points
    np.random.seed(927456)
    nsamples = 100

    uref = np.random.rand()

    longitude = 2*pi*np.random.rand(nsamples)
    latitude = 2*pi*np.random.rand(nsamples)

    # cartesian form
    x = r*np.cos(latitude)*np.cos(longitude)
    y = r*np.cos(latitude)*np.sin(longitude)
    z = r*np.sin(latitude)

    # get value of fd constant/expression
    def evalc(c):
        return c.evaluate(None, None, None, None)

    for i in range(nsamples):

        u = case2.velocity_expression(x[i], y[i], z[i], uref=uref)
        ucheck = case5.velocity_expression(x[i], y[i], z[i], uref=uref)

        assert (abs(evalc(u[0]) - evalc(ucheck[0])) < 1e-12)
        assert (abs(evalc(u[1]) - evalc(ucheck[1])) < 1e-12)
        assert (abs(evalc(u[2]) - evalc(ucheck[2])) < 1e-12)


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
