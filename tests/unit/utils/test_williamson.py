
from math import pi

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
        assert (abs(evalc(h[i]) - hcheck[i]) < 1e-12), "Williamson2 depth should match expression from paper"


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

        assert (abs(evalc(u[0]) - ucheck[i][0]) < 1e-12), "Williamson2 velocity should match expression from paper"
        assert (abs(evalc(u[1]) - ucheck[i][1]) < 1e-12), "Williamson2 velocity should match expression from paper"
        assert (abs(evalc(u[2]) - ucheck[i][2]) < 1e-12), "Williamson2 velocity should match expression from paper"


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
        assert (abs(evalc(h[i]) - evalc(hcheck[i])) < 1e-12), "Williamson5 depth should be identical to Williamson2"


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

        assert (abs(evalc(u[0]) - evalc(ucheck[0])) < 1e-12), "Williamson5 velocity should be identical to Williamson2"
        assert (abs(evalc(u[1]) - evalc(ucheck[1])) < 1e-12), "Williamson5 velocity should be identical to Williamson2"
        assert (abs(evalc(u[2]) - evalc(ucheck[2])) < 1e-12), "Williamson5 velocity should be identical to Williamson2"
