
from math import pi

import utils
import numpy as np


def test_williamson2_elevation2():
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

    hcheck = -(r*w*uref + 0.5*uref*uref)*pow(np.sin(latitude),2)/g

    # cartesian form
    x = r*np.cos(latitude)*np.cos(longitude)
    y = r*np.cos(latitude)*np.sin(longitude)
    z = r*np.sin(latitude)

    h = case2.elevation_expression(x, y, z, href=href, uref=uref)

    # get value of fd constant/expression
    def evalc(c):
        return c.evaluate(None, None, None, None)

    for i in range(nsamples):
        assert(abs(evalc(h[i]) - hcheck[i]) < 1e-12)
