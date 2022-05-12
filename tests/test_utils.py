
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

    x = r*np.random.rand(nsamples)
    y = r*np.random.rand(nsamples)
    z = r*np.random.rand(nsamples)

    # get value of fd constant/expression
    def evalc(c):
        return c.evaluate(None, None, None, None)

    for i in range(nsamples):
        hcheck = -(r*w*uref + 0.5*uref*uref)*z[i]*z[i]/(g*r*r)

        h = case2.elevation_expression(x[i], y[i], z[i], href=href, uref=uref)

        assert(abs(evalc(h) - hcheck) < 1e-12)
