
from math import pi

import firedrake as fd
from utils.planets import earth
from utils import units

# # # === --- constants --- === # # #

# gravitational constant * reference depth
gh0 = 2.94e4*pow(units.metre/units.second, 2)
Gh0 = fd.Constant(gh0)

# reference depth
h0 = gh0/earth.gravity
H0 = fd.Constant(h0)

# days taken for velocity to travel circumference
period = 12.0
Period = fd.Constant(period)

# reference velocity
u0 = 2*pi*earth.radius/(period*earth.day)
U0 = fd.Constant(u0)

# # # === --- analytical solution --- === # # #


# coriolis parameter f
def coriolis_expression(x, y, z):
    return 2*earth.Omega*z/earth.Radius


def coriolis_function(x, y, z, Vf, name="coriolis"):
    f = fd.Function(Vf, name=name)
    return f.interpolate(coriolis_expression(x, y, z))


# velocity field u
def velocity_expression(x, y, z, uref=U0):
    return fd.as_vector([-uref*y/earth.Radius, uref*x/earth.Radius, 0.0])


def velocity_function(x, y, z, V1, uref=U0, name="velocity"):
    v = fd.Function(V1, name=name)
    return v.project(velocity_expression(x, y, z, uref=uref))


# elevation field eta
def elevation_expression(x, y, z, href=H0, uref=U0):
    z0 = z/earth.Radius
    k = (earth.Radius*earth.Omega*uref + uref*uref/2.0)
    return - k*(z0*z0)/earth.Gravity


def elevation_function(x, y, z, V2, href=H0, uref=U0, name="elevation"):
    eta = fd.Function(V2, name=name)
    return eta.project(elevation_expression(x, y, z, href=href, uref=uref))


# topography field b
def topography_expression(x, y, z):
    return fd.Constant(0)


def topography_function(x, y, z, V2,
                        name="topography"):
    b = fd.Function(V2, name=name)
    return b.interpolate(topography_expression(x, y, z))
