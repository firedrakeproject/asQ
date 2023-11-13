
from math import pi

import firedrake as fd
from utils.planets import earth
from utils import units
from utils.shallow_water.williamson1992 import case2

# # # === --- constants --- === # # #

# reference depth
h0 = 5960*units.metre
H0 = fd.Constant(h0)

# reference velocity
u0 = 20*units.metre/units.second
U0 = fd.Constant(u0)

# mountain parameters
mountain_height = 2000*units.metre
Mountain_height = fd.Constant(mountain_height)

mountain_radius = pi/9.
Mountain_radius = fd.Constant(mountain_radius)

# different lambda_c because atan2 used for angle
mountain_centre_lambda = -pi/2.
Mountain_centre_lambda = fd.Constant(mountain_centre_lambda)

mountain_centre_theta = pi/6.
Mountain_centre_theta = fd.Constant(mountain_centre_theta)

# # # === --- analytical solution --- === # # #


# coriolis parameter f
def coriolis_expression(x, y, z):
    return case2.coriolis_expression(x, y, z)


def coriolis_function(x, y, z, Vf, name="coriolis"):
    return fd.Function(Vf, name=name).interpolate(coriolis_expression(x, y, z))


# velocity field u
def velocity_expression(x, y, z, uref=U0):
    return case2.velocity_expression(x, y, z, uref=uref)


def velocity_function(x, y, z, V1, uref=U0, name="velocity"):
    v = fd.Function(V1, name=name)
    return v.project(velocity_expression(x, y, z, uref=uref))


# elevation field eta
def elevation_expression(x, y, z, href=H0, uref=U0):
    return case2.elevation_expression(x, y, z, href=href, uref=uref)


def elevation_function(x, y, z, V2, href=H0, uref=U0, name="elevation"):
    eta = fd.Function(V2, name=name)
    return eta.project(elevation_expression(x, y, z, href=href, uref=uref))


def depth_expression(x, y, z, href=H0, uref=U0):
    b = topography_expression(x, y, z)
    eta = elevation_expression(x, y, z, href=href, uref=uref)
    return href - b + eta


# topography field b
def topography_expression(x, y, z,
                          radius=Mountain_radius,
                          height=Mountain_height,
                          theta_c=Mountain_centre_theta,
                          lambda_c=Mountain_centre_lambda):

    lambda_x = fd.atan2(y/earth.Radius, x/earth.Radius)
    theta_x = fd.asin(z/earth.Radius)

    radius2 = pow(radius, 2)
    lambda2 = pow(lambda_x - lambda_c, 2)
    theta2 = pow(theta_x - theta_c, 2)

    min_arg = fd.min_value(radius2, theta2 + lambda2)

    return height*(1 - fd.sqrt(min_arg)/radius)


def topography_function(x, y, z, V2,
                        radius=Mountain_radius,
                        height=Mountain_height,
                        theta_c=Mountain_centre_theta,
                        lambda_c=Mountain_centre_lambda,
                        name="topography"):

    b = fd.Function(V2, name=name)
    bexp = topography_expression(x, y, z,
                                 radius=radius,
                                 height=height,
                                 theta_c=theta_c,
                                 lambda_c=lambda_c)
    return b.interpolate(bexp)
