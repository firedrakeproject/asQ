
from math import pi

import firedrake as fd
from utils.planets import earth
from utils.shallow_water import earth_coriolis_expression, default_depth_function_space

import numpy as np

# # # === --- constants --- === # # #

umax = 80.
Umax = fd.Constant(umax)

h0 = 10e3
H0 = fd.Constant(h0)

theta0 = pi/7.
theta1 = pi/2. - theta0
en = np.exp(-4./((theta1-theta0)**2))

alpha = fd.Constant(1/3.)
beta = fd.Constant(1/15.)
hhat = fd.Constant(120)
theta2 = fd.Constant(pi/4.)

# # # === --- analytical solution --- === # # #


# coriolis parameter f
def coriolis_expression(x, y, z):
    return earth_coriolis_expression(x, y, z)


def coriolis_function(x, y, z, Vf, name="coriolis"):
    f = fd.Function(Vf, name=name)
    return f.interpolate(coriolis_expression(x, y, z))


# topography field b
def topography_expression(x, y, z):
    return fd.Constant(0)


def topography_function(x, y, z, V2, name="topography"):
    b = fd.Function(V2, name=name)
    return b.interpolate(topography_expression(x, y, z))


def zonal_velocity_expression(theta, exp):
    return (umax/en)*exp(1./((theta - theta0)*(theta - theta1)))


def velocity_expression(x, y, z):
    theta, lamda = earth.cart_to_sphere_coords(x, y, z)
    uzonal_expr = zonal_velocity_expression(theta, fd.exp)
    uzonal = fd.conditional(fd.ge(theta, theta0),
                            fd.conditional(fd.le(theta, theta1),
                                           uzonal_expr, 0.), 0.)
    umeridional = 0.
    return earth.sphere_to_cart_vector(x, y, z, uzonal, umeridional)


def velocity_function(x, y, z, V1, name="velocity"):
    v = fd.Function(V1, name=name)
    return v.project(velocity_expression(x, y, z))


def depth_integrand(theta):
    # Initial h field is calculated by integrating h_integrand w.r.t. theta
    # Assumes the input is between theta0 and theta1.
    # Note that this function operates on vectorized input.
    f = 2*earth.omega*np.sin(theta)
    uzonal = zonal_velocity_expression(theta, np.exp)
    return uzonal*(f + np.tan(theta)*uzonal/earth.radius)


def depth_calculation(x):
    # Function to return value of h at x
    from scipy import integrate

    # Preallocate output array
    depth = np.zeros(len(x))

    angles = np.zeros(len(x))

    # Minimize work by only calculating integrals for points with
    # theta between theta_0 and theta_1.
    # For theta <= theta_0, the integral is 0
    # For theta >= theta_1, the integral is constant.

    # Fixed quadrature with 64 points gives absolute errors below 1e-13
    # for a quantity of order 1e-3.
    # Precalculate this constant:
    poledepth, _ = integrate.fixed_quad(depth_integrand, theta0, theta1, n=64)
    poledepth *= -earth.radius/earth.gravity

    angles[:] = np.arcsin(x[:, 2]/earth.radius)

    for ii in range(len(x)):
        if angles[ii] <= theta0:
            depth[ii] = 0.0
        elif angles[ii] >= theta1:
            depth[ii] = poledepth
        else:
            v, _ = integrate.fixed_quad(depth_integrand, theta0, angles[ii], n=64)
            depth[ii] = -(earth.radius/earth.gravity)*v
    return depth


def depth_perturbation(theta, lamda, V):
    latitude_profile = fd.exp(-((theta2 - theta)/beta)**2)
    longitude_profile = fd.exp(-(lamda/alpha)**2)
    pole_scaling = fd.cos(theta)
    return fd.Function(V).interpolate(hhat*pole_scaling*longitude_profile*latitude_profile)


def depth_expression(x, y, z):
    # set up function spaces
    from ufl.domain import extract_unique_domain
    mesh = extract_unique_domain(x)
    V = default_depth_function_space(mesh)
    W = fd.VectorFunctionSpace(mesh, V.ufl_element())

    # initialise depth from coordinates
    coords = fd.Function(W).interpolate(mesh.coordinates)
    h = fd.Function(V)
    h.dat.data[:] = depth_calculation(coords.dat.data_ro)

    # correct mean depth and add unstable perturbation
    cells = fd.Function(V).assign(fd.Constant(1))
    area = fd.assemble(cells*fd.dx)
    hmean = fd.assemble(h*fd.dx)/area
    hpert = depth_perturbation(*earth.cart_to_sphere_coords(x, y, z), V)
    h += H0 - hmean + hpert

    return h


def depth_function(x, y, z, V2, name="depth"):
    eta = fd.Function(V2, name=name)
    return eta.project(depth_expression(x, y, z))
