
from math import pi
from functools import partial

import firedrake as fd
from utils.planets import earth
from utils.shallow_water import earth_coriolis_expression

# # # === --- constants --- === # # #

H = fd.Constant(1e4)

lamda_c1 = 0.2*pi
lamda_c2 = 1.2*pi
lamda_c3 = 1.6*pi

phi_c1 = pi/3
phi_c2 = pi/5
phi_c3 = -pi/4

p1 = 10
p2 = 80
p3 = 360

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


def velocity_expression(*x):
    return fd.Constant(fd.as_vector([0, 0, 0]))


def gauss_coord(l1, phi1, l2, phi2):
    sp1p2 = fd.sin(phi1)*fd.sin(phi2)
    cp1p2 = fd.cos(phi1)*fd.cos(phi2)
    return fd.acos(sp1p2 + cp1p2*fd.cos(l1-l2))


def bump(lc, phic, w, l, phi):
    d = gauss_coord(lc, phic, l, phi)
    return 0.1*H*fd.exp(-d*d*w)


def depth_expression(*x):
    bump1 = partial(bump, lc=lamda_c1, phic=phi_c1, w=p1)
    bump2 = partial(bump, lc=lamda_c2, phic=phi_c2, w=p2)
    bump3 = partial(bump, lc=lamda_c3, phic=phi_c3, w=p3)
    phi, l = earth.cart_to_sphere_coords(*x)
    return H + bump1(l=l, phi=phi) + bump2(l=l, phi=phi) + bump3(l=l, phi=phi)
