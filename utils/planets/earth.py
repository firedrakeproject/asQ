
import firedrake as fd

import utils.units as units

# # # === --- constants --- === # # #

# length of a single earth day
day = 24*units.hour
Day = fd.Constant(day)

# radius in metres
radius = 6371220*units.metre
Radius = fd.Constant(radius)

# rotation rate
omega = 7.292e-5/units.second
Omega = fd.Constant(omega)

# gravitational acceleration
gravity = 9.80616*units.metre/(units.second*units.second)
Gravity = fd.Constant(gravity)


# convert cartesian coordinates to latitude/longitude coordinates
def cart_to_sphere_coords(x, y, z):
    '''
    return latitude and longitude coordinates
    '''
    r = fd.sqrt(x*x + y*y + z*z)
    zr = z/r
    zr_corr = fd.min_value(fd.max_value(zr, -1), 1)  # avoid roundoff errors at poles
    theta = fd.asin(zr_corr)
    lamda0 = fd.atan2(y, x)
    return theta, lamda0


# convert vector in spherical coordinates to cartesian coordinates
def sphere_to_cart_vector(x, y, z, uzonal, umerid):
    theta, lamda = cart_to_sphere_coords(x, y, z)
    cart_u_expr = -uzonal*fd.sin(lamda) - umerid*fd.sin(theta)*fd.cos(lamda)
    cart_v_expr = uzonal*fd.cos(lamda) - umerid*fd.sin(theta)*fd.sin(lamda)
    cart_w_expr = umerid*fd.cos(theta)
    return fd.as_vector((cart_u_expr, cart_v_expr, cart_w_expr))
