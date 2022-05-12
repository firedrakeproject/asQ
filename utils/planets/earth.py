
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
