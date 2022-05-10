
import firedrake as fd

import firedrake_utils.units as units

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

# # # === --- planetary mesh --- === # # #


def IcosahedralMesh(refinement_level=0,
                    degree=1,
                    reorder=None,
                    distribution_parameters=None,
                    comm=fd.COMM_WORLD):

    globe = fd.IcosahedralSphereMesh(
                radius=radius,
                refinement_level=refinement_level,
                degree=degree,
                reorder=reorder,
                distribution_parameters=distribution_parameters,
                comm=comm)

    globe.init_cell_orientations(fd.SpatialCoordinate(globe))

    return globe
