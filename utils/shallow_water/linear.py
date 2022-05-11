
import firedrake as fd

# # # === --- finite element forms --- === # # #

# mass forms for depth and velocity fields


def form_mass_h(mesh, h, p):
    return (p*h)*fd.dx


def form_mass_u(mesh, u, w):
    return fd.inner(u, w)*fd.dx


def form_mass(mesh, h, u, p, w):
    return form_mass_h(mesh, h, p) + form_mass_u(mesh, u, w)


# spatial forms for depth and velocity fields

def form_function_h(mesh, H, h, u, p):
    return (H*p*fd.div(u))*fd.dx


def form_function_u(mesh, g, f, h, u, w, perp=fd.cross):

    outward_normals = fd.CellNormal(mesh)

    def prp(u):
        return perp(outward_normals, u)

    return (fd.inner(w, f*prp(u)) - g*h*fd.div(w))*fd.dx


def form_function(mesh, g, H, f, h, u, p, w):
    return form_function_h(mesh, H, h, u, p) + form_function_u(mesh, g, f, h, u, w)
