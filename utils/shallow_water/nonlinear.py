
import firedrake as fd

# # # === --- finite element forms --- === # # #

# mass forms for depth and velocity fields


def form_mass_h(mesh, h, p):
    return (p*h)*fd.dx


def form_mass_u(mesh, u, v):
    return fd.inner(u, v)*fd.dx


def form_mass(mesh, h, u, p, v):
    return form_mass_h(mesh, h, p) + form_mass_u(mesh, u, v)


# spatial forms for depth and velocity fields


def form_function_depth(mesh, h, u, p):
    n = fd.FacetNormal(mesh)
    uup = 0.5 * (fd.dot(u, n) + abs(fd.dot(u, n)))

    return (- fd.inner(fd.grad(p), u)*h*fd.dx
            + fd.jump(p)*(uup('+')*h('+') - uup('-')*h('-'))*fd.dS)


def form_function_velocity(mesh, g, b, f, h, u, v, perp=fd.cross):
    n = fd.FacetNormal(mesh)
    outward_normals = fd.CellNormal(mesh)

    def prp(u):
        return perp(outward_normals, u)

    def both(u):
        return 2*fd.avg(u)

    K = 0.5*fd.inner(u, u)
    upwind = 0.5 * (fd.sign(fd.dot(u, n)) + 1)

    return (fd.inner(v, f*prp(u))*fd.dx
            - fd.inner(prp(fd.grad(fd.inner(v, prp(u)))), u)*fd.dx
            + fd.inner(both(prp(n)*fd.inner(v, prp(u))),
                       both(upwind*u))*fd.dS
            - fd.div(v)*(g*(h + b) + K)*fd.dx)


def form_function(mesh, g, b, f, h, u, q, v):
    return form_function_velocity(mesh, g, b, f, h, u, v) + form_function_depth(mesh, h, u, q)
