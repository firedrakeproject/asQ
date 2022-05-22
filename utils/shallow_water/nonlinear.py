
import firedrake as fd

# # # === --- finite element forms --- === # # #

# mass forms for depth and velocity fields


def form_mass_h(mesh, h, p):
    return (p*h)*fd.dx


def form_mass_u(mesh, u, v):
    return fd.inner(u, v)*fd.dx


def form_mass(mesh, u, h, v, p):
    return form_mass_h(mesh, h, p) + form_mass_u(mesh, u, v)


# spatial forms for depth and velocity fields


def form_function_depth(mesh, u, h, p):
    n = fd.FacetNormal(mesh)
    uup = 0.5 * (fd.dot(u, n) + abs(fd.dot(u, n)))

    return (- fd.inner(fd.grad(p), u)*h*fd.dx
            + fd.jump(p)*(uup('+')*h('+') - uup('-')*h('-'))*fd.dS)


def form_function_velocity(mesh, g, b, f, u, h, v, perp=fd.cross):
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


def form_function(mesh, g, b, f, u, h, v, q):
    return form_function_velocity(mesh, g, b, f, u, h, v) + form_function_depth(mesh, u, h, q)


# calculate diagnostic fields

def convective_cfl_calculator(mesh):
    '''
    Return a function that, when passed a velocity field and a timestep, will return a function that is the cfl number.
    '''
    DG0 = fd.FunctionSpace(mesh, "DG", 0)
    v = fd.TestFunction(DG0)

    cfl_denominator = fd.Function(DG0, name="CFL denominator")
    cfl_numerator = fd.Function(DG0, name="CFL numerator")
    cfl = fd.Function(DG0, name="cfl")

    # mesh volume
    One = fd.Function(DG0).assign(1.0)
    fd.assemble(One*v*fd.dx, tensor=cfl_denominator)

    def cfl_calc(u, dt):
        # area weighted convective flux
        n = fd.FacetNormal(u.function_space().mesh())
        un = 0.5*(fd.inner(-u, n) + abs(fd.inner(-u, n)))
        cfl_numerator_form = (
            2*fd.avg(un*v)*fd.dS
            + un*v*fd.ds
        )
        fd.assemble(cfl_numerator_form, tensor=cfl_numerator)

        dT = fd.Constant(dt)
        cfl.assign(dT*cfl_numerator/cfl_denominator)
        return cfl

    return cfl_calc


def convective_cfl(u, dt):
    '''
    Return a function that, when passed a velocity field and a timestep, will return a function that is the cfl number.
    '''
    return convective_cfl_calculator(u.function_space().mesh())(u, dt)
