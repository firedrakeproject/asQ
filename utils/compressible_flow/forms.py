import firedrake as fd
from utils.misc import curl0, cross0, cross1
from utils.compressible_flow.gas import pi_formula


def theta_tendency(q, u, theta, n, Up, c_pen):
    unn = 0.5*(fd.inner(u, n) + abs(fd.inner(u, n)))

    # the basic consistent equation with horizontal upwinding
    eqn = (
        q*fd.inner(u, fd.grad(theta))*fd.dx
        + fd.jump(q)*(unn('+')*theta('+')
                      - unn('-')*theta('-'))*fd.dS_v
        - fd.jump(q*u*theta, n)*fd.dS_v
    )
    # jump stabilisation
    mesh = u.ufl_domain()
    h = fd.avg(fd.CellVolume(mesh))/fd.FacetArea(mesh)

    eqn += (
        h**2*c_pen*abs(fd.inner(u('+'), n('+')))
        * fd.inner(fd.jump(fd.grad(theta)),
                   fd.jump(fd.grad(q)))*(fd.dS_v+fd.dS_h))
    return eqn


def theta_mass(q, theta):
    return q*theta*fd.dx


def rho_mass(q, rho):
    return q*rho*fd.dx


def rho_tendency(q, rho, u, n):
    unn = 0.5*(fd.inner(u, n) + abs(fd.inner(u, n)))
    return (
        - fd.inner(fd.grad(q), u*rho)*fd.dx
        + fd.jump(q)*(unn('+')*rho('+')
                      - unn('-')*rho('-'))*(fd.dS_v + fd.dS_h)
    )


def u_mass(u, w):
    return fd.inner(u, w)*fd.dx


def u_tendency(w, n, u, theta, rho, gas, Up,
               mu=None, f=None, F=None,
               pi_degree=4):
    """
    Written in a dimension agnostic way
    """
    def both(u):
        return 2*fd.avg(u)

    Pi = pi_formula(rho, theta, gas)

    K = fd.Constant(0.5)*fd.inner(u, u)
    Upwind = 0.5*(fd.sign(fd.dot(u, n))+1)

    eqn = (
        + fd.inner(u, curl0(cross1(u, w)))*fd.dx
        - fd.inner(both(Upwind*u),
                   both(cross0(n, cross1(u, w))))*(fd.dS_h + fd.dS_v)
        - fd.div(w)*K*fd.dx
        - gas.cp*fd.div(theta*w)*Pi*fd.dx(degree=pi_degree)
        + gas.cp*fd.jump(w*theta, n)*fd.avg(Pi)*fd.dS_v(degree=pi_degree)
        + fd.inner(w, Up)*gas.g*fd.dx
    )

    if mu:  # Newtonian dissipation in vertical
        eqn += mu*fd.inner(w, Up)*fd.inner(u, Up)*fd.dx
    if f:  # Coriolis term
        eqn += f*fd.inner(w, fd.cross(Up, u))*fd.dx
    if F:  # additional source term
        eqn += fd.inner(w, F)*fd.dx
    return eqn


def get_form_mass():
    def form_mass(u, rho, theta, du, drho, dtheta):
        return u_mass(u, du) + rho_mass(rho, drho) + theta_mass(theta, dtheta)
    return form_mass


def get_form_function(n, Up, c_pen, gas, mu,
                      f=None, F=None,
                      viscosity=None, diffusivity=None,
                      pi_degree=4):
    def form_function(u, rho, theta, du, drho, dtheta, t):
        eqn = theta_tendency(dtheta, u, theta, n, Up, c_pen)
        eqn += rho_tendency(drho, rho, u, n)
        eqn += u_tendency(du, n, u, theta, rho,
                          gas, Up, mu, f, F, pi_degree=pi_degree)
        if viscosity:
            eqn += form_viscosity(u, du, viscosity)
        if diffusivity:
            eqn += form_viscosity(theta, dtheta, diffusivity)
        return eqn
    return form_function


def form_viscosity(u, v, kappa, mu=fd.Constant(10.0)):
    mesh = v.ufl_domain()
    n = fd.FacetNormal(mesh)
    a = fd.inner(fd.grad(u), fd.grad(v))*fd.dx
    h = fd.avg(fd.CellVolume(mesh))/fd.FacetArea(mesh)

    def get_flux_form(dS):
        fluxes = (-fd.inner(2*fd.avg(fd.outer(v, n)), fd.avg(fd.grad(u)))
                  - fd.inner(fd.avg(fd.grad(v)), 2*fd.avg(fd.outer(u, n)))
                  + mu*h*fd.inner(2*fd.avg(fd.outer(v, n)),
                                  2*fd.avg(fd.outer(u, n))))*dS
        return fluxes

    a += kappa*get_flux_form(fd.dS_v)
    a += kappa*get_flux_form(fd.dS_h)
    return a
