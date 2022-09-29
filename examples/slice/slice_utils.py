import firedrake as fd
import numpy as np
from firedrake import op2
from petsc4py import PETSc

def maximum(f):
    fmax = op2.Global(1, [-1e50], dtype=float)
    op2.par_loop(op2.Kernel("""
static void maxify(double *a, double *b) {
    a[0] = a[0] < b[0] ? b[0] : a[0];
}
""", "maxify"), f.dof_dset.set, fmax(op2.MAX), f.dat(op2.READ))
    return fmax.data[0]

def minimum(f):
    fmin = op2.Global(1, [1e50], dtype=float)
    op2.par_loop(op2.Kernel("""
static void minify(double *a, double *b) {
    a[0] = a[0] > b[0] ? b[0] : a[0];
}
""", "minify"), f.dof_dset.set, fmin(op2.MIN), f.dat(op2.READ))
    return fmin.data[0]

def pi_formula(rho, theta, R_d, p_0, kappa):
    
    return (rho * R_d * theta / p_0) ** (kappa / (1 - kappa))

def rho_formula(pi, theta, R_d, p_0, kappa):
    return p_0*pi**((1-kappa)/kappa)/R_d/theta

def hydrostatic_rho(Vv, V2, mesh, thetan, rhon, pi_boundary,
                    cp, R_d, p_0, kappa, g, Up,
                    top = False, Pi = None):
    # Calculate hydrostatic Pi, rho
    W_h = Vv * V2
    wh = fd.Function(W_h)
    n = fd.FacetNormal(mesh)
    dv, drho = fd.TestFunctions(W_h)

    v, Pi0 = fd.TrialFunctions(W_h)
    
    Pieqn = (
        (cp*fd.inner(v, dv) - cp*fd.div(dv*thetan)*Pi0)*fd.dx
        + drho*fd.div(thetan*v)*fd.dx
    )
    
    if top:
        bmeasure = fd.ds_t
        bstring = "bottom"
    else:
        bmeasure = fd.ds_b
        bstring = "top"
        
    zeros = []
    for i in range(Up.ufl_shape[0]):
        zeros.append(fd.Constant(0.))
        
    L = -cp*fd.inner(dv, n)*thetan*pi_boundary*bmeasure
    L -= g*fd.inner(dv, Up)*fd.dx
    bcs = [fd.DirichletBC(W_h.sub(0), zeros, bstring)]
    
    PiProblem = fd.LinearVariationalProblem(Pieqn, L, wh, bcs=bcs)
        
    lu_params = {
        'snes_monitor': None,
        'snes_stol': 1.0e-50,
        'ksp_monitor': None,
        'snes_converged_reason': None,
        'mat_type': 'aij',
        'pc_type': 'lu',
        "pc_factor_mat_ordering_type": "rcm",
    }

    PiSolver = fd.LinearVariationalSolver(PiProblem,
                                          solver_parameters=lu_params,
                                          options_prefix="pisolver")
    PiSolver.solve()
    v, Pi0 = wh.split()
    if Pi:
        Pi.assign(Pi0)
    PETSc.Sys.Print("pi",maximum(Pi0))
    if rhon:
        rhon.interpolate(rho_formula(Pi0, thetan, R_d, p_0, kappa))
        PETSc.Sys.Print(maximum(rhon), minimum(rhon))
        v,  rho = wh.split()
        rho.assign(rhon)
        v, rho = fd.split(wh)

        Pif = pi_formula(rho, thetan, R_d, p_0, kappa)
        
        rhoeqn = (
            (cp*fd.inner(v, dv) - cp*fd.div(dv*thetan)*Pif)*fd.dx
            + cp*drho*fd.div(thetan*v)*fd.dx
        )

        RF = fd.assemble(drho*Pif*fd.dx)

        if top:
            bmeasure = fd.ds_t
            bstring = "bottom"
        else:
            bmeasure = fd.ds_b
            bstring = "top"

        zeros = []
        for i in range(Up.ufl_shape[0]):
            zeros.append(fd.Constant(0.))

        rhoeqn += cp*fd.inner(dv, n)*thetan*pi_boundary*bmeasure
        rhoeqn += g*fd.inner(dv, Up)*fd.dx
        bcs = [fd.DirichletBC(W_h.sub(0), zeros, bstring)]

        RhoProblem = fd.NonlinearVariationalProblem(rhoeqn, wh, bcs=bcs)

        RhoSolver = fd.NonlinearVariationalSolver(RhoProblem,
                                                  solver_parameters=lu_params,
                                                  options_prefix="rhosolver")

        RhoSolver.solve()
        v, Rho0 = wh.split()
        rhon.assign(Rho0)

def theta_tendency(q, u, theta, n, Up, c_pen):
    unn = 0.5*(fd.inner(u, n) + abs(fd.inner(u, n)))

    #the basic consistent equation with horizontal upwinding
    eqn = (
        q*fd.inner(u,fd.grad(theta))*fd.dx
        + fd.jump(q)*(unn('+')*theta('+')
                      - unn('-')*theta('-'))*fd.dS_v
        - fd.jump(q*u*theta, n)*fd.dS_v
        )
    #jump stabilisation
    mesh = u.ufl_domain()
    h = fd.avg(fd.CellVolume(mesh))/fd.FacetArea(mesh)

    eqn += (
        h**2*c_pen*abs(fd.inner(u('+'),n('+')))
        *fd.inner(fd.jump(fd.grad(theta)),
                  fd.jump(fd.grad(q)))*(fd.dS_v + fd.dS_h))
    return eqn

def theta_mass(q, theta):
    return q*theta*fd.dx

def rho_mass(q, rho):
    return q*rho*fd.dx

def rho_tendency(q, rho, u, n):
    unn = 0.5*(fd.inner(u, n) + abs(fd.inner(u, n)))
    return (
        - fd.inner(fd.grad(q), u*rho)*fd.dx +
        fd.jump(q)*(unn('+')*rho('+')
                    - unn('-')*rho('-'))*(fd.dS_v + fd.dS_h)
    )

def u_mass(u, w):
    return fd.inner(u, w)*fd.dx

def curl0(u):
    """
    Curl function from y-cpt field to x-z field
    """
    mesh = u.ufl_domain()
    d = np.sum(mesh.cell_dimension())
    
    if d == 2:
        # equivalent vector is (0, u, 0)

        # |i   j   k  |
        # |d_x 0   d_z| = (- du/dz, 0, du/dx)
        # |0   u   0  |
        return fd.as_vector([-u.dx(1), u.dx(0)])
    elif d == 3:
        return fd.curl(u)
    else:
        raise NotImplementedError


def curl1(u):
    """
    dual curl function from dim-1 forms to dim-2 forms
    """
    mesh = u.ufl_domain()
    d = np.sum(mesh.cell_dimension())

    if d == 2:
        # we have vector in x-z plane and return scalar
        # representing y component of the curl

        # |i   j   k   |
        # |d_x 0   d_z | = (0, -du_3/dx + du_1/dz, 0)
        # |u_1 0   u_3 |
        
        return -u[1].dx(0) + u[0].dx(1)
    elif d == 3:
        return fd.curl(u)
    else:
        raise NotImplementedError


def cross1(u, w):
    """
    cross product (slice vector field with slice vector field)
    """
    mesh = u.ufl_domain()
    d = np.sum(mesh.cell_dimension())

    if d == 2:
        # cross product of two slice vectors goes into y cpt

        # |i   j   k   |
        # |u_1 0   u_3 | = (0, -u_1*w_3 + u_3*w_1, 0)
        # |w_1 0   w_3 |

        return w[0]*u[1] - w[1]*u[0]
    elif d == 3:
        return fd.cross(u, w)
    else:
        raise NotImplementedError


def cross0(u, w):
    """
    cross product (slice vector field with out-of-slice vector field)
    """

    # |i   j   k   |
    # |u_1 0   u_3 | = (-w*u_3, 0, w*u_1)
    # |0   w   0   |

    mesh = u.ufl_domain()
    d = np.sum(mesh.cell_dimension())

    if d == 2:
        # cross product of two slice vectors goes into y cpt
        return fd.as_vector([-w*u[1], w*u[0]])
    elif d == 3:
        return fd.cross(u, w)
    else:
        raise NotImplementedError
    

def both(u):
    return 2*fd.avg(u)

def u_tendency(w, n, u, theta, rho,
               cp, g, R_d, p_0, kappa, Up,
               mu=None, f=None, F=None):
    """
    Written in a dimension agnostic way
    """
    Pi = pi_formula(rho, theta, R_d, p_0, kappa)
    
    K = fd.Constant(0.5)*fd.inner(u, u)
    Upwind = 0.5*(fd.sign(fd.dot(u, n))+1)

    eqn = (
        + fd.inner(u, curl0(cross1(u, w)))*fd.dx
        - fd.inner(both(Upwind*u),
                      both(cross0(n, cross1(u, w))))*(fd.dS_h + fd.dS_v)
        - fd.div(w)*K*fd.dx
        - cp*fd.div(theta*w)*Pi*fd.dx
        + cp*fd.jump(w*theta, n)*fd.avg(Pi)*fd.dS_v
        + fd.inner(w, Up)*g*fd.dx
        )

    if mu: # Newtonian dissipation in vertical
        eqn += mu*fd.inner(w, Up)*fd.inner(u, Up)*fd.dx
    if f: # Coriolis term
        eqn += f*fd.inner(w, fd.cross(Up, u))*fd.dx
    if F: # additional source term
        eqn += fd.inner(w, F)*fd.dx
    return eqn

def get_form_mass():
    def form_mass(u, rho, theta, du, drho, dtheta):
        return u_mass(u, du) + rho_mass(rho, drho) + theta_mass(theta, dtheta)
    return form_mass
    
def get_form_function(n, Up, c_pen,
                      cp, g, R_d, p_0, kappa, mu, f=None, F=None):
    def form_function(u, rho, theta, du, drho, dtheta):
        eqn = theta_tendency(dtheta, u, theta, n, Up, c_pen)
        eqn += rho_tendency(drho, rho, u, n)
        eqn += u_tendency(du, n, u, theta, rho,
                          cp, g, R_d, p_0, kappa, Up, mu, f, F)
        return eqn
    return form_function

def form_viscosity(u, v, kappa, mu = fd.Constant(10.0)):
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

def slice_imr_form(un, unp1, rhon, rhonp1, thetan, thetanp1,
                   du, drho, dtheta,
                   dT, n, Up, c_pen,
                   cp, g, R_d, p_0, kappa, mu=None, f=None, F=None,
                   viscosity=None, diffusivity=None):
    form_mass = get_form_mass()
    form_function = get_form_function(n, Up, c_pen,
                                      cp, g, R_d, p_0, kappa, mu, f, F)

    eqn = form_mass(unp1, rhonp1, thetanp1, du, drho, dtheta)
    eqn -= form_mass(un, rhon, thetan, du, drho, dtheta)
    unph = fd.Constant(0.5)*(un + unp1)
    rhonph = fd.Constant(0.5)*(rhon + rhonp1)
    thetanph = fd.Constant(0.5)*(thetan + thetanp1)
    eqn += dT*form_function(unph, rhonph, thetanph,
                            du, drho, dtheta)

    if viscosity:
        eqn += dT*form_viscosity(unph, du, viscosity)
    if diffusivity:
        eqn += dT*form_viscosity(thetanph, dtheta, diffusivity)

    return eqn
