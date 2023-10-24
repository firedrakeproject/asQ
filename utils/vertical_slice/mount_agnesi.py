import firedrake as fd
from utils.compressible_flow import hydrostatic_rho
from utils.misc import function_maximum


def L(hydrostatic=False):
    """
    Length of the domain
    """
    if hydrostatic:
        return fd.Constant(240e3)
    else:
        return fd.Constant(144e3)


def H(hydrostatic=False):
    """
    Height of the domain
    """
    if hydrostatic:
        return fd.Constant(50e3)
    else:
        return fd.Constant(35e3)


def width(hydrostatic=False):
    """
    Width of the mountain
    """
    if hydrostatic:
        return fd.Constant(1e3)
    else:
        return fd.Constant(10e3)


def mesh(comm, ncolumns, nlayers,
         hydrostatic=False,
         distribution_parameters=None,
         smoother_profile=False):
    """
    Extruded mesh for Mount Agnesi
    """
    if distribution_parameters is None:
        distribution_parameters = {
            "partition": True,
            "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)
        }

    # domain dimensions
    L0 = L(hydrostatic)
    H0 = H(hydrostatic)

    # surface mesh of ground
    base_mesh = fd.PeriodicIntervalMesh(
        ncolumns, L0,
        distribution_parameters=distribution_parameters,
        comm=comm)

    # volume mesh of the slice
    msh = fd.ExtrudedMesh(base_mesh,
                          layers=nlayers,
                          layer_height=float(H0/nlayers))
    x, z = fd.SpatialCoordinate(msh)

    # making a mountain out of a molehill
    a = width(hydrostatic)
    xc = L0/2.
    hm = 1.
    zs = hm*a**2/((x-xc)**2 + a**2)

    if smoother_profile:
        zh = 5000.
        xexpr = fd.as_vector([x, fd.conditional(z < zh, z + fd.cos(0.5*fd.pi*z/zh)**6*zs, z)])
    else:
        xexpr = fd.as_vector([x, z + ((H0-z)/H0)*zs])
    msh.coordinates.interpolate(xexpr)

    return msh


def Tsurf(hydrostatic=False):
    """
    Surface temperature
    """
    if hydrostatic:
        return fd.Constant(250.)
    else:
        return fd.Constant(300.)


def initial_conditions(mesh, W, Vv, gas, hydrostatic=False):
    x, z = fd.SpatialCoordinate(mesh)
    V2 = W.subfunctions[1]
    Un = fd.Function(W)
    up = fd.as_vector([fd.Constant(0.0), fd.Constant(1.0)])  # up direction

    if not hydrostatic:
        thetab = Tsurf(hydrostatic)*fd.exp(gas.N**2*z/gas.g)

        un, rhon, thetan = Un.subfunctions
        un.project(fd.as_vector([10.0, 0.0]))
        thetan.interpolate(thetab)
        rhon.assign(1.0e-5)

        Pi = fd.Function(V2)

        hydrostatic_rho(Vv, V2, mesh, thetan, rhon,
                        pi_boundary=fd.Constant(0.02),
                        gas=gas, Up=up, top=True, Pi=Pi)
        p0 = function_maximum(Pi)

        hydrostatic_rho(Vv, V2, mesh, thetan, rhon,
                        pi_boundary=fd.Constant(0.05),
                        gas=gas, Up=up, top=True, Pi=Pi)
        p1 = function_maximum(Pi)
        alpha = 2.*(p1-p0)
        beta = p1-alpha
        pi_top = (1.-beta)/alpha

        hydrostatic_rho(Vv, V2, mesh, thetan, rhon,
                        pi_boundary=fd.Constant(pi_top),
                        gas=gas, Up=up, top=True)
    else:
        raise ValueError("Hydrostatic initial conditions not provided yet")
    return Un


def sponge_layer(mesh, V2, dt, hydrostatic=False):
    x, z = fd.SpatialCoordinate(mesh)
    H0 = H(hydrostatic)
    if hydrostatic:
        zb = H0 - 20e3
        mudt = 0.3
    else:
        zb = H0 - 10e3
        mudt = 0.15

    mubar = fd.Constant(mudt/dt)
    mu_top = fd.conditional(z <= zb, 0.0,
                            mubar*fd.sin(fd.Constant(fd.pi/2.)*(z-zb)/(H0-zb))**2)
    mu = fd.Function(V2).interpolate(mu_top)
    return mu


def boundary_conditions(W):
    zv = fd.as_vector([fd.Constant(0.), fd.Constant(0.)])
    bcs = [fd.DirichletBC(W.sub(0), zv, "bottom"),
           fd.DirichletBC(W.sub(0), zv, "top")]
    return bcs
