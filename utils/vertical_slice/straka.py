import firedrake as fd
from utils.compressible_flow import hydrostatic_rho, pi_formula

L = fd.Constant(51.2e3)
H = fd.Constant(6.4e3)
Tsurf = fd.Constant(300.)


def mesh(comm, ncolumns, nlayers,
         distribution_parameters=None):
    """
    Extruded mesh for Mount Agnesi
    """
    if distribution_parameters is None:
        distribution_parameters = {
            "partition": True,
            "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)
        }

    # surface mesh of ground
    base_mesh = fd.PeriodicIntervalMesh(
        ncolumns, float(L),
        distribution_parameters=distribution_parameters,
        comm=comm)
    base_mesh.coordinates.dat.data[:] -= float(L)/2

    # volume mesh of the slice
    msh = fd.ExtrudedMesh(base_mesh,
                          layers=nlayers,
                          layer_height=float(H/nlayers))
    return msh


def initial_conditions(mesh, W, Vv, gas):
    V1, V2, Vt = W.subfunctions
    Un = fd.Function(W)
    x, z = fd.SpatialCoordinate(mesh)

    # N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
    thetab = Tsurf

    up = fd.as_vector([fd.Constant(0.0), fd.Constant(1.0)])  # up direction

    un, rhon, thetan = Un.subfunctions
    thetan.interpolate(thetab)
    theta_back = fd.Function(Vt).assign(thetan)
    rhon.assign(1.0e-5)

    hydrostatic_rho(Vv, V2, mesh, thetan, rhon,
                    pi_boundary=fd.Constant(1.0),
                    gas=gas, Up=up, top=False)

    x = fd.SpatialCoordinate(mesh)
    xc = 0.
    xr = 4000.
    zc = 3000.
    zr = 2000.
    r = fd.sqrt(((x[0]-xc)/xr)**2 + ((x[1]-zc)/zr)**2)
    T_pert = fd.conditional(r > 1., 0., -7.5*(1.+fd.cos(fd.pi*r)))
    # T = theta*Pi so Delta theta = Delta T/Pi assuming Pi fixed

    Pi_back = pi_formula(rhon, thetan, gas)
    # this keeps perturbation at zero away from bubble
    thetan.project(theta_back + T_pert/Pi_back)
    # save the background stratification for rho
    rho_back = fd.Function(V2).assign(rhon)
    # Compute the new rho
    # using rho*theta = Pi which should be held fixed
    rhon.project(rhon*thetan/theta_back)

    return Un, rho_back, theta_back


def boundary_conditions(W):
    zv = fd.as_vector([fd.Constant(0.), fd.Constant(0.)])
    bcs = [fd.DirichletBC(W.sub(0), zv, "bottom"),
           fd.DirichletBC(W.sub(0), zv, "top")]
    return bcs
