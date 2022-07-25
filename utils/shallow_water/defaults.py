import firedrake as fd
from utils.planets import earth
from utils import mg


def default_degree():
    '''
    Default degree of the finite element space for the depth field.
    Finite element space for the velocity field has degree one higher.
    '''
    return 1


def default_velocity_function_space(mesh, degree=default_degree()):
    '''
    Default finite element function space for the velocity field: BDM

    :arg mesh: mesh that the function space is defined over
    :arg degree; the degree of the depth space, velocity space has degree one higher
    '''
    return fd.FunctionSpace(mesh, "BDM", degree+1)


def default_depth_function_space(mesh, degree=default_degree()):
    '''
    Default finite element function space for the depth field: DG

    :arg mesh: mesh that the function space is defined over
    :arg degree; the degree of the depth space
    '''
    return fd.FunctionSpace(mesh, "DG", degree)


def default_function_space(mesh, degree=default_degree()):
    '''
    Default finite element function space for the depth field: BDM*DG

    :arg mesh: mesh that the function space is defined over
    :arg degree; the degree of the depth space
    '''
    V1 = default_velocity_function_space(mesh, degree=degree)
    V2 = default_depth_function_space(mesh, degree=degree)
    return V1*V2


def earth_coriolis_expression(x, y, z):
    return 2*earth.Omega*z/earth.Radius


def default_icosahedral_refinement():
    return 5


def create_mg_globe_mesh(comm=fd.COMM_WORLD,
                         base_level=1,
                         ref_level=default_icosahedral_refinement(),
                         coords_degree=default_degree()+2,
                         radius=earth.radius):
    '''
    Create icosahedral sphere mesh with a multigrid heirarchy

    :arg comm: MPI communicator the mesh is defined over
    :arg base_level: refinement level of the coarsest mesh
    :arg ref_level: refinement level of the finest mesh
    :arg coords_degree: degree of the coordinates
    :arg radius: radius of the sphere
    '''
    distribution_parameters = {
        "partition": True,
        "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)
    }
    return mg.icosahedral_mesh(R0=radius,
                               base_level=base_level,
                               degree=coords_degree,
                               distribution_parameters=distribution_parameters,
                               nrefs=ref_level-base_level,
                               comm=comm)
