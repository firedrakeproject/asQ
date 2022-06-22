import firedrake as fd


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


def default_funcion_space(mesh, degree=default_degree()):
    '''
    Default finite element function space for the depth field: BDM*DG

    :arg mesh: mesh that the function space is defined over
    :arg degree; the degree of the depth space
    '''
    V1 = default_velocity_function_space(mesh, degree=degree)
    V2 = default_depth_function_space(mesh, degree=degree)
    return V1*V2
