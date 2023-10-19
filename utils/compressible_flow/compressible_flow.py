import firedrake as fd


class GasProperties:
    def __init__(self, g, cp, cv, R_d, kappa, N, p_0, T_0):
        """
        :arg g: Gravity
        :arg N: Brunt-Vaisala frequency (1/s)
        :arg cp: SHC of dry air at const. pressure (J/kg/K)
        :arg R_d: Gas constant for dry air (J/kg/K)
        :arg kappa: R_d/c_p
        :arg p_0: reference pressure (Pa, not hPa)
        :arg cv: SHC of dry air at const. volume (J/kg/K)
        :arg T_0: ref. temperature
        :arg cp: SHC of dry air at const. pressure (J/kg/K)
        """
        self.g = fd.Constant(g)
        self.cp = fd.Constant(cp)
        self.cv = fd.Constant(cv)
        self.R_d = fd.Constant(R_d)
        self.kappa = fd.Constant(kappa)
        self.N = fd.Constant(N)
        self.p_0 = fd.Constant(p_0)
        self.T_0 = fd.Constant(T_0)


def StandardAtmosphere(N=0):
    """
    :arg N: Brunt-Vaisala frequency (1/s)
    """
    return GasProperties(N=N, g=9.810616,
                         cp=1004.5, cv=717., R_d=287., kappa=2.0/7.0,
                         p_0=1000.0*100.0, T_0=273.15)


def function_space(mesh, horizontal_degree=1, vertical_degree=1,
                   vertical_velocity_space=False):
    if not mesh.extruded:
        msg = "Compressible Euler function space only defined for extruded meshes"
        raise ValueError(msg)

    dim = sum(mesh.cell_dimension())

    # horizontal base spaces
    if dim == 2:
        S1 = fd.FiniteElement("CG", fd.interval, horizontal_degree+1)
        S2 = fd.FiniteElement("DG", fd.interval, horizontal_degree)
    elif dim == 3:
        S1 = fd.FiniteElement("RTCF", fd.quadrilateral, horizontal_degree+1)
        S2 = fd.FiniteElement("DQ", fd.quadrilateral, horizontal_degree)
    else:
        msg = "Compressible Euler function space only defined for 2 or 3 dimensions"
        raise ValueError(msg)

    # vertical base spaces
    T0 = fd.FiniteElement("CG", fd.interval, vertical_degree+1)
    T1 = fd.FiniteElement("DG", fd.interval, vertical_degree)

    # build spaces V2, V3, Vt
    V2h_elt = fd.HDiv(fd.TensorProductElement(S1, T1))
    V2t_elt = fd.TensorProductElement(S2, T0)
    V3_elt = fd.TensorProductElement(S2, T1)
    V2v_elt = fd.HDiv(V2t_elt)
    V2_elt = V2h_elt + V2v_elt

    V1 = fd.FunctionSpace(mesh, V2_elt, name="Velocity")
    V2 = fd.FunctionSpace(mesh, V3_elt, name="Pressure")
    Vt = fd.FunctionSpace(mesh, V2t_elt, name="Temperature")

    W = V1 * V2 * Vt  # velocity, density, temperature

    if vertical_velocity_space:
        Vv = fd.FunctionSpace(mesh, V2v_elt, name="Vv")
        return W, Vv
    else:
        return W


def velocity_function_space(mesh, horizontal_degree=1, vertical_degree=1):
    return function_space(mesh, horizontal_degree, vertical_degree).subfunctions[0]


def pressure_function_space(mesh, horizontal_degree=1, vertical_degree=1):
    return function_space(mesh, horizontal_degree, vertical_degree).subfunctions[1]


def density_function_space(mesh, horizontal_degree=1, vertical_degree=1):
    return function_space(mesh, horizontal_degree, vertical_degree).subfunctions[1]


def temperature_function_space(mesh, horizontal_degree=1, vertical_degree=1):
    return function_space(mesh, horizontal_degree, vertical_degree).subfunctions[2]
