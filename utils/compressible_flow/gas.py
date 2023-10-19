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


def pi_formula(rho, theta, gas=None, R_d=None, p_0=None, kappa=None):
    if gas is None:
        if any(x is None for x in (R_d, p_0, kappa)):
            raise ValueError("R_d, p_0, and kappa must be specified if GasProperties not given")
    else:
        R_d = gas.R_d
        p_0 = gas.p_0
        kappa = gas.kappa
    return (rho * R_d * theta / p_0) ** (kappa / (1 - kappa))


def rho_formula(pi, theta, gas=None, R_d=None, p_0=None, kappa=None):
    if gas is None:
        if any(x is None for x in (R_d, p_0, kappa)):
            raise ValueError("R_d, p_0, and kappa must be specified if GasProperties not given")
    else:
        R_d = gas.R_d
        p_0 = gas.p_0
        kappa = gas.kappa
    return p_0*pi**((1-kappa)/kappa)/R_d/theta
