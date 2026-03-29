from ufl.duals import is_primal, is_dual
from firedrake import EnsembleFunctionSpace, EnsembleDualSpace

__all__ = ['AllAtOnceFunctionSpace', 'AllAtOnceDualSpace']


class AllAtOnceFunctionSpace(EnsembleFunctionSpace):
    def __new__(cls, domain, V):
        if is_primal(V):
            return super().__new__(cls)
        elif is_dual(V):
            return AllAtOnceDualSpace(domain, V)
        else:
            raise TypeError(
                "V must be a FunctionSpace or DualSpace"
                f" not a {type(V).__name__}")

    def __init__(self, domain, V):
        self._domain = domain
        self._V = V
        super().__init__(
            [V for _ in range(domain.nlocal_steps)],
            ensemble=domain.ensemble)

    @property
    def domain(self):
        return self._domain

    @property
    def timestep_function_space(self):
        return self._V


class AllAtOnceDualSpace(EnsembleDualSpace):
    def __init__(self, domain, V):
        self._domain = domain
        self._V = V
        super().__init__(
            [V for _ in range(domain.nlocal_steps)],
            ensemble=domain.ensemble)

    @property
    def domain(self):
        return self._domain

    @property
    def timestep_function_space(self):
        return self._V
