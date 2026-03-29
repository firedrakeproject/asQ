from functools import cached_property
from firedrake import EnsembleFunction, EnsembleCofunction, Function
from asQ.allatonce.function_space import AllAtOnceFunctionSpace, AllAtOnceDualSpace

__all__ = ['AllAtOnceFunction', 'AllAtOnceCofunction']


class AllAtOnceFunctionBase:
    @cached_property
    def initial_condition(self):
        return Function(self.function_space().timestep_function_space)

    @cached_property
    def uprev(self):
        return Function(self.function_space().timestep_function_space)

    @cached_property
    def unext(self):
        return Function(self.function_space().timestep_function_space)

    def __getitem__(self, i):
        return self.subfunctions[i]

    def update_time_halos(self, blocking=True):
        ensemble = self.function_space().domain.ensemble
        rank = ensemble.ensemble_rank
        size = ensemble.ensemble_size

        src = (rank - 1) % size
        dst = (rank + 1) % size

        sendrecv = ensemble.sendrecv if blocking else ensemble.isendrecv

        self.unext.assign(self[-1])

        return sendrecv(fsend=self.unext, dest=dst, sendtag=rank,
                        frecv=self.uprev, source=src, recvtag=src)

    def bcast_timestep(self, i, u=None):
        u = u or Function(
            self.function_space.timestep_function_space)

        domain = self.function_space().domain
        root = domain.layout.rank_of(i)

        if root == domain.time_rank:
            u.assign(self[self.global_to_local_index[i]])

        return self.ensemble.bcast(u, root=root)

    def time_average(self):
        W = self.function_space()
        local_average = Function(W.timestep_function_space)
        nt = W.domain.nsteps
        local_average.assign((1/nt)*sum(self.subfunctions))
        return W.ensemble.allreduce(local_average)


class AllAtOnceFunction(AllAtOnceFunctionBase, EnsembleFunction):
    def __new__(cls, function_space):
        if isinstance(function_space, AllAtOnceFunctionSpace):
            return super().__new__(cls)
        elif isinstance(function_space, AllAtOnceDualSpace):
            return AllAtOnceCofunction(function_space)
        else:
            raise TypeError(
                "function_space must be an AllAtOnceFunctionSpace or"
                f" AllAtOnceDualSpace not a {type(function_space).__name__}")


class AllAtOnceCofunction(AllAtOnceFunctionBase, EnsembleCofunction):
    def __init__(self, function_space):
        if not isinstance(function_space, AllAtOnceDualSpace):
            raise TypeError(
                "function_space must be an AllAtOnceDualSpace"
                f" not a {type(function_space).__name__}")
        super().__init__(function_space)
