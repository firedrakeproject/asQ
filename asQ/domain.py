from firedrake import Constant
from asQ.parallel_arrays import DistributedDataLayout1D


class SpaceTimeDomain:
    def __init__(self, mesh, dt, time_partition, ensemble, t0=None):
        self._ensemble = ensemble
        self._mesh = mesh
        self._dt = dt
        self._layout = DistributedDataLayout1D(
            time_partition, ensemble.ensemble_comm)

        self._t0 = t0 or Constant(0)
        self._times = [
            self.t0 + self.local_to_global_index(i)*self.dt
            for i in range(len(self.nlocal_steps))
        ]

    @property
    def ensemble(self):
        return self._ensemble

    @property
    def mesh(self):
        return self._mesh

    @property
    def layout(self):
        return self._layout

    @property
    def time_rank(self):
        return self.ensemble.ensemble_rank

    @property
    def time_partition(self):
        return self.layout.partition

    @property
    def nlocal_steps(self):
        return self.layout.local_size

    @property
    def nsteps(self):
        return self.layout.global_size

    def local_to_global_index(self, i):
        return self.layout.transform_index(i, itype='l', rtype='g')

    def global_to_local_index(self, i):
        return self.layout.transform_index(i, itype='g', rtype='l')

    @property
    def t0(self):
        return self._t0

    @property
    def dt(self):
        return self._dt

    def times(self, i):
        return self._times[i]
