from asQ.parallel_arrays import DistributedDataLayout1D


class TimePartitionMixin(object):
    """
    Mixin class for all-at-once types related to a timeseries
    distributed over the ranks of an Ensemble communicator.

    Provides the following member variables
    layout: a DistributedDataLayout describing the partition over the ensemble.
    ensemble: the time-parallel ensemble.
    time_partition: a list of integers for the number of timesteps stored on each ensemble rank.
    time_rank: the ensemble rank of the current process.
    nlocal_timesteps: the number of timesteps on the current ensemble member.
    ntimesteps: the number of timesteps in the entire time-series.
    """
    def __init__(self):
        pass

    def time_partition_setup(self, ensemble, time_partition):
        """
        Sets the provided member variables. This is not implemented in the
        the __init__ method to accomodate child classes which will be
        instantiated by PETSc (and hence we have no control over the values
        passed to __init__).

        :arg ensemble: the time-parallel ensemble communicator.
        :arg time_partition: a list of integers for the number of timesteps stored on each ensemble rank.
        """
        self.layout = DistributedDataLayout1D(time_partition, ensemble.ensemble_comm)
        self.ensemble = ensemble
        self.time_partition = self.layout.partition
        self.time_rank = ensemble.ensemble_comm.rank
        self.nlocal_timesteps = self.layout.local_size
        self.ntimesteps = self.layout.global_size
