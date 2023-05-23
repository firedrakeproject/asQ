from asQ.parallel_arrays import DistributedDataLayout1D


class TimePartitionMixin(object):
    def __init__(self):
        pass

    def time_partition_setup(self, ensemble, time_partition):
        self.layout = DistributedDataLayout1D(time_partition, ensemble.ensemble_comm)
        self.ensemble = ensemble
        self.time_partition = self.layout.partition
        self.time_rank = ensemble.ensemble_comm.rank
        self.nlocal_timesteps = self.layout.local_size
        self.ntimesteps = self.layout.global_size
