from firedrake import COMM_WORLD, Ensemble
from pyop2.mpi import internal_comm, decref

__all__ = ['create_ensemble',
           'split_ensemble', 'EnsembleConnector']


def create_ensemble(time_partition, comm=COMM_WORLD):
    '''
    Create an Ensemble for the given slice partition.
    Checks that the number of slices and the size of the communicator are compatible.

    :arg time_partition: a list of integers, the number of timesteps on each time-rank
    :arg comm: the global communicator for the ensemble
    '''
    nslices = 1 if type(time_partition) is int else len(time_partition)
    nranks = comm.size

    if nranks % nslices != 0:
        raise ValueError("Number of time slices must be exact factor of number of MPI ranks")

    nspatial_domains = nranks//nslices

    return Ensemble(comm, nspatial_domains)


def multi_ensemble(nchunks, chunk_length, comm=COMM_WORLD):
    global_comm = comm
    global_size = global_comm.size
    global_rank = global_comm.rank

    total_timesteps = nchunks*chunk_length

    if (global_size % total_timesteps) != 0:
        raise ValueError("Total number of time steps must be exact factor of number of MPI ranks")

    # number of ranks per timestep and per chunk
    spatial_size = global_size // total_timesteps
    chunk_size = spatial_size * chunk_length

    # create global ensemble
    global_ensemble = Ensemble(global_comm, spatial_size)
    assert global_ensemble.ensemble_comm.size == total_timesteps

    # create comm for local chunk
    chunk_id = global_rank // chunk_size
    chunk_comm = global_comm.Split(color=chunk_id, key=global_rank)
    assert chunk_comm.size == chunk_size

    # create ensemble for local chunk
    chunk_ensemble = Ensemble(chunk_comm, spatial_size)
    assert chunk_ensemble.ensemble_comm.size == chunk_length

    return global_ensemble, chunk_ensemble


def split_ensemble(ensemble, partition, split_size):
    # we just need to work out how many members of the global ensemble
    # needed to get `split_size` timesteps on each slice ensemble
    if ensemble.ensemble_comm.size != len(partition):
        msg = "partition must have the same number of members as the ensemble"
        raise ValueError(msg)

    if (sum(partition) % split_size) != 0:
        msg = "total number of timesteps must be integer multiple of split_size"
        raise ValueError(msg)

    if len(set(partition)) != 1:
        msg = "split_ensemble only implemented for balanced partitions yet"
        raise ValueError(msg)

    part = partition[0]

    # number of members of each split ensemble
    nmembers = split_size // part

    # which split are we part of?
    split_rank = ensemble.ensemble_comm.rank // nmembers

    # create split_ensemble.global_comm
    split_comm = ensemble.global_comm.Split(color=split_rank,
                                            key=ensemble.global_comm.rank)

    return EnsembleConnector(split_comm, ensemble.comm, nmembers)


class EnsembleConnector(Ensemble):
    def __init__(self, global_comm, local_comm, nmembers):
        assert nmembers*local_comm.size == global_comm.size

        self.global_comm = global_comm
        self._global_comm = internal_comm(self.global_comm)

        self.comm = local_comm
        self._comm = internal_comm(self.comm)

        self.ensemble_comm = self.global_comm.Split(color=self.comm.rank,
                                                    key=global_comm.rank)

        self._ensemble_comm = internal_comm(self.ensemble_comm)

    def __del__(self):
        if hasattr(self, "ensemble_comm"):
            self.ensemble_comm.Free()
            del self.ensemble_comm
        for comm_name in ["_global_comm", "_comm", "_ensemble_comm"]:
            if hasattr(self, comm_name):
                comm = getattr(self, comm_name)
                decref(comm)
