from firedrake import COMM_WORLD, Ensemble
from pyop2.mpi import internal_comm, decref

__all__ = ['create_ensemble', 'split_ensemble', 'EnsembleConnector']


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


def split_ensemble(ensemble, split_size):
    """
    Split an Ensemble into multiple smaller Ensembles which share the same
    spatial communicators `ensemble.comm`.

    Each smaller Ensemble returned is defined over a contiguous subset of the
    members of the large Ensemble.

    :arg ensemble: the large Ensemble to split.
    :arg split_size: the number of members in each smaller Ensemble.
    """
    if (ensemble.ensemble_comm.size % split_size) != 0:
        msg = "Ensemble size must be integer multiple of split_size"
        raise ValueError(msg)

    # which split are we part of?
    split_rank = ensemble.ensemble_comm.rank // split_size

    # create split_ensemble.global_comm
    split_comm = ensemble.global_comm.Split(color=split_rank,
                                            key=ensemble.global_comm.rank)

    return EnsembleConnector(split_comm, ensemble.comm, split_size)


class EnsembleConnector(Ensemble):
    def __init__(self, global_comm, local_comm, nmembers):
        """
        An Ensemble created from provided spatial communicators (ensemble.comm).

        :arg global_comm: global communicator the Ensemble is defined over.
        :arg local_comm: communicator to use for the Ensemble.comm member.
        :arg nmembers: number of Ensemble members (ensemble.ensemble_comm.size).
        """
        if nmembers*local_comm.size != global_comm.size:
            msg = "The global ensemble must have the same number of ranks as the sum of the local comms"
            raise ValueError(msg)

        self.global_comm = global_comm
        self._comm = internal_comm(self.global_comm, self)

        self.comm = local_comm
        self._spatial_comm = internal_comm(self.comm, self)

        self.ensemble_comm = self.global_comm.Split(color=self.comm.rank,
                                                    key=global_comm.rank)

        self._ensemble_comm = internal_comm(self.ensemble_comm, self)

    def __del__(self):
        if hasattr(self, "ensemble_comm"):
            self.ensemble_comm.Free()
            del self.ensemble_comm
        for comm_name in ["_global_comm", "_comm", "_ensemble_comm"]:
            if hasattr(self, comm_name):
                comm = getattr(self, comm_name)
                decref(comm)
