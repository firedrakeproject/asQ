from firedrake import COMM_WORLD, Ensemble
from pyop2.mpi import MPI, internal_comm
import weakref

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


def split_ensemble(ensemble, split_size, **kwargs):
    """
    Split an Ensemble into multiple smaller Ensembles which share the same
    spatial communicators `ensemble.comm`.

    Each smaller Ensemble returned is defined over a contiguous subset of the
    members of the large Ensemble.

    :arg ensemble: the large Ensemble to split.
    :arg split_size: the number of members in each smaller Ensemble.
    """
    if (ensemble.ensemble_comm.size % split_size) != 0:
        raise ValueError(
            "Ensemble size must be integer multiple of split_size")

    # how many splits in total?
    nsplits = ensemble.ensemble_comm.size // split_size

    # which split are we part of?
    split_index = ensemble.ensemble_comm.rank // split_size

    # create each split ensemble
    for split in range(nsplits):
        # which ranks are in this split?
        offset = split*split_size
        split_members = tuple(offset + i for i in range(split_size))

        maybe_comm = slice_ensemble(ensemble, split_members)
        if split == split_index:
            if maybe_comm == MPI.COMM_NULL:
                raise ValueError(
                    "expected slice_ensemble to return a valid communicator on this rank")
            else:
                ecomm = maybe_comm
        else:
            if maybe_comm != MPI.COMM_NULL:
                raise ValueError(
                    "expected slice_ensemble to return a null communicator on this rank")

    return ecomm


def slice_ensemble(ensemble, split_ranks, **kwargs):
    rank = ensemble.ensemble_comm.rank
    # should we end up in this split?
    color = 1 if rank in split_ranks else MPI.UNDEFINED

    # create split_ensemble.global_comm
    split_comm = ensemble.global_comm.Split(
        color=color, key=ensemble.global_comm.rank)

    if color == 1:
        if split_comm == MPI.COMM_NULL:
            raise ValueError(
                "expected slice_ensemble to return a valid communicator on this rank")
    else:
        if split_comm != MPI.COMM_NULL:
            raise ValueError(
                "expected slice_ensemble to return a null communicator on this rank")
        return split_comm

    split_ensemble = EnsembleConnector(
        split_comm, nmembers=len(split_ranks),
        spatial_comm=ensemble.comm, **kwargs)

    weakref.finalize(split_ensemble, split_comm.Free)

    return split_ensemble


class EnsembleConnector(Ensemble):
    def __init__(self, global_comm, *, nmembers=None,
                 spatial_comm=None, **kwargs):
        """
        An Ensemble created from provided spatial communicators (ensemble.comm).

        :arg global_comm: global communicator the Ensemble is defined over.
        :arg nmembers: number of Ensemble members (ensemble.ensemble_comm.size).
        :arg spatial_comm: communicator to use for the Ensemble.comm member.
        """
        if not isinstance(nmembers, int):
            raise TypeError(
                "nmembers must be an integer")
        if spatial_comm is None:
            raise TypeError(
                "spatial_comm must be an MPI Comm")
        if nmembers*spatial_comm.size != global_comm.size:
            raise ValueError(
                "The global ensemble must have the same number of ranks as the sum of the local comms")

        ensemble_name = kwargs.get("ensemble_name", "Ensemble")
        self.global_comm = global_comm
        self._comm = internal_comm(self.global_comm, self)

        self.comm = spatial_comm
        self.comm.name = f"{ensemble_name} spatial comm"
        self._spatial_comm = internal_comm(self.comm, self)

        self.ensemble_comm = self.global_comm.Split(color=self.comm.rank,
                                                    key=global_comm.rank)
        self.ensemble_comm.name = f"{ensemble_name} ensemble comm"
        self._ensemble_comm = internal_comm(self.ensemble_comm, self)
        weakref.finalize(self, self.ensemble_comm.Free)
