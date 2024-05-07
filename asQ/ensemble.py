from firedrake import COMM_WORLD, Ensemble
from pyop2.mpi import MPI, internal_comm, is_pyop2_comm, PyOP2CommError

__all__ = ['create_ensemble', 'split_ensemble']


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
        msg = "Ensemble size must be integer multiple of split_size"
        raise ValueError(msg)

    # which split are we part of?
    split_rank = ensemble.ensemble_comm.rank // split_size

    # create split_ensemble.global_comm
    split_global_comm = ensemble.global_comm.Split(color=split_rank,
                                                   key=ensemble.global_comm.rank)

    # create split_ensemble.ensemble_comm
    split_ensemble_comm = ensemble.ensemble_comm.Split(color=split_rank,
                                                       key=ensemble.global_comm.rank)

    return ManualEnsemble(split_global_comm, ensemble.comm, split_ensemble_comm, **kwargs)


class ManualEnsemble(Ensemble):
    def __init__(self, global_comm, spatial_comm, ensemble_comm, **kwargs):
        """
        An Ensemble created from provided comms.

        :arg global_comm: global communicator the Ensemble is defined over.
        :arg spatial_comm: communicator to use for the Ensemble.comm member.
        :arg ensemble_comm: communicator to use for the Ensemble.ensemble_comm member.

        The global_comm, spatial_comm, and ensemble_comm must have the same logical meaning
        as they do in firedrake.Ensemble. i.e. the global_comm is the union of a cartesian
        product of multiple spatial_comms and ensemble_comms.
            - ManualEnsemble is logically defined over all ranks in global_comm.
            - Each rank in global_comm belongs to only one spatial_comm and one ensemble_comm.
            - The size of the intersection of any (spatial_comm, ensemble_comm) pair is 1.

        WARNING: Not meeting these requirements may produce in errors, hangs, and nonsensical results.
        """
        # are we handed user comms?

        for comm in (global_comm, spatial_comm, ensemble_comm):
            if is_pyop2_comm(comm):
                raise PyOP2CommError("Cannot construct Ensemble from PyOP2 internal comm")

        # check cartesian product consistency

        if spatial_comm.size*ensemble_comm.size != global_comm.size:
            msg = "The global comm must have the same number of ranks as the product of spatial and ensemble comms"
            raise PyOP2CommError(msg)

        global_group = global_comm.Get_group()
        spatial_group = spatial_comm.Get_group()
        ensemble_group = ensemble_comm.Get_group()

        if MPI.Group.Intersection(spatial_group, ensemble_group).size != 1:
            raise PyOP2CommError("spatial and ensemble comms must be cartesian product in global_comm")
        if MPI.Group.Intersection(global_group, spatial_group).size != spatial_group.size:
            raise PyOP2CommError("spatial_comm must be subgroup of global_comm")
        if MPI.Group.Intersection(global_group, ensemble_group).size != ensemble_group.size:
            raise PyOP2CommError("ensemble_comm must be subgroup of global_comm")

        # create internal duplicates and name comms for debugging
        ensemble_name = kwargs.get("name", "Ensemble")

        self.global_comm = global_comm
        if not hasattr(self.global_comm, "name"):
            self.global_comm.name = f"{ensemble_name} global comm"
        self._comm = internal_comm(self.global_comm, self)

        self.comm = spatial_comm
        if not hasattr(self.comm, "name"):
            self.comm.name = f"{ensemble_name} spatial comm"
        self._spatial_comm = internal_comm(self.comm, self)

        self.ensemble_comm = ensemble_comm
        if not hasattr(self.ensemble_comm, "name"):
            self.ensemble_comm.name = f"{ensemble_name} ensemble comm"
        self._ensemble_comm = internal_comm(self.ensemble_comm, self)
