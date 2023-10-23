from firedrake import Ensemble
from mpi4py import MPI


def print_in_order(comm, string):
    for i in range(comm.size):
        if comm.rank == i:
            print(string)
        comm.Barrier()


def ensemble_square(nchunks, chunk_length, comm):
    """
    Split a comm into a square of Ensembles.

    First, `comm` is split to create `nchunks` equal sized Ensembles, each
    with `chunk_length` members.
    Then, `chunk_length` Ensembles with `nchunks` members are created
    which connect the members of the original ensembles with the same
    ensemble_comm.rank.
    The size of each ensemble member is calculated implicity from the size of `comm`.

    The first and second Ensembles that the local rank is a member of are returned.
    """
    big_world = comm
    big_rank = big_world.rank
    big_size = big_world.size

    # create ensembles for fine propogators

    nfine_ensembles = nchunks
    fine_size = big_size//nfine_ensembles
    assert fine_size*nfine_ensembles == big_size

    fine_world = big_world.Split(color=(big_rank//fine_size), key=big_rank)

    assert fine_world.size == fine_size

    fine_ensemble_size = chunk_length
    spatial_size = fine_size // fine_ensemble_size
    assert spatial_size*fine_ensemble_size == fine_size

    fine_ensemble = Ensemble(fine_world, spatial_size)
    fine_ensemble_rank = fine_ensemble.ensemble_comm.rank
    spatial_rank = fine_ensemble.comm.rank

    # create ensembles for coarse propogator

    ncoarse_ensembles = fine_ensemble_size
    coarse_size = big_size//ncoarse_ensembles
    assert coarse_size*ncoarse_ensembles == big_size

    coarse_world = big_world.Split(color=fine_ensemble_rank, key=big_rank)

    assert coarse_world.size == coarse_size

    coarse_ensemble_size = nchunks
    assert spatial_size*coarse_ensemble_size == coarse_size

    coarse_ensemble = Ensemble(coarse_world, spatial_size)
    assert coarse_ensemble.comm.rank == spatial_rank

    return fine_ensemble, coarse_ensemble


# print out rank layouts

nchunks = 3
chunk_length = 4
big_world = MPI.COMM_WORLD

fine_ensemble, coarse_ensemble = ensemble_square(nchunks,
                                                 chunk_length,
                                                 big_world)
big_rank = big_world.rank

fine_rank = fine_ensemble.global_comm.rank
fine_ensemble_rank = fine_ensemble.ensemble_comm.rank

coarse_rank = coarse_ensemble.global_comm.rank
coarse_ensemble_rank = coarse_ensemble.ensemble_comm.rank

spatial_rank = fine_ensemble.comm.rank
assert coarse_ensemble.comm.rank == spatial_rank

if big_rank == 0:
    print("big_rank | fine_rank | fine_ensemble_rank | coarse_rank | coarse_ensemble_rank | spatial_rank")

msg = f"{str(big_rank).ljust(3)} | " \
    + f"{str(fine_rank).ljust(3)} | " \
    + f"{str(fine_ensemble_rank).ljust(3)} | " \
    + f"{str(coarse_rank).ljust(3)} | " \
    + f"{str(coarse_ensemble_rank).ljust(3)} | " \
    + f"{str(spatial_rank).ljust(3)}"

print_in_order(big_world, msg)
