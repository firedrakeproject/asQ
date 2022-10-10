import pytest
# from pyop2.mpi import MPI
from asQ import DistributedArray


@pytest.mark.parallel(nprocs=4)
def test_distributed_array():
    array = DistributedArray(local_size=4, dtype=int)

    # check everything is zero to start
    for i in range(array.global_size):
        assert array.dglobal[i] == 0

    # each rank sets its own elements
    for i in range(array.local_size):
        array.dlocal[i] = array.rank + 1

    # synchronise
    array.synchronise()

    # check all elements are correct
    for rank in range(array.comm.size):
        check = rank + 1
        offset = rank*array.local_size
        for i in range(array.local_size):
            j = offset + i
            assert array.dglobal[j] == check
