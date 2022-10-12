import pytest
from pyop2.mpi import MPI
from asQ import DistributedDataLayout, SharedArray


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize("partition", [3, [2, 3, 4, 2]])
def test_distributed_data_layout(partition):
    comm = MPI.COMM_WORLD
    layout = DistributedDataLayout(partition, comm=comm)

    if isinstance(partition, int):
        partition = [partition for _ in range(layout.comm.size)]

    rank = layout.rank
    nlocal = partition[rank]
    nglobal = sum(partition)
    offset = sum(partition[:rank])

    # check attributes
    assert layout.partition == partition
    assert layout.rank == comm.rank
    assert layout.local_size == nlocal
    assert layout.global_size == nglobal
    assert layout.offset == offset

    max_indices = {
        'l': nlocal,
        'g': nglobal
    }

    # shift index: from_range == to_range
    for itype in max_indices.keys():

        imax = max_indices[itype]
        i = imax - 1

        # +ve index unchanged
        pos_shift = layout.shift_index(i, itype=itype, rtype=itype)
        assert (pos_shift == i)

        # -ve index changed to +ve
        neg_shift = layout.shift_index(-i, itype=itype, rtype=itype)
        assert (neg_shift == imax - i)

    # local address -> global address

    ilocal = 1

    # +ve index in local range
    iglobal = layout.shift_index(ilocal, itype='l', rtype='g')
    assert (iglobal == offset + ilocal)

    # -ve index in local range
    iglobal = layout.shift_index(-ilocal, itype='l', rtype='g')
    assert (iglobal == offset + nlocal - ilocal)

    ilocal = nlocal + 1

    # +ve index out of local range
    with pytest.raises(IndexError):
        iglobal = layout.shift_index(ilocal, itype='l', rtype='g')

    # -ve index out of local range
    with pytest.raises(IndexError):
        iglobal = layout.shift_index(-ilocal, itype='l', rtype='g')

    # global address -> local address

    # +ve index in range
    iglobal = offset + 1
    ilocal = layout.shift_index(iglobal, itype='g', rtype='l')
    assert (ilocal == 1)

    assert layout.is_local(iglobal)

    # -ve index in range
    iglobal = -sum(partition) + offset + 1
    ilocal = layout.shift_index(iglobal, itype='g', rtype='l')
    assert (ilocal == 1)

    assert layout.is_local(iglobal)

    # +ve index out of local range
    iglobal = (offset + nlocal + 1) % nglobal
    with pytest.raises(IndexError):
        ilocal = layout.shift_index(iglobal, itype='g', rtype='l')

    assert not layout.is_local(iglobal)

    with pytest.raises(IndexError):
        layout.is_local(iglobal, throws=True)

    # -ve index out of local range
    iglobal = -(nglobal - (offset + nlocal))
    with pytest.raises(IndexError):
        ilocal = layout.shift_index(iglobal, itype='g', rtype='l')

    assert not layout.is_local(iglobal)

    with pytest.raises(IndexError):
        layout.is_local(iglobal, throws=True)


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize("partition", [3, [2, 3, 4, 2]])
def test_shared_array(partition):
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if isinstance(partition, int):
        partition = [partition for _ in range(comm.size)]

    array = SharedArray(partition=partition, dtype=int, comm=comm)

    local_size = partition[rank]
    offset = sum(partition[:rank])

    # check everything is zero to start
    for i in range(array.global_size):
        assert array.dglobal[i] == 0

    # each rank sets its own elements using local access
    for i in range(local_size):
        array.dlocal[i] = array.rank + 1

    # synchronise
    array.synchronise()

    # check all elements are correct
    for rank in range(array.comm.size):
        check = rank + 1
        offset = sum(partition[:rank])
        for i in range(partition[rank]):
            j = offset + i
            assert array.dglobal[j] == check

    # each rank sets its own elements using global access
    for i in range(local_size):
        j = array.offset + i
        array.dglobal[j] = (array.rank + 1)*2

    array.synchronise()

    # check all elements are correct
    for i in range(local_size):
        assert array.dlocal[i] == (array.rank + 1)*2

    for rank in range(array.comm.size):
        check = (rank + 1)*2
        offset = sum(partition[:rank])
        for i in range(partition[rank]):
            j = offset + i
            assert array.dglobal[j] == check


@pytest.mark.skip
@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize("partition", [3])
def test_shared_array_manager(partition):
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if isinstance(partition, int):
        partition = [partition for _ in range(comm.size)]

    # local_size = partition[rank]

    array = SharedArray(partition=partition, dtype=int, comm=comm)

    # try to set a globally addressed element we don't own
    bad_global_index = array.offset-1

    # with pytest.raises(IndexError):
    #     array.dglobal[bad_global_index] = 1

    try:
        array.dglobal[bad_global_index] = 1
        assert False, 'should have thrown'
    except IndexError as err:
        print(f"IndexError on rank {rank}: "+str(err))
        pass
    except Exception as err:
        # print("unidentified error")
        print(f"Exception on rank {rank}: "+str(err))
        pass

    # with pytest.raises(IndexError):
    #     array.dglobal[bad_global_index] = 1

    # # try to set a locally addressed element we don't own
    # bad_local_index = local_size

    # with pytest.raises(IndexError):
    #     array.dlocal[bad_local_index] = 1

    # # try to get a locally addressed element we don't own
    # with pytest.raises(IndexError):
    #     x = array.dlocal[bad_local_index]  # noqa: F841 unused variable
