import pytest
from mpi4py import MPI
from asQ.parallel_arrays import in_range, DistributedDataLayout1D, SharedArray, OwnedArray
from numpy import allclose

partitions = [3, (2, 3, 4, 2)]


def test_in_range():
    length = 5
    inside_range = length - 1
    outside_range = length + 1

    # check inside +ve range
    assert in_range(inside_range, length), "length-1 should be in range"

    # check inside -ve range
    assert in_range(-inside_range, length), "negative indices should work"

    # check outside +ve range
    assert not in_range(outside_range, length), "length+1 should be in range"

    with pytest.raises(IndexError):
        in_range(outside_range, length, throws=True), "failures should raise error if requested"

    # check outside -ve range
    assert not in_range(-outside_range, length), "negative indices should work"

    with pytest.raises(IndexError):
        in_range(-outside_range, length, throws=True), "failures should raise error if requested"

    # reject -ve indices
    assert not in_range(-inside_range, length, allow_negative=False), "Fail if negative indices not allowed"

    with pytest.raises(IndexError):
        in_range(-inside_range, length, allow_negative=False, throws=True), "failures should raise if requested"


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize("partition", partitions)
def test_distributed_data_layout(partition):
    comm = MPI.COMM_WORLD
    layout = DistributedDataLayout1D(partition, comm=comm)

    if isinstance(partition, int):
        partition = tuple(partition for _ in range(layout.comm.size))

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
        pos_shift = layout.transform_index(i, itype=itype, rtype=itype)
        assert (pos_shift == i), "positive index should be unchanged if index type unchanged"

        # -ve index changed to +ve
        neg_shift = layout.transform_index(-i, itype=itype, rtype=itype)
        assert (neg_shift == imax - i), "negative index should be changed to equivalent positive index if index type unchanged"

    # local address -> global address

    ilocal = 1

    # +ve index in local range
    iglobal = layout.transform_index(ilocal, itype='l', rtype='g')
    assert (iglobal == offset + ilocal), "local to global index should offset by ilocal"

    # -ve index in local range
    iglobal = layout.transform_index(-ilocal, itype='l', rtype='g')
    assert (iglobal == offset + nlocal - ilocal), "negative local to global should offset from end of local range"

    ilocal = nlocal + 1

    # +ve index out of local range
    with pytest.raises(IndexError):
        iglobal = layout.transform_index(ilocal, itype='l', rtype='g'), "reject local index out of local range"

    # -ve index out of local range
    with pytest.raises(IndexError):
        iglobal = layout.transform_index(-ilocal, itype='l', rtype='g'), "reject local index input if out of local range"

    # global address -> local address

    # +ve index in range
    iglobal = offset + 1
    ilocal = layout.transform_index(iglobal, itype='g', rtype='l')
    assert (ilocal == 1), "local index should be global index minus offset"

    assert layout.is_local(iglobal), "True if iglobal-offset < nlocal"

    # -ve index in range
    iglobal = -sum(partition) + offset + 1
    ilocal = layout.transform_index(iglobal, itype='g', rtype='l')
    assert (ilocal == 1), "accept negative global indices in range"

    assert layout.is_local(iglobal), "accept negative global indices in range"

    # +ve index out of local range
    iglobal = (offset + nlocal + 1) % nglobal
    with pytest.raises(IndexError):
        ilocal = layout.transform_index(iglobal, itype='g', rtype='l'), "reject local index output if global index out of local range"

    assert not layout.is_local(iglobal), "reject local index output if global index out of local range"

    with pytest.raises(IndexError):
        layout.is_local(iglobal, throws=True), "reject local index output if global index out of local range"

    # -ve index out of local range
    iglobal = -(nglobal - (offset + nlocal))
    with pytest.raises(IndexError):
        ilocal = layout.transform_index(iglobal, itype='g', rtype='l'), "reject local index output if global index out of local range"

    assert not layout.is_local(iglobal), "reject local index output if global index out of local range"

    with pytest.raises(IndexError):
        layout.is_local(iglobal, throws=True), "reject local index output if global index out of local range"


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize("partition", partitions)
def test_distributed_data_layout_rankof(partition):
    comm = MPI.COMM_WORLD
    layout = DistributedDataLayout1D(partition, comm=comm)

    if isinstance(partition, int):
        partition = tuple(partition for _ in range(layout.comm.size))

    rank = layout.rank

    ranks = []
    for i in range(layout.global_size):
        for rank in range(layout.nranks):
            begin = sum(layout.partition[:rank])
            end = sum(layout.partition[:rank+1])
            if begin <= i < end:
                ranks.append(rank)
                break
    assert len(ranks) == layout.global_size

    for i in range(layout.global_size):
        assert ranks[i] == layout.rank_of(i), "correctly identify the rank owning index i"


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize("partition", partitions)
def test_shared_array(partition):
    comm = MPI.COMM_WORLD
    rank = comm.rank

    array = SharedArray(partition=partition, dtype=int, comm=comm)

    if isinstance(partition, int):
        partition = tuple(partition for _ in range(comm.size))

    local_size = partition[rank]
    offset = sum(partition[:rank])

    # check everything is zero to start
    for i in range(array.global_size):
        assert array.dglobal[i] == 0, "initial data should be zero"

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
            assert array.dglobal[j] == check, "Each rank should see all data after synchronise"

    # each rank sets its own elements using global access
    for i in range(local_size):
        j = array.offset + i
        array.dglobal[j] = (array.rank + 1)*2

    array.synchronise()

    # check all elements are correct
    for i in range(local_size):
        assert array.dlocal[i] == (array.rank + 1)*2, "Each rank should see all data after synchronise"

    for rank in range(array.comm.size):
        check = (rank + 1)*2
        offset = sum(partition[:rank])
        for i in range(partition[rank]):
            j = offset + i
            assert array.dglobal[j] == check, "Each rank should see all data after synchronise"

    # test copying data
    copy = array.data()
    assert copy is not array._data, "Copy should default to deepcopy of data"

    for i in range(array.global_size):
        assert copy[i] == array.dglobal[i], "Copy should copy values"

    dat = array.data(deepcopy=False)
    assert dat is array._data, "Shallow copy data if requested"

    # test synchronise to one rank
    for i in range(local_size):
        array.dlocal[i] = array.rank + 1

    root = 1
    before = array.data()
    array.synchronise(root=root)
    after = array.data()

    if array.rank == root:
        for rank in range(array.comm.size):
            check = rank + 1
            offset = sum(partition[:rank])
            for i in range(partition[rank]):
                j = offset + i
                assert array.dglobal[j] == check, "Data should be updated on root process after synchronisation"
    else:
        assert allclose(before, after), "Data should be unchanged for non-root processes after synchronisation"


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize("partition", partitions)
def test_shared_array_manager(partition):
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if isinstance(partition, int):
        partition = [partition for _ in range(comm.size)]

    local_size = partition[rank]

    array = SharedArray(partition=partition, dtype=int, comm=comm)

    # try to set a globally addressed element we don't own
    bad_global_index = array.offset-1

    with pytest.raises(IndexError):
        array.dglobal[bad_global_index] = 1, "Don't allow setting data we don't own via global accessor"

    # try to set a locally addressed element we don't own
    bad_local_index = local_size

    with pytest.raises(IndexError):
        array.dlocal[bad_local_index] = 1, "Don't allow setting data we don't own via local accessor"

    # try to get a locally addressed element we don't own
    with pytest.raises(IndexError):
        x = array.dlocal[bad_local_index], "Don't allow accessing data we don't own via the local accessor"  # noqa: F841 unused variable


@pytest.mark.parallel(nprocs=4)
def test_owned_array():
    size = 8
    comm = MPI.COMM_WORLD
    owner = 0

    array = OwnedArray(size, dtype=int, comm=comm, owner=owner)

    assert array.size == size
    assert array.owner == owner
    assert array.comm == comm
    assert array.rank == comm.rank

    if comm.rank == owner:
        assert array.is_owner(), "is_owner() should only return true for owning rank"
    else:
        assert not array.is_owner(), "is_owner() should only return true for owning rank"

    # initialise data
    for i in range(size):
        if array.is_owner():
            array[i] = 2*(i+1)
        else:
            assert array[i] == 0, "data should be initialised to zero"

    array.synchronise()

    # check data
    for i in range(size):
        assert array[i] == 2*(i+1), "data should be the same on all ranks after synchronisation"

    # only owner can modify
    if not array.is_owner():
        with pytest.raises(IndexError):
            array[0] = 0, "non-owning ranks are not allowed to modify data"

    # resize
    new_size = 2*size
    array.resize(new_size)

    assert array.size == new_size, "resizing should resize"

    # check original data is unmodified
    for i in range(size):
        assert array[i] == 2*(i+1), "data within new size should not be changed"

    # initialise new data
    for i in range(new_size):
        if array.is_owner():
            array[i] = 10*(i-5)

    array.synchronise()

    # check new data
    for i in range(new_size):
        assert array[i] == 10*(i-5), "synchronisation should still work after resizing"

    # test copying data
    copy = array.data()
    assert copy is not array._data, "copying should be a deepcopy"

    for i in range(array.size):
        assert copy[i] == array[i], "copying should result in the same values"

    dat = array.data(deepcopy=False)
    assert dat is array._data, "should allow access to underlying buffer if requested"
