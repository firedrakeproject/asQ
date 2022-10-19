from pyop2.mpi import MPI
from numpy import zeros as zero_array


def in_range(i, length, allow_negative=True, throws=False):
    '''
    Is the index i within the range of length?
    :arg i: index to check
    :arg length: the number of elements in the range
    :arg allow_negative: is negative indexing allowed?
    :arg throws: if True, an IndexError is raised if the index is out of range
    '''
    if allow_negative:
        result = (-length <= i < length)
    else:
        result = (0 <= i < length)
    if throws and result is False:
        raise IndexError(f"Index {i} is outside the range {length}")
    return result


class DistributedDataLayout1D(object):
    def __init__(self, partition, comm=MPI.COMM_WORLD):
        '''
        A representation of a 1D set of data distributed over several MPI ranks.

        :arg partition: The number of data elements on each rank. Can be a list of integers, in which case len(partition) must be comm.size. Can be a single integer, in which case all ranks have the same number of elements.
        :arg comm: MPI communicator the data is distributed over.
        '''
        if isinstance(partition, int):
            partition = tuple(partition for _ in range(comm.size))
        else:
            if len(partition) != comm.size:
                raise ValueError(f"Partition size {len(partition)} not equal to comm size {comm.size}")
            partition = tuple(partition)
        self.partition = partition
        self.comm = comm
        self.rank = comm.rank
        self.local_size = partition[self.rank]
        self.global_size = sum(partition)
        self.offset = sum(partition[:self.rank])

    def shift_index(self, i, itype='l', rtype='l'):
        '''
        Shift index between local and global addressing, and transform negative indices to their positive equivalent.

        For example if there are 3 ranks each owning two elements then:
            global indices 0,1 are local indices 0,1 on rank 0.
            global indices 2,3 are local indices 0,1 on rank 1.
            global indices 4,5 are local indices 0,1 on rank 2.
        Negative indices are shifted to their positive equivalent:
            local index -1 becomes local index 1.
            global index -2 becomes local index 0 on rank 2.
        Throws IndexError if original or shifted index is out of bounds.

        :arg i: index to shift.
        :arg itype: type of index i. 'l' for local, 'g' for global.
        :arg rtype: type of returned shifted index. 'l' for local, 'g' for global.
        '''
        if itype not in ['l', 'g']:
            raise ValueError(f"itype {itype} must be either 'l' or 'g'")
        if rtype not in ['l', 'g']:
            raise ValueError(f"rtype {rtype} must be either 'l' or 'g'")

        # validate
        sizes = {'l': self.local_size, 'g': self.global_size}
        in_range(i, sizes[itype], throws=True)

        # deal with -ve index
        i = i % sizes[itype]

        # no shift needed
        if itype == rtype:
            return i
        else:
            if itype == 'l':  # rtype == 'g'
                i += self.offset
            elif itype == 'g':  # rtype == 'l'
                i -= self.offset
            in_range(i, sizes[rtype], allow_negative=False, throws=True)
            return i

    def is_local(self, i, throws=False):
        '''
        Is the globally addressed index i owned by this time rank?

        :arg i: globally addressed index.
        :arg throws: if True, raises IndexError if i is outside the global address range
        '''
        try:
            self.shift_index(i, itype='g', rtype='l')
            return True
        except IndexError:
            if throws:
                raise
            else:
                return False


class SharedArray(object):
    def __init__(self, partition, dtype=None, comm=MPI.COMM_WORLD):
        '''
        1D array shared over an MPI comm.

        Each rank has a copy of the entire array of size sum(partition) but can only  modify the partition[comm.rank] section of the array.
        Provides method for synchronising the array over the comm.

        :arg partition: The number of data elements on each rank. Can be a list of integers, in which case len(partition) must be comm.size. Can be a single integer, in which case all ranks have the same number of elements.
        :arg dtype: datatype, defaults to numpy default dtype.
        :arg comm: MPI communicator that the array is distributed over.
        '''
        self.comm = comm
        self.rank = comm.rank
        self.layout = DistributedDataLayout1D(partition, comm=comm)
        self.partition = self.layout.partition
        self.local_size = self.layout.local_size
        self.global_size = self.layout.global_size
        self.offset = self.layout.offset

        self._data = zero_array(self.global_size, dtype=dtype)

        self.dglobal = self._GlobalAccessor(self.layout, self._data)
        self.dlocal = self._LocalAccessor(self.layout, self._data)

    class _GlobalAccessor(object):
        '''
        Manage access by global addressing
        '''
        def __init__(self, layout, data):
            self.layout = layout
            self._data = data

        def __getitem__(self, i):
            return self._data[i]

        def __setitem__(self, i, val):
            self.layout.is_local(i, throws=True)
            self._data[i] = val

    class _LocalAccessor(object):
        '''
        Manage access by local addressing
        '''
        def __init__(self, layout, data):
            self.layout = layout
            self._data = data

        def __getitem__(self, i):
            i = self.layout.shift_index(i, itype='l', rtype='g')
            return self._data[i]

        def __setitem__(self, i, val):
            i = self.layout.shift_index(i, itype='l', rtype='g')
            self._data[i] = val

    def synchronise(self):
        """
        Synchronise the array over the comm.

        Until this method is called, array elements not owned by the current rank are not guaranteed to be valid.
        """
        self.comm.Allgatherv(MPI.IN_PLACE, [self._data, self.partition])


class OwnedArray(object):
    def __init__(self, size, dtype=None, comm=MPI.COMM_WORLD, owner=0):
        '''
        Array owned by one rank but viewed over an MPI comm.

        The array can only be modified by the root rank, but every rank has a copy of the entire array.
        Modifying the array from any rank other than root invalidates the data.

        :arg size: length of the array.
        :arg dtype: datatype, defaults to numpy default dtype.
        :arg comm: MPI communicator the array is synchronised over.
        :arg owner: owning rank.
        '''
        self.size = size
        self.comm = comm
        self.owner = owner
        self.rank = comm.rank

        self._data = zero_array(size, dtype=dtype)

    def is_owner(self):
        '''
        Is the array owned by the current rank?
        '''
        return self.rank == self.owner

    def __getitem__(self, i):
        '''
        Get the value of the element at index i
        '''
        return self._data[i]

    def __setitem__(self, i, val):
        '''
        Set the value of the element at index i to val. Throws if the current rank does not own the array.
        '''
        if not self.is_owner():
            raise IndexError(f"Rank {self.rank} is not the owning rank {self.owner}")
        self._data[i] = val

    def synchronise(self):
        """
        Synchronise the array over the comm

        Until this method is called, array elements on any rank but root are not guaranteed to be valid
        """
        self.comm.Bcast(self._data, root=self.owner)
