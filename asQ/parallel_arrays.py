from pyop2.mpi import MPI
from numpy import zeros as zero_array


class DistributedArray(object):
    def __init__(self, local_size, dtype=None, comm=MPI.COMM_WORLD):
        '''
        1D array distributed over an MPI comm.

        Each rank has a copy of the entire array of size comm.size*size_per_rank.
        Each rank owns the array slice [comm.rank*local_size:(comm.rank+1)*local_size].
        Provides method for synchronising the array over the comm and testing if an element is owned by the current rank.

        :arg local_size: number of elements owned by each rank.
        :arg dtype: datatype, defaults to numpy default dtype.
        :arg comm: MPI communicator the array is distributed over.
        '''
        self.comm = comm
        self.local_size = local_size
        self.global_size = comm.size*local_size
        self.rank = comm.rank

        self._data = zero_array(self.global_size, dtype=dtype)
        self.offset = self.rank*self.local_size

        self.dglobal = self._GlobalAccessor(self)
        self.dlocal = self._LocalAccessor(self)

    class _GlobalAccessor(object):
        '''
        Manages access by global addressing
        '''
        def __init__(self, parent):
            self.parent = parent

        def __getitem__(self, i):
            return self.parent._data[i]

        def __setitem__(self, i, val):
            i = i % self.parent.global_size
            if not self.parent.is_owned(i):
                raise IndexError(f"Global index {i} is not on rank {self.parent.rank}")
            self.parent._data[i] = val

    class _LocalAccessor(object):
        '''
        Manages access by local addressing
        '''
        def __init__(self, parent):
            self.parent = parent

        def _validate_index(self, i):
            i = i % self.parent.local_size
            global_index = self.parent.offset + i
            if not self.parent.is_owned(global_index):
                raise IndexError(f"Local index {i} is not owned by rank {self.parent.rank}")
            return global_index

        def __getitem__(self, i):
            global_index = self._validate_index(i)
            return self.parent._data[global_index]

        def __setitem__(self, i, val):
            global_index = self._validate_index(i)
            self.parent._data[global_index] = val

    def is_owned(self, i):
        '''
        Is the index i owned by this time rank?
        '''
        return self.offset <= i < self.offset + self.local_size

    def synchronise(self):
        """
        Synchronise the array over the comm

        Until this method is called, array elements not owned by the current rank are not guaranteed to be vald
        """
        self.comm.Allgather(MPI.IN_PLACE, self._data)
