import firedrake as fd
from firedrake.petsc import PETSc
from pyop2 import MixedDat
from functools import reduce
from operator import mul
import contextlib
from asQ.profiling import profiler
from asQ.allatonce.mixin import TimePartitionMixin

__all__ = ['time_average', 'AllAtOnceFunction']


@profiler()
def time_average(aaofunc, uout, uwrk, average='window'):
    """
    Compute the time average of an all-at-once function
    over either the entire window or the current slice.

    :arg aaofunc: AllAtOnceFunction to average.
    :arg uout: Function to save average into.
    :arg uwrk: Function to use as working buffer.
        TODO: make this optional once Ensemble.allreduce accepts MPI.IN_PLACE.
    :arg average: range of time-average.
        'window': compute over all timesteps in all-at-once function.
        'slice': compute only over timesteps on local ensemble member.
    """
    # accumulate over local slice
    uout.assign(0)
    uouts = uout.subfunctions
    for i in range(aaofunc.nlocal_timesteps):
        for uo, uc in zip(uouts, aaofunc[i].subfunctions):
            uo.assign(uo + uc)

    if average == 'slice':
        nsamples = aaofunc.nlocal_timesteps
        uout /= fd.Constant(nsamples)
    elif average == 'window':
        aaofunc.ensemble.allreduce(uout, uwrk)
        nsamples = aaofunc.ntimesteps
        uout.assign(uwrk/fd.Constant(nsamples))
    else:
        raise ValueError(f"average type must be 'window' or 'slice', not {average}")

    return


class AllAtOnceFunction(TimePartitionMixin):
    @profiler()
    def __init__(self, ensemble, time_partition, function_space):
        """
        A function representing multiple timesteps of a time-dependent finite-element problem,
        i.e. the solution to an all-at-once system.

        :arg ensemble: time-parallel ensemble communicator. The timesteps are partitioned
            over the ensemble members according to time_partition so
            ensemble.ensemble_comm.size == len(time_partition) must be True.
        :arg time_partition: a list of integers for the number of timesteps stored on each
            ensemble rank.
        :arg function_space: a FunctionSpace for the solution at a single timestep.
        """
        self.time_partition_setup(ensemble, time_partition)

        # function space for single timestep
        self.field_function_space = function_space

        # function space for the slice of the all-at-once system on this process
        self.function_space = reduce(mul, (self.field_function_space
                                           for _ in range(self.nlocal_timesteps)))

        self.ncomponents = len(self.field_function_space.subfunctions)

        self.function = fd.Function(self.function_space)
        self.initial_condition = fd.Function(self.field_function_space)

        # Functions to view each timestep
        def field_function(i):
            dats = (self.function.subfunctions[j].dat
                    for j in self._component_indices(i))
            return fd.Function(self.field_function_space,
                               val=MixedDat(dats))

        self._fields = tuple(field_function(i)
                             for i in range(self.nlocal_timesteps))

        # functions containing the last step of the previous
        # and current slice for parallel communication
        self.uprev = fd.Function(self.field_function_space)
        self.unext = fd.Function(self.field_function_space)

        self.nlocal_dofs = self.function_space.node_set.size
        self.nglobal_dofs = self.ntimesteps*self.field_function_space.dim()

        with self.function.dat.vec as fvec:
            sizes = (self.nlocal_dofs, self.nglobal_dofs)
            self._vec = PETSc.Vec().createWithArray(fvec.array,
                                                    size=sizes,
                                                    comm=ensemble.global_comm)
            self._vec.setFromOptions()

    def transform_index(self, i, from_range='slice', to_range='slice'):
        '''
        Shift timestep index from one range to another, and account for pythonic -ve indices.

        For example, if there are 3 ensemble ranks with time_partition=(2, 3, 2), then:
            window index 0 is slice index 0 on ensemble rank 0
            window index 1 is slice index 1 on ensemble rank 0
            window index 2 is slice index 0 on ensemble rank 1
            window index 3 is slice index 1 on ensemble rank 1
            window index 4 is slice index 2 on ensemble rank 1
            window index 5 is slice index 0 on ensemble rank 2
            window index 6 is slice index 1 on ensemble rank 2

        Raises IndexError if original or shifted index is out of bounds.

        :arg i: timestep index to shift.
        :arg from_range: range of i. Either slice or window.
        :arg to_range: range to shift i to. Either 'slice' or 'window'.
        '''
        idxtypes = {'slice': 'l', 'window': 'g'}

        if from_range not in idxtypes:
            raise ValueError("from_range must be "+" or ".join(idxtypes.keys()))
        if to_range not in idxtypes:
            raise ValueError("to_range must be "+" or ".join(idxtypes.keys()))

        i = self.layout.transform_index(i, itype=idxtypes[from_range], rtype=idxtypes[to_range])

        return i

    def _component_indices(self, step, from_range='slice', to_range='slice'):
        '''
        Return indices of the components of a timestep in the all-at-once MixedFunction.

        :arg step: timestep index to get component indices for.
        :arg from_range: range of step. Either slice or window.
        :arg to_range: range to shift the indices to. Either 'slice' or 'window'.
        '''
        step = self.transform_index(step, from_range=from_range, to_range=to_range)
        return tuple(self.ncomponents*step + c
                     for c in range(self.ncomponents))

    @profiler()
    def __getitem__(self, i):
        '''
        Get a Function that is a view over a timestep.

        :arg i: index of timestep to view.
        :arg idx: is index in window or slice?
        '''
        index = i[0] if type(i) is tuple else i
        itype = i[1] if type(i) is tuple else 'slice'
        j = self.transform_index(index, from_range=itype, to_range='slice')
        return self._fields[j]

    @profiler()
    def bcast_field(self, step, u):
        """
        Broadcast solution at given timestep `step` to Function `u` on all time-ranks.

        :arg step: window index of field to broadcast.
        :arg u: fd.Function to place field into.
        """
        # find which rank step is on.
        root = self.layout.rank_of(step)

        # get u if step on this rank
        if self.time_rank == root:
            u.assign(self[step, 'window'])

        # bcast u
        self.ensemble.bcast(u, root=root)

        return u

    @profiler()
    def update_time_halos(self, blocking=True):
        '''
        Update uprev with the last step from the previous time slice (periodic).

        :arg blocking: Whether to use blocking MPI communications.
            If False then a list of MPI requests is returned.
        '''
        # sending last timestep on current slice to next slice
        self.unext.assign(self[-1])

        size = self.ensemble.ensemble_comm.size
        rank = self.ensemble.ensemble_comm.rank

        # ring communication
        dst = (rank+1) % size
        src = (rank-1) % size

        if blocking:
            sendrecv = self.ensemble.sendrecv
        else:
            sendrecv = self.ensemble.isendrecv

        return sendrecv(fsend=self.unext, dest=dst, sendtag=rank,
                        frecv=self.uprev, source=src, recvtag=src)

    @profiler()
    def copy(self, copy_values=True):
        """
        Return a deep copy of the AllAtOnceFunction.

        :arg copy_values: If true, the values of the current AllAtOnceFunction
            will be copied into the new AllAtOnceFunction.
        """
        new = AllAtOnceFunction(self.ensemble, self.time_partition,
                                self.field_function_space)
        if copy_values:
            new.assign(self)
        return new

    @profiler()
    def assign(self, src, update_halos=True, blocking=True):
        """
        Set value of AllAtOnceFunction from another object.

        :arg src: object to set value from. Can be one of:
            - AllAtOnceFunction: assign all values from src.
            - PETSc Vec: assign self.function from src via self.global_vec.
            - firedrake.Function in self.function_space:
                assign timesteps from src.
            - firedrake.Function in self.field_function_space:
                assign initial condition and all timesteps from src.
        :arg update_halos: if True then the time-halos will be updated.
        :arg blocking: if update_halos is True, then this argument determines
            whether blocking communication is used. A list of MPI Requests is returned
            if non-blocking communication is used.
        """
        if isinstance(src, AllAtOnceFunction):
            dst_funcs = [self.function, self.initial_condition]
            src_funcs = [src.function, src.initial_condition]
            # these buffers just will be overwritten if the halos are updated
            if not update_halos:
                dst_funcs.extend([self.uprev, self.unext])
                src_funcs.extend([src.uprev, src.unext])
            for dst, src in zip(dst_funcs, src_funcs):
                dst.assign(src)

        elif isinstance(src, PETSc.Vec):
            with self.global_vec_wo() as gvec:
                src.copy(gvec)

        elif isinstance(src, fd.Function):
            if src.function_space() == self.field_function_space:
                for i in range(self.nlocal_timesteps):
                    self[i].assign(src)
                self.initial_condition.assign(src)
                if not update_halos:
                    self.uprev.assign(src)
                    self.unext.assign(src)
            elif src.function_space() == self.function_space:
                self.function.assign(src)
            else:
                raise ValueError(f"src must be be in the `function_space` {self.function_space}"
                                 + " or `field_function_space` {self.field_function_space} of the"
                                 + " the AllAtOnceFunction, not in {src.function_space}")

        else:
            raise TypeError(f"src value must be AllAtOnceFunction or PETSc.Vec or field Function, not {type(src)}")

        if update_halos:
            return self.update_time_halos(blocking=blocking)

    @profiler()
    def zero(self, subset=None):
        """
        Set all values to zero.

        :arg subset: pyop2.types.set.Subset indicating the nodes to zero.
            If None then the whole function is zeroed.
        """
        funcs = (self.initial_condition, self.function, self.uprev, self.unext)
        for f in funcs:
            f.zero(subset=subset)
        return self

    @profiler()
    def axpy(self, a, x, update_ics=False,
             update_halos=False, blocking=True):
        """
        Compute y = a*x + y where y is this AllAtOnceFunction.

        :arg a: scalar to multiply x.
        :arg x: other object for calculation. Can be one of:
            - AllAtOnceFunction: all timesteps are updated, and optionally the ics.
            - PETSc Vec: all timesteps are updated.
            - firedrake.Function in self.function_space:
                all timesteps are updated.
            - firedrake.Function in self.field_function_space:
                all timesteps are updated, and optionally the ics.
        :arg update_ics: if True then the initial conditions will be updated
            from x as well as the timestep values (if possible).
        :arg update_halos: if True then the time-halos will be updated.
        :arg blocking: if update_halos is True, then this argument determines
            whether blocking communication is used. A list of MPI Requests is returned
            if non-blocking communication is used.
        """
        alpha = fd.Constant(a)

        def axpy(x, y):
            return y.assign(alpha*x + y)

        if isinstance(x, AllAtOnceFunction):
            axpy(x.function, self.function)
            if update_ics:
                axpy(x.initial_condition, self.initial_condition)

        elif isinstance(x, PETSc.Vec):
            with self.global_vec() as gvec:
                gvec.axpy(a, x)

        elif isinstance(x, fd.Function):
            if x.function_space() == self.field_function_space:
                for i in range(self.nlocal_timesteps):
                    axpy(x, self[i])
                if update_ics:
                    axpy(x, self.initial_condition)

            elif x.function_space() == self.function_space:
                axpy(x, self.function)

            else:
                raise ValueError(f"x must be be in the `function_space` {self.function_space}"
                                 + " or `field_function_space` {self.field_function_space} of the"
                                 + " the AllAtOnceFunction, not in {src.function_space}")

        else:
            raise TypeError(f"x value must be AllAtOnceFunction or PETSc.Vec or field Function, not {type(x)}")

        if update_halos:
            return self.update_time_halos(blocking=blocking)

    @contextlib.contextmanager
    @profiler()
    def global_vec(self):
        """
        Context manager for the global PETSc Vec with read/write access.

        It is invalid to access the Vec outside of a context manager.
        """
        # fvec shares the same storage as _vec, so we need this context
        # manager to make sure that the data gets copied to/from the
        # Function.dat storage and _vec.
        with self.function.dat.vec:
            yield self._vec

    @contextlib.contextmanager
    @profiler()
    def global_vec_ro(self):
        """
        Context manager for the global PETSc Vec with read only access.

        It is invalid to access the Vec outside of a context manager.
        """
        # fvec shares the same storage as _vec, so we need this context
        # manager to make sure that the data gets copied into _vec from
        # the Function.dat storage.
        with self.function.dat.vec_ro:
            yield self._vec

    @contextlib.contextmanager
    @profiler()
    def global_vec_wo(self):
        """
        Context manager for the global PETSc Vec with write only access.

        It is invalid to access the Vec outside of a context manager.
        """
        # fvec shares the same storage as _vec, so we need this context
        # manager to make sure that the data gets copied back into the
        # Function.dat storage from _vec.
        with self.function.dat.vec_wo:
            yield self._vec
