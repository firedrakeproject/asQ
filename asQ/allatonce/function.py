import firedrake as fd
from firedrake.petsc import PETSc
from functools import reduce
from operator import mul
import contextlib

from asQ.profiling import memprofile
from asQ.parallel_arrays import in_range
from asQ.allatonce.mixin import TimePartitionMixin

__all__ = ['time_average', 'AllAtOnceFunction']


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
        for uo, uc in zip(uouts, aaofunc.get_field_components(i)):
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
    @memprofile
    def __init__(self, ensemble, time_partition, function_space):
        """
        A function representing multiple timesteps of a time-dependent finite-element problem,
        i.e. the solution to an all-at-once system.

        :arg ensemble: time-parallel ensemble communicator.
        :arg time_partition: a list of integers for the number of timesteps stored on each ensemble rank.
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

    def transform_index(self, i, cpt=None, from_range='slice', to_range='slice'):
        '''
        Shift timestep index or component index from one range to another,
        and account for pythonic -ve indices.

        For example, if there are 3 ensemble ranks with time_partition=(2, 3, 2), then:
            window index 0 is slice index 0 on ensemble rank 0
            window index 1 is slice index 1 on ensemble rank 0
            window index 2 is slice index 0 on ensemble rank 1
            window index 3 is slice index 1 on ensemble rank 1
            window index 4 is slice index 2 on ensemble rank 1
            window index 5 is slice index 0 on ensemble rank 2
            window index 6 is slice index 1 on ensemble rank 2

        If cpt is None, shifts from one timestep range to another. If cpt is not None,
        returns index in flattened all-at-once function of component cpt in timestep i.
        This is only different from the value returned if cpt is None if a single timestep
        is defined on a MixedFunctionSpace.

        Raises IndexError if original or shifted index is out of bounds.

        :arg i: timestep index to shift.
        :arg cpt: None or component index in timestep i to shift.
        :arg from_range: range of i. Either slice or window.
        :arg to_range: range to shift i to. Either 'slice' or 'window'.
        '''
        idxtypes = {'slice': 'l', 'window': 'g'}

        if from_range not in idxtypes:
            raise ValueError("from_range must be "+" or ".join(idxtypes.keys()))
        if to_range not in idxtypes:
            raise ValueError("to_range must be "+" or ".join(idxtypes.keys()))

        i = self.layout.transform_index(i, itype=idxtypes[from_range], rtype=idxtypes[to_range])

        if cpt is None:
            return i
        else:  # cpt is not None:
            in_range(cpt, self.ncomponents, throws=True)
            cpt = cpt % self.ncomponents
            return i*self.ncomponents + cpt

    @PETSc.Log.EventDecorator()
    def set_component(self, step, cpt, usrc, index_range='slice', funcs=None):
        '''
        Set component of solution at a timestep to new value.

        :arg step: index of timestep.
        :arg cpt: index of component.
        :arg usrc: new solution for component cpt of timestep step.
        :arg index_range: is index in window or slice?
        :arg funcs: an indexable of the all-at-once function to set component in.
            If None, self.function.subfunctions is used.
        '''
        # index of component in all at once function
        aao_index = self.transform_index(step, cpt, from_range=index_range, to_range='slice')

        if funcs is None:
            funcs = self.function.subfunctions
        funcs[aao_index].assign(usrc)

    @PETSc.Log.EventDecorator()
    def get_component(self, step, cpt, uout=None, index_range='slice', funcs=None, name=None, deepcopy=False):
        '''
        Get component of solution at a timestep.

        :arg step: index of timestep.
        :arg cpt: index of component.
        :arg index_range: is timestep index in window or slice?
        :arg uout: Function to place value of component in (if None then component is returned).
        :arg name: name of returned function if deepcopy=True.
            Ignored if uout is not None or deepcopy=False.
        :arg funcs: an indexable of the all-at-once function to get component from.
            If None, self.function.subfunctions is used
        :arg deepcopy: if True, new function is returned. If False, handle to component
            of funcs is returned. Ignored if uout is not None.
        '''
        # index of component in all at once function
        aao_index = self.transform_index(step, cpt, from_range=index_range, to_range='slice')

        if funcs is None:
            funcs = self.function.subfunctions

        # required component
        uget = funcs[aao_index]

        if uout is not None:
            uout.assign(uget)
            return uout

        if deepcopy is False:
            return uget
        else:  # deepcopy is True
            ureturn = fd.Function(self.field_function_space.sub(cpt), name=name)
            ureturn.assign(uget)
            return ureturn

    def get_field_components(self, step, index_range='slice', funcs=None):
        '''
        Get tuple of the components of the all-at-once function for a timestep.

        This is equivalent to u.subfunctions if u was the Function for a single timestep.
        If the argument funcs is firedrake.split(self.function) then this is equivalent
        to firedrake.split(u) if u was the Function for a single timestep.

        :arg step: index of timestep.
        :arg index_range: is index in window or slice?
        :arg funcs: an indexable of the all-at-once function to get components from.
            If None, self.function.subfunctions is used.
        '''
        return tuple(self.get_component(step, cpt, index_range=index_range, funcs=funcs)
                     for cpt in range(self.ncomponents))

    @PETSc.Log.EventDecorator()
    def set_field(self, step, usrc, index_range='slice', funcs=None):
        '''
        Set solution at a timestep to new value.

        :arg step: index of timestep to set.
        :arg usrc: new solution for timestep.
        :arg index_range: is index in window or slice?
        :arg funcs: an indexable of the all-at-once function to set timestep in.
            If None, self.function.subfunctions is used.
        '''
        for cpt in range(self.ncomponents):
            self.set_component(step, cpt, usrc.subfunctions[cpt],
                               index_range=index_range, funcs=funcs)

    @PETSc.Log.EventDecorator()
    def get_field(self, step, uout=None, index_range='slice', name=None, funcs=None):
        '''
        Get solution at a timestep.

        :arg step: index of timestep to get.
        :arg index_range: is index in window or slice?
        :arg uout: function to set to value of timestep (timestep returned if None).
        :arg name: name of returned function. Ignored if uout is not None.
        :arg funcs: an indexable of the all-at-once function to get timestep from.
            If None, self.function.subfunctions is used.
        '''
        if uout is None:
            uget = fd.Function(self.field_function_space, name=name)
        else:
            uget = uout

        ucpts = self.get_field_components(step, index_range=index_range, funcs=funcs)

        for ug, uc in zip(uget.subfunctions, ucpts):
            ug.assign(uc)

        return uget

    @PETSc.Log.EventDecorator()
    def bcast_field(self, step, u):
        """
        Broadcast solution at given timestep to all time-ranks.

        :arg step: window index of field to broadcast.
        :arg u: fd.Function to place field into.
        """
        # find which rank step is on.
        root = self.layout.rank_of(step)

        # get u if step on this rank
        if self.time_rank == root:
            self.get_field(step, uout=u, index_range='window')

        # bcast u
        self.ensemble.bcast(u, root=root)

        return u

    @PETSc.Log.EventDecorator()
    def update_time_halos(self, blocking=True):
        '''
        Update uprev with the last step from the previous time slice (periodic).

        :arg blocking: Whether to use blocking MPI communications.
            If False then a list of MPI requests is returned.
        '''
        # sending last timestep on current slice to next slice
        self.get_field(-1, uout=self.unext, index_range='slice')

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

    @PETSc.Log.EventDecorator()
    @memprofile
    def assign(self, src, update_halos=True, blocking=True):
        """
        Set value of AllAtOnceFunction from another object.

        :arg src: object to set value from. Can be one of: Either another AllAtOnceFunction or a
            - AllAtOnceFunction: assign all values from src.
            - PETSc Vec: assign self.function from src via self.global_vec.
            - firedrake.Function: assign initial condition and all timesteps from src.
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
                    self.set_field(i, src, index_range='slice')
                self.initial_condition.assign(src)
                if not update_halos:
                    self.uprev.assign(src)
                    self.unext.assign(src)
            elif src.function_space() == self.function_space:
                self.function.assign(src)

        else:
            raise TypeError(f"src value must be AllAtOnceFunction or PETSc.Vec or field Function, not {type(src)}")

        if update_halos:
            return self.update_time_halos(blocking=blocking)

    @PETSc.Log.EventDecorator()
    @memprofile
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

    @contextlib.contextmanager
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
