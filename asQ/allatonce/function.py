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
    over either entire window or current slice.

    :arg aaofunc: AllAtOnceFunction to average.
    :arg uout: Function to save average into.
    :arg uwrk: Function to use as working buffer.
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
        The all-at-once system representing multiple timesteps of a time-dependent finite-element problem.

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

        self._vec = self._aao_vec()

    def _aao_vec(self):
        """
        Return a PETSc Vec representing the all-at-once function over the global comm
        """
        X = PETSc.Vec().create(comm=self.ensemble.global_comm)
        X.setSizes((self.nlocal_dofs, self.nglobal_dofs))
        X.setFromOptions()
        return X

    def copy(self):
        """
        Return a deep copy of the AllAtOnceFunction.
        """
        new = AllAtOnceFunction(self.ensemble, self.time_partition,
                                self.field_function_space)
        new.assign(self)
        return new

    def transform_index(self, i, cpt=None, from_range='slice', to_range='slice'):
        '''
        Shift timestep or component index from one range to another, and accounts for -ve indices.

        For example, if there are 3 ensemble ranks, each owning two timesteps, then:
            window index 0 is slice index 0 on ensemble rank 0
            window index 1 is slice index 1 on ensemble rank 0
            window index 2 is slice index 0 on ensemble rank 1
            window index 3 is slice index 1 on ensemble rank 1
            window index 4 is slice index 0 on ensemble rank 2
            window index 5 is slice index 1 on ensemble rank 2

        If cpt is None, shifts from one timestep range to another. If cpt is not None, returns index in all-at-once function of component cpt in timestep i.
        Throws IndexError if original or shifted index is out of bounds.

        :arg i: timestep index to shift.
        :arg cpt: None or component index in timestep i to shift.
        :arg from_range: range of i. Either slice or window.
        :arg to_range: range to shift i to. Either slice or window. Ignored if cpt is not None.
        '''
        if from_range == 'component' or to_range == 'component':
            raise ValueError("from_range and to_range apply to the timestep index and cannot be 'component'")

        idxtypes = {'slice': 'l', 'window': 'g'}

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
        Set component of solution at a timestep to new value

        :arg step: index of timestep
        :arg cpt: index of component
        :arg usrc: new solution for timestep
        :arg index_range: is index in window or slice?
        :arg funcs: an indexable of the (slice-local) all-at-once function to set timestep in.
            If None, self.function.subfunctions is used
        '''
        # index of component in all at once function
        aao_index = self.transform_index(step, cpt, from_range=index_range, to_range='slice')

        if funcs is None:
            funcs = self.function.subfunctions
        funcs[aao_index].assign(usrc)

    @PETSc.Log.EventDecorator()
    def get_component(self, step, cpt, uout=None, index_range='slice', funcs=None, name=None, deepcopy=False):
        '''
        Get component of solution at a timestep

        :arg step: index of timestep to get
        :arg cpt: index of component
        :arg index_range: is timestep index in window or slice?
        :arg uout: Function to set to component (component returned if None)
        :arg name: name of returned function if deepcopy=True.
            Ignored if uout is not None or deepcopy=False.
        :arg funcs: an all-at-once function to get timestep from.
            If None, self.function.subfunctions is used
        :arg deepcopy: if True, new function is returned. If false, handle to component
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

        :arg step: index of timestep.
        :arg index_range: is index in window or slice?
        :arg funcs: an all-at-once function to get timestep from. If None, self.w_alls is used
        '''
        if funcs is None:
            funcs = self.function.subfunctions

        return tuple(self.get_component(step, cpt, funcs=funcs)
                     for cpt in range(self.ncomponents))

    @PETSc.Log.EventDecorator()
    def set_field(self, step, usrc, index_range='slice', funcs=None):
        '''
        Set solution at a timestep to new value

        :arg step: index of timestep to set.
        :arg usrc: new solution for timestep
        :arg index_range: is index in window or slice?
        :arg funcs: an all-at-once function to set timestep in.
            If None, self.function.subfunctions is used
        '''
        for cpt in range(self.ncomponents):
            self.set_component(step, cpt, usrc.sub(cpt),
                               index_range=index_range, funcs=funcs)

    @PETSc.Log.EventDecorator()
    def get_field(self, step, uout=None, index_range='slice', name=None, funcs=None):
        '''
        Get solution at a timestep

        :arg step: index of timestep to set.
        :arg index_range: is index in window or slice?
        :arg uout: function to set to timestep (timestep returned if None)
        :arg name: name of returned function. Ignored if uout is not None
        :arg funcs: an all-at-once function to get timestep from.
            If None, self.function.subfunctions is used
        '''
        if uout is None:
            uget = fd.Function(self.field_function_space, name=name)
        else:
            uget = uout

        for cpt in range(self.ncomponents):
            ucpt = self.get_component(step, cpt, index_range=index_range, funcs=funcs)
            uget.sub(cpt).assign(ucpt)

        return uget

    @PETSc.Log.EventDecorator()
    def set_all_fields(self, usrc=None, index=None):
        """
        Set solution at ics and all timesteps using either provided
        Function or current value of given timestep.

        :arg usrc: solution to set all timesteps to.
        :arg index: index of timestep to use current solution for new value.
                    Must be in window range. Ignored if usrc is not None.
                    NotImplemented yet.
        """
        if usrc is not None:  # use given function
            self.initial_condition.assign(usrc)

        else:  # last rank broadcasts final timestep
            if index is not None:
                raise ValueError("Setting AllAtOnceFunction from given index not implemented yet")

            end_rank = self.ensemble.ensemble_comm.size - 1

            if self.time_rank == end_rank:
                self.get_field(-1, uout=self.initial_condition, index_range='window')

            self.ensemble.bcast(self.initial_condition, root=end_rank)

        # persistence forecast
        for i in range(self.nlocal_timesteps):
            self.set_field(i, self.initial_condition, index_range='slice')

        self.uprev.assign(self.initial_condition)

        return

    @PETSc.Log.EventDecorator()
    def update_time_halos(self, blocking=True):
        '''
        Update uprev with the last step from the previous slice (periodic) of walls

        :arg blocking: Whether to use blocking MPI communications. If False then a list of MPI requests is returned
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

    @PETSc.Log.EventDecorator()
    @memprofile
    def assign(self, src, update_halos=True, blocking=True):
        """
        Set value of function from another AllAtOnceFunction or PETSc Vec.
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

        else:
            raise TypeError(f"src value must be AllAtOnceFunction or PETSc.Vec, not {type(src)}")

        if update_halos:
            return self.update_time_halos(blocking=blocking)

    @contextlib.contextmanager
    def global_vec(self):
        """
        Context manager for the global PETSc Vec with read/write access.
        """
        with self.function.dat.vec as fvec:
            self._vec.placeArray(fvec.array)
            yield self._vec
            self._vec.resetArray()

    @contextlib.contextmanager
    def global_vec_ro(self):
        """
        Context manager for the global PETSc Vec with read only access.
        """
        with self.function.dat.vec_ro as fvec:
            self._vec.placeArray(fvec.array)
            yield self._vec
            self._vec.resetArray()

    @contextlib.contextmanager
    def global_vec_wo(self):
        """
        Context manager for the global PETSc Vec with write only access.
        """
        with self.function.dat.vec_wo as fvec:
            self._vec.placeArray(fvec.array)
            yield self._vec
            self._vec.resetArray()
