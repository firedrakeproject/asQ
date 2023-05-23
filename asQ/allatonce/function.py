import firedrake as fd
from firedrake.petsc import PETSc
from functools import reduce
from operator import mul

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

        self.ncomponents = len(self.field_function_space.split())

        self.function = fd.Function(self.function_space)
        self.initial_condition = fd.Function(self.field_function_space)

        # functions containing the last step of the previous
        # and current slice for parallel communication
        self.uprev = fd.Function(self.field_function_space)
        self.unext = fd.Function(self.field_function_space)

        self.vec = self._aao_vec()

    def _aao_vec(self):
        """
        Return a PETSc Vec representing the all-at-once function over the global comm
        """
        nlocal_space_dofs = self.field_function_space.node_set.size
        nspace_dofs = self.field_function_space.dim()
        nlocal = self.nlocal_timesteps*nlocal_space_dofs  # local times x local space
        nglobal = self.ntimesteps*nspace_dofs  # global times x global space

        X = PETSc.Vec().create(comm=self.ensemble.global_comm)
        X.setSizes((nlocal, nglobal))
        X.setFromOptions()

        return X

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
    def set_component(self, step, cpt, unew, index_range='slice'):
        '''
        Set component of solution at a timestep to new value

        :arg step: index of timestep
        :arg cpt: index of component
        :arg unew: new solution for timestep
        :arg index_range: is index in window or slice?
        '''
        # index of component in all at once function
        aao_index = self.transform_index(step, cpt=cpt, from_range=index_range, to_range='slice')
        self.function.subfunctions[aao_index].assign(unew)

    @PETSc.Log.EventDecorator()
    def get_component(self, step, cpt, index_range='slice', uout=None, name=None, deepcopy=False):
        '''
        Get component of solution at a timestep

        :arg step: index of timestep to get
        :arg cpt: index of component
        :arg index_range: is timestep index in window or slice?
        :arg uout: function to set to component (component returned if None)
        :arg name: name of returned function if deepcopy=True. Ignored if uout is not None
        :arg deepcopy: if True, new function is returned. If false, handle to component of f_alls is returned. Ignored if uout is not None
        '''
        # index of component in all at once function
        aao_index = self.transform_index(step, cpt=cpt, from_range=index_range, to_range='slice')

        # required component
        wget = self.function.subfunctions[aao_index]

        if uout is not None:
            uout.assign(wget)
            return uout

        if deepcopy is False:
            return wget
        else:  # deepcopy is True
            wreturn = fd.Function(self.function_space.sub(cpt), name=name)
            wreturn.assign(wget)
            return wreturn

    def get_field_components(self, step, index_range='slice'):
        '''
        Get tuple of the components of the all-at-once function for a timestep.

        :arg step: index of timestep.
        :arg index_range: is index in window or slice?
        '''
        return tuple(self.get_component(step, cpt, index_range=index_range)
                     for cpt in range(self.ncomponents))

    @PETSc.Log.EventDecorator()
    def set_field(self, step, unew, index_range='slice'):
        '''
        Set solution at a timestep to new value

        :arg step: index of timestep to set.
        :arg unew: new solution for timestep
        :arg index_range: is index in window or slice?
        '''
        for cpt in range(self.ncomponents):
            self.set_component(step, cpt, unew.sub(cpt),
                               index_range=index_range)

    @PETSc.Log.EventDecorator()
    def get_field(self, step, index_range='slice', uout=None, name=None):
        '''
        Get solution at a timestep

        :arg step: index of timestep to set.
        :arg index_range: is index in window or slice?
        :arg uout: function to set to timestep (timestep returned if None)
        :arg name: name of returned function. Ignored if uout is not None
        '''
        if uout is None:
            wget = fd.Function(self.function_space, name=name)
        else:
            wget = uout

        for cpt in range(self.ncomponents):
            wcpt = self.get_component(step, cpt, index_range=index_range)
            wget.subfunctions[cpt].assign(wcpt)

        return wget

    @PETSc.Log.EventDecorator()
    def set_all_fields(self, unew=None, index=None):
        """
        Set solution at ics and all timesteps using either provided
        Function or current value of given timestep.
        If both unew and index are None, value of last timestep will be used.

        :arg unew: initial solution for next time-window.
        :arg index: index of timestep to use current solution for new value.
                    Must be in window range. Ignored if unew is not None.
                    NotImplemented yet.
        """
        if unew is not None:  # use given function
            self.initial_condition.assign(unew)

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
    @memprofile
    def assign(self, new, update_halos=True, blocking=True, sync=False):
        """
        Set value of function from another AllAtOnceFunction or PETSc Vec.
        """
        if isinstance(new, AllAtOnceFunction):
            dst_funcs = [self.function, self.initial_condition]
            src_funcs = [new.function, new.initial_condition]
            # these buffers just will be overwritten if the halos are updated
            if not update_halos:
                dst_funcs.extend([self.uprev, self.unext])
                src_funcs.extend([new.uprev, new.unext])
            for dst, src in zip(dst_funcs, src_funcs):
                dst.assign(src)

        elif isinstance(new, PETSc.Vec):
            with self.function.dat.vec_wo as v:
                new.copy(v)

        else:
            raise TypeError(f"new value must be AllAtOnceFunction or PETSc.Vec, not {type(new)}")

        if sync:
            self.sync_vec()

        if update_halos:
            return self.update_time_halos(blocking=blocking)

    @PETSc.Log.EventDecorator()
    def sync_vec(self):
        '''
        Update the PETSc Vec with the values in the Function.
        '''
        with self.function.dat.vec_ro as fvec:
            fvec.copy(self.vec)

    @PETSc.Log.EventDecorator()
    def sync_function(self, update_halos=True, blocking=True):
        '''
        Update the Function with the values in the PETSc Vec

        :arg update_halos: If True, self.uprev is updated as well as self.function
        '''
        with self.function.dat.vec_wo as fvec:
            self.vec.copy(fvec)

        if update_halos:
            return self.update_time_halos(blocking=blocking)

    @PETSc.Log.EventDecorator()
    def update_time_halos(self, blocking=True):
        '''
        Update wrecv with the last step from the previous slice (periodic) of walls

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
