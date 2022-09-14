import firedrake as fd
from functools import reduce
from operator import mul


class AllAtOnceSystem(object):
    def __init__(self,
                 ensemble, slice_partition,
                 w0, bcs=[]):
        """
        The all-at-once system representing multiple timesteps of a time-dependent finite-element problem.

        :arg ensemble: time-parallel ensemble communicator.
        :arg slice_partition: a list of integers for the number of timesteps stored on each ensemble rank.
        :arg w0: a Function containing the initial data.
        :arg bcs: a list of DirichletBC boundary conditions on w0.function_space.
        """

        # check that the ensemble communicator is set up correctly
        if isinstance(slice_partition, int):
            slice_partition = [slice_partition]
        nsteps = len(slice_partition)
        ensemble_size = ensemble.ensemble_comm.size
        if nsteps != ensemble_size:
            raise ValueError(f"Number of timesteps {nsteps} must equal size of ensemble communicator {ensemble_size}")

        self.ensemble = ensemble
        self.slice_partition = slice_partition
        self.time_rank = ensemble.ensemble_comm.rank

        self.initial_condition = w0
        self.function_space = w0.function_space()
        self.boundary_conditions = bcs
        self.ncomponents = len(self.function_space.split())

        # function pace for the slice of the all-at-once system on this process
        self.function_space_all = reduce(mul, (self.function_space
                                               for _ in range(self.slice_partition[self.time_rank])))

        self.w_all = fd.Function(self.function_space_all)
        self.w_alls = self.w_all.split()

        # for i in range(self.slice_partition[self.time_rank]):
        #     self.set_timestep(i, self.initial_condition, index_range='slice')

        self.set_boundary_conditions()
        for bc in self.boundary_conditions_all:
            bc.apply(self.w_all)

    def set_boundary_conditions(self):
        """
        Set the boundary conditions onto each solution in the all-at-once system
        """
        is_mixed_element = isinstance(self.function_space.ufl_element(), fd.MixedElement)

        self.boundary_conditions_all = []
        for bc in self.boundary_conditions:
            for rank in range(self.slice_partition[self.time_rank]):
                if is_mixed_element:
                    i = bc.function_space().index
                    index = rank*self.ncomponents + i
                else:
                    index = rank
                all_bc = fd.DirichletBC(self.function_space_all.sub(index),
                                        bc.function_arg,
                                        bc.sub_domain)
                self.boundary_conditions_all.append(all_bc)

    def check_index(self, i, index_range='slice'):
        '''
        Check that timestep index is in range
        :arg i: timestep index to check
        :arg index_range: range that index is in. Either slice or window
        '''
        # set valid range
        if index_range == 'slice':
            maxidx = self.slice_partition[self.time_rank]
        elif index_range == 'window':
            maxidx = sum(self.slice_partition)
        else:
            raise ValueError("index_range must be one of 'window' or 'slice'")

        # allow for pythonic negative indices
        minidx = -maxidx

        if not (minidx <= i < maxidx):
            raise ValueError(f"index {i} outside {index_range} range {maxidx}")

    def shift_index(self, i, from_range='slice', to_range='slice'):
        '''
        Shift timestep index from one range to another, and accounts for -ve indices
        :arg i: timestep index to shift
        :arg from_range: range of i. Either slice or window
        :arg to_range: range to shift i to. Either slice or window
        '''
        self.check_index(i, index_range=from_range)

        # deal with -ve indices
        if from_range == 'slice':
            maxidx = self.slice_partition[self.time_rank]
        elif from_range == 'window':
            maxidx = sum(self.slice_partition)

        i = i % maxidx

        # no shift needed
        if to_range == from_range:
            return i

        # index of first timestep in slice
        index0 = sum(self.slice_partition[:self.time_rank])

        if to_range == 'slice':  # 'from_range' == 'window'
            i -= index0

        if to_range == 'window':  # 'from_range' == 'slice'
            i += index0

        self.check_index(i, index_range=to_range)

        return i

    def set_timestep(self, step, wnew, index_range='slice', f_alls=None):
        '''
        Set solution at a timestep to new value

        :arg step: index of timestep to set.
        :arg wnew: new solution for timestep
        :arg index_range: is index in window or slice?
        :arg f_alls: an all-at-once function to set timestep in. If None, self.w_alls is used
        '''

        step_local = self.shift_index(step, from_range=index_range, to_range='slice')

        if f_alls is None:
            f_alls = self.w_alls

        # index of first component of this step
        index0 = self.ncomponents*step_local

        for k in range(self.ncomponents):
            f_alls[index0+k].assign(wnew.sub(k))

    def get_timestep(self, step, index_range='slice', wout=None, name=None, f_alls=None):
        '''
        Get solution at a timestep to new value

        :arg step: index of timestep to set.
        :arg index_range: is index in window or slice?
        :arg wout: function to set to timestep (timestep returned if None)
        :arg name: name of returned function
        :arg f_alls: an all-at-once function to get timestep from. If None, self.w_alls is used
        '''

        step_local = self.shift_index(step, from_range=index_range, to_range='slice')

        if f_alls is None:
            f_alls = self.w_alls

        # where to put timestep?
        if wout is None:
            if name is None:
                wreturn = fd.Function(self.function_space)
            else:
                wreturn = fd.Function(self.function_space, name=name)
            wget = wreturn
        else:
            wget = wout

        # index of first component of this step
        index0 = self.ncomponents*step_local

        for k in range(self.ncomponents):
            wget.sub(k).assign(f_alls[index0+k])

        if wout is None:
            return wreturn
