import firedrake as fd
from pyop2.mpi import MPI
from functools import reduce
from operator import mul


class JacobianMatrix(object):
    def __init__(self, aaos):
        r"""
        Python matrix for the Jacobian of the all at once system
        :param aaos: The AllAtOnceSystem object
        """
        self.aaos = aaos
        self.u = fd.Function(self.aaos.function_space_all)  # for the input function
        self.F = fd.Function(self.aaos.function_space_all)  # for the output residual
        self.F_prev = fd.Function(self.aaos.function_space_all)  # Where we compute the
        # part of the output residual from neighbouring contributions
        self.u0 = fd.Function(self.aaos.function_space_all)  # Where we keep the state

        self.Fsingle = fd.Function(self.aaos.function_space)
        self.urecv = fd.Function(self.aaos.function_space)  # will contain the previous time value i.e. 3*r-1
        self.usend = fd.Function(self.aaos.function_space)  # will contain the next time value i.e. 3*(r+1)
        self.ulist = self.u.split()
        self.r = self.aaos.ensemble.ensemble_comm.rank
        self.n = self.aaos.ensemble.ensemble_comm.size
        # Jform missing contributions from the previous step
        # Find u1 s.t. F[u1, u2, u3; v] = 0 for all v
        # definition:
        # dF_{u1}[u1, u2, u3; delta_u, v] =
        #  lim_{eps -> 0} (F[u1+eps*delta_u,u2,u3;v]
        #                  - F[u1,u2,u3;v])/eps
        # Newton, solves for delta_u such that
        # dF_{u1}[u1, u2, u3; delta_u, v] = -F[u1,u2,u3; v], for all v
        # then updates u1 += delta_u
        self.Jform = fd.derivative(self.aaos.para_form, self.aaos.w_all)
        # Jform contributions from the previous step
        self.Jform_prev = fd.derivative(self.aaos.para_form,
                                        self.aaos.w_recv)

    def mult(self, mat, X, Y):
        n = self.n

        # copy the local data from X into self.u
        with self.u.dat.vec_wo as v:
            v.array[:] = X.array_r

        mpi_requests = []
        # Communication stage

        # send
        r = self.aaos.time_rank  # the time rank
        self.aaos.get_timestep(-1, index_range='slice', wout=self.usend, f_alls=self.ulist)

        request_send = self.aaos.ensemble.isend(self.usend, dest=((r+1) % n), tag=r)
        mpi_requests.extend(request_send)

        # receive
        request_recv = self.aaos.ensemble.irecv(self.urecv, source=((r-1) % n), tag=r-1)
        mpi_requests.extend(request_recv)

        # wait for the data [we should really do this after internal
        # assembly but have avoided that for now]
        MPI.Request.Waitall(mpi_requests)

        # Set the flag for the circulant option
        if self.aaos.circ in ["quasi", "picard"]:
            self.aaos.Circ.assign(1.0)
        else:
            self.aaos.Circ.assign(0.0)

        # assembly stage
        fd.assemble(fd.action(self.Jform, self.u), tensor=self.F)
        fd.assemble(fd.action(self.Jform_prev, self.urecv),
                    tensor=self.F_prev)
        self.F += self.F_prev

        # unset flag if alpha-circulant approximation only in Jacobian
        if self.aaos.circ not in ["picard"]:
            self.aaos.Circ.assign(0.0)

        # Apply boundary conditions
        # assumes aaos.w_all contains the current state we are
        # interested in
        # For Jacobian action we should just return the values in X
        # at boundary nodes
        for bc in self.aaos.boundary_conditions_all:
            bc.homogenize()
            bc.apply(self.F, u=self.u)
            bc.restore()

        with self.F.dat.vec_ro as v:
            v.copy(Y)


class AllAtOnceSystem(object):
    def __init__(self,
                 ensemble, slice_partition,
                 dt, theta,
                 form_mass, form_function,
                 w0, bcs=[],
                 circ="", alpha=1e-3):
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

        self.dt = dt
        self.theta = theta

        self.form_mass = form_mass
        self.form_function = form_function

        self.circ = circ
        self.alpha = alpha

        if self.circ == "picard":
            self.Circ = fd.Constant(1.0)
        else:
            self.Circ = fd.Constant(0.0)

        # function pace for the slice of the all-at-once system on this process
        self.function_space_all = reduce(mul, (self.function_space
                                               for _ in range(slice_partition[self.time_rank])))

        self.w_all = fd.Function(self.function_space_all)
        self.w_alls = self.w_all.split()

        for i in range(self.slice_partition[self.time_rank]):
            self.set_timestep(i, self.initial_condition, index_range='slice')

        self.set_boundary_conditions()
        for bc in self.boundary_conditions_all:
            bc.apply(self.w_all)

        # function to assemble the nonlinear residual
        self.F_all = fd.Function(self.function_space_all)

        # functions containing the last and next steps for parallel
        # communication timestep
        # from the previous iteration
        self.w_recv = fd.Function(self.function_space)
        self.w_send = fd.Function(self.function_space)

        self._set_para_form()
        self.jacobian = JacobianMatrix(self)

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

    def for_each_timestep(self, callback):
        '''
        call callback for each timestep in each slice in the current window
        callback arguments are: timestep index in window, timestep index in slice, Function at timestep

        :arg callback: the function to call for each timestep
        '''

        w = fd.Function(self.function_space)
        for slice_index in range(self.slice_partition[self.time_rank]):
            window_index = self.shift_index(slice_index,
                                            from_range='slice',
                                            to_range='window')
            self.get_timestep(slice_index, wout=w, index_range='slice')
            callback(window_index, slice_index, w)

    def next_window(self, w1=None):
        """
        Reset all-at-once-system ready for next time-window

        :arg w1: initial solution for next time-window.If None,
                 will use the final timestep from previous window
        """
        rank = self.time_rank
        ncomm = self.ensemble.ensemble_comm.size

        if w1 is not None:  # use given function
            self.initial_condition.assign(w1)
        else:  # last rank broadcasts final timestep
            if rank == ncomm-1:
                # index of start of final timestep
                self.get_timestep(-1, wout=self.initial_condition, index_range='slice')

            with self.initial_condition.dat.vec as vec:
                self.ensemble.ensemble_comm.Bcast(vec.array, root=ncomm-1)

        # persistence forecast
        for i in range(self.slice_partition[rank]):
            self.set_timestep(i, self.initial_condition, index_range='slice')

        return

    def update(self, X):
        '''
        Update self.w_alls and self.w_recv from PETSc Vec X.
        The local parts of X are copied into self.w_alls
        and the last step from the previous slice (periodic)
        is copied into self.u_prev
        '''

        with self.w_all.dat.vec_wo as v:
            v.array[:] = X.array_r

        n = self.ensemble.ensemble_comm.size
        r = self.time_rank

        mpi_requests = []
        # Communication stage
        # send
        self.get_timestep(-1, wout=self.w_send, index_range='slice')

        request_send = self.ensemble.isend(self.w_send, dest=((r+1) % n), tag=r)
        mpi_requests.extend(request_send)

        request_recv = self.ensemble.irecv(self.w_recv, source=((r-1) % n), tag=r-1)
        mpi_requests.extend(request_recv)

        # wait for the data [we should really do this after internal
        # assembly but have avoided that for now]
        MPI.Request.Waitall(mpi_requests)

    def _assemble_function(self, snes, X, Fvec):
        r"""
        This is the function we pass to the snes to assemble
        the nonlinear residual.
        """
        self.update(X)

        # Set the flag for the circulant option
        if self.circ == "picard":
            self.Circ.assign(1.0)
        else:
            self.Circ.assign(0.0)
        # assembly stage
        fd.assemble(self.para_form, tensor=self.F_all)

        # apply boundary conditions
        for bc in self.boundary_conditions_all:
            bc.apply(self.F_all, u=self.w_all)

        with self.F_all.dat.vec_ro as v:
            v.copy(Fvec)

    def _set_para_form(self):
        """
        Constructs the bilinear form for the all at once system.
        Specific to the theta-centred Crank-Nicholson method
        """

        w_all_cpts = fd.split(self.w_all)

        test_fns = fd.TestFunctions(self.function_space_all)

        dt = fd.Constant(self.dt)
        theta = fd.Constant(self.theta)
        alpha = fd.Constant(self.alpha)
        ncpts = self.ncomponents

        for n in range(self.slice_partition[self.time_rank]):
            # previous time level
            if n == 0:
                # self.w_recv will contain the adjacent data
                if self.time_rank == 0:
                    # need the initial data
                    w0list = fd.split(self.initial_condition)
                    wrecvlist = fd.split(self.w_recv)
                    w0s = [w0list[i] + self.Circ*alpha*wrecvlist[i]
                           for i in range(ncpts)]
                else:
                    w0s = fd.split(self.w_recv)
            else:
                w0s = w_all_cpts[ncpts*(n-1):ncpts*n]
            # current time level
            w1s = w_all_cpts[ncpts*n:ncpts*(n+1)]
            dws = test_fns[ncpts*n:ncpts*(n+1)]
            # time derivative
            if n == 0:
                p_form = (1.0/dt)*self.form_mass(*w1s, *dws)
            else:
                p_form += (1.0/dt)*self.form_mass(*w1s, *dws)
            p_form -= (1.0/dt)*self.form_mass(*w0s, *dws)
            # vector field
            p_form += theta*self.form_function(*w1s, *dws)
            p_form += (1-theta)*self.form_function(*w0s, *dws)
        self.para_form = p_form
