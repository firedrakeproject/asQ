import firedrake as fd
from firedrake.petsc import PETSc
from functools import reduce
from operator import mul
from .profiling import memprofile

from asQ.parallel_arrays import in_range, DistributedDataLayout1D


class JacobianMatrix(object):
    @memprofile
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

        self.urecv = fd.Function(self.aaos.function_space)  # will contain the previous time value i.e. 3*r-1
        # Jform missing contributions from the previous step
        # Find u1 s.t. F[u1, u2, u3; v] = 0 for all v
        # definition:
        # dF_{u1}[u1, u2, u3; delta_u, v] =
        #  lim_{eps -> 0} (F[u1+eps*delta_u,u2,u3;v]
        #                  - F[u1,u2,u3;v])/eps
        # Newton, solves for delta_u such that
        # dF_{u1}[u1, u2, u3; delta_u, v] = -F[u1,u2,u3; v], for all v
        # then updates u1 += delta_u
        self.Jform = fd.derivative(self.aaos.aao_form, self.aaos.w_all)
        # Jform contributions from the previous step
        self.Jform_prev = fd.derivative(self.aaos.aao_form,
                                        self.aaos.w_recv)

    @PETSc.Log.EventDecorator()
    @memprofile
    def mult(self, mat, X, Y):

        self.aaos.update(X, wall=self.u, wrecv=self.urecv, blocking=True)

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
    @memprofile
    def __init__(self,
                 ensemble, time_partition,
                 dt, theta,
                 form_mass, form_function,
                 w0, bcs=[],
                 circ="", alpha=1e-3):
        """
        The all-at-once system representing multiple timesteps of a time-dependent finite-element problem.

        :arg ensemble: time-parallel ensemble communicator.
        :arg time_partition: a list of integers for the number of timesteps stored on each ensemble rank.
        :arg w0: a Function containing the initial data.
        :arg bcs: a list of DirichletBC boundary conditions on w0.function_space.
        """
        self.layout = DistributedDataLayout1D(time_partition, ensemble.ensemble_comm)

        self.ensemble = ensemble
        self.time_partition = self.layout.partition
        self.time_rank = ensemble.ensemble_comm.rank
        self.nlocal_timesteps = self.layout.local_size
        self.ntimesteps = self.layout.global_size

        self.initial_condition = w0
        self.function_space = w0.function_space()
        self.boundary_conditions = bcs
        self.ncomponents = len(self.function_space.split())

        self.dt = dt
        self.time = tuple(fd.Constant(0) for _ in range(self.nlocal_timesteps))
        self.t0 = fd.Constant(0.0)
        self.theta = theta

        self.form_mass = form_mass
        self.form_function = form_function

        self.circ = circ
        self.alpha = alpha
        self.Circ = fd.Constant(0.0)

        self.max_indices = {
            'component': self.ncomponents,
            'slice': self.nlocal_timesteps,
            'window': self.ntimesteps
        }

        # function pace for the slice of the all-at-once system on this process
        self.function_space_all = reduce(mul, (self.function_space
                                               for _ in range(self.nlocal_timesteps)))

        self.w_all = fd.Function(self.function_space_all)
        self.w_alls = self.w_all.split()

        for i in range(self.nlocal_timesteps):
            self.set_field(i, self.initial_condition, index_range='slice')

        self.boundary_conditions_all = self.set_boundary_conditions(bcs)

        for bc in self.boundary_conditions_all:
            bc.apply(self.w_all)

        # function to assemble the nonlinear residual
        self.F_all = fd.Function(self.function_space_all)

        # functions containing the last and next steps for parallel
        # communication timestep
        # from the previous iteration
        self.w_recv = fd.Function(self.function_space)
        self.w_send = fd.Function(self.function_space)

        self._set_aao_form()
        self.jacobian = JacobianMatrix(self)

    def set_boundary_conditions(self, bcs):
        """
        Set the boundary conditions onto solution at each timestep in the all-at-once system.
        Returns a list of all boundary conditions on the all-at-once system.

        :arg bcs: a list of the boundary conditions to apply
        """
        is_mixed_element = isinstance(self.function_space.ufl_element(), fd.MixedElement)

        bcs_all = []
        for bc in bcs:
            for step in range(self.nlocal_timesteps):
                if is_mixed_element:
                    cpt = bc.function_space().index
                else:
                    cpt = 0
                index = self.transform_index(step, cpt)
                bc_all = fd.DirichletBC(self.function_space_all.sub(index),
                                        bc.function_arg,
                                        bc.sub_domain)
                bcs_all.append(bc_all)

        return bcs_all

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
            in_range(cpt, self.max_indices['component'], throws=True)
            cpt = cpt % self.max_indices['component']
            return i*self.ncomponents + cpt

    @PETSc.Log.EventDecorator()
    def set_component(self, step, cpt, wnew, index_range='slice', f_alls=None):
        '''
        Set component of solution at a timestep to new value

        :arg step: index of timestep
        :arg cpt: index of component
        :arg wout: new solution for timestep
        :arg index_range: is index in window or slice?
        :arg f_alls: an all-at-once function to set timestep in. If None, self.w_alls is used
        '''
        # index of component in all at once function
        aao_index = self.transform_index(step, cpt, from_range=index_range, to_range='slice')

        if f_alls is None:
            f_alls = self.w_alls

        f_alls[aao_index].assign(wnew)

    @PETSc.Log.EventDecorator()
    def get_component(self, step, cpt, index_range='slice', wout=None, name=None, f_alls=None, deepcopy=False):
        '''
        Get component of solution at a timestep

        :arg step: index of timestep to get
        :arg cpt: index of component
        :arg index_range: is timestep index in window or slice?
        :arg wout: function to set to component (component returned if None)
        :arg name: name of returned function if deepcopy=True. Ignored if wout is not None
        :arg f_alls: an all-at-once function to get timestep from. If None, self.w_alls is used
        :arg deepcopy: if True, new function is returned. If false, handle to component of f_alls is returned. Ignored if wout is not None
        '''
        # index of component in all at once function
        aao_index = self.transform_index(step, cpt, from_range=index_range, to_range='slice')

        if f_alls is None:
            f_alls = self.w_alls

        # required component
        wget = f_alls[aao_index]

        if wout is not None:
            wout.assign(wget)
            return wout

        if deepcopy is False:
            return wget
        else:  # deepcopy is True
            wreturn = fd.Function(self.function_space.sub(cpt), name=name)
            wreturn.assign(wget)
            return wreturn

    def get_field_components(self, step, index_range='slice', f_alls=None):
        '''
        Get tuple of the components of the all-at-once function for a timestep.

        :arg step: index of timestep.
        :arg index_range: is index in window or slice?
        :arg f_alls: an all-at-once function to get timestep from. If None, self.w_alls is used
        '''
        if f_alls is None:
            f_alls = self.w_alls

        return tuple(self.get_component(step, cpt, f_alls=f_alls)
                     for cpt in range(self.ncomponents))

    @PETSc.Log.EventDecorator()
    def set_field(self, step, wnew, index_range='slice', f_alls=None):
        '''
        Set solution at a timestep to new value

        :arg step: index of timestep to set.
        :arg wnew: new solution for timestep
        :arg index_range: is index in window or slice?
        :arg f_alls: an all-at-once function to set timestep in. If None, self.w_alls is used
        '''
        for cpt in range(self.ncomponents):
            self.set_component(step, cpt, wnew.sub(cpt),
                               index_range=index_range, f_alls=f_alls)

    @PETSc.Log.EventDecorator()
    def get_field(self, step, index_range='slice', wout=None, name=None, f_alls=None):
        '''
        Get solution at a timestep

        :arg step: index of timestep to set.
        :arg index_range: is index in window or slice?
        :arg wout: function to set to timestep (timestep returned if None)
        :arg name: name of returned function. Ignored if wout is not None
        :arg f_alls: an all-at-once function to get timestep from. If None, self.w_alls is used
        '''
        if wout is None:
            wget = fd.Function(self.function_space, name=name)
        else:
            wget = wout

        for cpt in range(self.ncomponents):
            wcpt = self.get_component(step, cpt, index_range=index_range, f_alls=f_alls)
            wget.sub(cpt).assign(wcpt)

        return wget

    @PETSc.Log.EventDecorator()
    def for_each_timestep(self, callback):
        '''
        call callback for each timestep in each slice in the current window
        callback arguments are: timestep index in window, timestep index in slice, Function at timestep

        :arg callback: the function to call for each timestep
        '''

        w = fd.Function(self.function_space)
        for slice_index in range(self.nlocal_timesteps):
            window_index = self.transform_index(slice_index,
                                                from_range='slice',
                                                to_range='window')
            self.get_field(slice_index, wout=w, index_range='slice')
            callback(window_index, slice_index, w)

    @PETSc.Log.EventDecorator()
    def next_window(self, w1=None):
        """
        Reset all-at-once-system ready for next time-window

        :arg w1: initial solution for next time-window.If None,
                 will use the final timestep from previous window
        """
        if w1 is not None:  # use given function
            self.initial_condition.assign(w1)

        else:  # last rank broadcasts final timestep

            end_rank = self.ensemble.ensemble_comm.size - 1

            if self.time_rank == end_rank:
                self.get_field(-1, wout=self.initial_condition, index_range='window')

            self.ensemble.bcast(self.initial_condition, root=end_rank)

        # persistence forecast
        for i in range(self.nlocal_timesteps):
            self.set_field(i, self.initial_condition, index_range='slice')
            self.time[i].assign(self.time[i] + self.dt*self.ntimesteps)
        self.t0.assign(self.t0 + self.dt*self.ntimesteps)
        return

    @PETSc.Log.EventDecorator()
    def update_time_halos(self, wsend=None, wrecv=None, walls=None, blocking=True):
        '''
        Update wrecv with the last step from the previous slice (periodic) of walls

        :arg wsend: Function to send last step of current slice to next slice. if None self.w_send is used
        :arg wrecv: Function to receive last step of previous slice. if None self.w_recv is used
        :arg walls: all at once function list to update wrecv from. if None self.w_alls is used
        :arg blocking: Whether to use blocking MPI communications. If False then a list of MPI requests is returned
        '''

        if wsend is None:
            wsend = self.w_send
        if wrecv is None:
            wrecv = self.w_recv
        if walls is None:
            walls = self.w_alls

        if blocking:
            sendrecv = self.ensemble.sendrecv
        else:
            sendrecv = self.ensemble.isendrecv

        # send last timestep on current slice to next slice
        self.get_field(-1, wout=wsend, index_range='slice', f_alls=walls)

        size = self.ensemble.ensemble_comm.size
        rank = self.ensemble.ensemble_comm.rank

        # ring communication
        dst = (rank+1) % size
        src = (rank-1) % size

        return sendrecv(fsend=wsend, dest=dst, sendtag=rank,
                        frecv=wrecv, source=src, recvtag=src)

    @PETSc.Log.EventDecorator()
    def update(self, X, wall=None, wsend=None, wrecv=None, blocking=True):
        '''
        Update self.w_alls and self.w_recv from PETSc Vec X.
        The local parts of X are copied into self.w_alls
        and the last step from the previous slice (periodic)
        is copied into self.u_prev
        '''
        if wall is None:
            wall = self.w_all
        if wsend is None:
            wsend = self.w_send
        if wrecv is None:
            wrecv = self.w_recv

        with wall.dat.vec_wo as v:
            v.array[:] = X.array_r

        return self.update_time_halos(wsend=wsend,
                                      wrecv=wrecv,
                                      walls=wall.split(),
                                      blocking=blocking)

    @PETSc.Log.EventDecorator()
    @memprofile
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
        fd.assemble(self.aao_form, tensor=self.F_all)

        # apply boundary conditions
        for bc in self.boundary_conditions_all:
            bc.apply(self.F_all, u=self.w_all)

        with self.F_all.dat.vec_ro as v:
            v.copy(Fvec)

    def _set_aao_form(self):
        """
        Constructs the bilinear form for the all at once system.
        Specific to the theta-centred Crank-Nicholson method
        """

        w_alls = fd.split(self.w_all)
        test_fns = fd.TestFunctions(self.function_space_all)

        dt = fd.Constant(self.dt)
        theta = fd.Constant(self.theta)
        alpha = fd.Constant(self.alpha)

        def get_step(i):
            return self.get_field_components(i, f_alls=w_alls)

        def get_test(i):
            return self.get_field_components(i, f_alls=test_fns)

        for n in range(self.nlocal_timesteps):
            self.time[n].assign(self.time[n] + dt*(self.layout.transform_index(n, 'l', 'g') + 1))
            # previous time level
            if n == 0:
                if self.time_rank == 0:
                    # need the initial data
                    w0list = fd.split(self.initial_condition)

                    # circulant option for quasi-Jacobian
                    wrecvlist = fd.split(self.w_recv)

                    w0s = [w0list[i] + self.Circ*alpha*wrecvlist[i]
                           for i in range(self.ncomponents)]
                else:
                    # self.w_recv will contain the data from the previous slice
                    w0s = fd.split(self.w_recv)
            else:
                w0s = get_step(n-1)

            # current time level
            w1s = get_step(n)
            dws = get_test(n)

            # time derivative
            if n == 0:
                aao_form = (1.0/dt)*self.form_mass(*w1s, *dws)
            else:
                aao_form += (1.0/dt)*self.form_mass(*w1s, *dws)
            aao_form -= (1.0/dt)*self.form_mass(*w0s, *dws)

            # vector field
            if self.layout.transform_index(n, 'l', 'g') == 0:
                aao_form += theta*self.form_function(*w1s, *dws, self.time[n])
                aao_form += (1-theta)*self.form_function(*w0s, *dws, self.t0)
            else:
                aao_form += theta*self.form_function(*w1s, *dws, self.time[n])
                aao_form += (1-theta)*self.form_function(*w0s, *dws, self.time[n-1])
        self.aao_form = aao_form
