import firedrake as fd
from firedrake.petsc import PETSc
from functools import reduce
from operator import mul
from .profiling import memprofile

from asQ.parallel_arrays import in_range, DistributedDataLayout1D


class JacobianMatrix(object):
    prefix = "aaos_jacobian_"

    @memprofile
    def __init__(self, aaos, snes=None):
        r"""
        Python matrix for the Jacobian of the all at once system
        :param aaos: The AllAtOnceSystem object
        """
        self.aaos = aaos

        if snes is not None:
            self.snes = snes

        # function to linearise around, and timestep from end of previous slice
        self.u = fd.Function(self.aaos.function_space_all)
        self.urecv = fd.Function(self.aaos.function_space)

        # function the Jacobian acts on, and contribution from timestep at end of previous slice
        self.x = fd.Function(self.aaos.function_space_all)
        self.xrecv = fd.Function(self.aaos.function_space)

        # output residual, and contribution from timestep at end of previous slice
        self.F = fd.Function(self.aaos.function_space_all)
        self.F_prev = fd.Function(self.aaos.function_space_all)

        # option for what form to linearise
        valid_linearisations = ['consistent', 'user']

        if snes is None:
            self.linearised_mass = aaos.linearised_mass
            self.linearised_function = aaos.linearised_function
        else:
            prefix = snes.getOptionsPrefix()
            prefix += self.prefix
            linear_option = f"{prefix}linearisation"

            linear = PETSc.Options().getString(linear_option, default=valid_linearisations[0])
            assert linear == valid_linearisations[0]
            if linear not in valid_linearisations:
                raise ValueError(f"{linear_option}={linear} but must be one of "+" or ".join(valid_linearisations))

            if linear == 'consistent':
                self.linearised_mass = aaos.form_mass
                self.linearised_function = aaos.form_function
            elif linear == 'user':
                self.linearised_mass = aaos.linearised_mass
                self.linearised_function = aaos.linearised_function

        # all-at-once form to linearise
        self.aao_form = self.aaos.construct_aao_form(wall=self.u, wrecv=self.urecv,
                                                     mass=self.linearised_mass,
                                                     function=self.linearised_function)

        # Jform without contributions from the previous step
        self.Jform = fd.derivative(self.aao_form, self.u)
        # Jform contributions from the previous step
        self.Jform_prev = fd.derivative(self.aao_form, self.urecv)

        # option for what state to linearise around
        valid_jacobian_states = ['current', 'linear', 'initial', 'reference']

        if snes is None:
            self.jacobian_state = lambda: 'current'
        else:
            prefix = snes.getOptionsPrefix()
            prefix += self.prefix
            state_option = f"{prefix}state"

            def jacobian_state():
                state = PETSc.Options().getString(state_option, default='current')
                if state not in valid_jacobian_states:
                    raise ValueError(f"{state_option} must be one of "+" or ".join(valid_jacobian_states))
                return state
            self.jacobian_state = jacobian_state

        jacobian_state = self.jacobian_state()

        if jacobian_state == 'reference' and self.aaos.reference_state is None:
            raise ValueError("AllAtOnceSystem must be provided a reference state to use \'reference\' for aaos_jacobian_state.")

    def update(self, X=None):
        # update the state to linearise around from the current all-at-once solution

        aaos = self.aaos
        jacobian_state = self.jacobian_state()

        if jacobian_state == 'linear':
            return

        elif jacobian_state == 'current':
            if X is None:
                self.u.assign(aaos.w_all)
                self.urecv.assign(aaos.w_recv)
            else:
                aaos.update(X, wall=self.u, wrecv=self.urecv, blocking=True)

        elif jacobian_state == 'initial':
            self.urecv.assign(aaos.initial_condition)
            for i in range(aaos.nlocal_timesteps):
                aaos.set_field(i, aaos.initial_condition, f_alls=self.u.subfunctions)

        elif jacobian_state == 'reference':
            self.urecv.assign(aaos.reference_state)
            for i in range(aaos.nlocal_timesteps):
                aaos.set_field(i, aaos.reference_state, f_alls=self.u.subfunctions)

    @PETSc.Log.EventDecorator()
    @memprofile
    def mult(self, mat, X, Y):

        self.aaos.update(X, wall=self.x, wrecv=self.xrecv, blocking=True)

        # Set the flag for the circulant option
        if self.aaos.circ in ["quasi", "picard"]:
            self.aaos.Circ.assign(1.0)
        else:
            self.aaos.Circ.assign(0.0)

        # assembly stage
        fd.assemble(fd.action(self.Jform, self.x), tensor=self.F)
        fd.assemble(fd.action(self.Jform_prev, self.xrecv),
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
            bc.apply(self.F, u=self.x)
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
                 reference_state=None,
                 linearised_function=None,
                 linearised_mass=None,
                 circ="", alpha=1e-3):
        """
        The all-at-once system representing multiple timesteps of a time-dependent finite-element problem.

        :arg ensemble: time-parallel ensemble communicator.
        :arg time_partition: a list of integers for the number of timesteps stored on each ensemble rank.
        :arg w0: a Function containing the initial data.
        :arg bcs: a list of DirichletBC boundary conditions on w0.function_space.
        :arg reference_state: a Function in W to use as a reference state
            e.g. in DiagFFTPC
        :arg circ: a string describing the option on where to use the
            alpha-circulant modification. "picard" - do a nonlinear wave
            form relaxation method. "quasi" - do a modified Newton
            method with alpha-circulant modification added to the
            Jacobian. To make the alpha circulant modification only in the
            preconditioner, simply set ksp_type:preonly in the solve options.
        :arg alpha: float, circulant matrix parameter
        """
        self.layout = DistributedDataLayout1D(time_partition, ensemble.ensemble_comm)

        self.ensemble = ensemble
        self.time_partition = self.layout.partition
        self.time_rank = ensemble.ensemble_comm.rank
        self.nlocal_timesteps = self.layout.local_size
        self.ntimesteps = self.layout.global_size

        self.function_space = w0.function_space()
        if reference_state is None:
            self.reference_state = None
        else:
            self.reference_state = fd.Function(self.function_space).assign(reference_state)
        self.initial_condition = fd.Function(self.function_space).assign(w0)
        # need to make copy of bcs too instead of taking a reference
        self.boundary_conditions = bcs
        self.ncomponents = len(self.function_space.subfunctions)

        if reference_state is not None and reference_state.function_space() != w0.function_space():
            raise ValueError("AllAtOnceSystem reference state must be in the same function space as the initial condition.")

        self.dt = dt
        self.theta = theta

        self.form_mass = form_mass
        self.form_function = form_function

        self.linearised_mass = linearised_mass if linearised_mass is not None else form_mass
        self.linearised_function = linearised_function if linearised_function is not None else form_function

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
        self.w_alls = self.w_all.subfunctions

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

        self.aao_form = self.construct_aao_form(self.w_all, self.w_recv)

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
                                      walls=wall.subfunctions,
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

    def construct_aao_form(self, wall=None, wrecv=None, mass=None, function=None):
        """
        Constructs the bilinear form for the all at once system.
        Specific to the theta-centred Crank-Nicholson method

        :arg wall: all-at-once function to construct the form over.
            Defaults to the AllAtOnceSystem's.
        :arg wrecv: last timestep from previous time slice.
            Defaults to the AllAtOnceSystem's.
        :arg mass: a function that returns a linear form on w0.function_space()
            providing the mass operator for the time derivative.
        :arg function: a function that returns a form on w0.function_space()
            providing f(w) for the ODE w_t + f(w) = 0.
        """
        if wall is None:
            w_alls = fd.split(self.w_all)
        else:
            w_alls = fd.split(wall)

        if wrecv is None:
            wrecv = self.w_recv

        if mass is None:
            mass = self.form_mass

        if function is None:
            function = self.form_function

        test_fns = fd.TestFunctions(self.function_space_all)

        dt1 = fd.Constant(1.0/self.dt)
        theta = fd.Constant(self.theta)
        thetam1 = fd.Constant(1.0 - self.theta)
        alpha = fd.Constant(self.alpha)

        def get_step(i):
            return self.get_field_components(i, f_alls=w_alls)

        def get_test(i):
            return self.get_field_components(i, f_alls=test_fns)

        for n in range(self.nlocal_timesteps):

            # previous time level
            if n == 0:
                if self.time_rank == 0:
                    # need the initial data
                    w0list = fd.split(self.initial_condition)

                    # circulant option for quasi-Jacobian
                    wrecvlist = fd.split(wrecv)

                    w0s = [w0list[i] + self.Circ*alpha*wrecvlist[i]
                           for i in range(self.ncomponents)]
                else:
                    # wrecv will contain the data from the previous slice
                    w0s = fd.split(wrecv)
            else:
                w0s = get_step(n-1)

            # current time level
            w1s = get_step(n)
            dws = get_test(n)

            # time derivative
            if n == 0:
                aao_form = dt1*mass(*w1s, *dws)
            else:
                aao_form += dt1*mass(*w1s, *dws)
            aao_form -= dt1*mass(*w0s, *dws)

            # vector field
            aao_form += theta*function(*w1s, *dws)
            aao_form += thetam1*function(*w0s, *dws)

        return aao_form
