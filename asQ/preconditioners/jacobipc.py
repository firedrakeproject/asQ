import firedrake as fd
from firedrake.petsc import PETSc
from asQ.profiling import profiler
from asQ.ensemble import split_ensemble
from asQ.parallel_arrays import SharedArray
from asQ.allatonce import (time_average, LinearSolver,
                           AllAtOnceFunction, AllAtOnceCofunction)
from asQ.preconditioners.base import (AllAtOnceBlockPCBase, AllAtOncePCBase,
                                      get_default_options)

__all__ = ['JacobiPC', 'SliceJacobiPC']


class JacobiPC(AllAtOnceBlockPCBase):
    """
    A block Jacobi preconditioner where each block is built from a single timestep.
    Each block is (approximately) solved using its own LinearVariatonalSolver.

    PETSc options:

    'aaojacobi_linearisation': <'consistent', 'user'>
        Which form to linearise when constructing the block Jacobians.
        Default is 'consistent'.

        'consistent': use the same form used in the AllAtOnceForm residual.
        'user': use the alternative forms given in the appctx.
            If this option is specified then the appctx must contain form_mass
            and form_function entries with keys 'pc_form_mass' and 'pc_form_function'.

    'aaojacobi_state': <'current', 'window', 'slice', 'linear', 'initial', 'reference', 'user'>
        Which state to linearise around when constructing the block Jacobians.
        Default is 'current'.

        'window': use the time average over the entire AllAtOnceFunction.
        'slice': use the time average over timesteps on the local Ensemble member.
        'linear': the form linearised is already linear, so no update to the state is needed.
        'initial': the initial condition of the AllAtOnceFunction is used for all timesteps.
        'reference': use the reference state of the AllAtOnceJacobian for all timesteps.

    'aaojacobi_block_%d': <LinearVariationalSolver options>
        The solver options for the %d'th block, enumerated globally.
        Use 'aaojacobi_block' to set options for all blocks.
        Default is the Firedrake default options.

    'aaojacobi_dt': <float>
        The timestep size to use in the preconditioning matrix.
        Defaults to the timestep size used in the AllAtOnceJacobian.

    'aaojacobi_theta': <float>
        The implicit theta method parameter to use in the preconditioning matrix.
        Defaults to the implicit theta method parameter used in the AllAtOnceJacobian.

    If the AllAtOnceSolver's appctx contains a 'block_appctx' dictionary, this is
    added to the appctx of each block solver.  The appctx of each block solver also
    contains the following:
        'blockid': index of the block.
        'dt': timestep of the block.
        'theta': implicit theta value of the block.
        'u0': state around which the block is linearised.
        't0': time at which the block is linearised.
        'bcs': block boundary conditions.
        'form_mass': function used to build the block mass matrix.
        'form_function': function used to build the block stiffness matrix.
    """
    prefix = "aaojacobi_"

    valid_jacobian_states = tuple(('current', 'window', 'slice', 'linear',
                                   'initial', 'reference', 'user'))

    @profiler()
    def initialize(self, pc):
        super().initialize(pc, final_initialize=False)

        aaofunc = self.aaofunc

        # all-at-once reference state
        self.state_func = aaofunc.copy()

        # single timestep function space
        field_function_space = aaofunc.field_function_space

        # Building the nonlinear operator
        self.block_solvers = []

        # zero out bc dofs
        self.block_bcs = tuple(
            fd.DirichletBC(field_function_space,
                           0*bc.function_arg,
                           bc.sub_domain)
            for bc in self.aaoform.field_bcs)

        # user appctx for the blocks
        block_appctx = self.appctx.get('block_appctx', {})

        dt1 = fd.Constant(1/self.dt)
        tht = fd.Constant(self.theta)

        # Block i has prefix 'aaojacobi_block_{i}', but we want to be able
        # to set default options for all blocks using 'aaojacobi_block'.
        # LinearVariationalSolver will prioritise options it thinks are from
        # the command line (including those in the `inserted_options` database
        # of the AllAtOnceSolver) over the ones passed to __init__, so we pull
        # the default options off the global dict and pass these explicitly to LVS.
        default_block_prefix = f"{self.full_prefix}block_"
        default_block_options = get_default_options(
            default_block_prefix, range(self.ntimesteps))

        # building the block problem solvers
        for i in range(self.nlocal_timesteps):

            # the reference states
            u0 = self.state_func[i]
            t0 = self.time[i]

            # the form
            vs = fd.TestFunctions(field_function_space)
            us = fd.split(u0)
            M = self.form_mass(*us, *vs)
            K = self.form_function(*us, *vs, t0)

            F = dt1*M + tht*K
            A = fd.derivative(F, u0)

            # pass parameters into PC:
            appctx_h = {
                "dt": self.dt,
                "theta": self.theta,
                "tref": t0,
                "uref": u0,
                "bcs": self.block_bcs,
                "form_mass": self.form_mass,
                "form_function": self.form_function,
            }

            appctx_h.update(block_appctx)

            # the global index of this block
            ii = aaofunc.transform_index(i, from_range='slice', to_range='window')

            # The block rhs/solution are the timestep i of the
            # input/output AllAtOnceCofunction/Function
            block_problem = fd.LinearVariationalProblem(A, self._x[i], self._y[i],
                                                        bcs=self.block_bcs,
                                                        constant_jacobian=True)

            block_solver = fd.LinearVariationalSolver(
                block_problem, appctx=appctx_h,
                options_prefix=default_block_prefix+str(ii),
                solver_parameters=default_block_options)

            self.block_solvers.append(block_solver)

        self.block_iterations = SharedArray(self.time_partition,
                                            dtype=int,
                                            comm=self.ensemble.ensemble_comm)
        self.initialized = True

    @profiler()
    def update(self, pc):
        """
        Update the state to linearise around according to aaojacobi_state.
        """

        aaofunc = self.aaofunc
        aaoform = self.aaoform
        state_func = self.state_func
        jacobian_state = self.jacobian_state

        for st, ft in zip(self.time, aaoform.time):
            st.assign(ft)

        if jacobian_state == 'linear':
            return

        elif jacobian_state == 'current':
            state_func.assign(aaofunc)

        elif jacobian_state in ('window', 'slice'):
            time_average(aaofunc, state_func.initial_condition,
                         state_func.uprev, average=jacobian_state)
            state_func.assign(state_func.initial_condition)

            for t in self.time:
                if jacobian_state == 'window':
                    t.assign(aaoform.t0 + self.dt*(self.ntimesteps + 1)/2)
                elif jacobian_state == 'slice':
                    i1 = aaofunc.transform_index(0, from_range='slice',
                                                 to_range='window')
                    t1 = aaoform.t0 + i1*self.dt
                    t.assign(t1 + self.dt*(self.nlocal_timesteps + 1)/2)

        elif jacobian_state == 'initial':
            state_func.assign(aaofunc.initial_condition)
            for t in self.time:
                t.assign(self.aaoform.t0)

        elif jacobian_state == 'reference':
            aaofunc.assign(self.jacobian.reference_state)

        elif jacobian_state == 'user':
            pass

        for block in self.block_solvers:
            block.invalidate_jacobian()

        return

    @profiler()
    def apply_impl(self, pc, x, y):
        # x and y are already the rhs and solution of the blocks
        self._y.zero()
        for i in range(self.nlocal_timesteps):
            self.block_solvers[i].solve()


class SliceJacobiPC(AllAtOncePCBase):
    """
    A block Jacobi preconditioner where each block (slice) is built from several timesteps.

    The all-at-once system is split into several slices of several
    timesteps each (these do not necessarily have to match the 'slices'
    of the ensemble partition, i.e. the timesteps on a single ensemble
    member).
    The preconditioner is constructed as a block-diagonal system where
    each block is the all-at-once system of a slice. Each block (slice)
    is (approximately) solved with its own KSP.

    PETSc options:

    'slice_jacobi_nsteps': <int>
        The number of timesteps per slice. Must be an integer multiple
        of the number of timesteps on each ensemble member i.e. all
        timesteps on an ensemble member must belong to the same slice.

    'slice_jacobi_slice_%d': <AllAtOnceSolver options>
        The solver options for the %d'th LinearSolver for each slice, enumerated globally.
        Use 'slice_jacobi_slice' to set options for all blocks.
    """
    prefix = "slice_jacobi_"

    @profiler()
    def initialize(self, pc, final_initialize=True):
        super().initialize(pc, final_initialize=False)

        # # # slice ensemble # # #

        # slice here means the smaller local ensemble, not the
        # section of the timeseries on the local ensemble member.

        # all ensemble members must have the same number of timesteps
        if len(set(self.time_partition)) != 1:
            msg = "SliceJacobiPC only implemented for balanced partitions yet"
            raise ValueError(msg)

        # how many timesteps in each slice?
        slice_size = PETSc.Options().getInt(
            f"{self.full_prefix}nsteps")

        # we need to work out how many members of the global ensemble
        # needed to get `split_size` timesteps on each slice ensemble
        slice_members = slice_size // self.time_partition[0]
        nslices = self.ntimesteps // slice_size

        # create the ensemble for the local slice by splitting the global ensemble
        self.slice_ensemble = split_ensemble(self.ensemble,
                                             split_size=slice_members)

        # which slice are we in?
        self.slice_rank = self.ensemble.ensemble_comm.rank // slice_members

        # the slice partition matches the corresponding
        # slice of the global partition.
        # only works for balanced partitions yet
        part = self.time_partition[0]
        slice_partition = tuple(part for i in range(slice_members))

        # # # slice aaofuncs - jacobian, yslice # # #

        # the slice jacobian is created around a slice
        # aaofunc that tracks the global aaofunc
        field_function_space = self.aaofunc.field_function_space
        slice_func = AllAtOnceFunction(self.slice_ensemble,
                                       slice_partition,
                                       field_function_space)
        self.slice_func = slice_func
        self._update_function()

        # the result is placed in here and then copied
        # out to the global result
        self.yslice = slice_func.copy()

        # this is the slice rhs
        self.xslice = AllAtOnceCofunction(self.slice_ensemble,
                                          slice_partition,
                                          field_function_space.dual())

        # # # slice aaoform - jacobian, # # #

        # here we abuse that aaoform.copy will create
        # the new form over the ensemble of the
        # aaofunc kwarg not its own ensemble.
        self.slice_form = self.aaoform.copy(aaofunc=slice_func)
        self._update_time()

        # # # slice parameters # # #
        default_slice_prefix = f"{self.full_prefix}slice_"
        default_slice_options = get_default_options(
            default_slice_prefix, range(nslices))

        # default to treating the slice as a PC not a KSP
        has_default_ksp_type = (
            'ksp_type' in default_slice_options
            or ('ksp' in default_slice_options
                and 'type' in default_slice_options['ksp']))

        if not has_default_ksp_type:
            default_slice_options['ksp_type'] = 'preonly'

        self.slice_solver = LinearSolver(
            self.slice_form, appctx=self.appctx,
            options_prefix=default_slice_prefix+str(self.slice_rank),
            solver_parameters=default_slice_options)

        self.initialized = final_initialize

    def _update_function(self):
        # # # update the timestep values
        aaofunc = self.aaofunc
        slice_func = self.slice_func

        # slice initial conditions
        aaofunc.update_time_halos()
        if self.slice_rank == 0:
            slice_func.initial_condition.assign(aaofunc.initial_condition)
        else:
            slice_func.initial_condition.assign(aaofunc.uprev)

        # slice values
        for i in range(self.nlocal_timesteps):
            slice_func[i].assign(aaofunc[i])
        slice_func.update_time_halos()

    def _update_time(self):
        # # # update the time values
        aaoform = self.aaoform
        slice_form = self.slice_form

        # slice initial time
        t0 = aaoform.t0 if self.slice_rank == 0 else aaoform.tprev
        slice_form.time_update(t0)

    @profiler()
    def update(self, pc):
        """
        Update the slice states.
        """
        self._update_function()
        self._update_time()

        # # # update the slice
        self.slice_solver.jacobian.update()

        return

    @profiler()
    def apply_impl(self, pc, x, y):

        # copy global rhs into slice rhs
        for i in range(self.nlocal_timesteps):
            self.xslice[i].assign(x[i])

        self.yslice.zero()
        self.slice_solver.solve(self.xslice, self.yslice)

        # copy slice result into global result
        for i in range(self.nlocal_timesteps):
            y[i].assign(self.yslice[i])
