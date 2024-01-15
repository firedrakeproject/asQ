import firedrake as fd
from asQ.profiling import profiler
from asQ.common import get_option_from_list
from asQ.allatonce.mixin import TimePartitionMixin

__all__ = ['JacobiPC', 'SliceJacobiPC']


class JacobiPC(TimePartitionMixin):
    """
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

    @profiler()
    def __init__(self):
        r"""A block diagonal Jacobi preconditioner for all-at-once systems.
        """
        self.initialized = False

    @profiler()
    def setUp(self, pc):
        """Setup method called by PETSc."""
        if not self.initialized:
            self.initialize(pc)
        self.update(pc)

    @profiler()
    def initialize(self, pc):
        if pc.getType() != "python":
            raise ValueError("Expecting PC type python")

        prefix = pc.getOptionsPrefix()
        prefix = prefix + self.prefix

        A, _ = pc.getOperators()
        jacobian = A.getPythonContext()
        self.jacobian = jacobian
        self._time_partition_setup(jacobian.ensemble, jacobian.time_partition)

        jacobian.pc = self
        aaofunc = jacobian.current_state
        self.aaofunc = aaofunc
        self.aaoform = jacobian.aaoform
        self.state_func = aaofunc.copy()

        appctx = jacobian.appctx

        # option for whether to use slice or window average for block jacobian
        valid_jacobian_states = tuple(('current', 'window', 'slice', 'linear',
                                       'initial', 'reference', 'user'))
        jac_option = f"{prefix}state"

        self.jacobian_state = partial(get_option_from_list,
                                      state_option, valid_jacobian_states,
                                      default_index=0)
        jac_state = self.jac_state()

        if jac_state == 'reference' and jacobian.reference_state is None:
            raise ValueError("AllAtOnceJacobian must be provided a reference state to use \'reference\' for diagfft_state.")

        # basic model function space
        self.blockV = aaofunc.field_function_space

        # Input/Output wrapper Functions for all-at-once residual being acted on
        self.xf = aaofunc.copy(copy_values=False)  # input
        self.yf = aaofunc.copy(copy_values=False)  # output

        # diagonalisation options
        self.dt = PETSc.Options().getReal(
            f"{prefix}dt", default=self.aaoform.dt)
        dt = self.dt

        self.theta = PETSc.Options().getReal(
            f"{prefix}theta", default=self.aaoform.theta)
        theta = self.theta

        nt = self.ntimesteps
        self.time = tuple(fd.Constant(0) for _ in range(aaofunc.nlocal_timesteps))

        self.block_rhs = fd.Cofunction(self.CblockV.dual())

        # Building the nonlinear operator
        self.block_solvers = []

        # which form to linearise around
        valid_linearisations = ['consistent', 'user']
        linearisation_option = f"{prefix}linearisation"

        linearisation = get_option_from_list(linearisation_option,
                                             valid_linearisations,
                                             default_index=0)

        if linearisation == 'consistent':
            form_mass = self.aaoform.form_mass
            form_function = self.aaoform.form_function
        elif linearisation == 'user':
            try:
                form_mass = appctx['pc_form_mass']
                form_function = appctx['pc_form_function']
            except KeyError as err:
                err_msg = "appctx must contain 'pc_form_mass' and 'pc_form_function' if " \
                          + f"{linearisation_option} = 'user'"
                raise type(err)(err_msg) from err

        self.form_mass = form_mass
        self.form_function = form_function

        # user appctx for the blocks
        block_appctx = appctx.get('block_appctx', {})

        dt1 = fd.Constant(1/dt)
        tht = fd.Constant(theta)

        # building the block problem solvers
        for i in range(nlocal_timesteps)

            # The rhs
            L = self.block_rhs

            # the reference states
            u0 = self.state_func[i]
            t0 = self.time[i]

            # the form
            vs = fd.TestFunctions(self.blockV)
            us = fd.split(u0)
            M = form_mass(*us, *vs)
            K = form_function(*us, *vs, t0)

            F = dt1*M + tht*K
            A = fd.derivative(F, u0)

            # pass parameters into PC:
            appctx_h = {
                "blockid": i,
                "dt": dt,
                "theta": theta,
                "t0": t0,
                "u0": u0,
                "bcs": self.block_bcs,
                "form_mass": self.form_mass,
                "form_function": self.form_function,
            }

            appctx_h.update(block_appctx)

            # Options with prefix 'aaojacobi_block_' apply to all blocks by default
            # If any options with prefix 'aaojacobi_block_{i}' exist, where i is the
            # block number, then this prefix is used instead (like pc fieldsplit)

            ii = aaofunc.transform_index(i, from_range='slice', to_range='window')

            block_prefix = f"{prefix}block_"
            for k, v in PETSc.Options().getAll().items():
                if k.startswith(f"{block_prefix}{str(ii)}_"):
                    block_prefix = f"{block_prefix}{str(ii)}_"
                    break

            block_problem = fd.LinearVariationalProblem(A, L, self.yf[i],
                                                        bcs=self.block_bcs)
            block_solver = fd.LinearVariationalSolver(block_problem,
                                                      appctx=appctx_h,
                                                      options_prefix=block_prefix)
            # multigrid transfer manager
            if f'{self.prefix}transfer_managers' in appctx:
                # block_solver.set_transfer_manager(jacobian.appctx['diagfft_transfer_managers'][ii])
                tm = appctx[f'{self.prefix}transfer_managers'][i]
                block_solver.set_transfer_manager(tm)
                tm_set = (block_solver._ctx.transfer_manager is tm)

                if tm_set is False:
                    print(f"transfer manager not set on block_solvers[{ii}]")

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
        state_func = self.state_func
        jacobian_state = self.jacobian_state()

        if jacobian_state == 'linear':
            pass

        elif jacobian_state == 'current':
            state_func.assign(aaofunc)

        elif jacobian_state in ('window', 'slice'):
            time_average(aaofunc, state_func.initial_condition,
                         state_func.uprev, average=jacobian_state)
            state_func.assign(state_func.initial_condition)

        elif jacobian_state == 'initial':
            state_func.assign(aaofunc.initial_condition)

        elif jacobian_state == 'reference':
            aaofunc.assign(self.jacobian.reference_state)

        elif jacobian_state == 'user':
            pass

        return

    @profiler()
    def apply(self, pc, x, y):

        # copy petsc vec into Function
        self.xf.assign(x)
        self.yf.zero()

        for i in range(self.nlocal_timesteps):
            self.block_solvers[i].solve()

        # copy result into petsc vec
        with self.yf.global_vec_ro() as v:
            v.copy(y)

    @profiler()
    def applyTranspose(self, pc, x, y):
        raise NotImplementedError


class SliceJacobiPC(TimePartitionMixin):
    @profiler()
    def __init__(self):
        raise NotImplementedError

    @profiler()
    def setUp(self, pc):
        raise NotImplementedError

    @profiler()
    def initialize(self, pc):
        raise NotImplementedError

    @profiler()
    def apply(self, pc, x, y):
        raise NotImplementedError

    @profiler()
    def applyTranspose(self, pc, x, y):
        raise NotImplementedError
