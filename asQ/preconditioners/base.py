import firedrake as fd
from asQ.profiling import profiler
from asQ.allatonce.mixin import TimePartitionMixin


class AllAtOncePCBase(TimePartitionMixin):
    """
    Base class for preconditioners for the all-at-once system.

    Child classes must:
        - implement apply_impl(pc, xf, yf) method
        - define prefix member
    """

    @profiler()
    def __init__(self):
        r"""A preconditioner for all-at-once systems.
        """
        self.initialized = False

    @profiler()
    def setUp(self, pc):
        """Setup method called by PETSc."""
        if not self.initialized:
            self.initialize(pc)
        self.update(pc)

    @profiler()
    def initialize(self, pc, final_initialize=True):
        if pc.getType() != "python":
            raise ValueError("Expecting PC type python")

        # grab aao objects off petsc mat python context
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

        appctx = jacobian.appctx

        # Input/Output wrapper Functions for all-at-once residual being acted on
        self.xf = aaofunc.copy(copy_values=False)  # input
        self.yf = aaofunc.copy(copy_values=False)  # output

        self.initialized = final_initialize

    @profiler()
    def apply(self, pc, x, y):

        # copy petsc vec into AllAtOnceCofunction
        with self.xf.global_vec_wo() as v:
            x.copy(v)

        self.apply_impl(pc, self.xf, self.yf)

        # copy result into petsc vec
        with self.yf.global_vec_ro() as v:
            v.copy(y)

    @profiler()
    def applyTranspose(self, pc, x, y):
        raise NotImplementedError


class AllAtOnceBlockPCBase(AllAtOncePCBase):
    """
    Base class for preconditioners for the all-at-once system

    Child classes must:
        - implement method: apply_impl(pc, xf, yf)
        - define class member str: prefix
        - define class member Iterable(str): valid_jacobian_states

    PETSc options:

    '{prefix}_linearisation': <'consistent', 'user'>
        Which form to linearise when constructing the block Jacobians.
        Default is 'consistent'.

        'consistent': use the same form used in the AllAtOnceForm residual.
        'user': use the alternative forms given in the appctx.
            If this option is specified then the appctx must contain form_mass
            and form_function entries with keys 'pc_form_mass' and 'pc_form_function'.

    '{prefix}_state': <'current', 'window', 'slice', 'linear', 'initial', 'reference', 'user'>
        Which state to linearise around when constructing the block Jacobians.
        Default is 'current'.

        'window': use the time average over the entire AllAtOnceFunction.
        'slice': use the time average over timesteps on the local Ensemble member.
        'linear': the form linearised is already linear, so no update to the state is needed.
        'initial': the initial condition of the AllAtOnceFunction is used for all timesteps.
        'reference': use the reference state of the AllAtOnceJacobian for all timesteps.

    '{prefix}_block_%d': <LinearVariationalSolver options>
        The solver options for the %d'th block, enumerated globally.
        Use 'aaojacobi_block' to set options for all blocks.
        Default is the Firedrake default options.

    '{prefix}_dt': <float>
        The timestep size to use in the preconditioning matrix.
        Defaults to the timestep size used in the AllAtOnceJacobian.

    '{prefix}_theta': <float>
        The implicit theta method parameter to use in the preconditioning matrix.
        Defaults to the implicit theta method parameter used in the AllAtOnceJacobian.
    """

    @profiler()
    def __init__(self):
        r"""A preconditioner for all-at-once systems.
        """
        self.initialized = False

    @profiler()
    def setUp(self, pc):
        """Setup method called by PETSc."""
        if not self.initialized:
            self.initialize(pc)
        self.update(pc)

    @profiler()
    def initialize(self, pc, final_initialize=True):
        super().initialize(pc, final_initialize=False)

        # option for what state to linearise PC around
        jac_option = f"{prefix}state"

        self.jacobian_state = get_option_from_list(
            state_option, self.valid_jacobian_states, default_index=0)

        if jac_state == 'reference' and jacobian.reference_state is None:
            msg = f"AllAtOnceJacobian must be provided a reference state to use \'reference\' for {prefix}state."
            raise ValueError(msg)

        # Problem parameter options
        self.dt = PETSc.Options().getReal(
            f"{prefix}dt", default=self.aaoform.dt)
        dt = self.dt

        self.theta = PETSc.Options().getReal(
            f"{prefix}theta", default=self.aaoform.theta)
        theta = self.theta

        self.time = tuple(fd.Constant(0) for _ in range(aaofunc.nlocal_timesteps))

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

        self.initialized = final_initialize

    @profiler()
    def apply(self, pc, x, y):

        # copy petsc vec into Function
        self.xf.assign(x)

        self.apply_impl(self.xf, self.yf)

        # copy result into petsc vec
        with self.yf.global_vec_ro() as v:
            v.copy(y)

    @profiler()
    def applyTranspose(self, pc, x, y):
        raise NotImplementedError
