import firedrake as fd
from firedrake.petsc import PETSc

from asQ.profiling import profiler
from asQ.common import get_option_from_list
from asQ.parallel_arrays import SharedArray

from asQ.allatonce.mixin import TimePartitionMixin
from asQ.allatonce import AllAtOnceFunction, AllAtOnceCofunction


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
        self.full_prefix = prefix + self.prefix

        A, _ = pc.getOperators()
        jacobian = A.getPythonContext()
        self.jacobian = jacobian
        self._time_partition_setup(jacobian.ensemble, jacobian.time_partition)

        jacobian.pc = self
        aaofunc = jacobian.current_state
        self.aaofunc = aaofunc
        self.aaoform = jacobian.aaoform

        self.appctx = jacobian.appctx

        # Input/Output wrapper Functions for all-at-once residual being acted on
        self.x = AllAtOnceCofunction(self.ensemble, self.time_partition,
                                     aaofunc.field_function_space.dual())

        self.y = AllAtOnceFunction(self.ensemble, self.time_partition,
                                   aaofunc.field_function_space)

        self.initialized = final_initialize

    @profiler()
    def _record_diagnostics(self):
        """
        Update diagnostic information from block linear solvers.

        Must be called exactly once at the end of each apply().
        """
        pass

    @profiler()
    def apply(self, pc, x, y):

        # copy petsc vec into AllAtOnceCofunction
        with self.x.global_vec_wo() as v:
            x.copy(v)

        self.apply_impl(pc, self.x, self.y)

        # copy result into petsc vec
        with self.y.global_vec_ro() as v:
            v.copy(y)

        self._record_diagnostics()

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
        jac_option = f"{self.full_prefix}state"

        self.jacobian_state = get_option_from_list(
            jac_option, self.valid_jacobian_states, default_index=0)

        if self.jacobian_state == 'reference' and self.jacobian.reference_state is None:
            msg = f"AllAtOnceJacobian must be provided a reference state to use \'reference\' for {self.full_prefix}state."
            raise ValueError(msg)

        # Problem parameter options
        self.dt = PETSc.Options().getReal(
            f"{self.full_prefix}dt", default=self.aaoform.dt)

        self.theta = PETSc.Options().getReal(
            f"{self.full_prefix}theta", default=self.aaoform.theta)

        self.time = tuple(fd.Constant(0) for _ in range(self.nlocal_timesteps))

        # which form to linearise around
        valid_linearisations = ['consistent', 'user']
        linearisation_option = f"{self.full_prefix}linearisation"

        linearisation = get_option_from_list(linearisation_option,
                                             valid_linearisations,
                                             default_index=0)

        if linearisation == 'consistent':
            form_mass = self.aaoform.form_mass
            form_function = self.aaoform.form_function
        elif linearisation == 'user':
            try:
                form_mass = self.appctx['pc_form_mass']
                form_function = self.appctx['pc_form_function']
            except KeyError as err:
                err_msg = "appctx must contain 'pc_form_mass' and 'pc_form_function' if " \
                          + f"{linearisation_option} = 'user'"
                raise type(err)(err_msg) from err

        self.form_mass = form_mass
        self.form_function = form_function

        self.block_iterations = SharedArray(self.time_partition,
                                            dtype=int,
                                            comm=self.ensemble.ensemble_comm)

        self.initialized = final_initialize

    @profiler()
    def _record_diagnostics(self):
        """
        Update diagnostic information from block linear solvers.

        Must be called exactly once at the end of each apply().
        """
        super()._record_diagnostics()
        for i in range(self.nlocal_timesteps):
            its = self.block_solvers[i].snes.getLinearSolveIterations()
            self.block_iterations.dlocal[i] += its
