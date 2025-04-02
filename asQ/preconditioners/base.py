from firedrake.petsc import PETSc

from asQ.profiling import profiler
from asQ.common import get_option_from_list
from asQ.parallel_arrays import SharedArray

from asQ.allatonce.mixin import TimePartitionMixin
from asQ.allatonce import AllAtOnceFunction, AllAtOnceCofunction


def get_default_options(default_prefix, custom_suffixes, options=PETSc.Options()):
    custom_prefixes = (default_prefix + str(suffix) for suffix in custom_suffixes)
    default_options = {
        k.removeprefix(default_prefix): v
        for k, v in options.getAll().items()
        if k.startswith(default_prefix)
        and not any(k.startswith(prefix) for prefix in custom_prefixes)
    }
    return default_options


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
        self.pc_prefix = prefix + "pc_" + self.prefix

        jacobian = self.get_jacobian(pc)
        self.jacobian = jacobian
        self._time_partition_setup(jacobian.ensemble, jacobian.time_partition)

        jacobian.pc = self
        aaofunc = jacobian.aaofunc
        self.aaofunc = aaofunc
        self.aaoform = jacobian.aaoform

        self.appctx = jacobian.appctx

        # Input/Output wrapper Functions for all-at-once residual being acted on
        self.x = AllAtOnceCofunction(
            self.ensemble, self.time_partition,
            aaofunc.field_function_space.dual(),
            full_function_space=aaofunc.dual_space,
            full_dual_space=aaofunc.function_space)

        self.y = AllAtOnceFunction(
            self.ensemble, self.time_partition,
            aaofunc.field_function_space,
            full_function_space=aaofunc.function_space,
            full_dual_space=aaofunc.dual_space)

        self.initialized = final_initialize

    def get_jacobian(self, pc, *args, **kwargs):
        _, P = pc.getOperators()
        jacobian = P.getPythonContext()
        return jacobian

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
    built on blocks matching each timestep of the timeseries.

    Child classes must:
        - implement method: apply_impl(pc, xf, yf)
        - define class member str: prefix
        - define class member Iterable(str): valid_jacobian_states

    PETSc options:

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
    """

    @profiler()
    def initialize(self, pc, final_initialize=True):
        super().initialize(pc, final_initialize=False)

        # option for what state to linearise PC around
        self.jacobian_state = get_option_from_list(
            self.full_prefix, "state", self.valid_jacobian_states,
            default_index=0)

        if self.jacobian_state == 'reference' and self.jacobian.reference_state is None:
            msg = f"AllAtOnceJacobian must be provided a reference state to use \'reference\' for {self.full_prefix}state."
            raise ValueError(msg)

        # time integration parameters
        self.dt = self.aaoform.dt
        self.theta = self.aaoform.theta
        self.time = self.aaoform.time

        # which form to linearise around
        self.form_mass = self.aaoform.form_mass
        self.form_function = self.aaoform.form_function

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
