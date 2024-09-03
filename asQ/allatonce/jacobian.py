import firedrake as fd
from firedrake.petsc import PETSc

from asQ.profiling import profiler
from asQ.common import get_option_from_list
from asQ.allatonce.mixin import TimePartitionMixin
from asQ.allatonce.function import time_average

__all__ = ['AllAtOnceJacobian']


class AllAtOnceJacobian(TimePartitionMixin):
    """
    PETSc options:

    'aaos_jacobian_state': <'current', 'window', 'slice', 'linear', 'initial', 'reference', 'user'>
        Which state to linearise around when constructing the Jacobian.
        Default is 'current'.

        'current': Use the current state of the AllAtOnceFunction (i.e. current Newton iterate).
        'window': Use the time average over the entire AllAtOnceFunction at all timesteps.
        'slice': Use the time average over timesteps on the local Ensemble member at each local timestep.
        'linear': Do not update the state. This option should be used when the form being linearised is linear.
        'initial': Use the initial condition at all timesteps.
        'reference': Use a provided reference state at all timesteps.
        'user': The state will be set manually by the user so no update is needed.
            The `pre_jacobian_callback` argument to the AllAtOnceSolver can be used to set the state.
    """
    prefix = "aaos_jacobian_"

    @profiler()
    def __init__(self, aaoform,
                 reference_state=None,
                 options_prefix="",
                 appctx={}):
        """
        Python context for a PETSc Mat for the Jacobian of an AllAtOnceForm.

        :arg aaoform: The AllAtOnceForm object to linearise.
        :arg reference_state: A firedrake.Function for a single timestep.
            Only needed if 'aaos_jacobian_state' is 'reference'.
        :arg options_prefix: string prefix for the Jacobian PETSc options.
        :arg appctx: the appcontext for the Jacobian and the preconditioner.
        """
        self._time_partition_setup(aaoform.ensemble, aaoform.time_partition)
        prefix = options_prefix + self.prefix

        aaofunc = aaoform.aaofunc
        self.aaoform = aaoform
        self.aaofunc = aaofunc

        self.appctx = appctx

        # function the Jacobian acts on, and contribution from timestep at end of previous slice
        self.x = aaofunc.copy()

        # output residual, and contribution from timestep at end of previous slice
        self.F = aaofunc.copy()
        self.Fprev = fd.Function(aaofunc.function_space)

        # working buffers for calculating time average when needed
        self.ureduce = fd.Function(aaofunc.field_function_space)
        self.uwrk = fd.Function(aaofunc.field_function_space)

        # form without contributions from the previous step
        self.form = fd.derivative(aaoform.form, aaofunc.function)
        self.action = fd.action(self.form, self.x.function)

        # form contributions from the previous step
        self._useprev = aaoform.alpha is not None or self.time_rank != 0
        if self._useprev:
            self.form_prev = fd.derivative(aaoform.form, aaofunc.uprev)
            self.action_prev = fd.action(self.form_prev, self.x.uprev)
        else:
            self.form_prev = None
            self.action_prev = None

        # matrix type for applying the Jacobian action.
        # 'matfree' or 'aij'
        self.mat_type = PETSc.Options().getString(
            options_prefix + 'mat_type', 'matfree')

        # option for what state to linearise around
        valid_jacobian_states = tuple(('current', 'window', 'slice', 'linear',
                                       'initial', 'reference', 'user'))

        if (prefix != "") and (not prefix.endswith("_")):
            prefix += "_"

        self.jacobian_state = get_option_from_list(
            prefix, "state", valid_jacobian_states, default_index=0)

        if reference_state is not None:
            self.reference_state = fd.Function(aaofunc.field_function_space)
            self.reference_state.assign(reference_state)
        else:
            self.reference_state = None

        if self.jacobian_state == 'reference' and self.reference_state is None:
            raise ValueError("AllAtOnceJacobian must be provided a reference state to use \'reference\' for aaos_jacobian_state.")

        self.update()

    @profiler()
    def update(self, X=None):
        """
        Update the state to linearise around according to aaos_jacobian_state.

        :arg X: an optional AllAtOnceFunction or global PETSc Vec.
            If X is not None then the state is updated from X.
        """

        aaofunc = self.aaofunc
        jacobian_state = self.jacobian_state

        if jacobian_state in ('linear', 'user'):
            return

        if X is not None:
            self.aaofunc.assign(X)

        if jacobian_state == 'current':
            return

        elif jacobian_state in ('window', 'slice'):
            time_average(self.aaofunc, self.ureduce, self.uwrk,
                         average=jacobian_state)
            aaofunc.assign(self.ureduce)

        elif jacobian_state == 'initial':
            aaofunc.assign(self.aaofunc.initial_condition)

        elif jacobian_state == 'reference':
            aaofunc.assign(self.reference_state)

        return

    @profiler()
    def mult(self, mat, X, Y):
        """
        Apply the action of the matrix to a PETSc Vec.

        :arg X: a PETSc Vec to apply the action on.
        :arg Y: a PETSc Vec for the result.
        """

        # we could use nonblocking here and overlap comms with assembling form
        self.x.assign(X, update_halos=True, blocking=True)

        # assembly stage
        fd.assemble(self.action, tensor=self.F.function, mat_type=self.mat_type)
        if self._useprev:
            fd.assemble(self.action_prev, tensor=self.Fprev, mat_type=self.mat_type)
            self.F.function += self.Fprev

        # Apply boundary conditions
        # For Jacobian action we should just return the values in X
        # at boundary nodes
        for bc in self.aaoform.bcs:
            bc.homogenize()
            bc.apply(self.F.function, u=self.x.function)
            bc.restore()

        with self.F.global_vec_ro() as v:
            v.copy(Y)

    @profiler()
    def petsc_mat(self):
        """
        Return a petsc4py.PETSc.Mat with this AllAtOnceJacobian as the python context.
        """
        mat = PETSc.Mat().create(comm=self.ensemble.global_comm)
        mat.setType("python")
        sizes = (self.aaofunc.nlocal_dofs, self.aaofunc.nglobal_dofs)
        mat.setSizes((sizes, sizes))
        mat.setPythonContext(self)
        mat.setUp()
        return mat
