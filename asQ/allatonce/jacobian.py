import firedrake as fd
from firedrake.petsc import PETSc

from asQ.profiling import profiler
from asQ.common import get_option_from_list
from asQ.allatonce.mixin import TimePartitionMixin
from asQ.allatonce.function import time_average

from functools import cached_property

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
        self._xn = fd.Function(self.x.field_function_space)
        self._xn1 = fd.Function(self.x.field_function_space)

        # output residual, and contribution from timestep at end of previous slice
        self.F = aaoform.F.copy(copy_values=False)
        self.Fprev = aaoform.F.copy(copy_values=False)

        # working buffers for calculating time average when needed
        self.ureduce = fd.Function(aaofunc.field_function_space)
        self.uwrk = fd.Function(aaofunc.field_function_space)

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
    def update(self, x=None):
        """
        Update the state to linearise around according to aaos_jacobian_state.

        :arg x: an optional AllAtOnceFunction or global PETSc Vec.
            If x is not None then the state is updated from x.
        """

        aaofunc = self.aaofunc
        jacobian_state = self.jacobian_state

        if jacobian_state in ('linear', 'user'):
            return

        if x is not None:
            self.aaofunc.assign(x)

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

    @cached_property
    def monolithic_form(self):
        return fd.derivative(self.aaoform.monolithic_form,
                             self.aaoform.aaofunc.function)

    @cached_property
    def monolithic_action(self):
        return fd.action(self.monolithic_form,
                         self.x.function)

    @cached_property
    def monolithic_form_prev(self):
        return fd.derivative(self.aaoform.monolithic_form,
                             self.aaoform.aaofunc.uprev)

    @cached_property
    def monolithic_action_prev(self):
        return fd.action(self.monolithic_form_prev,
                         self.x.uprev)

    @cached_property
    def stepwise_forms_implicit(self):
        return tuple(
            fd.derivative(self.aaoform.stepwise_forms[n],
                          self.aaoform.aaofunc[n])
            for n in range(self.nlocal_timesteps))

    @cached_property
    def stepwise_actions_implicit(self):
        return tuple(
            fd.action(self.stepwise_forms_implicit[n],
                      self.x[n])
            for n in range(self.nlocal_timesteps))

    @cached_property
    def stepwise_forms_explicit(self):
        return tuple(
            fd.derivative(self.aaoform.stepwise_forms[n],
                          self.aaoform.aaofunc[n-1])
            for n in range(1, self.nlocal_timesteps))

    @cached_property
    def stepwise_actions_explicit(self):
        return tuple(
            fd.action(self.stepwise_forms_explicit[n],
                      self.x[n])
            for n in range(self.nlocal_timesteps-1))

    @cached_property
    def stepwise_form_prev(self):
        return fd.derivative(
            self.aaoform.stepwise_forms[0],
            self.aaoform.aaofunc.uprev)

    @cached_property
    def stepwise_action_prev(self):
        return fd.action(self.stepwise_form_prev,
                         self.x.uprev)

    @cached_property
    def singlestep_form_implicit(self):
        return fd.derivative(
            self.aaoform.singlestep_form,
            self.aaoform._un1)

    @cached_property
    def singlestep_action_implicit(self):
        return fd.action(
            self.singlestep_form_implicit,
            self._xn1)

    @cached_property
    def singlestep_form_explicit(self):
        return fd.derivative(
            self.aaoform.singlestep_form,
            self.aaoform._un)

    @cached_property
    def singlestep_action_explicit(self):
        return fd.action(
            self.singlestep_form_explicit,
            self._xn)

    @profiler()
    def mult(self, A, x, y):
        """
        Apply the action of the matrix to a PETSc Vec.

        :arg x: a PETSc Vec to apply the action on.
        :arg y: a PETSc Vec for the result.
        """
        if self.aaoform.construct_type == "monolithic":
            self._mult_monolithic(A, x, y)
        elif self.aaoform.construct_type == "step-wise":
            self._mult_stepwise(A, x, y)
        elif self.aaoform.construct_type == "single-step":
            self._mult_singlestep(A, x, y)

    @profiler()
    def _mult_monolithic(self, A, x, y):
        """
        Apply the action of the matrix to a PETSc Vec.

        :arg x: a PETSc Vec to apply the action on.
        :arg y: a PETSc Vec for the result.
        """
        # we could use nonblocking here and overlap comms with assembling form
        self.x.assign(x, update_halos=True, blocking=True)

        # We use the same strategy as the implicit matrix context in firedrake
        # for dealing with the boundary nodes. From the comments in that file:

        # The matrix has an identity block corresponding to the Dirichlet
        # boundary conditions.
        # Our algorithm in this case is to save the BC values, zero them
        # out before computing the action so that they don't pollute
        # anything, and then set the values into the result.
        # This has the effect of applying [ A_ii 0 ; 0 A_bb ] where A_ii
        # is the block corresponding only to (non-fixed) internal dofs
        # and A_bb=I is the identity block on the (fixed) boundary dofs.

        # Zero the boundary nodes on the input so that A_ib = A_01 = 0
        for bc in self.aaoform.monolithic_bcs:
            bc.zero(self.x.function)

        # assembly stage
        fd.assemble(self.monolithic_action,
                    bcs=self.aaoform.monolithic_bcs,
                    tensor=self.F.cofunction)

        if self.aaoform.use_halo:
            # repeat for the halo part of the matrix action
            for bc in self.aaoform.field_bcs:
                bc.zero(self.x.uprev)

            fd.assemble(self.monolithic_action_prev,
                        bcs=self.aaoform.monolithic_bcs,
                        tensor=self.Fprev.cofunction)
            self.F.cofunction += self.Fprev.cofunction

        if len(self.aaoform.field_bcs) > 0:
            # just using Fprev as a buffer for the original values
            Fbuf = self.Fprev
            with Fbuf.global_vec_wo() as fvec:
                x.copy(fvec)

            # Set the output boundary nodes to the input boundary nodes.
            # This is equivalent to setting [A_bi, A_bb] = [0 I]
            for bc in self.aaoform.monolithic_bcs:
                bc.set(self.F.cofunction, Fbuf.cofunction)

        with self.F.global_vec_ro() as v:
            v.copy(y)

    @profiler()
    def _mult_stepwise(self, A, x, y):
        """
        Apply the action of the matrix to a PETSc Vec.

        :arg x: a PETSc Vec to apply the action on.
        :arg y: a PETSc Vec for the result.
        """
        # we could use nonblocking here and overlap comms with assembling form
        self.x.assign(x, update_halos=True, blocking=True)

        # We use the same strategy as the implicit matrix context in firedrake
        # for dealing with the boundary nodes. From the comments in that file:

        # The matrix has an identity block corresponding to the Dirichlet
        # boundary conditions.
        # Our algorithm in this case is to save the BC values, zero them
        # out before computing the action so that they don't pollute
        # anything, and then set the values into the result.
        # This has the effect of applying [ A_ii 0 ; 0 A_bb ] where A_ii
        # is the block corresponding only to (non-fixed) internal dofs
        # and A_bb=I is the identity block on the (fixed) boundary dofs.

        # Zero the boundary nodes on the input so that A_ib = A_01 = 0
        for n in range(self.nlocal_timesteps):
            for bc in self.aaoform.stepwise_bcs[n]:
                bc.zero(self.x[n])

        Fim = fd.Cofunction(self.F.field_function_space)
        Fex = fd.Cofunction(self.F.field_function_space)

        for n in range(self.nlocal_timesteps):
            Fim.zero()
            Fex.zero()

            fd.assemble(self.stepwise_actions_implicit[n],
                        bcs=self.aaoform.stepwise_bcs[n],
                        tensor=Fim)

            if n > 0:
                fd.assemble(self.stepwise_actions_explicit[n-1],
                            bcs=self.aaoform.stepwise_bcs[n-1],
                            tensor=Fex)

            self.F[n].assign(Fim + Fex)

        if self.aaoform.use_halo:
            # repeat for the halo part of the matrix action
            for bc in self.aaoform.field_bcs:
                bc.zero(self.x.uprev)

            fd.assemble(self.stepwise_action_prev,
                        bcs=self.aaoform.stepwise_bcs[0],
                        tensor=Fex)

            self.F[0].assign(self.F[0] + Fex)

        if len(self.aaoform.field_bcs) > 0:
            # just using Fprev as a buffer for the original values
            Fbuf = self.Fprev
            with Fbuf.global_vec_wo() as fvec:
                x.copy(fvec)

            # Set the output boundary nodes to the input boundary nodes.
            # This is equivalent to setting [A_bi, A_bb] = [0 I]
            for n in range(self.nlocal_timesteps):
                for bc in self.aaoform.stepwise_bcs[n]:
                    bc.set(self.F[n], Fbuf[n])

        with self.F.global_vec_ro() as v:
            v.copy(y)

    @profiler()
    def _mult_singlestep(self, A, x, y):
        """
        Apply the action of the matrix to a PETSc Vec.

        :arg x: a PETSc Vec to apply the action on.
        :arg y: a PETSc Vec for the result.
        """
        # we could use nonblocking here and overlap comms with assembling form
        self.x.assign(x, update_halos=True, blocking=True)

        # We use the same strategy as the implicit matrix context in firedrake
        # for dealing with the boundary nodes. From the comments in that file:

        # The matrix has an identity block corresponding to the Dirichlet
        # boundary conditions.
        # Our algorithm in this case is to save the BC values, zero them
        # out before computing the action so that they don't pollute
        # anything, and then set the values into the result.
        # This has the effect of applying [ A_ii 0 ; 0 A_bb ] where A_ii
        # is the block corresponding only to (non-fixed) internal dofs
        # and A_bb=I is the identity block on the (fixed) boundary dofs.

        # Zero the boundary nodes on the input so that A_ib = A_01 = 0
        for n in range(self.nlocal_timesteps):
            for bc in self.aaoform.singlestep_bcs:
                bc.zero(self.x[n])

        Fim = fd.Cofunction(self.F.field_function_space)
        Fex = fd.Cofunction(self.F.field_function_space)

        for n in range(self.nlocal_timesteps):
            Fim.zero()
            Fex.zero()

            self.aaoform._tn1.assign(self.aaoform.time[n])

            self._xn1.assign(self.x[n])
            fd.assemble(self.singlestep_action_implicit,
                        bcs=self.aaoform.singlestep_bcs,
                        tensor=Fim)

            if n > 0:
                self._xn.assign(self.x[n-1])
                fd.assemble(self.singlestep_action_explicit,
                            bcs=self.aaoform.singlestep_bcs,
                            tensor=Fex)

            self.F[n].assign(Fim + Fex)

        if self.aaoform.use_halo:
            # repeat for the halo part of the matrix action
            for bc in self.aaoform.field_bcs:
                bc.zero(self.x.uprev)

            self._xn.assign(self.x.uprev)
            self.aaoform._tn1.assign(self.aaoform.time[0])
            fd.assemble(self.singlestep_action_explicit,
                        bcs=self.aaoform.singlestep_bcs,
                        tensor=Fex)

            self.F[0].assign(self.F[0] + Fex)

        if len(self.aaoform.field_bcs) > 0:
            # just using Fprev as a buffer for the original values
            Fbuf = self.Fprev
            with Fbuf.global_vec_wo() as fvec:
                x.copy(fvec)

            # Set the output boundary nodes to the input boundary nodes.
            # This is equivalent to setting [A_bi, A_bb] = [0 I]
            for n in range(self.nlocal_timesteps):
                for bc in self.aaoform.singlestep_bcs:
                    bc.set(self.F[n], Fbuf[n])

        with self.F.global_vec_ro() as v:
            v.copy(y)

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
