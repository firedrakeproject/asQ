import firedrake as fd
from firedrake.petsc import PETSc
from functools import partial

from asQ.profiling import memprofile
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

        'current': use the current state of the AllAtOnceSystem (i.e. current Newton iterate).
        'window': use the time average over the entire AllAtOnceSystem.
        'slice': use the time average over timesteps on the local Ensemble member.
        'linear': the form being linearised is linear, so no update to the state is needed.
        'initial': use the initial condition is used for all timesteps.
        'reference': use the reference state of the AllAtOnceSystem for all timesteps.
        'user': the state will be set manually by the user so no update is needed.
    """
    prefix = "aaos_jacobian_"

    @memprofile
    def __init__(self, aaoform, current_state,
                 reference_state=None,
                 options_prefix="",
                 appctx={}):
        """
        Python matrix for the Jacobian of the all at once system
        :arg aaofunc: The AllAtOnceSystem object
        """
        self.time_partition_setup(aaoform.ensemble, aaoform.time_partition)

        aaofunc = aaoform.aaofunc
        self.aaoform = aaoform
        self.aaofunc = aaofunc

        self.current_state = current_state

        default_ctx = {
            'form_mass': aaoform.form_mass,
            'form_function': aaoform.form_function
        }
        for k, v in default_ctx.items():
            if k not in appctx:
                appctx[k] = v

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
        # form contributions from the previous step
        self.form_prev = fd.derivative(aaoform.form, aaofunc.uprev)

        # option for what state to linearise around
        valid_jacobian_states = ['current', 'window', 'slice', 'linear', 'initial', 'reference', 'user']

        if (options_prefix != "") and (not options_prefix.endswith("_")):
            options_prefix += "_"

        state_option = f"{options_prefix}{self.prefix}state"

        self.jacobian_state = partial(get_option_from_list,
                                      state_option, valid_jacobian_states,
                                      default_index=0)

        if reference_state is not None:
            self.reference_state = fd.Function(aaofunc.field_function_space)
            self.reference_state.assign(reference_state)
        else:
            self.reference_state = None

        jacobian_state = self.jacobian_state()

        if jacobian_state == 'reference' and self.reference_state is None:
            raise ValueError("AllAtOnceJacobian must be provided a reference state to use \'reference\' for aaos_jacobian_state.")

        self.update()

    @PETSc.Log.EventDecorator()
    def update(self, X=None):
        """
        Update the state to linearise around from the current all-at-once solution.
        """

        aaofunc = self.aaofunc
        jacobian_state = self.jacobian_state()

        if jacobian_state == 'linear':
            pass

        elif jacobian_state == 'current':
            if X is None:
                X = self.current_state
            self.aaofunc.assign(X)

        elif jacobian_state in ('window', 'slice'):
            time_average(self.current_state, self.ureduce, self.uwrk, average=jacobian_state)
            aaofunc.set_all_fields(self.ureduce)

        elif jacobian_state == 'initial':
            aaofunc.set_all_fields(self.current_state.initial_condition)

        elif jacobian_state == 'reference':
            aaofunc.set_all_fields(self.reference_state)

        elif jacobian_state == 'user':
            pass

        return

    @PETSc.Log.EventDecorator()
    @memprofile
    def mult(self, mat, X, Y):

        # we could use nonblocking here and overlap comms with assembling form
        self.x.assign(X, update_halos=True, blocking=True)

        # assembly stage
        fd.assemble(fd.action(self.form, self.x.function), tensor=self.F.function)
        fd.assemble(fd.action(self.form_prev, self.x.uprev), tensor=self.Fprev)
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
