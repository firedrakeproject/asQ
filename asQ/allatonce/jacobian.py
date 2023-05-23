import firedrake as fd
from firedrake.petsc import PETSc
from functools import partial

from asQ.profiling import memprofile
from asQ.common import get_option_from_list
from asQ.allatonce.mixin import TimePartitionMixin
from asQ.allatonce.function import time_average, AllAtOnceFunction

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
    def __init__(self, aaofunc, aaoform,
                 reference_state=None, snes=None):
        r"""
        Python matrix for the Jacobian of the all at once system
        :param aaofunc: The AllAtOnceSystem object
        """
        self.time_partition_setup(aaofunc.ensemble, aaofunc.time_partition)
        self.aaofunc = aaofunc
        self.aaoform = aaoform
        self.form = aaoform.form

        if snes is not None:
            self.snes = snes
            prefix = snes.getOptionsPrefix()
            prefix += self.prefix

        # function to linearise around, and timestep from end of previous slice
        self.u = AllAtOnceFunction(self.ensemble, self.time_partition,
                                   aaofunc.field_function_space)

        # function the Jacobian acts on, and contribution from timestep at end of previous slice
        self.x = AllAtOnceFunction(self.ensemble, self.time_partition,
                                   aaofunc.field_function_space)

        # output residual, and contribution from timestep at end of previous slice
        self.F = fd.Function(aaofunc.function_space)
        self.Fprev = fd.Function(aaofunc.function_space)

        # working buffers for calculating time average when needed
        self.ureduce = fd.Function(aaofunc.field_function_space)
        self.uwrk = fd.Function(aaofunc.field_function_space)

        # Jform without contributions from the previous step
        self.Jform = fd.derivative(self.aaoform.form, self.u)
        # Jform contributions from the previous step
        self.Jform_prev = fd.derivative(self.aaoform.form, self.urecv)

        # option for what state to linearise around
        valid_jacobian_states = ['current', 'window', 'slice', 'linear', 'initial', 'reference', 'user']

        if snes is None:
            self.jacobian_state = lambda: 'current'
        else:
            state_option = f"{prefix}state"

            self.jacobian_state = partial(get_option_from_list,
                                          state_option, valid_jacobian_states, default_index=0)

        if reference_state is not None:
            self.reference_state = fd.Function(aaofunc.field_function_space)
            self.reference_state.assign(reference_state)
        else:
            reference_state = None

        jacobian_state = self.jacobian_state()

        if jacobian_state == 'reference' and self.reference_state is None:
            raise ValueError("AllAtOnceSystem must be provided a reference state to use \'reference\' for aaofunc_jacobian_state.")

        self.update()

    @PETSc.Log.EventDecorator()
    def update(self, X=None):
        # update the state to linearise around from the current all-at-once solution

        aaofunc = self.aaofunc
        jacobian_state = self.jacobian_state()

        if jacobian_state == 'linear':
            pass

        elif jacobian_state == 'current':
            if X is None:
                self.u.assign(aaofunc)
            else:
                self.u.update(X, blocking=True)

        elif jacobian_state in ('window', 'slice'):
            time_average(aaofunc, self.ureduce, self.uwrk, average=jacobian_state)
            self.u.set_all_fields(self.ureduce)

        elif jacobian_state == 'initial':
            self.u.set_all_fields(aaofunc.initial_condition)

        elif jacobian_state == 'reference':
            self.u.set_all_fields(self.reference_state)

        elif jacobian_state == 'user':
            pass

        return

    @PETSc.Log.EventDecorator()
    @memprofile
    def mult(self, mat, X, Y):

        # we could use nonblocking and overlap comms with assembling Jform
        self.x.update(X, blocking=True)

        # assembly stage
        fd.assemble(fd.action(self.Jform, self.x.function), tensor=self.F)
        fd.assemble(fd.action(self.Jform_prev, self.x.uprev),
                    tensor=self.Fprev)
        self.F += self.Fprev

        # Apply boundary conditions
        # For Jacobian action we should just return the values in X
        # at boundary nodes
        for bc in self.aaoform.bcs:
            bc.homogenize()
            bc.apply(self.F, u=self.x.function)
            bc.restore()

        with self.F.dat.vec_ro as v:
            v.copy(Y)
