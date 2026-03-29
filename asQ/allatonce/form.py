from functools import cached_property
import firedrake as fd

from asQ.profiling import profiler
from asQ.allatonce import AllAtOnceCofunction
from asQ.allatonce.mixin import TimePartitionMixin

from functools import partial

__all__ = ['AllAtOnceForm']


class AllAtOnceForm(TimePartitionMixin):
    @profiler()
    def __init__(self, aaofunc, theta,
                 form_mass, form_function,
                 bcs=None, alpha=None):
        """
        The all-at-once form representing the implicit theta-method (trapezium rule version)
        over multiple timesteps of a time-dependent finite-element problem.

        :arg aaofunction: AllAtOnceFunction to create the form over.
        :arg theta: implicit timestepping parameter.
        :arg form_mass: a function that returns a linear form on aaofunction.field_function_space
            providing the mass operator for the time derivative.
            Must have signature `def form_mass(*u, *v):` where *u and *v are a split(TrialFunction)
            and a split(TestFunction) from aaofunction.field_function_space.
        :arg form_function: a function that returns a form on aaofunction.field_function_space
            providing f(w) for the ODE w_t + f(w) = 0.
            Must have signature `def form_function(*u, *v):` where *u and *v are a split(Function)
            and a split(TestFunction) from aaofunction.field_function_space.
        :arg bcs: a list of DirichletBC boundary conditions on aaofunc.field_function_space.
        :arg alpha: float, circulant matrix parameter. if None then no circulant approximation used.
        """

        self.aaofunc = aaofunc
        self._theta = fd.Constant(theta)

        self.form_mass = form_mass
        self.form_function = form_function

        self.alpha = None if alpha is None else fd.Constant(alpha)

        # should this make a copy of bcs instead of taking a reference?
        self._timestep_bcs = bcs or []

    @property
    def theta(self):
        return self._theta

    @property
    def timestep_bcs(self):
        return self._timestep_bcs

    @cached_property
    def bcs(self):
        """
        Create a list of  boundary conditions on the all-at-once function space corresponding
        to the boundary conditions `timestep_bcs` on a single timestep applied to every timestep.

        :arg timestep_bcs: a list of the boundary conditions to apply.
        """
        W = self.aaofunc.function_space()
        V = W.timestep_function_space
        is_mixed_element = isinstance(V.ufl_element(), fd.MixedElement)

        bcs_all = []
        for bc in self.timestep_bcs:
            for step in range(W.domain.nlocal_steps):
                if is_mixed_element:
                    cpt = bc.function_space().index
                else:
                    cpt = 0
                index = W._component_indices(step)[cpt]
                bc_all = fd.DirichletBC(W._full_local_space.sub(index),
                                        bc.function_arg,
                                        bc.sub_domain)
                bcs_all.append(bc_all)

        return bcs_all

    @profiler()
    def assemble(self, tensor=None):
        """
        Evaluates the form.

        By default the form will be evaluated at the state in self.aaofunc,
        and the result will be placed into self.F.

        :arg tensor: may optionally be an AllAtOnceCofunction, in which case
            the result will be placed into `tensor`.
        """
        if tensor is not None and not isinstance(tensor, AllAtOnceCofunction):
            raise TypeError(f"tensor must be an AllAtOnceCofunction, not {type(tensor)}")

        # Assembly stage
        # The residual on the DirichletBC nodes is set to zero,
        # so we need to make sure that the function conforms
        # with the boundary conditions.
        for bc in self.bcs:
            bc.apply(self.aaofunc.function)

        # Update the halos after enforcing the bcs so we
        # know they are correct. This doesn't make a
        # difference now because we only support constant
        # bcs, but it will be important once we support
        # time-dependent bcs.
        self.aaofunc.update_time_halos()

        fd.assemble(self.form, bcs=self.bcs,
                    tensor=self.F.cofunction)

        if tensor:
            tensor.assign(self.F)
            result = tensor
        else:
            result = self.F.copy()
        return result

    @cached_property
    def F(self):
        """AllAtOnceCofunction to assemble residual into."""
        self.F = AllAtOnceCofunction(self.aaofunc.function_space().dual())

    @cached_property
    def form(self):
        """
        The (possibly nonlinear) form for the all at once system.
        Specific to the implicit theta-method (trapezium rule version).
        """
        aaofunc = self.aaofunc
        W = aaofunc.function_space()
        funcs = fd.split(aaofunc._full_local_function)

        ics = fd.split(aaofunc.initial_condition)
        uprevs = fd.split(aaofunc.uprev)

        form_mass = self.form_mass
        form_function = self.form_function

        test = fd.TestFunction(W._full_local_space)
        tests = fd.split(test)

        dt = W.domain.dt
        theta = self.theta

        def get_components(i, funcs=None):
            return tuple(funcs[j] for j in W._component_indices(i))

        get_step = partial(get_components, funcs=funcs)
        get_test = partial(get_components, funcs=tests)

        form = fd.ZeroBaseForm(test)

        for n in range(W.domain.nlocal_timesteps):

            if n == 0:  # previous timestep is ic or is on previous slice
                if W.domain.time_rank == 0:
                    uns = ics
                    if self.alpha is not None:
                        uns = tuple(un + self.alpha*up for un, up in zip(uns, uprevs))
                    tn = W.domain.t0
                else:
                    uns = uprevs
                    tn = W.domain.times[0] - dt
            else:
                uns = get_step(n-1)
                tn = W.domain.times[n-1]

            # current time level
            un1s = get_step(n)
            vs = get_test(n)
            tn1 = W.domain.times[n]

            # time derivative
            form += (1.0/dt)*form_mass(*un1s, *vs)
            form -= (1.0/dt)*form_mass(*uns, *vs)

            # spatial terms
            form += theta*form_function(*un1s, *vs, tn1)
            form += (1.0 - theta)*form_function(*uns, *vs, tn)

        return form
