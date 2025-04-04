import firedrake as fd
from firedrake.assemble import get_assembler

from asQ.profiling import profiler
from asQ.allatonce.function import AllAtOnceCofunction
from asQ.allatonce.mixin import TimePartitionMixin

from functools import cached_property

__all__ = ['AllAtOnceForm']


class AllAtOnceForm(TimePartitionMixin):
    default_form_construct_type = "stepwise"

    @profiler()
    def __init__(self,
                 aaofunc, dt, theta,
                 form_mass, form_function,
                 bcs=[], alpha=None,
                 form_parameters=None):
        """
        The all-at-once form representing the implicit theta-method (trapezium rule version)
        over multiple timesteps of a time-dependent finite-element problem.

        :arg aaofunction: AllAtOnceFunction to create the form over.
        :arg dt: the timestep size.
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
        :arg form_parameters: dict, parameters for constructing the UFL form for the local slice.
            Possible entries:
            - 'form_construct_type': <'monolithic', 'stepwise'>
                - 'monolithic': construct a single UFL form for all timesteps in the slice.
                - 'stepwise': construct a separate UFL form for each timestep in the slice.
            - 'form_compiler_parameters': a dictionary to pass through to the assembly.
        """
        self._time_partition_setup(aaofunc.ensemble, aaofunc.time_partition)

        self.aaofunc = aaofunc
        self.field_function_space = aaofunc.field_function_space
        self.function_space = aaofunc.function_space

        self.dt = fd.Constant(dt)
        self.t0 = fd.Constant(0)
        self.time = tuple(fd.Constant(0) for _ in range(self.aaofunc.nlocal_timesteps))
        self.time_update(self.t0)

        self.theta = fd.Constant(theta)

        self.form_mass = form_mass
        self.form_function = form_function
        self.field_bcs = bcs

        self.alpha = alpha if alpha is None else fd.Constant(alpha)

        self.use_halo = (self.time_rank > 0) or (self.alpha is not None)

        self.form_parameters = form_parameters or {}

        self.construct_type = self.form_parameters.get(
            "form_construct_type",
            self.default_form_construct_type)

        construction_types = ["monolithic", "stepwise"]
        if self.construct_type not in construction_types:
            construction_options = " or ".join(construction_types)
            raise ValueError(
                "'form_construct_type' for AllAtOnceForm must be one"
                f" of {construction_options}, not {self.construct_type}")

        if self.construct_type == "monolithic":
            self.form = self.monolithic_form
            self.bcs = self.monolithic_bcs
        elif self.construct_type == "stepwise":
            self.form = self.stepwise_forms
            self.bcs = self.stepwise_bcs

        self.apply_bcs(aaofunc)

        # cofunction to assemble the nonlinear residual into
        self.F = AllAtOnceCofunction(
            self.ensemble, self.time_partition,
            aaofunc.field_function_space.dual(),
            full_function_space=aaofunc.dual_space,
            full_dual_space=aaofunc.function_space)

    def time_update(self, t=None):
        """
        Update the time points that the form is defined over.

        Default behaviour is to update the initial time t0 to be the
        time of the final timestep. The last timestep of the
        AllAtOnceFunction can then be used as the new initial condition.
        The time at each timestep is updated according to the initial time.

        :arg t: New initial time t0. If None then the current final
            time is used as the new initial time.
        """
        if t is not None:
            self.t0.assign(t)
        else:
            self.t0.assign(self.t0 + self.dt*self.ntimesteps)

        for n in range((self.nlocal_timesteps)):
            widx = 1 + self.aaofunc.transform_index(
                n, from_range='slice', to_range='window')
            self.time[n].assign(self.t0 + self.dt*widx)

    @cached_property
    def monolithic_bcs(self):
        """
        Create a list of  boundary conditions on the all-at-once function space corresponding
        to the boundary conditions `field_bcs` on a single timestep applied to every timestep.

        :arg field_bcs: a list of the boundary conditions to apply.
        """
        aaofunc = self.aaofunc
        is_mixed_element = isinstance(aaofunc.field_function_space.ufl_element(), fd.MixedElement)

        bcs_all = []
        for bc in self.field_bcs:
            for step in range(aaofunc.nlocal_timesteps):
                if is_mixed_element:
                    cpt = bc.function_space().index
                else:
                    cpt = 0
                index = aaofunc._component_indices(step)[cpt]
                bc_all = fd.DirichletBC(aaofunc.function_space.sub(index),
                                        bc.function_arg,
                                        bc.sub_domain)
                bcs_all.append(bc_all)

        return bcs_all

    @cached_property
    def stepwise_bcs(self):
        return tuple(self.field_bcs
                     for _ in range(self.nlocal_timesteps))

    def apply_bcs(self, u, t=None):
        if self.construct_type == "monolithic":
            for bc in self.monolithic_bcs:
                bc.apply(u.function)

        elif self.construct_type == "stepwise":
            for i in range(u.nlocal_timesteps):
                for bc in self.stepwise_bcs[i]:
                    bc.apply(u[i])

    @profiler()
    def copy(self, aaofunc=None):
        """
        Return a copy of the AllAtOnceForm.

        :arg aaofunc: An optional AllAtOnceFunction. If present, the new AllAtOnceForm
            will be defined over aaofunc. If None, the new AllAtOnceForm will be defined
            over a copy of self.aaofunc.
        """
        if aaofunc is None:
            aaofunc = self.aaofunc.copy()

        return AllAtOnceForm(aaofunc, self.dt, self.theta,
                             self.form_mass, self.form_function,
                             bcs=self.field_bcs, alpha=self.alpha,
                             form_parameters=self.form_parameters)

    @profiler()
    def assemble(self, func=None, tensor=None):
        """
        Evaluates the form.

        By default the form will be evaluated at the state in self.aaofunc,
        and the result will be placed into self.F.

        :arg func: may optionally be an AllAtOnceFunction or a global PETSc Vec.
            if not None, the form will be evaluated at the state in `func`.
        :arg tensor: may optionally be an AllAtOnceCofunction, in which case
            the result will be placed into `tensor`.
        """
        if tensor is not None and not isinstance(tensor, AllAtOnceCofunction):
            raise TypeError(f"tensor must be an AllAtOnceCofunction, not {type(tensor)}")

        # Set the current state
        if func is not None:
            self.aaofunc.assign(func, update_halos=False)

        # Assembly stage
        # The residual on the DirichletBC nodes is set to zero,
        # so we need to make sure that the function conforms
        # with the boundary conditions.
        self.apply_bcs(self.aaofunc)
        # for bc in self.bcs:
        #     bc.apply(self.aaofunc.function)

        # Update the halos after enforcing the bcs so we
        # know they are correct. This doesn't make a
        # difference now because we only support constant
        # bcs, but it will be important once we support
        # time-dependent bcs.
        self.aaofunc.update_time_halos()

        if self.construct_type == "monolithic":
            self.monolithic_assemble(tensor=self.F.cofunction)

        elif self.construct_type == "stepwise":
            for n in range(self.nlocal_timesteps):
                self.stepwise_assembles[n](tensor=self.F[n])

        else:
            raise ValueError(
                f"Unrecognised form_construct_type {self.construct_type}")

        if tensor:
            tensor.assign(self.F)
            result = tensor
        else:
            result = self.F
        return result

    @cached_property
    def monolithic_form(self):
        """
        Constructs the all-at-once UFL form for all timesteps on the local slice.
        """
        funcs = fd.split(self.aaofunc.function)
        tests = fd.TestFunctions(self.aaofunc.function_space)

        def cpts(fs, n):
            return tuple(fs[j] for j in self.aaofunc._component_indices(n))

        return sum(
            self._construct_step_form(
                cpts(funcs, n),
                self._get_uns(n, split=True, construct_type='monolithic'),
                cpts(tests, n), self.time[n])
            for n in range(self.nlocal_timesteps))

    @cached_property
    def monolithic_assemble(self):
        fc_params = self.form_parameters.get(
            "form_compiler_parameters", None)
        return get_assembler(
            self.monolithic_form,
            bcs=self.monolithic_bcs,
            form_compiler_parameters=fc_params
        ).assemble

    @cached_property
    def stepwise_forms(self):
        """
        Constructs the UFL forms for each timesteps on the local slice.
        """
        tests = fd.TestFunctions(self.aaofunc.field_function_space)
        return tuple(
            self._construct_step_form(fd.split(self.aaofunc[n]),
                                      self._get_uns(n, split=True,
                                                    construct_type="stepwise"),
                                      tests, self.time[n])
            for n in range(self.nlocal_timesteps))

    @cached_property
    def stepwise_assembles(self):
        fc_params = self.form_parameters.get(
            "form_compiler_parameters", None)
        return tuple(
            get_assembler(
                self.stepwise_forms[n],
                bcs=self.stepwise_bcs[n],
                form_compiler_parameters=fc_params
            ).assemble
            for n in range(self.nlocal_timesteps))

    def _get_uns(self, n, split=False, construct_type=None):
        construct_type = construct_type or self.construct_type

        get = fd.split if split else lambda u: u.subfunctions

        if n > 0:  # previous timestep is ic or is on previous slice
            if construct_type == "monolithic":
                funcs = get(self.aaofunc.function)
                uns = tuple(funcs[j] for j in self.aaofunc._component_indices(n-1))
            else:
                uns = get(self.aaofunc[n-1])
        else:
            uprevs = get(self.aaofunc.uprev)
            if self.time_rank > 0:
                uns = uprevs
            else:
                uns = get(self.aaofunc.initial_condition)
                if self.alpha is not None:
                    uns = tuple(un + self.alpha*up for un, up in zip(uns, uprevs))
        return uns

    def _construct_step_form(self, un1s, uns, vs, t):
        """
        Constructs the (possibly nonlinear) form for a single step.
        Specific to the implicit theta-method (trapezium rule version).
        """
        dt = self.dt
        theta = self.theta
        dt1 = (1.0/dt)
        imw = theta
        exw = (1.0 - theta)

        imt = t
        ext = t - dt

        M = dt1*(self.form_mass(*un1s, *vs) - self.form_mass(*uns, *vs))

        Kim = imw*self.form_function(*un1s, *vs, imt)
        Kex = exw*self.form_function(*uns, *vs, ext)
        K = Kim + Kex

        return M + K
