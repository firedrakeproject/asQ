import firedrake as fd
from firedrake.petsc import PETSc

from asQ.profiling import memprofile
from asQ.allatonce import AllAtOnceFunction
from asQ.allatonce.mixin import TimePartitionMixin

__all__ = ['AllAtOnceForm']


class AllAtOnceForm(TimePartitionMixin):
    @memprofile
    def __init__(self,
                 aaofunc, dt, theta,
                 form_mass, form_function,
                 bcs=[], alpha=None):
        """
        The all-at-once form representing the implicit theta-method over multiple timesteps of a time-dependent finite-element problem.

        :arg aaofunction: AllAtOnceFunction to create the form on.
        :arg dt: float, the timestep size.
        :arg theta: float, implicit timestepping parameter.
        :arg form_function: a function that returns a form on aaofunction.field_function_space
            providing f(w) for the ODE w_t + f(w) = 0.
        :arg form_mass: a function that returns a linear form on aaofunction.field_function_space
            providing the mass operator for the time derivative.
        :arg bcs: a list of DirichletBC boundary conditions on w0.function_space.
        :arg alpha: float if None, circulant matrix parameter
        """
        self.time_partition_setup(aaofunc.ensemble, aaofunc.time_partition)

        self.aaofunc = aaofunc

        self.field_function_space = aaofunc.field_function_space
        self.function_space = aaofunc.function_space

        self.dt = fd.Constant(dt)
        self.theta = fd.Constant(theta)

        self.form_mass = form_mass
        self.form_function = form_function

        self.alpha = None if alpha is None else fd.Constant(alpha)

        # should this make a copy of bcs instead of taking a reference?
        self.field_bcs = bcs
        self.bcs = self._set_bcs(self.field_bcs)

        for bc in self.bcs:
            bc.apply(aaofunc.function)

        # function to assemble the nonlinear residual into
        self.F = aaofunc.copy()
        self.F.set_all_fields(0)

        self.form = self._construct_form()

    def _set_bcs(self, field_bcs):
        """
        Set the boundary conditions onto solution at each timestep in the all-at-once system.
        Returns a list of all boundary conditions on the all-at-once system.

        :arg bcs: a list of the boundary conditions to apply
        """
        aaofunc = self.aaofunc
        is_mixed_element = isinstance(aaofunc.field_function_space.ufl_element(), fd.MixedElement)

        bcs_all = []
        for bc in field_bcs:
            for step in range(aaofunc.nlocal_timesteps):
                if is_mixed_element:
                    cpt = bc.function_space().index
                else:
                    cpt = 0
                index = aaofunc.transform_index(step, cpt)
                bc_all = fd.DirichletBC(aaofunc.function_space.sub(index),
                                        bc.function_arg,
                                        bc.sub_domain)
                bcs_all.append(bc_all)

        return bcs_all

    def copy(self, aaofunc=None):
        if aaofunc is None:
            aaofunc = self.aaofunc.copy()

        return AllAtOnceForm(aaofunc, self.dt, self.theta,
                             self.form_mass, self.form_function,
                             bcs=self.field_bcs, alpha=self.alpha)

    @PETSc.Log.EventDecorator()
    @memprofile
    def assemble(self, func=None, tensor=None):
        r"""
        This is the function we pass to the snes to assemble
        the nonlinear residual.
        """
        # set current state
        if func is not None:
            self.aaofunc.assign(func, update_halos=False)
        self.aaofunc.update_time_halos()

        # assembly stage
        fd.assemble(self.form, tensor=self.F.function)

        # apply boundary conditions
        for bc in self.bcs:
            bc.apply(self.F.function, u=self.aaofunc.function)

        # copy into return buffer

        if isinstance(tensor, AllAtOnceFunction):
            tensor.assign(self.F)

        elif isinstance(tensor, fd.Function):
            tensor.assign(self.F.function)

        elif isinstance(tensor, PETSc.Vec):
            with self.F.global_vec_ro() as v:
                v.copy(tensor)

        elif tensor is not None:
            raise TypeError(f"tensor must be AllAtOnceFunction, Function, or PETSc.Vec, not {type(tensor)}")

    def _construct_form(self):
        """
        Constructs the bilinear form for the all at once system.
        Specific to the theta-centred Crank-Nicholson method
        """
        aaofunc = self.aaofunc

        funcs = fd.split(aaofunc.function)

        ics = fd.split(aaofunc.initial_condition)
        uprevs = fd.split(aaofunc.uprev)

        form_mass = self.form_mass
        form_function = self.form_function

        test_funcs = fd.TestFunctions(aaofunc.function_space)

        dt = self.dt
        theta = self.theta
        alpha = fd.Constant(0) if self.alpha is None else self.alpha

        def get_step(i):
            return aaofunc.get_field_components(i, funcs=funcs)

        def get_test(i):
            return aaofunc.get_field_components(i, funcs=test_funcs)

        for n in range(self.nlocal_timesteps):

            if n == 0:  # previous timestep is ic or is on previous slice
                if self.time_rank == 0:
                    uns = tuple((ic + alpha*up for ic, up in zip(ics, uprevs)))
                else:
                    uns = uprevs
            else:
                uns = get_step(n-1)

            # current time level
            un1s = get_step(n)
            vs = get_test(n)

            # time derivative
            if n == 0:
                form = (1.0/dt)*form_mass(*un1s, *vs)
            else:
                form += (1.0/dt)*form_mass(*un1s, *vs)
            form -= (1.0/dt)*form_mass(*uns, *vs)

            # vector field
            form += theta*form_function(*un1s, *vs)
            form += (1.0 - theta)*form_function(*uns, *vs)

        return form
