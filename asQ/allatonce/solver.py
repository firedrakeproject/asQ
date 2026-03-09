from petsctools import (
    attach_options, set_from_options, get_options, inserted_options)
from firedrake.petsc import PETSc

from asQ.profiling import profiler
from asQ.allatonce import (AllAtOnceCofunction, AllAtOnceFunction,
                           AllAtOnceJacobian)
from asQ.allatonce.mixin import TimePartitionMixin

__all__ = ['AllAtOnceSolver', 'LinearSolver']


class AllAtOnceSolver(TimePartitionMixin):
    @profiler()
    def __init__(self, aaoform, aaofunc,
                 solver_parameters=None,
                 appctx=None,
                 options_prefix=None,
                 jacobian_form=None,
                 jacobian_reference_state=None,
                 pre_function_callback=None,
                 post_function_callback=None,
                 pre_jacobian_callback=None,
                 post_jacobian_callback=None):
        """
        Solve an all-at-once form over an all-at-once function.

        This is used to solve for a timeseries defined by the all-at-once form
        and the initial condition in the all-at-once function.

        :arg aaoform: the AllAtOnceForm to solve.
        :arg aaofunc: the AllAtOnceFunction solution.
        :arg solver_parameters: solver parameters to pass to PETSc.
            This should be a dict mapping PETSc options to values.
        :arg appctx: A dictionary containing application context that is
            passed to the preconditioner if matrix-free.
        :arg options_prefix: an optional prefix used to distinguish PETSc options.
            Use this option if you want to pass options to the solver from the
            command line in addition to through the solver_parameters dict.
        :arg jacobian_form: an AllAtOnceForm to create the AllAtOnceJacobian from.
            Allows the Jacobian to be defined around a form different from the form
            used to assemble the residual.
        :arg jacobian_reference_state: a firedrake.Function to pass to the
            AllAtOnceJacobian as a reference state.
        :arg pre_function_callback: A user-defined function that will be called immediately
            before residual assembly. This can be used, for example, to update a coefficient
            function that has a complicated dependence on the unknown solution.
        :arg post_function_callback: As above, but called immediately after residual assembly.
        :arg pre_jacobian_callback: As above, but called immediately before Jacobian assembly.
        :arg post_jacobian_callback: As above, but called immediately after Jacobian assembly.
        """
        self._time_partition_setup(aaofunc.ensemble, aaofunc.time_partition)
        self.aaofunc = aaofunc
        self.aaoform = aaoform

        self.appctx = appctx or {}

        self.jacobian_form = aaoform.copy() if jacobian_form is None else jacobian_form

        def passthrough(*args, **kwargs):
            pass

        # callbacks
        if pre_function_callback is None:
            self.pre_function_callback = passthrough
        else:
            self.pre_function_callback = pre_function_callback

        if post_function_callback is None:
            self.post_function_callback = passthrough
        else:
            self.post_function_callback = post_function_callback

        if pre_jacobian_callback is None:
            self.pre_jacobian_callback = passthrough
        else:
            self.pre_jacobian_callback = pre_jacobian_callback

        if post_jacobian_callback is None:
            self.post_jacobian_callback = passthrough
        else:
            self.post_jacobian_callback = post_jacobian_callback

        # the nonlinear solver
        self.snes = PETSc.SNES().create(comm=self.ensemble.global_comm)

        attach_options(self.snes, parameters=solver_parameters,
                       options_prefix=options_prefix,
                       default_prefix='asq')
        options_prefix = get_options(self.snes).options_prefix

        def assemble_function(snes, X, F):
            self.pre_function_callback(self, X)
            self.aaoform.assemble(X)
            with aaoform.F.global_vec_ro() as fvec:
                fvec.copy(F)
            self.post_function_callback(self, X, F)

        self.snes.setFunction(assemble_function)

        # Jacobian
        with inserted_options(self.snes):
            self.jacobian = AllAtOnceJacobian(self.jacobian_form,
                                              reference_state=jacobian_reference_state,
                                              options_prefix=options_prefix,
                                              appctx=self.appctx)

        self.jacobian_mat = self.jacobian.petsc_mat()

        def form_jacobian(snes, X, J, P):
            self.pre_jacobian_callback(self, X, J)
            self.jacobian.update(X)
            self.post_jacobian_callback(self, X, J)
            J.assemble()
            P.assemble()

        self.snes.setJacobian(form_jacobian,
                              J=self.jacobian_mat,
                              P=self.jacobian_mat)

        # complete the snes setup
        set_from_options(self.snes)

    @property
    def solver_parameters(self):
        return get_options(self.snes).parameters

    @property
    def options_prefix(self):
        return get_options(self.snes).options_prefix

    @profiler()
    def solve(self, rhs=None):
        """
        Solve the all-at-once system.

        :arg rhs: optional constant part of the system.
        """
        with inserted_options(self.snes):
            with self.aaofunc.global_vec() as gvec:
                if rhs is None:
                    self.snes.solve(None, gvec)
                else:
                    if not isinstance(rhs, AllAtOnceCofunction):
                        msg = f"Right hand side of all-at-once problem must be AllAtOnceCofunction not {type(rhs)}."
                        raise TypeError(msg)
                    with rhs.global_vec_ro() as rvec:
                        self.snes.solve(rvec, gvec)


class LinearSolver(TimePartitionMixin):
    @profiler()
    def __init__(self, aaoform,
                 solver_parameters=None,
                 appctx=None,
                 options_prefix=None):
        """
        Solve a linear system where the matrix is an all-at-once Jacobian.

        This does not solve for a timeseries (use an AllAtOnceSolver if this
        is what you need), but simply uses the AllAtOnceJacobian as the Mat
        for a KSP.

        :arg aaoform: the AllAtOnceForm to form the Jacobian from.
        :arg solver_parameters: solver parameters to pass to PETSc.
            This should be a dict mapping PETSc options to values.
        :arg appctx: A dictionary containing application context that is
            passed to the preconditioner if matrix-free.
        :arg options_prefix: an optional prefix used to distinguish PETSc options.
            Use this option if you want to pass options to the solver from the
            command line in addition to through the solver_parameters dict.
        """
        self._time_partition_setup(aaoform.ensemble, aaoform.time_partition)

        self.aaoform = aaoform
        self.appctx = appctx or {}

        # the linear solver
        self.ksp = PETSc.KSP().create(comm=self.ensemble.global_comm)

        # manage options from both dict and command line
        attach_options(self.ksp, parameters=solver_parameters,
                       options_prefix=options_prefix,
                       default_prefix='asq')
        options_prefix = get_options(self.ksp).options_prefix

        # create the all-at-once jacobian
        with inserted_options(self.ksp):
            self.jacobian = AllAtOnceJacobian(aaoform, appctx=self.appctx,
                                              options_prefix=options_prefix)

        # create petsc matrix
        self.ksp.setOperators(self.jacobian.petsc_mat())

        # finish setting up the ksp
        set_from_options(self.ksp)

    @property
    def solver_parameters(self):
        return get_options(self.snes).parameters

    @property
    def options_prefix(self):
        return get_options(self.snes).options_prefix

    @profiler()
    def solve(self, b, x):
        """
        Solve the all-at-once matrix Ax=b.

        :arg b: AllAtOnceCofunction right hand side vector.
        :arg x: AllAtOnceFunction solution vector.
        """
        if not isinstance(x, AllAtOnceFunction):
            msg = f"Solution of all-at-once problem must be AllAtOnceFunction not {type(x)}."
            raise TypeError(msg)

        if not isinstance(b, AllAtOnceCofunction):
            msg = f"Right hand side of all-at-once problem must be AllAtOnceCofunction not {type(b)}."
            raise TypeError(msg)

        with x.global_vec() as xvec, b.global_vec_ro() as bvec:
            with inserted_options(self.ksp):
                self.ksp.solve(bvec, xvec)
