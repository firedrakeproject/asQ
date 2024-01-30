from firedrake.petsc import PETSc
from asQ.profiling import profiler

from asQ.allatonce import AllAtOnceFunction, AllAtOnceForm, AllAtOnceSolver
from asQ.allatonce.mixin import TimePartitionMixin
from asQ.parallel_arrays import SharedArray

__all__ = ['Paradiag']


class Paradiag(TimePartitionMixin):
    @profiler()
    def __init__(self, ensemble,
                 time_partition,
                 form_mass, form_function,
                 ics, dt, theta,
                 solver_parameters={},
                 appctx={}, bcs=[],
                 options_prefix="",
                 reference_state=None,
                 function_alpha=None, jacobian_alpha=None,
                 jacobian_mass=None, jacobian_function=None,
                 pc_mass=None, pc_function=None,
                 pre_function_callback=None, post_function_callback=None,
                 pre_jacobian_callback=None, post_jacobian_callback=None):
        """A class to implement paradiag timestepping.

        :arg ensemble: time-parallel ensemble communicator. The timesteps are partitioned
            over the ensemble members according to time_partition so
            ensemble.ensemble_comm.size == len(time_partition) must be True.
        :arg time_partition: a list of integers for the number of timesteps stored on each
            ensemble rank.
        :arg form_mass: a function that returns a linear form on ics.function_space()
            providing the time derivative mass operator for  the PDE w_t + f(w) = 0.
            Must have signature `def form_mass(*u, *v):` where *u and *v are a split(TrialFunction)
            and a split(TestFunction) from ics.function_space().
        :arg form_function: a function that returns a form on ics.function_space()
            providing f(w) for the PDE w_t + f(w) = 0.
            Must have signature `def form_function(*u, *v):` where *u and *v are a split(Function)
            and a split(TestFunction) from ics.function_space().
        :arg ics: a Function containing the initial conditions.
        :arg dt: float, the timestep size.
        :arg theta: float, implicit timestepping parameter.
        :arg solver_parameters: options dictionary for nonlinear solver.
        :arg appctx: A dictionary containing application context that is
            passed to the preconditioner if matrix-free.
        :arg bcs: a list of DirichletBC boundary conditions on ics.function_space.
        :arg options_prefix: an optional prefix used to distinguish PETSc options.
            Use this option if you want to pass options to the solver from the
            command line in addition to through the solver_parameters dict.
        :arg reference_state: A reference firedrake.Function in ics.function_space().
            Only needed if 'aaos_jacobian_state' or 'diagfft_state' is 'reference'.
        :arg function_alpha: float, circulant matrix parameter used in the nonlinear residual.
            This is used for the waveform relaxation method. If None then no circulant
            approximation used.
        :arg jacobian_alpha: float, circulant matrix parameter used in the Jacobian.
            This introduces the circulant approximation in the AllAtOnceJacobian but not in the
            nonlinear residual. If None then no circulant approximation used in the Jacobian.
        :arg jacobian_mass: equivalent to form_mass, but used to construct the AllAtOnceJacobian
            not the nonlinear residual.
        :arg jacobian_function: equivalent to form_function, but used to construct the
            AllAtOnceJacobian not the nonlinear residual.
        :arg pc_mass: equivalent to form_mass, but used to construct the preconditioner.
        :arg pc_function: equivalent to form_function, but used to construct the preconditioner.
        :arg pre_function_callback: A user-defined function that will be called immediately
            before residual assembly. This can be used, for example, to update a coefficient
            function that has a complicated dependence on the unknown solution.
        :arg post_function_callback: As above, but called immediately after residual assembly.
        :arg pre_jacobian_callback: As above, but called immediately before Jacobian assembly.
        :arg post_jacobian_callback: As above, but called immediately after Jacobian assembly.
        """
        self._time_partition_setup(ensemble, time_partition)

        # all-at-once function and form

        function_space = ics.function_space()
        self.aaofunc = AllAtOnceFunction(ensemble, time_partition,
                                         function_space)
        self.aaofunc.assign(ics)

        self.aaoform = AllAtOnceForm(self.aaofunc, dt, theta,
                                     form_mass, form_function,
                                     bcs=bcs, alpha=function_alpha)

        # all-at-once jacobian
        if jacobian_mass is None:
            jacobian_mass = form_mass
        if jacobian_function is None:
            jacobian_function = form_function

        self.jacobian_aaofunc = self.aaofunc.copy()

        self.jacobian_form = AllAtOnceForm(self.jacobian_aaofunc, dt, theta,
                                           jacobian_mass, jacobian_function,
                                           bcs=bcs, alpha=jacobian_alpha)

        # pass seperate forms to the preconditioner
        if pc_mass is not None:
            appctx['pc_form_mass'] = pc_mass
        if pc_function is not None:
            appctx['pc_form_function'] = pc_function

        self.solver = AllAtOnceSolver(self.aaoform, self.aaofunc,
                                      solver_parameters=solver_parameters,
                                      options_prefix=options_prefix, appctx=appctx,
                                      jacobian_form=self.jacobian_form,
                                      jacobian_reference_state=reference_state,
                                      pre_function_callback=pre_function_callback,
                                      post_function_callback=post_function_callback,
                                      pre_jacobian_callback=pre_jacobian_callback,
                                      post_jacobian_callback=post_jacobian_callback)

        # iteration counts
        self.block_iterations = SharedArray(self.time_partition,
                                            dtype=int,
                                            comm=self.ensemble.ensemble_comm)
        self.reset_diagnostics()

    @profiler()
    def reset_diagnostics(self):
        """
        Set all diagnostic information to initial values, e.g. iteration counts to zero
        """
        self.linear_iterations = 0
        self.nonlinear_iterations = 0
        self.total_timesteps = 0
        self.total_windows = 0
        self.block_iterations.data()[:] = 0

    @profiler()
    def _record_diagnostics(self):
        """
        Update diagnostic information from snes.

        Must be called exactly once after each snes solve.
        """
        self.linear_iterations += self.solver.snes.getLinearSolveIterations()
        self.nonlinear_iterations += self.solver.snes.getIterationNumber()
        self.total_timesteps += sum(self.time_partition)
        self.total_windows += 1

    @profiler()
    def sync_diagnostics(self):
        """
        Synchronise diagnostic information over all time-ranks.

        Until this method is called, diagnostic information is not guaranteed to be valid.
        """
        if hasattr(self.solver.jacobian, "pc"):
            pc_block_iterations = self.solver.jacobian.pc.block_iterations
            pc_block_iterations.synchronise()
            self.block_iterations.data(deepcopy=False)[:] = pc_block_iterations.data(deepcopy=False)

    @profiler()
    def solve(self,
              nwindows=1,
              preproc=lambda pdg, w, rhs: None,
              postproc=lambda pdg, w, rhs: None,
              rhs=None,
              verbose=False):
        """
        Solve multiple windows of the all-at-once system.

        preproc and postproc must have call signature:
            (Paradiag, int, Any[AllAtOnceFunction, None]).

        :arg nwindows: number of windows to solve for
        :arg preproc: callback called before each window solve
        :arg postproc: callback called after each window solve
        """

        for wndw in range(nwindows):

            preproc(self, wndw, rhs)
            self.solver.solve(rhs=rhs)
            self._record_diagnostics()
            postproc(self, wndw, rhs)

            converged_reason = self.solver.snes.getConvergedReason()
            is_linear = self.solver.snes.getType() == 'ksponly'

            if is_linear and (converged_reason == 5):
                pass
            elif not (1 < converged_reason < 5):
                PETSc.Sys.Print(f'SNES diverged with error code {converged_reason}. Cancelling paradiag time integration.')
                return

            # reset window using last timestep as new initial condition
            # but don't wipe all-at-once function at last window
            if wndw != nwindows-1:
                self.aaofunc.bcast_field(-1, self.aaofunc.initial_condition)
                self.aaofunc.assign(self.aaofunc.initial_condition)
                self.aaoform.time_update()
                self.solver.jacobian_form.time_update()
        self.sync_diagnostics()
