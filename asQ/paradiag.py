import firedrake as fd
from firedrake.petsc import PETSc
from .profiling import memprofile

from asQ.allatonce import AllAtOnceFunction, AllAtOnceForm, AllAtOnceSolver
from asQ.allatonce.mixin import TimePartitionMixin
from asQ.parallel_arrays import SharedArray

__all__ = ['create_ensemble', 'Paradiag']


def create_ensemble(time_partition, comm=fd.COMM_WORLD):
    '''
    Create an Ensemble for the given slice partition
    Checks that the number of slices and the size of the communicator are compatible

    :arg time_partition: a list of integers, the number of timesteps on each time-rank
    :arg comm: the global communicator for the ensemble
    '''
    nslices = len(time_partition)
    nranks = comm.size

    if nranks % nslices != 0:
        raise ValueError("Number of time slices must be exact factor of number of MPI ranks")

    nspatial_domains = nranks//nslices

    return fd.Ensemble(comm, nspatial_domains)


class Paradiag(TimePartitionMixin):
    @memprofile
    def __init__(self, ensemble,
                 form_function, form_mass,
                 ics, dt, theta,
                 time_partition,
                 solver_parameters={},
                 appctx={}, bcs=[],
                 options_prefix="",
                 reference_state=None,
                 function_alpha=None, jacobian_alpha=None,
                 jacobian_function=None, jacobian_mass=None,
                 pc_function=None, pc_mass=None,
                 pre_function_callback=None, post_function_callback=None,
                 pre_jacobian_callback=None, post_jacobian_callback=None):
        """A class to implement paradiag timestepping.

        :arg ensemble: the ensemble communicator
        :arg form_function: a function that returns a form
            on ics.function_space() providing f(w) for the ODE w_t + f(w) = 0.
        :arg form_mass: a function that returns a linear form on
            ics.function_space providing the mass operator for the time derivative.
        :arg linearised_function: a function that returns a form
            on ics.function_space() which will be linearised to approximate
            derivative(f(w), w) for the ODE w_t + f(w) = 0.
        :arg linearised_mass: a function that returns a linear form on ics.function_space
            providing which will be linearised to approximate the form_mass
        :arg ics: a Function from ics.function_space containing the initial data.
        :arg dt: float, the timestep size.
        :arg theta: float, implicit timestepping parameter.
        :arg alpha: float, circulant matrix parameter.
        :arg time_partition: a list of integers, the number of timesteps
            assigned to each rank.
        :arg bcs: a list of DirichletBC boundary conditions on ics.function_space.
        :arg solver_parameters: options dictionary for nonlinear solver.
        :arg reference_state: a Function in W to use as a reference state
            e.g. in DiagFFTPC.
        :arg circ: a string describing the option on where to use the
            alpha-circulant modification. "picard" - do a nonlinear wave
            form relaxation method. "quasi" - do a modified Newton
            method with alpha-circulant modification added to the
            Jacobian. To make the alpha circulant modification only in the
            preconditioner, simply set ksp_type:preonly in the solve options.
        :arg tol: float, the tolerance for the relaxation method (if used)
        :arg maxits: integer, the maximum number of iterations for the
            relaxation method, if used.
        :arg block_ctx: non-petsc context for solvers.
        :arg block_mat_type: set the type of the diagonal block systems.
            Default is aij.
        """
        self.time_partition_setup(ensemble, time_partition)

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
                                      jacobian_reference_state=reference_state)

        # iteration counts
        self.block_iterations = SharedArray(self.time_partition,
                                            dtype=int,
                                            comm=self.ensemble.ensemble_comm)
        self.reset_diagnostics()

    def reset_diagnostics(self):
        """
        Set all diagnostic information to initial values, e.g. iteration counts to zero
        """
        self.linear_iterations = 0
        self.nonlinear_iterations = 0
        self.total_timesteps = 0
        self.total_windows = 0
        self.block_iterations.data()[:] = 0

    def _record_diagnostics(self):
        """
        Update diagnostic information from snes.

        Must be called exactly once after each snes solve.
        """
        self.linear_iterations += self.solver.snes.getLinearSolveIterations()
        self.nonlinear_iterations += self.solver.snes.getIterationNumber()
        self.total_timesteps += sum(self.time_partition)
        self.total_windows += 1

    def sync_diagnostics(self):
        """
        Synchronise diagnostic information over all time-ranks.

        Until this method is called, diagnostic information is not guaranteed to be valid.
        """
        pc_block_iterations = self.solver.jacobian.pc.block_iterations
        pc_block_iterations.synchronise()
        self.block_iterations.data(deepcopy=False)[:] = pc_block_iterations.data(deepcopy=False)

    @PETSc.Log.EventDecorator()
    @memprofile
    def solve(self,
              nwindows=1,
              preproc=lambda pdg, w, rhs: None,
              postproc=lambda pdg, w, rhs: None,
              rhs=None,
              verbose=False):
        """
        Solve the system (either in one shot or as a relaxation method).

        preproc and postproc must have call signature (paradiag, int)
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
            is_linear = (
                'snes_type' in self.solver.flat_solver_parameters
                and self.solver.flat_solver_parameters['snes_type'] == 'ksponly'
            )
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

        self.sync_diagnostics()
