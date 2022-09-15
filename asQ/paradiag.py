import firedrake as fd
from firedrake.petsc import flatten_parameters
from firedrake.petsc import PETSc, OptionsManager
from functools import partial

from asQ.allatoncesystem import AllAtOnceSystem

appctx = {}


def context_callback(pc, context):
    return context


get_context = partial(context_callback, context=appctx)


def create_ensemble(slice_partition, comm=fd.COMM_WORLD):
    '''
    Create an Ensemble for the given slice partition
    Checks that the number of slices and the size of the communicator are compatible

    :arg slice_partition: a list of integers, the number of timesteps on each time-rank
    :arg comm: the global communicator for the ensemble
    '''
    nslices = len(slice_partition)
    nranks = comm.size

    if nranks % nslices != 0:
        raise ValueError("Number of time slices must be exact factor of number of MPI ranks")

    nspatial_domains = nranks/nslices

    return fd.Ensemble(comm, nspatial_domains)


class paradiag(object):
    def __init__(self, ensemble,
                 form_function, form_mass, W, w0, dt, theta,
                 alpha, slice_partition, bcs=[],
                 solver_parameters={},
                 circ="picard",
                 tol=1.0e-6, maxits=10,
                 ctx={}, block_ctx={}, block_mat_type="aij"):
        """A class to implement paradiag timestepping.

        :arg ensemble: the ensemble communicator
        :arg form_function: a function that returns a linear form
        on W providing f(w) for the ODE w_t + f(w) = 0.
        :arg form_mass: a function that returns a linear form
        on W providing the mass operator for the time derivative.
        :arg W: the FunctionSpace on which f is defined.
        :arg w0: a Function from W containing the initial data
        :arg dt: float, the timestep size.
        :arg theta: float, implicit timestepping parameter
        :arg alpha: float, circulant matrix parameter
        :arg slice_partition: a list of integers, the number of timesteps
        assigned to each rank
        :arg bcs: a list of DirichletBC boundary conditions on W
        :arg solver_parameters: options dictionary for nonlinear solver
        :arg circ: a string describing the option on where to use the
        alpha-circulant modification. "picard" - do a nonlinear wave
        form relaxation method. "quasi" - do a modified Newton
        method with alpha-circulant modification added to the
        Jacobian. To make the alpha circulant modification only in the
        preconditioner, simply set ksp_type:preonly in the solve options.
        :arg tol: float, the tolerance for the relaxation method (if used)
        :arg maxits: integer, the maximum number of iterations for the
        relaxation method, if used.
        :arg ctx: application context for solvers.
        :arg block_ctx: non-petsc context for solvers.
        :arg block_mat_type: set the type of the diagonal block systems.
        Default is aij.
        """

        self.aaos = AllAtOnceSystem(ensemble, slice_partition,
                                    dt, theta,
                                    form_mass, form_function,
                                    w0, bcs,
                                    circ, alpha)

        self.ensemble = ensemble
        self.slice_partition = self.aaos.slice_partition
        self.time_rank = self.aaos.time_rank
        self.alpha = self.aaos.alpha
        self.tol = tol
        self.maxits = maxits
        self.circ = self.aaos.circ
        self.ctx = ctx
        self.block_ctx = block_ctx

        # set up the PETSc Vecs (X for coeffs and F for residuals)
        W = self.aaos.function_space

        nlocal = slice_partition[self.time_rank]*W.node_set.size  # local times x local space
        nglobal = sum(slice_partition)*W.dim()  # global times x global space

        self.X = PETSc.Vec().create(comm=fd.COMM_WORLD)
        self.X.setSizes((nlocal, nglobal))
        self.X.setFromOptions()
        # copy initial data into the PETSc vec
        with self.aaos.w_all.dat.vec_ro as v:
            v.copy(self.X)
        self.F = self.X.copy()

        # sort out the appctx
        if "pc_python_type" in solver_parameters:
            if solver_parameters["pc_python_type"] == "asQ.DiagFFTPC":
                appctx["paradiag"] = self
                solver_parameters["diagfft_context"] = "asQ.paradiag.get_context"
        solver_parameters = flatten_parameters(solver_parameters)

        # set up the snes
        self.snes = PETSc.SNES().create(comm=fd.COMM_WORLD)
        self.opts = OptionsManager(solver_parameters, '')
        self.snes.setOptionsPrefix('')
        self.snes.setFunction(self.aaos._assemble_function, self.F)

        # set up the Jacobian
        Jacmat = PETSc.Mat().create(comm=fd.COMM_WORLD)
        Jacmat.setType("python")
        Jacmat.setSizes(((nlocal, nglobal), (nlocal, nglobal)))
        Jacmat.setPythonContext(self.aaos.jacobian)
        Jacmat.setUp()
        self.Jacmat = Jacmat

        def form_jacobian(snes, X, J, P):
            # copy the snes state vector into self.X
            X.copy(self.X)
            self.aaos.update(X)
            J.assemble()
            P.assemble()

        self.snes.setJacobian(form_jacobian, J=Jacmat, P=Jacmat)

        # complete the snes setup
        self.opts.set_from_options(self.snes)

    def solve(self,
              nwindows=1,
              preproc=lambda pdg, w: None,
              postproc=lambda pdg, w: None,
              verbose=False):
        """
        Solve the system (either in one shot or as a relaxation method).

        preproc and postproc must have call signature (paradiag, int)
        :arg nwindows: number of windows to solve for
        :arg preproc: callback called before each window solve
        :arg postproc: callback called after each window solve
        """

        for wndw in range(nwindows):

            preproc(self, wndw)

            with self.opts.inserted_options():
                self.snes.solve(None, self.X)
            self.aaos.update(self.X)

            postproc(self, wndw)

            if not (1 < self.snes.getConvergedReason() < 5):
                PETSc.Sys.Print(f'SNES diverged with error code {self.snes.getConvergedReason()}. Cancelling paradiag time integration.')
                return

            if wndw != nwindows-1:
                self.aaos.next_window()
