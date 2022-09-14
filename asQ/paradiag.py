import firedrake as fd
from firedrake.petsc import flatten_parameters
from firedrake.petsc import PETSc, OptionsManager
from pyop2.mpi import MPI
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


class JacobianMatrix(object):
    def __init__(self, paradiag):
        r"""
        Python matrix for the Jacobian
        :param paradiag: The paradiag object
        """
        self.paradiag = paradiag
        self.u = fd.Function(paradiag.W_all)  # for the input function
        self.F = fd.Function(paradiag.W_all)  # for the output residual
        self.F_prev = fd.Function(paradiag.W_all)  # Where we compute the
        # part of the output residual from neighbouring contributions
        self.u0 = fd.Function(paradiag.W_all)  # Where we keep the state
        self.Fsingle = fd.Function(paradiag.W)
        self.urecv = fd.Function(paradiag.W)  # will contain the previous time value i.e. 3*r-1
        self.usend = fd.Function(paradiag.W)  # will contain the next time value i.e. 3*(r+1)
        self.ulist = self.u.split()
        self.r = paradiag.ensemble.ensemble_comm.rank
        self.n = paradiag.ensemble.ensemble_comm.size
        # Jform missing contributions from the previous step
        # Find u1 s.t. F[u1, u2, u3; v] = 0 for all v
        # definition:
        # dF_{u1}[u1, u2, u3; delta_u, v] =
        #  lim_{eps -> 0} (F[u1+eps*delta_u,u2,u3;v]
        #                  - F[u1,u2,u3;v])/eps
        # Newton, solves for delta_u such that
        # dF_{u1}[u1, u2, u3; delta_u, v] = -F[u1,u2,u3; v], for all v
        # then updates u1 += delta_u
        self.Jform = fd.derivative(paradiag.para_form, paradiag.w_all)
        # Jform contributions from the previous step
        self.Jform_prev = fd.derivative(paradiag.para_form,
                                        paradiag.w_recv)

    def mult(self, mat, X, Y):
        n = self.n

        # copy the local data from X into self.u
        with self.u.dat.vec_wo as v:
            v.array[:] = X.array_r

        mpi_requests = []
        # Communication stage

        # send
        r = self.paradiag.ensemble.ensemble_comm.rank  # the time rank
        self.paradiag.get_timestep(-1, index_range='slice', wout=self.usend, f_alls=self.ulist)

        request_send = self.paradiag.ensemble.isend(self.usend, dest=((r+1) % n), tag=r)
        mpi_requests.extend(request_send)

        # receive
        request_recv = self.paradiag.ensemble.irecv(self.urecv, source=((r-1) % n), tag=r-1)
        mpi_requests.extend(request_recv)

        # wait for the data [we should really do this after internal
        # assembly but have avoided that for now]
        MPI.Request.Waitall(mpi_requests)

        # Set the flag for the circulant option
        if self.paradiag.circ in ["quasi", "picard"]:
            self.paradiag.Circ.assign(1.0)
        else:
            self.paradiag.Circ.assign(0.0)

        # assembly stage
        fd.assemble(fd.action(self.Jform, self.u), tensor=self.F)
        fd.assemble(fd.action(self.Jform_prev, self.urecv),
                    tensor=self.F_prev)
        self.F += self.F_prev

        # unset flag if alpha-circulant approximation only in Jacobian
        if self.paradiag.circ not in ["picard"]:
            self.paradiag.Circ.assign(0.0)

        # Apply boundary conditions
        # assumes paradiag.w_all contains the current state we are
        # interested in
        # For Jacobian action we should just return the values in X
        # at boundary nodes
        for bc in self.paradiag.W_all_bcs:
            bc.homogenize()
            bc.apply(self.F, u=self.u)
            bc.restore()

        with self.F.dat.vec_ro as v:
            v.copy(Y)


class paradiag(object):
    def __init__(self, ensemble,
                 form_function, form_mass, W, w0, dt, theta,
                 alpha, M, bcs=[],
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
        :arg M: a list of integers, the number of timesteps
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

        self.aaos = AllAtOnceSystem(ensemble, M,
                                    dt, theta,
                                    form_mass, form_function,
                                    w0, bcs,
                                    circ, alpha)

        self.form_function = form_function
        self.ensemble = ensemble
        self.form_mass = form_mass
        self.W = self.aaos.function_space
        self.W_bcs = self.aaos.boundary_conditions
        self.ncpts = self.aaos.ncomponents
        self.w0 = self.aaos.initial_condition
        self.dt = self.aaos.dt
        self.M = self.aaos.slice_partition
        self.theta = self.aaos.theta
        self.alpha = self.aaos.alpha
        self.tol = tol
        self.maxits = maxits
        self.circ = self.aaos.circ
        self.ctx = ctx
        self.block_ctx = block_ctx
        self.rT = self.aaos.time_rank

        # A coefficient that switches the alpha-circulant term on
        self.Circ = self.aaos.Circ

        # function space for the component of them
        # all-at-once system assigned to this process
        # implemented as a massive mixed function space

        self.W_all = self.aaos.function_space_all

        # convert the W bcs into W_all bcs
        self.set_W_all_bcs()

        # function containing the part of the
        # all-at-once solution assigned to this rank
        self.w_all = self.aaos.w_all
        self.w_alls = self.aaos.w_alls

        # apply boundary conditions
        for bc in self.W_all_bcs:
            bc.apply(self.w_all)

        # function to assemble the nonlinear residual
        self.F_all = self.aaos.F_all

        # functions containing the last and next steps for parallel
        # communication timestep
        # from the previous iteration
        self.w_recv = self.aaos.w_recv
        self.w_send = self.aaos.w_send

        # set up the Vecs X (for coeffs and F for residuals)
        nlocal = M[self.rT]*W.node_set.size  # local times x local space
        nglobal = sum(M)*W.dim()  # global times x global space
        self.X = PETSc.Vec().create(comm=fd.COMM_WORLD)
        self.X.setSizes((nlocal, nglobal))
        self.X.setFromOptions()
        # copy initial data into the PETSc vec
        with self.w_all.dat.vec_ro as v:
            v.copy(self.X)
        self.F = self.X.copy()

        # construct the nonlinear form
        self._set_para_form()

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
        self.snes.setFunction(self._assemble_function, self.F)

        # set up the Jacobian
        mctx = JacobianMatrix(self)
        self.JacobianMatrix = mctx
        Jacmat = PETSc.Mat().create(comm=fd.COMM_WORLD)
        Jacmat.setType("python")
        Jacmat.setSizes(((nlocal, nglobal), (nlocal, nglobal)))
        Jacmat.setPythonContext(mctx)
        Jacmat.setUp()

        def form_jacobian(snes, X, J, P):
            # copy the snes state vector into self.X
            X.copy(self.X)
            self.update(X)
            J.assemble()
            P.assemble()

        self.snes.setJacobian(form_jacobian, J=Jacmat, P=Jacmat)

        # complete the snes setup
        self.opts.set_from_options(self.snes)

    def set_W_all_bcs(self):
        self.aaos.set_boundary_conditions()
        self.W_all_bcs = self.aaos.boundary_conditions_all

    def check_index(self, i, index_range='slice'):
        '''
        Check that timestep index is in range
        :arg i: timestep index to check
        :arg index_range: range that index is in. Either slice or window
        '''
        return self.aaos.check_index(i, index_range)

    def shift_index(self, i, from_range='slice', to_range='slice'):
        '''
        Shift timestep index from one range to another, and accounts for -ve indices
        :arg i: timestep index to shift
        :arg from_range: range of i. Either slice or window
        :arg to_range: range to shift i to. Either slice or window
        '''
        return self.aaos.shift_index(i, from_range, to_range)

    def set_timestep(self, step, wnew, index_range='slice', f_alls=None):
        '''
        Set solution at a timestep to new value

        :arg step: index of timestep to set.
        :arg wnew: new solution for timestep
        :arg index_range: is index in window or slice?
        :arg f_alls: an all-at-once function to set timestep in. If None, self.w_alls is used
        '''
        return self.aaos.set_timestep(step, wnew, index_range, f_alls)

    def get_timestep(self, step, index_range='slice', wout=None, name=None, f_alls=None):
        '''
        Get solution at a timestep to new value

        :arg step: index of timestep to set.
        :arg index_range: is index in window or slice?
        :arg wout: function to set to timestep (timestep returned if None)
        :arg name: name of returned function
        :arg f_alls: an all-at-once function to get timestep from. If None, self.w_alls is used
        '''
        return self.aaos.get_timestep(step, index_range, wout, name, f_alls)

    def for_each_timestep(self, callback):
        '''
        call callback for each timestep in each slice in the current window
        callback arguments are: timestep index in window, timestep index in slice, Function at timestep

        :arg callback: the function to call for each timestep
        '''
        self.aaos.for_each_timestep(callback)

    def update(self, X):
        '''
        Update self.w_alls and self.w_recv
        from X.
        The local parts of X are copied into self.w_alls
        and the last step from the previous slice (periodic)
        is copied into self.u_prev
        '''
        self.aaos.update(X)

    def next_window(self, w1=None):
        """
        Reset paradiag ready for next time-window

        :arg w1: initial solution for next time-window.If None,
                 will use the final timestep from previous window
        """
        self.aaos.next_window(w1)

    def _assemble_function(self, snes, X, Fvec):
        r"""
        This is the function we pass to the snes to assemble
        the nonlinear residual.
        """
        self.aaos._assemble_function(snes, X, Fvec)

    def _set_para_form(self):
        """
        Constructs the bilinear form for the all at once system.
        Specific to the theta-centred Crank-Nicholson method
        """
        self.aaos._set_para_form()
        self.para_form = self.aaos.para_form

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
            self.update(self.X)

            postproc(self, wndw)

            if not (1 < self.snes.getConvergedReason() < 5):
                PETSc.Sys.Print(f'SNES diverged with error code {self.snes.getConvergedReason()}. Cancelling paradiag time integration.')
                return

            if wndw != nwindows-1:
                self.next_window()
