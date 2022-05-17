import numpy as np
import firedrake as fd
from firedrake.petsc import flatten_parameters
from firedrake.petsc import PETSc, OptionsManager
from pyop2.mpi import MPI
from functools import reduce, partial
from operator import mul

appctx = {}


def context_callback(pc, context):
    return context


get_context = partial(context_callback, context=appctx)


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
        usends = self.usend.split()
        r = self.paradiag.ensemble.ensemble_comm.rank  # the time rank
        # r = ensemble.ensemble_comm.rank # the time rank
        for k in range(self.paradiag.ncpts):
            usends[k].assign(self.ulist[self.paradiag.ncpts*(self.paradiag.M[r]-1)+k])
        if r < n-1:
            request_send = self.paradiag.ensemble.isend(self.usend, dest=r+1, tag=r)
        else:
            request_send = self.paradiag.ensemble.isend(self.usend, dest=0, tag=r)
        mpi_requests.extend(request_send)
        # receive
        if r > 0:
            request_recv = self.paradiag.ensemble.irecv(self.urecv, source=r-1, tag=r-1)
        else:
            request_recv = self.paradiag.ensemble.irecv(self.urecv, source=n-1, tag=n-1)
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
                 ctx={}, block_mat_type="aij"):
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
        :arg block_mat_type: set the type of the diagonal block systems.
        Default is aij.
        """

        self.form_function = form_function
        self.ensemble = ensemble
        self.form_mass = form_mass
        self.W = W
        self.W_bcs = bcs
        self.ncpts = len(W.split())
        self.w0 = w0
        self.dt = dt
        if isinstance(M, int):
            M = [M]
        self.M = M
        self.theta = theta
        self.alpha = alpha
        self.tol = tol
        self.maxits = maxits
        self.circ = circ

        # A coefficient that switches the alpha-circulant term on
        self.Circ = fd.Constant(1.0)

        # checks that the ensemble communicator is set up correctly
        nM = len(M)  # the expected number of time ranks
        assert ensemble.ensemble_comm.size == nM
        rT = ensemble.ensemble_comm.rank  # the time rank
        self.rT = rT
        # function space for the component of them
        # all-at-once system assigned to this process
        # implemented as a massive mixed function space

        self.W_all = reduce(mul, (self.W for _ in range(M[rT])))

        # convert the W bcs into W_all bcs
        self.set_W_all_bcs()

        # function containing the part of the
        # all-at-once solution assigned to this rank
        self.w_all = fd.Function(self.W_all)
        w_alls = self.w_all.split()
        # initialise it from the initial condition
        for i in range(M[rT]):
            for k in range(self.ncpts):
                w_alls[self.ncpts*i+k].assign(self.w0.sub(k))
        self.w_alls = w_alls

        # function to assemble the nonlinear residual
        self.F_all = fd.Function(self.W_all)

        # functions containing the last and next steps for parallel
        # communication timestep
        # from the previous iteration
        self.w_recv = fd.Function(self.W)
        self.w_send = fd.Function(self.W)

        # set up the Vecs X (for coeffs and F for residuals)
        nlocal = M[rT]*W.node_set.size  # local times x local space
        nglobal = np.prod(M)*W.dim()  # global times x global space
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
        self.W_all_bcs = []
        for bc in self.W_bcs:
            if isinstance(self.W.ufl_element(), fd.MixedElement):
                i = bc.function_space().index
                ncpts = self.W.ufl_element().num_sub_elements()
                for r in range(self.M[self.rT]):
                    all_bc = fd.DirichletBC(self.W_all.sub(r*ncpts+i),
                                            bc.function_arg,
                                            bc.sub_domain)
                    self.W_all_bcs.append(all_bc)
            else:
                for r in range(self.M[self.rT]):
                    all_bc = fd.DirichletBC(self.W_all.sub(r),
                                            bc.function_arg,
                                            bc.sub_domain)
                    self.W_all_bcs.append(all_bc)

    def update(self, X):
        # Update self.w_alls and self.w_recv
        # from X.
        # The local parts of X are copied into self.w_alls
        # and the last step from the previous slice (periodic)
        # is copied into self.u_prev

        n = self.ensemble.ensemble_comm.size

        with self.w_all.dat.vec_wo as v:
            v.array[:] = X.array_r

        # apply the boundary conditions
        for bc in self.W_all_bcs:
            bc.apply(self.w_all)

        mpi_requests = []
        # Communication stage
        # send
        usends = self.w_send.split()
        r = self.ensemble.ensemble_comm.rank  # the time rank
        for k in range(self.ncpts):
            usends[k].assign(self.w_alls[self.ncpts*(self.M[r]-1)+k])
        if r < n-1:
            request_send = self.ensemble.isend(self.w_send, dest=r+1, tag=r)
        else:
            request_send = self.ensemble.isend(self.w_send, dest=0, tag=r)
        mpi_requests.extend(request_send)
        # receive
        if r > 0:
            request_recv = self.ensemble.irecv(self.w_recv, source=r-1, tag=r-1)
        else:
            request_recv = self.ensemble.irecv(self.w_recv, source=n-1, tag=n-1)
        mpi_requests.extend(request_recv)

        # wait for the data [we should really do this after internal
        # assembly but have avoided that for now]
        MPI.Request.Waitall(mpi_requests)

    def next_window(self, w1=None):
        """
        Reset paradiag ready for next time-window
        :arg w1: initial solution for next time-window.If None,
                 will use the final timestep from previous window
        """
        rank = self.rT
        ncomm = self.ensemble.ensemble_comm.size
        ncpts = self.ncpts

        if w1 is not None:  # use given function
            self.w0.assign(w1)
        else:  # last rank broadcasts final timestep
            if rank == ncomm-1:
                # index of start of final timestep
                i0 = ncpts*(self.M[-1]-1)
                for k in range(ncpts):
                    wend = self.w_alls[i0+k]
                    self.w0.sub(k).assign(wend)

            # there should really be a bcast method on the ensemble
            # if this gets added in firedrake then this will need updating
            for dat in self.w0.dat:
                self.ensemble.ensemble_comm.Bcast(dat.data, root=ncomm-1)

        # persistence forecast
        for i in range(self.M[rank]):
            for k in range(ncpts):
                self.w_alls[ncpts*i+k].assign(self.w0.sub(k))

        return

    def _assemble_function(self, snes, X, Fvec):
        r"""
        This is the function we pass to the snes to assemble
        the nonlinear residual.
        """
        self.update(X)

        # Set the flag for the circulant option
        if self.circ == "picard":
            self.Circ.assign(1.0)
        else:
            self.Circ.assign(0.0)
        # assembly stage
        fd.assemble(self.para_form, tensor=self.F_all)

        # apply boundary conditions
        for bc in self.W_all_bcs:
            bc.homogenize()
            bc.apply(self.F_all)
            bc.restore()

        with self.F_all.dat.vec_ro as v:
            v.copy(Fvec)

    def _set_para_form(self):
        """
        Constructs the bilinear form for the all at once system.
        Specific to the theta-centred Crank-Nicholson method
        """

        M = self.M[self.rT]
        w_all_cpts = fd.split(self.w_all)

        test_fns = fd.TestFunctions(self.W_all)

        dt = fd.Constant(self.dt)
        theta = fd.Constant(self.theta)
        alpha = fd.Constant(self.alpha)

        for n in range(M):
            # previous time level
            if n == 0:
                # self.w_recv will contain the adjacent data
                if self.rT == 0:
                    # need the initial data
                    w0list = fd.split(self.w0)
                    wrecvlist = fd.split(self.w_recv)
                    w0s = [w0list[i] + self.Circ*alpha*wrecvlist[i]
                           for i in range(self.ncpts)]
                else:
                    w0s = fd.split(self.w_recv)
            else:
                w0s = w_all_cpts[self.ncpts*(n-1):self.ncpts*n]
            # current time level
            w1s = w_all_cpts[self.ncpts*n:self.ncpts*(n+1)]
            dws = test_fns[self.ncpts*n:self.ncpts*(n+1)]
            # time derivative
            if n == 0:
                p_form = (1.0/dt)*self.form_mass(*w1s, *dws)
            else:
                p_form += (1.0/dt)*self.form_mass(*w1s, *dws)
            p_form -= (1.0/dt)*self.form_mass(*w0s, *dws)
            # vector field
            p_form += theta*self.form_function(*w1s, *dws)
            p_form += (1-theta)*self.form_function(*w0s, *dws)
        self.para_form = p_form

    def solve(self, verbose=False):
        """
        Solve the system (either in one shot or as a relaxation method).
        """

        with self.opts.inserted_options():
            self.snes.solve(None, self.X)
        self.update(self.X)
