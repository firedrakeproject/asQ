import numpy as np
import firedrake as fd
from scipy.fft import fft, ifft
from firedrake.petsc import PETSc, OptionsManager
from pyop2.mpi import MPI

class DiagFFTPC(fd.PCBase):

    r"""A preconditioner for all-at-once systems with alpha-circulant
    block diagonal structure, using FFT.
    """

    def initialize(self, pc):
        if pc.getType() != "python":
            raise ValueError("Expecting PC type python")
        prefix = pc.getOptionsPrefix() + "diagfft_"

        # we assume P has things stuffed inside of it
        _, P = pc.getOperators()
        context = P.getPythonContext()
        appctx = context.appctx
        self.appctx = appctx

        # all at once solution passed through the appctx
        self.w_all = appctx.get("w_all", None)

        # FunctionSpace checks
        test, trial = context.a.arguments()
        if test.function_space() != trial.function_space():
            raise ValueError("Pressure space test and trial space differ")
        W = test.function_space()

        # basic model function space
        get_blockV = appctx.get("get_blockV", None)
        self.blockV = get_blockV()
        M = int(W.dim()/self.blockV.dim())
        assert(self.blockV.dim()*M == W.dim())
        self.M = M
        self.NM = W.dim()

        # Input/Output wrapper Functions
        self.xf = fd.Function(W)  # input
        self.yf = fd.Function(W)  # output

        # Gamma coefficients
        Nt = M
        exponents = np.arange(Nt)/Nt
        alphav = appctx.get("alpha", None)
        self.Gam = alphav**exponents

        # Di coefficients
        thetav = appctx.get("theta", None)
        Dt = appctx.get("dt", None)
        C1col = np.zeros(Nt)
        C2col = np.zeros(Nt)
        C1col[:2] = np.array([1, -1])/Dt
        C2col[:2] = np.array([thetav, 1-thetav])
        self.D1 = np.sqrt(Nt)*fft(self.Gam*C1col)
        self.D2 = np.sqrt(Nt)*fft(self.Gam*C2col)

        # Block system setup
        # First need to build the vector function space version of
        # blockV
        mesh = self.blockV.mesh()
        Ve = self.blockV.ufl_element()
        if isinstance(Ve, fd.MixedElement):
            MixedCpts = []
            self.ncpts = Ve.num_sub_elements()
            for cpt in range(Ve.num_sub_elements()):
                SubV = Ve.sub_elements()[cpt]
                if isinstance(SubV, fd.FiniteElement):
                    MixedCpts.append(fd.VectorElement(SubV, dim=2))
                elif isinstance(SubV, fd.VectorElement):
                    shape = (2, SubV.num_sub_elements())
                    MixedCpts.append(fd.TensorElement(SubV, shape))
                elif isinstance(SubV, fd.TensorElement):
                    shape = (2,) + SubV._shape
                    MixedCpts.append(fd.TensorElement(SubV, shape))
                else:
                    raise NotImplementedError

            dim = len(MixedCpts)
            self.CblockV = np.prod([fd.FunctionSpace(mesh,
                                    MixedCpts[i]) for i in range(dim)])
        else:
            self.ncpts = 1
            if isinstance(Ve, fd.FiniteElement):
                self.CblockV = fd.FunctionSpace(mesh,
                                                fd.VectorElement(Ve, dim=2))
            elif isinstance(Ve, fd.VectorElement):
                shape = (2, Ve.num_sub_elements())
                self.CblockV = fd.FunctionSpace(mesh,
                                                fd.TensorElement(Ve, shape))
            elif isinstance(Ve, fd.TensorElement):
                shape = (2,) + Ve._shape
                self.CblockV = fd.FunctionSpace(mesh,
                                                fd.TensorElement(Ve, shape))
            else:
                raise NotImplementedError

        # Now need to build the block solver
        vs = fd.TestFunctions(self.CblockV)
        self.u0 = fd.Function(self.CblockV)  # we will create a linearisation
        us = fd.split(self.u0)

        # extract the real and imaginary parts
        vsr = []
        vsi = []
        usr = []
        usi = []

        if isinstance(Ve, fd.MixedElement):
            N = Ve.num_sub_elements()
            for i in range(N):
                SubV = Ve.sub_elements()[i]
                if len(SubV.value_shape()) == 0:
                    vsr.append(vs[i][0])
                    vsi.append(vs[i][1])
                    usr.append(us[i][0])
                    usi.append(us[i][1])
                elif len(SubV.value_shape()) == 1:
                    vsr.append(vs[i][0, :])
                    vsi.append(vs[i][1, :])
                    usr.append(us[i][0, :])
                    usi.append(us[i][1, :])
                elif len(SubV.value_shape()) == 2:
                    vsr.append(vs[i][0, :, :])
                    vsi.append(vs[i][1, :, :])
                    usr.append(us[i][0, :, :])
                    usi.append(us[i][1, :, :])
                else:
                    raise NotImplementedError
        else:
            if isinstance(Ve, fd.FiniteElement):
                vsr.append(vs[0])
                vsi.append(vs[1])
                usr.append(us[0])
                usi.append(us[1])
            elif isinstance(Ve, fd.VectorElement):
                vsr.append(vs[0, :])
                vsi.append(vs[1, :])
                usr.append(self.u0[0, :])
                usi.append(self.u0[1, :])
            elif isinstance(Ve, fd.TensorElement):
                vsr.append(vs[0, :])
                vsi.append(vs[1, :])
                usr.append(self.u0[0, :])
                usi.append(self.u0[1, :])
            else:
                raise NotImplementedError

        # input and output functions
        self.Jprob_in = fd.Function(self.CblockV)
        self.Jprob_out = fd.Function(self.CblockV)

        # A place to store all the inputs to the block problems
        self.xfi = fd.Function(W)
        self.xfr = fd.Function(W)

        #  Building the nonlinear operator
        self.Jsolvers = []
        self.Js = []
        form_mass = appctx.get("form_mass", None)
        form_function = appctx.get("form_function", None)

        # setting up the Riesz map
        # input for the Riesz map
        self.xtemp = fd.Function(self.CblockV)
        v = fd.TestFunction(self.CblockV)
        u = fd.TrialFunction(self.CblockV)
        a = fd.assemble(fd.inner(u, v)*fd.dx)
        self.Proj = fd.LinearSolver(a, options_prefix=prefix+"mass_")
        # building the block problem solvers
        for i in range(M):
            D1i = fd.Constant(np.imag(self.D1[i]))
            D1r = fd.Constant(np.real(self.D1[i]))
            D2i = fd.Constant(np.imag(self.D2[i]))
            D2r = fd.Constant(np.real(self.D2[i]))

            # pass sigma into PC:
            sigma = self.D1[i]**2/self.D2[i]
            sigma_inv = self.D2[i]**2/self.D1[i]
            appctx_h = appctx.copy()
            appctx_h["sr"] = fd.Constant(np.real(sigma))
            appctx_h["si"] = fd.Constant(np.imag(sigma))
            appctx_h["sinvr"] = fd.Constant(np.real(sigma_inv))
            appctx_h["sinvi"] = fd.Constant(np.imag(sigma_inv))
            appctx_h["D2r"] = D2r
            appctx_h["D2i"] = D2i
            appctx_h["D1r"] = D1r
            appctx_h["D1i"] = D1i

            A = (
                D1r*form_mass(*usr, *vsr)
                - D1i*form_mass(*usi, *vsr)
                + D2r*form_function(*usr, *vsr)
                - D2i*form_function(*usi, *vsr)
                + D1r*form_mass(*usi, *vsi)
                + D1i*form_mass(*usr, *vsi)
                + D2r*form_function(*usi, *vsi)
                + D2i*form_function(*usr, *vsi)
            )

            # The linear operator
            J = fd.derivative(A, self.u0)

            # The rhs
            v = fd.TestFunction(self.CblockV)
            L = fd.inner(v, self.Jprob_in)*fd.dx

            block_prefix = prefix+str(i)+'_'
            jprob = fd.LinearVariationalProblem(J, L, self.Jprob_out)
            Jsolver = fd.LinearVariationalSolver(jprob,
                                                 appctx=appctx_h,
                                                 options_prefix=block_prefix)
            self.Jsolvers.append(Jsolver)

    def update(self, pc):
        self.u0.assign(0)
        for i in range(self.M):
            # copy the data into solver input
            if self.ncpts > 1:
                u0s = self.u0.split()
                for cpt in range(self.ncpts):
                    u0s[cpt].sub(0).assign(u0s[cpt].sub(0)
                                           + self.w_all.split()[self.ncpts*i+cpt])
            else:
                self.u0.sub(0).assign(self.u0.sub(0)
                                      + self.w_all.split()[i])
        self.u0 /= self.M

    def apply(self, pc, x, y):

        # copy petsc vec into Function
        with self.xf.dat.vec_wo as v:
            x.copy(v)

        # get array of basis coefficients
        with self.xf.dat.vec_ro as v:
            parray = v.array.reshape((self.M, self.blockV.dim()))
        # This produces an array whose rows are time slices
        # and columns are finite element basis coefficients

        # Diagonalise
        Nt = self.M
        parray = fft((self.Gam*parray.T).T, axis=0)*np.sqrt(Nt)

        # Copy into xfi, xfr
        with self.xfr.dat.vec_wo as v:
            v.array[:] = parray.real.reshape((self.NM,))

        with self.xfi.dat.vec_wo as v:
            v.array[:] = parray.imag.reshape((self.NM,))

        # Do the block solves

        for i in range(self.M):
            # copy the data into solver input
            self.xtemp.assign(0.)
            if self.ncpts > 1:
                Jins = self.xtemp.split()
                for cpt in range(self.ncpts):
                    Jins[cpt].sub(0).assign(
                        self.xfr.split()[self.ncpts*i+cpt])
                    Jins[cpt].sub(1).assign(
                        self.xfi.split()[self.ncpts*i+cpt])
            else:
                self.xtemp.sub(0).assign(self.xfr.split()[i])
                self.xtemp.sub(1).assign(self.xfi.split()[i])
            # Do a project for Riesz map, to be superceded
            # when we get Cofunction
            self.Proj.solve(self.Jprob_in, self.xtemp)

            # solve the block system
            self.Jprob_out.assign(0.)
            self.Jsolvers[i].solve()

            # copy the data from solver output
            if self.ncpts > 1:
                Jpouts = self.Jprob_out.split()
                for cpt in range(self.ncpts):
                    self.xfr.split()[self.ncpts*i+cpt].assign(
                        Jpouts[cpt].sub(0))
                    self.xfi.split()[self.ncpts*i+cpt].assign(
                        Jpouts[cpt].sub(1))
            else:
                Jpouts = self.Jprob_out
                self.xfr.split()[i].assign(Jpouts.sub(0))
                self.xfi.split()[i].assign(Jpouts.sub(1))

        # Undiagonalise
        # get array of basis coefficients
        with self.xfi.dat.vec_ro as v:
            parray = 1j*v.array.reshape((self.M, self.blockV.dim()))
        with self.xfr.dat.vec_ro as v:
            parray += v.array.reshape((self.M, self.blockV.dim()))
        parray = ((1.0/self.Gam)*ifft(parray, axis=0).T).T
        # get array of basis coefficients
        with self.yf.dat.vec_wo as v:
            v.array[:] = parray.reshape((self.M*self.blockV.dim(),)).real

        with self.yf.dat.vec_ro as v:
            v.copy(y)

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError


#python matrix for the Jacobian
class JacobianMatrix(object):
    def __init__(self, paradiag):
        r"""
        :param paradiag: The paradiag object
        """
        self.u = Function(paradiag.W_all) #Where we copy the input function
        self.F = Function(paradiag.W_all) #Where we copy the output residual
        self.F_prev = Function(paradiag.W_all) #Where we compute the
                                               #part of the output
                                               #residual from neighbouring
                                               #contributions
        self.u0 = Function(paradiag.W_all) #Where we keep the state
        self.Fsingle = Function(paradiag.W)
        self.urecv = Function(paradiag.W) #will contain the previous time value i.e. 3*r-1
        self.usend = Function(paradiag.W) #will contain the next time value i.e. 3*(r+1)
        self.ulist = self.u.split()
        self.r = paradiag.ensemble.ensemble_comm.rank
        self.n = paradiag.ensemble.ensemble_comm.size
        # Jform missing contributions from the previous step
        self.Jform = fd.derivative(paradiag.para_form, paradiag.w_all)
        # Jform contributions from the previous step
        self.Jform_prev = fd.derivative(paradiag.para_form, 
                                        paradiag.w_recv)

    def mult(self, mat, X, Y):
        #update the contents of paradiag.u from paradiag.X
        self.paradiag.update(self.paradiag.Z)

        n = self.n

        with self.u.dat.vec_wo as v:
            v.array[:] = X.array_r

        mpi_requests = []
        #Communication stage                
        #send
        usends = self.usend.split()
        r = ensemble.ensemble_comm.rank # the time rank
        for k in range(self.ncpts):
            usends[k].assign(u[self.ncpts*(self.M[r]-1)+k])
        if r < n-1:
            request_send = ens_comm.isend(self.usend, dest=r+1, tag=r)
        else:
            request_send = ens_comm.isend(self.usend, dest=0, tag=r)
        mpi_requests.extend(request_send)
        #receive
        if r > 0:
            request_recv = ens_comm.irecv(self.urecv, source=r-1, tag=r-1)
        else:
            request_recv = ens_comm.irecv(self.urecv, source=n-1, tag=n-1)
        mpi_requests.extend(request_recv)

        #wait for the data [we should really do this after internal
        #assembly but have avoided that for now]
        MPI.Request.Waitall(mpi_requests)

        #Set the flag for the circulant option
        if self.circ == "quasi":
            self.Circ.assign(1.0)
        else:
            self.Circ.assign(0.0)

        #assembly stage
        fd.assemble(fd.action(self.Jform, self.u), tensor=self.F)
        fd.assemble(fd.action(self.Jform_prev, self.usend),
                    tensor=self.F_prev)
        self.F += self.F_prev

        with self.F.dat.vec_ro as v:
            v.copy(Y)


class paradiag(object):
    def __init__(self, ensemble,
                 form_function, form_mass, W, w0, dt, theta,
                 alpha, M, solver_parameters=None,
                 circ="picard",
                 jac_average="newton", tol=1.0e-6, maxits=10,
                 ctx={}, block_mat_type="aij"):
        """A class to implement paradiag timestepping.

        : arg ensemble: the ensemble communicator
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
        :arg solver_parameters: options dictionary for nonlinear solver
        :arg circ: a string describing the option on where to use the
        alpha-circulant modification. "picard" - do a nonlinear wave
        form relaxation method. "quasi" - do a modified Newton
        method with alpha-circulant modification added to the
        Jacobian. To make the alpha circulant modification only in the
       preconditioner, simply set ksp_type:preonly in the solve options.
        :arg jac_average: a string describing the option for when to
        average the jacobian. "newton" - make a quasi-Newton method by
        time averaging the Jacobian. "preconditioner" - only do the
        average in the preconditioner.
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
        self.jac_average = jac_average

        # A coefficient that switches the alpha-circulant term on
        self.Circ = fd.Constant(1.0)

        # checks that the ensemble communicator is set up correctly
        nM = len(M) # the expected number of time ranks
        assert(ensemble.ensemble_comm.size == nM,
               ensemble.ensemble_comm.size, nM)
        rT = ensemble.ensemble_comm.rank # the time rank
        self.rT = rT
        # function space for the component of the
        # all-at-once system assigned to this process
        # implemented as a massive mixed function space
        self.W_all = np.prod([self.W for i in range(M[rT])])

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
        nlocal = W.node_set.size
        nglobal = W.dim()
        self.X = PETSc.Vec().create(comm=fd.COMM_WORLD)
        self.X.setSizes((nlocal, nglobal))
        self.X.setFromOptions()
        #copy initial data into the PETSc vec
        with self.w_all.dat.vec_ro as v:
            v.copy(self.X)
        self.F = X.copy()

        # construct the nonlinear form
        self._set_para_form()

        # set up the snes
        self.snes = PETSc.SNES().create(comm=fd.COMM_WORLD)
        opts = OptionsManager(solver_parameters, 'paradiag')
        self.snes.setOptionsPrefix('paradiag')
        self.snes.setFunction(self._assemble_function, self.F)

        # set up the Jacobian
        mctx = JacobianMatrix(self)
        Jacmat = PETSc.Mat().create(comm=fd.COMM_WORLD)
        Jacmat.setType("python")
        Jacmat.setSizes(((nlocal, nglobal), (nlocal, nglobal)))
        Jacmat.setPythonContext(mctx)
        Jacmat.setUp()
        def form_jacobian(snes, X, J, P):
            J.assemble()
            P.assemble()

        snes.setJacobian(form_jacobian, J=Jacmat, P=Jacmat)

        # complete the snes setup
        opts.set_from_options(self.snes)

        # passing stuff to the diag preconditioner using the appctx
        # This stuff all needs moving to the preconditioner
        def getW():
            return W

        ctx["get_blockV"] = getW
        ctx["alpha"] = self.alpha
        ctx["theta"] = self.theta
        ctx["dt"] = self.dt
        ctx["form_mass"] = self.form_mass
        ctx["form_function"] = self.form_function
        ctx["w_all"] = self.w_all
        ctx["block_mat_type"] = block_mat_type

    def update(self, X):
        #Update self.u and self.u_recv
        #from X.
        #The local parts of X are copied into self.u
        #and the last step from the previous slice (periodic)
        #is copied into self.u_prev

        n = self.ensemble.ensemble_comm.size

        with self.w_all.dat.vec_wo as v:
            v.array[:] = X.array_r
        
        mpi_requests = []
        #Communication stage                
        #send
        usends = self.usend.split()
        r = ensemble.ensemble_comm.rank # the time rank
        for k in range(self.ncpts):
            usends[k].assign(w_alls[self.ncpts*(self.M[r]-1)+k])
        if r < n-1:
            request_send = ens_comm.isend(self.usend, dest=r+1, tag=r)
        else:
            request_send = ens_comm.isend(self.usend, dest=0, tag=r)
        mpi_requests.extend(request_send)
        #receive
        if r > 0:
            request_recv = ens_comm.irecv(self.urecv, source=r-1, tag=r-1)
        else:
            request_recv = ens_comm.irecv(self.urecv, source=n-1, tag=n-1)
        mpi_requests.extend(request_recv)

        #wait for the data [we should really do this after internal
        #assembly but have avoided that for now]
        MPI.Request.Waitall(mpi_requests)        
        
    def _assemble_function(self, snes, X, Fvec):
        r"""
        This is the function we pass to the snes to assemble
        the nonlinear residual.
        """
        self.update(X)

        #Set the flag for the circulant option
        if self.circ == "picard":
            self.Circ.assign(1.0)
        else:
            self.Circ.assign(0.0)
        #assembly stage
        fd.assemble(self.para_form, tensor=self.F_all)

        with self.F.dat.vec_ro as v:
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
        wMs = w_all_cpts[self.ncpts*(M-1):]

        for n in range(M):
            # previous time level
            if n == 0:
                #self.w_recv will contain the adjacent data
                if self.rT == 0:
                    #need the initial data
                    w0s = fd.split(self.w0 + self.Circ*alpha*self.w_recv)
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

        M = self.M
        if self.circ == "picard":
            raise NotImplementedError
        else:
            # One shot
            snes.solve(self.X)
            self.update(self.X)
