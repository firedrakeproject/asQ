import numpy as np
import firedrake as fd
from scipy.fft import fft, ifft
from firedrake.petsc import PETSc, OptionsManager
from pyop2.mpi import MPI

from functools import reduce
from operator import mul

from mpi4py_fft.pencil import Pencil, Subcomm
from mpi4py_fft import fftw

class DiagFFTPC(object):
    def __init__(self):
        r"""A preconditioner for all-at-once systems with alpha-circulant
        block diagonal structure, using FFT.
        """
        self.initialized = False

    def setUp(self, pc):
        """Setup method called by PETSc."""

        if self.initialized:
            self.update(pc)
        else:
            self.initialize(pc)
            self.initialized = True
        
    def initialize(self, pc):
        if pc.getType() != "python":
            raise ValueError("Expecting PC type python")
        prefix = pc.getOptionsPrefix() + "diagfft_"

        # we assume P has things stuffed inside of it
        #_, P = pc.getOperators()
        #context = P.getPythonContext()
        #appctx = context.appctx
        #self.appctx = appctx

        paradiag = pc.getAttr("paradiag")

        # this time slice part of the all at once solution
        self.w_all = paradiag.w_all
        # this is bad naming
        W = paradiag.W_all

        # basic model function space
        self.blockV = paradiag.W
        M = paradiag.M
        ensemble = paradiag.ensemble
        rT = ensemble.ensemble_comm.rank  # the time rank
        assert(self.blockV.dim()*M[rT] == W.dim())
        self.M = M
        self.rT = rT
        self.NM = W.dim()

        # Input/Output wrapper Functions
        self.xf = fd.Function(W)  # input
        self.yf = fd.Function(W)  # output

        # Gamma coefficients
        self.Nt = np.sum(M)
        exponents = np.arange(Nt)/Nt
        alphav = paradiag.alpha
        self.Gam = alphav**exponents
        self.Gam_slice = self.Gam[np.sum(M[:rT]):np.sum(M[:rT+1])]

        # Di coefficients
        thetav = paradiag.theta
        Dt = paradiag.dt
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
        form_mass = paradiag.form_mass
        form_function = paradiag.form_function

        # setting up the FFT stuff
        # construct simply dist array and 1d fftn:
        subcomm = Subcomm(ens_comm.ensemble_comm, [0, 1])
        # get some dimensions
        nlocal = self.blockV.node_set.size
        NN = np.array([np.sum(M), nlocal], dtype=int)
        # transfer pencil is aligned along axis 1
        self.p0 = Pencil(subcomm, NN, axis=1)
        # a0 is the local part of our fft working array
        # has shape of (M/P, nlocal)
        self.a0 = np.zeros(p0.subshape, complex)
        self.p1 = self.p0.pencil(0)
        # a0 is the local part of our other fft working array
        self.a1 = np.zeros(p1.subshape, complex)
        self.transfer = p0.transfer(p1, complex)
        # the FFTW working arrays
        self.b_fftin = fftw.aligned(a1.shape, dtype=complex)
        self.b_fftout = fftw.aligned_like(self.b_fft)
        # FFTW plans
        self.fft = fftw.fftn(self.b_fftin,
                             flags=(fftw.FFTW_MEASURE,),
                             axes=(0,), output_array=self.b_fftout)
        self.fft = fftw.ifftn(self.b_fftin,
                              flags=(fftw.FFTW_MEASURE,),
                              axes=(0,), output_array=self.b_fftout)

        # setting up the Riesz map
        # input for the Riesz map
        self.xtemp = fd.Function(self.CblockV)
        v = fd.TestFunction(self.CblockV)
        u = fd.TrialFunction(self.CblockV)
        a = fd.assemble(fd.inner(u, v)*fd.dx)
        self.Proj = fd.LinearSolver(a, options_prefix=prefix+"mass_")
        # building the block problem solvers
        for i in range(M[rT]):
            ii = np.sum(M[:rT])+i # global time time index
            D1i = fd.Constant(np.imag(self.D1[ii]))
            D1r = fd.Constant(np.real(self.D1[ii]))
            D2i = fd.Constant(np.imag(self.D2[ii]))
            D2r = fd.Constant(np.real(self.D2[ii]))

            # pass sigma into PC:
            sigma = self.D1[ii]**2/self.D2[ii]
            sigma_inv = self.D2[ii]**2/self.D1[ii]
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

            block_prefix = prefix+str(ii)+'_'
            jprob = fd.LinearVariationalProblem(J, L, self.Jprob_out)
            Jsolver = fd.LinearVariationalSolver(jprob,
                                                 appctx=appctx_h,
                                                 options_prefix=block_prefix)
            self.Jsolvers.append(Jsolver)

    def update(self, pc):
        self.u0.assign(0)
        print("Need to check that contents from X get into w_all")
        for i in range(self.M[self.rT]):
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
        # hopefully this works
        with self.xf.dat.vec_wo as v:
            x.copy(v)

        # get array of basis coefficients
        with self.xf.dat.vec_ro as v:
            parray = v.array_r.reshape((self.M[rT],
                                        self.blockV.node_set.size))
        # This produces an array whose rows are time slices
        # and columns are finite element basis coefficients

        ######################
        # Diagonalise - scale, transfer, FFT, transfer, Copy
        # Scale
        # is there a better way to do this with broadcasting?
        parray = (self.Gam_slice*parray.T).T*np.sqrt(self.Nt)
        # transfer forward
        self.a0[:] = parray[:]
        self.transfer.forward(self.a0, self.a1)
        # FFT
        self.b_fftin[:] = self.a1[:]
        self.fft()
        a1[:] = self.b_fftout[:]
        # transfer backward
        self.transfer.backward(self.a1, self.a0)
        # Copy into xfi, xfr
        parray[:] = self.a0[:]
        with self.xfr.dat.vec_wo as v:
            v.array[:] = parray.real.reshape(-1)
        with self.xfi.dat.vec_wo as v:
            v.array[:] = parray.imag.reshape(-1)
        #####################
            
        # Do the block solves

        for i in range(self.M[rT]):
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

        ######################
        # Undiagonalise - Copy, transfer, IFFT, transfer, scale, copy
        # get array of basis coefficients
        with self.xfi.dat.vec_ro as v:
            parray = 1j*v.array_r.reshape((self.M,
                                           self.blockV.node_set.size))
        with self.xfr.dat.vec_ro as v:
            parray += v.array_r.reshape((self.M,
                                         self.blockV.node_set.size))
        # transfer forward
        self.a0[:] = parray[:]
        self.transfer.forward(self.a0, self.a1)
        # IFFT
        self.b_fftin[:] = self.a1[:]
        self.ifft()
        a1[:] = self.b_fftout[:]
        # transfer backward
        self.transfer.backward(self.a1, self.a0)
        parray[:] = self.a0[:]
        ifft(parray, axis=0)
        #scale
        parray = ((1.0/self.Gam)*parray.T).T
        # Copy into xfi, xfr
        with self.yf.dat.vec_wo as v:
            v.array[:] = parray.reshape(-1).real
        with self.yf.dat.vec_ro as v:
            v.copy(y)
        ################

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError


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
        # update the contents of paradiag.u from paradiag.X
        self.paradiag.update(self.paradiag.X)

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
        if self.paradiag.circ == "quasi":
            self.paradiag.Circ.assign(1.0)
        else:
            self.paradiag.Circ.assign(0.0)

        # assembly stage
        fd.assemble(fd.action(self.Jform, self.u), tensor=self.F)
        fd.assemble(fd.action(self.Jform_prev, self.urecv),
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
        nM = len(M)  # the expected number of time ranks
        assert ensemble.ensemble_comm.size == nM
        rT = ensemble.ensemble_comm.rank  # the time rank
        self.rT = rT
        # function space for the component of them
        # all-at-once system assigned to this process
        # implemented as a massive mixed function space

        self.W_all = reduce(mul, (self.W for _ in range(M[rT])))
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

        # set up the snes
        self.snes = PETSc.SNES().create(comm=fd.COMM_WORLD)
        self.opts = OptionsManager(solver_parameters, 'paradiag')
        self.snes.setOptionsPrefix('paradiag')
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
            J.assemble()
            P.assemble()

        self.snes.setJacobian(form_jacobian, J=Jacmat, P=Jacmat)

        pc = PETSc.PC().create()
        pc.setType("python")
        pc.setPythonType("DiagFFTPC")
        pc.setAttr("paradiag", self)
        print("ask lawrence how to set the pc in the snes")
        print("will need to put pc.delAttr('paradiag') somewhere")

        # complete the snes setup
        self.opts.set_from_options(self.snes)

    def update(self, X):
        # Update self.w_alls and self.w_recv
        # from X.
        # The local parts of X are copied into self.w_alls
        # and the last step from the previous slice (periodic)
        # is copied into self.u_prev

        n = self.ensemble.ensemble_comm.size

        with self.w_all.dat.vec_wo as v:
            v.array[:] = X.array_r

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

        if self.circ == "picard":
            raise NotImplementedError
        else:
            # One shot
            with self.opts.inserted_options():
                self.snes.solve(None, self.X)
            self.update(self.X)
