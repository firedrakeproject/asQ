import numpy as np
import firedrake as fd
from scipy.fft import fft, ifft
from firedrake.petsc import PETSc, OptionsManager
from pyop2.mpi import MPI
from mpi4py_fft.pencil import Pencil, Subcomm
from mpi4py_fft import fftw
import importlib


class DiagFFTPC(object):
    prefix = "diagfft_"

    def __init__(self):
        r"""A preconditioner for all-at-once systems with alpha-circulant
        block diagonal structure, using FFT.
        """
        self.initialized = False

    def setUp(self, pc):
        """Setup method called by PETSc."""
        prefix = pc.getOptionsPrefix()

        sentinel = object()
        print(f"{prefix}{self.prefix}context")
        constructor = PETSc.Options().getString(
            f"{prefix}{self.prefix}context", default=sentinel)
        if constructor == sentinel:
            raise ValueError

        mod, fun = constructor.rsplit(".", 1)
        mod = importlib.import_module(mod)
        fun = getattr(mod, fun)
        if isinstance(fun, type):
            fun = fun()
        self.context = fun(pc)
        
        if self.initialized:
            self.update(pc)
        else:
            self.initialize(pc)
            self.initialized = True
        
    def initialize(self, pc):
        if pc.getType() != "python":
            raise ValueError("Expecting PC type python")

        paradiag = self.context["paradiag"]
        self.paradiag = paradiag
        
        # this time slice part of the all at once solution
        self.w_all = paradiag.w_all
        # this is bad naming
        W = paradiag.W_all

        # basic model function space
        self.blockV = paradiag.W
        M = np.array(paradiag.M)
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
        Nt = self.Nt
        print(self.M)
        exponents = np.arange(self.Nt)/self.Nt
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
        subcomm = Subcomm(paradiag.ensemble.ensemble_comm, [0, 1])
        # get some dimensions
        nlocal = self.blockV.node_set.size
        NN = np.array([np.sum(M), nlocal], dtype=int)
        # transfer pencil is aligned along axis 1
        self.p0 = Pencil(subcomm, NN, axis=1)
        # a0 is the local part of our fft working array
        # has shape of (M/P, nlocal)
        self.a0 = np.zeros(self.p0.subshape, complex)
        self.p1 = self.p0.pencil(0)
        # a0 is the local part of our other fft working array
        self.a1 = np.zeros(self.p1.subshape, complex)
        self.transfer = self.p0.transfer(self.p1, complex)
        # the FFTW working arrays
        #self.b_fftin = fftw.aligned(self.a1.shape, dtype=complex)
        #self.b_fftout = fftw.aligned_like(self.b_fftin)
        # FFTW plans
        #self.fft = fftw.fftn(self.b_fftin,
        #                     flags=(fftw.FFTW_MEASURE,),
        #                     axes=(0,), output_array=self.b_fftout)
        #self.fft = fftw.ifftn(self.b_fftin,
        #                      flags=(fftw.FFTW_MEASURE,),
        #                      axes=(0,), output_array=self.b_fftout)

        # setting up the Riesz map
        # input for the Riesz map
        self.xtemp = fd.Function(self.CblockV)
        v = fd.TestFunction(self.CblockV)
        u = fd.TrialFunction(self.CblockV)
        a = fd.assemble(fd.inner(u, v)*fd.dx)
        self.Proj = fd.LinearSolver(a, options_prefix=self.prefix+"mass_")
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
            appctx_h = {}
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

            block_prefix = self.prefix+str(ii)+'_'
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
        self.a1[:] = fft(self.a1, axis=0)
        #self.b_fftin[:] = self.a1[:]
        #self.fft()
        #self.a1[:] = self.b_fftout[:]

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
        self.a1[:] = fft(self.a1, axis=0)
        #self.b_fftin[:] = self.a1[:]
        #self.ifft()
        #self.a1[:] = self.b_fftout[:]
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
