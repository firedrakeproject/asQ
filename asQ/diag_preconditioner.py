import numpy as np
import firedrake as fd
from scipy.fft import fft, ifft
from firedrake.petsc import PETSc
# from mpi4py_fft.pencil import Pencil, Subcomm
from asQ.pencil import Pencil, Subcomm
from operator import mul
from functools import reduce
import importlib
from ufl.classes import MultiIndex, FixedIndex, Indexed


class DiagFFTPC(object):
    prefix = "diagfft_"

    def __init__(self):
        r"""A preconditioner for all-at-once systems with alpha-circulant
        block diagonal structure, using FFT.
        """
        self.initialized = False

    def setUp(self, pc):
        """Setup method called by PETSc."""
        if not self.initialized:
            self.initialize(pc)
        self.update(pc)

    def initialize(self, pc):
        if pc.getType() != "python":
            raise ValueError("Expecting PC type python")

        prefix = pc.getOptionsPrefix()

        # get hook to paradiag object
        sentinel = object()
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

        paradiag = self.context["paradiag"]
        self.paradiag = paradiag
        aaos = paradiag.aaos
        self.aaos = paradiag.aaos

        paradiag.diagfftpc = self

        # option for whether to use slice or window average for block jacobian
        self.jac_average = PETSc.Options().getString(
            f"{prefix}{self.prefix}jac_average", default='window')

        valid_jac_averages = ['window', 'slice']

        if self.jac_average not in valid_jac_averages:
            raise ValueError("diagfft_jac_average must be one of "+" or ".join(valid_jac_averages))

        # this time slice part of the all at once solution
        self.w_all = aaos.w_all

        partition = np.array(paradiag.time_partition)
        self.time_partition = partition

        ensemble = paradiag.ensemble
        self.ensemble = ensemble

        time_rank = paradiag.time_rank  # the time rank
        self.time_rank = time_rank

        # basic model function space
        self.blockV = aaos.function_space

        W_all = aaos.function_space_all
        # sanity check
        assert (self.blockV.dim()*partition[time_rank] == W_all.dim())

        # Input/Output wrapper Functions
        self.xf = fd.Function(W_all)  # input
        self.yf = fd.Function(W_all)  # output

        # Gamma coefficients
        self.Nt = np.sum(partition)
        Nt = self.Nt
        exponents = np.arange(self.Nt)/self.Nt
        alphav = paradiag.alpha
        self.Gam = alphav**exponents
        self.Gam_slice = self.Gam[np.sum(partition[:time_rank]):np.sum(partition[:time_rank+1])]

        # Di coefficients
        thetav = aaos.theta
        Dt = aaos.dt
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
        self.ncpts = len(self.blockV)
        V_cpts = self.blockV.split()
        ComplexCpts = []
        for V_cpt in V_cpts:
            rank = V_cpt.rank
            V_cpt_ele = V_cpt.ufl_element()
            if rank == 0:  # scalar basis coefficients
                ComplexCpts.append(fd.VectorElement(V_cpt_ele, dim=2))
            elif rank == 1:  # vector basis coefficients
                dim = V_cpt_ele.num_sub_elements()
                shape = (2, dim)
                scalar_element = V_cpt_ele.sub_elements()[0]
                ComplexCpts.append(fd.TensorElement(scalar_element, shape))
            else:
                assert (rank > 0)
                shape = (2,) + V_cpt_ele._shape
                scalar_element = V_cpt_ele.sub_elements()[0]
                ComplexCpts.append(fd.TensorElement(scalar_element, shape))
        self.CblockV = reduce(mul, [fd.FunctionSpace(mesh, ComplexCpt)
                                    for ComplexCpt in ComplexCpts])

        # get the boundary conditions
        self.set_CblockV_bcs()

        # Now need to build the block solver
        vs = fd.TestFunctions(self.CblockV)
        uts = fd.TrialFunctions(self.CblockV)
        self.u0 = fd.Function(self.CblockV)  # we will create a linearisation
        us = fd.split(self.u0)

        # function to do global reduction into for average block jacobian
        if self.jac_average == 'window':
            self.ureduce = fd.Function(self.blockV)
            self.ubuf = fd.Function(self.blockV)
            self.ureduceC = fd.Function(self.CblockV)

        # extract the real and imaginary parts
        vsr = []
        vsi = []
        utsr = []
        utsi = []
        usr = []
        usi = []

        if isinstance(Ve, fd.MixedElement):
            N = Ve.num_sub_elements()
            for i in range(N):
                part = vs[i]
                idxs = fd.indices(len(part.ufl_shape) - 1)
                vsr.append(fd.as_tensor(Indexed(part, MultiIndex((FixedIndex(0), *idxs))), idxs))
                vsi.append(fd.as_tensor(Indexed(part, MultiIndex((FixedIndex(1), *idxs))), idxs))
                part = us[i]
                idxs = fd.indices(len(part.ufl_shape) - 1)
                usr.append(fd.as_tensor(Indexed(part, MultiIndex((FixedIndex(0), *idxs))), idxs))
                usi.append(fd.as_tensor(Indexed(part, MultiIndex((FixedIndex(1), *idxs))), idxs))
                part = uts[i]
                idxs = fd.indices(len(part.ufl_shape) - 1)
                utsr.append(fd.as_tensor(Indexed(part, MultiIndex((FixedIndex(0), *idxs))), idxs))
                utsi.append(fd.as_tensor(Indexed(part, MultiIndex((FixedIndex(1), *idxs))), idxs))
        else:
            vsr.append(vs[0])
            vsi.append(vs[1])
            usr.append(us[0])
            usi.append(us[1])
            utsr.append(uts[0])
            utsi.append(uts[1])

        # input and output functions
        self.Jprob_in = fd.Function(self.CblockV)
        self.Jprob_out = fd.Function(self.CblockV)

        # A place to store all the inputs to the block problems
        self.xfi = fd.Function(W_all)
        self.xfr = fd.Function(W_all)

        #  Building the nonlinear operator
        self.Jsolvers = []
        self.Js = []
        form_mass = aaos.form_mass
        form_function = aaos.form_function

        # setting up the FFT stuff
        # construct simply dist array and 1d fftn:
        subcomm = Subcomm(self.ensemble.ensemble_comm, [0, 1])
        # get some dimensions
        nlocal = self.blockV.node_set.size
        NN = np.array([np.sum(partition), nlocal], dtype=int)
        # transfer pencil is aligned along axis 1
        self.p0 = Pencil(subcomm, NN, axis=1)
        # a0 is the local part of our fft working array
        # has shape of (partition/P, nlocal)
        self.a0 = np.zeros(self.p0.subshape, complex)
        self.p1 = self.p0.pencil(0)
        # a0 is the local part of our other fft working array
        self.a1 = np.zeros(self.p1.subshape, complex)
        self.transfer = self.p0.transfer(self.p1, complex)

        # setting up the Riesz map
        # input for the Riesz map
        self.xtemp = fd.Function(self.CblockV)
        v = fd.TestFunction(self.CblockV)
        u = fd.TrialFunction(self.CblockV)
        a = fd.assemble(fd.inner(u, v)*fd.dx)
        self.Proj = fd.LinearSolver(a, options_prefix=self.prefix+"mass_")

        # building the Jacobian of the nonlinear term
        # what we want is a block diagonal matrix in the 2x2 system
        # coupling the real and imaginary parts.
        # We achieve this by copying w_all into both components of u0
        # building the nonlinearity separately for the real and imaginary
        # parts and then linearising.

        Nrr = form_function(*usr, *vsr)
        Nri = form_function(*usr, *vsi)
        Nir = form_function(*usi, *vsr)
        Nii = form_function(*usi, *vsi)
        Jrr = fd.derivative(Nrr, self.u0)
        Jri = fd.derivative(Nri, self.u0)
        Jir = fd.derivative(Nir, self.u0)
        Jii = fd.derivative(Nii, self.u0)

        # building the block problem solvers
        for i in range(partition[time_rank]):
            ii = np.sum(partition[:time_rank])+i  # global time time index
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

            # The linear operator
            A = (
                D1r*form_mass(*utsr, *vsr)
                - D1i*form_mass(*utsi, *vsr)
                + D2r*Jrr
                - D2i*Jir
                + D1r*form_mass(*utsi, *vsi)
                + D1i*form_mass(*utsr, *vsi)
                + D2r*Jii
                + D2i*Jri
            )

            # The rhs
            v = fd.TestFunction(self.CblockV)
            L = fd.inner(v, self.Jprob_in)*fd.dx

            # PETSc has hard-coded limit of 512 options per manager
            # Having different options for each block can easily reach this limit
            # Switched to using the same options for all blocks until this is fixed in PETSc
            # block_prefix = self.prefix+str(ii)+'_'
            block_prefix = self.prefix+'block_'

            jprob = fd.LinearVariationalProblem(A, L, self.Jprob_out,
                                                bcs=self.CblockV_bcs)
            Jsolver = fd.LinearVariationalSolver(jprob,
                                                 appctx=appctx_h,
                                                 options_prefix=block_prefix)
            # multigrid transfer manager
            if 'diag_transfer_managers' in paradiag.block_ctx:
                # Jsolver.set_transfer_manager(paradiag.block_ctx['diag_transfer_managers'][ii])
                tm = paradiag.block_ctx['diag_transfer_managers'][i]
                Jsolver.set_transfer_manager(tm)
                tm_set = (Jsolver._ctx.transfer_manager is tm)

                if tm_set is False:
                    print(f"transfer manager not set on Jsolvers[{ii}]")

            self.Jsolvers.append(Jsolver)

        self.initialized = True

    def set_CblockV_bcs(self):
        self.CblockV_bcs = []
        for bc in self.aaos.boundary_conditions:
            is_mixed_element = isinstance(self.aaos.function_space.ufl_element(),
                                          fd.MixedElement)
            for r in range(2):  # Complex coefficient index
                if is_mixed_element:
                    i = bc.function_space().index
                    all_bc = fd.DirichletBC(self.CblockV.sub(i).sub(r),
                                            0*bc.function_arg,
                                            bc.sub_domain)
                else:
                    all_bc = fd.DirichletBC(self.CblockV.sub(r),
                                            0*bc.function_arg,
                                            bc.sub_domain)
                self.CblockV_bcs.append(all_bc)

    @PETSc.Log.EventDecorator()
    def update(self, pc):
        '''
        we need to update u0 from w_all, containing state.
        we copy w_all into the "real" and "imaginary" parts of u0
        this is so that when we linearise the nonlinearity, we get
        an operator that is block diagonal in the 2x2 system coupling
        real and imaginary parts.
        '''
        self.u0.assign(0)
        for i in range(self.aaos.nlocal_timesteps):
            # copy the data into solver input
            if self.ncpts > 1:
                u0s = self.u0.split()
            for r in range(2):
                if self.ncpts > 1:
                    for cpt in range(self.ncpts):
                        u0s[cpt].sub(r).assign(u0s[cpt].sub(r)
                                               + self.w_all.split()[self.ncpts*i+cpt])
                else:
                    self.u0.sub(r).assign(self.u0.sub(r)
                                          + self.w_all.split()[i])

        # average only over current time-slice
        if self.jac_average == 'slice':
            self.u0 /= self.nlocal_timesteps
        else:  # implies self.jac_average == 'window':
            self.paradiag.ensemble.allreduce(self.u0, self.ureduceC)
            self.u0.assign(self.ureduceC)
            self.u0 /= sum(self.time_partition)

    @PETSc.Log.EventDecorator()
    def apply(self, pc, x, y):

        # copy petsc vec into Function
        # hopefully this works
        with self.xf.dat.vec_wo as v:
            x.copy(v)

        # get array of basis coefficients
        with self.xf.dat.vec_ro as v:
            parray = v.array_r.reshape((self.aaos.nlocal_timesteps,
                                        self.blockV.node_set.size))
        # This produces an array whose rows are time slices
        # and columns are finite element basis coefficients

        ######################
        # Diagonalise - scale, transfer, FFT, transfer, Copy
        # Scale
        # is there a better way to do this with broadcasting?
        parray = (1.0+0.j)*(self.Gam_slice*parray.T).T*np.sqrt(self.Nt)
        # transfer forward
        self.a0[:] = parray[:]
        self.transfer.forward(self.a0, self.a1)
        # FFT
        self.a1[:] = fft(self.a1, axis=0)

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

        for i in range(self.aaos.nlocal_timesteps):
            # copy the data into solver input
            self.xtemp.assign(0.)

            Jins = self.xtemp.split()
            for cpt in range(self.ncpts):

                self.aaos.get_component(i, cpt, wout=Jins[cpt].sub(0), f_alls=self.xfr.split())
                self.aaos.get_component(i, cpt, wout=Jins[cpt].sub(1), f_alls=self.xfi.split())

            # Do a project for Riesz map, to be superceded
            # when we get Cofunction
            self.Proj.solve(self.Jprob_in, self.xtemp)

            # solve the block system
            self.Jprob_out.assign(0.)
            self.Jsolvers[i].solve()

            # copy the data from solver output
            Jpouts = self.Jprob_out.split()
            for cpt in range(self.ncpts):

                self.aaos.set_component(i, cpt, Jpouts[cpt].sub(0), f_alls=self.xfr.split())
                self.aaos.set_component(i, cpt, Jpouts[cpt].sub(1), f_alls=self.xfi.split())

        ######################
        # Undiagonalise - Copy, transfer, IFFT, transfer, scale, copy
        # get array of basis coefficients
        with self.xfi.dat.vec_ro as v:
            parray = 1j*v.array_r.reshape((self.aaos.nlocal_timesteps,
                                           self.blockV.node_set.size))
        with self.xfr.dat.vec_ro as v:
            parray += v.array_r.reshape((self.aaos.nlocal_timesteps,
                                         self.blockV.node_set.size))
        # transfer forward
        self.a0[:] = parray[:]
        self.transfer.forward(self.a0, self.a1)
        # IFFT
        self.a1[:] = ifft(self.a1, axis=0)
        # transfer backward
        self.transfer.backward(self.a1, self.a0)
        parray[:] = self.a0[:]
        # scale
        parray = ((1.0/self.Gam_slice)*parray.T).T
        # Copy into xfi, xfr
        with self.yf.dat.vec_wo as v:
            v.array[:] = parray.reshape(-1).real
        with self.yf.dat.vec_ro as v:
            v.copy(y)
        ################

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError
