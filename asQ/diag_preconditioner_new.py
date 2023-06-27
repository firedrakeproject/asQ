import numpy as np
import firedrake as fd
from scipy.fft import fft, ifft
from firedrake.petsc import PETSc
# from mpi4py_fft.pencil import Pencil, Subcomm
from asQ.pencil import Pencil, Subcomm
import importlib
from asQ.profiling import profiler
from asQ.common import get_option_from_list
from asQ.allatoncesystem import time_average

from functools import partial

import asQ.complex_proxy.vector as cpx


class DiagFFTPC(object):
    """
    PETSc options:

    'diagfft_linearisation': <'consistent', 'user'>
        Which form to linearise when constructing the block Jacobians.
        Default is 'consistent'.

        'consistent': use the same form used in the AllAtOnceSystem residual.
        'user': use the alternative forms given to the AllAtOnceSystem.

    'diagfft_state': <'window', 'slice', 'linear', 'initial', 'reference'>
        Which state to linearise around when constructing the block Jacobians.
        Default is 'window'.

        'window': use the time average over the entire AllAtOnceSystem.
        'slice': use the time average over timesteps on the local Ensemble member.
        'linear': the form linearised is already linear, so no update to the state is needed
        'initial': use the initial condition is used for all timesteps.
        'reference': use the reference state of the AllAtOnceSystem for all timesteps.

    'diagfft_mass': <LinearSolver options>
        The solver options for the Riesz map.
        Default is {'pc_type': 'lu'}
        Use 'diagfft_mass_fieldsplit' if the single-timestep function space is mixed.

    'diagfft_block_%d': <LinearVariationalSolver options>
        Default is the Firedrake default options.
        The solver options for the %d'th block, enumerated globally.
        Use 'diagfft_block' to set options for all blocks.
    """
    prefix = "diagfft_"

    @profiler()
    def __init__(self):
        r"""A preconditioner for all-at-once systems with alpha-circulant
        block diagonal structure, using FFT.
        """
        self.initialized = False

    @profiler()
    def setUp(self, pc):
        """Setup method called by PETSc."""
        if not self.initialized:
            self.initialize(pc)
        self.update(pc)

    @profiler()
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

        self.ensemble = paradiag.ensemble
        self.paradiag = paradiag
        self.aaos = paradiag.aaos
        self.layout = paradiag.layout
        self.time_partition = paradiag.time_partition
        self.time_rank = paradiag.time_rank
        self.ntimesteps = paradiag.ntimesteps
        self.nlocal_timesteps = paradiag.nlocal_timesteps

        paradiag.diagfftpc = self

        # option for whether to use slice or window average for block jacobian
        valid_jac_state = ['window', 'slice', 'linear', 'initial', 'reference']
        jac_option = f"{prefix}{self.prefix}state"

        self.jac_state = partial(get_option_from_list,
                                 jac_option, valid_jac_state, default_index=0)
        jac_state = self.jac_state()

        if jac_state == 'reference' and self.aaos.reference_state is None:
            raise ValueError("AllAtOnceSystem must be provided a reference state to use \'reference\' for diagfft_jac_state.")

        # this time slice part of the all at once solution
        self.w_all = self.aaos.w_all

        # basic model function space
        self.blockV = self.aaos.function_space

        W_all = self.aaos.function_space_all
        # sanity check
        assert (self.blockV.dim()*paradiag.nlocal_timesteps == W_all.dim())

        # Input/Output wrapper Functions for all-at-once residual being acted on

        # Gamma coefficients
        exponents = np.arange(self.ntimesteps)/self.ntimesteps
        self.Gam = paradiag.alpha**exponents

        slice_begin = self.aaos.transform_index(0, from_range='slice', to_range='window')
        slice_end = slice_begin + self.nlocal_timesteps
        self.Gam_slice = self.Gam[slice_begin:slice_end]

        # circulant eigenvalues
        C1col = np.zeros(self.ntimesteps)
        C2col = np.zeros(self.ntimesteps)

        dt = self.aaos.dt
        self.t_average = fd.Constant(self.aaos.t0 + (self.ntimesteps + 1)*dt/2)
        theta = self.aaos.theta
        C1col[:2] = np.array([1, -1])/dt
        C2col[:2] = np.array([theta, 1-theta])

        self.D1 = np.sqrt(self.ntimesteps)*fft(self.Gam*C1col)
        self.D2 = np.sqrt(self.ntimesteps)*fft(self.Gam*C2col)

        # Block system setup
        # First need to build the vector function space version of blockV
        self.CblockV = cpx.FunctionSpace(self.blockV)

        # set the boundary conditions to zero for the residual
        self.CblockV_bcs = tuple((cb
                                  for bc in self.aaos.boundary_conditions
                                  for cb in cpx.DirichletBC(self.CblockV, self.blockV,
                                                            bc, 0*bc.function_arg)))

        # function to do global reduction into for average block jacobian
        if jac_state in ('window', 'slice'):
            self.ureduce = fd.Function(self.blockV)
            self.uwrk = fd.Function(self.blockV)

        # input and output functions to the block solve
        self.Jprob_in = fd.Function(self.CblockV)
        self.Jprob_out = fd.Function(self.CblockV)

        # A place to store the real/imag components of the all-at-once residual after fft
        self.xfi = fd.Function(W_all)
        self.xfr = fd.Function(W_all)

        # setting up the FFT stuff
        self.smaller_transpose = False
        # construct simply dist array and 1d fftn:
        subcomm = Subcomm(self.ensemble.ensemble_comm, [0, 1])

        # dimensions of space-time data in this ensemble_comm
        nlocal = self.blockV.node_set.size
        NN = np.array([self.ntimesteps, nlocal], dtype=int)

        # transfer pencil is aligned along axis 1
        self.p0 = Pencil(subcomm, NN, axis=1)
        self.p1 = self.p0.pencil(0)

        # data types
        rtype = self.xfr.dat[0].data.dtype
        ctype = complex

        # set up real valued transfer
        self.rtransfer = self.p0.transfer(self.p1, rtype)

        # a0 is the local part of the original data (i.e. distributed in time)
        # has shape of (nlocal_timesteps, nlocal)
        self.ra0 = np.zeros(self.p0.subshape, rtype)

        # a1 is the local part of the transposed data (i.e. distributed in space)
        # has shape of (ntimesteps, ...)
        self.ra1 = np.zeros(self.p1.subshape, rtype)

        # set up complex valued transfer

        # a0 is the local part of the original data (i.e. distributed in time)
        # has shape of (nlocal_timesteps, nlocal)
        self.ca0 = np.zeros(self.p0.subshape, ctype)

        # a1 is the local part of the transposed data (i.e. distributed in space)
        # has shape of (ntimesteps, ...)
        self.ca1 = np.zeros(self.p1.subshape, ctype)

        self.ctransfer = self.p0.transfer(self.p1, ctype)

        # setting up the Riesz map
        default_riesz_method = {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
            'mat_type': 'aij'
        }

        # mixed mass matrices are decoupled so solve seperately
        if isinstance(self.blockV.ufl_element(), fd.MixedElement):
            default_riesz_parameters = {
                'ksp_type': 'preonly',
                'mat_type': 'nest',
                'pc_type': 'fieldsplit',
                'pc_field_split_type': 'additive',
                'fieldsplit': default_riesz_method
            }
        else:
            default_riesz_parameters = default_riesz_method

        # we need to pass the mat_types to assemble directly because
        # it won't pick them up from Options

        riesz_mat_type = PETSc.Options().getString(
            f"{prefix}{self.prefix}mass_mat_type",
            default=default_riesz_parameters['mat_type'])

        riesz_sub_mat_type = PETSc.Options().getString(
            f"{prefix}{self.prefix}mass_fieldsplit_mat_type",
            default=default_riesz_method['mat_type'])

        # input for the Riesz map
        self.xtemp = fd.Function(self.CblockV)
        v = fd.TestFunction(self.CblockV)
        u = fd.TrialFunction(self.CblockV)

        a = fd.assemble(fd.inner(u, v)*fd.dx,
                        mat_type=riesz_mat_type,
                        sub_mat_type=riesz_sub_mat_type)

        self.Proj = fd.LinearSolver(a, solver_parameters=default_riesz_parameters,
                                    options_prefix=f"{prefix}{self.prefix}mass_")

        # building the Jacobian of the nonlinear term
        # what we want is a block diagonal matrix in the 2x2 system
        # coupling the real and imaginary parts.
        # We achieve this by copying w_all into both components of u0
        # building the nonlinearity separately for the real and imaginary
        # parts and then linearising.
        # This is constructed by cpx.derivative

        #  Building the nonlinear operator
        self.Jsolvers = []

        # which form to linearise around
        valid_linearisations = ['consistent', 'user']
        linear_option = f"{prefix}{self.prefix}linearisation"

        linear = get_option_from_list(linear_option, valid_linearisations, default_index=0)

        if linear == 'consistent':
            form_mass = self.aaos.form_mass
            form_function = self.aaos.form_function
        elif linear == 'user':
            form_mass = self.aaos.jacobian_mass
            form_function = self.aaos.jacobian_function

        form_function = partial(form_function, t=self.t_average)

        # Now need to build the block solver
        self.u0 = fd.Function(self.CblockV)  # time average to linearise around

        # building the block problem solvers
        for i in range(self.nlocal_timesteps):
            ii = self.aaos.transform_index(i, from_range='slice', to_range='window')
            d1 = self.D1[ii]
            d2 = self.D2[ii]

            M, D1r, D1i = cpx.BilinearForm(self.CblockV, d1, form_mass, return_z=True)
            K, D2r, D2i = cpx.derivative(d2, form_function, self.u0, return_z=True)

            A = M + K

            # The rhs
            v = fd.TestFunction(self.CblockV)
            L = fd.inner(v, self.Jprob_in)*fd.dx

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

            # Options with prefix 'diagfft_block_' apply to all blocks by default
            # If any options with prefix 'diagfft_block_{i}' exist, where i is the
            # block number, then this prefix is used instead (like pc fieldsplit)

            block_prefix = f"{prefix}{self.prefix}block_"
            for k, v in PETSc.Options().getAll().items():
                if k.startswith(f"{block_prefix}{str(ii)}_"):
                    block_prefix = f"{block_prefix}{str(ii)}_"
                    break

            jprob = fd.LinearVariationalProblem(A, L, self.Jprob_out,
                                                bcs=self.CblockV_bcs)
            Jsolver = fd.LinearVariationalSolver(jprob,
                                                 appctx=appctx_h,
                                                 options_prefix=block_prefix)
            # multigrid transfer manager
            if f'{prefix}transfer_managers' in paradiag.block_ctx:
                # Jsolver.set_transfer_manager(paradiag.block_ctx['diag_transfer_managers'][ii])
                tm = paradiag.block_ctx[f'{prefix}transfer_managers'][i]
                Jsolver.set_transfer_manager(tm)
                tm_set = (Jsolver._ctx.transfer_manager is tm)

                if tm_set is False:
                    print(f"transfer manager not set on Jsolvers[{ii}]")

            self.Jsolvers.append(Jsolver)

        self.initialized = True

    @profiler()
    def _record_diagnostics(self):
        """
        Update diagnostic information from block linear solvers.

        Must be called exactly once at the end of each apply()
        """
        for i in range(self.aaos.nlocal_timesteps):
            its = self.Jsolvers[i].snes.getLinearSolveIterations()
            self.paradiag.block_iterations.dlocal[i] += its

    @profiler()
    def update(self, pc):
        '''
        we need to update u0 from w_all, containing state.
        we copy w_all into the "real" and "imaginary" parts of u0
        this is so that when we linearise the nonlinearity, we get
        an operator that is block diagonal in the 2x2 system coupling
        real and imaginary parts.
        '''
        jac_state = self.jac_state()
        if jac_state == 'linear':
            return

        elif jac_state == 'initial':
            ustate = self.aaos.initial_condition

        elif jac_state == 'reference':
            ustate = self.aaos.reference_state

        elif jac_state in ('window', 'slice'):
            time_average(self.aaos, self.ureduce, self.uwrk, average=jac_state)
            ustate = self.ureduce

        cpx.set_real(self.u0, ustate)
        cpx.set_imag(self.u0, ustate)

        self.t_average.assign(self.aaos.t0 + (self.aaos.ntimesteps + 1)*self.aaos.dt/2)

        return

    @profiler()
    def fft(self):
        """
        FFT of ca1
        """
        self.ca1[:] = fft(self.ca1, axis=0)

    @profiler()
    def ifft(self):
        """
        IFFT of ca1
        """
        self.ca1[:] = ifft(self.ca1, axis=0)

    @profiler()
    def forward_transfer1(self):
        """
        Forward transfer from ra0 to ca1
        """
        if self.smaller_transpose:
            self.rtransfer.forward(self.ra0, self.ra1)
            self.ca1.real[:] = self.ra1
        else:
            self.ca0.real[:] = self.ra0
            self.ctransfer.forward(self.ca0, self.ca1)

    @profiler()
    def backward_transfer2(self):
        """
        Backward transfer from ca1 to ca0
        """
        self.ctransfer.backward(self.ca1, self.ca0)

    @profiler()
    def forward_transfer3(self):
        """
        Forward transfer from ca0 to ca1
        """
        self.ctransfer.forward(self.ca0, self.ca1)

    @profiler()
    def backward_transfer4(self):
        """
        Backward transfer from ca1 to ra1
        """
        if self.smaller_transpose:
            self.ra1[:] = self.ca1.real[:]
            self.rtransfer.backward(self.ra1, self.ra0)
        else:
            self.ctransfer.backward(self.ca1, self.ca0)
            self.ra0[:] = self.ca0.real[:]

    @profiler()
    def apply(self, pc, x, y):

        # This produces an array whose rows are time slices
        # and columns are finite element basis coefficients
        self.ra0[:] = x.array_r.reshape((self.aaos.nlocal_timesteps,
                                         self.blockV.node_set.size))

        ######################
        # Diagonalise - scale, transfer, FFT, transfer, Copy
        # Scale

        self.ra0[:] = (self.Gam_slice*self.ra0.T).T*np.sqrt(self.ntimesteps)

        self.forward_transfer1()
        self.fft()
        self.backward_transfer2()

        # Copy into xfi, xfr
        with self.xfr.dat.vec_wo as v:
            v.array[:] = self.ca0.real.reshape(-1)

        with self.xfi.dat.vec_wo as v:
            v.array[:] = self.ca0.imag.reshape(-1)

        #####################

        # Do the block solves

        for i in range(self.aaos.nlocal_timesteps):
            # copy the data into solver input
            self.xtemp.assign(0.)

            cpx.set_real(self.xtemp, self.aaos.get_field_components(i, f_alls=self.xfr.subfunctions))
            cpx.set_imag(self.xtemp, self.aaos.get_field_components(i, f_alls=self.xfi.subfunctions))

            # Do a project for Riesz map, to be superceded
            # when we get Cofunction
            with PETSc.Log.Event("asQ.DiagFFTPC.apply.riesz_map"):
                self.Proj.solve(self.Jprob_in, self.xtemp)

            # solve the block system
            with PETSc.Log.Event("asQ.DiagFFTPC.apply.block_solves"):
                self.Jprob_out.assign(0.)
                self.Jsolvers[i].solve()

            # copy the data from solver output
            cpx.get_real(self.Jprob_out, self.aaos.get_field_components(i, f_alls=self.xfr.subfunctions))
            cpx.get_imag(self.Jprob_out, self.aaos.get_field_components(i, f_alls=self.xfi.subfunctions))

        ######################
        # Undiagonalise - Copy, transfer, IFFT, transfer, scale, copy
        # get array of basis coefficients
        with self.xfr.dat.vec_ro as v:
            self.ca0.real[:] = v.array_r.reshape((self.aaos.nlocal_timesteps,
                                                  self.blockV.node_set.size))
        with self.xfi.dat.vec_ro as v:
            self.ca0.imag[:] = v.array_r.reshape((self.aaos.nlocal_timesteps,
                                                  self.blockV.node_set.size))

        self.forward_transfer3()
        self.ifft()
        self.backward_transfer4()

        self.ra0[:] = ((1.0/self.Gam_slice)*self.ra0.T).T

        y.array[:] = self.ra0.reshape(-1)

        ################

        self._record_diagnostics()

    @profiler()
    def applyTranspose(self, pc, x, y):
        raise NotImplementedError
