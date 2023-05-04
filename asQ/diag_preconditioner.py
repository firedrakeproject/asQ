import numpy as np
import firedrake as fd
from scipy.fft import fft, ifft
from firedrake.petsc import PETSc
# from mpi4py_fft.pencil import Pencil, Subcomm
from asQ.pencil import Pencil, Subcomm
import importlib
from asQ.profiling import memprofile

import asQ.complex_proxy.vector as cpx


def construct_riesz_map(V, W, prefix, riesz_options=None):
    """
    Construct projection into W assuming W is a complex-proxy
    FunctionSpace for the real FunctionSpace V.

    :arg V: a real-valued FunctionSpace.
    :arg W: a complex-proxy FunctionSpace for V.
    :arg prefix: the prefix for the PETSc options for the projection solve.
    :arg riesz_options: PETSc options for the projection solve. Defaults to direct solve.
    """
    # default is to solve directly
    if riesz_options is None:
        riesz_options = {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
            'mat_type': 'aij'
        }

    # mixed mass matrices are decoupled so solve seperately
    if isinstance(V.ufl_element(), fd.MixedElement):
        full_riesz_options = {
            'ksp_type': 'preonly',
            'mat_type': 'nest',
            'pc_type': 'fieldsplit',
            'pc_field_split_type': 'additive',
            'fieldsplit': riesz_options
        }
    else:
        full_riesz_options = riesz_options

    # mat types
    mat_type = PETSc.Options().getString(
        f"{prefix}mat_type",
        default=riesz_options['mat_type'])

    sub_mat_type = PETSc.Options().getString(
        f"{prefix}fieldsplit_mat_type",
        default=riesz_options['mat_type'])

    # input for riesz map
    rhs = fd.Function(W)

    # construct forms
    v = fd.TestFunction(W)
    u = fd.TrialFunction(W)

    a = fd.assemble(fd.inner(u, v)*fd.dx,
                    mat_type=mat_type,
                    sub_mat_type=sub_mat_type)

    # create LinearSolver
    rmap = fd.LinearSolver(a, solver_parameters=full_riesz_options,
                           options_prefix=f"{prefix}")

    return rmap, rhs


class DiagFFTPC(object):
    prefix = "diagfft_"

    def __init__(self):
        r"""A preconditioner for all-at-once systems with alpha-circulant
        block diagonal structure, using FFT.
        """
        self.initialized = False

    @memprofile
    def setUp(self, pc):
        """Setup method called by PETSc."""
        if not self.initialized:
            self.initialize(pc)
        self.update(pc)

    @memprofile
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
        self.jac_average = PETSc.Options().getString(
            f"{prefix}{self.prefix}jac_average", default='window')

        valid_jac_averages = ['window', 'slice']

        if self.jac_average not in valid_jac_averages:
            raise ValueError("diagfft_jac_average must be one of "+" or ".join(valid_jac_averages))

        # this time slice part of the all at once solution
        self.w_all = self.aaos.w_all

        # basic model function space
        self.blockV = self.aaos.function_space

        W_all = self.aaos.function_space_all
        # sanity check
        assert (self.blockV.dim()*paradiag.nlocal_timesteps == W_all.dim())

        # Input/Output wrapper Functions for all-at-once residual being acted on
        self.xfunc = fd.Function(W_all)  # input
        self.yfunc = fd.Function(W_all)  # output

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
        theta = self.aaos.theta
        C1col[:2] = np.array([1, -1])/dt
        C2col[:2] = np.array([theta, 1-theta])

        self.D1 = fft(self.Gam*C1col, norm='backward')
        self.D2 = fft(self.Gam*C2col, norm='backward')

        # Block system setup
        # First need to build the vector function space version of blockV
        self.CblockV = cpx.FunctionSpace(self.blockV)

        # set the boundary conditions to zero for the residual
        self.CblockV_bcs = tuple((cb
                                  for bc in self.aaos.boundary_conditions
                                  for cb in cpx.DirichletBC(self.CblockV, self.blockV,
                                                            bc, 0*bc.function_arg)))

        # function to do global reduction into for average block jacobian
        if self.jac_average == 'window':
            self.ureduce = fd.Function(self.blockV)
            self.uwrk = fd.Function(self.blockV)

        # input and output functions to the block solve
        self.block_rhs = fd.Function(self.CblockV)
        self.block_sol = fd.Function(self.CblockV)

        # A place to store the real/imag components of the all-at-once residual after fft
        self.xreal = fd.Function(W_all)
        self.ximag = fd.Function(W_all)

        # setting up the FFT stuff
        # construct simply dist array and 1d fftn:
        subcomm = Subcomm(self.ensemble.ensemble_comm, [0, 1])
        # dimensions of space-time data in this ensemble_comm
        nlocal = self.blockV.node_set.size
        NN = np.array([self.ntimesteps, nlocal], dtype=int)
        # transfer pencil is aligned along axis 1
        self.p0 = Pencil(subcomm, NN, axis=1)
        # a0 is the local part of our fft working array
        # has shape of (partition/P, nlocal)
        self.a0 = np.zeros(self.p0.subshape, complex)
        self.p1 = self.p0.pencil(0)
        # a0 is the local part of our other fft working array
        self.a1 = np.zeros(self.p1.subshape, complex)
        self.transfer = self.p0.transfer(self.p1, complex)
        self.cwrk = np.zeros((self.nlocal_timesteps, nlocal), dtype=complex)

        # setting up the Riesz map to project residual into complex space
        rmap_rhs = construct_riesz_map(self.blockV,
                                       self.CblockV,
                                       prefix=f"{prefix}{self.prefix}mass_")
        self.riesz_proj, self.riesz_rhs = rmap_rhs

        # building the Jacobian of the nonlinear term
        # what we want is a block diagonal matrix in the 2x2 system
        # coupling the real and imaginary parts.
        # We achieve this by copying w_all into both components of u0
        # building the nonlinearity separately for the real and imaginary
        # parts and then linearising.
        # This is constructed by cpx.derivative

        #  Building the nonlinear operator
        self.Jsolvers = []
        form_mass = self.aaos.form_mass
        form_function = self.aaos.form_function

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
            L = fd.inner(v, self.block_rhs)*fd.dx

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

            jprob = fd.LinearVariationalProblem(A, L, self.block_sol,
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

    def _record_diagnostics(self):
        """
        Update diagnostic information from block linear solvers.

        Must be called exactly once at the end of each apply()
        """
        for i in range(self.aaos.nlocal_timesteps):
            its = self.Jsolvers[i].snes.getLinearSolveIterations()
            self.paradiag.block_iterations.dlocal[i] += its

    @PETSc.Log.EventDecorator()
    @memprofile
    def update(self, pc):
        '''
        we need to update u0 from w_all, containing state.
        we copy w_all into the "real" and "imaginary" parts of u0
        this is so that when we linearise the nonlinearity, we get
        an operator that is block diagonal in the 2x2 system coupling
        real and imaginary parts.
        '''
        self.ureduce.assign(0)

        urs = self.ureduce.subfunctions
        for i in range(self.nlocal_timesteps):
            for ur, ui in zip(urs, self.aaos.get_field_components(i)):
                ur.assign(ur + ui)

        # average only over current time-slice
        if self.jac_average == 'slice':
            self.ureduce /= fd.Constant(self.nlocal_timesteps)
        else:  # implies self.jac_average == 'window':
            self.paradiag.ensemble.allreduce(self.ureduce, self.uwrk)
            self.ureduce.assign(self.uwrk/fd.Constant(self.ntimesteps))

        cpx.set_real(self.u0, self.ureduce)
        cpx.set_imag(self.u0, self.ureduce)

    @PETSc.Log.EventDecorator()
    @memprofile
    def apply(self, pc, x, y):

        # copy petsc vec into Function
        # hopefully this works
        with self.xfunc.dat.vec_wo as v:
            x.copy(v)

        # get array of basis coefficients
        with self.xfunc.dat.vec_ro as v:
            self.cwrk.real[:] = v.array_r.reshape((self.aaos.nlocal_timesteps,
                                                   self.blockV.node_set.size))
            self.cwrk.imag[:] = 0

        # This produces an array whose rows are time slices
        # and columns are finite element basis coefficients

        ######################
        # Diagonalise - scale, transfer, FFT, transfer, Copy
        # Scale
        # is there a better way to do this with broadcasting?

        self.cwrk[:] = (1.0+0.j)*(self.Gam_slice*self.cwrk.T).T

        # transfer forward
        self.a0[:] = self.cwrk[:]
        self.transfer.forward(self.a0, self.a1)
        # FFT
        self.a1[:] = fft(self.a1, axis=0)

        # transfer backward
        self.transfer.backward(self.a1, self.a0)
        # Copy into ximag, xreal
        self.cwrk[:] = self.a0[:]

        with self.xreal.dat.vec_wo as v:
            v.array[:] = self.cwrk.real.reshape(-1)
        with self.ximag.dat.vec_wo as v:
            v.array[:] = self.cwrk.imag.reshape(-1)
        #####################

        # Do the block solves

        for i in range(self.aaos.nlocal_timesteps):
            # copy the data into solver input
            self.riesz_rhs.assign(0.)

            cpx.set_real(self.riesz_rhs, self.aaos.get_field_components(i, f_alls=self.xreal.subfunctions))
            cpx.set_imag(self.riesz_rhs, self.aaos.get_field_components(i, f_alls=self.ximag.subfunctions))

            # Do a project for Riesz map, to be superceded
            # when we get Cofunction
            self.riesz_proj.solve(self.block_rhs, self.riesz_rhs)

            # solve the block system
            self.block_sol.assign(0.)
            self.Jsolvers[i].solve()

            # copy the data from solver output
            cpx.get_real(self.block_sol, self.aaos.get_field_components(i, f_alls=self.xreal.subfunctions))
            cpx.get_imag(self.block_sol, self.aaos.get_field_components(i, f_alls=self.ximag.subfunctions))

        ######################
        # Undiagonalise - Copy, transfer, IFFT, transfer, scale, copy
        # get array of basis coefficients
        with self.ximag.dat.vec_ro as v:
            self.cwrk[:] = 1j*v.array_r.reshape((self.aaos.nlocal_timesteps,
                                                 self.blockV.node_set.size))
        with self.xreal.dat.vec_ro as v:
            self.cwrk[:] += v.array_r.reshape((self.aaos.nlocal_timesteps,
                                               self.blockV.node_set.size))
        # transfer forward
        self.a0[:] = self.cwrk[:]
        self.transfer.forward(self.a0, self.a1)
        # IFFT
        self.a1[:] = ifft(self.a1, axis=0)
        # transfer backward
        self.transfer.backward(self.a1, self.a0)
        self.cwrk[:] = self.a0[:]
        # scale
        self.cwrk[:] = ((1.0/self.Gam_slice)*self.cwrk.T).T

        # Copy into ximag, xreal
        with self.yfunc.dat.vec_wo as v:
            v.array[:] = self.cwrk.reshape(-1).real
        with self.yfunc.dat.vec_ro as v:
            v.copy(y)
        ################

        self._record_diagnostics()

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError
