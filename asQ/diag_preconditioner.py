import numpy as np
import firedrake as fd
from scipy.fft import fft, ifft
from firedrake.petsc import PETSc
# from mpi4py_fft.pencil import Pencil, Subcomm
from asQ.pencil import Pencil, Subcomm
import importlib
from asQ.profiling import memprofile
from asQ.common import get_option_from_list
from asQ.allatoncesystem import time_average

from functools import partial

import asQ.complex_proxy.vector as cpx


def construct_riesz_map(W, prefix, fieldsplit=False, riesz_options=None):
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
    if fieldsplit:
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
        self.function_space = self.aaos.function_space

        W_all = self.aaos.function_space_all
        # sanity check
        assert (self.function_space.dim()*paradiag.nlocal_timesteps == W_all.dim())

        # Gamma coefficients
        exponents = np.arange(self.ntimesteps)/self.ntimesteps
        self.gamma = paradiag.alpha**exponents

        slice_begin = self.aaos.transform_index(0, from_range='slice', to_range='window')
        slice_end = slice_begin + self.nlocal_timesteps
        self.gamma_slice = self.gamma[slice_begin:slice_end]

        # circulant eigenvalues
        C1col = np.zeros(self.ntimesteps)
        C2col = np.zeros(self.ntimesteps)

        dt = self.aaos.dt
        self.t_average = fd.Constant(self.aaos.t0 + (self.ntimesteps + 1)*dt/2)
        theta = self.aaos.theta
        C1col[:2] = np.array([1, -1])/dt
        C2col[:2] = np.array([theta, 1-theta])

        self.D1 = fft(self.gamma*C1col, norm='backward')
        self.D2 = fft(self.gamma*C2col, norm='backward')

        # Block system setup
        # First need to build the vector function space version of function_space
        self.cpx_function_space = cpx.FunctionSpace(self.function_space)

        # set the boundary conditions to zero for the residual
        self.block_bcs = tuple((cb
                                for bc in self.aaos.boundary_conditions
                                for cb in cpx.DirichletBC(self.cpx_function_space, self.function_space,
                                                          bc, 0*bc.function_arg)))

        # function to do global reduction into for average block jacobian
        if jac_state in ('window', 'slice'):
            self.ureduce = fd.Function(self.function_space)
            self.uwrk = fd.Function(self.function_space)

        # input and output functions to the block solve
        self.block_rhs = fd.Function(self.cpx_function_space)
        self.block_sol = fd.Function(self.cpx_function_space)

        # A place to store the real/imag components of the all-at-once residual
        self.xreal = fd.Function(W_all)
        self.ximag = fd.Function(W_all)

        # setting up the FFT stuff
        # construct simply dist array and 1d fftn:
        subcomm = Subcomm(self.ensemble.ensemble_comm, [0, 1])
        # dimensions of space-time data in this ensemble_comm
        nlocal = self.function_space.node_set.size
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

        # setting up the Riesz map to project residual into complex space
        is_mixed = isinstance(self.function_space.ufl_element(), fd.MixedElement)
        rmap_rhs = construct_riesz_map(self.cpx_function_space,
                                       prefix=f"{prefix}{self.prefix}mass_",
                                       fieldsplit=is_mixed)
        self.riesz_proj, self.riesz_rhs = rmap_rhs

        # Now need to build the block solvers

        # time-average function to linearise around
        self.u0 = fd.Function(self.cpx_function_space)

        self.block_solvers = tuple((self._make_block(i, f"{prefix}{self.prefix}")
                                    for i in range(self.nlocal_timesteps)))

        self.initialized = True

    def _make_block(self, i, prefix):
        """
        Construct the LinearVariationalSolver for block index i.
        """
        # building the Jacobian of the nonlinear term
        # what we want is a block diagonal matrix in the 2x2 system
        # coupling the real and imaginary parts.
        # We achieve this by copying w_all into both components of u0
        # building the nonlinearity separately for the real and imaginary
        # parts and then linearising.
        # This is constructed by cpx.derivative

        #  Building the nonlinear operator

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

        ii = self.aaos.transform_index(i, from_range='slice', to_range='window')
        d1 = self.D1[ii]
        d2 = self.D2[ii]

        M, D1r, D1i = cpx.BilinearForm(self.cpx_function_space, d1, form_mass, return_z=True)
        K, D2r, D2i = cpx.derivative(d2, form_function, self.u0, return_z=True)

        A = M + K

        # The rhs
        v = fd.TestFunction(self.cpx_function_space)
        L = fd.inner(v, self.block_rhs)*fd.dx

        # pass sigma into PC:
        appctx_h = {}

        # Options with prefix 'diagfft_block_' apply to all blocks by default
        # If any options with prefix 'diagfft_block_{i}' exist, where i is the
        # block number, then this prefix is used instead (like pc fieldsplit)

        block_prefix = f"{prefix}block_"
        for k, v in PETSc.Options().getAll().items():
            if k.startswith(f"{block_prefix}{str(ii)}_"):
                block_prefix = f"{block_prefix}{str(ii)}_"
                break

        block_prob = fd.LinearVariationalProblem(A, L, self.block_sol,
                                                 bcs=self.block_bcs)
        block_solver = fd.LinearVariationalSolver(block_prob,
                                                  appctx=appctx_h,
                                                  options_prefix=block_prefix)
        # multigrid transfer manager
        if 'diag_transfer_managers' in self.paradiag.block_ctx:
            # block_solver.set_transfer_manager(self.paradiag.block_ctx['diag_transfer_managers'][ii])
            tm = self.paradiag.block_ctx['diag_transfer_managers'][i]
            block_solver.set_transfer_manager(tm)
            tm_set = (block_solver._ctx.transfer_manager is tm)

            if tm_set is False:
                print(f"transfer manager not set on block_solvers[{ii}]")

        return block_solver

    def _record_diagnostics(self):
        """
        Update diagnostic information from block linear solvers.

        Must be called exactly once at the end of each apply()
        """
        for i in range(self.aaos.nlocal_timesteps):
            its = self.block_solvers[i].snes.getLinearSolveIterations()
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

    @PETSc.Log.EventDecorator()
    @memprofile
    def to_eigenbasis(self, xreal, ximag, output='real,imag'):
        """
        In-place transform of the complex vector (xreal, ximag) to the preconditioner (block-)eigenbasis.
        :arg xreal: real part of input and output
        :arg ximag: real part of input and output
        :arg output: which parts of the result to copy the back into xreal and/or ximag.
        """
        # copy data into working array
        with xreal.dat.vec_ro as v:
            self.a0.real[:] = v.array_r.reshape((self.aaos.nlocal_timesteps,
                                                 self.function_space.node_set.size))
        with ximag.dat.vec_ro as v:
            self.a0.imag[:] = v.array_r.reshape((self.aaos.nlocal_timesteps,
                                                 self.function_space.node_set.size))

        # alpha-weighting
        self.a0.real[:] = (self.gamma_slice*self.a0.real.T).T

        # transpose forward
        self.transfer.forward(self.a0, self.a1)

        # FFT
        self.a1[:] = fft(self.a1, axis=0)

        # transpose backward
        self.transfer.backward(self.a1, self.a0)

        # copy back into output
        if 'real' in output:
            with xreal.dat.vec_wo as v:
                v.array[:] = self.a0.real.reshape(-1)
        if 'imag' in output:
            with ximag.dat.vec_wo as v:
                v.array[:] = self.a0.imag.reshape(-1)

    @PETSc.Log.EventDecorator()
    @memprofile
    def from_eigenbasis(self, xreal, ximag, output='real,imag'):
        """
        In-place transform of the complex vector (xreal, ximag) from the preconditioner (block-)eigenbasis.
        :arg xreal: real part of input and output
        :arg ximag: real part of input and output
        :arg output: which parts of the result to copy the back into xreal and/or ximag.
        """
        # copy data into working array
        with xreal.dat.vec_ro as v:
            self.a0.real[:] = v.array_r.reshape((self.aaos.nlocal_timesteps,
                                                 self.function_space.node_set.size))
        with ximag.dat.vec_ro as v:
            self.a0.imag[:] = v.array_r.reshape((self.aaos.nlocal_timesteps,
                                                 self.function_space.node_set.size))

        # transpose forward
        self.transfer.forward(self.a0, self.a1)

        # IFFT
        self.a1[:] = ifft(self.a1, axis=0)

        # transpose backward
        self.transfer.backward(self.a1, self.a0)

        # alpha-weighting
        self.a0[:] = ((1.0/self.gamma_slice)*self.a0.T).T

        # copy back into output
        if 'real' in output:
            with xreal.dat.vec_wo as v:
                v.array[:] = self.a0.real.reshape(-1)
        if 'imag' in output:
            with ximag.dat.vec_wo as v:
                v.array[:] = self.a0.imag.reshape(-1)

    @PETSc.Log.EventDecorator()
    @memprofile
    def solve_blocks(self, xreal, ximag):
        """
        Solve each of the blocks in the diagonalised preconditioner with
        complex vector (xreal,ximag) as the right-hand-sides.
        :arg xreal: real part of input and output
        :arg ximag: real part of input and output
        """
        def get_field(i, x):
            return self.aaos.get_field_components(i, f_alls=x.subfunctions)

        for i in range(self.aaos.nlocal_timesteps):
            self.block_rhs.assign(0.)
            self.block_sol.assign(0.)

            # copy the data into solver input
            cpx.set_real(self.riesz_rhs, get_field(i, xreal))
            cpx.set_imag(self.riesz_rhs, get_field(i, ximag))

            # Do a project for Riesz map, to be superceded when we get Cofunction
            self.riesz_proj.solve(self.block_rhs, self.riesz_rhs)

            # solve the block system
            self.block_solvers[i].solve()

            # copy the data from solver output
            cpx.get_real(self.block_sol, get_field(i, xreal))
            cpx.get_imag(self.block_sol, get_field(i, ximag))

    @PETSc.Log.EventDecorator()
    @memprofile
    def apply(self, pc, x, y):

        # copy input Vec into Function
        with self.xreal.dat.vec_wo as v:
            x.copy(v)
        self.ximag.assign(0.)

        # forward FFT
        self.to_eigenbasis(self.xreal, self.ximag)

        # Do the block solves
        self.solve_blocks(self.xreal, self.ximag)

        # backward IFFT
        self.from_eigenbasis(self.xreal, self.ximag, output='real')

        # copy solution into output Vec
        with self.xreal.dat.vec_ro as v:
            v.copy(y)

        self._record_diagnostics()

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError
