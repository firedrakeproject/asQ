import numpy as np
import firedrake as fd
from scipy.fft import fft, ifft
from firedrake.petsc import PETSc
# from mpi4py_fft.pencil import Pencil, Subcomm
from asQ.pencil import Pencil, Subcomm
import importlib
from asQ.profiling import memprofile
from asQ.common import get_option_from_list
from asQ.allatoncesystem import time_average as time_average_system
from asQ.allatonce.function import time_average as time_average_function
from asQ.allatonce.mixin import TimePartitionMixin
from asQ.parallel_arrays import SharedArray

from functools import partial

import asQ.complex_proxy.vector as cpx

__all__ = ['DiagFFTPC', 'ParaDiagPC']


class ParaDiagPC(TimePartitionMixin):
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
        Use 'diagfft_mass_fieldsplit' if the single-timestep function space is mixed.
        Default is {'pc_type': 'lu'}

    'diagfft_block_%d': <LinearVariationalSolver options>
        The solver options for the %d'th block, enumerated globally.
        Use 'diagfft_block' to set options for all blocks.
        Default is the Firedrake default options.

    'diagfft_dt': <float>
        The timestep size to use in the preconditioning matrix.
        Defaults to the timestep size used in the Jacobian.

    'diagfft_theta': <float>
        The implicit theta method parameter to use in the preconditioning matrix.
        Defaults to the implicit theta method parameter used in the Jacobian.

    'diagfft_alpha': <float>
        The circulant parameter to use in the preconditioning matrix.
        Defaults to 1e-3.
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
        prefix = prefix + self.prefix

        A, _ = pc.getOperators()
        jacobian = A.getPythonContext()
        self.jacobian = jacobian
        self.time_partition_setup(jacobian.ensemble, jacobian.time_partition)

        jacobian.pc = self
        aaofunc = jacobian.current_state
        self.aaofunc = aaofunc

        aaoform = jacobian.aaoform

        appctx = jacobian.appctx

        # option for whether to use slice or window average for block jacobian
        valid_jac_state = ['window', 'slice', 'linear', 'initial', 'reference']
        jac_option = f"{prefix}state"

        self.jac_state = partial(get_option_from_list,
                                 jac_option, valid_jac_state, default_index=0)
        jac_state = self.jac_state()

        if jac_state == 'reference' and jacobian.reference_state is None:
            raise ValueError("AllAtOnceJacobian must be provided a reference state to use \'reference\' for diagfft_state.")

        # basic model function space
        self.blockV = aaofunc.field_function_space

        # Input/Output wrapper Functions for all-at-once residual being acted on
        self.xf = fd.Function(aaofunc.function_space)  # input
        self.yf = fd.Function(aaofunc.function_space)  # output

        # diagonalisation options
        self.dt = PETSc.Options().getReal(
            f"{prefix}dt", default=aaoform.dt)

        self.theta = PETSc.Options().getReal(
            f"{prefix}theta", default=aaoform.theta)

        self.alpha = PETSc.Options().getReal(
            f"{prefix}alpha", default=1e-3)

        dt = self.dt
        theta = self.theta
        alpha = self.alpha
        nt = self.ntimesteps

        # Gamma coefficients
        exponents = np.arange(nt)/nt
        self.Gam = alpha**exponents

        slice_begin = aaofunc.transform_index(0, from_range='slice', to_range='window')
        slice_end = slice_begin + self.nlocal_timesteps
        self.Gam_slice = self.Gam[slice_begin:slice_end]

        # circulant eigenvalues
        C1col = np.zeros(nt)
        C2col = np.zeros(nt)

        C1col[:2] = np.array([1, -1])/dt
        C2col[:2] = np.array([theta, 1-theta])

        self.D1 = np.sqrt(nt)*fft(self.Gam*C1col)
        self.D2 = np.sqrt(nt)*fft(self.Gam*C2col)

        # Block system setup
        # First need to build the vector function space version of blockV
        self.CblockV = cpx.FunctionSpace(self.blockV)

        # set the boundary conditions to zero for the residual
        self.CblockV_bcs = tuple((cb
                                  for bc in aaoform.field_bcs
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
        self.xfi = aaofunc.copy()
        self.xfr = aaofunc.copy()

        # setting up the FFT stuff
        # construct simply dist array and 1d fftn:
        subcomm = Subcomm(self.ensemble.ensemble_comm, [0, 1])
        # dimensions of space-time data in this ensemble_comm
        nlocal = self.blockV.node_set.size
        NN = np.array([nt, nlocal], dtype=int)
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
            f"{prefix}mass_mat_type",
            default=default_riesz_parameters['mat_type'])

        riesz_sub_mat_type = PETSc.Options().getString(
            f"{prefix}mass_fieldsplit_mat_type",
            default=default_riesz_method['mat_type'])

        # input for the Riesz map
        self.xtemp = fd.Function(self.CblockV)
        v = fd.TestFunction(self.CblockV)
        u = fd.TrialFunction(self.CblockV)

        a = fd.assemble(fd.inner(u, v)*fd.dx,
                        mat_type=riesz_mat_type,
                        sub_mat_type=riesz_sub_mat_type)

        self.Proj = fd.LinearSolver(a, solver_parameters=default_riesz_parameters,
                                    options_prefix=f"{prefix}mass_")

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
        linearisation_option = f"{prefix}linearisation"

        linearisation = get_option_from_list(linearisation_option,
                                             valid_linearisations,
                                             default_index=0)

        if linearisation == 'consistent':
            form_mass = aaoform.form_mass
            form_function = aaoform.form_function
        elif linearisation == 'user':
            try:
                form_mass = appctx['pc_form_mass']
                form_function = appctx['pc_form_function']
            except KeyError as err:
                err_msg = "appctx must contain 'pc_form_mass' and 'pc_form_function' if " \
                          + f"{linearisation_option} = 'user'"
                raise type(err)(err_msg) from err

        self.form_mass = form_mass
        self.form_function = form_function

        # Now need to build the block solver
        self.u0 = fd.Function(self.CblockV)  # time average to linearise around

        # building the block problem solvers
        for i in range(self.nlocal_timesteps):
            ii = aaofunc.transform_index(i, from_range='slice', to_range='window')
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

            block_prefix = f"{prefix}block_"
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
            if f'{self.prefix}transfer_managers' in appctx:
                # Jsolver.set_transfer_manager(jacobian.appctx['diagfft_transfer_managers'][ii])
                tm = appctx[f'{self.prefix}transfer_managers'][i]
                Jsolver.set_transfer_manager(tm)
                tm_set = (Jsolver._ctx.transfer_manager is tm)

                if tm_set is False:
                    print(f"transfer manager not set on Jsolvers[{ii}]")

            self.Jsolvers.append(Jsolver)

        self.block_iterations = SharedArray(self.time_partition,
                                            dtype=int,
                                            comm=self.ensemble.ensemble_comm)
        self.initialized = True

    def _record_diagnostics(self):
        """
        Update diagnostic information from block linear solvers.

        Must be called exactly once at the end of each apply()
        """
        for i in range(self.nlocal_timesteps):
            its = self.Jsolvers[i].snes.getLinearSolveIterations()
            self.block_iterations.dlocal[i] += its

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
            ustate = self.aaofunc.initial_condition

        elif jac_state == 'reference':
            ustate = self.jacobian.reference_state

        elif jac_state in ('window', 'slice'):
            time_average_function(self.aaofunc, self.ureduce, self.uwrk, average=jac_state)
            ustate = self.ureduce

        cpx.set_real(self.u0, ustate)
        cpx.set_imag(self.u0, ustate)

        return

    @PETSc.Log.EventDecorator()
    @memprofile
    def apply(self, pc, x, y):

        # copy petsc vec into Function
        # hopefully this works
        with self.xf.dat.vec_wo as v:
            x.copy(v)

        # get array of basis coefficients
        with self.xf.dat.vec_ro as v:
            parray = v.array_r.reshape((self.nlocal_timesteps,
                                        self.blockV.node_set.size))
        # This produces an array whose rows are time slices
        # and columns are finite element basis coefficients

        ######################
        # Diagonalise - scale, transfer, FFT, transfer, Copy
        # Scale
        # is there a better way to do this with broadcasting?
        parray = (1.0+0.j)*(self.Gam_slice*parray.T).T*np.sqrt(self.ntimesteps)
        # transfer forward
        self.a0[:] = parray[:]
        self.transfer.forward(self.a0, self.a1)
        # FFT
        self.a1[:] = fft(self.a1, axis=0)

        # transfer backward
        self.transfer.backward(self.a1, self.a0)
        # Copy into xfi, xfr
        parray[:] = self.a0[:]
        with self.xfr.function.dat.vec_wo as v:
            v.array[:] = parray.real.reshape(-1)
        with self.xfi.function.dat.vec_wo as v:
            v.array[:] = parray.imag.reshape(-1)
        #####################

        # Do the block solves

        for i in range(self.nlocal_timesteps):
            # copy the data into solver input
            self.xtemp.assign(0.)

            cpx.set_real(self.xtemp, self.xfr.get_field_components(i))
            cpx.set_imag(self.xtemp, self.xfi.get_field_components(i))

            # Do a project for Riesz map, to be superceded
            # when we get Cofunction
            self.Proj.solve(self.Jprob_in, self.xtemp)

            # solve the block system
            self.Jprob_out.assign(0.)
            self.Jsolvers[i].solve()

            # copy the data from solver output
            cpx.get_real(self.Jprob_out, self.xfr.get_field_components(i))
            cpx.get_imag(self.Jprob_out, self.xfi.get_field_components(i))

        ######################
        # Undiagonalise - Copy, transfer, IFFT, transfer, scale, copy
        # get array of basis coefficients
        with self.xfi.function.dat.vec_ro as v:
            parray = 1j*v.array_r.reshape((self.nlocal_timesteps,
                                           self.blockV.node_set.size))
        with self.xfr.function.dat.vec_ro as v:
            parray += v.array_r.reshape((self.nlocal_timesteps,
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

        self._record_diagnostics()

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError


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
        self.blockV = self.aaos.function_space

        W_all = self.aaos.function_space_all
        # sanity check
        assert (self.blockV.dim()*paradiag.nlocal_timesteps == W_all.dim())

        # Input/Output wrapper Functions for all-at-once residual being acted on
        self.xf = fd.Function(W_all)  # input
        self.yf = fd.Function(W_all)  # output

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
            form_mass = self.aaos.linearised_mass
            form_function = self.aaos.linearised_function

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
        jac_state = self.jac_state()
        if jac_state == 'linear':
            return

        elif jac_state == 'initial':
            ustate = self.aaos.initial_condition

        elif jac_state == 'reference':
            ustate = self.aaos.reference_state

        elif jac_state in ('window', 'slice'):
            time_average_system(self.aaos, self.ureduce, self.uwrk, average=jac_state)
            ustate = self.ureduce

        cpx.set_real(self.u0, ustate)
        cpx.set_imag(self.u0, ustate)

        return

    @PETSc.Log.EventDecorator()
    @memprofile
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
        parray = (1.0+0.j)*(self.Gam_slice*parray.T).T*np.sqrt(self.ntimesteps)
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

            cpx.set_real(self.xtemp, self.aaos.get_field_components(i, f_alls=self.xfr.subfunctions))
            cpx.set_imag(self.xtemp, self.aaos.get_field_components(i, f_alls=self.xfi.subfunctions))

            # Do a project for Riesz map, to be superceded
            # when we get Cofunction
            self.Proj.solve(self.Jprob_in, self.xtemp)

            # solve the block system
            self.Jprob_out.assign(0.)
            self.Jsolvers[i].solve()

            # copy the data from solver output
            cpx.get_real(self.Jprob_out, self.aaos.get_field_components(i, f_alls=self.xfr.subfunctions))
            cpx.get_imag(self.Jprob_out, self.aaos.get_field_components(i, f_alls=self.xfi.subfunctions))

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

        self._record_diagnostics()

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError
