import firedrake as fd
from firedrake.petsc import PETSc

from warnings import warn
import numpy as np
from scipy.fft import fft, ifft

from asQ.pencil import Pencil, Subcomm
from asQ.profiling import profiler
from asQ.common import get_option_from_list, get_deprecated_option

from asQ.allatonce.function import time_average as time_average_function
from asQ.preconditioners.base import AllAtOnceBlockPCBase, get_default_options

from functools import partial

__all__ = ['CirculantPC']


class CirculantPC(AllAtOnceBlockPCBase):
    """
    A block alpha-circulant Paradiag preconditioner for the all-at-once system.

    For details see:
    "ParaDiag: parallel-in-time algorithms based on the diagonalization technique"
    Martin J. Gander, Jun Liu, Shu-Lin Wu, Xiaoqiang Yue, Tao Zhou.
    arXiv:2005.09158

    PETSc options:

    'diagfft_linearisation': <'consistent', 'user'>
        Which form to linearise when constructing the block Jacobians.
        Default is 'consistent'.

        'consistent': use the same form used in the AllAtOnceForm residual.
        'user': use the alternative forms given in the appctx.
            If this option is specified then the appctx must contain form_mass
            and form_function entries with keys 'pc_form_mass' and 'pc_form_function'.

    'diagfft_state': <'window', 'slice', 'linear', 'initial', 'reference'>
        Which state to linearise around when constructing the block Jacobians.
        Default is 'window'.

        'window': use the time average over the entire AllAtOnceFunction.
        'slice': use the time average over timesteps on the local Ensemble member.
        'linear': the form linearised is already linear, so no update to the state is needed.
        'initial': the initial condition of the AllAtOnceFunction is used for all timesteps.
        'reference': use the reference state of the AllAtOnceJacobian for all timesteps.

    'diagfft_block_%d': <LinearVariationalSolver options>
        The solver options for the %d'th block, enumerated globally.
        Use 'diagfft_block' to set options for all blocks.
        Default is the Firedrake default options.

    'diagfft_dt': <float>
        The timestep size to use in the preconditioning matrix.
        Defaults to the timestep size used in the AllAtOnceJacobian.

    'diagfft_theta': <float>
        The implicit theta method parameter to use in the preconditioning matrix.
        Defaults to the implicit theta method parameter used in the AllAtOnceJacobian.

    'diagfft_alpha': <float>
        The circulant parameter to use in the preconditioning matrix.
        Defaults to 1e-3.

    'diagfft_complex_proxy': <'vector', 'mixed'>
        Which implementation of the complex-proxy module to use.
        Default is 'vector'.

        'vector': the real-imag components of the complex function space are implemented
            as a 2-component VectorFunctionSpace
        'mixed': the real-imag components of the complex function space are implemented
            as a 2-component MixedFunctionSpace

    If the AllAtOnceSolver's appctx contains a 'block_appctx' dictionary, this is
    added to the appctx of each block solver.  The appctx of each block solver also
    contains the following:
        'd1': circulant eigenvalue of the mass matrix.
        'd2': circulant eigenvalue of the stiffness matrix.
        'cpx': complex-proxy module implementation set with 'diagfft_complex_proxy'.
        'uref': state around which the blocks are linearised.
        'tref': time at which the blocks are linearised.
        'bcs': block boundary conditions.
        'form_mass': function used to build the block mass matrix.
        'form_function': function used to build the block stiffness matrix.
    """
    prefix = "circulant_"
    deprecated_prefix = "diagfft_"
    valid_jacobian_states = tuple(('window', 'slice', 'linear', 'initial', 'reference'))

    default_alpha = 1e-3

    @profiler()
    def initialize(self, pc):
        super().initialize(pc, final_initialize=False)

        # these were setup by super
        prefix = self.full_prefix
        aaofunc = self.aaofunc
        appctx = self.appctx

        # basic model function space
        self.blockV = aaofunc.field_function_space

        # Input/Output wrapper Functions for all-at-once residual being acted on
        self.yf = fd.Function(aaofunc.function_space)  # output

        self.alpha = get_deprecated_option(
            PETSc.Options().getReal, prefix, self.deprecated_prefix,
            "alpha", default=self.default_alpha)

        dt = self.dt
        self.t_average = fd.Constant(self.aaoform.t0 + (self.aaofunc.ntimesteps + 1)*self.dt/2)
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
        # First need to build the complex function space version of blockV
        valid_cpx_type = ['vector', 'mixed']

        cpx_type = get_option_from_list(
            prefix, "complex_proxy", valid_cpx_type,
            default_index=0, deprecated_prefix=self.deprecated_prefix)

        if cpx_type == 'vector':
            import asQ.complex_proxy.vector as cpx
            self.cpx = cpx
        elif cpx_type == 'mixed':
            import asQ.complex_proxy.mixed as cpx
            self.cpx = cpx

        self.CblockV = cpx.FunctionSpace(self.blockV)

        # set the boundary conditions to zero for the residual
        self.block_bcs = tuple((cb
                                for bc in self.aaoform.field_bcs
                                for cb in cpx.DirichletBC(self.CblockV, self.blockV,
                                                          bc, 0*bc.function_arg)))

        # function to do global reduction into for average block jacobian
        if self.jacobian_state in ('window', 'slice'):
            self.ureduce = fd.Function(self.blockV)
            self.uwrk = fd.Function(self.blockV)

        # input and output functions to the block solve
        self.block_sol = fd.Function(self.CblockV)
        self.block_rhs = fd.Cofunction(self.CblockV.dual())
        # input for the cofunc rhs map
        self.xtemp = fd.Function(self.CblockV)

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

        # building the Jacobian of the nonlinear term
        # what we want is a block diagonal matrix in the 2x2 system
        # coupling the real and imaginary parts.
        # We achieve this by copying w_all into both components of u0
        # building the nonlinearity separately for the real and imaginary
        # parts and then linearising.
        # This is constructed by cpx.derivative

        #  Building the nonlinear operator
        self.block_solvers = []

        # which form to linearise around
        form_mass = self.form_mass
        form_function = partial(self.form_function, t=self.t_average)

        # Now need to build the block solver
        self.u0 = fd.Function(self.CblockV)  # time average to linearise around

        # user appctx for the blocks
        block_appctx = appctx.get('block_appctx', {})

        # Block i has prefix 'circulant_block_{i}', but we want to be able
        # to set default options for all blocks using 'circulant_block'.
        # LinearVariationalSolver will prioritise options it thinks are from
        # the command line (including those in the `inserted_options` database
        # of the AllAtOnceSolver) over the ones passed to __init__, so we pull
        # the default options off the global dict and pass these explicitly to LVS.

        default_block_prefix = f"{prefix}block_"
        deprecated_block_prefix = f"{self.deprecated_prefix}block_"
        for k, v in PETSc.Options().getAll().items():
            if k.startswith(f"{deprecated_block_prefix}"):
                msg = "Prefix 'diagfft' is deprecated and will be removed in the future. Use 'circulant' instead."
                warn(msg)
                default_block_prefix = deprecated_block_prefix

        default_block_options = get_default_options(
            default_block_prefix, range(self.ntimesteps))

        # building the block problem solvers
        for i in range(self.nlocal_timesteps):
            ii = aaofunc.transform_index(i, from_range='slice', to_range='window')
            d1 = self.D1[ii]
            d2 = self.D2[ii]

            M, D1r, D1i = cpx.BilinearForm(self.CblockV, d1, form_mass, return_z=True)
            K, D2r, D2i = cpx.derivative(d2, form_function, self.u0, return_z=True)

            A = M + K

            # The rhs
            L = self.block_rhs

            # pass parameters into PC:
            appctx_h = {
                "d1": d1,
                "d2": d2,
                "cpx": cpx,
                "uref": self.u0,
                "tref": self.t_average,
                "bcs": self.block_bcs,
                "form_mass": self.form_mass,
                "form_function": self.form_function,
            }
            appctx_h.update(block_appctx)

            block_problem = fd.LinearVariationalProblem(A, L, self.block_sol,
                                                        bcs=self.block_bcs,
                                                        constant_jacobian=True)

            block_solver = fd.LinearVariationalSolver(
                block_problem, appctx=appctx_h,
                options_prefix=default_block_prefix+str(ii),
                solver_parameters=default_block_options)

            self.block_solvers.append(block_solver)

        self.initialized = True

    @profiler()
    def _record_diagnostics(self):
        """
        Update diagnostic information from block linear solvers.

        Must be called exactly once at the end of each apply().
        """
        for i in range(self.nlocal_timesteps):
            its = self.block_solvers[i].snes.getLinearSolveIterations()
            self.block_iterations.dlocal[i] += its

    @profiler()
    def update(self, pc):
        '''
        we need to update u0 according to the diagfft_state option.
        we copy the state into both the "real" and "imaginary" parts
        of u0. this is so that when we linearise the nonlinearity,
        we get an operator that is block diagonal in the 2x2 system
        coupling real and imaginary parts.
        '''
        cpx = self.cpx

        # default to time at centre of window
        self.t_average.assign(self.aaoform.t0 + self.dt*(self.ntimesteps + 1)/2)

        jacobian_state = self.jacobian_state
        if jacobian_state == 'linear':
            return

        elif jacobian_state == 'initial':
            ustate = self.aaofunc.initial_condition
            self.t_average.assign(self.aaoform.t0)

        elif jacobian_state == 'reference':
            ustate = self.jacobian.reference_state

        elif jacobian_state in ('window', 'slice'):
            time_average_function(self.aaofunc, self.ureduce,
                                  self.uwrk, average=jacobian_state)
            ustate = self.ureduce

            if jacobian_state == 'slice':
                i1 = self.aaofunc.transform_index(0, from_range='slice',
                                                  to_range='window')
                t1 = self.aaoform.t0 + i1*self.dt
                self.t_average.assign(t1 + self.dt*(self.nlocal_timesteps + 1)/2)

        cpx.set_real(self.u0, ustate)
        cpx.set_imag(self.u0, ustate)

        for block in self.block_solvers:
            block.invalidate_jacobian()
        return

    @profiler()
    def apply_impl(self, pc, x, y):
        cpx = self.cpx

        # get array of basis coefficients
        with x.global_vec_ro() as xvec:
            parray = xvec.array_r.reshape((self.nlocal_timesteps,
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
        with PETSc.Log.Event("asQ.diag_preconditioner.CirculantPC.apply.transfer"):
            self.transfer.forward(self.a0, self.a1)

        # FFT
        with PETSc.Log.Event("asQ.diag_preconditioner.CirculantPC.apply.fft"):
            self.a1[:] = fft(self.a1, axis=0)

        # transfer backward
        with PETSc.Log.Event("asQ.diag_preconditioner.CirculantPC.apply.transfer"):
            self.transfer.backward(self.a1, self.a0)

        # Copy into xfi, xfr
        parray[:] = self.a0[:]
        with self.xfr.function.dat.vec_wo as v:
            v.array[:] = parray.real.reshape(-1)
        with self.xfi.function.dat.vec_wo as v:
            v.array[:] = parray.imag.reshape(-1)
        #####################

        # Do the block solves

        with PETSc.Log.Event("asQ.diag_preconditioner.CirculantPC.apply.block_solves"):
            for i in range(self.nlocal_timesteps):
                # copy the data into solver input
                cpx.set_real(self.xtemp, self.xfr[i])
                cpx.set_imag(self.xtemp, self.xfi[i])

                for cdat, xdat in zip(self.block_rhs.dat, self.xtemp.dat):
                    cdat.data[:] = xdat.data[:]

                # solve the block system
                self.block_sol.zero()
                self.block_solvers[i].solve()

                # copy the data from solver output
                cpx.get_real(self.block_sol, self.xfr[i])
                cpx.get_imag(self.block_sol, self.xfi[i])

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
        with PETSc.Log.Event("asQ.diag_preconditioner.CirculantPC.apply.transfer"):
            self.transfer.forward(self.a0, self.a1)

        # IFFT
        with PETSc.Log.Event("asQ.diag_preconditioner.CirculantPC.apply.fft"):
            self.a1[:] = ifft(self.a1, axis=0)

        # transfer backward
        with PETSc.Log.Event("asQ.diag_preconditioner.CirculantPC.apply.transfer"):
            self.transfer.backward(self.a1, self.a0)
        parray[:] = self.a0[:]

        # scale
        parray = ((1.0/self.Gam_slice)*parray.T).T
        # Copy into xfi, xfr

        with y.global_vec_wo() as yvec:
            yvec.array[:] = parray.reshape(-1).real
        ################
