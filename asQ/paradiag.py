import numpy as np
import firedrake as fd
from scipy.fft import fft, ifft

print('reload paradiag')
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

        # w_all contains all functions, used for nonlinear update!
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
        self.xf = fd.Function(W) #input
        self.yf = fd.Function(W) #output

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
        ## First need to build the vector function space version of
        ## blockV
        mesh = self.blockV.mesh()
        Ve = self.blockV.ufl_element()
        if isinstance(Ve, fd.MixedElement):
            MixedCpts = []
            self.ncpts=Ve.num_sub_elements()
            for cpt in range(Ve.num_sub_elements()):
                SubV = Ve.sub_elements()[cpt]
                if isinstance(SubV, fd.FiniteElement):
                    MixedCpts.append(fd.VectorElement(SubV, dim=2))
                elif isinstance(SubV, fd.VectorElement):
                    shape = (2,SubV.num_sub_elements())
                    MixedCpts.append(fd.TensorElement(SubV, shape))
                elif isinstance(SubV, fd.TensorElement):
                    shape = (2,) + SubV._shape
                    MixedCpts.append(fd.TensorElement(SubV, shape))
                else:
                    raise(NotImplementedError)

            dim = len(MixedCpts)
            self.CblockV = np.prod([fd.FunctionSpace(mesh,
                                    MixedCpts[i]) for i in range(dim)])
        else:
            self.ncpts = 1
            if isinstance(Ve, fd.FiniteElement):
                self.CblockV = fd.FunctionSpace(mesh,
                                                fd.VectorElement(Ve, dim=2))
            elif isinstance(Ve, fd.VectorElement):
                shape = (2,Ve.num_sub_elements())
                self.CblockV = fd.FunctionSpace(mesh,
                                                fd.TensorElement(Ve, shape))
            elif isinstance(Ve, fd.TensorElement):
                shape = (2,) + SubV._shape
                self.CblockV = fd.FunctionSpace(mesh,
                                                fd.TensorElement(Ve, shape))
            else:
                raise(NotImplementedError)


        ##Now need to build the block solver
        vs = fd.TestFunctions(self.CblockV)
        self.u0 = fd.Function(self.CblockV) #we will create a linearisation
        us = fd.split(self.u0)

        ##extract the real and imaginary parts
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
                    vsr.append(vs[i][0,:])
                    vsi.append(vs[i][1,:])
                    usr.append(us[i][0,:])
                    usi.append(us[i][1,:])
                elif len(SubV.value_shape()) == 2:
                    vsr.append(vs[i][0,:,:])
                    vsi.append(vs[i][1,:,:])
                    usr.append(us[i][0,:,:])
                    usi.append(us[i][1,:,:])
                else:
                    raise(NotImplementedError)
        else:
            if isinstance(Ve, fd.FiniteElement):
                vsr.append(vs[0])
                vsi.append(vs[1])
                usr.append(us[0])
                usi.append(us[1])
            elif isinstance(Ve, fd.VectorElement):
                vsr.append(vs[0,:])
                vsi.append(vs[1,:])
                usr.append(self.u0[0,:])
                usi.append(self.u0[1,:])
            elif isinstance(Ve, fd.TensorElement):
                vsr.append(vs[0,:])
                vsi.append(vs[1,:])
                usr.append(self.u0[0,:])
                usi.append(self.u0[1,:])
            else:
                raise(NotImplementedError)

                   
        ## input and output functions
        self.Jprob_in = fd.Function(self.CblockV)
        self.Jprob_out = fd.Function(self.CblockV)
        
        #A place to store all the inputs to the block problems
        self.xfi = fd.Function(W)
        self.xfr = fd.Function(W)
            
        ## Building the nonlinear operator
        self.Jsolvers = []
        self.Js = []
        form_mass = appctx.get("form_mass", None)
        form_function = appctx.get("form_function", None)
        for i in range(M):
            D1i = fd.Constant(np.imag(self.D1[i]))
            D1r = fd.Constant(np.real(self.D1[i]))
            D2i = fd.Constant(np.imag(self.D2[i]))
            D2r = fd.Constant(np.real(self.D2[i]))

            L = (
                D1r*form_mass(*usr, *vsr)
                - D1i*form_mass(*usi, *vsr)
                + D2r*form_function(*usr, *vsr)
                - D2i*form_function(*usi, *vsr)
                + D1r*form_mass(*usi, *vsi)
                + D1i*form_mass(*usr, *vsi)
                + D2r*form_function(*usi, *vsi)
                + D2i*form_function(*usr, *vsi)
            )

            ## The linear operator
            J = fd.derivative(L, self.u0)
            Jsolver = fd.LinearSolver(fd.assemble(J),
                                                 options_prefix=prefix)
            self.Js.append(J)
            self.Jsolvers.append(Jsolver)
        
    def update(self, pc):
        self.u0.assign(0)
        for i in range(self.M):
            #copy the data into solver input
            if self.ncpts > 1:
                u0s = self.u0.split()
                for cpt in range(self.ncpts):
                    u0s[cpt].sub(0).assign(
                        u0s[cpt].sub(0) + self.w_all.split()[self.ncpts*i+cpt])
            else:
                self.u0.sub(0).assign(self.u0.sub(0) + self.w_all.split()[i])

            solver =  self.Jsolvers[i]
            fd.assemble(self.Js[i],tensor=solver.A)
        self.u0 /= self.M



    def apply(self, pc, x, y):

        #copy petsc vec into Function
        with self.xf.dat.vec_wo as v:
            x.copy(v)

        #get array of basis coefficients
        with self.xf.dat.vec_ro as v:
            parray = v.array.reshape((self.M, self.blockV.dim()))
        #This produces an array whose rows are time slices
        #and columns are finite element basis coefficients
            
        # Diagonalise
        Nt = self.M
        parray = fft((self.Gam*parray.T).T, axis=0)*np.sqrt(Nt)

        # Copy into xfi, xfr
        with self.xfr.dat.vec_wo as v:
            v.array[:] = parray.real.reshape((self.NM,))

        with self.xfi.dat.vec_wo as v:
            v.array[:] = parray.imag.reshape((self.NM,))

        # Do the block solves

        # print("Warning - for nonlinear we need to update u0")
        #The above will be done via self.get_appctx(pc)["state"]

        for i in range(self.M):
            #copy the data into solver input
            if self.ncpts > 1:
                Jins = self.Jprob_in.split()
                for cpt in range(self.ncpts):
                    Jins[cpt].sub(0).assign(
                        self.xfr.split()[self.ncpts*i+cpt])
                    Jins[cpt].sub(1).assign(
                        self.xfi.split()[self.ncpts*i+cpt])
            else:
                self.Jprob_in.sub(0).assign(self.xfr.split()[i])
                self.Jprob_in.sub(1).assign(self.xfi.split()[i])

            #solve the block system
            self.Jsolvers[i].solve(self.Jprob_out, self.Jprob_in)

            #copy the data from solver output
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
        #get array of basis coefficients
        with self.xfi.dat.vec_ro as v:
            parray = 1j*v.array.reshape((self.M, self.blockV.dim()))
        with self.xfr.dat.vec_ro as v:
            parray += v.array.reshape((self.M, self.blockV.dim()))
        
        parray = ((1.0/self.Gam)*ifft(parray, axis=0).T).T

        #get array of basis coefficients
        with self.yf.dat.vec_wo as v:
            v.array[:] = parray.reshape((self.M*self.blockV.dim(),)).real

        with self.yf.dat.vec_ro as v:
            v.copy(y)

    def applyTranspose(self, pc, x, y):
        RaiseNotImplementedError


class paradiag(object):
    def __init__(self, form_function, form_mass, W, w0, dt, theta,
                 alpha, M, solver_parameters=None,
                 circ="outside",
                 jac_average="newton", tol=1.0e-6, maxits=10):
        """A class to implement paradiag timestepping.
        
        :arg form_function: a function that returns a linear form
        on W providing f(w) for the ODE w_t + f(w) = 0.
        :arg form_mass: a function that returns a linear form
        on W providing the mass operator for the time derivative.
        :arg W: the FunctionSpace on which f is defined.
        :arg w0: a Function from W containing the initial data
        :arg dt: float, the timestep size.
        :arg theta: float, implicit timestepping parameter
        :arg alpha: float, circulant matrix parameter
        :arg M: integer, the number of timesteps
        :arg solver_parameters: options dictionary for nonlinear solver
        :arg circ: a string describing the option on where to use the
        alpha-circulant modification. "outside" - do a nonlinear wave
        form relaxation method. "jacobian" - do a modified Newton
        method with alpha-circulant modification added to the
        Jacobian. "preconditioner" - only make the alpha-circulant
        modification in the preconditioner.
        :arg jac_average: a string describing the option for when to
        average the jacobian. "newton" - make a quasi-Newton method by
        time averaging the Jacobian. "preconditioner" - only do the
        average in the preconditioner.
        :arg tol: float, the tolerance for the relaxation method (if used)
        :arg maxits: integer, the maximum number of iterations for the 
        relaxation method, if used.
        """

        self.form_function = form_function
        self.form_mass = form_mass
        self.W = W
        self.ncpts = len(W.split()) #count how many components in
                                    #mixed space W
        self.w0 = w0
        self.dt = dt
        self.M = M
        self.theta = theta
        self.alpha = alpha
        self.tol = tol
        self.maxits = maxits
        self.circ = circ
        self.jac_average = jac_average
        
        #function space for the all-at-once system
        #implemented as a massive mixed function space
        self.W_all = np.prod([self.W for i in range(M)])
        
        #function containing the all-at-once solution
        self.w_all = fd.Function(self.W_all)
        w_alls = self.w_all.split()
        #initialise it from the initial condition
        for i in range(M):
            for k in range(self.ncpts):
                w_alls[self.ncpts*i+k].assign(self.w0.sub(k))

        #function containing the last timestep
        #from the previous iteration
        if self.circ == "outside":
            self.w_prev = fd.Function(self.W)
            
        self._set_para_form()

        #passing stuff to the diag preconditioner using the appctx
        def getW():
            return W
        ctx = {"get_blockV":getW,
               "alpha":self.alpha,
               "theta":self.theta,
               "dt":self.dt,
               "form_mass":self.form_mass,
               "form_function":self.form_function,
               "w_all":self.w_all}
        vproblem = fd.NonlinearVariationalProblem(self.para_form, self.w_all)
        self.vsolver = fd.NonlinearVariationalSolver(vproblem,
                                                     solver_parameters
                                                     =
                                                     solver_parameters,
                                                     appctx = ctx)


    def _set_para_form(self):
        """
        Constructs the bilinear form for the all at once system.
        Specific to the theta-centred Crank-Nicholson method
        """

        M = self.M
        w_all_cpts = fd.split(self.w_all)
        
        test_fns = fd.TestFunctions(self.W_all)
        
        dt = fd.Constant(self.dt)
        theta = fd.Constant(self.theta)
        alpha = fd.Constant(self.alpha)
        wMs = w_all_cpts[self.ncpts*(M-1):]
        if self.circ == "outside":
            if self.ncpts == 1:
                wMkm1s = [self.w_prev]
            else:
                wMkm1s = fd.split(self.w_prev)

        for n in range(M):
            #previous time level
            if n==0:
                w0ss = fd.split(self.w0)
                if self.circ == "outside":
                    w0s = [w0ss[i] + alpha*(wMs[i] - wMkm1s[i]) for i in range(self.ncpts)]
                else:
                    w0s = w0ss
            else:
                w0s = w_all_cpts[self.ncpts*(n-1):self.ncpts*n]
            #current time level
            w1s = w_all_cpts[self.ncpts*n:self.ncpts*(n+1)]
            dws = test_fns[self.ncpts*n:self.ncpts*(n+1)]
            #time derivative
            if n == 0:
                p_form = (1.0/dt)*self.form_mass(*w1s, *dws)
            else:
                p_form += (1.0/dt)*self.form_mass(*w1s, *dws)
            p_form -= (1.0/dt)*self.form_mass(*w0s, *dws)
            #vector field
            p_form += theta*self.form_function(*w1s, *dws)
            p_form += (1-theta)*self.form_function(*w0s, *dws)
        self.para_form = p_form


    def solve(self, verbose=False):
        """
        Solve the system (either in one shot or as a relaxation method).
        """
        M = self.M
        if self.circ == "outside":
            #Relaxation method
            wMs = fd.split(self.w_all)[self.ncpts*(M-1):]
            residual = 2*self.tol
            its = 0
            while residual > self.tol and its < self.maxits:
                its += 1
                self.vsolver.solve()

                #compute the residual
                err = [wMs[i] - self.w_prev.sub(i)
                       for i in range(self.ncpts)]
                residual = fd.assemble(self.form_mass(*err, *err))**0.5
                if verbose:
                    print(its, residual)
                
                #copy the last time slice into w_prev
                wMs = self.w_all.split()[self.ncpts*(M-1):]
                for i in range(self.ncpts):
                    self.w_prev.sub(i).assign(wMs[i])
            if its == self.maxits:
                print("Exited due to maxits.", its)
        else:
            #One shot
            self.vsolver.solve()

#with mixed_function.dat.vec_ro as v:
#   process_local_array = v.array # (or v.array_r if you only need readonly)
