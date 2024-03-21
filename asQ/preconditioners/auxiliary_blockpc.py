import firedrake as fd
from firedrake.petsc import PETSc

from functools import partial

__all__ = ['AuxiliaryRealBlockPC', 'AuxiliaryComplexBlockPC']


class AuxiliaryBlockPCBase(fd.AuxiliaryOperatorPC):
    def _setup(self, pc, v, u):
        appctx = self.get_appctx(pc)
        self.appctx = appctx

        self.prefix = pc.getOptionsPrefix() + self.prefix
        self.options = PETSc.Options(self.prefix)

        self.uref = appctx.get('uref')
        assert self.uref.function_space() == u.function_space()

        self.bcs = appctx['bcs']
        self.tref = appctx['tref']

        form_mass = appctx['form_mass']
        form_function = appctx['form_function']

        self.form_mass = appctx.get('aux_form_mass', form_mass)
        self.form_function = appctx.get('aux_form_function', form_function)


class AuxiliaryRealBlockPC(AuxiliaryBlockPCBase):
    """
    A preconditioner for the serial-in-time problem that builds a PC using a specified form.

    This preconditioner is analogous to firedrake.AuxiliaryOperatorPC. Given
    `form_mass` and `form_function` functions (with the usual call-signatures),
    it constructs an AuxiliaryOperatorPC.

    By default, the timestep, theta parameter, and the form_mass and form_function
    of the original problem are used (i.e. exactly the same operator as the Jacobian).

    User-defined `form_mass` and `form_function` functions and dt and theta can be
    passed through the appctx using the following keys:
        'aux_form_mass': function used to build the mass matrix.
        'aux_form_function': function used to build the stiffness matrix.
        'aux_dt': timestep size.
        'aux_theta': implicit theta parameter.
    """
    def form(self, pc, v, u):
        self._setup(pc, v, u)

        dt = self.appctx['dt']
        theta = self.appctx['theta']

        dt = self.options.getReal('dt', default=dt)
        theta = self.options.getReal('theta', default=theta)

        us = fd.split(self.uref)
        vs = fd.split(v)

        M = self.form_mass(*us, *vs)
        K = self.form_function(*us, *vs, self.tref)

        dt1 = fd.Constant(1/dt)
        thet = fd.Constant(theta)

        F = dt1*M + thet*K
        a = fd.derivative(F, self.uref)

        return (a, self.bcs)


class AuxiliaryComplexBlockPC(fd.AuxiliaryOperatorPC):
    """
    A preconditioner for the complex blocks that builds a PC using a specified form.

    This preconditioner is analogous to firedrake.AuxiliaryOperatorPC. Given
    `form_mass` and `form_function` functions on the real function space (with the
    usual call-signatures), it constructs an AuxiliaryOperatorPC on the complex
    block function space.

    By default, the circulant eigenvalues and the form_mass and form_function of the
    circulant preconditioner are used (i.e. exactly the same operator as the block).

    User-defined `form_mass` and `form_function` functions and complex coeffiecients
    can be passed through the block_appctx using the following keys:
        'aux_form_mass': function used to build the mass matrix.
        'aux_form_function': function used to build the stiffness matrix.
        'aux_%d_d1': complex coefficient on the mass matrix of the %d'th block.
        'aux_%d_d2': complex coefficient on the stiffness matrix of the %d'th block.
    """
    def form(self, pc, v, u):
        self._setup(pc, v, u)

        cpx = self.appctx['cpx']

        d1 = self.appctx['d1']
        d2 = self.appctx['d2']

        # PETScScalar is real so we can't get a complex number directly from options
        d1r = self.options.getReal('d1r', default=d1.real)
        d1i = self.options.getReal('d1i', default=d1.imag)

        d2r = self.options.getReal('d2r', default=d1.real)
        d2i = self.options.getReal('d2i', default=d1.imag)

        d1 = complex(d1r, d1i)
        d2 = complex(d2r, d2i)

        # complex and real valued function spaces
        W = v.function_space()
        V = W.sub(0)

        bcs = tuple((cb
                     for bc in self.bcs
                     for cb in cpx.DirichletBC(W, V, bc, 0*bc.function_arg)))

        M = cpx.BilinearForm(W, d1, self.form_mass)
        K = cpx.derivative(d2, partial(self.form_function, t=self.tref), self.uref)

        a = M + K

        return (a, bcs)
