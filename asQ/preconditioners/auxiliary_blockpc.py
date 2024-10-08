import firedrake as fd
from firedrake.petsc import PETSc

from functools import partial

__all__ = ['AuxiliaryRealBlockPC', 'AuxiliaryComplexBlockPC']


class AuxiliaryBlockPCBase(fd.AuxiliaryOperatorPC):
    def _setup(self, pc, v, u):
        appctx = self.get_appctx(pc)
        self.appctx = appctx

        self.prefix = pc.getOptionsPrefix() + self._prefix
        self.options = PETSc.Options(self.prefix)

        uref = appctx.get('uref')
        self.uref = appctx.get('aux_uref', uref)
        self.u = fd.Function(u.function_space())

        self.frozen = self.options.getBool("frozen", default=False)

        self.bcs = appctx['bcs']
        self.tref = appctx['tref']

        form_mass = appctx['form_mass']
        form_function = appctx['form_function']

        self.form_mass = appctx.get('aux_form_mass', form_mass)
        self.form_function = appctx.get('aux_form_function', form_function)

    def update(self, pc):
        if not self.frozen:
            self.update_state(pc)
            super().update(pc)
        return


class AuxiliaryRealBlockPC(AuxiliaryBlockPCBase):
    """
    A preconditioner for the serial-in-time problem that builds a PC using a specified form.

    This preconditioner is analogous to firedrake.AuxiliaryOperatorPC. Given
    `form_mass` and `form_function` functions (with the usual call-signatures),
    it constructs an AuxiliaryOperatorPC.

    Required appctx entries (usually filled by the all-at-once preconditioner):

    'uref': Firedrake Function around which to linearise the form_function.
    'tref': The time at which to linearise the form_function.
    'bcs': A list of the boundary conditions.
    'form_mass': The function to generate the mass matrix.
    'form_function': The function to generate the stiffness matrix.
    'dt': The timestep size.
    'theta': The implicit parameter.

    Optional appctx entries. Used instead of the required entries if present.

    'aux_form_mass': Alternative function used to generate the mass matrix.
    'aux_form_function': Alternative function used to generate the stiffness matrix.

    PETSc options. Used instead of the appctx entries if present.

    'aux_dt': <float>
        Alternative timestep size.
    'aux_theta': <float>
        Alternative implicit theta parameter.
    """
    def update_state(self, pc):
        self.u.assign(self.uref)

    def form(self, pc, v, u):
        self._setup(pc, v, u)
        self.update_state(pc)

        dt = self.appctx['dt']
        theta = self.appctx['theta']

        dt = self.options.getReal('dt', default=dt)
        theta = self.options.getReal('theta', default=theta)

        us = fd.split(self.u)
        vs = fd.split(v)

        dt1 = fd.Constant(1/dt)
        thet = fd.Constant(theta)

        M = self.form_mass(*fd.split(u), *vs)

        F = self.form_function(*us, *vs, self.tref)
        K = fd.derivative(F, self.u)

        a = dt1*M + thet*K

        return (a, self.bcs)


class AuxiliaryComplexBlockPC(AuxiliaryBlockPCBase):
    """
    A preconditioner for the complex blocks that builds a PC using a specified form.

    This preconditioner is analogous to firedrake.AuxiliaryOperatorPC. Given
    `form_mass` and `form_function` functions on the real function space (with the
    usual call-signatures), it constructs an AuxiliaryOperatorPC on the complex
    block function space.

    Required appctx entries (usually filled by the all-at-once preconditioner).

    'uref': Firedrake Function around which to linearise the form_function.
    'tref': The time at which to linearise the form_function.
    'bcs': A list of the boundary conditions on the real space.
    'form_mass': The function to generate the mass matrix.
    'form_function': The function to generate the stiffness matrix.
    'd1': The complex coefficient on the mass matrix.
    'd2': The complex coefficient on the stiffness matrix.
    'cpx': The complex_proxy submodule to generate the complex-valued forms.

    Optional appctx entries. Used instead of the required entries if present.

    'aux_form_mass': Alternative function used to generate the mass matrix.
    'aux_form_function': Alternative function used to generate the stiffness matrix.

    PETSc options. Used instead of the appctx entries if present.

    'aux_d1r': <float>
        Real part of an alternative complex coefficient on the mass matrix.
    'aux_d1i': <float>
        Imaginary part of an alternative complex coefficient on the mass matrix.
    'aux_d2r': <float>
        Real part of an alternative complex coefficient on the stiffness matrix.
    'aux_d2i': <float>
        Imaginary part of an alternative complex coefficient on the stiffness matrix.
    """
    def update_state(self, pc):
        if self.u.function_space() == self.uref.function_space():
            self.u.assign(self.uref)
        else:
            self.cpx.set_real(self.u, self.uref)
            self.cpx.set_imag(self.u, self.uref)

    def form(self, pc, v, u):
        self._setup(pc, v, u)
        cpx = self.appctx['cpx']
        self.cpx = cpx
        self.update_state(pc)

        d1 = self.appctx['d1']
        d2 = self.appctx['d2']

        # PETScScalar is real so we can't get a complex number directly from options
        d1r = self.options.getReal('d1r', default=d1.real)
        d1i = self.options.getReal('d1i', default=d1.imag)

        d2r = self.options.getReal('d2r', default=d2.real)
        d2i = self.options.getReal('d2i', default=d2.imag)

        d1 = complex(d1r, d1i)
        d2 = complex(d2r, d2i)

        # complex and real valued function spaces
        W = v.function_space()

        M = cpx.BilinearForm(W, d1, self.form_mass)
        K = cpx.derivative(d2, partial(self.form_function, t=self.tref), self.u)

        a = M + K

        return (a, self.bcs)
