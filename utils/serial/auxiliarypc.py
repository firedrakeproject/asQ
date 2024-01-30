import firedrake as fd

__all__ = ['AuxiliarySerialPC']


class AuxiliarySerialPC(fd.AuxiliaryOperatorPC):
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
        appctx = self.get_appctx(pc)

        w1 = appctx.get('w1')
        assert w1.function_space() == v.function_space()

        bcs = appctx['bcs']
        t1 = appctx['t1']

        dt = appctx['dt']
        theta = appctx['theta']

        aux_dt = appctx.get('aux_dt', dt)
        aux_theta = appctx.get('aux_theta', theta)

        form_mass = appctx['form_mass']
        form_function = appctx['form_function']

        aux_form_mass = appctx.get('aux_form_mass', form_mass)
        aux_form_function = appctx.get('aux_form_function', form_function)

        us = fd.split(w1)
        vs = fd.split(v)

        M = aux_form_mass(*us, *vs)
        K = aux_form_function(*us, *vs, t1)

        dt1 = fd.Constant(1/aux_dt)
        thet = fd.Constant(aux_theta)

        F = dt1*M + thet*K
        a = fd.derivative(F, w1)
        return (a, bcs)
