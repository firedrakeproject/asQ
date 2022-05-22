
import firedrake as fd


def convective_cfl_calculator(mesh):
    '''
    Return a function that, when passed a velocity field and a timestep, will return a function that is the cfl number.
    '''
    DG0 = fd.FunctionSpace(mesh, "DG", 0)
    v = fd.TestFunction(DG0)

    cfl_denominator = fd.Function(DG0, name="CFL denominator")
    cfl_numerator = fd.Function(DG0, name="CFL numerator")
    cfl = fd.Function(DG0, name="cfl")

    # mesh volume
    One = fd.Function(DG0).assign(1.0)
    fd.assemble(One*v*fd.dx, tensor=cfl_denominator)

    def cfl_calc(u, dt):
        # area weighted convective flux
        n = fd.FacetNormal(u.function_space().mesh())
        un = 0.5*(fd.inner(-u, n) + abs(fd.inner(-u, n)))
        cfl_numerator_form = (
            2*fd.avg(un*v)*fd.dS
            + un*v*fd.ds
        )
        fd.assemble(cfl_numerator_form, tensor=cfl_numerator)

        dT = fd.Constant(dt)
        cfl.assign(dT*cfl_numerator/cfl_denominator)
        return cfl

    return cfl_calc


def convective_cfl(u, dt):
    '''
    Return a function that, when passed a velocity field and a timestep, will return a function that is the cfl number.
    '''
    return convective_cfl_calculator(u.function_space().mesh())(u, dt)
