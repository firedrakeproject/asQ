
import firedrake as fd


def convective_cfl_calculator(mesh,
                              name="CFL"):
    '''
    Return a function that calculates the cfl field from a velocity field and a timestep

    :arg mesh: the mesh over which the velocity field is defined
    :arg name: name of the CFL Function
    '''
    DG0 = fd.FunctionSpace(mesh, "DG", 0)
    v = fd.TestFunction(DG0)

    cfl_denominator = fd.Function(DG0, name="CFL denominator")
    cfl_numerator = fd.Function(DG0, name="CFL numerator")
    cfl = fd.Function(DG0, name=name)

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


def potential_vorticity(velocity_function_space,
                        element="CG",
                        degree=3,
                        name="Relative vorticity",
                        perp=fd.cross):
    '''
    Return a function that calculates the potential vorticity field of a velocity field

    :arg velocity_function_space: the FunctionSpace that the velocity field lives in
    :arg element: the element type of the potential vorticity FunctionSpace
    :arg degree: the degree of the potential vorticity FunctionSpace
    :arg name: name of the potential vorticity Function
    :arg perp: the perpendicular operation required by the potential vorticity calculation
    '''

    mesh = velocity_function_space().mesh
    outward_normals = fd.CellNormal(mesh)

    def prp(v):
        return perp(outward_normals, v)

    V = fd.FunctionSpace(mesh, element, degree)

    q = fd.TrialFunction(V)
    p = fd.TestFunction(V)

    vel = fd.Function(velocity_function_space)
    pv = fd.Function(V, name=name)

    pv_eqn = q*p*fd.dx + fd.inner(prp(fd.grap(p)), vel)*fd.dx

    pv_prob = fd.LinearVariationalProblem(fd.lhs(pv_eqn),
                                          fd.rhs(pv_eqn),
                                          pv)

    params = {'ksp_type': 'cg'}
    pv_solver = fd.LinearVariationalSolver(pv_prob, params)

    def pv_calc(u):
        vel.assign(u)
        pv_solver.solve()
        return pv

    return pv_calc
