
import firedrake as fd
from asQ.profiling import profiler


def convective_cfl_calculator(mesh,
                              name="CFL"):
    '''
    Return a function that calculates the cfl field from a velocity field and a timestep

    :arg mesh: the mesh over which the velocity field is defined
    :arg name: name of the CFL Function
    '''
    DG0 = fd.FunctionSpace(mesh, "DG", 0)
    v = fd.TestFunction(DG0)

    cell_volume = fd.Function(DG0, name="Cell volume")
    cell_flux = fd.Function(DG0, name="Cell surface flux")
    cfl = fd.Function(DG0, name=name)

    # mesh volume
    One = fd.Function(DG0, name="One").assign(1)
    fd.assemble(One*v*fd.dx, tensor=cell_volume)

    # choose correct facet integral for mesh type
    if mesh.extruded:
        dS = fd.dS_v + fd.dS_h
        ds = fd.ds_v + fd.ds_t + fd.ds_b
    else:
        dS = fd.dS
        ds = fd.ds

    def both(u):
        return 2*fd.avg(u)

    @profiler()
    def cfl_calc(u, dt):
        # area weighted convective flux
        n = fd.FacetNormal(u.function_space().mesh())
        un = 0.5*(fd.inner(-u, n) + abs(fd.inner(-u, n)))
        cell_flux_form = (
            both(un*v)*dS
            + un*v*ds
        )
        fd.assemble(cell_flux_form, tensor=cell_flux)

        dT = fd.Constant(dt)
        cfl.interpolate(dT*cell_flux/cell_volume)
        return cfl

    return cfl_calc


@profiler()
def convective_cfl(u, dt):
    '''
    Return the convective CFL number for the velocity field u with timestep dt
    :arg u: the velocity Function
    :arg dt: the timestep
    '''
    return convective_cfl_calculator(u.function_space().mesh())(u, dt)


def potential_vorticity_calculator(
        velocity_function_space,
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

    mesh = velocity_function_space.mesh()
    outward_normals = fd.CellNormal(mesh)

    def prp(v):
        return perp(outward_normals, v)

    V = fd.FunctionSpace(mesh, element, degree)

    q = fd.TrialFunction(V)
    p = fd.TestFunction(V)

    vel = fd.Function(velocity_function_space)
    pv = fd.Function(V, name=name)

    pv_eqn = q*p*fd.dx + fd.inner(prp(fd.grad(p)), vel)*fd.dx

    pv_prob = fd.LinearVariationalProblem(fd.lhs(pv_eqn),
                                          fd.rhs(pv_eqn),
                                          pv)

    params = {'ksp_type': 'cg'}
    pv_solver = fd.LinearVariationalSolver(pv_prob,
                                           solver_parameters=params)

    @profiler()
    def pv_calc(u):
        vel.assign(u)
        pv_solver.solve()
        return pv

    return pv_calc
