import firedrake as fd


def function_space(mesh, horizontal_degree=1, vertical_degree=1,
                   vertical_velocity_space=False):
    if not mesh.extruded:
        msg = "Compressible Euler function space only defined for extruded meshes"
        raise ValueError(msg)

    dim = sum(mesh.cell_dimension())

    # horizontal base spaces
    if dim == 2:
        S1 = fd.FiniteElement("CG", fd.interval, horizontal_degree+1)
        S2 = fd.FiniteElement("DG", fd.interval, horizontal_degree)
    elif dim == 3:
        S1 = fd.FiniteElement("RTCF", fd.quadrilateral, horizontal_degree+1)
        S2 = fd.FiniteElement("DQ", fd.quadrilateral, horizontal_degree)
    else:
        msg = "Compressible Euler function space only defined for 2 or 3 dimensions"
        raise ValueError(msg)

    # vertical base spaces
    T0 = fd.FiniteElement("CG", fd.interval, vertical_degree+1)
    T1 = fd.FiniteElement("DG", fd.interval, vertical_degree)

    # build spaces V2, V3, Vt
    V2h_elt = fd.HDiv(fd.TensorProductElement(S1, T1))
    V2t_elt = fd.TensorProductElement(S2, T0)
    V3_elt = fd.TensorProductElement(S2, T1)
    V2v_elt = fd.HDiv(V2t_elt)
    V2_elt = V2h_elt + V2v_elt

    V1 = fd.FunctionSpace(mesh, V2_elt, name="Velocity")
    V2 = fd.FunctionSpace(mesh, V3_elt, name="Pressure")
    Vt = fd.FunctionSpace(mesh, V2t_elt, name="Temperature")

    W = V1 * V2 * Vt  # velocity, density, temperature

    if vertical_velocity_space:
        Vv = fd.FunctionSpace(mesh, V2v_elt, name="Vv")
        return W, Vv
    else:
        return W


def velocity_function_space(mesh, horizontal_degree=1, vertical_degree=1):
    return function_space(mesh, horizontal_degree, vertical_degree).subfunctions[0]


def pressure_function_space(mesh, horizontal_degree=1, vertical_degree=1):
    return function_space(mesh, horizontal_degree, vertical_degree).subfunctions[1]


def density_function_space(mesh, horizontal_degree=1, vertical_degree=1):
    return function_space(mesh, horizontal_degree, vertical_degree).subfunctions[1]


def temperature_function_space(mesh, horizontal_degree=1, vertical_degree=1):
    return function_space(mesh, horizontal_degree, vertical_degree).subfunctions[2]


def hydrostatic_rho(Vv, V2, mesh, thetan, rhon, pi_boundary,
                    gas, Up, top=False, Pi=None, verbose=0):
    from utils.compressible_flow.gas import pi_formula, rho_formula
    # Calculate hydrostatic Pi, rho
    W_h = Vv * V2
    wh = fd.Function(W_h)
    n = fd.FacetNormal(mesh)
    dv, drho = fd.TestFunctions(W_h)

    v, Pi0 = fd.TrialFunctions(W_h)

    Pieqn = (
        gas.cp*(fd.inner(v, dv) - fd.div(dv*thetan)*Pi0)*fd.dx
        + drho*fd.div(thetan*v)*fd.dx
    )

    if top:
        bmeasure = fd.ds_t
        bstring = "bottom"
    else:
        bmeasure = fd.ds_b
        bstring = "top"

    zeros = []
    for i in range(Up.ufl_shape[0]):
        zeros.append(fd.Constant(0.))

    L = -gas.cp*fd.inner(dv, n)*thetan*pi_boundary*bmeasure
    L -= gas.g*fd.inner(dv, Up)*fd.dx
    bcs = [fd.DirichletBC(W_h.sub(0), zeros, bstring)]

    PiProblem = fd.LinearVariationalProblem(Pieqn, L, wh, bcs=bcs)

    lu_params = {
        'snes_stol': 1e-10,
        'mat_type': 'aij',
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        "pc_factor_mat_ordering_type": "rcm",
        "pc_factor_mat_solver_type": "mumps",
    }

    verbose_options = {
        'snes_monitor': None,
        'snes_converged_reason': None,
        'ksp_monitor': None,
        'ksp_converged_reason': None,
    }

    if verbose == 0:
        pass
    elif verbose == 1:
        lu_params['snes_converged_reason'] = None
    elif verbose >= 2:
        lu_params.update(verbose_options)

    PiSolver = fd.LinearVariationalSolver(PiProblem,
                                          solver_parameters=lu_params,
                                          options_prefix="pisolver")
    PiSolver.solve()
    v = wh.subfunctions[0]
    Pi0 = wh.subfunctions[1]
    if Pi:
        Pi.assign(Pi0)

    if rhon:
        rhon.interpolate(rho_formula(Pi0, thetan, gas))
        v = wh.subfunctions[0]
        rho = wh.subfunctions[1]
        rho.assign(rhon)
        v, rho = fd.split(wh)

        Pif = pi_formula(rho, thetan, gas)

        rhoeqn = gas.cp*(
            (fd.inner(v, dv) - fd.div(dv*thetan)*Pif)*fd.dx(degree=6)
            + drho*fd.div(thetan*v)*fd.dx
        )

        if top:
            bmeasure = fd.ds_t
            bstring = "bottom"
        else:
            bmeasure = fd.ds_b
            bstring = "top"

        zeros = []
        for i in range(Up.ufl_shape[0]):
            zeros.append(fd.Constant(0.))

        rhoeqn += gas.cp*fd.inner(dv, n)*thetan*pi_boundary*bmeasure
        rhoeqn += gas.g*fd.inner(dv, Up)*fd.dx
        bcs = [fd.DirichletBC(W_h.sub(0), zeros, bstring)]

        RhoProblem = fd.NonlinearVariationalProblem(rhoeqn, wh, bcs=bcs)

        RhoSolver = fd.NonlinearVariationalSolver(RhoProblem,
                                                  solver_parameters=lu_params,
                                                  options_prefix="rhosolver")

        RhoSolver.solve()
        v = wh.subfunctions[0]
        Rho0 = wh.subfunctions[1]
        rhon.assign(Rho0)
        del RhoSolver
    del PiSolver
    from firedrake.petsc import PETSc
    import gc
    gc.collect()
    PETSc.garbage_cleanup(mesh._comm)
