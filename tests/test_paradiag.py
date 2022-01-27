import asQ
import firedrake as fd
import numpy as np
import pytest
from petsc4py import PETSc


@pytest.mark.parallel(nprocs=8)
def test_jacobian_heat_equation():
    # tests the basic snes setup
    # using the heat equation
    # solves using unpreconditioned GMRES

    from petsc4py import PETSc
    PETSc.Sys.popErrorHandler()

    # only one spatial domain
    ensemble = fd.Ensemble(fd.COMM_WORLD, 2)

    mesh = fd.UnitSquareMesh(4, 4, comm=ensemble.comm)
    V = fd.FunctionSpace(mesh, "CG", 1)

    x, y = fd.SpatialCoordinate(mesh)
    u0 = fd.Function(V).interpolate(fd.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.5 ** 2))
    dt = 0.01
    theta = 0.5
    alpha = 0.001
    M = [2, 2, 2, 2]
    solver_parameters = {'ksp_type': 'gmres', 'pc_type': 'none',
                         'ksp_rtol': 1.0e-10, 'ksp_atol': 1.0e-10,
                         'ksp_monitor': None,
                         'ksp_max_it': 45,
                         'ksp_converged_reason': None,
                         # 'snes_test_jacobian': None,
                         # 'snes_test_jacobian_view': None,
                         'snes_converged_reason': None,
                         'snes_max_it': 2,
                         # 'snes_view': None
                         }

    def form_function(u, v):
        return fd.inner(fd.grad(u), fd.grad(v)) * fd.dx

    def form_mass(u, v):
        return u * v * fd.dx

    PD = asQ.paradiag(ensemble=ensemble,
                      form_function=form_function,
                      form_mass=form_mass, W=V, w0=u0,
                      dt=dt, theta=theta,
                      alpha=alpha,
                      M=M, solver_parameters=solver_parameters,
                      circ="none",
                      jac_average="newton", tol=1.0e-6, maxits=None,
                      ctx={}, block_mat_type="aij")
    PD.solve()

    assert (1 < PD.snes.getConvergedReason() < 5)


@pytest.mark.parallel(nprocs=8)
def test_set_para_form():
    # checks that the all-at-once system is the same as solving
    # timesteps sequentially using the heat equation as an example by
    # substituting the sequential solution and evaluating the residual

    # only one spatial domain
    ensemble = fd.Ensemble(fd.COMM_WORLD, 2)

    mesh = fd.UnitSquareMesh(4, 4, comm=ensemble.comm)
    V = fd.FunctionSpace(mesh, "CG", 1)

    x, y = fd.SpatialCoordinate(mesh)
    u0 = fd.Function(V).interpolate(fd.exp(-((x-0.5)**2 + (y-0.5)**2)/0.5**2))
    dt = 0.01
    theta = 0.5
    alpha = 0.001
    M = [2, 2, 2, 2]
    solver_parameters = {'ksp_type': 'gmres', 'pc_type': 'none',
                         'ksp_rtol': 1.0e-8, 'ksp_atol': 1.0e-8,
                         'ksp_monitor': None}

    def form_function(u, v):
        return fd.inner(fd.grad(u), fd.grad(v))*fd.dx

    def form_mass(u, v):
        return u*v*fd.dx

    PD = asQ.paradiag(ensemble=ensemble,
                      form_function=form_function,
                      form_mass=form_mass, W=V, w0=u0,
                      dt=dt, theta=theta,
                      alpha=alpha,
                      M=M, solver_parameters=solver_parameters,
                      circ="none",
                      jac_average="newton", tol=1.0e-6, maxits=None,
                      ctx={}, block_mat_type="aij")

    # sequential assembly
    WFull = V * V * V * V * V * V * V * V
    ufull = fd.Function(WFull)
    np.random.seed(132574)
    ufull_list = ufull.split()
    for i in range(8):
        ufull_list[i].dat.data[:] = np.random.randn(*(ufull_list[i].dat.data.shape))

    rT = ensemble.ensemble_comm.rank
    # copy the data from the full list into the time slice for this rank in PD.w_all
    w_alls = PD.w_all.split()
    w_alls[0].assign(ufull_list[rT*2])
    w_alls[1].assign(ufull_list[rT*2+1])
    # copy from w_all into the PETSc vec PD.X
    with PD.w_all.dat.vec_ro as v:
        v.copy(PD.X)

    # make a form for all of the time slices
    vfull = fd.TestFunction(WFull)
    ufulls = fd.split(ufull)
    vfulls = fd.split(vfull)
    for i in range(8):
        if i == 0:
            un = u0
        else:
            un = ufulls[i-1]
        unp1 = ufulls[i]
        v = vfulls[i]
        tform = form_mass(unp1 - un, v/dt) + form_function((unp1+un)/2, v)
        if i == 0:
            fullform = tform
        else:
            fullform += tform

    Ffull = fd.assemble(fullform)

    PD._assemble_function(PD.snes, PD.X, PD.F)
    PD_F1 = fd.Function(V)
    PD_F2 = fd.Function(V)
    vlen = V.node_set.size
    with PD_F1.dat.vec_ro as v:
        v.array[:] = PD.F.array_r[0:vlen]
    with PD_F2.dat.vec_ro as v:
        v.array[:] = PD.F.array_r[vlen:]

    error1 = fd.Function(V)
    error2 = fd.Function(V)
    r1 = fd.Function(V)
    r2 = fd.Function(V)
    r1.assign(Ffull.sub(rT * 2))
    r2.assign(Ffull.sub(rT * 2 + 1))
    error1.assign(r1 - PD_F1)
    error2.assign(r2 - PD_F2)

    assert(fd.norm(error1) < 1.0e-12)
    assert(fd.norm(error2) < 1.0e-12)


@pytest.mark.parallel(nprocs=8)
def test_set_para_form_mixed_parallel():
    # checks that the all-at-once system is the same as solving
    # timesteps sequentially using the heat equation as an example by
    # substituting the sequential solution and evaluating the residual

    # only one spatial domain
    ensemble = fd.Ensemble(fd.COMM_WORLD, 2)

    mesh = fd.UnitSquareMesh(4, 4, comm=ensemble.comm)
    V = fd.FunctionSpace(mesh, "BDM", 1)
    Q = fd.FunctionSpace(mesh, "DG", 0)
    W = V * Q

    x, y = fd.SpatialCoordinate(mesh)
    # u0 = fd.Function(V).interpolate(fd.exp(-((x-0.5)**2 + (y-0.5)**2)/0.5**2))
    w0 = fd.Function(W)
    u0, p0 = w0.split()
    p0.interpolate(fd.exp(-((x-0.5)**2 + (y-0.5)**2)/0.5**2))
    dt = 0.01
    theta = 0.5
    alpha = 0.001
    M = [2, 2, 2, 2]
    solver_parameters = {'ksp_type': 'gmres', 'pc_type': 'none',
                         'ksp_rtol': 1.0e-8, 'ksp_atol': 1.0e-8,
                         'ksp_monitor': None}

    def form_function(uu, up, vu, vp):
        return (fd.div(vu)*up - fd.div(uu)*vp)*fd.dx

    def form_mass(uu, up, vu, vp):
        return (fd.inner(uu, vu) + up*vp)*fd.dx

    PD = asQ.paradiag(ensemble=ensemble,
                      form_function=form_function,
                      form_mass=form_mass, W=W, w0=w0,
                      dt=dt, theta=theta,
                      alpha=alpha,
                      M=M, solver_parameters=solver_parameters,
                      circ="none",
                      jac_average="newton", tol=1.0e-6, maxits=None,
                      ctx={}, block_mat_type="aij")

    # sequential assembly
    WFull = W * W * W * W * W * W * W * W
    ufull = fd.Function(WFull)
    np.random.seed(132574)
    ufull_list = ufull.split()
    for i in range((2*8)):
        # print(i)
        ufull_list[i].dat.data[:] = np.random.randn(*(ufull_list[i].dat.data.shape))

    rT = ensemble.ensemble_comm.rank
    # copy the data from the full list into the time slice for this rank in PD.w_all
    w_alls = PD.w_all.split()
    w_alls[0].assign(ufull_list[4 * rT])   # 1st time slice V
    w_alls[1].assign(ufull_list[4 * rT + 1])  # 1st time slice Q
    w_alls[2].assign(ufull_list[4 * rT + 2])  # 2nd time slice V
    w_alls[3].assign(ufull_list[4 * rT + 3])  # 2nd time slice Q

    # copy from w_all into the PETSc vec PD.X
    with PD.w_all.dat.vec_ro as v:
        v.copy(PD.X)

    # make a form for all of the time slices
    vfull = fd.TestFunction(WFull)
    ufulls = fd.split(ufull)
    vfulls = fd.split(vfull)

    for i in range(8):
        if i == 0:
            un = u0
            pn = p0
        else:
            un = ufulls[2 * i - 2]
            pn = ufulls[2 * i - 1]
        unp1 = ufulls[2 * i]
        pnp1 = ufulls[2 * i + 1]
        vu = vfulls[2 * i]
        vp = vfulls[2 * i + 1]
        # forms have 2 components and 2 test functions: (u, h, w, phi)
        tform = form_mass(unp1 - un, pnp1 - pn, vu / dt, vp / dt) \
            + form_function((unp1 + un) / 2, (pnp1 + pn) / 2, vu, vp)
        if i == 0:
            fullform = tform
        else:
            fullform += tform

    Ffull = fd.assemble(fullform)

    PD._assemble_function(PD.snes, PD.X, PD.F)
    PD_F1 = fd.Function(W)
    PD_F2 = fd.Function(W)
    vlen = V.node_set.size
    plen = Q.node_set.size

    with PD_F1.sub(0).dat.vec_ro as v:
        v.array[:] = PD.F.array_r[0:vlen]
    with PD_F1.sub(1).dat.vec_ro as v:
        v.array[:] = PD.F.array_r[vlen:vlen + plen]
    with PD_F2.sub(0).dat.vec_ro as v:
        v.array[:] = PD.F.array_r[vlen + plen:vlen + plen + vlen]
    with PD_F2.sub(1).dat.vec_ro as v:
        v.array[:] = PD.F.array_r[vlen + plen + vlen:]

    r11, r12 = fd.Function(W).split()
    r21, r22 = fd.Function(W).split()
    e11, e12 = fd.Function(W).split()
    e21, e22 = fd.Function(W).split()
    r11.assign(Ffull.sub(rT * 4))
    r12.assign(Ffull.sub(rT * 4 + 1))
    r21.assign(Ffull.sub(rT * 4 + 2))
    r22.assign(Ffull.sub(rT * 4 + 3))
    e11.assign(r11 - PD_F1.sub(0))
    e12.assign(r12 - PD_F1.sub(1))
    e21.assign(r21 - PD_F2.sub(0))
    e22.assign(r22 - PD_F2.sub(1))

    assert(fd.norm(e11) < 1.0e-12)
    assert(fd.norm(e12) < 1.0e-12)
    assert(fd.norm(e21) < 1.0e-12)
    assert(fd.norm(e22) < 1.0e-12)


@pytest.mark.parallel(nprocs=8)
def test_jacobian_mixed_parallel1():
    ensemble = fd.Ensemble(fd.COMM_WORLD, 2)

    mesh = fd.UnitSquareMesh(4, 4, comm=ensemble.comm)
    V = fd.FunctionSpace(mesh, "BDM", 1)
    Q = fd.FunctionSpace(mesh, "DG", 0)
    W = V * Q

    x, y = fd.SpatialCoordinate(mesh)
    w0 = fd.Function(W)
    u0, p0 = w0.split()
    # p0, u0 = w0.split()
    p0.interpolate(fd.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.5 ** 2))
    dt = 0.01
    theta = 0.5
    alpha = 0.001
    M = [2, 2, 2, 2]
    Ml = np.sum(M)
    c = fd.Constant(0.1)
    eps = fd.Constant(0.001)

    solver_parameters = {'ksp_type': 'gmres', 'pc_type': 'none',
                         'ksp_rtol': 1.0e-8, 'ksp_atol': 1.0e-8,
                         'ksp_monitor': None}

    def form_function(uu, up, vu, vp):
        return (fd.div(vu) * up +
                c * fd.sqrt(fd.inner(uu, uu) + eps) * fd.inner(uu, vu)
                - fd.div(uu) * vp) * fd.dx

    def form_mass(uu, up, vu, vp):
        return (fd.inner(uu, vu) + up * vp) * fd.dx

    PD = asQ.paradiag(ensemble=ensemble,
                      form_function=form_function,
                      form_mass=form_mass, W=W, w0=w0,
                      dt=dt, theta=theta,
                      alpha=alpha,
                      M=M, solver_parameters=solver_parameters,
                      circ="none",
                      jac_average="newton", tol=1.0e-6, maxits=None,
                      ctx={}, block_mat_type="aij")

    # sequential assembly
    WFull = np.prod([W for i in range(Ml)])
    ufull = fd.Function(WFull)
    np.random.seed(132574)
    ufull_list = ufull.split()
    for i in range((2 * Ml)):
        ufull_list[i].dat.data[:] = np.random.randn(*(ufull_list[i].dat.data.shape))

    # make another function v_alls:
    vfull = fd.Function(WFull)
    vfull_list = vfull.split()
    for i in range((2 * Ml)):
        vfull_list[i].dat.data[:] = np.random.randn(*(vfull_list[i].dat.data.shape))

    rT = ensemble.ensemble_comm.rank

    # copy the data from the full list into the time slice for this rank in PD.w_all
    w_alls = PD.w_all.split()
    v_all = fd.Function(PD.W_all)
    v_alls = v_all.split()

    nM = M[rT]
    for i in range(nM):
        # sum over the entries of M until rT determines left position left
        left = np.sum(M[:rT],dtype=int)
        ind1 = 2*left + 2*i
        ind2 = 2*left + 2*i + 1
        w_alls[2*i].assign(ufull_list[ind1])  # ith time slice V
        w_alls[2*i + 1].assign(ufull_list[ind2])  # ith time slice Q
        v_alls[2*i].assign(vfull_list[ind1])  # ith time slice V
        v_alls[2*i + 1].assign(vfull_list[ind2])  # ith time slice Q

    # Parallel PARADIAG: calculate Jac1 with PD
    # copy from w_all into the PETSc vec PD.X
    with PD.w_all.dat.vec_ro as v:
        v.copy(PD.X)

    # use PD to calculate the Jacobian
    Jac1 = PD.JacobianMatrix

    # construct Petsc vector X1, Y1:
    nlocal = M[rT]*W.node_set.size  # local times x local space
    nglobal = np.prod(M)*W.dim()  # global times x global space
    X1 = PETSc.Vec().create(comm=fd.COMM_WORLD)
    X1.setSizes((nlocal, nglobal))
    X1.setFromOptions()
    Y1 = PETSc.Vec().create(comm=fd.COMM_WORLD)
    Y1.setSizes((nlocal, nglobal))
    Y1.setFromOptions()

    # copy from v_all into the PETSc vec X1
    with v_all.dat.vec_ro as v:
        v.copy(X1)

    # do the matrix multiplication of Jac1 with X1, write output into Y1
    Jac1.mult(None, X1, Y1)

    # SERIAL: make a form for all of the time slices
    tfull = fd.TestFunction(WFull)
    ufulls = fd.split(ufull)
    tfulls = fd.split(tfull)

    for i in range(Ml):
        if i == 0:
            un = u0
            pn = p0
        else:
            un, pn = ufulls[2*i - 2: 2*i]
        unp1, pnp1 = ufulls[2*i: 2*i + 2]
        vu, vp = tfulls[2*i: 2*i + 2]

        tform = form_mass(unp1 - un, pnp1 - pn, vu / dt, vp / dt) \
            + 0.5*form_function(unp1, pnp1, vu, vp) \
            + 0.5*form_function(un, pn, vu, vp)
        if i == 0:
            fullform = tform
        else:
            fullform += tform

    # calculate derivative of Jac2 directly from serial fullform wrt ufull:
    Jac2 = fd.derivative(fullform, ufull)
    # do the matrix multiplication with vfull:
    jacout = fd.assemble(fd.action(Jac2, vfull))

    vlen, plen = V.node_set.size, Q.node_set.size

    # generalization of the error evaluation to nM time slices
    PD_J = fd.Function(PD.W_all)
    PD_Js = PD_J.split()
    for i in range(nM):
        with PD_Js[2*i].dat.vec_ro as v:
            v.array[:] = Y1.array_r[i*vlen + i*plen: (i+1)*vlen + i*plen]
        with PD_Js[2*i+1].dat.vec_ro as v:
            v.array[:] = Y1.array_r[(i+1)*vlen + i*plen: (i+1)*vlen + (i+1)*plen]

    err_all = fd.Function(PD.W_all)
    err_alls = err_all.split()
    for i in range(nM):
        left = np.sum(M[:rT], dtype=int)
        ind1 = 2*left + 2*i
        ind2 = 2*left + 2*i + 1
        err_alls[2*i].assign(jacout.sub(ind1))  # ith time slice V
        err_alls[2*i+1].assign(jacout.sub(ind2))  # ith time slice Q
        err_alls[2*i].assign(jacout.sub(ind1))  # ith time slice V
        err_alls[2*i+1].assign(jacout.sub(ind2))  # ith time slice Q

    error_all = fd.Function(PD.W_all)
    error_alls = error_all.split()
    for i in range(2 * nM):
        error_alls[i].assign(err_alls[i] - PD_Js[i])
        assert(fd.norm(error_alls[i]) < 1.0e-12)


@pytest.mark.xfail
def test_set_para_form_mixed():
    # checks that the all-at-once system is the same as solving
    # timesteps sequentially using the mixed wave equation as an
    # example by substituting the sequential solution and evaluating
    # the residual

    mesh = fd.PeriodicUnitSquareMesh(20, 20)
    V = fd.FunctionSpace(mesh, "BDM", 1)
    Q = fd.FunctionSpace(mesh, "DG", 0)
    W = V * Q

    x, y = fd.SpatialCoordinate(mesh)
    w0 = fd.Function(W)
    u0, p0 = w0.split()
    p0.interpolate(fd.exp(-((x-0.5)**2 + (y-0.5)**2)/0.5**2))
    dt = 0.01
    theta = 0.5
    alpha = 0.001
    M = 4
    solver_parameters = {'ksp_type': 'gmres', 'pc_type': 'none',
                         'ksp_rtol': 1.0e-8, 'ksp_atol': 1.0e-8,
                         'ksp_monitor': None}

    def form_function(uu, up, vu, vp):
        return (fd.div(vu)*up - fd.div(uu)*vp)*fd.dx

    def form_mass(uu, up, vu, vp):
        return (fd.inner(uu, vu) + up*vp)*fd.dx

    PD = asQ.paradiag(form_function=form_function,
                      form_mass=form_mass, W=W, w0=w0, dt=dt,
                      theta=theta, alpha=alpha, M=M,
                      solver_parameters=solver_parameters,
                      circ="none")

    # sequential solver
    un = fd.Function(W)
    unp1 = fd.Function(W)

    un.assign(w0)
    v = fd.TestFunction(W)

    eqn = (1.0/dt)*form_mass(*(fd.split(unp1)), *(fd.split(v)))
    eqn -= (1.0/dt)*form_mass(*(fd.split(un)), *(fd.split(v)))
    eqn += fd.Constant((1-theta))*form_function(*(fd.split(un)),
                                                *(fd.split(v)))
    eqn += fd.Constant(theta)*form_function(*(fd.split(unp1)),
                                            *(fd.split(v)))

    sprob = fd.NonlinearVariationalProblem(eqn, unp1)
    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu',
                         'pc_factor_mat_solver_type': 'mumps',
                         'mat_type': 'aij'}
    ssolver = fd.NonlinearVariationalSolver(sprob,
                                            solver_parameters=solver_parameters)

    for i in range(M):
        ssolver.solve()
        for k in range(2):
            PD.w_all.sub(2*i+k).assign(unp1.sub(k))
        un.assign(unp1)

    Pres = fd.assemble(PD.para_form)
    for i in range(M):
        assert(dt*np.abs(Pres.sub(i).dat.data[:]).max() < 1.0e-16)


@pytest.mark.xfail
def test_solve_para_form():
    # checks that the all-at-once system is the same as solving
    # timesteps sequentially using the heat equation as an example by
    # solving the all-at-once system and comparing with the sequential
    # solution

    mesh = fd.UnitSquareMesh(20, 20)
    V = fd.FunctionSpace(mesh, "CG", 1)

    x, y = fd.SpatialCoordinate(mesh)
    u0 = fd.Function(V).interpolate(fd.exp(-((x-0.5)**2 + (y-0.5)**2)/0.5**2))
    dt = 0.01
    theta = 0.5
    alpha = 0.001
    M = 4
    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu',
                         'pc_factor_mat_solver_type': 'mumps',
                         'mat_type': 'aij'}

    def form_function(u, v):
        return fd.inner(fd.grad(u), fd.grad(v))*fd.dx

    def form_mass(u, v):
        return u*v*fd.dx

    PD = asQ.paradiag(form_function=form_function,
                      form_mass=form_mass, W=V, w0=u0, dt=dt,
                      theta=theta, alpha=alpha, M=M,
                      solver_parameters=solver_parameters,
                      circ="none")
    PD.solve()

    # sequential solver
    un = fd.Function(V)
    unp1 = fd.Function(V)

    un.assign(u0)
    v = fd.TestFunction(V)

    eqn = (unp1 - un)*v*fd.dx
    eqn += fd.Constant(dt*(1-theta))*form_function(un, v)
    eqn += fd.Constant(dt*theta)*form_function(unp1, v)

    sprob = fd.NonlinearVariationalProblem(eqn, unp1)
    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu'}
    ssolver = fd.NonlinearVariationalSolver(sprob,
                                            solver_parameters=solver_parameters)

    err = fd.Function(V, name="err")
    pun = fd.Function(V, name="pun")
    for i in range(M):
        ssolver.solve()
        un.assign(unp1)
        pun.assign(PD.w_all.sub(i))
        err.assign(un-pun)
        assert(fd.norm(err) < 1.0e-15)


@pytest.mark.xfail
def test_solve_para_form_mixed():
    # checks that the all-at-once system is the same as solving
    # timesteps sequentially using the mixed wave equation as an
    # example by substituting the sequential solution and evaluating
    # the residual

    mesh = fd.PeriodicUnitSquareMesh(20, 20)
    V = fd.FunctionSpace(mesh, "BDM", 1)
    Q = fd.FunctionSpace(mesh, "DG", 0)
    W = V * Q

    x, y = fd.SpatialCoordinate(mesh)
    w0 = fd.Function(W)
    u0, p0 = w0.split()
    p0.interpolate(fd.exp(-((x-0.5)**2 + (y-0.5)**2)/0.5**2))
    dt = 0.01
    theta = 0.5
    alpha = 0.001
    M = 4
    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu',
                         'pc_factor_mat_solver_type': 'mumps',
                         'mat_type': 'aij'}

    def form_function(uu, up, vu, vp):
        return (fd.div(vu)*up - fd.div(uu)*vp)*fd.dx

    def form_mass(uu, up, vu, vp):
        return (fd.inner(uu, vu) + up*vp)*fd.dx

    PD = asQ.paradiag(form_function=form_function,
                      form_mass=form_mass, W=W, w0=w0, dt=dt,
                      theta=theta, alpha=alpha, M=M,
                      solver_parameters=solver_parameters,
                      circ="none")
    PD.solve()

    # sequential solver
    un = fd.Function(W)
    unp1 = fd.Function(W)

    un.assign(w0)
    v = fd.TestFunction(W)

    eqn = form_mass(*(fd.split(unp1)), *(fd.split(v)))
    eqn -= form_mass(*(fd.split(un)), *(fd.split(v)))
    eqn += fd.Constant(dt*(1-theta))*form_function(*(fd.split(un)),
                                                   *(fd.split(v)))
    eqn += fd.Constant(dt*theta)*form_function(*(fd.split(unp1)),
                                               *(fd.split(v)))

    sprob = fd.NonlinearVariationalProblem(eqn, unp1)
    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu',
                         'pc_factor_mat_solver_type': 'mumps',
                         'mat_type': 'aij'}
    ssolver = fd.NonlinearVariationalSolver(sprob,
                                            solver_parameters=solver_parameters)
    ssolver.solve()

    err = fd.Function(W, name="err")
    pun = fd.Function(W, name="pun")
    puns = pun.split()
    for i in range(M):
        ssolver.solve()
        un.assign(unp1)
        walls = PD.w_all.split()[2*i:2*i+2]
        for k in range(2):
            puns[k].assign(walls[k])
        err.assign(un-pun)
        assert(fd.norm(err) < 1.0e-15)


@pytest.mark.xfail
def test_relax():
    # tests the relaxation method
    # using the heat equation as an example

    mesh = fd.UnitSquareMesh(20, 20)
    V = fd.FunctionSpace(mesh, "CG", 1)

    x, y = fd.SpatialCoordinate(mesh)
    u0 = fd.Function(V).interpolate(fd.exp(-((x-0.5)**2 + (y-0.5)**2)/0.5**2))
    dt = 0.01
    theta = 0.5
    alpha = 0.001
    M = 4
    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu',
                         'pc_factor_mat_solver_type': 'mumps',
                         'mat_type': 'aij'}

    # Solving U_t + F(U) = 0
    # defining F(U)
    def form_function(u, v):
        return fd.inner(fd.grad(u), fd.grad(v))*fd.dx

    # defining the structure of U_t
    def form_mass(u, v):
        return u*v*fd.dx

    PD = asQ.paradiag(form_function=form_function,
                      form_mass=form_mass, W=V, w0=u0, dt=dt,
                      theta=theta, alpha=alpha, M=M,
                      solver_parameters=solver_parameters,
                      circ="picard", tol=1.0e-12)
    PD.solve()

    # sequential solver
    un = fd.Function(V)
    unp1 = fd.Function(V)

    un.assign(u0)
    v = fd.TestFunction(V)

    eqn = (unp1 - un)*v*fd.dx
    eqn += fd.Constant(dt*(1-theta))*form_function(un, v)
    eqn += fd.Constant(dt*theta)*form_function(unp1, v)

    sprob = fd.NonlinearVariationalProblem(eqn, unp1)
    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu'}
    ssolver = fd.NonlinearVariationalSolver(sprob,
                                            solver_parameters=solver_parameters)

    err = fd.Function(V, name="err")
    pun = fd.Function(V, name="pun")
    for i in range(M):
        ssolver.solve()
        un.assign(unp1)
        pun.assign(PD.w_all.sub(i))
        err.assign(un-pun)
        assert(fd.norm(err) < 1.0e-15)


@pytest.mark.xfail
def test_relax_mixed():
    # checks that the all-at-once system is the same as solving
    # timesteps sequentially using the mixed wave equation as an
    # example by substituting the sequential solution and evaluating
    # the residual

    mesh = fd.PeriodicUnitSquareMesh(20, 20)
    V = fd.FunctionSpace(mesh, "BDM", 1)
    Q = fd.FunctionSpace(mesh, "DG", 0)
    W = V * Q

    x, y = fd.SpatialCoordinate(mesh)
    w0 = fd.Function(W)
    u0, p0 = w0.split()
    p0.interpolate(fd.exp(-((x-0.5)**2 + (y-0.5)**2)/0.5**2))
    dt = 0.01
    theta = 0.5
    alpha = 0.001
    M = 4
    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu',
                         'pc_factor_mat_solver_type': 'mumps',
                         'mat_type': 'aij'}

    def form_function(uu, up, vu, vp):
        return (fd.div(vu)*up - fd.div(uu)*vp)*fd.dx

    def form_mass(uu, up, vu, vp):
        return (fd.inner(uu, vu) + up*vp)*fd.dx

    PD = asQ.paradiag(form_function=form_function,
                      form_mass=form_mass, W=W, w0=w0, dt=dt,
                      theta=theta, alpha=alpha, M=M,
                      solver_parameters=solver_parameters,
                      circ="picard", tol=1.0e-12)
    PD.solve(verbose=True)

    # sequential solver
    un = fd.Function(W)
    unp1 = fd.Function(W)

    un.assign(w0)
    v = fd.TestFunction(W)

    eqn = form_mass(*(fd.split(unp1)), *(fd.split(v)))
    eqn -= form_mass(*(fd.split(un)), *(fd.split(v)))
    eqn += fd.Constant(dt*(1-theta))*form_function(*(fd.split(un)),
                                                   *(fd.split(v)))
    eqn += fd.Constant(dt*theta)*form_function(*(fd.split(unp1)),
                                               *(fd.split(v)))

    sprob = fd.NonlinearVariationalProblem(eqn, unp1)
    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu',
                         'pc_factor_mat_solver_type': 'mumps',
                         'mat_type': 'aij'}
    ssolver = fd.NonlinearVariationalSolver(sprob,
                                            solver_parameters=solver_parameters)
    ssolver.solve()

    err = fd.Function(W, name="err")
    pun = fd.Function(W, name="pun")
    puns = pun.split()
    for i in range(M):
        ssolver.solve()
        un.assign(unp1)
        walls = PD.w_all.split()[2*i:2*i+2]
        for k in range(2):
            puns[k].assign(walls[k])
        err.assign(un-pun)
        assert(fd.norm(err) < 1.0e-15)


@pytest.mark.xfail
def test_diag_precon():
    # Test PCDIAGFFT by using it
    # within the relaxation method
    # using the heat equation as an example
    # we compare one iteration using just the diag PC
    # with the direct solver

    mesh = fd.UnitSquareMesh(20, 20)
    V = fd.FunctionSpace(mesh, "CG", 1)

    x, y = fd.SpatialCoordinate(mesh)
    u0 = fd.Function(V).interpolate(fd.exp(-((x-0.5)**2 + (y-0.5)**2)/0.5**2))
    dt = 0.01
    theta = 0.5
    alpha = 0.01
    M = 4

    diagfft_options = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
        'mat_type': 'aij'}

    solver_parameters = {
        'snes_type': 'ksponly',
        'mat_type': 'matfree',
        'ksp_type': 'preonly',
        'ksp_rtol': 1.0e-10,
        'ksp_converged_reason': None,
        'pc_type': 'python',
        'pc_python_type': 'asQ.DiagFFTPC',
    }
    for i in range(M):
        solver_parameters["diagfft_"+str(i)+"_"] = diagfft_options

    def form_function(u, v):
        return fd.inner(fd.grad(u), fd.grad(v))*fd.dx

    def form_mass(u, v):
        return u*v*fd.dx

    PD = asQ.paradiag(form_function=form_function,
                      form_mass=form_mass, W=V, w0=u0, dt=dt,
                      theta=theta, alpha=alpha, M=M,
                      solver_parameters=solver_parameters,
                      circ="picard", tol=1.0e-12, maxits=1)
    PD.solve(verbose=True)
    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu',
                         'pc_factor_mat_solver_type': 'mumps',
                         'mat_type': 'aij'}
    PDe = asQ.paradiag(form_function=form_function,
                       form_mass=form_mass, W=V, w0=u0, dt=dt,
                       theta=theta, alpha=alpha, M=M,
                       solver_parameters=solver_parameters,
                       circ="picard", tol=1.0e-12, maxits=1)
    PDe.solve(verbose=True)
    unD = fd.Function(V, name='diag')
    un = fd.Function(V, name='full')
    err = fd.Function(V, name='error')
    unD.assign(u0)
    un.assign(u0)
    for i in range(M):
        walls = PD.w_all.split()[i]
        wallsE = PDe.w_all.split()[i]
        unD.assign(walls)
        un.assign(wallsE)
        err.assign(un-unD)
        assert(fd.norm(err) < 1.0e-13)


@pytest.mark.xfail
def test_diag_precon_mixed():
    # checks that the all-at-once system is the same as solving
    # timesteps sequentially using the mixed wave equation as an
    # example by substituting the sequential solution and evaluating
    # the residual

    mesh = fd.PeriodicUnitSquareMesh(20, 20)
    V = fd.FunctionSpace(mesh, "BDM", 1)
    Q = fd.FunctionSpace(mesh, "DG", 0)
    W = V * Q

    x, y = fd.SpatialCoordinate(mesh)
    w0 = fd.Function(W)
    u0, p0 = w0.split()
    p0.interpolate(fd.exp(-((x-0.5)**2 + (y-0.5)**2)/0.5**2))
    dt = 0.01
    theta = 0.5
    alpha = 0.001
    M = 4

    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu',
                         'pc_factor_mat_solver_type': 'mumps',
                         'mat_type': 'aij'}

    def form_function(uu, up, vu, vp):
        return (fd.div(vu)*up - fd.div(uu)*vp)*fd.dx

    def form_mass(uu, up, vu, vp):
        return (fd.inner(uu, vu) + up*vp)*fd.dx

    diagfft_options = {'ksp_type': 'gmres', 'pc_type': 'lu',
                       'ksp_monitor': None,
                       'ksp_converged_reason': None,
                       'pc_factor_mat_solver_type': 'mumps',
                       'mat_type': 'aij'}

    solver_parameters_diag = {
        'snes_type': 'ksponly',
        'mat_type': 'matfree',
        'ksp_type': 'preonly',
        'ksp_rtol': 1.0e-10,
        'ksp_converged_reason': None,
        'pc_type': 'python',
        'pc_python_type': 'asQ.DiagFFTPC',
    }
    for i in range(M):
        solver_parameters_diag["diagfft_"+str(i)+"_"] = diagfft_options

    PD = asQ.paradiag(form_function=form_function,
                      form_mass=form_mass, W=W, w0=w0, dt=dt,
                      theta=theta, alpha=alpha, M=M,
                      solver_parameters=solver_parameters_diag,
                      circ="picard", tol=1.0e-12)
    PD.solve(verbose=True)

    # sequential solver
    un = fd.Function(W)
    unp1 = fd.Function(W)

    un.assign(w0)
    v = fd.TestFunction(W)

    eqn = form_mass(*(fd.split(unp1)), *(fd.split(v)))
    eqn -= form_mass(*(fd.split(un)), *(fd.split(v)))
    eqn += fd.Constant(dt*(1-theta))*form_function(*(fd.split(un)),
                                                   *(fd.split(v)))
    eqn += fd.Constant(dt*theta)*form_function(*(fd.split(unp1)),
                                               *(fd.split(v)))

    sprob = fd.NonlinearVariationalProblem(eqn, unp1)
    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu',
                         'pc_factor_mat_solver_type': 'mumps',
                         'mat_type': 'aij'}
    ssolver = fd.NonlinearVariationalSolver(sprob,
                                            solver_parameters=solver_parameters)
    ssolver.solve()

    err = fd.Function(W, name="err")
    pun = fd.Function(W, name="pun")
    puns = pun.split()
    for i in range(M):
        ssolver.solve()
        un.assign(unp1)
        walls = PD.w_all.split()[2*i:2*i+2]
        for k in range(2):
            puns[k].assign(walls[k])
        err.assign(un-pun)
        assert(fd.norm(err) < 1.0e-15)


@pytest.mark.xfail
def test_diag_precon_nl():
    # Test PCDIAGFFT by using it within the relaxation method
    # using the NONLINEAR heat equation as an example
    # we compare one iteration using just the diag PC
    # with the direct solver

    mesh = fd.UnitSquareMesh(20, 20)
    V = fd.FunctionSpace(mesh, "CG", 1)

    x, y = fd.SpatialCoordinate(mesh)
    u0 = fd.Function(V).interpolate(fd.exp(-((x-0.5)**2 + (y-0.5)**2)/0.5**2))
    dt = 0.01
    theta = 0.5
    alpha = 0.01
    M = 4
    c = fd.Constant(0.1)

    diagfft_options = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'ksp_monitor': None,
        'ksp_converged_reason': None,
        'pc_factor_mat_solver_type': 'mumps',
        'mat_type': 'aij'}

    solver_parameters = {
        'snes_monitor': None,
        'snes_converged_reason': None,
        'mat_type': 'matfree',
        'ksp_rtol': 1.0e-10,
        'ksp_max_it': 12,
        'ksp_converged_reason': None,
        'pc_type': 'python',
        'pc_python_type': 'asQ.DiagFFTPC',
    }
    for i in range(M):
        solver_parameters["diagfft_"+str(i)+"_"] = diagfft_options

    def form_function(u, v):
        return fd.inner((1.+c*fd.inner(u, u))*fd.grad(u), fd.grad(v))*fd.dx

    def form_mass(u, v):
        return u*v*fd.dx

    PD = asQ.paradiag(form_function=form_function,
                      form_mass=form_mass, W=V, w0=u0, dt=dt,
                      theta=theta, alpha=alpha, M=M,
                      solver_parameters=solver_parameters,
                      circ="picard", tol=1.0e-12, maxits=1)
    PD.solve(verbose=True)
    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu',
                         'pc_factor_mat_solver_type': 'mumps',
                         'mat_type': 'aij'
                         }
    PDe = asQ.paradiag(form_function=form_function,
                       form_mass=form_mass, W=V, w0=u0, dt=dt,
                       theta=theta, alpha=alpha, M=M,
                       solver_parameters=solver_parameters,
                       circ="picard", tol=1.0e-12, maxits=1)
    PDe.solve(verbose=True)
    unD = fd.Function(V, name='diag')
    un = fd.Function(V, name='full')
    err = fd.Function(V, name='error')
    unD.assign(u0)
    un.assign(u0)
    for i in range(M):
        walls = PD.w_all.split()[i]
        wallsE = PDe.w_all.split()[i]
        unD.assign(walls)
        un.assign(wallsE)
        err.assign(un-unD)
        assert(fd.norm(err) < 1.0e-12)


@pytest.mark.xfail
def test_quasi():
    # tests the quasi-Newton option
    # using the heat equation as an example

    mesh = fd.UnitSquareMesh(20, 20)
    V = fd.FunctionSpace(mesh, "CG", 1)

    x, y = fd.SpatialCoordinate(mesh)
    u0 = fd.Function(V).interpolate(fd.exp(-((x-0.5)**2 + (y-0.5)**2)/0.5**2))
    dt = 0.01
    theta = 0.5
    alpha = 0.001
    M = 4
    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu',
                         'pc_factor_mat_solver_type': 'mumps',
                         'mat_type': 'aij',
                         'snes_monitor': None}

    # Solving U_t + F(U) = 0
    # defining F(U)
    def form_function(u, v):
        return fd.inner(fd.grad(u), fd.grad(v))*fd.dx

    # defining the structure of U_t
    def form_mass(u, v):
        return u*v*fd.dx

    PD = asQ.paradiag(form_function=form_function,
                      form_mass=form_mass, W=V, w0=u0, dt=dt,
                      theta=theta, alpha=alpha, M=M,
                      solver_parameters=solver_parameters,
                      circ="quasi")
    PD.solve()

    # sequential solver
    un = fd.Function(V)
    unp1 = fd.Function(V)

    un.assign(u0)
    v = fd.TestFunction(V)

    eqn = (unp1 - un)*v*fd.dx
    eqn += fd.Constant(dt*(1-theta))*form_function(un, v)
    eqn += fd.Constant(dt*theta)*form_function(unp1, v)

    sprob = fd.NonlinearVariationalProblem(eqn, unp1)
    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu'}
    ssolver = fd.NonlinearVariationalSolver(sprob,
                                            solver_parameters=solver_parameters)

    err = fd.Function(V, name="err")
    pun = fd.Function(V, name="pun")
    for i in range(M):
        ssolver.solve()
        un.assign(unp1)
        pun.assign(PD.w_all.sub(i))
        err.assign(un-pun)
        assert(fd.norm(err) < 1.0e-10)


@pytest.mark.xfail
def test_diag_precon_nl_mixed():
    # Test PCDIAGFFT by using it
    # within the relaxation method
    # using the NONLINEAR wave equation as an example
    # we compare one iteration using just the diag PC
    # with the direct solver

    mesh = fd.PeriodicUnitSquareMesh(20, 20)
    V = fd.FunctionSpace(mesh, "BDM", 1)
    Q = fd.FunctionSpace(mesh, "DG", 0)
    W = V * Q

    x, y = fd.SpatialCoordinate(mesh)
    w0 = fd.Function(W)
    u0, p0 = w0.split()
    p0.interpolate(fd.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.5 ** 2))
    dt = 0.01
    theta = 0.5
    alpha = 0.001
    M = 4
    c = fd.Constant(10)
    eps = fd.Constant(0.001)

    def form_function(uu, up, vu, vp):
        return (fd.div(vu) * up + c * fd.sqrt(fd.inner(uu, uu) + eps) * fd.inner(uu, vu)
                - fd.div(uu) * vp) * fd.dx

    def form_mass(uu, up, vu, vp):
        return (fd.inner(uu, vu) + up * vp) * fd.dx

    diagfft_options = {
        'ksp_type': 'gmres',
        'pc_type': 'lu',
        'ksp_monitor': None,
        'ksp_converged_reason': None,
        'pc_factor_mat_solver_type': 'mumps',
        'mat_type': 'aij'}

    solver_parameters_diag = {
        'snes_monitor': None,
        'snes_converged_reason': None,
        'mat_type': 'matfree',
        'ksp_rtol': 1.0e-10,
        'ksp_max_it': 6,
        'ksp_converged_reason': None,
        'pc_type': 'python',
        'pc_python_type': 'asQ.DiagFFTPC',
    }
    for i in range(M):
        solver_parameters_diag["diagfft_"+str(i)+"_"] = diagfft_options

    PD = asQ.paradiag(form_function=form_function,
                      form_mass=form_mass, W=W, w0=w0, dt=dt,
                      theta=theta, alpha=alpha, M=M,
                      solver_parameters=solver_parameters_diag,
                      circ="quasi", tol=1.0e-12,
                      maxits=1)
    PD.solve(verbose=True)

    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu',
                         'pc_factor_mat_solver_type': 'mumps',
                         'mat_type': 'aij',
                         # 'snes_monitor':None,
                         # 'snes_converged_reason':None,
                         }
    PDe = asQ.paradiag(form_function=form_function,
                       form_mass=form_mass, W=W, w0=w0, dt=dt,
                       theta=theta, alpha=alpha, M=M,
                       solver_parameters=solver_parameters,
                       circ="quasi", tol=1.0e-12,
                       maxits=1)
    PDe.solve(verbose=True)

    # define functions
    un = fd.Function(W, name='full')
    unD = fd.Function(W, name='diag')
    err = fd.Function(W, name='error')
    # write initial conditions
    un.assign(w0)
    unD.assign(w0)
    # split
    pun = fd.Function(W, name='pun')
    punD = fd.Function(W, name='punD')
    puns = pun.split()
    punDs = punD.split()

    for i in range(M):
        walls = PD.w_all.split()[2 * i:2 * i + 2]
        wallsE = PDe.w_all.split()[2 * i:2 * i + 2]
        for k in range(2):
            puns[k].assign(walls[k])
            punDs[k].assign(wallsE[k])
        err.assign(punD - pun)
        print(fd.norm(err))
        assert (fd.norm(err) < 1.0e-15)
