import asQ
import firedrake as fd
import numpy as np


def test_set_para_form():
    # checks that the all-at-once system is the same as solving
    # timesteps sequentially using the heat equation as an example by
    # substituting the sequential solution and evaluating the residual

    mesh = fd.UnitSquareMesh(20, 20)
    V = fd.FunctionSpace(mesh, "CG", 1)

    x, y = fd.SpatialCoordinate(mesh)
    u0 = fd.Function(V).interpolate(fd.exp(-((x-0.5)**2 + (y-0.5)**2)/0.5**2))
    dt = 0.01
    theta = 0.5
    alpha = 0.001
    M = 4
    solver_parameters = {'ksp_type': 'gmres', 'pc_type': 'none',
                         'ksp_rtol': 1.0e-8, 'ksp_atol': 1.0e-8,
                         'ksp_monitor': None}

    def form_function(u, v):
        return fd.inner(fd.grad(u), fd.grad(v))*fd.dx

    def form_mass(u, v):
        return u*v*fd.dx

    PD = asQ.paradiag(form_function=form_function,
                      form_mass=form_mass, W=V, w0=u0,
                      dt=dt, theta=theta,
                      alpha=alpha,
                      M=M, solver_parameters=solver_parameters,
                      circ="none")

    # sequential solver
    un = fd.Function(V)
    unp1 = fd.Function(V)

    un.assign(u0)
    v = fd.TestFunction(V)

    eqn = (unp1 - un)*v/dt*fd.dx
    eqn += fd.Constant((1-theta))*form_function(un, v)
    eqn += fd.Constant(theta)*form_function(unp1, v)

    sprob = fd.NonlinearVariationalProblem(eqn, unp1)
    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu'}
    ssolver = fd.NonlinearVariationalSolver(sprob,
                                            solver_parameters  # noqa: E251
                                            =solver_parameters)  # noqa: E251

    for i in range(M):
        ssolver.solve()
        PD.w_all.sub(i).assign(unp1)
        un.assign(unp1)

    Pres = fd.assemble(PD.para_form)
    for i in range(M):
        assert(dt*np.abs(Pres.sub(i).dat.data[:]).max() < 1.0e-16)


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
                                            solver_parameters  # noqa: E251
                                            =solver_parameters)

    for i in range(M):
        ssolver.solve()
        for k in range(2):
            PD.w_all.sub(2*i+k).assign(unp1.sub(k))
        un.assign(unp1)

    Pres = fd.assemble(PD.para_form)
    for i in range(M):
        assert(dt*np.abs(Pres.sub(i).dat.data[:]).max() < 1.0e-16)


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
                                            solver_parameters  # noqa: E251
                                            =solver_parameters)

    err = fd.Function(V, name="err")
    pun = fd.Function(V, name="pun")
    for i in range(M):
        ssolver.solve()
        un.assign(unp1)
        pun.assign(PD.w_all.sub(i))
        err.assign(un-pun)
        assert(fd.norm(err) < 1.0e-15)


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
                                            solver_parameters  # noqa: E251
                                            =solver_parameters)
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
                      circ="outside", tol=1.0e-12)
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
                                            solver_parameters  # noqa:
                                            =solver_parameters)

    err = fd.Function(V, name="err")
    pun = fd.Function(V, name="pun")
    for i in range(M):
        ssolver.solve()
        un.assign(unp1)
        pun.assign(PD.w_all.sub(i))
        err.assign(un-pun)
        assert(fd.norm(err) < 1.0e-15)


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
                      circ="outside", tol=1.0e-12)
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
                                            solver_parameters  # noqa:
                                            =solver_parameters)
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
        'diagfft': diagfft_options}

    def form_function(u, v):
        return fd.inner(fd.grad(u), fd.grad(v))*fd.dx

    def form_mass(u, v):
        return u*v*fd.dx

    PD = asQ.paradiag(form_function=form_function,
                      form_mass=form_mass, W=V, w0=u0, dt=dt,
                      theta=theta, alpha=alpha, M=M,
                      solver_parameters=solver_parameters,
                      circ="outside", tol=1.0e-12, maxits=1)
    PD.solve(verbose=True)
    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu',
                         'pc_factor_mat_solver_type': 'mumps',
                         'mat_type': 'aij'}
    PDe = asQ.paradiag(form_function=form_function,
                       form_mass=form_mass, W=V, w0=u0, dt=dt,
                       theta=theta, alpha=alpha, M=M,
                       solver_parameters=solver_parameters,
                       circ="outside", tol=1.0e-12, maxits=1)
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
        'diagfft': diagfft_options}

    PD = asQ.paradiag(form_function=form_function,
                      form_mass=form_mass, W=W, w0=w0, dt=dt,
                      theta=theta, alpha=alpha, M=M,
                      solver_parameters=solver_parameters_diag,
                      circ="outside", tol=1.0e-12)
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
                                            solver_parameters  # noqa:
                                            =solver_parameters)
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
