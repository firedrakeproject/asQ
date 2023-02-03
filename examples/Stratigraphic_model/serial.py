from firedrake import *
from firedrake.petsc import PETSc
print = lambda x: PETSc.Sys.Print(x)


class Problem(object):
    def __init__(self, N, degree):
        super().__init__()
        self.N = N
        self.degree = degree

    def mesh(self):
        mesh = RectangleMesh(self.N, self.N, 100, 100, quadrilateral=False)
        return mesh

    def initial_condition(self, Z):
        (x, y) = SpatialCoordinate(Z.mesh())
        s = Function(Z)
        s.interpolate(Constant(0))
        return s

    def function_space(self, mesh):
        Ve = FiniteElement("CG", mesh.ufl_cell(), self.degree)
        return FunctionSpace(mesh, Ve)


if __name__ == "__main__":
    N = 1000
    problem = Problem(N, 1)
    mesh = problem.mesh()
    Z = problem.function_space(mesh)
    s0 = problem.initial_condition(Z)
    s = Function(Z)
    q = TestFunction(Z)
    x, y = SpatialCoordinate(mesh)
    print("Z.dim():%s" % Z.dim())

    sp = {
        "snes_max_it": 100,
        "snes_atol": 1.0e-8,
        "snes_rtol": 1.0e-8,
        "snes_monitor": None,
        "snes_converged_reason": None,
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    A = Constant(50)
    dt = Constant(1000)
    t = Constant(0)
    D_c = Constant(2e-3)
    G_0 = Constant(4e-3)
    b = 100*tanh((x-50)/20)
    l = A*sin(2*pi*t/500000)
    d = l-b-s
    D = 2*D_c/sqrt(2*pi)*exp(-1/2*((d-5)/10)**2)
    G = G_0/(1+exp(50*d))*exp(-d/10)
    F = D*inner(grad(s), grad(q))*dx - G*q*dx
    F_euler = (inner((s), (q))*dx - inner((s0), (q))*dx + dt*(F))
    nvproblem = NonlinearVariationalProblem(F_euler, s)
    solver = NonlinearVariationalSolver(nvproblem, solver_parameters=sp)

    while (float(t) < 2*float(dt)):
        t.assign(float(t+dt))
        solver.solve()
        s0.assign(s)
