import firedrake as fd
from firedrake.petsc import PETSc
from math import pi

# A toy equation for Stratigraphic modelling.


class Problem(object):
    def __init__(self, N, degree):
        super().__init__()
        self.N = N
        self.degree = degree

    def mesh(self):
        mesh = fd.RectangleMesh(self.N, self.N, 100, 100, quadrilateral=False)
        return mesh

#    Initial condition on thickness of the carbonate layer.
    def initial_condition(self, Z):
        (x, y) = fd.SpatialCoordinate(Z.mesh())
        s = fd.Function(Z)
        s.interpolate(fd.Constant(0))
        return s

    def function_space(self, mesh):
        Ve = fd.FiniteElement("CG", mesh.ufl_cell(), self.degree)
        return fd.FunctionSpace(mesh, Ve)


if __name__ == "__main__":
    N = 1000
    problem = Problem(N, 1)
    mesh = problem.mesh()
    Z = problem.function_space(mesh)
    s0 = problem.initial_condition(Z)
    s = fd.Function(Z)
    q = fd.TestFunction(Z)
    x, y = fd.SpatialCoordinate(mesh)
    PETSc.Sys.Print("Z.dim():%s" % Z.dim())
    sp = {
        "snes_max_it": 2000,
        "snes_atol": 1.0e-8,
        "snes_rtol": 1.0e-8,
        "snes_monitor": None,
        "snes_linesearch_type": "l2",
        "snes_converged_reason": None,
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"
    }

    dt = fd.Constant(1000)
    A = fd.Constant(50)
    t = fd.Constant(0)
# D_c is a constant diffusion parameter.
    D_c = fd.Constant(2e-3)
# G_0 is a growth constant for the carbonate.
    G_0 = fd.Constant(4e-3)
# b is the seabed.
    b = 100*fd.tanh((x-50)/20)
# l is the sea level.
    l = A*fd.sin(2*pi*t/500000)
# d is the water depth.
    d = l-b-s
    D = 2*D_c/fd.sqrt(2*pi)*fd.exp(-1/2*((d-5)/10)**2)
    G = G_0*fd.conditional(d > 0, fd.exp(-d/10)/(1 + fd.exp(-50*d)), fd.exp((50-1/10)*d)/(fd.exp(50*d) + 1))
    F = D*fd.inner(fd.grad(s), fd.grad(q))*fd.dx - G*q*fd.dx
    F_euler = (fd.inner(s, q)*fd.dx - fd.inner(s0, q)*fd.dx + dt*(F))
    nvproblem = fd.NonlinearVariationalProblem(F_euler, s)
    solver = fd.NonlinearVariationalSolver(nvproblem, solver_parameters=sp)
    outfile = fd.File("s.pvd")
    while (float(t) < 128*float(dt)):
        t.assign(float(t+dt))
        solver.solve()
        s0.assign(s)
        outfile.write(s)
