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

    def function_space(self, mesh):
        Ve = fd.FiniteElement("CG", mesh.ufl_cell(), self.degree)
        return fd.FunctionSpace(mesh, Ve)


if __name__ == "__main__":

    for m in [1, 2, 4]:
        N = m*500
        problem = Problem(N, 1)
        mesh = problem.mesh()
        n = fd.FacetNormal(mesh)
        Z = problem.function_space(mesh)
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

        dt = fd.Constant(.001/m)
        A = fd.Constant(50)
        t = fd.Constant(0.0)
        s0 = fd.Function(Z)
        s0.interpolate(fd.sin((x+2*y)))
        s_exact = fd.Function(Z)
#        print(fd.norm(s_exact))
        s = fd.Function(Z)
        q = fd.TestFunction(Z)
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
        d_exact = l-b-s_exact

        D = 2*D_c/fd.sqrt(2*pi)*fd.exp(-1/2*((d-5)/10)**2)
        D_exact = 2*D_c/fd.sqrt(2*pi)*fd.exp(-1/2*((d_exact-5)/10)**2)

        G = G_0*fd.conditional(d > 0, fd.exp(-d/10)/(1 + fd.exp(-50*d)), fd.exp((50-1/10)*d)/(fd.exp(50*d) + 1))
        G_exact = G_0*fd.conditional(d_exact > 0, fd.exp(-d_exact/10)/(1 + fd.exp(-50*d_exact)), fd.exp((50-1/10)*d_exact)/(fd.exp(50*d_exact) + 1))

        Rhs = fd.div(D_exact*fd.grad(s_exact)) + G_exact

        F = D*fd.inner(fd.grad(s), fd.grad(q))*fd.dx - G*q*fd.dx - Rhs*q*fd.dx - D_exact*fd.inner(fd.inner(fd.grad(s_exact), n), q)*fd.ds

        F_euler = (fd.inner(s, q)*fd.dx - fd.inner(s0, q)*fd.dx + dt*(F))
        nvproblem = fd.NonlinearVariationalProblem(F_euler, s)
        solver = fd.NonlinearVariationalSolver(nvproblem, solver_parameters=sp)

        for i in range(1):
            t.assign(t+dt)
            s_exact = fd.Function(Z).interpolate((t+1)*fd.sin((x+2*y)))
            solver.solve()
            A = fd.errornorm(s, s_exact)/fd.norm(s_exact)
            PETSc.Sys.Print("E :%s" % A)
            s0.assign(s)
