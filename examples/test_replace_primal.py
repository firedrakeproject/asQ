import firedrake as fd
import ufl

mesh = fd.UnitIntervalMesh(8)

V = fd.FunctionSpace(mesh, "CG", 1)

def form(u, v):
    return fd.inner(u, v)*fd.dx

u = fd.TrialFunction(V)
v = fd.TestFunction(V)

eqn = form(u, v)

W = V*V

u0, u1 = fd.TrialFunctions(W)
v0, v1 = fd.TestFunctions(W)

new0 = ufl.replace(eqn, {u: u0})
new0 = ufl.replace(new0, {v: v0})

new1 = ufl.replace(eqn, {u: u1})
new1 = ufl.replace(new1, {v: v1})
