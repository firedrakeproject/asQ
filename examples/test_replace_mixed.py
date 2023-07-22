import firedrake as fd
import ufl

mesh = fd.UnitSquareMesh(8, 8)

Vu = fd.FunctionSpace(mesh, "RT", 2)
Vh = fd.FunctionSpace(mesh, "DG", 1)

V = Vu * Vh

def form(u, h, v, q):
    return (fd.inner(u, v) + fd.inner(h, q))*fd.dx

u, h = fd.split(fd.Function(V))
v, q = fd.TestFunctions(V)

eqn = form(u, h, v, q)

# W = Vu * Vh * Vu * Vh
# u0, h0, u1, h1 = fd.split(fd.Function(W))
# v0, q0, v1, q1 = fd.TestFunctions(W)

W = Vu * Vu * Vh * Vh
u0, u1, h0, h1 = fd.split(fd.Function(W))
v0, v1, q0, q1 = fd.TestFunctions(W)

# first eqn

new0 = ufl.replace(eqn, {u: u0})
new0 = ufl.replace(new0, {h: h0})

new0 = ufl.replace(new0, {v: v0})
new0 = ufl.replace(new0, {q: q0})

# second eqn

new1 = ufl.replace(eqn, {u: u1})
new1 = ufl.replace(new1, {h: h1})

new1 = ufl.replace(new1, {v: v1})
new1 = ufl.replace(new1, {q: q1})
