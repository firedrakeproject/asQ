import firedrake as fd
import ufl

mesh = fd.UnitSquareMesh(8, 8)

Vu = fd.FunctionSpace(mesh, "RT", 2)
Vh = fd.FunctionSpace(mesh, "DG", 1)

V = Vu * Vh

def form(sigma, u, tau, v):
    return (fd.dot(sigma, tau) + fd.div(tau)*u + fd.div(sigma)*v)*fd.dx
    # return (fd.inner(u, v) + fd.inner(h, q))*fd.dx

sigma, u = fd.split(fd.Function(V))
tau, v = fd.TestFunctions(V)

eqn = form(sigma, u, tau, v)

# W = Vu * Vh * Vu * Vh
# sigma0, u0, sigma1, u1 = fd.split(fd.Function(W))
# tau0, v0, tau1, v1 = fd.TestFunctions(W)

W = Vu * Vu * Vh * Vh
sigma0, sigma1, u0, u1 = fd.split(fd.Function(W))
tau0, tau1, v0, v1 = fd.TestFunctions(W)

# first eqn

new0 = ufl.replace(eqn, {tau: tau0})
new0 = ufl.replace(new0, {v: v0})

new0 = ufl.replace(new0, {sigma: sigma0})
new0 = ufl.replace(new0, {u: u0})

# second eqn

new1 = ufl.replace(eqn, {tau: tau1})
new1 = ufl.replace(new1, {v: v1})

new1 = ufl.replace(new1, {sigma: sigma1})
new1 = ufl.replace(new1, {u: u1})
