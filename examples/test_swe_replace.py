from utils import shallow_water as swe
from utils.planets import earth
import firedrake as fd
import ufl

from functools import partial

mesh = swe.create_mg_globe_mesh(coords_degree=1)

W = swe.default_function_space(mesh)
Vu, Vh = W.subfunctions[:]

g = earth.Gravity
b = fd.Constant(0)
f = fd.Constant(0)
t = fd.Constant(0)

form_mass = lambda *args: swe.nonlinear.form_mass(mesh, *args)
form_function = lambda *args: swe.nonlinear.form_function(mesh, g, b, f, *args, t)

form = form_function

u, h = fd.TrialFunctions(W)
v, q = fd.TestFunctions(W)

M = form(u, h, v, q)

# replace from W

u0, h0 = fd.TrialFunctions(W)
v0, q0 = fd.TestFunctions(W)

# subject
M0 = ufl.replace(M,  {u:u0})
M0 = ufl.replace(M0, {h:h0})

# tests
M0 = ufl.replace(M0, {v:v0})
M0 = ufl.replace(M0, {q:q0})

# replace from (W * W)

WW = W * W

u1, h1, u_, h_ = fd.TrialFunctions(WW)
v1, q1, v_, q_ = fd.TestFunctions(WW)

# subject
M1 = ufl.replace(M,  {u:u1})
M1 = ufl.replace(M1, {h:h1})

# tests
M1 = ufl.replace(M1, {v:v1})
M1 = ufl.replace(M1, {q:q1})

# replace from (Vu * Vu * Vh * Vh)

VV = Vu * Vu * Vh * Vh

u2, u_, h2, h_ = fd.TrialFunctions(VV)
v2, v_, q2, q_ = fd.TestFunctions(VV)

# subject
M2 = ufl.replace(M,  {u:u2})
M2 = ufl.replace(M2, {h:h2})

# tests
M2 = ufl.replace(M2, {v:v2})
M2 = ufl.replace(M2, {q:q2})
