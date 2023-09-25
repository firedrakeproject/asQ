import firedrake as fd
import gusto

from gusto import Term
from gusto.fml import replace_subject, replace_test_function, replace_trial_function
from gusto.labels import time_derivative, prognostic
from gusto.fml import all_terms, drop

from firedrake.formmanipulation import split_form

# set up the equation
mesh = fd.UnitIcosahedralSphereMesh(refinement_level=1, degree=1)
swe_params = gusto.ShallowWaterParameters(H=1, g=1, Omega=1)
domain = gusto.Domain(mesh, dt=1, family='BDM', degree=1)
eqn = gusto.ShallowWaterEquations(domain, swe_params)

residual = eqn.residual

print()
print("Original mass matrix:")
print(residual.label_map(lambda t: t.has_label(time_derivative),
                         map_if_false=drop).form)
print()

# function to replace the test function and subject of the time-derivative with different test function and subject.
def form_mass(u, h, v, q):
    M = residual.label_map(lambda t: t.has_label(time_derivative),
                           map_if_false=drop)

    M = M.label_map(all_terms, replace_subject((u, h)))
    M = M.label_map(all_terms, replace_test_function((v, q)))
    return M.form

V = eqn.function_space
Vu = V.subfunctions[0]
Vh = V.subfunctions[1]

# the usual FunctionSpace

u, h = fd.split(fd.Function(V))
v, q = fd.TestFunctions(V)

M = form_mass(u, h, v, q)
print("Simple mass matrix replacement:")
print(u, h, v, q)
print(M)
print()

# A FunctionSpace for multiple timesteps (u0, h0, u1, h1)

W0 = V*V
u0, h0, u1, h1 = fd.split(fd.Function(W0))
v0, q0, v1, q1 = fd.TestFunctions(W0)

Mt = form_mass(u1, h1, v1, q1)
print("Multiple timesteps mass matrix replacement:")
print(u1, h1, v1, q1)
print(Mt)
print()

# A FunctionSpace to proxy the complex problem (real u, imag u, real h, imag h)

W1 = fd.MixedFunctionSpace((Vu, Vu, Vh, Vh))
ur, ui, hr, hi = fd.split(fd.Function(W1))
vr, vi, qr, qi = fd.TestFunctions(W1)

Mc = form_mass(ui, hi, vi, qi)
print("Complex mass matrix replacement:")
print(ui, hi, vi, qi)
print(Mc)
print()
