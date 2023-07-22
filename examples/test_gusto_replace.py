from utils.planets import earth
import firedrake as fd
import gusto
import asQ.complex_proxy.mixed as cpx

from gusto.labels import replace_subject, replace_test_function, replace_trial_function, time_derivative
from gusto.fml.form_manipulation_labelling import all_terms, drop

mesh = fd.IcosahedralSphereMesh(radius=earth.radius,
                                refinement_level=2,
                                degree=1)

swe_params = gusto.ShallowWaterParameters(H=10e4,
                                          g=earth.gravity,
                                          Omega=earth.omega)

dt = 1

domain = gusto.Domain(mesh, dt, 'BDM', degree=1)

eqn = gusto.ShallowWaterEquations(domain, swe_params)

residual = eqn.residual

def form_mass(u, h, v, q):
    M = residual.label_map(lambda t: t.has_label(time_derivative),
                           map_if_false=drop)

    M = M.label_map(all_terms, replace_subject(u, idx=0))
    M = M.label_map(all_terms, replace_subject(h, idx=1))

    M = M.label_map(all_terms, replace_test_function(v, idx=0))
    M = M.label_map(all_terms, replace_test_function(q, idx=1))
    return M.form

V = eqn.function_space
Vu = V.subfunctions[0]
Vh = V.subfunctions[1]

u, h = fd.TrialFunctions(V)
v, q = fd.TestFunctions(V)

M = form_mass(u, h, v, q)

# W = V*V
# u0, h0, u1, h1 = fd.split(fd.Function(W))
# v0, q0, v1, q1 = fd.TestFunctions(W)

W = fd.MixedFunctionSpace((Vu, Vu, Vh, Vh))
u0, u1, h0, h1 = fd.split(fd.Function(W))
v0, v1, q0, q1 = fd.TestFunctions(W)
 
M0 = form_mass(u0, h0, v0, q0)
# M1 = form_mass(u1, h1, v1, q1)
