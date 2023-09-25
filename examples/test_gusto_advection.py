from utils.planets import earth
import firedrake as fd
import gusto
import asQ.complex_proxy.mixed as cpx

from gusto.labels import replace_subject, replace_test_function, time_derivative
from gusto.fml.form_manipulation_labelling import all_terms, drop

dt = 0.1
L =1.0
nx = 8

# mesh = fd.UnitSquareMesh(8, 8)

base_mesh = fd.PeriodicIntervalMesh(nx, L)
mesh = fd.ExtrudedMesh(base_mesh, nx, L/nx)

x, y = fd.SpatialCoordinate(mesh)

finit = fd.exp(-((x-0.5*L)**2 + (y-0.5*L)**2))

domain = gusto.Domain(mesh, dt, family='CG', degree=1)

V = domain.spaces("DG")

eqn = gusto.AdvectionEquation(domain, V, "f")

residual = eqn.residual

def form_mass(u, v):
    M = residual.label_map(lambda t: t.has_label(time_derivative),
                           map_if_false=drop)

    M = M.label_map(all_terms, replace_subject(u, idx=0))
    M = M.label_map(all_terms, replace_test_function(v, idx=0))
    return M.form

V = eqn.function_space

u = fd.TrialFunction(V)
v = fd.TestFunction(V)

M = form_mass(u, v)

C = cpx.FunctionSpace(V)

# print(V)
# print()
# print(C)

N = cpx.BilinearForm(C, 1+0j, form_mass)
