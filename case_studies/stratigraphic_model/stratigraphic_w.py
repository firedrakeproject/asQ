from math import pi
import firedrake as fd
from firedrake.petsc import PETSc
from utils.serial import SerialMiniApp

# The mesh
nx = 500
mesh = fd.SquareMesh(nx, nx, 100000, quadrilateral=False, name='meshA')

V = fd.FunctionSpace(mesh, "CG", 1)

# # # === --- initial conditions --- === # # #

x, y = fd.SpatialCoordinate(mesh)

s0 = fd.Function(V, name="scalar_initial")
s0.interpolate(fd.Constant(0.0))


# The sediment movement D
def D(D_c, d):
    return D_c*2*1e6/fd.Constant(fd.sqrt(2*pi))*fd.exp(-1/2*((d-5)/10)**2)


# The carbonate growth L.
def L(G_0, d):
    return G_0*fd.conditional(d > 0, fd.exp(-d/10)/(1 + fd.exp(-50*d)), fd.exp((50-1/10)*d)/(fd.exp(50*d) + 1))


# # # === --- finite element forms --- === # # #


def form_mass(s, q):
    return s*q*fd.dx


D_c = fd.Constant(.002)
G_0 = fd.Constant(.004)
A = fd.Constant(50)
b = 100*fd.tanh(1/20000*(x-50000))


def form_function(s, q, t):
    return D(D_c, A*fd.sin(2*pi*t/500000)-b-s)*fd.inner(fd.grad(s), fd.grad(q))*fd.dx-L(G_0, A*fd.sin(2*pi*t/500000)-b-s)*q*fd.dx


sp = {
    "snes_max_it": 2000,
    "snes_rtol": 1.0e-8,
    "snes_atol": 1.0e-100,
    "snes_stol": 1.0e-100,
    "snes_monitor": None,
    "snes_linesearch_type": "l2",
    "snes_converged_reason": None,
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"
}

dt = 1000
theta = 0.5
miniapp = SerialMiniApp(dt, theta,
                        s0,
                        form_mass,
                        form_function,
                        sp)


# Solve of the required number of timesteps j
def preproc(miniapp, step, time):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'### === --- Calculating solution at step {step + 1} and time {time} --- === ###')
    PETSc.Sys.Print('')


j = 4
miniapp.solve(j, preproc=preproc)

u = fd.Function(V, name="Ic")
u.assign(miniapp.w1)

with fd.CheckpointFile("example.h5", 'w') as afile:
    afile.save_mesh(mesh)  # optional
    afile.save_function(u)
