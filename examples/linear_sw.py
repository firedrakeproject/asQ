import firedrake as fd
from petsc4py import PETSc
import asQ
import numpy as np

PETSc.Sys.popErrorHandler()

# some domain, parameters and FS setup
R0 = 6371220.
H = fd.Constant(5960.)

# set up the ensemble communicator for space-time parallelism
nspatial_domains = 2
ensemble = fd.Ensemble(fd.COMM_WORLD, nspatial_domains)
mesh = fd.IcosahedralSphereMesh(radius=R0,
                                refinement_level=3,
                                degree=2,
                                comm=ensemble.comm)
x = fd.SpatialCoordinate(mesh)
mesh.init_cell_orientations(x)
R0 = fd.Constant(R0)
cx, cy, cz = fd.SpatialCoordinate(mesh)

outward_normals = fd.CellNormal(mesh)

def perp(u):
    return fd.cross(outward_normals, u)


degree = 1
V1 = fd.FunctionSpace(mesh, "BDFM", degree+1)
V2 = fd.FunctionSpace(mesh, "DG", degree)
V0 = fd.FunctionSpace(mesh, "CG", degree+2)
W = fd.MixedFunctionSpace((V1, V2))

u, eta = fd.TrialFunctions(W)
v, phi = fd.TestFunctions(W)

Omega = fd.Constant(7.292e-5)  # rotation rate
f = 2*Omega*cz/fd.Constant(R0)  # Coriolis parameter
g = fd.Constant(9.8)  # Gravitational constant
dT = fd.Constant(0.)

def form_function(u, h, v, q):
    n = fd.FacetNormal(mesh)

    eqn = (
        fd.inner(v, f*perp(u))*fd.dx
        - fd.div(v)*g*h*fd.dx
        - q*H*fd.div(u)*fd.dx
    )
    return eqn


def form_mass(u, h, v, q):
    return fd.inner(u, v)*fd.dx + h*q*fd.dx

# Parameters for the diag
sparameters = {
    "ksp_type": "preonly",
    'pc_python_type': 'firedrake.HybridizationPC',
    'hybridization': {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type':'mumps'}
}

solver_parameters_diag = {
    'snes_monitor': None,
    'mat_type': 'matfree',
    'ksp_type': 'preonly',
    'pc_type': 'python',
    'pc_python_type': 'asQ.DiagFFTPC'}

M = [2, 2, 2, 2]
for i in range(np.sum(M)):
    solver_parameters_diag["diagfft_"+str(i)+"_"] = sparameters
    
dt = 60*60*3600
dT.assign(dt)
t = 0.

# W = V1 * V2
w0 = fd.Function(W)
un, etan = w0.split()
rl = fd.pi/9.0
lambda_x = fd.atan_2(x[1]/R0, x[0]/R0)
lambda_c = -fd.pi/2.0
phi_x = fd.asin(x[2]/R0)
phi_c = fd.pi/6.0
minarg = fd.Min(pow(rl, 2),
                pow(phi_x - phi_c, 2) + pow(lambda_x - lambda_c, 2))
bexpr = 2000.0*(1 - fd.sqrt(minarg)/rl)
etan.project(bexpr)

alpha = 1.0e5
theta = 0.5

PD = asQ.paradiag(ensemble=ensemble,
                  form_function=form_function,
                  form_mass=form_mass, W=W, w0=w0,
                  dt=dt, theta=theta,
                  alpha=alpha,
                  M=M, solver_parameters=solver_parameters_diag,
                  circ="none",
                  jac_average="newton", tol=1.0e-6, maxits=None,
                  ctx={}, block_mat_type="aij")
PD.solve()
