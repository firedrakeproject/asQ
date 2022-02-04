import firedrake as fd
from petsc4py import PETSc
import asQ

PETSc.Sys.popErrorHandler()

# some domain, parameters and FS setup
R0 = 6371220.
H = fd.Constant(5960.)

mesh = fd.IcosahedralSphereMesh(radius=R0,
                                refinement_level=3, degree=2)
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
    "mat_type": "matfree",
    "ksp_type": "fgmres",
    "ksp_max_it": 50,
    "ksp_gmres_modifiedgramschmidt": None,
    "ksp_rtol": 1e-8,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",
    "pc_fieldsplit_off_diag_use_amat": True,
}

bottomright = {
    "ksp_type": "gmres",
    "ksp_max_it": 3,
    "pc_type": "python",
    "pc_python_type": "__main__.HelmholtzPC",
    "aux_pc_type": "lu"
}

sparameters["fieldsplit_1"] = bottomright

topleft_LU = {
    "ksp_type": "preonly",
    "pc_type": "python",
    "pc_python_type": "firedrake.AssembledPC",
    "assembled_pc_type": "lu",
    "assembled_pc_factor_mat_solver_type": "mumps"
}

sparameters["fieldsplit_0"] = topleft_LU

lu_parameters = {
    "pc_type": "python",
    "pc_python_type": "firedrake.AssembledPC",
    "assembled_mat_type": "aij",
    "assembled_pc_type": "lu",
    "assembled_pc_factor_mat_solver_type": "mumps"
}

appctx = {}
contextfn = asQ.set_context(appctx)

solver_parameters_diag = {
    'snes_monitor': None,
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    "ksp_gmres_modifiedgramschmidt": None,
    'ksp_max_it': 60,
    'ksp_rtol': 1.0e-5,
    'ksp_atol': 1.0e-30,
    'pc_type': 'python',
    'pc_python_type': 'asQ.DiagFFTPC',
    'pc_context':'__main__.contextfn'}

for i in range(M):
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

PD = asQ.paradiag(form_function=form_function,
                  form_mass=form_mass, W=W, w0=w0, dt=dt,
                  theta=theta, alpha=alpha, M=M,
                  solver_parameters=solver_parameters_diag,
                  circ="quasi")
PD.solve()
