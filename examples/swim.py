import firedrake as fd
from petsc4py import PETSc
import numpy as np
import asQ
PETSc.Sys.popErrorHandler()

# get command arguments
import argparse
parser = argparse.ArgumentParser(description='Williamson 5 testcase for approximate Schur complement solver.')
parser.add_argument('--base_level', type=int, default=1, help='Base refinement level of icosahedral grid for MG solve. Default 1.')
parser.add_argument('--ref_level', type=int, default=3, help='Refinement level of icosahedral grid. Default 3.')
#parser.add_argument('--nsteps', type=int, default=10, help='Number of timesteps. Default 10.')
parser.add_argument('--alpha', type=float, default=0.0001, help='Circulant coefficient. Default 0.0001.')
parser.add_argument('--dt', type=float, default=0.05, help='Timestep in hours. Default 0.05.')
parser.add_argument('--filename', type=str, default='w5diag')
parser.add_argument('--coords_degree', type=int, default=1, help='Degree of polynomials for sphere mesh approximation.')
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space).')
parser.add_argument('--kspschur', type=int, default=1, help='Number of KSP iterations on the Schur complement.')
parser.add_argument('--kspmg', type=int, default=3, help='Number of KSP iterations in the MG levels.')
parser.add_argument('--tlblock', type=str, default='patch', help='Solver for the velocity-velocity block. mg==Multigrid with patchPC, lu==direct solver with MUMPS, patch==just do a patch smoother. Default is patch')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

# some domain, parameters and FS setup
R0 = 6371220.
H = fd.Constant(5960.)
base_level = args.base_level
nrefs = args.ref_level - base_level
filename = args.filename
deg = args.coords_degree
distribution_parameters = {"partition": True, "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)}

nspatial_domains = 1
ensemble = fd.Ensemble(fd.COMM_WORLD, nspatial_domains)

if args.tlblock == "mg":
    basemesh = fd.IcosahedralSphereMesh(radius=R0,
                                        refinement_level=base_level,
                                        degree=deg,
                                        distribution_parameters=
                                        distribution_parameters,
                                        comm=ensemble.comm)
    mh = fd.MeshHierarchy(basemesh, nrefs)
    for mesh in mh:
        x = fd.SpatialCoordinate(mesh)
        mesh.init_cell_orientations(x)
    mesh = mh[-1]
else:
    mesh = fd.IcosahedralSphereMesh(radius=R0,
                                    refinement_level=args.ref_level,
                                    degree=deg,
                                    distribution_parameters=
                                    distribution_parameters,
                                    comm=ensemble.comm)
    x = fd.SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)

R0 = fd.Constant(R0)
x = fd.SpatialCoordinate(mesh)

outward_normals = fd.CellNormal(mesh)

def perp(u):
    return fd.cross(outward_normals, u)


degree = args.degree
V1 = fd.FunctionSpace(mesh, "BDM", degree+1)
V2 = fd.FunctionSpace(mesh, "DG", degree)
V0 = fd.FunctionSpace(mesh, "CG", degree+2)
W = fd.MixedFunctionSpace((V1, V2))

Omega = fd.Constant(7.292e-5)  # rotation rate
f = 2*Omega*x[2]/fd.Constant(R0)  # Coriolis parameter
g = fd.Constant(9.8)  # Gravitational constant
b = fd.Function(V2, name="Topography")
# D = eta + b


def both(u):
    return 2*fd.avg(u)


def form_function(u, h, v, q):
    K = 0.5*fd.inner(u, u)
    n = fd.FacetNormal(mesh)
    uup = 0.5 * (fd.dot(u, n) + abs(fd.dot(u, n)))
    Upwind = 0.5 * (fd.sign(fd.dot(u, n)) + 1)

    eqn = (
        fd.inner(v, f*perp(u))*fd.dx
        - fd.inner(perp(fd.grad(fd.inner(v, perp(u)))), u)*fd.dx
        + fd.inner(both(perp(n)*fd.inner(v, perp(u))),
                   both(Upwind*u))*fd.dS
        - fd.div(v)*(g*(h + b) + K)*fd.dx
        - fd.inner(fd.grad(q), u)*h*fd.dx
        + fd.jump(q)*(uup('+')*h('+')
                      - uup('-')*h('-'))*fd.dS
    )
    return eqn


def form_mass(u, h, v, q):
    return fd.inner(u, v)*fd.dx + h*q*fd.dx

# Parameters for the diag
sparameters = {
    "ksp_type": "preonly",
    'pc_python_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps'}

solver_parameters_diag = {
    "snes_linesearch_type": "basic",
    'snes_monitor': None,
    'snes_converged_reason': None,
    'mat_type': 'matfree',
    'ksp_type': 'gmres',
    'ksp_monitor': None,
    'pc_type': 'python',
    'pc_python_type': 'asQ.DiagFFTPC'}

M = [2, 2, 2, 2]
for i in range(sum(M)): # should this be sum(M) or max(M)?
    solver_parameters_diag["diagfft_"+str(i)+"_"] = sparameters

dt = 60*60*args.dt
t = 0.

u_0 = 20.0  # maximum amplitude of the zonal wind [m/s]
u_max = fd.Constant(u_0)
u_expr = fd.as_vector([-u_max*x[1]/R0, u_max*x[0]/R0, 0.0])
eta_expr = - ((R0 * Omega * u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R0*R0)))/g
# W = V1 * V2
w0 = fd.Function(W)
un, etan = w0.split()
un.project(u_expr)
etan.project(eta_expr)

# Topography.
rl = fd.pi/9.0
lambda_x = fd.atan_2(x[1]/R0, x[0]/R0)
lambda_c = -fd.pi/2.0
phi_x = fd.asin(x[2]/R0)
phi_c = fd.pi/6.0
minarg = fd.Min(pow(rl, 2),
                pow(phi_x - phi_c, 2) + pow(lambda_x - lambda_c, 2))
bexpr = 2000.0*(1 - fd.sqrt(minarg)/rl)
b.interpolate(bexpr)

alpha = args.alpha
theta = 0.5

PD = asQ.paradiag(ensemble=ensemble,
                  form_function=form_function,
                  form_mass=form_mass, W=W, w0=w0,
                  dt=dt, theta=theta,
                  alpha=alpha,
                  M=M, solver_parameters=solver_parameters_diag,
                  circ="quasi",
                  jac_average="newton", tol=1.0e-6, maxits=None,
                  ctx={}, block_mat_type="aij")
PD.solve()

u_out = fd.Function(V1,name='velocity')
h_out = fd.Function(V2,name='depth')

# write output:
r = PD.ensemble.ensemble_comm.rank
if r == len(M) - 1:
    file0 = fd.File("output/"+filename+".pvd", comm=ensemble.comm)
    u0, h0, u1, h1 = PD.w_all.split()
    u_out.assign(u1)
    h_out.assign(h1)
    file0.write(u_out, h_out)
