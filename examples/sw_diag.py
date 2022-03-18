import firedrake as fd
from petsc4py import PETSc
import asQ
PETSc.Sys.popErrorHandler()

# get command arguments
import argparse
parser = argparse.ArgumentParser(description='Williamson 5 testcase for approximate Schur complement solver.')
parser.add_argument('--base_level', type=int, default=1, help='Base refinement level of icosahedral grid for MG solve. Default 1.')
parser.add_argument('--ref_level', type=int, default=2, help='Refinement level of icosahedral grid. Default 3.')
parser.add_argument('--nsteps', type=int, default=10, help='Number of timesteps. Default 10.')
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
    print(args)

# some domain, parameters and FS setup
R0 = 6371220.
H = fd.Constant(5960.)
base_level = args.base_level
nrefs = args.ref_level - base_level
name = args.filename
deg = args.coords_degree
distribution_parameters = {"partition": True, "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)}

if args.tlblock == "mg":
    basemesh = fd.IcosahedralSphereMesh(radius=R0,
                                        refinement_level=base_level, degree=deg,
                                        distribution_parameters=distribution_parameters)
    mh = fd.MeshHierarchy(basemesh, nrefs)
    for mesh in mh:
        x = fd.SpatialCoordinate(mesh)
        mesh.init_cell_orientations(x)
    mesh = mh[-1]
else:
    mesh = fd.IcosahedralSphereMesh(radius=R0,
                                    refinement_level=args.ref_level, degree=deg,
                                    distribution_parameters=distribution_parameters)
    x = fd.SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)
R0 = fd.Constant(R0)
cx, cy, cz = fd.SpatialCoordinate(mesh)

outward_normals = fd.CellNormal(mesh)

M = args.nsteps


def perp(u):
    return fd.cross(outward_normals, u)


degree = args.degree
V1 = fd.FunctionSpace(mesh, "BDFM", degree+1)
V2 = fd.FunctionSpace(mesh, "DG", degree)
V0 = fd.FunctionSpace(mesh, "CG", degree+2)
W = fd.MixedFunctionSpace((V1, V2))

u, eta = fd.TrialFunctions(W)
v, phi = fd.TestFunctions(W)

Omega = fd.Constant(7.292e-5)  # rotation rate
f = 2*Omega*cz/fd.Constant(R0)  # Coriolis parameter
g = fd.Constant(9.8)  # Gravitational constant
b = fd.Function(V2, name="Topography")
c = fd.sqrt(g*H)
dT = fd.Constant(0.)
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


class HelmholtzPC(fd.AuxiliaryOperatorPC):

    def form(self, pc, v, u):
        _, P = pc.getOperators()
        context = P.getPythonContext()

        vr = v[0]
        vi = v[1]
        ur = u[0]
        ui = u[1]

        eta = fd.Constant(context.appctx.get("eta", 10.))
        D1r = context.appctx.get("D1r", None)
        assert(D1r)
        D1i = context.appctx.get("D1i", None)
        assert(D1i)
        sr = context.appctx.get("sinvr", None)
        assert(sr)
        si = context.appctx.get("sinvi", None)
        assert(si)

        def get_laplace(q, phi):
            h = fd.avg(fd.CellVolume(mesh))/fd.FacetArea(mesh)
            mu = eta/h
            n = fd.FacetNormal(mesh)
            ad = (- fd.inner(2 * fd.avg(phi*n),
                             fd.avg(fd.grad(q)))
                  - fd.inner(fd.avg(fd.grad(phi)),
                             2 * fd.avg(q*n))
                  + mu * fd.inner(2 * fd.avg(phi*n),
                                  2 * fd.avg(q*n))) * fd.dS
            ad += fd.inner(fd.grad(q), fd.grad(phi)) * fd.dx
            return ad

        D1u_r = D1r*ur - D1i*ui
        D1u_i = D1i*ur + D1r*ui
        su_r = sr*ur - si*ui
        su_i = si*ur + sr*ui

        a = vr * D1u_r * fd.dx + get_laplace(vr, g*H*su_r)
        a += vi * D1u_i * fd.dx + get_laplace(vi, g*H*su_i)

        # Returning None as bcs
        return (a, None)


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
    "ksp_max_it": args.kspschur,
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

topleft_ILU = {
    "ksp_type": "gmres",
    "ksp_max_it": 3,
    "pc_type": "python",
    "pc_python_type": "firedrake.AssembledPC",
    "assembled_pc_type": "ilu",
    "assembled_pc_factor_mat_solver_type": "petsc"
}

topleft_MG = {
    "ksp_type": "preonly",
    "ksp_max_it": 3,
    "pc_type": "mg",
    "mg_coarse_ksp_type": "preonly",
    "mg_coarse_pc_type": "python",
    "mg_coarse_pc_python_type": "firedrake.AssembledPC",
    "mg_coarse_assembled_pc_type": "lu",
    "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
    "mg_levels_ksp_type": "gmres",
    "mg_levels_ksp_max_it": args.kspmg,
    "mg_levels_pc_type": "python",
    "mg_levels_pc_python_type": "firedrake.PatchPC",
    "mg_levels_patch_pc_patch_save_operators": True,
    "mg_levels_patch_pc_patch_partition_of_unity": False,
    "mg_levels_patch_pc_patch_sub_mat_type": "seqaij",
    "mg_levels_patch_pc_patch_construct_type": "star",
    "mg_levels_patch_pc_patch_multiplicative": False,
    "mg_levels_patch_pc_patch_symmetrise_sweep": False,
    "mg_levels_patch_pc_patch_construct_dim": 0,
    "mg_levels_patch_sub_ksp_type": "preonly",
    "mg_levels_patch_sub_pc_type": "lu",
}

topleft_MGs = {
    "ksp_type": "preonly",
    "ksp_max_it": 3,
    "pc_type": "mg",
    "mg_coarse_ksp_type": "preonly",
    "mg_coarse_pc_type": "python",
    "mg_coarse_pc_python_type": "firedrake.AssembledPC",
    "mg_coarse_assembled_pc_type": "lu",
    "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
    "mg_levels_ksp_type": "gmres",
    "mg_levels_ksp_max_it": args.kspmg,
    "mg_levels_pc_type": "python",
    "mg_levels_pc_python_type": "firedrake.AssembledPC",
    "mg_levels_assembled_pc_type": "python",
    "mg_levels_assembled_pc_python_type": "firedrake.ASMStarPC",
    "mg_levels_assembled_pc_star_backend": "tinyasm",
    "mg_levels_assmbled_pc_star_construct_dim": 0
}

topleft_smoother = {
    "ksp_type": "gmres",
    "ksp_max_it": 3,
    "pc_type": "python",
    "pc_python_type": "firedrake.PatchPC",
    "patch_pc_patch_save_operators": True,
    "patch_pc_patch_partition_of_unity": False,
    "patch_pc_patch_sub_mat_type": "seqaij",
    "patch_pc_patch_construct_type": "star",
    "patch_pc_patch_multiplicative": False,
    "patch_pc_patch_symmetrise_sweep": False,
    "patch_pc_patch_construct_dim": 0,
    "patch_sub_ksp_type": "preonly",
    "patch_sub_pc_type": "lu",
}

if args.tlblock == "mg":
    sparameters["fieldsplit_0"] = topleft_MG
elif args.tlblock == "patch":
    sparameters["fieldsplit_0"] = topleft_smoother
elif args.tlblock == "ilu":
    sparameters["fieldsplit_0"] = topleft_ILU
else:
    assert(args.tlblock == "lu")
    sparameters["fieldsplit_0"] = topleft_LU

lu_parameters = {
    "pc_type": "python",
    "pc_python_type": "firedrake.AssembledPC",
    "assembled_mat_type": "aij",
    "assembled_pc_type": "lu",
    "assembled_pc_factor_mat_solver_type": "mumps"
}

solver_parameters_diag = {
    'snes_monitor': None,
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    "ksp_gmres_modifiedgramschmidt": None,
    'ksp_max_it': 60,
    'ksp_rtol': 1.0e-5,
    'ksp_atol': 1.0e-30,
    'pc_type': 'python',
    'pc_python_type': 'asQ.DiagFFTPC'}

for i in range(M):
    solver_parameters_diag["diagfft_"+str(i)+"_"] = sparameters

dt = 60*60*args.dt
# dT.assign(dt)
t = 0.

x = fd.SpatialCoordinate(mesh)
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

PD = asQ.paradiag(form_function=form_function,
                  form_mass=form_mass, W=W, w0=w0, dt=dt,
                  theta=theta, alpha=alpha, M=M,
                  solver_parameters=solver_parameters_diag,
                  circ="quasi")
PD.solve()


# write output:
file0 = fd.File("output/output1.pvd")
pun = fd.Function(W, name="pun")
puns = pun.split()

for i in range(M):
    walls = PD.w_all.split()[2 * i:2 * i + 2]
    for k in range(2):
        puns[k].assign(walls[k])
    u_out = puns[0]
    h_out = puns[1]
    file0.write(u_out, h_out)


# # write output:
# r = PD.ensemble.ensemble_comm.rank
# if r == len(M) - 1:
#     file0 = fd.File("output/output1.pvd", comm=ensemble.comm)
#     u0, h0, u1, h1 = PD.w_all.split()
#     file0.write(u1, h1)