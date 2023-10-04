import firedrake as fd
from firedrake.petsc import PETSc
from math import pi
from utils.diagnostics import convective_cfl_calculator
from utils.serial import SerialMiniApp
from utils.vertical_slice import hydrostatic_rho, \
    get_form_mass, get_form_function, maximum

PETSc.Sys.Print("Setting up problem")

comm = fd.COMM_WORLD

# set up the mesh

output_freq = 10
nt = 1800
dt = 5.

nlayers = 70  # horizontal layers
base_columns = 180  # number of columns
L = 144e3
H = 35e3  # Height position of the model top

distribution_parameters = {
    "partition": True,
    "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)
}

# surface mesh of ground
base_mesh = fd.PeriodicIntervalMesh(base_columns, L,
                                    distribution_parameters=distribution_parameters,
                                    comm=comm)

# volume mesh of the slice
mesh = fd.ExtrudedMesh(base_mesh,
                       layers=nlayers,
                       layer_height=H/nlayers)
n = fd.FacetNormal(mesh)
x, z = fd.SpatialCoordinate(mesh)

g = fd.Constant(9.810616)
N = fd.Constant(0.01)  # Brunt-Vaisala frequency (1/s)
cp = fd.Constant(1004.5)  # SHC of dry air at const. pressure (J/kg/K)
R_d = fd.Constant(287.)  # Gas constant for dry air (J/kg/K)
kappa = fd.Constant(2.0/7.0)  # R_d/c_p
p_0 = fd.Constant(1000.0*100.0)  # reference pressure (Pa, not hPa)
cv = fd.Constant(717.)  # SHC of dry air at const. volume (J/kg/K)
T_0 = fd.Constant(273.15)  # ref. temperature

dT = fd.Constant(dt)

# making a mountain out of a molehill
a = 10000.
xc = L/2.
hm = 1.
zs = hm*a**2/((x-xc)**2 + a**2)

smooth_z = True
name = "mountain_nh"
if smooth_z:
    name += '_smootherz'
    zh = 5000.
    xexpr = fd.as_vector([x, fd.conditional(z < zh, z + fd.cos(0.5*pi*z/zh)**6*zs, z)])
else:
    xexpr = fd.as_vector([x, z + ((H-z)/H)*zs])
mesh.coordinates.interpolate(xexpr)

horizontal_degree = 1
vertical_degree = 1

S1 = fd.FiniteElement("CG", fd.interval, horizontal_degree+1)
S2 = fd.FiniteElement("DG", fd.interval, horizontal_degree)

# vertical base spaces
T0 = fd.FiniteElement("CG", fd.interval, vertical_degree+1)
T1 = fd.FiniteElement("DG", fd.interval, vertical_degree)

# build spaces V2, V3, Vt
V2h_elt = fd.HDiv(fd.TensorProductElement(S1, T1))
V2t_elt = fd.TensorProductElement(S2, T0)
V3_elt = fd.TensorProductElement(S2, T1)
V2v_elt = fd.HDiv(V2t_elt)
V2_elt = V2h_elt + V2v_elt

V1 = fd.FunctionSpace(mesh, V2_elt, name="Velocity")
V2 = fd.FunctionSpace(mesh, V3_elt, name="Pressure")
Vt = fd.FunctionSpace(mesh, V2t_elt, name="Temperature")
Vv = fd.FunctionSpace(mesh, V2v_elt, name="Vv")

W = V1 * V2 * Vt  # velocity, density, temperature

PETSc.Sys.Print(f"DoFs: {W.dim()}")
PETSc.Sys.Print(f"DoFs/core: {W.dim()/comm.size}")

Un = fd.Function(W)

# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
Tsurf = fd.Constant(300.)
thetab = Tsurf*fd.exp(N**2*z/g)

Up = fd.as_vector([fd.Constant(0.0), fd.Constant(1.0)])  # up direction

un, rhon, thetan = Un.subfunctions
un.project(fd.as_vector([10.0, 0.0]))
thetan.interpolate(thetab)
theta_back = fd.Function(Vt).assign(thetan)
rhon.assign(1.0e-5)

PETSc.Sys.Print("Calculating hydrostatic state")

Pi = fd.Function(V2)

hydrostatic_rho(Vv, V2, mesh, thetan, rhon, pi_boundary=fd.Constant(0.02),
                cp=cp, R_d=R_d, p_0=p_0, kappa=kappa, g=g, Up=Up,
                top=True, Pi=Pi)
p0 = maximum(Pi)

hydrostatic_rho(Vv, V2, mesh, thetan, rhon, pi_boundary=fd.Constant(0.05),
                cp=cp, R_d=R_d, p_0=p_0, kappa=kappa, g=g, Up=Up,
                top=True, Pi=Pi)
p1 = maximum(Pi)
alpha = 2.*(p1-p0)
beta = p1-alpha
pi_top = (1.-beta)/alpha

hydrostatic_rho(Vv, V2, mesh, thetan, rhon, pi_boundary=fd.Constant(pi_top),
                cp=cp, R_d=R_d, p_0=p_0, kappa=kappa, g=g, Up=Up,
                top=True)

rho_back = fd.Function(V2).assign(rhon)

zc = fd.Constant(H-10000.)
mubar = fd.Constant(0.15/dt)
mu_top = fd.conditional(z <= zc, 0.0,
                        mubar*fd.sin(fd.Constant(pi/2.)*(z-zc)/(fd.Constant(H)-zc))**2)
mu = fd.Function(V2).interpolate(mu_top)

form_function = get_form_function(n, Up, c_pen=2.0**(-7./2),
                                  cp=cp, g=g, R_d=R_d,
                                  p_0=p_0, kappa=kappa, mu=mu)

form_mass = get_form_mass()

zv = fd.as_vector([fd.Constant(0.), fd.Constant(0.)])
bcs = [fd.DirichletBC(W.sub(0), zv, "bottom"),
       fd.DirichletBC(W.sub(0), zv, "top")]

for bc in bcs:
    bc.apply(Un)

# Parameters for the newton iterations
solver_parameters = {
    "snes": {
        "monitor": None,
        "converged_reason": None,
        "stol": 1e-12,
        "atol": 1e-6,
        "rtol": 1e-6,
        "ksp_ew": None,
        "ksp_ew_version": 1,
        "ksp_ew_threshold": 1e-5,
        "ksp_ew_rtol0": 1e-3,
    },
    "ksp_type": "gmres",
    "ksp": {
        "monitor": None,
        "converged_reason": None,
        "atol": 1e-7,
    },
    "pc_type": "python",
    "pc_python_type": "firedrake.AssembledPC",
    "assembled": {
        "pc_type": "python",
        "pc_python_type": "firedrake.ASMVankaPC",
        "pc_vanka": {
            "construct_dim": 0,
            "sub_sub_pc_type": "lu",
            "sub_sub_pc_factor_mat_solver_type": 'mumps',
        },
    },
}

theta = 0.5

miniapp = SerialMiniApp(dt=dt, theta=theta, w_initial=Un,
                        form_mass=form_mass,
                        form_function=form_function,
                        solver_parameters=solver_parameters,
                        bcs=bcs)

PETSc.Sys.Print("Solving problem")

uout = fd.Function(V1, name='velocity')
thetaout = fd.Function(Vt, name='temperature')
rhoout = fd.Function(V2, name='density')

ofile = fd.File('output/slice_mountain.pvd',
                comm=comm)


def assign_out_functions():
    uout.assign(miniapp.w0.subfunctions[0])
    rhoout.assign(miniapp.w0.subfunctions[1])
    thetaout.assign(miniapp.w0.subfunctions[2])

    rhoout.assign(rhoout - rho_back)
    thetaout.assign(thetaout - theta_back)


def write_to_file(time):
    ofile.write(uout, rhoout, thetaout, t=time)


assign_out_functions()
write_to_file(time=0)

PETSc.Sys.Print('### === --- Timestepping loop --- === ###')
linear_its = 0
nonlinear_its = 0

cfl_calc = convective_cfl_calculator(mesh)
cfl_series = []


def max_cfl(u, dt):
    with cfl_calc(u, dt).dat.vec_ro as v:
        return v.max()[1]


def preproc(app, it, time):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'### === --- Calculating time-step {it} --- === ###')
    PETSc.Sys.Print('')


def postproc(app, it, time):
    global linear_its
    global nonlinear_its

    linear_its += miniapp.nlsolver.snes.getLinearSolveIterations()
    nonlinear_its += miniapp.nlsolver.snes.getIterationNumber()

    if (it % output_freq) == 0:
        assign_out_functions()
        write_to_file(time=time)

    cfl = max_cfl(uout, dt)
    cfl_series.append(cfl)
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'Time = {time}')
    PETSc.Sys.Print(f'Maximum CFL = {cfl}')


# solve for each window
miniapp.solve(nt=nt,
              preproc=preproc,
              postproc=postproc)

PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Iteration counts --- === ###')
PETSc.Sys.Print('')

PETSc.Sys.Print(f'linear iterations: {linear_its} | iterations per timestep: {linear_its/nt}')
PETSc.Sys.Print(f'nonlinear iterations: {nonlinear_its} | iterations per timestep: {nonlinear_its/nt}')
PETSc.Sys.Print('')
