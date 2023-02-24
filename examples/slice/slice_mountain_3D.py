import firedrake as fd
import asQ
from math import pi
from utils.vertical_slice import hydrostatic_rho, \
    get_form_mass, get_form_function, maximum
from petsc4py import PETSc

# set up the ensemble communicator for space-time parallelism
time_partition = tuple((1 for _ in range(4)))

ensemble = asQ.create_ensemble(time_partition, comm=fd.COMM_WORLD)

# set up the mesh

nx = 12  # number streamwise of columns
ny = 6  # number spanwise of columns
nz = 12  # horizontal layers
Lx = 16e3
Ly = 12e3
Lz = 12e3  # Height position of the model top

distribution_parameters = {
    "partition": True,
    "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)
}

# surface mesh of ground
base_mesh = fd.PeriodicRectangleMesh(nx, ny, Lx, Ly,
                                     direction='both', quadrilateral=True,
                                     distribution_parameters=distribution_parameters,
                                     comm=ensemble.comm)

# volume mesh of the slice
mesh = fd.ExtrudedMesh(base_mesh, layers=nz, layer_height=Lz/nz)
n = fd.FacetNormal(mesh)


g = fd.Constant(9.810616)
N = fd.Constant(0.01)  # Brunt-Vaisala frequency (1/s)
cp = fd.Constant(1004.5)  # SHC of dry air at const. pressure (J/kg/K)
R_d = fd.Constant(287.)  # Gas constant for dry air (J/kg/K)
kappa = fd.Constant(2.0/7.0)  # R_d/c_p
p_0 = fd.Constant(1000.0*100.0)  # reference pressure (Pa, not hPa)
cv = fd.Constant(717.)  # SHC of dry air at const. volume (J/kg/K)
T_0 = fd.Constant(273.15)  # ref. temperature

dt = 1
dT = fd.Constant(dt)

# making a mountain out of a molehill
a = 2000.
xc = Lx/2.
yc = Ly/2.
x, y, z = fd.SpatialCoordinate(mesh)
hm = 1.
r2 = ((x - xc)/a)**2 + ((y - yc)/(2*a))**2
zs = hm*fd.exp(-r2)

smooth_z = True
name = "mountain_nh"
if smooth_z:
    name += '_smootherz'
    zh = 5000.
    xexpr = fd.as_vector([x, y, fd.conditional(z < zh, z + fd.cos(0.5*pi*z/zh)**6*zs, z)])
else:
    xexpr = fd.as_vector([x, y, z + ((Lz-z)/Lz)*zs])
mesh.coordinates.interpolate(xexpr)

horizontal_degree = 1
vertical_degree = 1

S1 = fd.FiniteElement("RTCF", fd.quadrilateral, horizontal_degree+1)
S2 = fd.FiniteElement("DQ", fd.quadrilateral, horizontal_degree)

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

Un = fd.Function(W)

# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
Tsurf = fd.Constant(300.)
thetab = Tsurf*fd.exp(N**2*z/g)

cp = fd.Constant(1004.5)  # SHC of dry air at const. pressure (J/kg/K)
Up = fd.as_vector([fd.Constant(0.0), fd.Constant(0.0), fd.Constant(1.0)])  # up direction

un = Un.subfunctions[0]
rhon = Un.subfunctions[1]
thetan = Un.subfunctions[2]
un.project(fd.as_vector([20.0, 0.0, 0.0]))
thetan.interpolate(thetab)
theta_back = fd.Function(Vt).assign(thetan)
rhon.assign(1.0e-5)

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

zc = Lz-10000.
mubar = 0.3
mu_top = fd.conditional(z <= zc, 0.0, mubar*fd.sin((pi/2.)*(z-zc)/(Lz-zc))**2)
mu = fd.Function(V2).interpolate(mu_top/dT)

form_function = get_form_function(n, Up, c_pen=2.0**(-7./2),
                                  cp=cp, g=g, R_d=R_d,
                                  p_0=p_0, kappa=kappa, mu=mu)

form_mass = get_form_mass()

zv = fd.as_vector([fd.Constant(0.), fd.Constant(0.), fd.Constant(0.)])
bcs = [fd.DirichletBC(W.sub(0), zv, "bottom"),
       fd.DirichletBC(W.sub(0), zv, "top")]

for bc in bcs:
    bc.apply(Un)

# Parameters for the diag
lines_parameters = {
    'ksp_type': 'gmres',
    # 'ksp_converged_reason': None,
    'ksp_rtol': 1e-5,
    'pc_type': 'python',
    'pc_python_type': 'firedrake.AssembledPC',
    'assembled': {
        'pc_type': 'python',
        'pc_python_type': 'firedrake.ASMVankaPC',
        'pc_vanka_construct_dim': 0,
        'pc_vanka_sub_sub_pc_type': 'lu',
        'pc_vanka_sub_sub_pc_factor_mat_solver_type': 'mumps',
    }
}

solver_parameters_diag = {
    'snes_monitor': None,
    'snes_converged_reason': None,
    'snes_rtol': 1e-8,
    'ksp_type': 'fgmres',
    'ksp_monitor': None,
    'ksp_converged_reason': None,
    'mat_type': 'matfree',
    'pc_type': 'python',
    'pc_python_type': 'asQ.DiagFFTPC'
}

for i in range(sum(time_partition)):
    solver_parameters_diag["diagfft_block_"+str(i)+"_"] = lines_parameters

alpha = 1.0e-4
theta = 0.5

pdg = asQ.paradiag(ensemble=ensemble,
                   form_function=form_function,
                   form_mass=form_mass, w0=Un,
                   dt=dt, theta=theta,
                   alpha=alpha, time_partition=time_partition, bcs=bcs,
                   solver_parameters=solver_parameters_diag,
                   circ="none")

aaos = pdg.aaos
is_last_slice = pdg.layout.is_local(-1)

# only last slice does diagnostics/output
if is_last_slice:

    uout = fd.Function(V1, name='velocity')
    thetaout = fd.Function(Vt, name='temperature')
    rhoout = fd.Function(V2, name='density')

    ofile = fd.File('output/slice_mountain_diag.pvd',
                    comm=ensemble.comm)

    def assign_out_functions():
        aaos.get_component(-1, 0, wout=uout)
        aaos.get_component(-1, 1, wout=rhoout)
        aaos.get_component(-1, 2, wout=thetaout)

        rhoout.assign(rhoout - rho_back)
        thetaout.assign(thetaout - theta_back)

    def write_to_file():
        ofile.write(uout, rhoout, thetaout)


def window_preproc(pdg, wndw):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'### === --- Calculating time-window {wndw} --- === ###')
    PETSc.Sys.Print('')


def window_postproc(pdg, wndw):
    # postprocess this timeslice
    if is_last_slice:
        assign_out_functions()
        write_to_file()


# solve for each window
pdg.solve(nwindows=2,
          preproc=window_preproc,
          postproc=window_postproc)
