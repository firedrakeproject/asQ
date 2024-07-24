"""
The Williamson 5 shallow-water test case (flow over topography), solved with a
discretisation of the non-linear shallow-water equations.

This uses an icosahedral mesh of the sphere, and runs a series of resolutions.
"""

from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate,
                       errornorm, norm, as_vector, pi, sqrt, min_value)
from firedrake.petsc import PETSc
from firedrake.output import VTKFile
Print = PETSc.Sys.Print

from utils.serial import SerialMiniApp
from gusto_interface import asq_forms

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

hour = 60.*60.
day = 24.*60.*60.
ref_level = 3
dt = 1800
# tmax = 50*day
tmax = day/3
ndumps = 1

# setup shallow water parameters
R = 6371220.
H = 5960.

# setup input that doesn't change with ref level or dt
parameters = ShallowWaterParameters(H=H)

# ------------------------------------------------------------------------ #
# Set up model objects
# ------------------------------------------------------------------------ #

# Domain
mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=ref_level, degree=2)
x = SpatialCoordinate(mesh)
domain = Domain(mesh, dt, 'BDM', 1)

# Equation
Omega = parameters.Omega
fexpr = 2*Omega*x[2]/R
lamda, theta, _ = lonlatr_from_xyz(x[0], x[1], x[2])
R0 = pi/9.
R0sq = R0**2
lamda_c = -pi/2.
lsq = (lamda - lamda_c)**2
theta_c = pi/6.
thsq = (theta - theta_c)**2
rsq = min_value(R0sq, lsq+thsq)
r = sqrt(rsq)
bexpr = 2000 * (1 - r/R0)
eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr, bexpr=bexpr)

# I/O
dirname = "williamson_5_ref%s_dt%s" % (ref_level, dt)
# dumpfreq = int(tmax / (ndumps*dt))
dumpfreq = 1
output = OutputParameters(
    dirname=dirname,
    dumplist_latlon=['D'],
    dumpfreq=dumpfreq,
)
diagnostic_fields = [Sum('D', 'topography')]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport schemes
transport_methods = [DGUpwind(eqns, "u"), DGUpwind(eqns, "D")]

params = {
    'snes_monitor': None,
    'snes_converged_reason': None,
    'snes_rtol': 1e-12,
    'snes_ksp_ew': None,
    'snes_ksp_rtol0': 1e-2,
    'ksp_type': 'fgmres',
    'pc_type': 'lu',
    'ksp_monitor': None,
    'ksp_converged_rate': None,
    'snes_lag_preconditioner': -2,
    'snes_lag_preconditioner_persists': None,
}

# Time stepper
stepper = Timestepper(eqns, TrapeziumRule(domain, solver_parameters=params),
                      io, transport_methods)

# ------------------------------------------------------------------------ #
# Setup Gusto solver
# ------------------------------------------------------------------------ #

u0 = stepper.fields('u')
D0 = stepper.fields('D')
u_max = 20.   # Maximum amplitude of the zonal wind (m/s)
uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
g = parameters.g
Rsq = R**2
Dexpr = H - ((R * Omega * u_max + 0.5*u_max**2)*x[2]**2/Rsq)/g - bexpr

u0.project(uexpr)
D0.interpolate(Dexpr)

Dbar = Function(D0.function_space()).assign(H)
stepper.set_reference_profiles([('D', Dbar)])

# --------------------------------------------------------------------- #
# Setup asQ stepper
# --------------------------------------------------------------------- #
form_mass, form_function = asq_forms(eqns, transport_velocity_index=0)
theta = 0.5
w0 = Function(eqns.function_space)
w0.subfunctions[0].project(uexpr)
w0.subfunctions[1].interpolate(Dexpr)
asqstepper = SerialMiniApp(dt, theta, w0, form_mass, form_function,
                           solver_parameters=params)

# ------------------------------------------------------------------------ #
# Run Gusto solver
# ------------------------------------------------------------------------ #

stepper.run(t=0, tmax=tmax)

gusto_u = stepper.fields('u')
gusto_D = stepper.fields('D')

# ------------------------------------------------------------------------ #
# Run asQ solver
# ------------------------------------------------------------------------ #

asq_file = VTKFile(f"results/{dirname}/asq_timeseries.pvd")
asq_file.write(*asqstepper.w1.subfunctions)


def postprocess(app, step, t):
    asq_file.write(*asqstepper.w1.subfunctions)


asqstepper.solve(nt=stepper.step-1, postproc=postprocess)
asq_u = asqstepper.w1.subfunctions[0]
asq_D = asqstepper.w1.subfunctions[1]

# ------------------------------------------------------------------------ #
# Compare results
# ------------------------------------------------------------------------ #

uinit, Dinit = w0.subfunctions
Print(f"{errornorm(asq_u, uinit)/norm(uinit) = }")
Print(f"{errornorm(asq_D, Dinit)/norm(Dinit) = }")

Print(f"{errornorm(asq_u, gusto_u)/norm(uinit) = }")
Print(f"{errornorm(asq_D, gusto_D)/norm(uinit) = }")
