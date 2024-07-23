from firedrake import (
    PeriodicSquareMesh, SpatialCoordinate, Constant,
    sin, cos, pi, as_vector, errornorm, norm)
from gusto import *
from firedrake.petsc import PETSc
Print = PETSc.Sys.Print

from utils.serial import SerialMiniApp
from gusto_interface import asq_forms


dt = 0.001
tmax = 30*dt
H = 1
wx = 2
wy = 1
g = 1

# Domain
mesh_name = 'linear_sw_mesh'
L = 1
nx = ny = 20
mesh = PeriodicSquareMesh(nx, ny, L, direction='both', name=mesh_name)
x, y = SpatialCoordinate(mesh)
domain = Domain(mesh, dt, 'BDM', 1)

# Equation
parameters = ShallowWaterParameters(H=H, g=g)
fexpr = Constant(1)
eqns = LinearShallowWaterEquations(domain, parameters, fexpr=fexpr)

# I/O
output = OutputParameters(
    dirname=str('output')+"/linear_sw_wave",
    dumpfreq=1,
    checkpoint=True
)
io = IO(domain, output)
transport_methods = [DefaultTransport(eqns, "D")]

# Timestepper
stepper = Timestepper(eqns, TrapeziumRule(domain), io, transport_methods)

# ---------------------------------------------------------------------- #
# Initial conditions
# ---------------------------------------------------------------------- #

eta = sin(2*pi*(x-L/2)*wx)*cos(2*pi*(y-L/2)*wy) - (1/5)*cos(2*pi*(x-L/2)*wx)*sin(4*pi*(y-L/2)*wy)
Dexpr = H + eta

u = cos(4*pi*(x-L/2)*wx)*cos(2*pi*(y-L/2)*wy)
v = cos(2*pi*(x-L/2)*wx)*cos(4*pi*(y-L/2)*wy)
uexpr = as_vector([u, v])

u0 = stepper.fields("u")
D0 = stepper.fields("D")

u0.project(uexpr)
D0.interpolate(Dexpr)

# --------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------- #

stepper.run(t=0, tmax=tmax)

gusto_u = stepper.fields('u')
gusto_D = stepper.fields('D')


# --------------------------------------------------------------------- #
# Setup asQ stepper
# --------------------------------------------------------------------- #
form_mass, form_function = asq_forms(eqns)
theta = 0.5
w0 = Function(eqns.function_space)
w0.subfunctions[0].project(uexpr)
w0.subfunctions[1].interpolate(Dexpr)
params = {
    'snes_type': 'ksponly',
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'ksp_converged_reason': None,
}
asqstepper = SerialMiniApp(dt, theta, w0, form_mass, form_function,
                           solver_parameters=params)
asqstepper.solve(nt=stepper.step-1)
asq_u = asqstepper.w1.subfunctions[0]
asq_D = asqstepper.w1.subfunctions[1]

Print(f"{errornorm(asq_u, gusto_u)/norm(gusto_u) = }")
Print(f"{errornorm(asq_D, gusto_D)/norm(gusto_D) = }")
