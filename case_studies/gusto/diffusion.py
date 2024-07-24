from firedrake import (
    PeriodicSquareMesh, SpatialCoordinate,
    exp, errornorm, Function)
from gusto import *
from firedrake.petsc import PETSc
Print = PETSc.Sys.Print

from utils.serial import SerialMiniApp
from gusto_interface import asq_forms

dt = 0.01
L = 10.
mesh = PeriodicSquareMesh(10, 10, L, direction='x')

output = OutputParameters(dirname=str('output'), dumpfreq=25)
domain = Domain(mesh, dt, family="CG", degree=1)
io = IO(domain, output)

tmax = 1.0
x = SpatialCoordinate(mesh)
f_init = exp(-((x[0]-0.5*L)**2 + (x[1]-0.5*L)**2))

tol = 5.e-2
kappa = 1.

f_end_expr = (1/(1+4*tmax))*f_init**(1/(1+4*tmax))

V = domain.spaces("DG")
Print(f"{V=}")

mu = 5.

# gusto stepper

diffusion_params = DiffusionParameters(kappa=kappa, mu=mu)
eqn = DiffusionEquation(domain, V, "f", diffusion_parameters=diffusion_params)
diffusion_scheme = BackwardEuler(domain)
diffusion_methods = [InteriorPenaltyDiffusion(eqn, "f", diffusion_params)]
timestepper = Timestepper(eqn, diffusion_scheme, io, spatial_methods=diffusion_methods)
timestepper.fields("f").interpolate(f_init)

# asq stepper
form_mass, form_function = asq_forms(eqn)
theta = 1
f0 = Function(V).interpolate(f_init)
params = {
    'snes_type': 'ksponly',
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'ksp_converged_reason': None,
}
asqstepper = SerialMiniApp(dt, theta, f0, form_mass, form_function,
                           solver_parameters=params)

# Run gusto stepper
timestepper.run(0., tmax)
gusto_f_end = timestepper.fields('f')
Print(f"{errornorm(f_end_expr, gusto_f_end)=}")

# Run asQ stepper
asqstepper.solve(nt=timestepper.step-1)
asq_f_end = asqstepper.w1
Print(f"{errornorm(f_end_expr, asq_f_end)=}")

# error
Print(f"{errornorm(gusto_f_end, asq_f_end)=}")
