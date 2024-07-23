
from firedrake import (SpatialCoordinate, PeriodicIntervalMesh, exp, as_vector,
                       norm, Constant, conditional, sqrt, VectorFunctionSpace)
from firedrake.petsc import PETSc
from gusto import *

from utils.serial import SerialMiniApp

Print = PETSc.Sys.Print


def asq_forms(equation):
    from firedrake.fml import (
        Term, all_terms, drop,
        replace_subject, replace_test_function
    )
    from gusto.labels import time_derivative, prognostic
    NullTerm = Term(None)

    V = equation.function_space
    ncpts = len(V)
    residual = equation.residual

    # split equation into mass matrix and linear operator
    mass = residual.label_map(
        lambda t: t.has_label(time_derivative),
        map_if_false=drop)

    function = residual.label_map(
        lambda t: t.has_label(time_derivative),
        map_if_true=drop)

    # generate ufl for mass matrix over given trial/tests
    def form_mass(*trials_and_tests):
        trials = trials_and_tests[:ncpts]
        tests = trials_and_tests[ncpts:]
        m = mass.label_map(
            all_terms,
            replace_test_function(tests[0]))
        m = m.label_map(
            all_terms,
            replace_subject(trials[0]))
        return m.form

    # generate ufl for linear operator over given trial/tests
    def form_function(*trials_and_tests):
        trials = trials_and_tests[:ncpts]
        tests = trials_and_tests[ncpts:]

        f = NullTerm
        for i in range(ncpts):
            fi = function.label_map(
                lambda t: t.get(prognostic) == equation.field_name,
                lambda t: Term(
                    split_form(t.form)[i].form,
                    t.labels),
                map_if_false=drop)

            fi = fi.label_map(
                all_terms,
                replace_test_function(tests[i]))

            fi = fi.label_map(
                all_terms,
                replace_subject(trials[i]))

            f += fi
        f = f.label_map(lambda t: t is NullTerm, drop)
        return f.form

    return form_mass, form_function


# Set up model objects
# ------------------------------------------------------------------------ #

# Domain
dt = 0.02
tmax = 1.0
L = 10
mesh = PeriodicIntervalMesh(20, L)
domain = Domain(mesh, dt, "CG", 1)

# Equation
diffusion_params = DiffusionParameters(kappa=0.75, mu=5)
V = domain.spaces("DG")
Vu = VectorFunctionSpace(mesh, "CG", 1)

equation = AdvectionDiffusionEquation(domain, V, "f", Vu=Vu,
                                      diffusion_parameters=diffusion_params)
spatial_methods = [DGUpwind(equation, "f"),
                   InteriorPenaltyDiffusion(equation, "f", diffusion_params)]

# I/O
output = OutputParameters(dirname=str('tmp'), dumpfreq=25)
io = IO(domain, output)

# Time stepper
stepper = PrescribedTransport(equation, TrapeziumRule(domain), io, spatial_methods)

# ------------------------------------------------------------------------ #
# Initial conditions
# ------------------------------------------------------------------------ #

x = SpatialCoordinate(mesh)
xc_init = 0.25*L
xc_end = 0.75*L
umax = 0.5*L/tmax
# umax = 1

# Get minimum distance on periodic interval to xc
x_init = conditional(sqrt((x[0] - xc_init)**2) < 0.5*L,
                     x[0] - xc_init, L + x[0] - xc_init)

x_end = conditional(sqrt((x[0] - xc_end)**2) < 0.5*L,
                    x[0] - xc_end, L + x[0] - xc_end)

f_init = 5.0
f_end = f_init / 2.0
f_width_init = L / 10.0
f_width_end = f_width_init * 2.0
f_init_expr = f_init*exp(-(x_init / f_width_init)**2)
f_end_expr = f_end*exp(-(x_end / f_width_end)**2)

stepper.fields('f').interpolate(f_init_expr)
stepper.fields('u').interpolate(as_vector([Constant(umax)]))
f_end = stepper.fields('f_end', space=V)
f_end.interpolate(f_end_expr)

# ------------------------------------------------------------------------ #
# Run Gusto stepper
# ------------------------------------------------------------------------ #

stepper.run(0, tmax=tmax)

# ------------------------------------------------------------------------ #
# Set up asQ stepper
# ------------------------------------------------------------------------ #

Print(f"{equation.function_space = }")
Print(f"{V = }")
params = {
    'snes_type': 'ksponly',
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'ksp_converged_reason': None,
}
form_mass, form_function = asq_forms(equation)
theta = 0.5
f0 = Function(V).interpolate(f_init_expr)
asqstep = SerialMiniApp(dt, theta, f0, form_mass, form_function,
                        solver_parameters=params)

# ------------------------------------------------------------------------ #
# Run Gusto stepper
# ------------------------------------------------------------------------ #

asqstep.solve(nt=stepper.step-1)

# ------------------------------------------------------------------------ #
# Check errors
# ------------------------------------------------------------------------ #

gusto_error = norm(stepper.fields('f') - f_end) / norm(f_end)
Print(f"{gusto_error = }")

asq_error = norm(asqstep.w1 - f_end) / norm(f_end)
Print(f"{asq_error = }")

gusto_asq_error = norm(stepper.fields('f') - asqstep.w1) / norm(f_end)
Print(f"{gusto_asq_error = }")
