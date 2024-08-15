try:
    from gusto import time_derivative, prognostic, transporting_velocity
except ModuleNotFoundError as err:
    raise type(err)("Gusto must be installed to use the asQ-Gusto interface.") from err

from firedrake.fml import (
    Term, all_terms, drop,
    replace_subject, replace_test_function
)
from firedrake.formmanipulation import split_form
from ufl import replace


def asq_forms(equation, transport_velocity_index=None):
    NullTerm = Term(None)
    ncpts = len(equation.function_space)

    # split equation into mass matrix and linear operator
    mass = equation.residual.label_map(
        lambda t: t.has_label(time_derivative),
        map_if_false=drop)

    function = equation.residual.label_map(
        lambda t: t.has_label(time_derivative),
        map_if_true=drop)

    # generate ufl for mass matrix over given trial/tests
    def form_mass(*trials_and_tests):
        trials = trials_and_tests[:ncpts] if ncpts > 1 else trials_and_tests[0]
        tests = trials_and_tests[ncpts:] if ncpts > 1 else trials_and_tests[1]

        m = mass.label_map(
            all_terms,
            replace_test_function(tests))
        m = m.label_map(
            all_terms,
            replace_subject(trials))
        return m.form

    # generate ufl for linear operator over given trial/tests
    def form_function(*trials_and_tests):
        trials = trials_and_tests[:ncpts] if ncpts > 1 else trials_and_tests[0]
        tests = trials_and_tests[ncpts:2*ncpts]

        fields = equation.field_names if ncpts > 1 else [equation.field_name]

        f = NullTerm
        for i in range(ncpts):
            fi = function.label_map(
                lambda t: t.get(prognostic) == fields[i],
                lambda t: Term(
                    split_form(t.form)[i].form,
                    t.labels),
                map_if_false=drop)

            fi = fi.label_map(
                all_terms,
                replace_test_function(tests[i]))

            fi = fi.label_map(
                all_terms,
                replace_subject(trials))

            if transport_velocity_index is not None:
                transport_velocity = trials[transport_velocity_index]
                fi = fi.label_map(
                    lambda t: t.has_label(transporting_velocity),
                    map_if_true=lambda t: Term(replace(t.form, {t.get(transporting_velocity): transport_velocity}), t.labels))

            f += fi
        f = f.label_map(lambda t: t is NullTerm, drop)
        return f.form

    return form_mass, form_function
