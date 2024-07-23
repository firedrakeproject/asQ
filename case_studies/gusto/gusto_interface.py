from firedrake.fml import (
    Term, all_terms, drop,
    replace_subject, replace_test_function
)
from firedrake.formmanipulation import split_form
from gusto.labels import time_derivative, prognostic


def asq_forms(equation):
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
        tests = trials_and_tests[ncpts:]

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

            f += fi
        f = f.label_map(lambda t: t is NullTerm, drop)
        return f.form

    return form_mass, form_function
