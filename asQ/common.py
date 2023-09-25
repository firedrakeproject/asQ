from firedrake.petsc import PETSc


def get_option_from_list(option_name, option_list, default_index=None):
    """
    Get a string option from the global PETSc.Options and check it is one of a valid list.

    :arg option_name: the name of the PETSc option.
    :arg option_list: an iterable with all valid values for the option.
    :arg default_index: the index of the default option in the option_list.
        if None then no default used.
    """
    default = option_list[default_index] if default_index is not None else None
    option = PETSc.Options().getString(option_name, default=default)
    if option not in option_list:
        raise ValueError(f"{option} must be one of "+" or ".join(option_list))
    return option
