from firedrake.petsc import PETSc


def get_option_from_list(prefix, option_name, option_list,
                         default_index=None):
    """
    Get a string option from the global PETSc.Options and check it is one of a valid list.

    :arg option_name: the name of the PETSc option.
    :arg option_list: an iterable with all valid values for the option.
    :arg default_index: the index of the default option in the option_list.
        if None then no default used.
    """
    default = None if default_index is None else option_list[default_index]
    option = PETSc.Options().getString(prefix+option_name,
                                       default=default)
    if option not in option_list:
        msg = f"{option} must be one of "+" or ".join(option_list)
        raise ValueError(msg)
    return option
