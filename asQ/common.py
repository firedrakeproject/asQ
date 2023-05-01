from firedrake.petsc import PETSc


def get_option_from_list(option_name, option_list, default_index=None):
    default = option_list[default_index] if default_index is not None else None
    option = PETSc.Options().getString(option_name, default=default)
    if option not in option_list:
        raise ValueError(f"{option} must be one of "+" or ".join(option_list))
    return option
