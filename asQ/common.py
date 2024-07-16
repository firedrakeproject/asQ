from firedrake.petsc import PETSc
from warnings import warn


def get_option_from_list(prefix, option_name, option_list,
                         default_index=None, deprecated_prefix=None):
    """
    Get a string option from the global PETSc.Options and check it is one of a valid list.

    :arg option_name: the name of the PETSc option.
    :arg option_list: an iterable with all valid values for the option.
    :arg default_index: the index of the default option in the option_list.
        if None then no default used.
    """
    default = None if default_index is None else option_list[default_index]
    if deprecated_prefix is not None:
        option = get_deprecated_option(PETSc.Options().getString,
                                       prefix, deprecated_prefix,
                                       option_name, default)
    else:
        option = PETSc.Options().getString(prefix+option_name,
                                           default=default)
    if option not in option_list:
        msg = f"{option} must be one of "+" or ".join(option_list)
        raise ValueError(msg)
    return option


def get_deprecated_option(getOption, prefix, deprecated_prefix,
                          option_name, default=None):
    deprecated_name = deprecated_prefix + option_name
    option_name = prefix + option_name

    deprecated_option = getOption(deprecated_name,
                                  default=default)
    option = getOption(option_name,
                       default=default)

    if deprecated_option != default:
        msg = f"Prefix {deprecated_prefix} is deprecated and will be removed in the future. Use {prefix} instead."
        warn(msg)
        if option != default:
            msg = f"{deprecated_name} ignored in favour of {option_name}"
            warn(msg)
        else:
            option = deprecated_option

    return option
