from argparse import ArgumentDefaultsHelpFormatter, RawDescriptionHelpFormatter


class DefaultsAndRawTextFormatter(ArgumentDefaultsHelpFormatter, RawDescriptionHelpFormatter):
    '''
    This just combines the effect of the two subclassed argparse formatters.
    ArgumentDefaultsHelpFormatter will print the default argument values with `-h`.
    RawDescriptionHelpFormatter means we can format the PETSc argument help more nicely.
    '''
    pass
