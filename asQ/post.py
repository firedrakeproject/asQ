
import firedrake as fd

def write_timesteps( pdg,
                     file_name='paradiag_output',
                     function_names=[] ):
    """Writes each timestep of a paradiag object to seperate file.

    :arg pdg: the paradiag object
    :arg file_name: optional name for the files. The full filename will be file_name.{timestep}
    :arg function_names: a list of names for each function in the (mixed) function space at each timestep
    """

    # TODO: This implementation assumes that a MixedFunctionSpace is used at each timestep
    #       Once there is an example using a plain FunctionSpace this will need updating

    # if given, check we have the right number of function_names
    if len(function_names)!=0 and len(function_names)!=pdg.ncpts:
        raise ValueError(  "function_names must be same length as pdg.ncpts,"
                         +f" {len(function_names)} provided, {pdg.ncpts} needed." )

    # functions for writing to file
    functions=[]
    for i in range( pdg.ncpts ):
        V = pdg.W.split()[i]
        if len(function_names)!=0:
            functions.append( fd.Function(V, name=function_names[i]) )
        else:
            functions.append( fd.Function(V) )

    # functions from entire local time-slice
    walls = pdg.w_all.split()

    # first timestep of this local time-slice
    timestep0 = sum(pdg.M[:pdg.rT])

    for i in range( pdg.M[pdg.rT] ):
        timestep = timestep0+i

        # index of first split function in this timestep
        index0 = pdg.ncpts*i

        for j in range( pdg.ncpts ):
            functions[j].assign(walls[index0+j])

        fd.File( file_name+"."+str(timestep)+".pvd",
                 comm=pdg.ensemble.comm
               ).write(*functions)

