
import firedrake as fd

from pyop2.mpi import MPI


def write_timesteps(pdg,
                    file_name='paradiag_output',
                    function_names=[],
                    only_last_step=False):
    """Writes each timestep of a paradiag object to seperate vtk files.

    :arg pdg: the paradiag object
    :arg file_name: optional name for the files. The full filename will be file_name.{timestep}
    :arg function_names: a list of names for each function in the (mixed) function space at each timestep
    :arg only_last_step: only writes the last timestep and does not use .{timestep} suffix on filename
    """

    # TODO: This implementation assumes that a MixedFunctionSpace is used at each timestep
    #       Once there is an example using a plain FunctionSpace this will need updating

    # if given, check we have the right number of function_names
    if (len(function_names) != 0) and (len(function_names) != pdg.ncpts):
        raise ValueError("function_names must be same length as pdg.ncpts,"
                         + f" {len(function_names)} provided, {pdg.ncpts} needed.")

    # functions for writing to file
    functions = []
    for cpt in range(pdg.ncpts):
        V = pdg.W.split()[cpt]
        if len(function_names) != 0:
            functions.append(fd.Function(V, name=function_names[cpt]))
        else:
            functions.append(fd.Function(V))

    # functions from entire local time-slice
    walls = pdg.w_all.split()

    # first timestep of this local time-slice
    timestep0 = sum(pdg.M[:pdg.rT])

    if only_last_step:
        if pdg.rT == (len(pdg.M) - 1):
            i = pdg.M[pdg.rT]-1
            timestep = timestep0+i

            # index of first split function in this timestep
            index0 = pdg.ncpts*i

            for cpt in range(pdg.ncpts):
                functions[j].assign(walls[index0+cpt])

            fd.File(file_name+".pvd",
                    comm=pdg.ensemble.comm).write(*functions)

            return

    else:  # write every timestep

        for i in range(pdg.M[pdg.rT]):
            timestep = timestep0+i

            # index of first split function in this timestep
            index0 = pdg.ncpts*i

            for cpt in range(pdg.ncpts):
                functions[cpt].assign(walls[index0+cpt])

            fd.File(file_name+"."+str(timestep)+".pvd",
                    comm=pdg.ensemble.comm).write(*functions)


def write_timeseries(pdg,
                     file_name='paradiag_output',
                     function_names=[]):
    """Writes timesteps of a paradiag object to a timeseries vtk file.

    :arg pdg: the paradiag object
    :arg file_name: optional name for the file
    :arg function_names: a list of names for each function in the (mixed) function space at each timestep
    """

    # TODO: This implementation assumes that a MixedFunctionSpace is used at each timestep
    #       Once there is an example using a plain FunctionSpace this will need updating

    # if given, check we have the right number of function_names
    if (len(function_names) != 0) and (len(function_names) != pdg.ncpts):
        raise ValueError("function_names must be same length as pdg.ncpts,"
                         + f" {len(function_names)} provided, {pdg.ncpts} needed.")

    # functions for writing to file
    functions = []
    for cpt in range(pdg.ncpts):
        V = pdg.W.split()[cpt]
        if len(function_names) != 0:
            functions.append(fd.Function(V, name=function_names[cpt]))
        else:
            functions.append(fd.Function(V))

    # only first time slice writes to file
    if pdg.rT == 0:
        outfile = fd.File(file_name+".pvd",
                          comm=pdg.ensemble.comm)

    # functions from entire local time-slice
    walls = pdg.w_all.split()

    # first timestep of this local time-slice
    timestep_begin = sum(pdg.M[:pdg.rT])

    # one past last timestep of the local time-slice
    timestep_end = timestep_begin + pdg.M[pdg.rT]

    for timestep in range(sum(pdg.M)):

        # which rank is this timestep on?
        for r in range(len(pdg.M)):
            t0 = sum(pdg.M[:r])
            t1 = t0 + pdg.M[r]
            if (t0 <= timestep) and (timestep < t1):
                time_rank = r

        mpi_requests = []
        # if timestep on this time-rank, send functions
        if time_rank == pdg.rT:

            # index of first split function in this timestep
            index0 = pdg.ncpts*(timestep - timestep_begin)

            # send functions in timestep to time-rank 0
            for cpt in range(pdg.ncpts):
                request_send = pdg.ensemble.isend(walls[index0+cpt],
                                                  dest=0,
                                                  tag=timestep)
                mpi_requests.extend(request_send)

        # if time-rank 0: recv functions
        if pdg.rT == 0:

            # recv functions in timestep
            for cpt in range(pdg.ncpts):
                request_recv = pdg.ensemble.irecv(functions[cpt],
                                                  source=time_rank,
                                                  tag=timestep)
                mpi_requests.extend(request_recv)

        MPI.Request.Waitall(mpi_requests)

        # if time-rank 0: write to file
        if pdg.rT == 0:
            outfile.write(*functions)
