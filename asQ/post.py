
import firedrake as fd

from pyop2.mpi import MPI

__all__ = ["write_timesteps",
           "write_timeseries",
           "write_solver_parameters",
           "write_paradiag_setup",
           "write_aaos_solve_metrics",
           "write_block_solve_metrics",
           "write_paradiag_metrics"]


def write_timesteps(pdg,
                    file_name='paradiag_output',
                    function_names=[],
                    frequency=1,
                    only_last_step=False):
    """Writes each timestep of a paradiag object to seperate vtk files.

    :arg pdg: the paradiag object
    :arg file_name: optional name for the files. The full filename will be file_name.{timestep}
    :arg function_names: a list of names for each function in the (mixed) function space at each timestep
    :arg frequency: frequency at which to write timesteps
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
        V = pdg.W.subfunctions[cpt]
        if len(function_names) != 0:
            functions.append(fd.Function(V, name=function_names[cpt]))
        else:
            functions.append(fd.Function(V))

    # functions from entire local time-slice
    walls = pdg.w_all.subfunctions

    # first timestep of this local time-slice
    timestep0 = sum(pdg.M[:pdg.rT])

    if only_last_step:
        if pdg.rT == (len(pdg.M) - 1):
            i = pdg.M[pdg.rT]-1
            timestep = timestep0+i

            # index of first split function in this timestep
            index0 = pdg.ncpts*i

            for cpt in range(pdg.ncpts):
                functions[cpt].assign(walls[index0+cpt])

            fd.File(file_name+".pvd",
                    comm=pdg.ensemble.comm).write(*functions)

            return

    else:  # write timesteps with freqency

        for i in range(pdg.M[pdg.rT]):
            timestep = timestep0+i

            if timestep % frequency != 0:
                continue

            # index of first split function in this timestep
            index0 = pdg.ncpts*i

            for cpt in range(pdg.ncpts):
                functions[cpt].assign(walls[index0+cpt])

            fd.File(file_name+"."+str(timestep)+".pvd",
                    comm=pdg.ensemble.comm).write(*functions)


def write_timeseries(pdg,
                     file_name='paradiag_output',
                     function_names=[],
                     frequency=1,
                     time_scale=1):
    """Writes timesteps of a paradiag object to a timeseries vtk file.

    :arg pdg: the paradiag object
    :arg file_name: optional name for the file
    :arg function_names: a list of names for each function in the (mixed) function space at each timestep
    :arg frequency: frequency at which to write timesteps
    :arg time_scale: coefficient on timestamp for each timestep (eg 1./60 for timestamp in minutes)
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
        V = pdg.W.subfunctions[cpt]
        if len(function_names) != 0:
            functions.append(fd.Function(V, name=function_names[cpt]))
        else:
            functions.append(fd.Function(V))

    # only first time slice writes to file
    if pdg.rT == 0:
        outfile = fd.File(file_name+".pvd",
                          comm=pdg.ensemble.comm)

    # functions from entire local time-slice
    walls = pdg.w_all.subfunctions

    # first timestep of this local time-slice
    timestep_begin = sum(pdg.M[:pdg.rT])

    for timestep in range(0, sum(pdg.M), frequency):

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
            outfile.write(*functions, time=time_scale*timestep*pdg.dt)


def write_solver_parameters(sparams, directory=""):
    """
    Write the solver options dictionary sparams to file, formatted so that it can be parsed as python code

    :arg sparams: the solver options dictionary
    :arg directory: the directory to write the file into
    """
    from json import dumps

    if not isinstance(sparams, dict):
        raise ValueError("sparams must be dictionary")
    if not isinstance(directory, str):
        raise ValueError("directory must be string")

    if (directory != "") and (directory[-1] != "/"):
        directory += "/"
    file_name = directory + "solver_parameters.txt"

    with open(file_name, "w") as f:
        f.write(dumps(sparams, indent=4))

    return


def write_paradiag_setup(pdg, directory=""):
    """
    Write various parameters for the paradiag object to file e.g. dt, alpha

    :arg pdg: paradiag object
    :arg directory: the directory to write the files into
    """
    if not isinstance(directory, str):
        raise ValueError("directory must be string")

    is_root = (pdg.ensemble.global_comm.rank == 0)

    if is_root:
        if (directory != "") and (directory[-1] != "/"):
            directory += "/"
        file_name = directory + "paradiag_setup.txt"

        with open(file_name, "w") as f:
            info = \
                f"dt = {pdg.aaoform.dt}\n" + \
                f"theta = {pdg.aaoform.theta}\n" + \
                f"time_partition = {pdg.time_partition}\n" + \
                f"comm.size = {pdg.ensemble.comm.size}\n" + \
                f"ensemble_comm.size = {pdg.ensemble.ensemble_comm.size}\n" + \
                f"global_comm.size = {pdg.ensemble.global_comm.size}\n"
            if hasattr(pdg.solver.jacobian.pc, "alpha"):
                info += f"alpha = {pdg.solver.jacobian.pc.alpha}"
            f.write(info)
    return


def write_aaos_solve_metrics(pdg, directory=""):
    """
    Write various metrics for the all-at-once solve from the paradiag object to file e.g. number of windows, number of iterations etc

    :arg pdg: paradiag object
    :arg directory: the directory to write the files into
    """
    if not isinstance(directory, str):
        raise ValueError("directory must be string")

    pdg.sync_diagnostics()

    is_root = (pdg.ensemble.global_comm.rank == 0)

    if is_root:
        if (directory != "") and (directory[-1] != "/"):
            directory += "/"
        file_name = directory + "aaos_metrics.txt"

        with open(file_name, "w") as f:
            nt = pdg.total_timesteps
            nw = pdg.total_windows
            nlits = pdg.nonlinear_iterations
            lits = pdg.linear_iterations
            blits = max(pdg.block_iterations._data)
            info = \
                f"total timesteps = {nt}\n" + \
                f"total windows = {nw}\n" + \
                f"total nonlinear iterations = {nlits}\n" + \
                f"total linear iterations = {lits}\n" + \
                f"nonlinear iterations per window = {nlits/nw}\n" + \
                f"linear iterations per window = {lits/nw}\n" + \
                f"linear iterations per nonlinear iteration = {lits/nlits}\n" + \
                f"max iterations per block solve = {blits/lits}"
            f.write(info)
    return


def write_block_solve_metrics(pdg, directory=""):
    """
    Write various metrics for the block solves from the paradiag object to file e.g. number of linear iterations, eigenvalues etc

    :arg pdg: paradiag object
    :arg directory: the directory to write the files into
    """
    from asQ import DiagFFTPC
    jacobian = pdg.solver.jacobian
    if not hasattr(jacobian, "pc"):
        return
    elif not isinstance(jacobian.pc, DiagFFTPC):
        return

    from numpy import real, imag, abs
    from numpy import angle as arg

    if not isinstance(directory, str):
        raise ValueError("directory must be string")

    pdg.sync_diagnostics()

    is_root = (pdg.ensemble.global_comm.rank == 0)

    if is_root:
        if (directory != "") and (directory[-1] != "/"):
            directory += "/"
        file_name = directory + "block_metrics.txt"

        with open(file_name, "w") as f:
            lits = pdg.linear_iterations
            blits = pdg.block_iterations._data/lits
            if hasattr(pdg.solver.jacobian, "pc"):
                if hasattr(pdg.solver.jacobian.pc, "D1"):
                    l1 = pdg.solver.jacobian.pc.D1
                    l2 = pdg.solver.jacobian.pc.D2
                else:
                    l1 = float(1/pdg.solver.aaoform.dt)
                    l2 = float(1/pdg.solver.aaoform.theta)
            else:
                l1 = 0
                l2 = 0
            l12 = l1/l2

            # header
            info = \
                f"# iterations per block solve = {blits}\n" + \
                f"# max iterations per block solve = {max(blits)}\n" + \
                f"# min iterations per block solve = {min(blits)}\n" + \
                "# n     " + \
                "R(l1)        I(l1)        abs(l1)      arg(l1)      " + \
                "R(l2)        I(l2)        abs(l2)      arg(l2)      " + \
                "R(l1/l2)     I(l1/l2)     abs(l1/l2)   arg(l1/l2)   " + \
                "its\n"

            # row for each block
            for i, (d1, d2, d12, n) in enumerate(zip(l1, l2, l12, blits)):
                line = "{:3d}"
                for _ in range(3*4 + 1):  # 4 values for each l1, l2, l1/l2, and iteration count
                    line += "   {:10.6f}"
                line += "\n"
                line = line.format(i,
                                   real(d1), imag(d1), abs(d1), arg(d1),
                                   real(d2), imag(d2), abs(d2), arg(d2),
                                   real(d12), imag(d12), abs(d12), arg(d12),
                                   n)
                info += line

            f.write(info)
    return


def write_paradiag_metrics(pdg, directory=""):
    """
    Write various information for the paradiag object to files in the specified directory

    Information written:
    - solver parameters dictionary
    - parameters for paradiag setup (dt, alpha etc)
    - metrics for all-at-once solver
    - metrics for each block solver

    :arg pdg: paradiag object
    :arg directory: the directory to write the files into
    """
    is_root = (pdg.ensemble.global_comm.rank == 0)

    if is_root:
        # write solver parameters
        write_solver_parameters(pdg.solver.solver_parameters, directory)

    # write paradiag setup
    write_paradiag_setup(pdg, directory)

    # write aaos solve metrics
    write_aaos_solve_metrics(pdg, directory)

    # write block solve metrics
    write_block_solve_metrics(pdg, directory)
