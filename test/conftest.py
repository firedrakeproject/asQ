"""Global test configuration."""

import pytest
from subprocess import check_call


def parallel(item):
    """Run a test in parallel.

    :arg item: The test item to run.
    """
    from mpi4py import MPI
    if MPI.COMM_WORLD.size > 1:
        raise RuntimeError("parallel test can't be run within parallel environment")
    marker = item.get_closest_marker("parallel")
    if marker is None:
        raise RuntimeError("Parallel test doesn't have parallel marker")
    nprocs = marker.kwargs.get("nprocs", 3)
    if nprocs < 2:
        raise RuntimeError("Need at least two processes to run parallel test")

    # Only spew tracebacks on rank 0.
    # Run xfailing tests to ensure that errors are reported to calling process

    call = ["mpiexec", "-n", "1", "python", "-m", "pytest", "--runxfail", "-s", "-q", "%s::%s" % (item.fspath, item.name)]
    call.extend([":", "-n", "%d" % (nprocs - 1), "python", "-m", "pytest", "--runxfail", "--tb=no", "-q",
                 "%s::%s" % (item.fspath, item.name)])
    check_call(call)


def pytest_configure(config):
    """Register an additional marker."""
    config.addinivalue_line(
        "markers",
        "parallel(nprocs): mark test to run in parallel on nprocs processors")
