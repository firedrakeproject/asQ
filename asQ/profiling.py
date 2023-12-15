import os
from functools import wraps, partial

__all__ = ['profile_type', 'profiler', 'coll_barrier']

_default_profile_type = "PETSC_LOG"

profile_type = os.getenv("ASQ_PROFILE", _default_profile_type)

COLL_BARRIER = os.getenv("ASQ_COLL_BARRIER")


def pass_through(func):
    @wraps(func)
    def pass_through(*args, **kwargs):
        return func(*args, **kwargs)
    return pass_through


# # #
# Set profiling decorator
# # #

# profile timing
if profile_type == "PETSC_LOG":
    from firedrake.petsc import PETSc

    def profiler():
        return PETSc.Log.EventDecorator()

# profile memory use
elif profile_type == "MEMORY":
    from memory_profiler import profile

    def profiler():
        return profile

# profile memory use and output results to one file per rank
elif profile_type == "MEMORY_FILE":
    from memory_profiler import profile
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    mprof_file = open(f"asq_memprof_log{rank}.dat", "w")
    prof = partial(profile, stream=mprof_file)

    def profiler():
        return prof

# do not profile
elif profile_type == "NONE":
    def prof(func):
        return pass_through(func)

    def profiler():
        return prof

# unknown profile type
else:
    import warnings
    warnings.warn(f"Unknown profile type ASQ_PROFILE={profile_type}. Defaulting to no profiling.")

    def prof(func):
        return pass_through(func)

    def profiler():
        return prof

# # #
# Set collective communications barrier
# # #

if COLL_BARRIER is None:
    def coll_barrier(func):
        return pass_through(func)
else:
    def coll_barrier(func, comm_name='ensemble.ensemble_comm'):
        @wraps(func)
        def coll_barrier_decorator(self, *args, **kwargs):
            comm = getattr(self, comm_name)
            comm.Barrier()
            return func(self, *args, **kwargs)
        return coll_barrier_decorator
