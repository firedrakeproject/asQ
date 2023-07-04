import os

profile_type = os.getenv("ASQ_PROFILE")

# default is to profile timing
if profile_type is None:
    profile_type = "PETSC_LOG"

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
    from functools import partial
    rank = MPI.COMM_WORLD.rank
    mprof_file = open(f"asq_memprof_log{rank}.dat", "w")
    prof = partial(profile, stream=mprof_file)

    def profiler():
        return prof

# do not profile
elif profile_type == "NONE":
    def prof(func):
        def pass_through(*args, **kwargs):
            return func(*args, **kwargs)
        return pass_through

    def profiler():
        return prof

# unknown profile type
else:
    import warnings
    warnings.warn(f"Unknown profile type ASQ_PROFILE={profile_type}. Defaulting to no profiling.")

    def prof(func):
        def pass_through(*args, **kwargs):
            return func(*args, **kwargs)
        return pass_through

    def profiler():
        return prof
