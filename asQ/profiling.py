import os

# profile memory use
if os.getenv("ASQ_PROFILE_MEM") is not None:
    from memory_profiler import profile
    if os.getenv("ASQ_PROFILE_MEM") == "file":
        # log memory profile to file (one per MPI rank)
        from mpi4py import MPI
        from functools import partial
        rank = MPI.COMM_WORLD.rank
        mprof_file = open(f"asq_memprof_log{rank}.dat", "w")
        memprofile = partial(profile, stream=mprof_file)
    else:
        # log memory profile to stdout
        memprofile = profile

# no memory profile
else:
    def memprofile(func):
        def pass_through(*args, **kwargs):
            return func(*args, **kwargs)
        return pass_through
