
import os

if os.getenv('ASQ_PROFILE') is not None:
    from memory_profiler import profile
else:
    def profile(func):
        def pass_through(*args, **kwargs):
            return func(*args, **kwargs)
        return pass_through
