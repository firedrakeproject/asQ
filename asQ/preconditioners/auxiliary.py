from asQ.profiling import profiler
from asQ.preconditioners.base import AllAtOncePCBase
from asQ.allatonce.solver import LinearSolver

__all__ = ['AuxiliaryOperatorPC']


class AuxiliaryOperatorPC(AllAtOncePCBase):
    prefix = "aaoaux_"

    @profiler()
    def initialize(self, pc, final_initialize=True):
        super().initialize(pc, final_initialize=False)

        self.solver = LinearSolver(
            self.aaoform, appctx=self.appctx,
            options_prefix=self.full_prefix)

        self.initialized = final_initialize

    @profiler()
    def get_jacobian(self, pc):
        return NotImplementedError(
            f"AuxiliaryOperatorPC child class {type(self)} must implement get_jacobian method")

    @profiler()
    def apply_impl(self, pc, x, y):
        self.solver.solve(x, y)

    @profiler()
    def update(self, pc):
        pass
