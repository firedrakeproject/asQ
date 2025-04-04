from firedrake.petsc import PETSc, flatten_parameters
from asQ.profiling import profiler
from asQ.ensemble import split_ensemble
from asQ.allatonce import (
    LinearSolver, AllAtOnceFunction, AllAtOnceCofunction)
from asQ.preconditioners.base import (
    AllAtOncePCBase, get_default_options)

__all__ = ("IntervalJacobiPC", "IntervalGaussSeidelPC")


class IntervalJacobiGaussSeidelPCBase(AllAtOncePCBase):
    @profiler()
    def initialize(self, pc, final_initialize=True):
        super().initialize(pc, final_initialize=False)

        # # # interval ensemble # # #

        # interval here means the smaller local ensemble, not the
        # section of the timeseries on the local ensemble member.

        # all ensemble members must have the same number of timesteps
        if len(set(self.time_partition)) != 1:
            msg = f"{type(self).__name__} only implemented for balanced partitions yet"
            raise ValueError(msg)

        # how many timesteps in each interval?
        self.interval_length = PETSc.Options().getInt(
            f"{self.pc_prefix}interval_length")

        if (self.interval_length % self.time_partition[0]) != 0:
            raise ValueError(
                f"Interval length {self.interval_length} must be a"
                f"  multiple of the slice length {self.time_partition[0]}")

        # we need to work out how many members of the global ensemble
        # needed to get `interval_length` timesteps on each interval ensemble
        interval_nmembers = self.interval_length // self.time_partition[0]
        self.nintervals = self.ntimesteps // self.interval_length

        # create the ensemble for the local interval by splitting the global ensemble
        self.interval_ensemble = split_ensemble(
            self.ensemble, split_size=interval_nmembers)

        # which interval are we in?
        self.interval_index = self.ensemble.ensemble_comm.rank // interval_nmembers

        # the interval partition matches the corresponding
        # interval of the global partition.
        # only works for balanced partitions yet
        part = self.time_partition[0]
        interval_partition = tuple(part for i in range(interval_nmembers))

        # # # interval aaofuncs - jacobian, yinterval # # #

        # the interval jacobian is created around a interval
        # aaofunc that views the global aaofunc via val.
        field_function_space = self.aaofunc.field_function_space
        self.interval_func = AllAtOnceFunction(
            self.interval_ensemble, interval_partition,
            field_function_space, val=self.aaofunc)

        self.interval_rank = self.interval_func.time_rank

        # create a view of this interval of y so that
        # the interval solution is written directly
        # into the global solution buffer.
        self.yinterval = AllAtOnceFunction(
            self.interval_ensemble, interval_partition,
            field_function_space, val=self.y)

        # create a view of this interval of x so that
        # the interval rhs is accessed directly from
        # the global solution buffer.
        self.xinterval = AllAtOnceCofunction(
            self.interval_ensemble, interval_partition,
            field_function_space.dual(), val=self.x)

        # # # interval aaoform - jacobian, # # #

        # here we abuse that aaoform.copy will create
        # the new form over the ensemble of the
        # aaofunc kwarg and not its own ensemble.
        self.interval_form = self.aaoform.copy(aaofunc=self.interval_func)

        # # # interval parameters # # #
        default_interval_prefix = self.full_prefix
        default_interval_options = get_default_options(
            default_interval_prefix, range(self.nintervals))

        interval_prefix = default_interval_prefix+str(self.interval_index)

        # default to treating the interval as a PC not a KSP.
        default_interval_options = flatten_parameters(default_interval_options)
        default_interval_options.setdefault("ksp_type", "preonly")

        self.interval_solver = LinearSolver(
            self.interval_form, appctx=self.appctx,
            options_prefix=interval_prefix,
            solver_parameters=default_interval_options)

        self.interval_solver.ksp.incrementTabLevel(1, parent=pc)
        self.interval_solver.ksp.pc.incrementTabLevel(1, parent=pc)

        self.initialized = final_initialize

    @profiler()
    def update(self, pc):
        """
        Update the interval states.
        """
        # Update the timestep values.
        # interval_func mostly views aaofunc data,
        # except that interval_func circulant halos
        # do not necessarily coincide with aaofunc
        # circulant halos. This means we must ask
        # interval_func to do the halo update.
        self.interval_func.update_time_halos()

        # # # update the time values
        aaoform = self.aaoform

        # interval initial time
        if self.interval_index == 0:
            t0 = aaoform.t0
        else:
            t0 = float(aaoform.time[0] - aaoform.dt)
            t0 = self.interval_ensemble.ensemble_comm.bcast(
                t0, root=0)
        self.interval_form.time_update(t0)

        # # # update the interval
        self.interval_solver.jacobian.update()

        return


class IntervalJacobiPC(IntervalJacobiGaussSeidelPCBase):
    """
    A block Jacobi preconditioner where each block (interval)
    is built from several timesteps.

    The all-at-once system is split into several intervals of several
    timesteps each (these do not necessarily have to match the 'slices'
    of the ensemble partition, i.e. the timesteps on a single ensemble
    member).
    The preconditioner is constructed as a block-diagonal system where
    each block is the all-at-once system of an interval. Each block
    (interval) is (approximately) solved with its own KSP.

    The current implementation is limited to interval boundaries that
    coincide with slice boundaries i.e. each slice must belong entirely
    in a single interval (but one interval can include several slices).

    PETSc options:

    'pc_ijacobi_interval_type': <'regular', 'repeating'>
        The method to calculate the intervals.
        - 'regular': all intervals are the same length.
        - 'repeating': interval lengths varying with a repeating pattern,
            for example all odd numbered intervals have length 4 and
            all even numbered intervals have length 8.

    'pc_ijacobi_interval_length': <int,list[int]>
        The number of timesteps per interval.
        - If 'interval_type' is 'regular' then this should be a
          single integer for the length of all intervals.
        - If 'interval_type' is 'repeating' then this should be a
          list of integers specifying the length of each interval
          in the repeating pattern, e.g. '4,8' in the example above.

    'ijacobi_%d': <AllAtOnceSolver options>
        The solver options for the LinearSolver for the %d-th interval,
        enumerated globally.
        Use 'ijacobi_' to set default options for all intervals.
    """
    prefix = "ijacobi_"

    @profiler()
    def apply_impl(self, pc, x, y):
        self.yinterval.zero()
        self.interval_solver.solve(self.xinterval, self.yinterval)


class IntervalGaussSeidelPC(IntervalJacobiGaussSeidelPCBase):
    """
    A block Gauss-Seidel preconditioner where each block (interval)
    is built from several timesteps.

    The all-at-once system is split into several intervals of several
    timesteps each (these do not necessarily have to match the 'slices'
    of the ensemble partition, i.e. the timesteps on a single ensemble
    member).
    The preconditioner is constructed as a block lower-triangular system
    where each diagonal block is the all-at-once system of an interval.
    Each block (interval) is (approximately) solved with its own KSP.

    The current implementation is limited to interval boundaries that
    coincide with slice boundaries i.e. each slice must belong entirely
    in a single interval (but one interval can include several slices).

    PETSc options:

    'pc_igs_interval_type': <'regular', 'repeating'>
        The method to calculate the intervals.
        - 'regular': all intervals are the same length.
        - 'repeating': interval lengths varying with a repeating pattern,
            for example all odd numbered intervals have length 4 and
            all even numbered intervals have length 8.

    'pc_igs_interval_length': <int,list[int]>
        The number of timesteps per interval.
        - If 'interval_type' is 'regular' then this should be a
          single integer for the length of all intervals.
        - If 'interval_type' is 'repeating' then this should be a
          list of integers specifying the length of each interval
          in the repeating pattern, e.g. '4,8' in the example above.

    'igs_%d': <AllAtOnceSolver options>
        The solver options for the LinearSolver for the %d-th interval,
        enumerated globally.
        Use 'igs_' to set default options for all intervals.
    """
    prefix = "igs_"

    def initialize(self, pc, final_initialize=True):
        super().initialize(pc, final_initialize=False)
        if self.jacobian.aaoform.alpha is not None:
            raise ValueError(
                "Cannot apply Gauss-Seidel iterations to a circulant form")
        self.initialized = final_initialize

    @profiler()
    def apply_impl(self, pc, x, y):
        self.yinterval.zero()

        first_interval = (self.interval_index == 0)
        last_interval = (self.interval_index == self.nintervals - 1)

        first_rank = (self.interval_rank == 0)
        last_rank = (self.interval_rank == len(self.xinterval.time_partition) - 1)

        # buffer to store gauss-seidel increment to the rhs from the
        # solution of the previous timestep: rhs = x_{n} - Ay_{n-1}
        Ay_prev = x.uprev.copy(deepcopy=True).zero()

        if first_rank and not first_interval:
            block_bcs, yprev, assemble = self.jacobian.step_explicit_action(0)

            self.ensemble.recv(
                yprev, source=self.time_rank-1,
                tag=self.time_rank)

            for bc in block_bcs:
                bc.zero(yprev)

            assemble(tensor=Ay_prev)
            x[0].assign(x[0] - Ay_prev)

        # solve slice
        self.interval_solver.solve(self.xinterval, self.yinterval)

        if last_rank and not last_interval:
            self.ensemble.send(
                self.yinterval[-1], dest=self.time_rank+1,
                tag=self.time_rank+1)
