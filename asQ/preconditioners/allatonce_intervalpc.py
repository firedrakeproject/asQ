from firedrake.petsc import PETSc, flatten_parameters
from pyop2.mpi import MPI
from asQ.profiling import profiler
from asQ.ensemble import split_ensemble, slice_ensemble
from asQ.allatonce import (
    LinearSolver, AllAtOnceFunction, AllAtOnceCofunction)
from asQ.preconditioners.base import (
    AllAtOncePCBase, get_default_options)

__all__ = ("IntervalJacobiPC", "IntervalGaussSeidelPC")


class IntervalJacobiGaussSeidelPCBase(AllAtOncePCBase):
    @profiler()
    def initialize(self, pc, final_initialize=True):
        super().initialize(pc, final_initialize=False)

        # grab the interval types and lengths
        option_interval_type = f"{self.pc_prefix}interval_type"
        self.interval_type = PETSc.Options().getString(
            option_interval_type, default="regular")

        valid_interval_types = ("regular", "repeating")
        if self.interval_type not in valid_interval_types:
            type_options = " or ".join(valid_interval_types)
            raise ValueError(
                f"{option_interval_type} option for {type(self).__name__}"
                f" must be one of {type_options}, not {self.interval_type}.")

        option_interval_length = f"{self.pc_prefix}interval_length"
        interval_lengths = PETSc.Options().getIntArray(
            option_interval_length)

        # do we have compatible interval types and lengths?
        if self.interval_type == "regular":
            if len(interval_lengths) != 1:
                raise ValueError(
                    f"Must provide {type(self).__name__} exactly one argument to"
                    f" f{option_interval_length} if interval_type is 'regular'.")
        else:
            if len(interval_lengths) == 0:
                raise ValueError(
                    f"Must provide {type(self).__name__} the interval"
                    f" lengths in the {option_interval_length} argument.")

        # set up the interval partition
        self.init_ensembles(pc, interval_lengths)

        # # # interval aaofuncs - jacobian, yinterval # # #

        # the interval jacobian is created around a interval
        # aaofunc that views the global aaofunc via val.
        field_function_space = self.aaofunc.field_function_space
        self.interval_func = AllAtOnceFunction(
            self.interval_ensemble, self.interval_partition,
            field_function_space, val=self.aaofunc)

        self.interval_rank = self.interval_func.time_rank

        # create a view of this interval of y so that
        # the interval solution is written directly
        # into the global solution buffer.
        self.yinterval = AllAtOnceFunction(
            self.interval_ensemble, self.interval_partition,
            field_function_space, val=self.y)

        # create a view of this interval of x so that
        # the interval rhs is accessed directly from
        # the global solution buffer.
        self.xinterval = AllAtOnceCofunction(
            self.interval_ensemble, self.interval_partition,
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

    def init_ensembles(self, pc, interval_lengths):
        """
        Initialise the time partitions and ensembles for each interval.
        """

        # how many intervals per repetition?
        intervals_per_repeat = len(interval_lengths)

        # how long is each repetition?
        repeat_len = sum(interval_lengths)

        if (self.ntimesteps % repeat_len) != 0:
            raise ValueError(
                f"Total length of repeating interval lengths {interval_lengths}"
                f" for {type(self).__name__} must be an exact divisor of the"
                f" total number of timesteps {self.ntimesteps}")

        # how many of the repetitions do we have?
        nrepeats = self.ntimesteps // repeat_len

        self.nintervals = nrepeats*intervals_per_repeat

        interval_initialised = False

        # for i in nintervals:
        for n in range(nrepeats):

            # first timestep of this repetition
            r0 = n*repeat_len

            for i in range(intervals_per_repeat):
                interval_index = n*intervals_per_repeat + i

                # first timestep of this repetition
                j0 = r0 + sum(interval_lengths[:i])

                ilen = interval_lengths[i]

                # what timesteps are in this interval?
                interval_timesteps = [
                    int(j0 + j) for j in range(ilen)]

                # are any of my timesteps in this interval?
                local_timesteps = [
                    self.layout.offset + j
                    for j in range(self.nlocal_timesteps)]

                matching_timesteps = [
                    j in interval_timesteps
                    for j in local_timesteps]

                if any(matching_timesteps):
                    if not all(matching_timesteps):
                        raise ValueError(
                            "All or no timesteps of each slice must be in an interval")
                    belongs_to_interval = True
                else:
                    belongs_to_interval = False

                interval_ranks = set(
                    self.layout.rank_of(j)
                    for j in interval_timesteps)

                # create interval_ensemble
                maybe_ensemble = slice_ensemble(
                    self.ensemble, interval_ranks)

                if belongs_to_interval:
                    assert not interval_initialised

                    interval_ensemble = maybe_ensemble
                    assert interval_ensemble != MPI.COMM_NULL

                    self.interval_ensemble = interval_ensemble
                    self.interval_index = interval_index

                    # gather interval_time_partition
                    self.interval_partition = \
                        interval_ensemble.ensemble_comm.allgather(
                            self.nlocal_timesteps)

                    interval_initialised = True

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
            yprev, assemble = self.jacobian.step_explicit_action(0)

            self.ensemble.recv(
                yprev, source=self.time_rank-1,
                tag=self.time_rank)

            assemble(tensor=Ay_prev)
            x[0].assign(x[0] - Ay_prev)

        # solve slice
        self.interval_solver.solve(self.xinterval, self.yinterval)

        if last_rank and not last_interval:
            self.ensemble.send(
                self.yinterval[-1], dest=self.time_rank+1,
                tag=self.time_rank+1)
