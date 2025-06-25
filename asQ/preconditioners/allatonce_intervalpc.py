from firedrake.petsc import PETSc
from firedrake import MixedFunctionSpace, Function
from petsctools import flatten_parameters
from pyop2.mpi import MPI
from pyop2 import MixedDat
from asQ.profiling import profiler
from asQ.ensemble import slice_ensemble
from asQ.allatonce import (
    LinearSolver, AllAtOnceFunction, AllAtOnceCofunction)
from asQ.preconditioners.base import (
    AllAtOncePCBase, get_default_options)

__all__ = ("IntervalJacobiPC", "IntervalGaussSeidelPC")


def slice_aaofunc_fbuf(aaofunc, idxs):
    idxs = [
        aaofunc.transform_index(
            i, from_range='window', to_range='slice')
        for i in idxs]
    V = MixedFunctionSpace([aaofunc.field_function_space
                            for _ in range(len(idxs))])
    dat = MixedDat((sub.dat
                    for idx in idxs
                    for sub in aaofunc[idx].subfunctions))
    return Function(V, val=dat)
    # return Function(
    #     MixedFunctionSpace(
    #         [aaofunc.field_function_space
    #          for _ in range(len(idxs))]),
    #     val=MixedDat(
    #         (sub.dat
    #          for idx in idxs
    #          for sub in aaofunc[idx].subfunctions))
    # )


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

        self.interval_funcs = []
        self.interval_ranks = []
        self.interval_forms = []
        self.interval_solvers = []
        self.yintervals = []
        self.xintervals = []

        for (interval_index,
             interval_ensemble,
             interval_partition,
             interval_timesteps) in zip(self.interval_indices,
                                        self.interval_ensembles,
                                        self.interval_partitions,
                                        self.interval_local_timesteps):

            fbuf = slice_aaofunc_fbuf(self.aaofunc, interval_timesteps)
            interval_func = AllAtOnceFunction(
                interval_ensemble, interval_partition,
                field_function_space, fval=fbuf,
                full_function_space=fbuf.function_space())

            interval_rank = interval_func.time_rank

            # create a view of this interval of y so that
            # the interval solution is written directly
            # into the global solution buffer.
            ybuf = slice_aaofunc_fbuf(self.y, interval_timesteps)
            yinterval = AllAtOnceFunction(
                interval_ensemble, interval_partition,
                field_function_space, fval=ybuf,
                full_function_space=ybuf.function_space())

            # create a view of this interval of x so that
            # the interval rhs is accessed directly from
            # the global solution buffer.
            xbuf = slice_aaofunc_fbuf(self.x, interval_timesteps)
            xinterval = AllAtOnceCofunction(
                interval_ensemble, interval_partition,
                field_function_space.dual(), fval=xbuf,
                full_function_space=xbuf.function_space())

            # # # interval aaoform - jacobian, # # #

            # here we abuse that aaoform.copy will create
            # the new form over the ensemble of the
            # aaofunc kwarg and not its own ensemble.
            interval_form = self.aaoform.copy(aaofunc=interval_func)

            # # # interval parameters # # #
            default_interval_prefix = self.full_prefix
            default_interval_options = get_default_options(
                default_interval_prefix, range(self.nintervals))

            interval_prefix = default_interval_prefix+str(interval_index)

            # default to treating the interval as a PC not a KSP.
            default_interval_options = flatten_parameters(default_interval_options)
            default_interval_options.setdefault("ksp_type", "preonly")

            interval_solver = LinearSolver(
                interval_form, appctx=self.appctx,
                options_prefix=interval_prefix,
                solver_parameters=default_interval_options)

            interval_solver.ksp.incrementTabLevel(1, parent=pc)
            interval_solver.ksp.pc.incrementTabLevel(1, parent=pc)

            self.interval_funcs.append(interval_func)
            self.interval_ranks.append(interval_rank)
            self.interval_forms.append(interval_form)
            self.interval_solvers.append(interval_solver)
            self.xintervals.append(xinterval)
            self.yintervals.append(yinterval)

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

        self.interval_ensembles = []
        self.interval_indices = []
        self.interval_timesteps = []
        self.interval_local_timesteps = []
        self.interval_partitions = []

        # timesteps on the local slice
        local_timesteps = [
            self.layout.offset + j
            for j in range(self.nlocal_timesteps)]

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

                # are any timesteps on this slice in this interval?
                interval_local_timesteps = [
                    j for j in local_timesteps
                    if j in interval_timesteps
                ]

                belongs_to_interval = (len(interval_local_timesteps) > 0)

                interval_ranks = set(
                    self.layout.rank_of(j)
                    for j in interval_timesteps)

                # create interval_ensemble
                maybe_ensemble = slice_ensemble(
                    self.ensemble, interval_ranks)

                if belongs_to_interval:
                    interval_ensemble = maybe_ensemble
                    assert interval_ensemble != MPI.COMM_NULL

                    self.interval_ensembles.append(interval_ensemble)
                    self.interval_indices.append(interval_index)
                    self.interval_timesteps.append(interval_timesteps)
                    self.interval_local_timesteps.append(interval_local_timesteps)

                    # gather interval_time_partition
                    self.interval_partitions.append(
                        interval_ensemble.ensemble_comm.allgather(
                            len(interval_local_timesteps)))

        self.nlocal_intervals = len(self.interval_ensembles)

        # make sure all local timesteps are in one interval each
        for j in local_timesteps:
            participating_intervals = len([
                i for i, interval in enumerate(self.interval_local_timesteps)
                if j in interval])
            if participating_intervals != 1:
                raise ValueError(
                    "All timesteps on all ranks must participate in exactly"
                    f" one interval, not {participating_intervals}")

        # # transform local_timesteps to local index range
        # for isteps in self.interval_local_timesteps:
        #     for j in range(len(isteps)):
        #         isteps[j] = self.aaofunc.transform_index(
        #             isteps[j], from_range='window', to_range='slice')

    @profiler()
    def update(self, pc):
        """
        Update the interval states.
        """
        aaoform = self.aaoform
        # Update the timestep values.
        # interval_func mostly views aaofunc data,
        # except that interval_func circulant halos
        # do not necessarily coincide with aaofunc
        # circulant halos. This means we must ask
        # interval_func to do the halo update.
        for isteps, iensemble, ifunc, iform, isolver in zip(self.interval_local_timesteps,
                                                            self.interval_ensembles,
                                                            self.interval_funcs,
                                                            self.interval_forms,
                                                            self.interval_solvers):
            # update values
            ifunc.update_time_halos()

            # # # update the time values

            # interval initial time
            if ifunc.time_rank == 0:
                i0 = aaoform.aaofunc.transform_index(
                    isteps[0], from_range='window', to_range='slice')
                t0 = float(aaoform.time[i0] - aaoform.dt)
            else:
                t0 = None

            t0 = iensemble.ensemble_comm.bcast(t0, root=0)
            iform.time_update(t0)

            # # # update the interval
            isolver.jacobian.update()


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
        y.zero()
        for isolver, ix, iy in zip(self.interval_solvers,
                                   self.xintervals,
                                   self.yintervals):
            isolver.solve(ix, iy)


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
    def apply_impl_single(self, pc, x, y):
        y.zero()

        first_interval = (self.interval_indices[0] == 0)
        last_interval = (self.interval_indices[0] == self.nintervals - 1)

        first_rank = (self.interval_ranks[0] == 0)
        last_rank = (self.interval_ranks[0] == len(self.xintervals[0].time_partition) - 1)

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
        self.interval_solvers[0].solve(self.xintervals[0], self.yintervals[0])

        if last_rank and not last_interval:
            self.ensemble.send(
                self.yintervals[0][-1], dest=self.time_rank+1,
                tag=self.time_rank+1)

    @profiler()
    def apply_impl(self, pc, x, y):
        y.zero()

        # buffer to store gauss-seidel increment to the rhs from the
        # solution of the previous timestep: rhs = x_{n} - Ay_{n-1}
        Ay_prev = x.uprev.copy(deepcopy=True).zero()
        yprev = y.uprev.copy(deepcopy=True).zero()

        # do we receive a halo?
        #   - not first global rank
        #   - first rank of first local interval
        grank = self.time_rank
        irank0 = self.interval_ranks[0]
        if (grank != 0) and (irank0 == 0):
            self.ensemble.recv(
                yprev,
                source=self.time_rank-1,
                tag=self.time_rank)

        # loop over local intervals
        for i, (isolver, ix, iy, irank, isteps) in enumerate(zip(self.interval_solvers,
                                                                 self.xintervals,
                                                                 self.yintervals,
                                                                 self.interval_ranks,
                                                                 self.interval_local_timesteps)):

            # do we need an increment from the last timestep?
            if (irank == 0) and not (grank == 0 and i == 0):
                # what step are we incrementing?
                n = self.aaofunc.transform_index(
                    isteps[0], from_range='window', to_range='slice')

                yn, assemble = self.jacobian.step_explicit_action(n)

                yn.assign(yprev if (n == 0) else y[n-1])
                assemble(tensor=Ay_prev)
                x[n].assign(x[n] - Ay_prev)

                for bc in self.aaoform.stepwise_bcs[n]:
                    bc.zero(x[n])

            # solve slice
            isolver.solve(ix, iy)

        # do we send a halo?
        #   - not last global rank
        #   - last rank of last local interval
        gsize = self.ensemble.ensemble_comm.size
        ilast_rank = self.interval_ranks[-1]
        ilast_size = self.interval_ensembles[-1].ensemble_comm.size

        if (grank != gsize - 1) and (ilast_rank == ilast_size - 1):
            self.ensemble.send(
                self.yintervals[-1][-1],
                dest=self.time_rank+1,
                tag=self.time_rank+1)
