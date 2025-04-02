from firedrake.petsc import PETSc
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

        # # # slice ensemble # # #

        # slice here means the smaller local ensemble, not the
        # section of the timeseries on the local ensemble member.

        # all ensemble members must have the same number of timesteps
        if len(set(self.time_partition)) != 1:
            msg = f"{type(self).__name__} only implemented for balanced partitions yet"
            raise ValueError(msg)

        # how many timesteps in each slice?
        interval_length = PETSc.Options().getInt(
            f"{self.pc_prefix}interval_length")

        # we need to work out how many members of the global ensemble
        # needed to get `split_size` timesteps on each slice ensemble
        slice_members = interval_length // self.time_partition[0]
        nslices = self.ntimesteps // interval_length

        # create the ensemble for the local slice by splitting the global ensemble
        self.slice_ensemble = split_ensemble(self.ensemble,
                                             split_size=slice_members)

        # which slice are we in?
        self.slice_rank = self.ensemble.ensemble_comm.rank // slice_members

        # the slice partition matches the corresponding
        # slice of the global partition.
        # only works for balanced partitions yet
        part = self.time_partition[0]
        slice_partition = tuple(part for i in range(slice_members))

        # # # slice aaofuncs - jacobian, yslice # # #

        # the slice jacobian is created around a slice
        # aaofunc that views the global aaofunc via val.
        field_function_space = self.aaofunc.field_function_space
        self.slice_func = AllAtOnceFunction(
            self.slice_ensemble, slice_partition,
            field_function_space, val=self.aaofunc)

        # the result is placed in here and then copied
        # out to the global result
        self.yslice = AllAtOnceFunction(
            self.slice_ensemble, slice_partition,
            field_function_space, val=self.y)

        # this is the slice rhs
        self.xslice = AllAtOnceCofunction(
            self.slice_ensemble, slice_partition,
            field_function_space.dual(), val=self.x)

        # # # slice aaoform - jacobian, # # #

        # here we abuse that aaoform.copy will create
        # the new form over the ensemble of the
        # aaofunc kwarg not its own ensemble.
        self.slice_form = self.aaoform.copy(aaofunc=self.slice_func)

        # # # slice parameters # # #
        default_slice_prefix = self.full_prefix
        default_slice_options = get_default_options(
            default_slice_prefix, range(nslices))

        slice_prefix = default_slice_prefix+str(self.slice_rank)

        # default to treating the slice as a PC not a KSP.
        # we can't just use dict.setdefault in case the
        # the ksp_type is specificed with a nested dict.
        has_default_ksp_type = (
            'ksp_type' in default_slice_options
            or ('ksp' in default_slice_options
                and 'type' in default_slice_options['ksp']))

        if not has_default_ksp_type:
            default_slice_options['ksp_type'] = 'preonly'

        self.slice_solver = LinearSolver(
            self.slice_form, appctx=self.appctx,
            options_prefix=slice_prefix,
            solver_parameters=default_slice_options)

        self.initialized = final_initialize

    @profiler()
    def update(self, pc):
        """
        Update the slice states.
        """
        # Update the timestep values.
        # slice_func mostly views aaofunc data,
        # except that slice_func circulant halos
        # do not necessarily coincide with aaofunc
        # circulant halos. This means we must ask
        # slice_func to do the halo update.
        self.slice_func.update_time_halos()

        # # # update the time values
        aaoform = self.aaoform

        # slice initial time
        if self.slice_rank == 0:
            t0 = aaoform.t0
        else:
            t0 = float(aaoform.time[0]) - float(aaoform.dt)
            t0 = self.slice_ensemble.ensemble_comm.bcast(
                t0, root=0)
        self.slice_form.time_update(t0)

        # # # update the slice
        self.slice_solver.jacobian.update()

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

    PETSc options:

    'pc_ijacobi_interval_length': <int>
        The number of timesteps per interval. Must be an integer multiple
        of the number of timesteps on each ensemble member i.e. all
        timesteps on an ensemble member must belong to the same interval.

    'ijacobi_%d': <AllAtOnceSolver options>
        The solver options for the LinearSolver for the %d-th interval,
        enumerated globally.
        Use 'ijacobi_' to set default options for all intervals.
    """
    prefix = "ijacobi_"

    @profiler()
    def apply_impl(self, pc, x, y):
        self.yslice.zero()
        self.slice_solver.solve(self.xslice, self.yslice)


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

    PETSc options:

    'pc_igs_interval_length': <int>
        The number of timesteps per interval. Must be an integer multiple
        of the number of timesteps on each ensemble member i.e. all
        timesteps on an ensemble member must belong to the same interval.

    'igs_%d': <AllAtOnceSolver options>
        The solver options for the LinearSolver for the %d-th interval,
        enumerated globally.
        Use 'igs_' to set default options for all intervals.
    """
    prefix = "igs_"
    pass
