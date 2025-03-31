from firedrake.petsc import PETSc
from asQ.profiling import profiler
from asQ.ensemble import split_ensemble
from asQ.allatonce import (
    LinearSolver, AllAtOnceFunction, AllAtOnceCofunction)
from asQ.preconditioners.base import (
    AllAtOncePCBase, get_default_options)

__all__ = ("SliceJacobiPC", "SliceGaussSeidelPC")


class SliceJacobiPC(AllAtOncePCBase):
    """
    A block Jacobi preconditioner where each block (slice) is built from several timesteps.

    The all-at-once system is split into several slices of several
    timesteps each (these do not necessarily have to match the 'slices'
    of the ensemble partition, i.e. the timesteps on a single ensemble
    member).
    The preconditioner is constructed as a block-diagonal system where
    each block is the all-at-once system of a slice. Each block (slice)
    is (approximately) solved with its own KSP.

    PETSc options:

    'slice_jacobi_nsteps': <int>
        The number of timesteps per slice. Must be an integer multiple
        of the number of timesteps on each ensemble member i.e. all
        timesteps on an ensemble member must belong to the same slice.

    'slice_jacobi_slice_%d': <AllAtOnceSolver options>
        The solver options for the %d'th LinearSolver for each slice, enumerated globally.
        Use 'slice_jacobi_slice' to set options for all blocks.
    """
    prefix = "slice_jacobi_"

    @profiler()
    def initialize(self, pc, final_initialize=True):
        super().initialize(pc, final_initialize=False)

        # # # slice ensemble # # #

        # slice here means the smaller local ensemble, not the
        # section of the timeseries on the local ensemble member.

        # all ensemble members must have the same number of timesteps
        if len(set(self.time_partition)) != 1:
            msg = "SliceJacobiPC only implemented for balanced partitions yet"
            raise ValueError(msg)

        # how many timesteps in each slice?
        slice_size = PETSc.Options().getInt(
            f"{self.full_prefix}nsteps")

        # we need to work out how many members of the global ensemble
        # needed to get `split_size` timesteps on each slice ensemble
        slice_members = slice_size // self.time_partition[0]
        nslices = self.ntimesteps // slice_size

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
        # aaofunc that tracks the global aaofunc
        field_function_space = self.aaofunc.field_function_space
        slice_func = AllAtOnceFunction(
            self.slice_ensemble, slice_partition, field_function_space,
            full_function_space=self.aaofunc.function_space,
            full_dual_space=self.aaofunc.dual_space)
        slice_func.zero()
        self.slice_func = slice_func

        for i in range(self.nlocal_timesteps):
            slice_func[i].assign(self.aaofunc[i])

        # the result is placed in here and then copied
        # out to the global result
        self.yslice = slice_func.copy()

        # this is the slice rhs
        self.xslice = AllAtOnceCofunction(
            self.slice_ensemble, slice_partition,
            field_function_space.dual(),
            full_function_space=self.aaofunc.dual_space,
            full_dual_space=self.aaofunc.function_space)

        # # # slice aaoform - jacobian, # # #

        # here we abuse that aaoform.copy will create
        # the new form over the ensemble of the
        # aaofunc kwarg not its own ensemble.
        self.slice_form = self.aaoform.copy(aaofunc=slice_func)

        # # # slice parameters # # #
        default_slice_prefix = f"{self.full_prefix}slice_"
        default_slice_options = get_default_options(
            default_slice_prefix, range(nslices))

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
            options_prefix=default_slice_prefix+str(self.slice_rank),
            solver_parameters=default_slice_options)

        self.initialized = final_initialize

    @profiler()
    def update(self, pc):
        """
        Update the slice states.
        """
        # # # update the timestep values
        aaofunc = self.aaofunc
        slice_func = self.slice_func

        # slice initial conditions
        aaofunc.update_time_halos()
        if self.slice_rank == 0:
            slice_func.initial_condition.assign(aaofunc.initial_condition)
        else:
            slice_func.initial_condition.assign(aaofunc.uprev)

        # slice values
        for i in range(self.nlocal_timesteps):
            slice_func[i].assign(aaofunc[i])
        slice_func.update_time_halos()

        # # # update the time values
        aaoform = self.aaoform
        slice_form = self.slice_form

        # slice initial time
        if self.slice_rank == 0:
            slice_form.t0.assign(aaoform.t0)
        else:
            slice_form.t0.assign(aaoform.time[0] - aaoform.dt)

        # slice times
        for i in range(self.nlocal_timesteps):
            slice_form.time[i].assign(aaoform.time[i])

        # # # update the slice
        self.slice_solver.jacobian.update()

        return

    @profiler()
    def apply_impl(self, pc, x, y):

        # copy global rhs into slice rhs
        for i in range(self.nlocal_timesteps):
            self.xslice[i].assign(x[i])

        self.yslice.zero()
        self.slice_solver.solve(self.xslice, self.yslice)

        # copy slice result into global result
        for i in range(self.nlocal_timesteps):
            y[i].assign(self.yslice[i])


class SliceGaussSeidelPC(AllAtOncePCBase):
    pass
