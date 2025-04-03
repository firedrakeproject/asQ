import firedrake as fd
from asQ.profiling import profiler
from asQ.parallel_arrays import SharedArray
from asQ.preconditioners.base import (
    AllAtOnceBlockPCBase, get_default_options)

__all__ = ("JacobiPC", "GaussSeidelPC")


class JacobiGaussSeidelPCBase(AllAtOnceBlockPCBase):
    """
    Base class for block preconditioners where each block is
    builtfrom the diagonal block of a single timestep from the
    all-at-once Jacobian.  Each block is (approximately) solved
    with its own LinearVariationalSolver.
    """
    valid_jacobian_states = tuple(("state",))

    @profiler()
    def initialize(self, pc, final_initialize=True):
        super().initialize(pc, final_initialize=False)

        # Block n has prefix 'aao<type>_block_{n}', with type\in(jacobi, gs),
        # but we want to be able to set default options for all blocks using
        # 'aao<type>_block'.
        # LinearVariationalSolver will prioritise options it thinks are from
        # the command line (including those in the `inserted_options` database
        # of the AllAtOnceSolver) over the ones passed to __init__, so we pull
        # the default options off the global dict and pass these explicitly to LVS.
        default_block_prefix = f"{self.full_prefix}block_"
        default_block_options = get_default_options(
            default_block_prefix, range(self.ntimesteps))

        # user appctx for the blocks
        block_appctx = self.appctx.get('block_appctx', {})

        # Building the solvers for the diagonal blocks
        self.block_solvers = []
        for n in range(self.nlocal_timesteps):

            # grab the form from the diagonal of the matrix
            if self.aaoform.construct_type == "single_step":
                # respect how many forms we generate
                A = self.jacobian.singlestep_form_implicit
                block_bcs = self.aaoform.singlestep_bcs
            else:
                A = self.jacobian.stepwise_forms_implicit[n]
                block_bcs = self.aaoform.stepwise_bcs[n]

            block_bcs = tuple(
                fd.DirichletBC(
                    self.aaofunc.field_function_space,
                    0*bc.function_arg, bc.sub_domain)
                for bc in block_bcs)

            # pass parameters into PC:
            appctx_block = {
                "dt": self.dt,
                "theta": self.theta,
                "tref": self.aaoform.time[n],
                "uref": self.aaofunc[n],
                "bcs": block_bcs,
                "form_mass": self.form_mass,
                "form_function": self.form_function,
            }

            appctx_block.update(block_appctx)

            # use the global index of this block for the prefix
            ii = self.aaofunc.transform_index(
                n, from_range='slice', to_range='window')

            # The block rhs/solution are the timestep n of the
            # input/output AllAtOnceCofunction/Function
            block_problem = fd.LinearVariationalProblem(
                A, self.x[n], self.y[n],
                bcs=block_bcs,
                constant_jacobian=True)

            block_solver = fd.LinearVariationalSolver(
                block_problem, appctx=appctx_block,
                options_prefix=default_block_prefix+str(ii),
                solver_parameters=default_block_options)

            block_solver.snes.incrementTabLevel(1, parent=pc)
            block_solver.snes.ksp.incrementTabLevel(1, parent=pc)
            block_solver.snes.ksp.pc.incrementTabLevel(1, parent=pc)

            self.block_solvers.append(block_solver)

        self.block_solvers = tuple(self.block_solvers)

        self.block_iterations = SharedArray(
            self.time_partition, dtype=int,
            comm=self.ensemble.ensemble_comm)

        self.initialized = final_initialize

    @profiler()
    def update(self, pc):
        """
        Rebuild the block solvers from the current Jacobian state.
        """
        # the blocks are built from the Jacobian state
        # so we assume that is up to date and just
        # rebuild the matrices in the solvers.
        for block in self.block_solvers:
            block.invalidate_jacobian()


class JacobiPC(JacobiGaussSeidelPCBase):
    """
    A block Jacobi preconditioner where each block is built from
    the diagonal block of a single timestep from the all-at-once
    Jacobian.
    Each block is (approximately) solved with its own LinearVariationalSolver.

    PETSc options:

    'aaojacobi_block_%d': <LinearVariationalSolver options>
        The solver options for the %d'th block, enumerated globally.
        Use 'aaojacobi_block' to set options for all blocks.
        Default is the Firedrake default options.

    If the AllAtOnceSolver's appctx contains a 'block_appctx' dictionary, this is
    added to the appctx of each block solver.  The appctx of each block solver also
    contains the following:
        'blockid': index of the block.
        'dt': timestep of the block.
        'theta': implicit theta value of the block.
        'u0': state around which the block is linearised.
        't0': time at which the block is linearised.
        'bcs': block boundary conditions.
        'form_mass': function used to build the block mass matrix.
        'form_function': function used to build the block stiffness matrix.
    """
    prefix = "aaojacobi_"

    @profiler()
    def apply_impl(self, pc, x, y):
        # x and y are already the rhs and solution vectors of the blocks
        y.zero()
        for n in range(self.nlocal_timesteps):
            if self.aaoform.construct_type == "single_step":
                self.aaoform.singlestep_set_state(n)
            self.block_solvers[n].solve()


class GaussSeidelPC(JacobiGaussSeidelPCBase):
    """
    A block Gauss-Seidel preconditioner where each block
    is built from the diagonal block of a single timestep
    from the all-at-once Jacobian.
    Each block is (approximately) solved with its own LinearVariationalSolver.

    PETSc options:

    'aaogs_block_%d': <LinearVariationalSolver options>
        The solver options for the %d'th block, enumerated globally.
        Use 'aaojacobi_block' to set options for all blocks.
        Default is the Firedrake default options.

    If the AllAtOnceSolver's appctx contains a 'block_appctx' dictionary, this is
    added to the appctx of each block solver.  The appctx of each block solver also
    contains the following:
        'blockid': index of the block.
        'dt': timestep of the block.
        'theta': implicit theta value of the block.
        'u0': state around which the block is linearised.
        't0': time at which the block is linearised.
        'bcs': block boundary conditions.
        'form_mass': function used to build the block mass matrix.
        'form_function': function used to build the block stiffness matrix.
    """
    prefix = "aaogs_"

    @profiler()
    def apply_impl(self, pc, x, y):
        # x and y are already the rhs and solution vectors of the blocks
        y.zero()

        jacobian = self.jacobian

        # buffer to store gauss-seidel increment to the rhs
        # from the solution of the previous timestep
        xprev = fd.Cofunction(x.field_function_space)

        if self.time_rank > 0:
            self.ensemble.recv(
                y.uprev, source=self.time_rank-1,
                tag=self.time_rank)

        for n in range(self.nlocal_timesteps):

            # Explicit action Ax=b of block sub-diagonal
            # bcs,
            explicit_action = jacobian.step_explicit_action(n)

            if any(o is None for o in explicit_action):
                continue

            block_bcs, x_action, assemble = explicit_action

            # 1. zero out the boundary nodes
            for bc in block_bcs:
                bc.zero(x[n])

            # 2. action of A0 on the latest value of previous timestep.
            #   - for first timestep on rank this is the halo.
            #   - for later timesteps this is the previous solution step.
            if (n > 0) or self.aaoform.use_halo:
                x_action.assign(y.uprev if (n == 0) else y[n-1])
                assemble(tensor=xprev)
                x[n].assign(x[n] - xprev)

            # 3. solve diagonal block for this timestep
            # Implicit solve Ax=b of block diagonal
            self.block_solvers[n].solve()

        # nslices not ntimesteps
        last_rank = len(self.time_partition) - 1
        if self.time_rank < last_rank:
            self.ensemble.send(
                y[-1], dest=self.time_rank + 1,
                tag=self.time_rank + 1)
