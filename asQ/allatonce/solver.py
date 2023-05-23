import firedrake as fd
from firedrake.petsc import PETSc

__all__ = ['AllAtOnceSolver']


class AllAtOnceSolver(TimePartitionMixin):
    def __init__(self, aaoform, aaofunc,
                 solver_parameters={},
                 options_prefix=None,
                 jacobian_form=None,
                 jacobian_reference_state=None,
                 pre_function_callback=None,
                 post_function_callback=None,
                 pre_jacobian_callback=None,
                 post_jacobian_callback=None):
        """
        Solves an all-at-once form over an all-at-once function.
        """
        self.time_partition_setup(aaofunc.ensemble, aaofunc.time_partition)
        self.aaofunc = aaofunc
        self.aaoform = aaoform
        self.jacobian_form = aaoform if jacobian_form is None else jacobian_form

        # callbacks
        self.pre_function_callback = pre_function_callback 
        self.post_function_callback = post_function_callback
        self.pre_jacobian_callback = pre_jacobian_callback 
        self.post_jacobian_callback = post_jacobian_callback

        # solver options
        self.solver_parameters = solver_parameters
        self.flat_solver_parameters = flatten_parameters(solver_parameters)
        self.options = OptionsManager(self.flat_solver_parameters, '')

        # dof counts
        nlocal_space_dofs = self.field_function_space.node_set.size
        nspace_dofs = self.field_function_space.dim()
        nlocal = self.nlocal_timesteps*nlocal_space_dofs  # local times x local space
        nglobal = self.ntimesteps*nspace_dofs  # global times x global space

        # snes
        self.snes = PETSc.SNES().create(comm=self.ensemble.global_comm)
        self.snes.setOptionsPrefix(options_prefix)

        # residual function
        self.F = aaofunc.vector.copy()

        def assemble_function(snes, X, Fvec):
            self.pre_function_callback(X)
            self.aaoform.assemble(snes, X, Fvec)
            self.post_function_callback(X, Fvec)

        self.snes.setFunction(assemble_function, self.F)

        # Jacobian
        with self.options.inserted_options():
            self.jacobian = AllAtOnceJacobian(aaofunc,
                                              self.jacobian_form,
                                              reference_state=jacobian_reference_state
                                              snes=snes)

        mat = PETSc.Mat().create(comm=ensemble.global_comm)
        mat.setType("python")
        mat.setSizes(((nlocal, nglobal), (nlocal, nglobal)))
        mat.setPythonContext(self.jacobian)
        mat.setUp()
        self.mat = mat

        def form_jacobian(snes, X, J, P):
            # copy the snes state vector into self.X
            self.pre_jacobian_callback(X)
            self.jacobian.update(X)
            self.post_jacobian_callback(X, J)
            J.assemble()
            P.assemble()

        self.snes.setJacobian(form_jacobian, J=mat, P=mat)

    def solve(self):

        self.aaofunc.sync_vector()

        with self.options.inserted_options():
            self.snes.solve(None, self.aaofunc.vector)

        self.aaofunc.sync_function()
