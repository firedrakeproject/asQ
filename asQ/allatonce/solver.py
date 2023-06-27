from firedrake.petsc import PETSc, OptionsManager, flatten_parameters

from asQ.profiling import memprofile
from asQ.allatonce import AllAtOnceJacobian
from asQ.allatonce.mixin import TimePartitionMixin

__all__ = ['AllAtOnceSolver']


class AllAtOnceSolver(TimePartitionMixin):
    @memprofile
    def __init__(self, aaoform, aaofunc,
                 solver_parameters={},
                 appctx={},
                 options_prefix="",
                 jacobian_form=None,
                 jacobian_reference_state=None,
                 pre_function_callback=lambda solver, X: None,
                 post_function_callback=lambda solver, X, F: None,
                 pre_jacobian_callback=lambda solver, X, J: None,
                 post_jacobian_callback=lambda solver, X, J: None):
        """
        Solves an all-at-once form over an all-at-once function.
        """
        self.time_partition_setup(aaofunc.ensemble, aaofunc.time_partition)
        self.aaofunc = aaofunc
        self.aaoform = aaoform

        self.appctx = appctx

        self.jacobian_form = aaoform.copy() if jacobian_form is None else jacobian_form

        # callbacks
        self.pre_function_callback = pre_function_callback
        self.post_function_callback = post_function_callback
        self.pre_jacobian_callback = pre_jacobian_callback
        self.post_jacobian_callback = post_jacobian_callback

        # solver options
        self.solver_parameters = solver_parameters
        self.flat_solver_parameters = flatten_parameters(solver_parameters)
        self.options = OptionsManager(self.flat_solver_parameters, options_prefix)
        options_prefix = self.options.options_prefix

        # snes
        self.snes = PETSc.SNES().create(comm=self.ensemble.global_comm)

        self.snes.setOptionsPrefix(options_prefix)

        # residual vector
        self.F = aaofunc._vec.duplicate()

        def assemble_function(snes, X, F):
            self.pre_function_callback(self, X)
            self.aaoform.assemble(X, tensor=F)
            self.post_function_callback(self, X, F)

        self.snes.setFunction(assemble_function, self.F)

        # Jacobian
        with self.options.inserted_options():
            self.jacobian = AllAtOnceJacobian(self.jacobian_form,
                                              current_state=aaofunc,
                                              reference_state=jacobian_reference_state,
                                              options_prefix=options_prefix,
                                              appctx=appctx)

        jacobian_mat = PETSc.Mat().create(comm=self.ensemble.global_comm)
        jacobian_mat.setType("python")
        sizes = (aaofunc.nlocal_dofs, aaofunc.nglobal_dofs)
        jacobian_mat.setSizes((sizes, sizes))
        jacobian_mat.setPythonContext(self.jacobian)
        jacobian_mat.setUp()
        self.jacobian_mat = jacobian_mat

        def form_jacobian(snes, X, J, P):
            # copy the snes state vector into self.X
            self.pre_jacobian_callback(self, X, J)
            self.jacobian.update(X)
            self.post_jacobian_callback(self, X, J)
            J.assemble()
            P.assemble()

        self.snes.setJacobian(form_jacobian, J=jacobian_mat, P=jacobian_mat)

        # complete the snes setup
        self.options.set_from_options(self.snes)

    @PETSc.Log.EventDecorator()
    @memprofile
    def solve(self, rhs=None):
        with self.aaofunc.global_vec() as gvec, self.options.inserted_options():
            if rhs is None:
                self.snes.solve(None, gvec)
            else:
                with rhs.global_vec_ro() as rvec:
                    self.snes.solve(rvec, gvec)
