import firedrake as fd
from firedrake.petsc import PETSc


class SerialMiniApp(object):
    def __init__(self,
                 dt, theta,
                 w_initial,
                 form_mass,
                 form_function,
                 sparameters):
        '''
        A miniapp to integrate a finite element form forward in time using the implicit theta method
        '''
        self.dt = dt
        self.theta = theta

        self.initial_condition = w_initial
        self.function_space = w_initial.function_space()

        self.form_mass = form_mass
        self.form_function = form_function

        self.solver_parameters = sparameters

        # current and next timesteps
        self.w0 = fd.Function(self.function_space).assign(self.initial_condition)
        self.w1 = fd.Function(self.function_space).assign(self.initial_condition)

        # time integration form
        self.form_full = self.set_theta_form(self.form_mass,
                                             self.form_function,
                                             self.dt, self.theta,
                                             self.w0, self.w1)

        self.nlproblem = fd.NonlinearVariationalProblem(self.form_full, self.w1)

        self.nlsolver = fd.NonlinearVariationalSolver(self.nlproblem,
                                                      solver_parameters=self.solver_parameters)

    def set_theta_form(self, form_mass, form_function, dt, theta, w0, w1):
        '''
        Construct the finite element form for a single step of the implicit theta method
        '''

        dt1 = fd.Constant(1/dt)
        theta = fd.Constant(theta)

        v = fd.TestFunctions(w0.function_space())

        dqdt = form_mass(*(fd.split(w1)), *v) - form_mass(*(fd.split(w0)), *v)

        L = theta*form_function(*(fd.split(w1)), *v) + (1 - theta)*form_function(*(fd.split(w0)), *v)

        return dt1*dqdt + L

    def solve(self, nt,
              preproc=lambda miniapp, it, t: None,
              postproc=lambda miniapp, it, t: None):
        '''
        Integrate forward nt timesteps
        '''
        time = 0
        for step in range(nt):
            preproc(self, step, time)

            self.nlsolver.solve()
            self.w0.assign(self.w1)

            time += self.dt

            postproc(self, step, time)
