import firedrake as fd
from pyop2.mpi import MPI

from functools import partial

import numpy as np

import asQ


class SerialMiniApp(object):
    def __init__(self,
                 dt, theta,
                 w_initial,
                 form_mass,
                 form_function,
                 solver_parameters):
        '''
        A miniapp to integrate a finite element form forward in time using the implicit theta method

        :arg dt: the timestep
        :arg theta: parameter for the implicit theta method
        :arg w_initial: initial conditions
        :arg form_mass: a function that returns a linear form on w_initial.function_space() providing the mass operator M for the ODE M*w_t + f(w) = 0
        :arg form_function: a function that returns a linear form on w_initial.function_space() providing f(w) for the ODE M*w_t + f(w) = 0
        :arg solver_parameters: options dictionary for nonlinear solver
        '''
        self.dt = dt
        self.theta = theta

        self.initial_condition = w_initial
        self.function_space = w_initial.function_space()

        self.form_mass = form_mass
        self.form_function = form_function

        self.solver_parameters = solver_parameters

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
        w1s = fd.split(w1)
        w0s = fd.split(w0)

        dqdt = form_mass(*w1s, *v) - form_mass(*w0s, *v)

        L = theta*form_function(*w1s, *v) + (1 - theta)*form_function(*w0s, *v)

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


class ComparisonMiniapp(object):
    def __init__(self,
                 ensemble, time_partition,
                 form_mass,
                 form_function,
                 w_initial,
                 dt, theta,
                 alpha,
                 serial_sparameters,
                 parallel_sparameters,
                 boundary_conditions=[],
                 circ=None,
                 block_ctx={}):
        '''
        A miniapp to run the same problem in serial and with paradiag and compare the results

        :arg ensemble: the ensemble communicator
        :arg time_partition: a list of integers, the number of timesteps
        :arg form_mass: a function that returns a linear form on w_initial.function_space() providing the mass operator M for the ODE M*w_t + f(w) = 0
        :arg form_function: a function that returns a linear form on w_initial.function_space() providing f(w) for the ODE M*w_t + f(w) = 0
        :arg w_initial: initial conditions
        :arg dt: the timestep
        :arg theta: parameter for the implicit theta method
        :arg alpha: float, circulant matrix parameter
        :arg serial_sparameters: options dictionary for nonlinear serial solver
        :arg parallel_sparameters: options dictionary for nonlinear parallel solver
        :arg circ: a string describing the option on where to use the
        :arg block_ctx: non-petsc context for solvers.
        '''

        self.ensemble = ensemble
        self.time_partition = time_partition
        self.form_mass = form_mass
        self.form_function = form_function
        self.w_initial = w_initial
        self.dt = dt
        self.theta = theta
        self.alpha = alpha
        self.serial_sparameters = serial_sparameters
        self.parallel_sparameters = parallel_sparameters
        self.boundary_conditions = boundary_conditions
        self.circ = circ
        self.block_ctx = block_ctx

        self.function_space = self.w_initial.function_space()

        self.wserial = fd.Function(self.function_space)
        self.wparallel = fd.Function(self.function_space)

        # set up serial solver
        self.serial_app = SerialMiniApp(dt, theta, w_initial,
                                        form_mass, form_function,
                                        serial_sparameters)

        # set up paradiag
        self.paradiag = asQ.paradiag(ensemble,
                                     form_function, form_mass,
                                     w_initial, dt, theta,
                                     alpha, time_partition,
                                     bcs=boundary_conditions,
                                     solver_parameters=parallel_sparameters,
                                     circ=circ, block_ctx=block_ctx)

    def solve(self, nwindows,
              preproc=lambda srl, pdg, w: None,
              postproc=lambda srl, pdg, w: None,
              pdg_preproc=lambda pdg, w: None,
              pdg_postproc=lambda pdg, w: None,
              serial_preproc=lambda app, it, t: None,
              serial_postproc=lambda app, it, t: None):
        '''
        Solve nwindows*sum(time_partition) timesteps using both serial and paradiag solvers and return an array of the errornorm of the two solutions

        :arg nwindows: the number of time-windows to solve
        '''

        errors = np.zeros(nwindows*sum(self.time_partition))
        window_length = sum(self.time_partition)

        # set up function to calculate errornorm after each timestep

        def step_on_slice(it):
            try:
                self.paradiag.aaos.check_index(it, index_type='slice')
            except ValueError:
                return False
            return True

        def serial_error_postproc(app, it, t, wndw):
            # nonlocal errors

            # only calculate error it timestep it is on this parallel time-slice
            try:
                # try get parallel step (will throw if not here)
                self.paradiag.aaos.get_timestep(it, wout=self.wparallel, index_range='slice')

                # get serial solution at this timestep
                self.wserial.assign(self.serial_app.w0)

                # calculate error
                err = fd.errornorm(self.wserial, self.wparallel)

                # store in full timeseries
                global_timestep = wndw*window_length + it

                errors[global_timestep] = err

            except ValueError:
                pass

            # run the users postprocessing
            serial_postproc(app, it, t)

        for wndw in range(nwindows):

            preproc(self.serial_app, self.paradiag, wndw)

            self.paradiag.aaos.next_window()
            self.paradiag.solve(nwindows=1,
                                preproc=pdg_preproc,
                                postproc=pdg_postproc)

            self.serial_app.solve(nt=window_length,
                                  preproc=serial_preproc,
                                  postproc=partial(serial_error_postproc, wndw=wndw))

            postproc(self.serial_app, self.paradiag, wndw)

        # collect full error series on all ranks
        global_errors = np.zeros_like(errors)
        self.ensemble.ensemble_comm.Allreduce(errors, global_errors, op=MPI.SUM)

        return global_errors
