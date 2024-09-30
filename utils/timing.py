from mpi4py import MPI

__all__ = ['Timer', 'SolverTimer']


class Timer:
    '''
    Time multiple similar actions.
    '''
    def __init__(self):
        self.times = []

    def start_timing(self):
        '''
        Start timing an action. This should be the last statement before the action starts.
        '''
        self.times.append(MPI.Wtime())

    def stop_timing(self):
        '''
        Stop timing an action. This should be the first statement after the action stops.
        '''
        etime = MPI.Wtime()
        stime = self.times[-1]
        self.times[-1] = etime - stime

    def total_time(self):
        '''
        The total duration of all actions timed.
        '''
        return sum(self.times)

    def ntimes(self):
        '''
        The total number of actions timed.
        '''
        return len(self.times)

    def average_time(self):
        '''
        The average duration of an action.
        '''
        return self.total_time()/self.ntimes()


class SolverTimer(Timer):
    '''
    Time multiple solves and print out total/average etc times.
    '''
    def string(self, timesteps_per_solve=1,
               total_iterations=1, ndigits=None):
        rnd = lambda x: x if ndigits is None else round(x, ndigits)
        total_time = self.total_time()
        average_time = self.average_time()
        timestep_time = average_time/timesteps_per_solve
        iteration_time = total_time/total_iterations
        string = ''\
            + f'Total solution time: {rnd(total_time)}\n' \
            + f'Average solve solution time: {rnd(average_time)}\n' \
            + f'Average timestep solution time: {rnd(timestep_time)}\n' \
            + f'Average iteration solution time: {rnd(iteration_time)}'
        return string
