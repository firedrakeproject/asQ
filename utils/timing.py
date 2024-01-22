from mpi4py import MPI

__all__ = ['SolverTimer']


class SolverTimer:
    def __init__(self):
        self.solve_times = []

    def start_timing(self):
        self.solve_times.append(MPI.Wtime())

    def stop_timing(self):
        etime = MPI.Wtime()
        stime = self.solve_times[-1]
        self.solve_times[-1] = etime - stime

    def total_time(self):
        return sum(self.solve_times)

    def nsolves(self):
        return len(self.solve_times)

    def average_time(self):
        return self.total_time()/self.nsolves()

    def string(self, timesteps_per_solve=1, ndigits=None):
        rnd = lambda x: x if ndigits is None else round(x, ndigits)
        total_time = self.total_time()
        average_time = self.average_time()
        timestep_time = average_time/timesteps_per_solve
        string = ''\
            + f'Total solution time: {rnd(total_time)}\n' \
            + f'Average solve solution time: {rnd(average_time)}\n' \
            + f'Average timestep solution time: {rnd(timestep_time)}'
        return string
