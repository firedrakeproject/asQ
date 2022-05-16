import firedrake as fd
from petsc4py import PETSc

from utils.shallow_water.verifications.williamson5 import serial_solve, parallel_solve

PETSc.Sys.popErrorHandler()

# get command arguments
import argparse
parser = argparse.ArgumentParser(description='Williamson 5 testcase for approximate Schur complement solver.')
parser.add_argument('--base_level', type=int, default=1, help='Base refinement level of icosahedral grid for MG solve. Default 1.')
parser.add_argument('--ref_level', type=int, default=2, help='Refinement level of icosahedral grid. Default 2.')
parser.add_argument('--nslices', type=int, default=2, help='Number of time-slices. Default 2.')
parser.add_argument('--slice_length', type=int, default=2, help='Number of timesteps per time-slice. Default 2.')
parser.add_argument('--nspatial_domains', type=int, default=2, help='Size of spatial partition. Default 2.')
parser.add_argument('--alpha', type=float, default=0.0001, help='Circulant coefficient. Default 0.0001.')
parser.add_argument('--dt', type=float, default=0.05, help='Timestep in hours. Default 0.05.')
parser.add_argument('--filename', type=str, default='w5diag')
parser.add_argument('--coords_degree', type=int, default=3, help='Degree of polynomials for sphere mesh approximation.')
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space).')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

M = [args.slice_length for _ in range(args.nslices)]

nsteps = sum(M)

# mesh set up
ensemble = fd.Ensemble(fd.COMM_WORLD, args.nspatial_domains)

r = ensemble.ensemble_comm.rank

# list of serial timesteps
PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Calculating serial solution --- === ###')
PETSc.Sys.Print('')

wserial = serial_solve(base_level=args.base_level,
                       ref_level=args.ref_level,
                       tmax=nsteps,
                       dumpt=1,
                       dt=args.dt,
                       coords_degree=args.coords_degree,
                       degree=args.degree,
                       comm=ensemble.comm,
                       verbose=False)

# only keep the timesteps on the current time-slice
timestep_start = sum(M[:r])
timestep_end = timestep_start + M[r]

wserial = wserial[timestep_start:timestep_end]

PETSc.Sys.Print('')


PETSc.Sys.Print('### === --- Calculating parallel solution --- === ###')
PETSc.Sys.Print('')

wparallel = parallel_solve(base_level=args.base_level,
                           ref_level=args.ref_level,
                           M=M,
                           dumpt=1,
                           dt=args.dt,
                           coords_degree=args.coords_degree,
                           degree=args.degree,
                           ensemble=ensemble,
                           alpha=args.alpha,
                           verbose=True)


PETSc.Sys.Print('### === --- Comparing solutions --- === ###')
PETSc.Sys.Print('')

W = wserial[0].function_space()

ws = fd.Function(W)
wp = fd.Function(W)

us, hs = ws.split()
up, hp = wp.split()

for i in range(M[r]):

    tstep = sum(M[:r]) + i

    us.assign(wserial[i].split()[0])
    hs.assign(wserial[i].split()[1])

    up.assign(wparallel[i].split()[0])
    hp.assign(wparallel[i].split()[1])

    herror = fd.errornorm(hs, hp)/fd.norm(hs)
    uerror = fd.errornorm(us, up)/fd.norm(us)

    PETSc.Sys.Print('timestep:', tstep, '|', 'uerror:', uerror, '|', 'herror: ', herror, comm=ensemble.comm)

PETSc.Sys.Print('')
