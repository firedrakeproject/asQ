import firedrake as fd
from firedrake.petsc import PETSc
import asQ
from asQ.pencil import Pencil, Subcomm

from math import pi
from scipy.fft import rfft, fft
import numpy as np
from functools import partial

fftnorm = 'forward'
rfft = partial(rfft, norm=fftnorm)
fft = partial(fft, norm=fftnorm)

PETSc.Sys.popErrorHandler()
Print = PETSc.Sys.Print


def print_in_order(comm, thing):
    comm.Barrier()
    for rank in range(comm.size):
        if comm.rank == rank:
            print(thing)
        comm.Barrier()
    comm.Barrier()


def to_zero(arr, eps=1e-12):
    arr.real[abs(arr.real) < eps] = 0
    arr.imag[abs(arr.imag) < eps] = 0
    return arr


def transposer(comm, sizes, dtype):
    subcomm = Subcomm(comm, [0, 1])
    pencilA = Pencil(subcomm, sizes, axis=1)
    pencilB = pencilA.pencil(0)
    transfer = pencilA.transfer(pencilB, dtype)
    return transfer


def aaofft_v(transfer, aaofunc, aaofreq, xarray, yarray):
    local_size = [aaofunc.nlocal_timesteps,
                  aaofunc.field_function_space.node_set.size]

    with aaofunc.global_vec_ro() as fvec:
        xarray[:] = fvec.array_r.reshape(local_size)
    Print("xarray:")
    print_in_order(comm, xarray.round(4))

    transfer.forward(xarray, yarray)
    Print("yarray:")
    print_in_order(comm, yarray.round(4))

    fftarray = rfft(yarray, axis=0)
    Print("fft:")
    print_in_order(comm, fftarray.round(4))

    yarray[:] = 0
    yarray[:1+nt//2, :] = np.abs(fftarray)
    Print("abs(fft):")
    print_in_order(comm, yarray.round(4))

    transfer.backward(yarray, xarray)
    Print("xarray:")
    print_in_order(comm, xarray.round(4))

    with aaofreq.global_vec_wo() as fvec:
        fvec.array.reshape(local_size)[:] = xarray[:]


def aaofft(transfer, aaofunc, aaofreq, xarray, yarray):
    local_size = [aaofunc.nlocal_timesteps,
                  aaofunc.field_function_space.node_set.size]

    with aaofunc.global_vec_ro() as fvec:
        xarray[:] = fvec.array_r.reshape(local_size)[:]

    transfer.forward(xarray, yarray)

    fftarray = rfft(yarray, axis=0)

    yarray[:1+nt//2, :] = np.abs(fftarray)
    yarray[1+nt//2:, :] = 0

    transfer.backward(yarray, xarray)

    with aaofreq.global_vec_wo() as fvec:
        fvec.array.reshape(local_size)[:] = xarray[:]


nlocal = 4
time_partition = tuple((nlocal, nlocal))

ensemble = asQ.create_ensemble(time_partition)
comm = ensemble.ensemble_comm
rank = comm.rank

mesh = fd.UnitIntervalMesh(2, comm=ensemble.comm)
V = fd.FunctionSpace(mesh, "DG", 0)
aaofunc = asQ.AllAtOnceFunction(ensemble, time_partition, V)
aaofreq = aaofunc.copy()
nt = aaofunc.ntimesteps

local_size = [nlocal, V.node_set.size]
aaosizes = [nt, V.node_set.size]
transfer = transposer(ensemble.ensemble_comm, aaosizes, dtype=float)
xarray = np.zeros(transfer.subshapeA, dtype=transfer.dtype)
yarray = np.zeros(transfer.subshapeB, dtype=transfer.dtype)
Print(transfer.subshapeA)
Print(transfer.subshapeB)

x = np.linspace(0, 2*pi, nt, endpoint=False)
with aaofunc.function.dat.vec as fvec:
    farr = fvec.array.reshape(local_size)
    farr[:, 0] = (np.sin(3*x) + np.sin(x))[rank*nlocal:(rank+1)*nlocal]
    farr[:, 1] = (np.cos(3*x) + np.sin(2*x-2))[rank*nlocal:(rank+1)*nlocal] + 1


aaofft(transfer, aaofunc, aaofreq, xarray, yarray)

norms = [fd.norm(aaofreq[i]).round(4) for i in range(nlocal)]
Print("norms:")
print_in_order(comm, norms)
