import asQ
import firedrake as fd
import pytest
import numpy as np
from functools import reduce
from operator import mul


max_ncpts = 2
ncpts = [i for i in range(1, max_ncpts + 1)]


@pytest.fixture
def ensemble():
    if fd.COMM_WORLD.size == 1:
        return

    return fd.Ensemble(fd.COMM_WORLD, 2)


@pytest.fixture
def mesh(ensemble):
    if fd.COMM_WORLD.size == 1:
        return

    return fd.UnitSquareMesh(4, 4, comm=ensemble.comm)


@pytest.fixture
def V(mesh):
    if fd.COMM_WORLD.size == 1:
        return

    return fd.FunctionSpace(mesh, "CG", 1)


# mixed space
@pytest.fixture(params=ncpts)
def W(request, V):
    if fd.COMM_WORLD.size == 1:
        return

    return reduce(mul, [V for _ in range(request.param)])


@pytest.fixture
def form_function(W):
    # product of first test and first trial functions
    def form(*args):
        return (args[0]*args[len(args)//2])*fd.dx
    return form


@pytest.fixture
def form_mass(W):
    # product of first test and first trial functions
    def form(*args):
        return (args[0]*args[len(W.split())])*fd.dx
    return form


@pytest.mark.parallel(nprocs=4)
def test_for_each_timestep(ensemble, W,
                           form_function, form_mass):
    '''
    test aaos.for_each_timestep
    '''
    if len(W.split()) != 1:
        return

    # prep paradiag setup
    M = [2, 2]
    dt = 1
    theta = 0.5

    v0 = fd.Function(W, name="v0")
    v0.assign(0)

    aaos = asQ.AllAtOnceSystem(ensemble, M,
                               dt, theta,
                               form_function, form_mass,
                               w0=v0)

    def check_timestep(expression, w):
        assert (fd.errornorm(expression, w) < 1e-12)

    # set each timestep as slice timestep index
    for step in range(aaos.slice_partition[aaos.time_rank]):
        v0.assign(step)
        aaos.set_timestep(step, v0, index_range='slice')

    aaos.for_each_timestep(
        lambda wi, si, w: check_timestep(fd.Constant(si), w))

    # set each timestep as window timestep index
    for step in range(aaos.slice_partition[aaos.time_rank]):
        v0.assign(aaos.shift_index(step, from_range='slice', to_range='window'))
        aaos.set_timestep(step, v0, index_range='slice')

    aaos.for_each_timestep(
        lambda wi, si, w: check_timestep(fd.Constant(wi), w))


@pytest.mark.parallel(nprocs=4)
def test_get_timestep(ensemble, mesh, W,
                      form_function, form_mass):
    '''
    test getting a specific timestep
    only valid if test_set_timestep passes
    '''

    # prep paradiag setup
    M = [2, 2]
    dt = 1
    theta = 0.5

    v0 = fd.Function(W, name="v0")
    v1 = fd.Function(W, name="v1")

    # two random solutions
    np.random.seed(572046)

    for dat in v0.dat:
        dat.data[:] = np.random.rand(*(dat.data.shape))
    for dat in v1.dat:
        dat.data[:] = np.random.rand(*(dat.data.shape))

    aaos = asQ.AllAtOnceSystem(ensemble, M,
                               dt, theta,
                               form_function, form_mass,
                               w0=v0)

    # set each step using window index
    rank = aaos.time_rank

    for slice_index in range(aaos.slice_partition[rank]):
        window_index = aaos.shift_index(slice_index,
                                        from_range='slice',
                                        to_range='window')
        aaos.set_timestep(window_index, v1, index_range='window')
        err = fd.errornorm(v1, aaos.get_timestep(window_index, index_range='window'))
        assert (err < 1e-12)

    # set each step using slice index
    vcheck = fd.Function(W)
    for step in range(aaos.slice_partition[rank]):
        aaos.set_timestep(step, v0, index_range='slice')

        aaos.get_timestep(step, index_range='slice', wout=vcheck)
        err = fd.errornorm(v0, vcheck)
        assert (err < 1e-12)


@pytest.mark.parallel(nprocs=4)
def test_set_timestep(ensemble, mesh, W,
                      form_function, form_mass):
    '''
    test setting a specific timestep
    '''

    # prep paradiag setup
    M = [2, 2]
    dt = 1
    theta = 0.5

    v0 = fd.Function(W, name="v0")
    v1 = fd.Function(W, name="v1")

    # two random solutions
    np.random.seed(572046)

    for dat in v0.dat:
        dat.data[:] = np.random.rand(*(dat.data.shape))
    for dat in v1.dat:
        dat.data[:] = np.random.rand(*(dat.data.shape))

    aaos = asQ.AllAtOnceSystem(ensemble, M,
                               dt, theta,
                               form_function, form_mass,
                               w0=v0)

    vcheck = fd.Function(W)
    vchecks = vcheck.split()

    # set each step using window index
    rank = aaos.time_rank
    ncpt = aaos.ncomponents

    for slice_index in range(aaos.slice_partition[rank]):
        window_index = aaos.shift_index(slice_index,
                                        from_range='slice',
                                        to_range='window')
        aaos.set_timestep(window_index, v1, index_range='window')
        for i in range(ncpt):
            vchecks[i].assign(aaos.w_alls[ncpt*slice_index+i])
        err = fd.errornorm(v1, vcheck)
        assert (err < 1e-12)

    # set each step using slice index
    for step in range(aaos.slice_partition[rank]):
        aaos.set_timestep(step, v0, index_range='slice')

        for i in range(ncpt):
            vchecks[i].assign(aaos.w_alls[ncpt*step+i])

        err = fd.errornorm(v0, vcheck)
        assert (err < 1e-12)


@pytest.mark.parallel(nprocs=4)
def test_next_window(ensemble, mesh, W,
                     form_function, form_mass):
    # test resetting paradiag to start to next time-window

    # prep paradiag setup
    M = [2, 2]
    dt = 1
    theta = 0.5

    v0 = fd.Function(W, name="v0")
    v1 = fd.Function(W, name="v1")

    # two random solutions
    np.random.seed(572046)
    for dat0, dat1 in zip(v0.dat, v1.dat):
        dat0.data[:] = np.random.rand(*(dat0.data.shape))
        dat1.data[:] = np.random.rand(*(dat1.data.shape))

    aaos = asQ.AllAtOnceSystem(ensemble, M,
                               dt, theta,
                               form_function, form_mass,
                               w0=v0)

    # set next window from new solution
    aaos.next_window(v1)

    # check all timesteps == v1
    rank = aaos.time_rank

    for step in range(aaos.slice_partition[rank]):
        err = fd.errornorm(v1, aaos.get_timestep(step))
        assert (err < 1e-12)

    # force last timestep = v0
    ncomm = ensemble.ensemble_comm.size
    if rank == ncomm-1:
        aaos.set_timestep(-1, v0)

    # set next window from end of last window
    aaos.next_window()

    # check all timesteps == v0
    for step in range(aaos.slice_partition[rank]):
        err = fd.errornorm(v0, aaos.get_timestep(step))
        assert (err < 1e-12)
