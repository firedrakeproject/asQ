import asQ
import firedrake as fd
import pytest
import numpy as np
from functools import reduce
from operator import mul


max_ncpts = 3
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


@pytest.mark.parallel(nprocs=4)
def test_transform_index(ensemble, W):
    '''
    test transforming between window, slice, and component indexes
    '''

    # prep aaof setup
    slice_length = 3
    assert (fd.COMM_WORLD.size % 2 == 0)
    nslices = fd.COMM_WORLD.size//2
    time_partition = [slice_length for _ in range(nslices)]
    time_rank = ensemble.ensemble_comm.rank

    ncpts = len(W.subfunctions)

    window_length = sum(time_partition)

    aaof = asQ.AllAtOnceFunction(ensemble, time_partition, W)

    max_indices = {
        'slice': slice_length,
        'window': window_length
    }

    # from_range == to_range
    for index_type in max_indices.keys():

        max_index = max_indices[index_type]
        index = max_index - 1

        # +ve index unchanged
        pos_shift = aaof.transform_index(index, from_range=index_type, to_range=index_type)
        assert (pos_shift == index)

        # -ve index changed to +ve
        neg_shift = aaof.transform_index(-index, from_range=index_type, to_range=index_type)
        assert (neg_shift == max_index - index)

    # slice_range -> window_range

    slice_index = 1

    # +ve index in slice range
    window_index = aaof.transform_index(slice_index, from_range='slice', to_range='window')
    assert (window_index == time_rank*slice_length + slice_index)

    # -ve index in slice range
    window_index = aaof.transform_index(-slice_index, from_range='slice', to_range='window')
    assert (window_index == (time_rank+1)*slice_length - slice_index)

    slice_index = slice_length + 1

    # +ve index out of slice range
    with pytest.raises(IndexError):
        window_index = aaof.transform_index(slice_index, from_range='slice', to_range='window')

    # -ve index out of slice range
    with pytest.raises(IndexError):
        window_index = aaof.transform_index(-slice_index, from_range='slice', to_range='window')

    # window_range -> slice_range

    # +ve index in range
    window_index = time_rank*slice_length + 1
    slice_index = aaof.transform_index(window_index, from_range='window', to_range='slice')
    assert (slice_index == 1)

    # -ve index in range
    window_index = -window_length + time_rank*slice_length + 1
    slice_index = aaof.transform_index(window_index, from_range='window', to_range='slice')
    assert (slice_index == 1)

    # +ve index out of slice range
    window_index = ((time_rank + 1) % nslices)*slice_length + 1
    # for some reason pytest.raises doesn't work with this call
    try:
        slice_index = aaof.transform_index(window_index, from_range='window', to_range='slice')
    except IndexError:
        pass

    # -ve index out of slice range
    window_index = -window_index
    # for some reason pytest.raises doesn't work with this call
    try:
        slice_index = aaof.transform_index(-window_index, from_range='window', to_range='slice')
    except IndexError:
        pass

    # component indices

    # +ve slice and +ve component indices
    slice_index = 1
    cpt_index = ncpts-1
    aao_index = aaof.transform_index(slice_index, cpt_index, from_range='slice')
    check_index = ncpts*slice_index + cpt_index
    assert (aao_index == check_index)

    # +ve window and -ve component indices
    window_index = time_rank*slice_length + 1
    slice_index = aaof.transform_index(window_index, from_range='window', to_range='slice')
    cpt_index = -1
    aao_index = aaof.transform_index(window_index, cpt_index, from_range='window')
    check_index = ncpts*(slice_index+1) + cpt_index

    # reject component index out of range
    with pytest.raises(IndexError):
        aaof.transform_index(0, 100, from_range='slice')
    with pytest.raises(IndexError):
        window_index = aaof.transform_index(0, from_range='slice', to_range='window')
        aaof.transform_index(window_index, -100, from_range='window')


@pytest.mark.parallel(nprocs=4)
def test_set_component(ensemble, mesh, W):
    '''
    test setting a specific component of a timestep
    '''

    # prep aaof setup
    slice_length = 2
    nslices = fd.COMM_WORLD.size//2
    time_partition = [slice_length for _ in range(nslices)]

    v0 = fd.Function(W, name="v0")
    v1 = fd.Function(W, name="v1")

    # two random solutions
    np.random.seed(572046)

    for dat in v0.dat:
        dat.data[:] = np.random.rand(*(dat.data.shape))
    for dat in v1.dat:
        dat.data[:] = np.random.rand(*(dat.data.shape))

    aaof = asQ.AllAtOnceFunction(ensemble, time_partition, W)

    ncpts = aaof.ncomponents

    for slice_index in range(aaof.nlocal_timesteps):
        window_index = aaof.transform_index(slice_index, from_range='slice', to_range='window')

        for cpt in range(ncpts):
            vcheck = aaof.function.subfunctions[ncpts*slice_index + cpt]

            # set component using slice index
            aaof.set_component(slice_index, cpt, v1.subfunctions[cpt], index_range='slice')
            assert (fd.errornorm(v1.subfunctions[cpt], vcheck) < 1e-12)

            # set component using window index
            aaof.set_component(window_index, cpt, v0.subfunctions[cpt], index_range='window')
            assert (fd.errornorm(v0.subfunctions[cpt], vcheck) < 1e-12)


@pytest.mark.parallel(nprocs=4)
def test_get_component(ensemble, mesh, W):
    '''
    test getting a specific component of a timestep
    only valid if test_set_component passes
    '''

    # prep aaof setup
    slice_length = 2
    nslices = fd.COMM_WORLD.size//2
    time_partition = [slice_length for _ in range(nslices)]

    v0 = fd.Function(W, name="v0")
    v1 = fd.Function(W, name="v1")

    # two random solutions
    np.random.seed(572046)

    for dat in v0.dat:
        dat.data[:] = np.random.rand(*(dat.data.shape))
    for dat in v1.dat:
        dat.data[:] = np.random.rand(*(dat.data.shape))

    aaof = asQ.AllAtOnceFunction(ensemble, time_partition, W)
    aaof.set_all_fields(v0)

    ncpts = aaof.ncomponents

    vcheck = fd.Function(W)

    # get each component using slice index
    for step in range(aaof.nlocal_timesteps):
        for cpt in range(ncpts):
            aaof.get_component(step, cpt, index_range='slice', uout=vcheck.subfunctions[cpt])

            err = fd.errornorm(v0.subfunctions[cpt], vcheck.subfunctions[cpt])
            assert (err < 1e-12)

            v2 = aaof.get_component(step, cpt, index_range='slice', deepcopy=True)
            err = fd.errornorm(v2, v0.subfunctions[cpt])
            assert (err < 1e-12)

            v2 = aaof.get_component(step, cpt, index_range='slice')
            assert v2 is aaof.function.subfunctions[step*ncpts + cpt]

    # get each component using window index
    for slice_index in range(aaof.nlocal_timesteps):
        aaof.set_field(slice_index, v1, index_range='slice')

        window_index = aaof.transform_index(slice_index, from_range='slice', to_range='window')

        for cpt in range(ncpts):
            aaof.get_component(window_index, cpt, index_range='window', uout=vcheck.subfunctions[cpt])
            err = fd.errornorm(v1.subfunctions[cpt], vcheck.subfunctions[cpt])
            assert (err < 1e-12)

            v2 = aaof.get_component(window_index, cpt, index_range='window', deepcopy=True)
            err = fd.errornorm(v2, v1.subfunctions[cpt])
            assert (err < 1e-12)

            v2 = aaof.get_component(window_index, cpt, index_range='window')
            assert v2 is aaof.function.subfunctions[slice_index*ncpts + cpt]


@pytest.mark.parallel(nprocs=4)
def test_set_field(ensemble, mesh, W,
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
    vchecks = vcheck.subfunctions

    # set each step using window index
    rank = aaos.time_rank
    ncpt = aaos.ncomponents

    for slice_index in range(aaos.time_partition[rank]):
        window_index = aaos.transform_index(slice_index,
                                            from_range='slice',
                                            to_range='window')
        aaos.set_field(window_index, v1, index_range='window')
        for i in range(ncpt):
            vchecks[i].assign(aaos.w_alls[ncpt*slice_index+i])
        err = fd.errornorm(v1, vcheck)
        assert (err < 1e-12)

    # set each step using slice index
    for step in range(aaos.time_partition[rank]):
        aaos.set_field(step, v0, index_range='slice')

        for i in range(ncpt):
            vchecks[i].assign(aaos.w_alls[ncpt*step+i])

        err = fd.errornorm(v0, vcheck)
        assert (err < 1e-12)


@pytest.mark.parallel(nprocs=4)
def test_get_field(ensemble, mesh, W,
                   form_function, form_mass):
    '''
    test getting a specific timestep
    only valid if test_set_field passes
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

    vcheck = fd.Function(W)
    # get each step using slice index
    for step in range(aaos.time_partition[rank]):
        aaos.get_field(step, index_range='slice', wout=vcheck)
        err = fd.errornorm(v0, vcheck)
        assert (err < 1e-12)

        v2 = aaos.get_field(step, index_range='slice')
        err = fd.errornorm(v2, v0)
        assert (err < 1e-12)

    # get each step using window index
    for slice_index in range(aaos.time_partition[rank]):
        window_index = aaos.transform_index(slice_index,
                                            from_range='slice',
                                            to_range='window')

        aaos.set_field(window_index, v1, index_range='window')

        aaos.get_field(window_index, index_range='window', wout=vcheck)
        err = fd.errornorm(v1, vcheck)
        assert (err < 1e-12)

        v2 = aaos.get_field(window_index, index_range='window')
        err = fd.errornorm(v2, v1)
        assert (err < 1e-12)


@pytest.mark.parallel(nprocs=4)
def test_set_all_fields(ensemble, mesh, W):
    """
    test setting all timesteps/ics to given function.
    """

    # prep aaof setup
    slice_length = 2
    nslices = fd.COMM_WORLD.size//2
    time_partition = [slice_length for _ in range(nslices)]

    v0 = fd.Function(W, name="v0")
    v1 = fd.Function(W, name="v1")

    # two random solutions
    np.random.seed(572046)

    for dat in v0.dat:
        dat.data[:] = np.random.rand(*(dat.data.shape))

    aaof = asQ.AllAtOnceFunction(ensemble, time_partition, W)

    # set next window from new solution
    aaof.set_all_fields(v0)

    # check all timesteps == v0

    for step in range(aaof.nlocal_timesteps):
        v1.assign(0)
        err = fd.errornorm(v0, aaof.get_field(step, uout=v1))
        assert (err < 1e-12)

    err = fd.errornorm(v0, aaof.initial_condition)
    assert (err < 1e-12)

    err = fd.errornorm(v0, aaof.uprev)
    assert (err < 1e-12)


def test_assign(): pass


def test_sync_vec(): pass


def test_sync_function(): pass


@pytest.mark.parallel(nprocs=4)
def test_update_time_halos(ensemble, mesh, V,
                           form_function, form_mass):
    # prep aaos setup
    M = [2, 2]
    dt = 1
    theta = 0.5

    rank = ensemble.ensemble_comm.rank
    size = ensemble.ensemble_comm.size

    v0 = fd.Function(V).assign(-1)
    v1 = fd.Function(V).assign(rank)

    aaos = asQ.AllAtOnceSystem(ensemble, M,
                               dt, theta,
                               form_function, form_mass,
                               w0=v0)

    # test updating own halos
    for step in range(aaos.time_partition[rank]):
        aaos.set_field(step, v1)

    aaos.update_time_halos()

    # solution on previous slice
    v0.assign((rank - 1) % size)

    assert (fd.errornorm(aaos.w_recv, v0) < 1e-12)

    wsend = fd.Function(aaos.function_space).assign(-1)
    wrecv = fd.Function(aaos.function_space).assign(-1)
    wall = fd.Function(aaos.function_space_all)
    walls = wall.subfunctions

    for step in range(aaos.time_partition[rank]):
        walls[step].assign(v1)

    aaos.update_time_halos(wsend=wsend, wrecv=wrecv, walls=walls)

    assert (fd.errornorm(wrecv, v0) < 1e-12)
