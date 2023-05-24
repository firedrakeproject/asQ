import asQ
import firedrake as fd
import pytest
import numpy as np
from functools import reduce
from operator import mul


def random_aaof(aaof, v):
    for dat in v.dat:
        dat.data[:] = np.random.rand(*(dat.data.shape))
    aaof.initial_condition.assign(v)
    for step in range(aaof.nlocal_timesteps):
        for dat in v.dat:
            dat.data[:] = np.random.rand(*(dat.data.shape))
        aaof.set_field(step, v)
    aaof.update_time_halos()


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
def test_set_get_component(ensemble, W):
    '''
    test setting a specific component of a timestep
    '''

    # prep aaof setup
    slice_length = 2
    nslices = fd.COMM_WORLD.size//2
    time_partition = [slice_length for _ in range(nslices)]

    aaof = asQ.AllAtOnceFunction(ensemble, time_partition, W)

    ncpts = aaof.ncomponents

    vc = fd.Function(W, name="vc")
    v0 = fd.Function(W, name="v0")
    v1 = fd.Function(W, name="v1")

    # two random solutions
    np.random.seed(572046)

    for dat in aaof.function.dat:
        dat.data[:] = np.random.rand(*(dat.data.shape))

    for slice_index in range(aaof.nlocal_timesteps):
        window_index = aaof.transform_index(slice_index, from_range='slice', to_range='window')

        for cpt in range(ncpts):
            for dat in v0.dat:
                dat.data[:] = np.random.rand(*(dat.data.shape))
            for dat in v1.dat:
                dat.data[:] = np.random.rand(*(dat.data.shape))

            # set component using slice index
            vc.assign(0)
            aaof.set_component(slice_index, cpt, v0.subfunctions[cpt], index_range='slice')
            aaof.get_component(slice_index, cpt, vc.subfunctions[cpt], index_range='slice')
            assert (fd.errornorm(v0.subfunctions[cpt], vc.subfunctions[cpt]) < 1e-12)

            # set component using window index
            vc.assign(0)
            aaof.set_component(window_index, cpt, v1.subfunctions[cpt], index_range='window')
            aaof.get_component(window_index, cpt, vc.subfunctions[cpt], index_range='window')
            assert (fd.errornorm(v1.subfunctions[cpt], vc.subfunctions[cpt]) < 1e-12)

            # get handle to component index in aaof.function
            vcpt = aaof.get_component(slice_index, cpt)
            for datcpt, dat0 in zip(vcpt.dat, v0.subfunctions[cpt].dat):
                datcpt.data[:] = dat0.data[:]
            aaof.get_component(slice_index, cpt, vc.subfunctions[cpt], index_range='slice')
            assert (fd.errornorm(v0.subfunctions[cpt], vc.subfunctions[cpt]) < 1e-12)


@pytest.mark.parallel(nprocs=4)
def test_set_get_field(ensemble, W):
    '''
    test getting a specific timestep
    only valid if test_set_field passes
    '''
    # prep aaof setup
    slice_length = 2
    nslices = fd.COMM_WORLD.size//2
    time_partition = [slice_length for _ in range(nslices)]

    aaof = asQ.AllAtOnceFunction(ensemble, time_partition, W)

    ncpts = aaof.ncomponents

    vc = fd.Function(W, name="vc")
    v0 = fd.Function(W, name="v1")
    v1 = fd.Function(W, name="v1")

    # two random solutions
    np.random.seed(572046)

    for dat in aaof.function.dat:
        dat.data[:] = np.random.rand(*(dat.data.shape))

    # get each step using slice index
    for step in range(aaof.nlocal_timesteps):
        windx = aaof.transform_index(step, from_range='slice', to_range='window')

        for dat in v0.dat:
            dat.data[:] = np.random.rand(*(dat.data.shape))
        for dat in v1.dat:
            dat.data[:] = np.random.rand(*(dat.data.shape))

        aaof.set_field(step, v0, index_range='slice')
        aaof.get_field(step, vc, index_range='slice')
        err = fd.errornorm(v0, vc)
        assert (err < 1e-12)

        aaof.set_field(step, v1, index_range='slice')
        v = aaof.get_field(step, index_range='slice')
        err = fd.errornorm(v1, v)
        assert (err < 1e-12)

        aaof.set_field(windx, v0, index_range='window')
        aaof.get_field(windx, vc, index_range='window')
        err = fd.errornorm(v0, vc)
        assert (err < 1e-12)

        aaof.set_field(windx, v1, index_range='window')
        v = aaof.get_field(windx, index_range='window')
        err = fd.errornorm(v1, v)
        assert (err < 1e-12)


@pytest.mark.parallel(nprocs=4)
def test_set_all_fields(ensemble, W):
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


@pytest.mark.parallel(nprocs=4)
def test_copy(ensemble, W):
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

    aaof0 = asQ.AllAtOnceFunction(ensemble, time_partition, W)

    # set next window from new solution
    random_aaof(aaof0, v0)
    aaof1 = aaof0.copy()

    # check all timesteps == v0

    for step in range(aaof0.nlocal_timesteps):
        err = fd.errornorm(aaof0.get_field(step, uout=v0),
                           aaof1.get_field(step, uout=v1))
        assert (err < 1e-12)

    err = fd.errornorm(aaof0.initial_condition, aaof1.initial_condition)
    assert (err < 1e-12)

    err = fd.errornorm(aaof0.uprev, aaof1.uprev)
    assert (err < 1e-12)


@pytest.mark.parallel(nprocs=4)
def test_assign(ensemble, W):
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

    aaof0 = asQ.AllAtOnceFunction(ensemble, time_partition, W)
    aaof1 = asQ.AllAtOnceFunction(ensemble, time_partition, W)

    # assign from another aaof
    random_aaof(aaof0, v0)
    aaof1.assign(aaof0)

    for step in range(aaof0.nlocal_timesteps):
        err = fd.errornorm(aaof0.get_field(step, v0),
                           aaof1.get_field(step, v1))
        assert (err < 1e-12)

    err = fd.errornorm(aaof0.initial_condition,
                       aaof1.initial_condition)
    assert (err < 1e-12)

    err = fd.errornorm(aaof0.uprev, aaof1.uprev)
    assert (err < 1e-12)

    # set from PETSc Vec
    random_aaof(aaof0, v0)
    aaof0.sync_vec()
    aaof1.assign(aaof0.vec)

    for step in range(aaof0.nlocal_timesteps):
        err = fd.errornorm(aaof0.get_field(step, v0),
                           aaof1.get_field(step, v1))
        assert (err < 1e-12)

    err = fd.errornorm(aaof0.uprev, aaof1.uprev)
    assert (err < 1e-12)


@pytest.mark.parallel(nprocs=4)
def test_sync_vec(ensemble, W):
    """
    test synchronising the global Vec with the local Functions
    """

    # prep aaof setup
    slice_length = 2
    nslices = fd.COMM_WORLD.size//2
    time_partition = [slice_length for _ in range(nslices)]

    v0 = fd.Function(W, name="v0")
    v1 = fd.Function(W, name="v1")

    aaof0 = asQ.AllAtOnceFunction(ensemble, time_partition, W)
    aaof1 = asQ.AllAtOnceFunction(ensemble, time_partition, W)
    mvec = aaof0._aao_vec()

    # random solutions
    np.random.seed(572046)
    random_aaof(aaof0, v0)
    aaof0.sync_vec()
    aaof1.assign(aaof0.vec)

    for step in range(aaof0.nlocal_timesteps):
        err = fd.errornorm(aaof0.get_field(step, v0),
                           aaof1.get_field(step, v1))
        assert (err < 1e-12)


@pytest.mark.parallel(nprocs=4)
def test_sync_function(ensemble, W):
    """
    test synchronising the global Vec with the local Functions
    """

    # prep aaof setup
    slice_length = 2
    nslices = fd.COMM_WORLD.size//2
    time_partition = [slice_length for _ in range(nslices)]

    v0 = fd.Function(W, name="v0")
    v1 = fd.Function(W, name="v1")

    aaof0 = asQ.AllAtOnceFunction(ensemble, time_partition, W)
    aaof1 = asQ.AllAtOnceFunction(ensemble, time_partition, W)
    mvec = aaof0._aao_vec()

    # random solutions
    np.random.seed(572046)

    random_aaof(aaof0, v0)
    aaof0.sync_vec()
    aaof0.vec.copy(aaof1.vec)
    aaof1.sync_function()

    for step in range(aaof0.nlocal_timesteps):
        err = fd.errornorm(aaof0.get_field(step, v0),
                           aaof1.get_field(step, v1))
        assert (err < 1e-12)


@pytest.mark.parallel(nprocs=4)
def test_update_time_halos(ensemble, W):
    """
    test updating the time halo functions
    """

    # prep aaof setup
    slice_length = 2
    nslices = fd.COMM_WORLD.size//2
    time_partition = [slice_length for _ in range(nslices)]

    v0 = fd.Function(W, name="v0")
    v1 = fd.Function(W, name="v1")

    aaof = asQ.AllAtOnceFunction(ensemble, time_partition, W)

    # test updating own halos
    aaof.set_all_fields(0)

    rank = ensemble.ensemble_comm.rank
    size = ensemble.ensemble_comm.size

    # solution on this slice
    v0.assign(rank)
    # solution on previous slice
    v1.assign((rank - 1) % size)

    # set last field from each slice
    aaof.set_field(-1, v0, index_range='slice')

    aaof.update_time_halos()

    assert (fd.errornorm(aaof.uprev, v1) < 1e-12)
