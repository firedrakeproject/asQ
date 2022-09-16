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


# dummy finite element forms to initialise the aaos
@pytest.fixture
def form_function():
    # product of first test and first trial functions
    def form(*args):
        return (args[0]*args[len(args)//2])*fd.dx
    return form


@pytest.fixture
def form_mass():
    # product of first test and first trial functions
    def form(*args):
        return (args[0]*args[len(args)//2])*fd.dx
    return form


@pytest.mark.parallel(nprocs=4)
def test_check_index(ensemble, W,
                     form_function, form_mass):
    # prep aaos setup
    slice_length = 3
    assert (fd.COMM_WORLD.size % 2 == 0)
    nslices = fd.COMM_WORLD.size//2
    time_partition = [slice_length for _ in range(nslices)]

    window_length = sum(time_partition)

    dt = 1
    theta = 0.5

    v0 = fd.Function(W).assign(0)

    aaos = asQ.AllAtOnceSystem(ensemble, time_partition,
                               dt, theta,
                               form_function, form_mass,
                               w0=v0)

    max_indices = {
        'component': len(W.split()),
        'slice': slice_length,
        'window': window_length
    }

    for index_type in max_indices.keys():

        inside_range = max_indices[index_type] - 1
        outside_range = max_indices[index_type] + 1

        # check inside +ve range
        try:
            aaos.check_index(inside_range, index_type)
        except ValueError as err:
            assert False, f"{err}"

        # check inside -ve range
        try:
            aaos.check_index(-inside_range, index_type)
        except ValueError as err:
            assert False, f"{err}"

        # check outside +ve range
        with pytest.raises(ValueError):
            aaos.check_index(outside_range, index_type)

        # check outside -ve range
        with pytest.raises(ValueError):
            aaos.check_index(-outside_range, index_type)


@pytest.mark.parallel(nprocs=4)
def test_shift_index(ensemble, V,
                     form_function, form_mass):
    # prep aaos setup
    slice_length = 3
    assert (fd.COMM_WORLD.size % 2 == 0)
    nslices = fd.COMM_WORLD.size//2
    time_partition = [slice_length for _ in range(nslices)]
    time_rank = ensemble.ensemble_comm.rank

    window_length = sum(time_partition)

    dt = 1
    theta = 0.5

    v0 = fd.Function(V).assign(0)

    aaos = asQ.AllAtOnceSystem(ensemble, time_partition,
                               dt, theta,
                               form_function, form_mass,
                               w0=v0)

    max_indices = {
        'slice': slice_length,
        'window': window_length
    }

    # from_range == to_range
    for index_type in max_indices.keys():

        max_index = max_indices[index_type]
        index = max_index - 1

        # +ve index unchanged
        pos_shift = aaos.shift_index(index, from_range=index_type, to_range=index_type)
        assert (pos_shift == index)

        # -ve index changed to +ve
        neg_shift = aaos.shift_index(-index, from_range=index_type, to_range=index_type)
        assert (neg_shift == max_index - index)

    # slice_range -> window_range

    slice_index = 1

    # +ve index in slice range
    window_index = aaos.shift_index(slice_index, from_range='slice', to_range='window')
    assert (window_index == time_rank*slice_length + slice_index)

    # -ve index in slice range
    window_index = aaos.shift_index(-slice_index, from_range='slice', to_range='window')
    assert (window_index == (time_rank+1)*slice_length - slice_index)

    slice_index = slice_length + 1

    # +ve index out of slice range
    with pytest.raises(ValueError):
        window_index = aaos.shift_index(slice_index, from_range='slice', to_range='window')

    # -ve index out of slice range
    with pytest.raises(ValueError):
        window_index = aaos.shift_index(-slice_index, from_range='slice', to_range='window')

    # window_range -> slice_range

    # +ve index in range
    window_index = time_rank*slice_length + 1
    slice_index = aaos.shift_index(window_index, from_range='window', to_range='slice')
    assert (slice_index == 1)

    # -ve index in range
    window_index = -window_length + time_rank*slice_length + 1
    slice_index = aaos.shift_index(window_index, from_range='window', to_range='slice')
    assert (slice_index == 1)

    # +ve index out of slice range
    window_index = ((time_rank + 1) % nslices)*slice_length + 1
    # for some reason pytest.raises doesn't work with this call
    try:
        slice_index = aaos.shift_index(window_index, from_range='window', to_range='slice')
    except ValueError:
        pass

    # -ve index out of slice range
    window_index = -window_index
    # for some reason pytest.raises doesn't work with this call
    try:
        slice_index = aaos.shift_index(-window_index, from_range='window', to_range='slice')
    except ValueError:
        pass

    # reject component indices
    with pytest.raises(ValueError):
        aaos.shift_index(0, from_range='slice', to_range='component')
    with pytest.raises(ValueError):
        aaos.shift_index(0, from_range='component', to_range='slice')


@pytest.mark.parallel(nprocs=4)
def test_set_component(ensemble, mesh, W,
                       form_function, form_mass):
    '''
    test setting a specific component of a timestep
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

    rank = aaos.time_rank
    ncpts = aaos.ncomponents

    for slice_index in range(aaos.slice_partition[rank]):
        window_index = aaos.shift_index(slice_index, from_range='slice', to_range='window')

        for cpt in range(ncpts):
            vcheck = aaos.w_alls[ncpts*slice_index + cpt]

            # set component using slice index
            aaos.set_component(slice_index, cpt, v1.sub(cpt), index_range='slice')
            assert (fd.errornorm(v1.sub(cpt), vcheck) < 1e-12)

            # set component using window index
            aaos.set_component(window_index, cpt, v0.sub(cpt), index_range='window')
            assert (fd.errornorm(v0.sub(cpt), vcheck) < 1e-12)


@pytest.mark.parallel(nprocs=4)
def test_get_component(ensemble, mesh, W,
                       form_function, form_mass):
    '''
    test getting a specific component of a timestep
    only valid if test_set_component passes
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
    ncpts = aaos.ncomponents

    vcheck = fd.Function(W)

    # get each component using slice index
    for step in range(aaos.slice_partition[rank]):
        for cpt in range(ncpts):
            aaos.get_component(step, cpt, index_range='slice', wout=vcheck.sub(cpt))

            err = fd.errornorm(v0.sub(cpt), vcheck.sub(cpt))
            assert (err < 1e-12)

            v2 = aaos.get_component(step, cpt, index_range='slice', deepcopy=True)
            err = fd.errornorm(v2, v0.sub(cpt))
            assert (err < 1e-12)

            v2 = aaos.get_component(step, cpt, index_range='slice')
            assert v2 is aaos.w_alls[step*ncpts + cpt]

    # get each component using window index
    for slice_index in range(aaos.slice_partition[rank]):
        aaos.set_timestep(slice_index, v1, index_range='slice')

        window_index = aaos.shift_index(slice_index, from_range='slice', to_range='window')

        for cpt in range(ncpts):
            aaos.get_component(window_index, cpt, index_range='window', wout=vcheck.sub(cpt))
            err = fd.errornorm(v1.sub(cpt), vcheck.sub(cpt))
            assert (err < 1e-12)

            v2 = aaos.get_component(window_index, cpt, index_range='window', deepcopy=True)
            err = fd.errornorm(v2, v1.sub(cpt))
            assert (err < 1e-12)

            v2 = aaos.get_component(window_index, cpt, index_range='window')
            assert v2 is aaos.w_alls[slice_index*ncpts + cpt]


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

    vcheck = fd.Function(W)
    # get each step using slice index
    for step in range(aaos.slice_partition[rank]):
        aaos.get_timestep(step, index_range='slice', wout=vcheck)
        err = fd.errornorm(v0, vcheck)
        assert (err < 1e-12)

        v2 = aaos.get_timestep(step, index_range='slice')
        err = fd.errornorm(v2, v0)
        assert (err < 1e-12)

    # get each step using window index
    for slice_index in range(aaos.slice_partition[rank]):
        window_index = aaos.shift_index(slice_index,
                                        from_range='slice',
                                        to_range='window')

        aaos.set_timestep(window_index, v1, index_range='window')

        aaos.get_timestep(window_index, index_range='window', wout=vcheck)
        err = fd.errornorm(v1, vcheck)
        assert (err < 1e-12)

        v2 = aaos.get_timestep(window_index, index_range='window')
        err = fd.errornorm(v2, v1)
        assert (err < 1e-12)


@pytest.mark.parallel(nprocs=4)
def test_for_each_timestep(ensemble, V,
                           form_function, form_mass):
    '''
    test aaos.for_each_timestep
    '''
    # prep aaos setup
    M = [2, 2]
    dt = 1
    theta = 0.5

    v0 = fd.Function(V, name="v0")
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
def test_next_window(ensemble, mesh, V,
                     form_function, form_mass):
    # test resetting paradiag to start to next time-window

    # prep paradiag setup
    M = [2, 2]
    dt = 1
    theta = 0.5

    v0 = fd.Function(V, name="v0")
    v1 = fd.Function(V, name="v1")

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
