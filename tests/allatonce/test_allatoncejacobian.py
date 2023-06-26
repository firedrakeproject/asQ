import asQ
import firedrake as fd
import numpy as np
import pytest
from functools import reduce
from operator import mul


@pytest.mark.parallel(nprocs=4)
def test_heat_jacobian():
    """
    Test that the AllAtOnceJacobian is the same as the action of the
    slice-local part of the derivative of an all-at-once form for
    the whole timeseries.
    """

    # build the all-at-once function
    nslices = fd.COMM_WORLD.size//2
    slice_length = 2

    time_partition = tuple((slice_length for _ in range(nslices)))
    ensemble = asQ.create_ensemble(time_partition, comm=fd.COMM_WORLD)

    mesh = fd.UnitSquareMesh(4, 4, comm=ensemble.comm)
    x, y = fd.SpatialCoordinate(mesh)
    V = fd.FunctionSpace(mesh, "CG", 1)

    aaofunc = asQ.AllAtOnceFunction(ensemble, time_partition, V)

    ics = fd.Function(V, name="ics")
    ics.interpolate(fd.exp(-((x-0.5)**2 + (y-0.5)**2)/0.5**2))
    aaofunc.assign(ics)

    # build the all-at-once form

    dt = fd.Constant(0.01)
    theta = fd.Constant(0.75)

    def form_function(u, v):
        return fd.inner(fd.grad(u), fd.grad(v))*fd.dx

    def form_mass(u, v):
        return u*v*fd.dx

    aaoform = asQ.AllAtOnceForm(aaofunc, dt, theta,
                                form_mass, form_function)

    # build the all-at-once jacobian
    aaojac = asQ.AllAtOnceJacobian(aaoform, aaofunc)

    # on each time-slice, build the form for the entire timeseries
    full_function_space = reduce(mul, (V for _ in range(sum(time_partition))))
    ufull = fd.Function(full_function_space)

    vfull = fd.TestFunction(full_function_space)
    ufulls = fd.split(ufull)
    vfulls = fd.split(vfull)
    for i in range(aaofunc.ntimesteps):
        if i == 0:
            un = ics
        else:
            un = ufulls[i-1]
        unp1 = ufulls[i]
        v = vfulls[i]
        tform = form_mass(unp1 - un, v/dt)
        tform += theta*form_function(unp1, v) + (1-theta)*form_function(un, v)

        if i == 0:
            fullform = tform
        else:
            fullform += tform

    # create the full jacobian
    full_jacobian = fd.derivative(fullform, ufull)

    # evaluate the form on some random data
    np.random.seed(132574)
    for dat in ufull.dat:
        dat.data[:] = np.random.randn(*(dat.data.shape))

    # copy the data from the full list into the local time slice
    for step in range(aaofunc.nlocal_timesteps):
        windx = aaofunc.transform_index(step, from_range='slice', to_range='window')
        aaofunc.set_field(step, ufull.subfunctions[windx])

    # apply the form to some random data
    xfull = fd.Function(full_function_space)
    for dat in xfull.dat:
        dat.data[:] = np.random.randn(*(dat.data.shape))

    # input/output vectors for aaojac
    x = aaofunc.copy()
    y = aaofunc.copy()
    for step in range(aaofunc.nlocal_timesteps):
        windx = aaofunc.transform_index(step, from_range='slice', to_range='window')
        x.set_field(step, xfull.subfunctions[windx])

    # assemble the aao jacobian
    with x.global_vec_ro() as xvec, y.global_vec_wo() as yvec:
        aaojac.mult(None, xvec, yvec)

    # assemble the full jacobian
    yfull = fd.assemble(fd.action(full_jacobian, xfull))

    # check they match

    for step in range(aaofunc.nlocal_timesteps):
        windx = aaofunc.transform_index(step, from_range='slice', to_range='window')
        yserial = yfull.subfunctions[windx]
        yparallel = y.get_component(step, 0)
        err = fd.errornorm(yserial, yparallel)
        assert (err < 1e-12)


@pytest.mark.parallel(nprocs=4)
def test_mixed_heat_jacobian():
    """
    Test that the AllAtOnceJacobian is the same as the action of the
    slice-local part of the derivative of an all-at-once form for
    the whole timeseries.
    """

    # build the all-at-once function
    nslices = fd.COMM_WORLD.size//2
    slice_length = 2

    time_partition = tuple((slice_length for _ in range(nslices)))
    ensemble = asQ.create_ensemble(time_partition, comm=fd.COMM_WORLD)

    mesh = fd.UnitSquareMesh(4, 4, comm=ensemble.comm)
    x, y = fd.SpatialCoordinate(mesh)
    V = fd.MixedFunctionSpace((fd.FunctionSpace(mesh, "BDM", 1),
                               fd.FunctionSpace(mesh, "DG", 0)))

    aaofunc = asQ.AllAtOnceFunction(ensemble, time_partition, V)

    ics = fd.Function(V, name="ics")
    ics.subfunctions[1].interpolate(fd.exp(-((x-0.5)**2 + (y-0.5)**2)/0.5**2))
    aaofunc.assign(ics)

    # build the all-at-once form

    dt = fd.Constant(0.01)
    theta = fd.Constant(0.75)

    def form_function(u, p, v, q):
        return (fd.div(v)*p - fd.div(u)*q)*fd.dx

    def form_mass(u, p, v, q):
        return (fd.inner(u, v) + p*q)*fd.dx

    aaoform = asQ.AllAtOnceForm(aaofunc, dt, theta,
                                form_mass, form_function)

    # build the all-at-once jacobian
    aaojac = asQ.AllAtOnceJacobian(aaoform, aaofunc)

    # on each time-slice, build the form for the entire timeseries
    full_function_space = reduce(mul, (V for _ in range(sum(time_partition))))
    ufull = fd.Function(full_function_space)

    vfull = fd.TestFunction(full_function_space)
    ufulls = fd.split(ufull)
    vfulls = fd.split(vfull)
    for i in range(aaofunc.ntimesteps):
        if i == 0:
            un = fd.split(ics)
        else:
            un = ufulls[2*(i-1):2*i]

        unp1 = ufulls[2*i:2*(i+1)]
        v = vfulls[2*i:2*(i+1)]

        tform = (1/dt)*(form_mass(*unp1, *v) - form_mass(*un, *v))
        tform += theta*form_function(*unp1, *v) + (1-theta)*form_function(*un, *v)

        if i == 0:
            fullform = tform
        else:
            fullform += tform

    # create the full jacobian
    full_jacobian = fd.derivative(fullform, ufull)

    # evaluate the form on some random data
    np.random.seed(132574)
    for dat in ufull.dat:
        dat.data[:] = np.random.randn(*(dat.data.shape))

    # copy the data from the full list into the local time slice
    for step in range(aaofunc.nlocal_timesteps):
        for cpt in range(2):
            windx = aaofunc.transform_index(step, cpt, from_range='slice', to_range='window')
            aaofunc.set_component(step, cpt, ufull.subfunctions[windx])

    # apply the form to some random data
    xfull = fd.Function(full_function_space)
    for dat in xfull.dat:
        dat.data[:] = np.random.randn(*(dat.data.shape))

    # input/output vectors for aaojac
    x = aaofunc.copy()
    y = aaofunc.copy()
    for step in range(aaofunc.nlocal_timesteps):
        for cpt in range(2):
            windx = aaofunc.transform_index(step, cpt, from_range='slice', to_range='window')
            x.set_component(step, cpt, xfull.subfunctions[windx])

    # assemble the aao jacobian
    with x.global_vec_ro() as xvec, y.global_vec_wo() as yvec:
        aaojac.mult(None, xvec, yvec)

    # assemble the full jacobian
    yfull = fd.assemble(fd.action(full_jacobian, xfull))

    for step in range(aaofunc.nlocal_timesteps):
        for cpt in range(2):
            windx = aaofunc.transform_index(step, cpt, from_range='slice', to_range='window')
            yserial = yfull.subfunctions[windx]
            yparallel = y.get_component(step, cpt)
            err = fd.errornorm(yserial, yparallel)
            assert (err < 1e-12)
