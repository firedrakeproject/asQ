import asQ
import firedrake as fd
import numpy as np
import pytest
from functools import reduce
from operator import mul

bc_opts = ["no_bcs", "homogeneous_bcs", "inhomogeneous_bcs"]


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize("bc_opt", bc_opts)
def test_heat_form(bc_opt):
    """
    Test that assembling the AllAtOnceForm is the same as assembling the
    slice-local part of an all-at-once form for the whole timeseries.
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
    aaofunc.set_all_fields(ics)

    # build the all-at-once form

    dt = fd.Constant(0.01)
    theta = fd.Constant(0.75)

    def form_function(u, v):
        return fd.inner(fd.grad(u), fd.grad(v))*fd.dx

    def form_mass(u, v):
        return u*v*fd.dx

    if bc_opt == "inhomogeneous_bcs":
        bc_val = fd.sin(2*fd.pi*x)
        bc_domain = "on_boundary"
        bcs = [fd.DirichletBC(V, bc_val, bc_domain)]
    elif bc_opt == "homogeneous_bcs":
        bc_val = 0.
        bc_domain = 1
        bcs = [fd.DirichletBC(V, bc_val, bc_domain)]
    else:
        bcs = []

    aaoform = asQ.AllAtOnceForm(aaofunc, dt, theta,
                                form_mass, form_function,
                                bcs=bcs)

    # on each time-slice, build the form for the entire timeseries
    full_function_space = reduce(mul, (V for _ in range(sum(time_partition))))
    ufull = fd.Function(full_function_space)

    if bc_opt == "no_bcs":
        bcs_full = []
    else:
        bcs_full = []
        for i in range(sum(time_partition)):
            bcs_full.append(fd.DirichletBC(full_function_space.sub(i), bc_val, bc_domain))

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

    # evaluate the form on some random data
    np.random.seed(132574)
    for dat in ufull.dat:
        dat.data[:] = np.random.randn(*(dat.data.shape))

    # copy the data from the full list into the local time slice
    for step in range(aaofunc.nlocal_timesteps):
        windx = aaofunc.transform_index(step, from_range='slice', to_range='window')
        aaofunc.set_field(step, ufull.subfunctions[windx])

    # assemble and compare
    aaoform.assemble()
    Ffull = fd.assemble(fullform)
    for bc in bcs_full:
        bc.apply(Ffull, u=ufull)

    for step in range(aaofunc.nlocal_timesteps):
        windx = aaofunc.transform_index(step, from_range='slice', to_range='window')
        userial = Ffull.subfunctions[windx]
        uparallel = aaoform.F.get_component(step, 0)
        err = fd.errornorm(userial, uparallel)
        assert (err < 1e-12)


@pytest.mark.parametrize("bc_opt", bc_opts)
@pytest.mark.parallel(nprocs=4)
def test_mixed_heat_form(bc_opt):
    """
    Test that assembling the AllAtOnceForm is the same as assembling the
    slice-local part of an all-at-once form for the whole timeseries.
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
    aaofunc.set_all_fields(ics)

    # build the all-at-once form

    dt = fd.Constant(0.01)
    theta = fd.Constant(0.75)

    def form_function(u, p, v, q):
        return (fd.div(v)*p - fd.div(u)*q)*fd.dx

    def form_mass(u, p, v, q):
        return (fd.inner(u, v) + p*q)*fd.dx

    if bc_opt == "inhomogeneous_bcs":
        bc_val = fd.as_vector([fd.sin(2*fd.pi*x), -fd.cos(fd.pi*y)])
        bc_domain = "on_boundary"
        bcs = [fd.DirichletBC(V.sub(0), bc_val, bc_domain)]
    elif bc_opt == "homogeneous_bcs":
        bc_val = fd.as_vector([0., 0.])
        bc_domain = "on_boundary"
        bcs = [fd.DirichletBC(V.sub(0), bc_val, bc_domain)]
    else:
        bcs = []

    aaoform = asQ.AllAtOnceForm(aaofunc, dt, theta,
                                form_mass, form_function,
                                bcs=bcs)

    # on each time-slice, build the form for the entire timeseries
    full_function_space = reduce(mul, (V for _ in range(sum(time_partition))))
    ufull = fd.Function(full_function_space)

    if bc_opt == "no_bcs":
        bcs_full = []
    else:
        bcs_full = []
        for i in range(sum(time_partition)):
            bcs_full.append(fd.DirichletBC(full_function_space.sub(2*i),
                                           bc_val,
                                           bc_domain))

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

    # evaluate the form on some random data
    np.random.seed(132574)
    for dat in ufull.dat:
        dat.data[:] = np.random.randn(*(dat.data.shape))

    # copy the data from the full list into the local time slice
    for step in range(aaofunc.nlocal_timesteps):
        for cpt in range(2):
            windx = aaofunc.transform_index(step, cpt, from_range='slice', to_range='window')
            aaofunc.set_component(step, cpt, ufull.subfunctions[windx])

    # assemble and compare
    aaoform.assemble()
    Ffull = fd.assemble(fullform)
    for bc in bcs_full:
        bc.apply(Ffull, u=ufull)

    for step in range(aaofunc.nlocal_timesteps):
        for cpt in range(2):
            windx = aaofunc.transform_index(step, cpt, from_range='slice', to_range='window')
            userial = Ffull.subfunctions[windx]
            uparallel = aaoform.F.get_component(step, cpt)
            err = fd.errornorm(userial, uparallel)
            assert (err < 1e-12)
