import asQ
import firedrake as fd
import numpy as np
import pytest
from functools import reduce
from operator import mul


def assemble(form, *args, **kwargs):
    return fd.assemble(form, *args, **kwargs).riesz_representation(riesz_map='l2')


bc_options = ["no_bcs", "homogeneous_bcs", "inhomogeneous_bcs"]


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize("bc_option", bc_options)
def test_heat_jacobian(bc_option):
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
    time = tuple(fd.Constant(0) for _ in range(aaofunc.ntimesteps))
    for i in range(aaofunc.ntimesteps):
        time[i].assign((i+1)*dt)

    theta = fd.Constant(0.75)

    def form_function(u, v, t):
        return (fd.Constant(1) + t)*fd.inner(fd.grad(u), fd.grad(v))*fd.dx

    def form_mass(u, v):
        return u*v*fd.dx

    if bc_option == "inhomogeneous_bcs":
        bc_val = fd.sin(2*fd.pi*x)
        bc_domain = "on_boundary"
        bcs = [fd.DirichletBC(V, bc_val, bc_domain)]
    elif bc_option == "homogeneous_bcs":
        bc_val = 0.
        bc_domain = 1
        bcs = [fd.DirichletBC(V, bc_val, bc_domain)]
    else:
        bcs = []

    aaoform = asQ.AllAtOnceForm(aaofunc, dt, theta,
                                form_mass, form_function,
                                bcs=bcs)

    # build the all-at-once jacobian
    aaojac = asQ.AllAtOnceJacobian(aaoform)

    # on each time-slice, build the form for the entire timeseries
    full_function_space = reduce(mul, (V for _ in range(sum(time_partition))))
    ufull = fd.Function(full_function_space)

    if bc_option == "no_bcs":
        bcs_full = []
    else:
        bcs_full = [fd.DirichletBC(full_function_space.sub(i),
                                   bc_val, bc_domain)
                    for i in range(sum(time_partition))]

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
        tform += theta*form_function(unp1, v, time[i]) + (1-theta)*form_function(un, v, time[i]-dt)
        if i == 0:
            fullform = tform
        else:
            fullform += tform

    # create the full jacobian
    full_jacobian = fd.derivative(fullform, ufull)

    # evaluate the form at some random state
    np.random.seed(132574)
    for dat in ufull.dat:
        dat.data[:] = np.random.randn(*(dat.data.shape))

    # make sure the state conforms with the boundary conditions
    for bc in bcs_full:
        bc.apply(ufull)

    # copy the data from the full list into the local time slice
    for step in range(aaofunc.nlocal_timesteps):
        windx = aaofunc.transform_index(step, from_range='slice', to_range='window')
        aaofunc[step].assign(ufull.subfunctions[windx])

    # apply the form to some random data
    xfull = fd.Function(full_function_space)
    for dat in xfull.dat:
        dat.data[:] = np.random.randn(*(dat.data.shape))

    # input/output vectors for aaojac
    x = aaofunc.copy(copy_values=False)
    y = aaoform.F.copy(copy_values=False)
    for step in range(aaofunc.nlocal_timesteps):
        windx = aaofunc.transform_index(step, from_range='slice', to_range='window')
        x[step].assign(xfull.subfunctions[windx])

    # assemble the aao jacobian
    with x.global_vec_ro() as xvec, y.global_vec_wo() as yvec:
        aaojac.mult(None, xvec, yvec)

    # assemble the full jacobian
    jac_full = fd.assemble(full_jacobian, bcs=bcs_full,
                           mat_type='matfree')
    mat_full = jac_full.petscmat.getPythonContext()

    yfull = fd.Cofunction(full_function_space.dual())

    with xfull.dat.vec_ro as xvec, yfull.dat.vec_wo as yvec:
        mat_full.mult(None, xvec, yvec)

    # check they match
    for step in range(aaofunc.nlocal_timesteps):
        windx = aaofunc.transform_index(step, from_range='slice', to_range='window')
        yserial = yfull.subfunctions[windx]
        yparallel = y[step].subfunctions[0]
        for pdat, sdat in zip(yparallel.dat, yserial.dat):
            assert np.allclose(pdat.data, sdat.data), f"Timestep {step} of AllAtOnceJacobian action doesn't match component of monolithic residual calculated locally"


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize("bc_option", bc_options)
def test_mixed_heat_jacobian(bc_option):
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
    time = tuple(fd.Constant(0) for _ in range(aaofunc.ntimesteps))
    for i in range(aaofunc.ntimesteps):
        time[i].assign((i+1)*dt)

    theta = fd.Constant(0.75)

    def form_function(u, p, v, q, t):
        return (fd.div(v)*p - fd.div(u)*q)*fd.dx

    def form_mass(u, p, v, q):
        return (fd.inner(u, v) + p*q)*fd.dx

    if bc_option == "inhomogeneous_bcs":
        bc_val = fd.as_vector([fd.sin(2*fd.pi*x), -fd.cos(fd.pi*y)])
        bc_domain = "on_boundary"
        bcs = [fd.DirichletBC(V.sub(0), bc_val, bc_domain)]
    elif bc_option == "homogeneous_bcs":
        bc_val = fd.as_vector([0., 0.])
        bc_domain = "on_boundary"
        bcs = [fd.DirichletBC(V.sub(0), bc_val, bc_domain)]
    else:
        bcs = []

    aaoform = asQ.AllAtOnceForm(aaofunc, dt, theta,
                                form_mass, form_function,
                                bcs=bcs)

    # build the all-at-once jacobian
    aaojac = asQ.AllAtOnceJacobian(aaoform)

    # on each time-slice, build the form for the entire timeseries
    full_function_space = reduce(mul, (V for _ in range(sum(time_partition))))
    ufull = fd.Function(full_function_space)

    if bc_option == "no_bcs":
        bcs_full = []
    else:
        bcs_full = [fd.DirichletBC(full_function_space.sub(2*i),
                                   bc_val, bc_domain)
                    for i in range(sum(time_partition))]

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
        tform += theta*form_function(*unp1, *v, time[i]) + (1-theta)*form_function(*un, *v, time[i]-dt)
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

    # make sure the state conforms with the boundary conditions
    for bc in bcs_full:
        bc.apply(ufull)

    # copy the data from the full list into the local time slice
    for step in range(aaofunc.nlocal_timesteps):
        windices = aaofunc._component_indices(step, to_range='window')
        for cpt in range(2):
            windx = windices[cpt]
            aaofunc[step].subfunctions[cpt].assign(ufull.subfunctions[windx])

    # apply the form to some random data
    xfull = fd.Function(full_function_space)
    for dat in xfull.dat:
        dat.data[:] = np.random.randn(*(dat.data.shape))

    # input/output vectors for aaojac
    x = aaofunc.copy(copy_values=False)
    y = aaoform.F.copy(copy_values=False)
    for step in range(aaofunc.nlocal_timesteps):
        windices = aaofunc._component_indices(step, to_range='window')
        for cpt in range(2):
            windx = windices[cpt]
            x[step].subfunctions[cpt].assign(xfull.subfunctions[windx])

    # assemble the aao jacobian
    with x.global_vec_ro() as xvec, y.global_vec_wo() as yvec:
        aaojac.mult(None, xvec, yvec)

    # assemble the full jacobian
    yfull = assemble(fd.action(full_jacobian, xfull), bcs=bcs_full)

    for step in range(aaofunc.nlocal_timesteps):
        windices = aaofunc._component_indices(step, to_range='window')
        for cpt in range(2):
            windx = windices[cpt]
            yserial = yfull.subfunctions[windx]
            yparallel = y[step].subfunctions[cpt]
            for pdat, sdat in zip(yparallel.dat, yserial.dat):
                assert np.allclose(pdat.data, sdat.data), f"Timestep {step}, component {cpt}, of AllAtOnceJacobian action doesn't match component of monolithic residual calculated locally"


if __name__ == '__main__':
    for bc in bc_options[:]:
        test_heat_jacobian(bc)
