
import firedrake as fd

ensemble = fd.Ensemble(fd.COMM_WORLD, 2)

mesh = fd.UnitIntervalMesh(8, comm=ensemble.comm, name="interval_mesh")

V = fd.FunctionSpace(mesh, "CG", 1)

v = fd.Function(V, name="func").interpolate(
    *(fd.SpatialCoordinate(mesh)))

with fd.CheckpointFile('test.h5', 'w', comm=ensemble.comm) as cfile:
    cfile.save_mesh(mesh)
    cfile.save_function(v)
