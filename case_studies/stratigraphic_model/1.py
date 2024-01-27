from firedrake import *

mesh = SquareMesh(100, 100, 100000)

V = FunctionSpace(mesh, 'CG', 1)
u = Function(V)
x, y = SpatialCoordinate(mesh)
b = 90*tanh(1/20000*(x-50000)) + 10*sin(x/4000)*sin(y/3000)

u.interpolate(b)



outfile = File("output.pvd")
outfile.write(u)
