from sys import float_info
import firedrake as fd
from firedrake import op2


def function_maximum(f):
    """
    Find the maximum basis coefficient of a firedrake.Function

    :arg f: a firedrake.Function
    """
    fmax = op2.Global(1, [-float_info.max], dtype=float, comm=f.comm)
    op2.par_loop(op2.Kernel("""
static void maxify(double *a, double *b) {
    a[0] = a[0] < b[0] ? b[0] : a[0];
}
""", "maxify"), f.dof_dset.set, fmax(op2.MAX), f.dat(op2.READ))
    return fmax.data[0]


def function_minimum(f):
    """
    Find the minimum basis coefficient of a firedrake.Function

    :arg f: a firedrake.Function
    """
    fmin = op2.Global(1, [float_info.max], dtype=float, comm=f.comm)
    op2.par_loop(op2.Kernel("""
static void minify(double *a, double *b) {
    a[0] = a[0] > b[0] ? b[0] : a[0];
}
""", "minify"), f.dof_dset.set, fmin(op2.MIN), f.dat(op2.READ))
    return fmin.data[0]


def function_mean(f):
    """
    Find the volume averaged mean of a function.
    """
    mesh = f.function_space().mesh()
    cells = fd.Function(fd.FunctionSpace(mesh, "DG", 0))
    cells.assign(1)
    area = fd.assemble(cells*fd.dx)
    ftotal = fd.assemble(f*fd.dx)
    return ftotal / area


def curl0(u):
    """
    Curl function from y-cpt field to x-z field
    """
    mesh = u.ufl_domain()
    d = sum(mesh.cell_dimension())

    if d == 2:
        # equivalent vector is (0, u, 0)

        # |i   j   k  |
        # |d_x 0   d_z| = (- du/dz, 0, du/dx)
        # |0   u   0  |
        return fd.as_vector([-u.dx(1), u.dx(0)])
    elif d == 3:
        return fd.curl(u)
    else:
        raise NotImplementedError("curl only implemented for 2 or 3 dimensions")


def curl1(u):
    """
    dual curl function from dim-1 forms to dim-2 forms
    """
    mesh = u.ufl_domain()
    d = sum(mesh.cell_dimension())

    if d == 2:
        # we have vector in x-z plane and return scalar
        # representing y component of the curl

        # |i   j   k   |
        # |d_x 0   d_z | = (0, -du_3/dx + du_1/dz, 0)
        # |u_1 0   u_3 |

        return -u[1].dx(0) + u[0].dx(1)
    elif d == 3:
        return fd.curl(u)
    else:
        raise NotImplementedError("curl only implemented for 2 or 3 dimensions")


def cross0(u, w):
    """
    cross product (slice vector field with out-of-slice vector field)
    """

    # |i   j   k   |
    # |u_1 0   u_3 | = (-w*u_3, 0, w*u_1)
    # |0   w   0   |

    mesh = u.ufl_domain()
    d = sum(mesh.cell_dimension())

    if d == 2:
        # cross product of two slice vectors goes into y cpt
        return fd.as_vector([-w*u[1], w*u[0]])
    elif d == 3:
        return fd.cross(u, w)
    else:
        raise NotImplementedError("cross only implemented for 2 or 3 dimensions")


def cross1(u, w):
    """
    cross product (slice vector field with slice vector field)
    """
    mesh = u.ufl_domain()
    d = sum(mesh.cell_dimension())

    if d == 2:
        # cross product of two slice vectors goes into y cpt

        # |i   j   k   |
        # |u_1 0   u_3 | = (0, -u_1*w_3 + u_3*w_1, 0)
        # |w_1 0   w_3 |

        return w[0]*u[1] - w[1]*u[0]
    elif d == 3:
        return fd.cross(u, w)
    else:
        raise NotImplementedError("cross only implemented for 2 or 3 dimensions")
