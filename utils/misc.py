from sys import float_info
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
