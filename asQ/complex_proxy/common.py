
from enum import IntEnum

# flags for real and imaginary parts
Part = IntEnum("Part", (("Real", 0), ("Imag", 1)))
re = Part.Real
im = Part.Imag

api_names = ["FiniteElement", "FunctionSpace", "DirichletBC",
             "split", "subfunctions",
             "get_real", "get_imag", "set_real", "set_imag",
             "LinearForm", "BilinearForm", "derivative",
             "Part", "re", "im"]


def _flatten_tree(root, is_leaf, get_children, container=tuple):
    """
    Return the recursively flattened tree below root in the order that the leafs appear in the traversal.

    :arg root: the current root node.
    :arg is_leaf: predicate on root returning True if root has no children.
    :arg get_children: unary op on root that returns an iterable of root's children if is_leaf(root) evaluates False.
    :arg container: the container type to return the flattened tree in.
    """
    if is_leaf(root):
        return container((root,))
    else:
        return container((leaf
                          for child in get_children(root)
                          for leaf in _flatten_tree(child, is_leaf, get_children)))


def _build_twoform(W, z, A, u, split, return_z):
    """
    Return a bilinear Form on the complex FunctionSpace W equal to a complex multiple of a bilinear Form on the real FunctionSpace.
    If z = zr + i*zi is a complex number, u = ur + i*ui is a complex (Trial)Function, and b = br + i*bi is a complex linear Form, we want to construct a Form such that (zA)u=b

    (zA)u = (zr*A + i*zi*A)(ur + i*ui)
          = (zr*A*ur - zi*A*ui) + i*(zr*A*ui + zi*A*ur)

          = | zr*A   -zi*A | | ur | = | br |
            |              | |    |   |    |
            | zi*A    zr*A | | ui | = | bi |

    :arg W: the complex-proxy FunctionSpace
    :arg z: a complex number.
    :arg A: a generator function for a bilinear Form on the real FunctionSpace, callable as A(*u, *v) where u and v are TrialFunctions and TestFunctions on the real FunctionSpace.
    :arg u: a Function or TrialFunction on the complex space
    :arg return_z: If true, return Constants for the real/imaginary parts of z used in the BilinearForm.
    """
    from firedrake import TestFunction, Constant

    v = TestFunction(W)

    ur = split(u, Part.Real)
    ui = split(u, Part.Imag)

    vr = split(v, Part.Real)
    vi = split(v, Part.Imag)

    zr = Constant(z.real)
    zi = Constant(z.imag)

    A11 = zr*A(*ur, *vr)
    A12 = -zi*A(*ui, *vr)
    A21 = zi*A(*ur, *vi)
    A22 = zr*A(*ui, *vi)
    Ac = A11 + A12 + A21 + A22

    if return_z:
        return Ac, zr, zi
    else:
        return Ac


def _build_oneform(W, z, f, split, return_z):
    """
    Return a Linear Form on the complex FunctionSpace W equal to a complex multiple of a linear Form on the real FunctionSpace.
    If z = zr + i*zi is a complex number, v = vr + i*vi is a complex TestFunction, we want to construct the Form:
    <zr*vr,f> + i<zi*vi,f>

    :arg W: the complex-proxy FunctionSpace.
    :arg z: a complex number.
    :arg f: a generator function for a linear Form on the real FunctionSpace, callable as f(*v) where v are TestFunctions on the real FunctionSpace.
    :arg return_z: If true, return Constants for the real/imaginary parts of z used in the LinearForm.
    """
    from firedrake import TestFunction, Constant

    v = TestFunction(W)
    vr = split(v, Part.Real)
    vi = split(v, Part.Imag)

    zr = Constant(z.real)
    zi = Constant(z.imag)

    fr = zr*f(*vr)
    fi = zi*f(*vi)
    fc = fr + fi

    if return_z:
        return fc, zr, zi
    else:
        return fc
