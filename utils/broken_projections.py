from firedrake import FunctionSpace, BrokenElement, Function, dx
from firedrake.parloops import par_loop, READ, INC
from math import prod

__all__ = ['BrokenHDivProjector']


class BrokenHDivProjector:
    def __init__(self, V):
        """
        Projects between an HDiv space and it's broken space
        using equal splitting at the facets.

        :arg V: the HDiv space.
        """

        # check V is actually HDiv
        sobolev_space = V.ufl_element().sobolev_space.name
        if sobolev_space != "HDiv":
            raise TypeError(f"V must be in HDiv not {sobolev_space}")

        # create the broken function space
        self.mesh = V.mesh()
        self.V = V
        self.Vb = FunctionSpace(self.mesh, BrokenElement(V.ufl_element()))

        # parloop domain
        shapes = (V.finat_element.space_dimension(),
                  prod(V.shape))

        domain = "{[i,j]: 0 <= i < %d and 0 <= j < %d}" % shapes

        # calculate weights
        weight_instructions = """
        for i, j
            weights[i,j] = weights[i,j] + 1
        end
        """
        weight_kernel = (domain, weight_instructions)

        self.weights = Function(V).assign(0)
        par_loop(weight_kernel, dx, {"weights": (self.weights, INC)})

        # create projection kernel
        projection_instructions = """
        for i, j
            dst[i,j] = dst[i,j] + src[i,j]/weights[i,j]
        end
        """
        self.projection_kernel = (domain, projection_instructions)

    def project(self, src, dst):
        """
        Project from src to dst using equal weighting at the facets.

        :arg src: the Function/Cofunction to be projected from.
        :arg dst: the Function/Cofunction to be projected to.
        """
        dst.assign(0)
        par_loop(self.projection_kernel, dx,
                 {"weights": (self.weights, READ),
                  "src": (src, READ),
                  "dst": (dst, INC)})
