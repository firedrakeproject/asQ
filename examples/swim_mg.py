
import firedrake as fd


class ManifoldTransfer(object):
    def __init__(self):
        '''
        Object to manage transfer operators for MG
        where we pull back to a piecewise flat mesh
        before doing transfers
        '''

        # list of flags to say if we've set up before
        self.ready = {}
        self.Ftransfer = fd.TransferManager()  # is this the firedrake warning?

    def prolong(self, coarse, fine):
        Vfine = fd.FunctionSpace(fine.ufl_domain(),
                                 fine.function_space().ufl_element())
        key = Vfine.dim()

        firsttime = self.ready.get(key, None) is None
        if firsttime:
            self.ready[key] = True

            coarse_mesh = coarse.ufl_domain()
            fine_mesh = fine.ufl_domain()
            if not hasattr(coarse_mesh, "coordinates_bk"):
                coarse_mesh.coordinates_bk = fd.Function(coarse_mesh.coordinates)
            if not hasattr(fine_mesh, "coordinates_bk"):
                fine_mesh.coordinates_bk = fd.Function(fine_mesh.coordinates)

        # change to the transfer coordinates for prolongation
        coarse.ufl_domain().coordinates.assign(
            coarse.ufl_domain().transfer_coordinates)
        fine.ufl_domain().coordinates.assign(
            fine.ufl_domain().transfer_coordinates)

        # standard transfer preserves divergence-free subspaces
        self.Ftransfer.prolong(coarse, fine)

        # change back to deformed mesh
        coarse.ufl_domain().coordinates.assign(
            coarse.ufl_domain().coordinates_bk)
        fine.ufl_domain().coordinates.assign(
            fine.ufl_domain().coordinates_bk)

    def restrict(self, fine, coarse):
        Vfine = fd.FunctionSpace(fine.ufl_domain(),
                                 fine.function_space().ufl_element())
        key = Vfine.dim()

        firsttime = self.ready.get(key, None) is None
        if firsttime:
            self.ready[key] = True

            coarse_mesh = coarse.ufl_domain()
            fine_mesh = fine.ufl_domain()
            if not hasattr(coarse_mesh, "coordinates_bk"):
                coarse_mesh.coordinates_bk = fd.Function(coarse_mesh.coordinates)
            if not hasattr(fine_mesh, "coordinates_bk"):
                fine_mesh.coordinates_bk = fd.Function(fine_mesh.coordinates)

        # change to the transfer coordinates for prolongation
        coarse.ufl_domain().coordinates.assign(
            coarse.ufl_domain().transfer_coordinates)
        fine.ufl_domain().coordinates.assign(
            fine.ufl_domain().transfer_coordinates)

        # standard transfer preserves divergence-free subspaces
        self.Ftransfer.restrict(fine, coarse)

        # change back to deformed mesh
        coarse.ufl_domain().coordinates.assign(
            coarse.ufl_domain().coordinates_bk)
        fine.ufl_domain().coordinates.assign(
            fine.ufl_domain().coordinates_bk)

    def inject(self, fine, coarse):
        Vfine = fd.FunctionSpace(fine.ufl_domain(),
                                 fine.function_space().ufl_element())
        key = Vfine.dim()

        firsttime = self.ready.get(key, None) is None
        if firsttime:
            self.ready[key] = True

            coarse_mesh = coarse.ufl_domain()
            fine_mesh = fine.ufl_domain()
            if not hasattr(coarse_mesh, "coordinates_bk"):
                coarse_mesh.coordinates_bk = fd.Function(coarse_mesh.coordinates)
            if not hasattr(fine_mesh, "coordinates_bk"):
                fine_mesh.coordinates_bk = fd.Function(fine_mesh.coordinates)

        # change to the transfer coordinates for prolongation
        coarse.ufl_domain().coordinates.assign(
            coarse.ufl_domain().transfer_coordinates)
        fine.ufl_domain().coordinates.assign(
            fine.ufl_domain().transfer_coordinates)

        # standard transfer preserves divergence-free subspaces
        self.Ftransfer.inject(fine, coarse)

        # change back to deformed mesh
        coarse.ufl_domain().coordinates.assign(
            coarse.ufl_domain().coordinates_bk)
        fine.ufl_domain().coordinates.assign(
            fine.ufl_domain().coordinates_bk)
