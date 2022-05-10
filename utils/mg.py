
import firedrake as fd


# transfer between mesh levels on a manifold
# by first ensuring meshes are nested
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
                coordinates_bk_c = fd.Function(coarse_mesh.coordinates)
                coarse_mesh.coordinates_bk = coordinates_bk_c
            if not hasattr(fine_mesh, "coordinates_bk"):
                coordinates_bk_f = fd.Function(fine_mesh.coordinates)
                fine_mesh.coordinates_bk = coordinates_bk_f

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
                coordinates_bk_c = fd.Function(coarse_mesh.coordinates)
                coarse_mesh.coordinates_bk = coordinates_bk_c
            if not hasattr(fine_mesh, "coordinates_bk"):
                coordinates_bk_f = fd.Function(fine_mesh.coordinates)
                fine_mesh.coordinates_bk = coordinates_bk_f

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
                coordinates_bk_c = fd.Function(coarse_mesh.coordinates)
                coarse_mesh.coordinates_bk = coordinates_bk_c
            if not hasattr(fine_mesh, "coordinates_bk"):
                coordinates_bk_f = fd.Function(fine_mesh.coordinates)
                fine_mesh.coordinates_bk = coordinates_bk_f

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


# create a transfer manager using the ManifoldTransfer operators
# for the MixedFunctionSpace W
def manifold_transfer_manager(W):
    Vs = W.split()
    vtransfer = ManifoldTransfer()
    transfers = {}
    for V in Vs:
        transfers[V.ufl_element()] = (vtransfer.prolong, vtransfer.restrict,
                                      vtransfer.inject)
    return fd.TransferManager(native_transfers=transfers)


# set up mesh levels for multigrid scheme
def high_order_icosahedral_mesh_hierarchy(mh, degree, R0):
    meshes = []
    for m in mh:
        X = fd.VectorFunctionSpace(m, "Lagrange", degree)
        new_coords = fd.interpolate(m.coordinates, X)
        x, y, z = new_coords
        r = (x**2 + y**2 + z**2)**0.5
        new_coords.assign(R0*new_coords/r)
        new_mesh = fd.Mesh(new_coords)
        meshes.append(new_mesh)

    return fd.HierarchyBase(meshes, mh.coarse_to_fine_cells,
                            mh.fine_to_coarse_cells,
                            mh.refinements_per_level, mh.nested)


# multigrid mesh for an icosahedral sphere
def icosahedral_mesh(R0,
                     base_level,
                     degree,
                     distribution_parameters,
                     nrefs,
                     comm=fd.COMM_WORLD):
    basemesh = fd.IcosahedralSphereMesh(
                    radius=R0,
                    refinement_level=base_level,
                    degree=1,
                    distribution_parameters=distribution_parameters,
                    comm=comm)
    del basemesh._radius
    mh = fd.MeshHierarchy(basemesh, nrefs)
    mh = high_order_icosahedral_mesh_hierarchy(mh, degree, R0)
    for mesh in mh:
        xf = mesh.coordinates
        mesh.transfer_coordinates = fd.Function(xf)
        x = fd.SpatialCoordinate(mesh)
        r = (x[0]**2 + x[1]**2 + x[2]**2)**0.5
        xf.interpolate(R0*xf/r)
        mesh.init_cell_orientations(x)
    mesh = mh[-1]
    return mesh
