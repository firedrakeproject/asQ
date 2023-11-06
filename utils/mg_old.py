
import firedrake as fd
from ufl.domain import extract_unique_domain as ufl_domain


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
        Vfine = fd.FunctionSpace(ufl_domain(fine),
                                 fine.function_space().ufl_element())
        key = Vfine.dim()

        firsttime = self.ready.get(key, None) is None
        if firsttime:
            self.ready[key] = True

            coarse_mesh = ufl_domain(coarse)
            fine_mesh = ufl_domain(fine)
            if not hasattr(coarse_mesh, "coordinates_bk"):
                coordinates_bk_c = fd.Function(coarse_mesh.coordinates)
                coarse_mesh.coordinates_bk = coordinates_bk_c
            if not hasattr(fine_mesh, "coordinates_bk"):
                coordinates_bk_f = fd.Function(fine_mesh.coordinates)
                fine_mesh.coordinates_bk = coordinates_bk_f

        # change to the transfer coordinates for prolongation
        ufl_domain(coarse).coordinates.assign(
            ufl_domain(coarse).transfer_coordinates)
        ufl_domain(fine).coordinates.assign(
            ufl_domain(fine).transfer_coordinates)

        # standard transfer preserves divergence-free subspaces
        self.Ftransfer.prolong(coarse, fine)

        # change back to deformed mesh
        ufl_domain(coarse).coordinates.assign(
            ufl_domain(coarse).coordinates_bk)
        ufl_domain(fine).coordinates.assign(
            ufl_domain(fine).coordinates_bk)

    def restrict(self, fine, coarse):
        Vfine = fd.FunctionSpace(ufl_domain(fine),
                                 fine.function_space().ufl_element())
        key = Vfine.dim()

        firsttime = self.ready.get(key, None) is None
        if firsttime:
            self.ready[key] = True

            coarse_mesh = ufl_domain(coarse)
            fine_mesh = ufl_domain(fine)
            if not hasattr(coarse_mesh, "coordinates_bk"):
                coordinates_bk_c = fd.Function(coarse_mesh.coordinates)
                coarse_mesh.coordinates_bk = coordinates_bk_c
            if not hasattr(fine_mesh, "coordinates_bk"):
                coordinates_bk_f = fd.Function(fine_mesh.coordinates)
                fine_mesh.coordinates_bk = coordinates_bk_f

        # change to the transfer coordinates for prolongation
        ufl_domain(coarse).coordinates.assign(
            ufl_domain(coarse).transfer_coordinates)
        ufl_domain(fine).coordinates.assign(
            ufl_domain(fine).transfer_coordinates)

        # standard transfer preserves divergence-free subspaces
        self.Ftransfer.restrict(fine, coarse)

        # change back to deformed mesh
        ufl_domain(coarse).coordinates.assign(
            ufl_domain(coarse).coordinates_bk)
        ufl_domain(fine).coordinates.assign(
            ufl_domain(fine).coordinates_bk)

    def inject(self, fine, coarse):
        Vfine = fd.FunctionSpace(ufl_domain(fine),
                                 fine.function_space().ufl_element())
        key = Vfine.dim()

        firsttime = self.ready.get(key, None) is None
        if firsttime:
            self.ready[key] = True

            coarse_mesh = ufl_domain(coarse)
            fine_mesh = ufl_domain(fine)
            if not hasattr(coarse_mesh, "coordinates_bk"):
                coordinates_bk_c = fd.Function(coarse_mesh.coordinates)
                coarse_mesh.coordinates_bk = coordinates_bk_c
            if not hasattr(fine_mesh, "coordinates_bk"):
                coordinates_bk_f = fd.Function(fine_mesh.coordinates)
                fine_mesh.coordinates_bk = coordinates_bk_f

        # change to the transfer coordinates for prolongation
        ufl_domain(coarse).coordinates.assign(
            ufl_domain(coarse).transfer_coordinates)
        ufl_domain(fine).coordinates.assign(
            ufl_domain(fine).transfer_coordinates)

        # standard transfer preserves divergence-free subspaces
        self.Ftransfer.inject(fine, coarse)

        # change back to deformed mesh
        ufl_domain(coarse).coordinates.assign(
            ufl_domain(coarse).coordinates_bk)
        ufl_domain(fine).coordinates.assign(
            ufl_domain(fine).coordinates_bk)


def manifold_transfer_manager(W):
    '''
    Return a multigrid transfer manager for manifold meshes

    Uses the ManifoldTransfer operations to manage the
    prolongation, restriction and injection
    arg: W: a MixedFunctionSpace over the manifold mesh
    '''

    Vs = W.subfunctions
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
        new_coords.interpolate(R0*new_coords/r)
        new_mesh = fd.Mesh(new_coords)
        meshes.append(new_mesh)

    return fd.HierarchyBase(meshes, mh.coarse_to_fine_cells,
                            mh.fine_to_coarse_cells,
                            mh.refinements_per_level, mh.nested)


def icosahedral_mesh(R0,
                     base_level,
                     degree,
                     distribution_parameters,
                     nrefs,
                     comm=fd.COMM_WORLD):
    '''
    Return an icosahedral sphere mesh with a multigrid mesh hierarchy

    The coordinates of each mesh in the hierarchy are pushed
    onto the sphere surface. This means that the meshes are not
    nested and a manifold transfer manager is necessary.
    '''

    basemesh = \
        fd.IcosahedralSphereMesh(radius=R0,
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
