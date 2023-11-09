
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
        self.registered_meshes = set()
        self.Ftransfer = fd.TransferManager()  # is this the firedrake warning?

    def prolong(self, coarse, fine):
        self._transfer(coarse, fine, self.Ftransfer.prolong)

    def restrict(self, fine, coarse):
        self._transfer(fine, coarse, self.Ftransfer.restrict)

    def inject(self, fine, coarse):
        self._transfer(fine, coarse, self.Ftransfer.inject)

    def _transfer(self, f0, f1, transfer_op):
        mesh0 = f0.function_space().mesh()
        mesh1 = f1.function_space().mesh()

        # backup the original coordinates the first time we see a mesh
        self._register_mesh(mesh0)
        self._register_mesh(mesh1)

        # change to the transfer coordinates for transfer operations
        mesh0.coordinates.assign(mesh0.transfer_coordinates)
        mesh1.coordinates.assign(mesh1.transfer_coordinates)

        # standard transfer preserves divergence-free subspaces
        transfer_op(f0, f1)

        # change back to original mesh
        mesh0.coordinates.assign(mesh0.original_coordinates)
        mesh1.coordinates.assign(mesh1.original_coordinates)

    def _register_mesh(self, mesh):
        mesh_key = mesh.coordinates.function_space().dim()
        if mesh_key not in self.registered_meshes:
            mesh.original_coordinates = fd.Function(mesh.coordinates)
            self.registered_meshes.add(mesh_key)


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
