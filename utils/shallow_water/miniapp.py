import firedrake as fd
import asQ
import utils.shallow_water as swe
from utils import mg


class ShallowWaterMiniApp(object):
    def __init__(self,
                 create_mesh,
                 gravity,
                 topography_expression,
                 coriolis_expression,
                 velocity_expression,
                 depth_expression,
                 dt, theta, alpha,
                 slice_partition,
                 paradiag_sparameters,
                 block_ctx={},
                 velocity_function_space=swe.default_velocity_function_space,
                 depth_function_space=swe.default_depth_function_space):
        '''
        A miniapp to integrate the rotating shallow water equations on the sphere using the paradiag method.

        :arg create_mesh: function to generate the mesh, given an MPI communicator
        :arg gravity: the gravitational constant.
        :arg topography_expression: firedrake expression for the topography field.
        :arg coriolis_expression: firedrake expression for the coriolis parameter.
        :arg velocity_expression: firedrake expression for initialising the velocity field.
        :arg depth_expression: firedrake expression for initialising the depth field.
        :arg dt: timestep size.
        :arg theta: parameter for the implicit theta-method integrator
        :arg alpha: value used for the alpha-circulant approximation in the paradiag method.
        :arg slice_partition: a list with how many timesteps are on each of the ensemble time-ranks.
        :paradiag_sparameters: a dictionary of PETSc solver parameters for the solution of the all-at-once system
        :block_ctx: a dictionary of extra values required for the block system solvers.
        :velocity_function_space: function to return a firedrake FunctionSpace for the velocity field, given a mesh
        :depth_function_space: function to return a firedrake FunctionSpace for the depth field, given a mesh
        '''

        # calculate nspatial_domains and set up ensemble
        nslices = len(slice_partition)
        nranks = fd.COMM_WORLD.size
        if nranks % nslices != 0:
            raise ValueError("Number of time slices must be exact factor of number of MPI ranks")

        nspatial_domains = nranks/nslices

        self.ensemble = fd.Ensemble(fd.COMM_WORLD, nspatial_domains)

        self.mesh = create_mesh(self.ensemble.comm)
        x = fd.SpatialCoordinate(self.mesh)

        # Mixed function space for velocity and depth
        self.V1 = velocity_function_space(self.mesh)
        self.V2 = depth_function_space(self.mesh)
        self.W = fd.MixedFunctionSpace((self.V1, self.V2))

        # nonlinear swe forms

        self.gravity = gravity
        self.topography = topography_expression(*x)
        self.coriolis = coriolis_expression(*x)

        def form_function(u, h, v, q):
            g = self.gravity
            b = self.topography
            f = self.coriolis
            return swe.nonlinear.form_function(self.mesh, g, b, f, u, h, v, q)

        def form_mass(u, h, v, q):
            return swe.nonlinear.form_mass(self.mesh, u, h, v, q)

        self.form_function = form_function
        self.form_mass = form_mass

        # initial conditions

        w0 = fd.Function(self.W)
        u0, h0 = w0.split()

        u0.project(velocity_expression(*x))
        h0.project(depth_expression(*x))

        # non-petsc information for block solve

        # mesh transfer operators
        # probably shouldn't be here, but has to be at the moment because we can't get at the mesh to make W before initialising.
        # should look at removing this once the manifold transfer manager has found a proper home
        transfer_managers = []
        for _ in range(sum(slice_partition)):
            tm = mg.manifold_transfer_manager(self.W)
            transfer_managers.append(tm)

        block_ctx['diag_transfer_managers'] = transfer_managers

        self.paradiag = asQ.paradiag(
            ensemble=self.ensemble,
            form_function=form_function,
            form_mass=form_mass,
            W=self.W, w0=w0,
            dt=dt, theta=theta,
            alpha=alpha, M=slice_partition,
            solver_parameters=paradiag_sparameters,
            circ=None, ctx={}, block_ctx=block_ctx)
