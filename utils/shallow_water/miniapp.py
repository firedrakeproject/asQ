import firedrake as fd
import asQ
import utils.shallow_water as swe
from utils import mg


class ShallowWaterMiniApp(object):
    def __init__(self,
                 ensemble,
                 form_function,
                 form_mass,
                 W, w0,
                 dt, theta, alpha, M,
                 solver_parameters,
                 block_ctx):

        self.paradiag = asQ.paradiag(
            ensemble=ensemble,
            form_function=form_function,
            form_mass=form_mass, W=W, w0=w0,
            dt=dt, theta=0.5,
            alpha=alpha,
            M=M, solver_parameters=solver_parameters,
            circ=None, tol=1.0e-6, maxits=None,
            ctx={}, block_ctx=block_ctx, block_mat_type="aij")


class ShallowWaterMiniApp_donotuse(object):
    def __init__(self,
                 create_mesh,
                 gravity,
                 topography_expression,
                 coriolis_expression,
                 velocity_expression,
                 depth_expression,
                 dt,
                 alpha,
                 slice_partition,
                 paradiag_sparameters,
                 block_sparameters,
                 block_ctx={}):
        '''
        A miniapp to integrate the rotating shallow water equations on the sphere using the paradiag method.

        :arg create_mesh: function to generate the mesh given an MPI communicator
        :arg gravity: the gravitational constant.
        :arg topography_expression: firedrake expression for the topography field.
        :arg coriolis_expression: firedrake expression for the coriolis parameter.
        :arg velocity_expression: firedrake expression for initialising the velocity field.
        :arg depth_expression: firedrake expression for initialising the depth field.
        :arg dt: timestep size.
        :arg alpha: value used for the alpha-circulant approximation in the paradiag method.
        :arg slice_partition: a list with how many timesteps are on each of the ensemble time-ranks.
        :paradiag_sparameters: a dictionary of PETSc solver parameters for the solution of the all-at-once system
        :block_sparamaters: a dictionary of PETSc solver parameters for the solution of the block systems in the paradiag matrix
        :block_ctx: a dictionary of extra values required for the block system solvers.
        '''

        # calculate nspatial_domains and set up ensemble
        nslices = len(slice_partition)
        nranks = fd.COMM_WORLD.size
        if nranks % nslices != 0:
            raise ValueError("Number of time slices must be exact factor of number of MPI ranks")

        nspatial_domains = nranks/nslices

        self.ensemble = fd.Ensemble(fd.COMM_WORLD, nspatial_domains)

        # set up mesh
        self.mesh = create_mesh(self.ensemble.comm)
        x = fd.SpatialCoordinate(self.mesh)

        # function spaces
        self.V1 = swe.default_velocity_function_space(self.mesh)
        self.V2 = swe.default_depth_function_space(self.mesh)
        self.W = fd.MixedFunctionSpace((self.V1, self.V2))

        # mesh transfer operators
        # probably shouldn't be here, but has to be at the moment because we can't get at the mesh to make W before initialising.
        # should look at removing this once the manifold transfer manager has found a proper home
        transfer_managers = []
        for _ in range(sum(slice_partition)):
            tm = mg.manifold_transfer_manager(self.W)
            transfer_managers.append(tm)
        block_ctx['diag_transfer_managers'] = transfer_managers

        # initial conditions
        self.gravity = gravity
        self.b = fd.Function(self.V2, name='Elevation')
        self.b.interpolate(topography_expression(*x))

        self.f = coriolis_expression(*x)

        w0 = fd.Function(self.W)
        u0, h0 = w0.split()

        u0.project(velocity_expression(*x))
        h0.project(depth_expression(*x))

        # nonlinear swe forms

        def form_function(u, h, v, q):
            return swe.nonlinear.form_function(self.mesh, self.gravity, self.b, self.f, u, h, v, q)

        def form_mass(u, h, v, q):
            return swe.nonlinear.form_mass(self.mesh, u, h, v, q)

        self.form_function = form_function
        self.form_mass = form_mass

        # set up paradiag
        self.paradiag = asQ.paradiag(
            ensemble=self.ensemble,
            form_function=form_function,
            form_mass=form_mass, W=self.W, w0=w0,
            dt=dt, theta=0.5,
            alpha=alpha,
            M=slice_partition,
            solver_parameters=paradiag_sparameters,
            circ=None,
            block_ctx=block_ctx)
