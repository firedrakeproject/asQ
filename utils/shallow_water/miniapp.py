import firedrake as fd
import asQ
from utils import diagnostics
import utils.shallow_water as swe
from utils import mg


class ShallowWaterMiniApp(object):
    def __init__(self,
                 gravity,
                 topography_expression,
                 velocity_expression,
                 depth_expression,
                 dt, theta, alpha,
                 slice_partition,
                 paradiag_sparameters,
                 create_mesh=swe.create_mg_globe_mesh,
                 coriolis_expression=swe.earth_coriolis_expression,
                 block_ctx={},
                 reference_depth=0,
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

        self.ensemble = asQ.create_ensemble(slice_partition)

        self.mesh = create_mesh(comm=self.ensemble.comm)
        x = fd.SpatialCoordinate(self.mesh)

        # Mixed function space for velocity and depth
        V1 = velocity_function_space(self.mesh)
        V2 = depth_function_space(self.mesh)
        self.W = fd.MixedFunctionSpace((V1, V2))
        self.velocity_index = 0
        self.depth_index = 1

        # nonlinear swe forms

        self.gravity = gravity
        self.topography = topography_expression(*x)
        self.coriolis = coriolis_expression(*x)
        self.reference_depth = reference_depth

        self.topography_function = fd.Function(self.depth_function_space(),
                                               name='topography')
        self.topography_function.interpolate(self.topography)

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

        w0 = fd.Function(self.function_space())
        u0, h0 = w0.split()

        u0.project(velocity_expression(*x))
        h0.project(depth_expression(*x))

        # non-petsc information for block solve

        # mesh transfer operators
        # probably shouldn't be here, but has to be at the moment because we can't get at the mesh to make W before initialising.
        # should look at removing this once the manifold transfer manager has found a proper home
        transfer_managers = []
        for _ in range(slice_partition[self.ensemble.ensemble_comm.rank]):
            tm = mg.manifold_transfer_manager(self.function_space())
            transfer_managers.append(tm)

        block_ctx['diag_transfer_managers'] = transfer_managers

        self.paradiag = asQ.paradiag(
            ensemble=self.ensemble,
            form_function=form_function,
            form_mass=form_mass,
            W=self.function_space(), w0=w0,
            dt=dt, theta=theta,
            alpha=alpha, M=slice_partition,
            solver_parameters=paradiag_sparameters,
            circ=None, ctx={}, block_ctx=block_ctx)

        # set up swe diagnostics

        # cfl
        self.cfl = diagnostics.convective_cfl_calculator(self.mesh)

        # potential vorticity
        self.potential_vorticity = diagnostics.potential_vorticity_calculator(
            self.velocity_function_space(), name='vorticity')

    def function_space(self):
        return self.W

    def velocity_function_space(self):
        return self.W.sub(self.velocity_index)

    def depth_function_space(self):
        return self.W.sub(self.depth_index)

    def max_cfl(self, dt, step=None, index_range='slice', v=None):
        '''
        Return the maximum convective CFL number for the field u with timestep dt
        :arg dt: the timestep
        :arg step: timestep to calculate CFL number for. If None, cfl calculated for function v.
        :arg index_range: type of index of step: slice or window
        :arg v: velocity Function from FunctionSpace V1 or a full MixedFunction from W if None, calculate cfl of timestep step. Ignored if step is not None.
        '''
        if step is not None:
            u = self.get_velocity(step, index_range=index_range)
        elif v is not None:
            if v.function_space() == self.velocity_function_space():
                u = v
            elif v.function_space() == self.function_space():
                u = v.split()[self.velocity_index]
            else:
                raise ValueError("function v must be in FunctionSpace V1 or MixedFunctionSpace W")
        else:
            raise ValueError("v or step must be not None")

        with self.cfl(u, dt).dat.vec_ro as cfl_vec:
            return cfl_vec.max()[1]

    def solve(self,
              nwindows=1,
              preproc=lambda miniapp, pdg, w: None,
              postproc=lambda miniapp, pdg, w: None,
              verbose=False):
        """
        Solve the paradiag system

        preproc and postproc must have call signature (miniapp, paradiag, int)
        :arg nwindows: number of windows to solve for
        :arg preproc: callback called before each window solve
        :arg postproc: callback called after each window solve
        """

        # wrap the pre/post processing functions

        def preprocess(pdg, w):
            preproc(self, pdg, w)

        def postprocess(pdg, w):
            postproc(self, pdg, w)

        self.paradiag.solve(nwindows,
                            preproc=preprocess,
                            postproc=postprocess,
                            verbose=verbose)

    def get_velocity(self, step, index_range='slice', uout=None, name='velocity'):
        '''
        Return the velocity function at index step

        :arg step: the index, either in window or slice
        :arg index_range: whether step is a slice or window index
        :arg uout: if None, velocity function is returned, else the velocity function is assigned to uout
        :arg name: if uout is None, the returned velocity function will have this name
        '''
        w = self.paradiag.get_timestep(step,
                                       index_range=index_range)
        if uout is None:
            u = fd.Function(self.velocity_function_space(), name=name)
            u.assign(w.split()[self.velocity_index])
            return u
        elif uout.function_space() == self.velocity_function_space():
            uout.assign(w.split()[self.velocity_index])
            return
        else:
            raise ValueError("uout must be None or a Function from velocity_function_space")

    def get_depth(self, step, index_range='slice', hout=None, name='depth'):
        '''
        Return the depth function at index step

        :arg step: the index, either in window or slice
        :arg index_range: whether step is a slice or window index
        :arg hout: if None, depth function is returned, else the depth function is assigned to hout
        :arg name: if hout is None, the returned depth function will have this name
        '''
        w = self.paradiag.get_timestep(step,
                                       index_range=index_range)
        if hout is None:
            h = fd.Function(self.depth_function_space(), name=name)
            h.assign(w.split()[self.depth_index])
            return h
        elif hout.function_space() == self.depth_function_space():
            hout.assign(w.split()[self.depth_index])
            return
        else:
            raise ValueError("hout must be None or a Function from depth_function_space")

    def get_elevation(self, step, index_range='slice', hout=None, name='depth'):
        '''
        Return the elevation around the reference depth at index step

        :arg step: the index, either in window or slice
        :arg index_range: whether step is a slice or window index
        :arg hout: if None, depth function is returned, else the depth function is assigned to hout
        :arg name: if hout is None, the returned depth function will have this name
        '''
        if hout is None:
            h = self.get_depth(step, index_range=index_range, name=name)
            h.assign(h + self.topography_function - self.reference_depth)
            return h
        elif hout.function_space() == self.depth_function_space():
            self.get_depth(step, index_range=index_range, hout=hout)
            hout.assign(hout + self.topography_function - self.reference_depth)
            return
        else:
            raise ValueError("hout must be None or a Function from depth_function_space")
