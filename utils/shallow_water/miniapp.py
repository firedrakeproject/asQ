import firedrake as fd
import asQ
from asQ.allatonce.mixin import TimePartitionMixin

from utils import units
from utils import diagnostics
import utils.shallow_water as swe
from utils import mg


class ShallowWaterMiniApp(TimePartitionMixin):
    def __init__(self,
                 gravity,
                 topography_expression,
                 velocity_expression,
                 depth_expression,
                 dt, theta,
                 time_partition,
                 paradiag_sparameters,
                 create_mesh=swe.create_mg_globe_mesh,
                 coriolis_expression=swe.earth_coriolis_expression,
                 appctx={},
                 reference_depth=0,
                 reference_state=False,
                 linear=False,
                 velocity_function_space=swe.default_velocity_function_space,
                 depth_function_space=swe.default_depth_function_space,
                 record_diagnostics={'cfl': True, 'file': True},
                 save_step=-1, file_name='swe',
                 pre_function_callback=None, post_function_callback=None,
                 pre_jacobian_callback=None, post_jacobian_callback=None):
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
        :arg time_partition: a list with how many timesteps are on each of the ensemble time-ranks.
            arg :paradiag_sparameters: a dictionary of PETSc solver parameters for the solution of the all-at-once system
        :arg appctx: a dictionary of extra values required for the preconditioner.
        :arg reference_depth: constant used to calculate elevation
        :arg reference_state: Whether to create a reference state for the AllAtOnceSystem
        :arg linear: if False, solve nonlinear shallow water equations, if True solve linear equations
        :arg velocity_function_space: function to return a firedrake FunctionSpace for the velocity field, given a mesh
        :arg depth_function_space: function to return a firedrake FunctionSpace for the depth field, given a mesh
        :arg record_diagnostics: List of bools whether to: record CFL at each timestep; save timesteps to file
        :arg save_step: if record_diagnostics['file'] is True, save timestep with this window index to file
        :arg file_name: if record_diagnostics['file'] is True, save timesteps to file with this name
        '''
        self.ensemble = asQ.create_ensemble(time_partition)
        self.time_partition_setup(self.ensemble, time_partition)

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
        if linear:
            def form_function(u, h, v, q, t):
                g = self.gravity
                H = self.reference_depth
                f = self.coriolis
                return swe.linear.form_function(self.mesh, g, H, f, u, h, v, q, t)

            def form_mass(u, h, v, q):
                return swe.linear.form_mass(self.mesh, u, h, v, q)

        else:
            def form_function(u, h, v, q, t):
                g = self.gravity
                b = self.topography
                f = self.coriolis
                return swe.nonlinear.form_function(self.mesh, g, b, f, u, h, v, q, t)

            def form_mass(u, h, v, q):
                return swe.nonlinear.form_mass(self.mesh, u, h, v, q)

        self.form_function = form_function
        self.form_mass = form_mass

        # initial conditions

        w0 = fd.Function(self.function_space())
        u0 = w0.subfunctions[self.velocity_index]
        h0 = w0.subfunctions[self.depth_index]

        u0.project(velocity_expression(*x))
        h0.project(depth_expression(*x))

        if reference_state:
            self.reference_state = fd.Function(self.function_space())
            reference_state = self.reference_state
        else:
            reference_state = None

        # non-petsc information for block solve

        # mesh transfer operators
        # probably shouldn't be here, but has to be at the moment because we can't get at the mesh to make W before initialising.
        # should look at removing this once the manifold transfer manager has found a proper home
        transfer_managers = []
        for _ in range(self.nlocal_timesteps):
            tm = mg.manifold_transfer_manager(self.function_space())
            transfer_managers.append(tm)

        appctx['diagfft_transfer_managers'] = transfer_managers

        self.paradiag = asQ.Paradiag(
            ensemble=self.ensemble,
            form_function=form_function,
            form_mass=form_mass,
            ics=w0, dt=dt, theta=theta,
            time_partition=time_partition,
            appctx=appctx,
            reference_state=reference_state,
            solver_parameters=paradiag_sparameters,
            pre_function_callback=pre_function_callback,
            post_function_callback=post_function_callback,
            pre_jacobian_callback=pre_jacobian_callback,
            post_jacobian_callback=post_jacobian_callback)

        self.aaofunc = self.paradiag.aaofunc
        self.aaoform = self.paradiag.aaofunc
        self.solver = self.paradiag.solver

        # set up swe diagnostics
        self.record_diagnostics = record_diagnostics
        self.save_step = self.aaofunc.transform_index(save_step, from_range='window', to_range='window')

        # cfl
        self.cfl = diagnostics.convective_cfl_calculator(self.mesh)

        # potential vorticity
        self.potential_vorticity = diagnostics.potential_vorticity_calculator(
            self.velocity_function_space(), name='vorticity')

        if record_diagnostics['file']:
            if self.layout.is_local(self.save_step):
                self.uout = fd.Function(self.velocity_function_space(), name='velocity')
                self.hout = fd.Function(self.depth_function_space(), name='elevation')
                self.ofile = fd.File(file_name+'.pvd',
                                     comm=self.ensemble.comm)
                # save initial conditions
                self.uout.assign(u0)
                self.hout.assign(h0 + self.topography_function - self.reference_depth)
                self.ofile.write(self.uout, self.hout,
                                 self.potential_vorticity(self.uout),
                                 time=0)

        if record_diagnostics['cfl']:
            # which rank is the owner?
            owner = self.layout.rank_of(self.save_step)
            self.cfl_series = asQ.OwnedArray(size=1,
                                             owner=owner,
                                             comm=self.ensemble.ensemble_comm)

    def function_space(self):
        return self.W

    def velocity_function_space(self):
        return self.W.sub(self.velocity_index)

    def depth_function_space(self):
        return self.W.sub(self.depth_index)

    def max_cfl(self, step=None, dt=None, index_range='slice', v=None):
        '''
        Return the maximum convective CFL number for the field u with timestep dt

        :arg step: timestep to calculate CFL number for. If None, cfl calculated for function v.
        :arg dt: the timestep. If None, the timestep of the all-at-once system is used
        :arg index_range: type of index of step: slice or window
        :arg v: velocity Function from FunctionSpace V1 or a full MixedFunction from W if None, calculate cfl of timestep step. Ignored if step is not None.
        '''
        if step is not None:
            u = self.get_velocity(step, index_range=index_range)
        elif v is not None:
            if v.function_space() == self.velocity_function_space():
                u = v
            elif v.function_space() == self.function_space():
                u = v.subfunctions[self.velocity_index]
            else:
                raise ValueError("function v must be in FunctionSpace V1 or MixedFunctionSpace W")
        else:
            raise ValueError("v or step must be not None")

        if dt is None:
            dt = self.paradiag.aaoform.dt

        with self.cfl(u, dt).dat.vec_ro as cfl_vec:
            return cfl_vec.max()[1]

    def _record_diagnostics(self):
        '''
        Update diagnostic information after each solve

        :arg w: index of window in current solve loop
        '''
        if self.layout.is_local(self.save_step):

            window = self.paradiag.total_windows

            if self.record_diagnostics['file']:
                self.get_velocity(self.save_step, uout=self.uout, index_range='window')
                self.get_elevation(self.save_step, hout=self.hout, index_range='window')

                # global timestep over all windows
                window_length = self.paradiag.ntimesteps

                nt = (window - 1)*window_length + (self.save_step + 1)
                dt = self.paradiag.aaoform.dt
                t = nt*dt

                # save to file
                self.ofile.write(self.uout, self.hout,
                                 self.potential_vorticity(self.uout),
                                 time=t/units.hour)

            if self.record_diagnostics['cfl']:
                self.cfl_series[window-1] = self.max_cfl(self.save_step, index_range='window')

        self.ensemble.global_comm.Barrier()

    def sync_diagnostics(self):
        """
        Synchronise diagnostic information over all time-ranks.

        Until this method is called, diagnostic information is not guaranteed to be valid.
        """
        self.cfl_series.synchronise()

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

        def preprocess(pdg, w, rhs=None):
            preproc(self, pdg, w)

        def postprocess(pdg, w, rhs=None):
            self._record_diagnostics()
            postproc(self, pdg, w)

        # extend cfl array
        self.cfl_series.resize(self.paradiag.total_windows + nwindows)

        self.paradiag.solve(nwindows,
                            preproc=preprocess,
                            postproc=postprocess,
                            verbose=verbose)

        self.sync_diagnostics()

    def get_velocity(self, step, index_range='slice', uout=None, name='velocity', deepcopy=False):
        '''
        Return the velocity function at index step

        :arg step: the index, either in window or slice
        :arg index_range: whether step is a slice or window index
        :arg uout: if None, velocity function is returned, else the velocity function is assigned to uout
        :arg name: if uout is None, the returned velocity function will have this name
        :arg deepcopy: if True, new function is returned. If false, handle to the velocity component of the all-at-once function is returned. Ignored if wout is not None
        '''
        return self.aaofunc.get_component(step, self.velocity_index, index_range=index_range,
                                          uout=uout, name=name, deepcopy=deepcopy)

    def get_depth(self, step, index_range='slice', hout=None, name='depth', deepcopy=False):
        '''
        Return the depth function at index step

        :arg step: the index, either in window or slice
        :arg index_range: whether step is a slice or window index
        :arg hout: if None, depth function is returned, else the depth function is assigned to hout
        :arg name: if hout is None, the returned depth function will have this name
        :arg deepcopy: if True, new function is returned. If false, handle to the depth component of the all-at-once function is returned. Ignored if wout is not None
        '''
        return self.aaofunc.get_component(step, self.depth_index, index_range=index_range,
                                          uout=hout, name=name, deepcopy=deepcopy)

    def get_elevation(self, step, index_range='slice', hout=None, name='depth'):
        '''
        Return the elevation around the reference depth at index step

        :arg step: the index, either in window or slice
        :arg index_range: whether step is a slice or window index
        :arg hout: if None, depth function is returned, else the depth function is assigned to hout
        :arg name: if hout is None, the returned depth function will have this name
        '''
        if hout is None:
            h = self.get_depth(step, index_range=index_range, name=name, deepcopy=True)
            h.assign(h + self.topography_function - self.reference_depth)
            return h
        elif hout.function_space() == self.depth_function_space():
            h = self.get_depth(step, index_range=index_range)
            hout.assign(h + self.topography_function - self.reference_depth)
            return
        else:
            raise ValueError("hout must be None or a Function from depth_function_space")
