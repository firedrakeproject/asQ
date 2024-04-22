import firedrake as fd
from firedrake.parloops import par_loop, READ, INC
from math import prod
from functools import partial

__all__ = ['BrokenHDivProjector', 'HybridisedSCPC']


class BrokenHDivProjector:
    def __init__(self, V, Vb=None):
        """
        Projects between an HDiv space and it's broken space
        using equal splitting at the facets.

        :arg V: The HDiv space.
        :arg Vb: optional, the broken space.
                 If not provided then BrokenElement(V.ufl_element()) is used.
        """

        # check V is actually HDiv
        sobolev_space = V.ufl_element().sobolev_space.name
        if sobolev_space != "HDiv":
            raise TypeError(f"V must be in HDiv not {sobolev_space}")

        # create the broken function space
        self.mesh = V.mesh()
        self.V = V
        if Vb is None:
            Vb = fd.FunctionSpace(self.mesh, fd.BrokenElement(V.ufl_element()))
        self.Vb = Vb

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

        self.weights = fd.Function(V).assign(0)
        par_loop(weight_kernel, fd.dx, {"weights": (self.weights, INC)})

        # create projection kernel
        projection_instructions = """
        for i, j
            dst[i,j] = dst[i,j] + src[i,j]/weights[i,j]
        end
        """
        self.projection_kernel = (domain, projection_instructions)

    def _check_functions(self, x, y):
        """
        Ensure the Functions/Cofunctions x and y are from the correct HDiv and broken spaces.
        """
        ex, ey = x.ufl_element(), y.ufl_element()
        ev, eb = self.V.ufl_element(), self.Vb.ufl_element()
        echeck = (ev, eb)
        mx, my = x.function_space().mesh(), y.function_space().mesh()
        mcheck = self.V.mesh()

        valid_mesh = (mx == mcheck) and (my == mcheck)
        valid_elements = (ex in echeck) and (ey in echeck) and (ex != ey)

        if not (valid_mesh and valid_elements):
            msg = f"Can only project between the HDiv space {ev} and the broken space {eb} on the mesh {mcheck}, not between the spaces {ex} on mesh {mx} and {ey} on mesh {my}."
            raise TypeError(msg)

    def project(self, src, dst):
        """
        Project from src to dst using equal weighting at the facets.

        :arg src: the Function/Cofunction to be projected from.
        :arg dst: the Function/Cofunction to be projected to.
        """
        self._check_functions(src, dst)
        dst.assign(0)
        par_loop(self.projection_kernel, fd.dx,
                 {"weights": (self.weights, READ),
                  "src": (src, READ),
                  "dst": (dst, INC)})


def _break_function_space(V, appctx):
    cpx = appctx.get('cpx', None)
    mesh = V.mesh()

    # find HDiv space
    iu = None
    for i, Vi in enumerate(V):
        if Vi.ufl_element().sobolev_space.name == "HDiv":
            iu = i
            break
    if iu is None:
        msg = "Hybridised space must have one HDiv component"
        raise ValueError(msg)

    # hybridisable space - broken HDiv and Trace
    Vs = V.subfunctions

    Vu = Vs[iu]

    # broken HDiv space - either we are given one or we build one.
    # If we are using complex-proxy then we need to build the real
    # broken space first and then convert to complex-proxy.
    if 'broken_space' in appctx:
        Vub = appctx['broken_space']
    else:
        if cpx is None:
            broken_element = fd.BrokenElement(Vu.ufl_element())
            Vub = fd.FunctionSpace(mesh, broken_element)
        else:
            broken_element = fd.BrokenElement(Vu.sub(0).ufl_element())
            broken_element = cpx.FiniteElement(broken_element)
            Vub = fd.FunctionSpace(mesh, broken_element)

    # trace space - possibly complex-valued
    Tr = fd.FunctionSpace(mesh, "HDivT", Vu.ufl_element().degree())
    if cpx is not None:
        Tr = cpx.FunctionSpace(Tr)

    # trace space always last component
    trsubs = [Vs[i] if (i != iu) else Vub
              for i in range(len(Vs))] + [Tr]
    Vtr = fd.MixedFunctionSpace(trsubs)

    return Vtr, iu


class HybridisedSCPC(fd.PCBase):
    """
    A preconditioner to solve a mixed Hdiv problem using hybridisation.

    Hybridises the HDiv component of the mixed space, and reduces the
    system down to the trace variable. The petsc options for the trace
    problem have the prefix:
    "-hybridscpc_condensed_field"

    This preconditioner can be applied to an Hdiv problem formed with
    the vector module of complex_proxy.

    Required appctx entries (usually filled by the all-at-once preconditioner):

    'uref': Firedrake Function around which to linearise the form_function.
    'tref': The time at which to linearise the form_function.
    'bcs': A list of the boundary conditions.
    'form_mass': The function to generate the mass matrix.
    'form_function': The function to generate the stiffness matrix.

    If solving a real-valued problem the following entries are also required:

    'dt': The timestep size.
    'theta': The implicit parameter.

    If solving a complex-valued problem the following entries are also required:

    'cpx': The complex_proxy submodule to generate the complex-valued forms.
    'd1': The complex coefficient on the mass matrix.
    'd2': The complex coefficient on the stiffness matrix.

    Optional appctx entries. Used instead of the required entries if present.

    'hybridscpc_form_mass': Alternative function used to generate the mass matrix.
    'hybridscpc_form_function': Alternative function used to generate the stiffness matrix.
    """

    _prefix = "hybridscpc"

    def initialize(self, pc):
        if pc.getType() != "python":
            raise ValueError("Expecting PC type python")

        self._process_context(pc)

        # break the HDiv component of the function space,
        # leaving the rest untouched
        V = self.V
        Vtr, iu = _break_function_space(V, self.appctx)
        self.Vtr = Vtr
        self.iu = iu

        # breaks/mends the velocity residual
        self.projector = BrokenHDivProjector(V[iu], Vtr[iu])

        # allocate working buffers
        self.x = fd.Cofunction(V.dual())
        self.y = fd.Function(V)

        self.xtr = fd.Cofunction(Vtr.dual())
        self.ytr = fd.Function(Vtr)

        # build the hybridised system
        self.utr_ref = fd.Function(Vtr)
        us = fd.split(self.utr_ref)
        utrs = fd.TrialFunctions(Vtr)
        vtrs = fd.TestFunctions(Vtr)

        # the trace bit
        n = fd.FacetNormal(self.mesh)
        ncpts = len(V.subfunctions)

        def form_trace(*args):
            trls = args[:ncpts+1]
            tsts = args[ncpts+1:]
            return (
                fd.jump(tsts[iu], n)*trls[-1]('+')
                + fd.jump(trls[iu], n)*tsts[-1]('+')
            )*fd.dS

        if self._complex_proxy:
            cpx = self.cpx

            # forms from only the broken components (not the trace component)
            def form_mass(*args):
                trls = args[:ncpts+1]
                tsts = args[ncpts+1:]
                return self.form_mass(*trls[:-1], *tsts[:-1])

            def form_function(*args):
                trls = args[:ncpts+1]
                tsts = args[ncpts+1:]
                return self.form_function(*trls[:-1], *tsts[:-1])

            M = cpx.BilinearForm(Vtr, self.d1, form_mass)
            K = cpx.derivative(self.d2, form_function, self.utr_ref)

            A = M + K

            # now add the trace bit
            A += cpx.BilinearForm(Vtr, 1, form_trace)

        else:
            M = self.form_mass(*utrs[:-1], *vtrs[:-1])
            F = self.form_function(*us[:-1], *vtrs[:-1])
            K = fd.derivative(F, self.utr_ref)

            dt1 = fd.Constant(1/self.dt)
            tht = fd.Constant(self.theta)

            A = dt1*M + tht*K
            A += form_trace(*utrs, *vtrs)

        L = self.xtr

        default_trace_params = {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps'
        }

        # eliminate everything except the trace variable
        eliminate_fields = ", ".join(map(str, range(ncpts)))

        scpc_params = {
            "mat_type": "matfree",
            "ksp_type": "preonly",
            "pc_type": "python",
            "pc_python_type": "firedrake.SCPC",
            "pc_sc_eliminate_fields": eliminate_fields,
            "condensed_field": default_trace_params
        }

        problem = fd.LinearVariationalProblem(A, L, self.ytr, constant_jacobian=True)
        self.solver = fd.LinearVariationalSolver(
            problem, appctx=self.appctx,
            solver_parameters=scpc_params,
            options_prefix=self.prefix)

    def apply(self, pc, x, y):
        # copy into unbroken vector
        with self.x.dat.vec_wo as v:
            x.copy(v)

        # break velocity, other spaces already broken
        xs = zip(self.x.subfunctions, self.xtr.subfunctions[:-1])
        self._mend_or_break(xs)

        self.ytr.assign(0)
        self.solver.solve()

        # mend velocity, other spaces already mended
        ys = zip(self.ytr.subfunctions[:-1], self.y.subfunctions)
        self._mend_or_break(ys)

        # copy out to petsc
        with self.y.dat.vec_ro as v:
            v.copy(y)

    def update(self, pc):
        usubs = zip(self.uref.subfunctions, self.utr_ref.subfunctions[:-1])
        self._mend_or_break(usubs)

        self.solver.invalidate_jacobian()

    def _process_context(self, pc):
        appctx = self.get_appctx(pc)
        self.appctx = appctx

        self.prefix = pc.getOptionsPrefix() + self._prefix

        self.uref = appctx.get('uref')
        self.V = self.uref.function_space()
        self.mesh = self.V.mesh()

        self.bcs = appctx['bcs']
        self.tref = appctx['tref']

        form_mass = appctx['form_mass']
        form_function = appctx['form_function']

        self.form_mass = appctx.get('hybridscpc_form_mass', form_mass)
        self.form_function = appctx.get('hybridscpc_form_function', form_function)

        self.form_function = partial(self.form_function, t=self.tref)

        if 'cpx' in self.appctx:
            self.cpx = self.appctx['cpx']
            self.d1 = self.appctx['d1']
            self.d2 = self.appctx['d2']
            self._complex_proxy = True
        else:
            self.dt = self.appctx['dt']
            self.theta = self.appctx['theta']
            self._complex_proxy = False

    def _mend_or_break(self, usubs):
        for i, (ux, uy) in enumerate(usubs):
            if i == self.iu:
                self.projector.project(ux, uy)
            else:
                uy.assign(ux)

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError
