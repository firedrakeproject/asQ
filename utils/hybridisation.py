import firedrake as fd
from firedrake.parloops import par_loop, READ, INC
from math import prod

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
    _prefix = "hybridscpc"

    def initialize(self, pc):
        if pc.getType() != "python":
            raise ValueError("Expecting PC type python")

        from ufl import replace

        appctx = self.get_appctx(pc)
        _, P = pc.getOperators()
        ctx = P.getPythonContext()
        test, trial = ctx.a.arguments()
        tests, trials = fd.split(test), fd.split(trial)

        V = test.function_space()
        print(f"V = {V}")
        mesh = V.mesh()
        ncpts = len(V)

        # break the HDiv component of the function space,
        # leaving the rest untouched
        Vtr, iu = _break_function_space(V, appctx)
        self.iu = iu
        print(f"Vtr = {Vtr}")
        print(f"iu = {iu}")

        # breaks/mends the velocity residual
        self.projector = BrokenHDivProjector(V[iu], Vtr[iu])

        # build working buffers
        self.x = fd.Cofunction(V.dual())
        self.y = fd.Function(V)

        self.xtr = fd.Cofunction(Vtr.dual())
        self.ytr = fd.Function(Vtr)

        # build the hybridised system
        utrs = fd.TrialFunctions(Vtr)
        vtrs = fd.TestFunctions(Vtr)

        print("utrs =")
        print()
        for u in utrs:
            print(u)
        print()

        print("vtrs =")
        print()
        for v in vtrs:
            print(v)
        print()

        # break the original form
        olds = (*tests, *trials)
        news = (*vtrs[:-1], *utrs[:-1])
        arg_map = {old: new for old, new in zip(olds, news)}
        A = replace(ctx.a, arg_map)

        # add the trace bit
        n = fd.FacetNormal(mesh)

        def form_trace(*args):
            trls = args[:ncpts+1]
            tsts = args[ncpts+1:]
            return (
                fd.jump(tsts[iu], n)*trls[-1]('+')
                + fd.jump(trls[iu], n)*tsts[-1]('+')
            )*fd.dS

        # are we using complex-proxy?
        if 'cpx' in appctx:
            cpx = appctx['cpx']
            A += cpx.BilinearForm(Vtr, 1, form_trace)
        else:
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

        problem = fd.LinearVariationalProblem(A, L, self.ytr)
        self.solver = fd.LinearVariationalSolver(
            problem, appctx=appctx,
            solver_parameters=scpc_params,
            options_prefix=pc.getOptionsPrefix()+self._prefix)

    def apply(self, pc, x, y):
        # copy into unbroken vector
        with self.x.dat.vec_wo as v:
            x.copy(v)

        # break velocity, other spaces already broken
        iu = self.iu
        xs = zip(self.x.subfunctions, self.xtr.subfunctions[:-1])
        for i, (vi, vbi) in enumerate(xs):
            if i == iu:
                self.projector.project(vi, vbi)
            else:
                vbi.assign(vi)

        self.ytr.assign(0)
        self.solver.solve()

        # mend velocity, other spaces already mended
        ys = zip(self.y.subfunctions, self.ytr.subfunctions[:-1])
        for i, (vi, vbi) in enumerate(ys):
            if i == iu:
                self.projector.project(vbi, vi)
            else:
                vi.assign(vbi)

        # copy out to petsc
        with self.y.dat.vec_ro as v:
            v.copy(y)

    def update(self, pc):
        pass

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError
