import firedrake as fd
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
from utils.diagnostics import convective_cfl_calculator
from utils.serial import SerialMiniApp
from utils.vertical_slice import straka
from utils import compressible_flow as euler

import argparse
parser = argparse.ArgumentParser(description='Straka testcase.')
parser.add_argument('--nlayers', type=int, default=16, help='Number of layers, default 10.')
parser.add_argument('--ncolumns', type=int, default=128, help='Number of columns, default 10.')
parser.add_argument('--nt', type=int, default=1, help='Number of timesteps.')
parser.add_argument('--output_freq', type=int, default=1, help='Output frequency in timesteps.')
parser.add_argument('--dt', type=float, default=2, help='Timestep in seconds. Default 1.')
parser.add_argument('--filename', type=str, default='straka')
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space). Default 1.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

PETSc.Sys.Print("Setting up problem")

comm = fd.COMM_WORLD

mesh = straka.mesh(comm, ncolumns=args.ncolumns, nlayers=args.nlayers)
n = fd.FacetNormal(mesh)

dt = args.dt
gas = euler.StandardAtmosphere(N=0.01)

W, Vv = euler.function_space(mesh, horizontal_degree=args.degree,
                             vertical_degree=args.degree,
                             vertical_velocity_space=True)
V1, V2, Vt = W.subfunctions  # velocity, density, temperature

PETSc.Sys.Print(f"DoFs: {W.dim()}")
PETSc.Sys.Print(f"DoFs/core: {W.dim()/mesh.comm.size}")

Un, rho_back, theta_back = straka.initial_conditions(mesh, W, Vv, gas)

# The timestepping forms

viscosity = fd.Constant(75.)

form_mass = euler.get_form_mass()

up = fd.as_vector([fd.Constant(0.0), fd.Constant(1.0)])  # up direction
form_function = euler.get_form_function(
    n=n, Up=up, c_pen=fd.Constant(2.0**(-7./2)),
    gas=gas, mu=None, viscosity=viscosity, diffusivity=viscosity)

bcs = straka.boundary_conditions(W)
for bc in bcs:
    bc.apply(Un)

lu_params = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    # "pc_factor_mat_solver_type": "mumps",
}

jacobi_params = {
    "ksp_type": "gmres",
    "pc_type": "jacobi",
}

patch_params = {
    "mat_type": "matfree",
    "pc_type": "python",
    "pc_python_type": "firedrake.AssembledPC",
    "assembled": {
        "pc_type": "python",
        "pc_python_type": "firedrake.ASMVankaPC",
        "pc_vanka": {
            "construct_dim": 0,
            "sub_sub_pc_type": "lu",
            "sub_sub_pc_factor_mat_ordering_type": "rcm",
            "sub_sub_pc_factor_reuse_ordering": None,
            "sub_sub_pc_factor_reuse_fill": None,
        },
    },
}

iwave = 0
itemp = int(not iwave)
ntemp = "Temperature"

hybridization_sparams = {
    # "mat_type": "matfree",
    "ksp_type": "fgmres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",
    f"pc_fieldsplit_{iwave}_fields": "0,1",
    f"pc_fieldsplit_{itemp}_fields": "2",
    f"fieldsplit_{iwave}": lu_params,
    f"fieldsplit_{ntemp}": lu_params,
    # f"fieldsplit_{ntemp}": {
    #     "ksp_type": "gmres",
    #     "pc_type": "none",
    # },
    # f"fieldsplit_{iwave}": {
    #     # "mat_type": "matfree",
    #     "pc_type": "python",
    #     "pc_python_type": "firedrake.HybridizationPC",
    #     # "hybridization": lu_params
    #     "hybridization": {
    #         "ksp_rtol": 1e-10,
    #         "ksp_type": "cg",
    #         # "ksp_converged_rate": None,
    #         'pc_type': 'gamg',
    #         'pc_gamg_sym_graph': None,
    #         'pc_mg_type': 'multiplicative',
    #         'mg': {
    #             'levels': {
    #                 'ksp_type': 'richardson',
    #                 'ksp_max_it': 3,
    #                 'pc_type': 'bjacobi',
    #                 'sub_pc_type': 'ilu',
    #             },
    #         },
    #     },
    # },
    "ksp_view": None,
}

atol = 1e0
stol = 1e-100
sparameters = {
    "snes": {
        "monitor": None,
        "converged_reason": None,
        "stol": stol,
        "atol": atol,
        "rtol": 1e-8,
        "ksp_ew": None,
        "ksp_ew_version": 1,
        "ksp_ew_threshold": 1e-10,
        "ksp_ew_rtol0": 1e-2,
    },
    "ksp_type": "fgmres",
    "ksp": {
        "monitor": None,
        "converged_reason": None,
        "stol": stol,
        "atol": atol,
    },
}

# sparameters.update(patch_params)
sparameters.update(hybridization_sparams)

theta = 0.5

miniapp = SerialMiniApp(dt=dt, theta=theta, w_initial=Un,
                        form_mass=form_mass,
                        form_function=form_function,
                        solver_parameters=sparameters,
                        bcs=bcs)

PETSc.Sys.Print("Solving problem")

uout = fd.Function(V1, name='velocity').assign(Un.subfunctions[0])
rhoout = fd.Function(V2, name='density').assign(Un.subfunctions[1])
thetaout = fd.Function(Vt, name='temperature').assign(Un.subfunctions[2])

ofile = fd.File(f'output/straka/{args.filename}.pvd',
                comm=comm)


def output_iteration(it):
    return (it+1) % args.output_freq


def assign_out_functions(it):
    uout.assign(miniapp.w1.subfunctions[0])

    if output_iteration(it):
        rhoout.assign(miniapp.w1.subfunctions[1])
        thetaout.assign(miniapp.w1.subfunctions[2])

        rhoout.assign(rhoout - rho_back)
        thetaout.assign(thetaout - theta_back)


def write_to_file(time):
    ofile.write(uout, rhoout, thetaout, t=time)


assign_out_functions(0)
write_to_file(time=0)

PETSc.Sys.Print('### === --- Timestepping loop --- === ###')
linear_its = 0
nonlinear_its = 0
solver_time = []

cfl_calc = convective_cfl_calculator(mesh)
cfl_series = []


def max_cfl(u, dt):
    with cfl_calc(u, dt).dat.vec_ro as v:
        return v.max()[1]


def preproc(app, it, time):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'### === --- Calculating time-step {it} --- === ###')
    PETSc.Sys.Print('')
    stime = MPI.Wtime()
    solver_time.append(stime)


def postproc(app, it, time):
    global linear_its
    global nonlinear_its

    etime = MPI.Wtime()
    stime = solver_time[-1]
    duration = etime - stime
    solver_time[-1] = duration
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'Timestep solution time: {duration}\n')
    PETSc.Sys.Print('')

    linear_its += miniapp.nlsolver.snes.getLinearSolveIterations()
    nonlinear_its += miniapp.nlsolver.snes.getIterationNumber()

    assign_out_functions(it)

    if output_iteration(it):
        write_to_file(time=time)

    cfl = max_cfl(uout, dt)
    cfl_series.append(cfl)
    PETSc.Sys.Print(f'Time = {time}')
    PETSc.Sys.Print(f'Maximum CFL = {cfl}')


# solve for each window
miniapp.solve(nt=args.nt,
              preproc=preproc,
              postproc=postproc)

PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Iteration counts --- === ###')
PETSc.Sys.Print('')

PETSc.Sys.Print(f'Maximum CFL = {max(cfl_series)}')
PETSc.Sys.Print('')

PETSc.Sys.Print(f'linear iterations: {linear_its} | iterations per timestep: {linear_its/args.nt}')
PETSc.Sys.Print(f'nonlinear iterations: {nonlinear_its} | iterations per timestep: {nonlinear_its/args.nt}')
PETSc.Sys.Print('')

PETSc.Sys.Print(f'Total DoFs: {W.dim()}')
PETSc.Sys.Print(f'Number of MPI ranks: {mesh.comm.size} ')
PETSc.Sys.Print(f'DoFs/rank: {W.dim()/mesh.comm.size}')
PETSc.Sys.Print('')

if len(solver_time) > 1:
    # solver_time = solver_time[1:]
    solver_time[0] = solver_time[1]

PETSc.Sys.Print(f'Total solution time: {sum(solver_time)}')
PETSc.Sys.Print(f'Average timestep solution time: {sum(solver_time)/len(solver_time)}')
PETSc.Sys.Print('')
