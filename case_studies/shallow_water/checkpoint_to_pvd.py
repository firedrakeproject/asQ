import firedrake as fd
from firedrake.petsc import PETSc

import argparse
parser = argparse.ArgumentParser(
    description='Read a timeseries from a checkpoint file and write to a pvd file',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--ifilename', type=str, default='time_series', help='Name of checkpoint file.')
parser.add_argument('--ofilename', type=str, default='time_series', help='Name of vtk file.')
parser.add_argument('--ifuncname', type=str, default='func', help='Name of the Function in the input checkpoint file.')
parser.add_argument('--ofuncnames', type=str, nargs='+', default='func', help='Names of the (sub)Function(s) to write to the output pvd file.')
parser.add_argument('--nsteps', type=int, default=0, help='How many timesteps in the checkpoint file. If nsteps is 0 then only one Function is written and the idx argument to CheckpointFile.load_function is not used.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

is_series = args.nsteps > 0

with fd.CheckpointFile(f"{args.ifilename}.h5", "r") as checkpoint:
    pfile = fd.File(f"{args.ofilename}.pvd")
    mesh = checkpoint.load_mesh()

    if is_series:
        idx = 0
        func = checkpoint.load_function(mesh, args.ifuncname, idx=idx)
    else:
        func = checkpoint.load_function(mesh, args.ifuncname)

    if len(args.ofuncnames) != len(func.subfunctions):
        msg = "--ofuncnames should contain one name for every component of the Function in the CheckpointFile"
        raise ValueError(msg)

    outputfuncs = tuple(fd.Function(f.function_space(), name=fname).assign(f)
                        for f, fname in zip(func.subfunctions, args.ofuncnames))
    if is_series:
        pfile.write(*outputfuncs, t=idx)
    else:
        pfile.write(*outputfuncs)

    for idx in range(1, args.nsteps):
        func = checkpoint.load_function(mesh, args.ifuncname, idx=idx)
        for g, f in zip(outputfuncs, func.subfunctions):
            g.assign(f)
        pfile.write(*outputfuncs, t=idx)
