# Example scripts for "asQ: parallel-in-time finite element simulations using ParaDiag for geoscientific models and beyond"

These are the python scripts used to generate the data for the asQ library paper https://arxiv.org/abs/2409.18792.

## Section 3.2 "Heat equation example" script
`heat.py` is the example in Section 3.2 "A heat equation example".
It explicitly uses the `AllAtOnce*` objects to create each component of the all-at-once system.
Because it is hard-coded to four ensemble ranks (i.e. four MPI ranks in time), it must be run with a multiple of 4 MPI ranks, e.g.

```mpiexec -np 4 heat.py```

will run with 4 ranks in time, and serial in space, whereas

```mpiexec -np 8 heat.py```

will run with 4 ranks in time, and 2 ranks in each spatial communicator.
To change the time-parallelism, change the `time_partition` list in the script.

## Section 4 "Numerical Examples" scripts

All scripts use `argparse` to process command line argument, so will print out information on how to use them if run with the `-h` flag. They do not have to be run in parallel to do this.

`python <script_name.py> -h`

All scripts will also accept a `--show_args` argument, which will print out the value of all argparse arguments at the beginning of the script.
The default arguments for the `*_paradiag.py` scripts do not use time-parallelism, so can be run in serial.
To specify the time-parallelism, see the help for the `--nslices` and `--slice_length` command line arguments.

- The data in Section 4.1 "Advection equation" was generated with
  - `advection_serial.py` for the serial-in-time results.
  - `advection_paradiag.py` for the parallel-in-time results.
- The data in Section 4.2 "Linear shallow water equations" was generated with
  - `linear_shallow_water_serial.py` for the serial-in-time results.
  - `linear_shallow_water_paradiag.py` for the parallel-in-time results.
- The data in Section 4.3 "Nonlinear shallow water equations" was generated with
  - `nonlinear_shallow_water_serial.py` for the serial-in-time results.
  - `nonlinear_shallow_water_paradiag.py` for the parallel-in-time results.
- The data in Section 4.4 "Compressible Euler equations" was generated with
  - `vertical_slice_serial.py` for the serial-in-time results.
  - `vertical_slice_paradiag.py` for the parallel-in-time results.

 The `*_serial.py` scripts all use the `SerialMiniApp` class to run the serial-in-time method.
 The parallel-in-time shallow water equation scripts use the `ShallowWaterMiniApp` to set up the all-at-once system specifically fo the shallow water equations.
 The parallel-in-time advection and vertical slice scripts use the `Paradiag` class to construct the all-at-once system without having to manually create each `AllAtOnce*` object.
