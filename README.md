[![Lint and test](https://github.com/firedrakeproject/asQ/actions/workflows/test.yml/badge.svg)](https://github.com/firedrakeproject/asQ/actions/workflows/test.yml) [![DOI](https://zenodo.org/badge/329630815.svg)](https://doi.org/10.5281/zenodo.14592039)



# asQ: Parallel-in-time simulation of finite element models using ParaDiag

asQ is a python package for developing parallel-in-time ParaDiag methods for finite element models.
asQ is designed to allow fast prototyping of new ParaDiag methods, while still being scalable to large HPC systems with parallelism in both time and space.

This is achieved using the Firedrake and PETSc libraries.
The finite element models are defined by specifying the weak form using [Firedrake, "*an automated system for the portable solution of partial differential equations using the finite element method*](https://www.firedrakeproject.org/)", and the linear and nonlinear solvers required are provided by [PETSc, "*the Portable, Extensible Toolkit for Scientific Computation*"](https://petsc.org/release/).

See the arXiv paper https://arxiv.org/abs/2409.18792 for a description and demonstration of asQ. The code used for this paper can be found in the [asq_manuscript_examples](https://github.com/firedrakeproject/asQ/tree/master/asq_manuscript_examples) directory.

## ParaDiag

ParaDiag is a parallel-in-time method, meaning that it solves for multiple timesteps of a timeseries simultaneously, rather than one at a time like traditional serial-in-time methods.
This [review article](https://arxiv.org/abs/2005.09158) provides a good introduction to the method.
asQ implements the ParaDiag-II family of methods based on creating a block-circulant approximation to the all-at-once system which can be block-diagonalised with the FFT and solved efficiently in parallel.

## Requirements

asQ can be installed as part of a Firedrake installation. You can find instructions for installing Firedrake here: ([download instructions](https://www.firedrakeproject.org/download)).
To install asQ, pass the arguments `--install asQ` to the `firedrake-install` script.

## Getting started

The best place to start is the arXiv paper and associated examples in [this directory](https://github.com/firedrakeproject/asQ/tree/master/asq_manuscript_examples).

Other examples can be found in the [examples directory](https://github.com/firedrakeproject/asQ/tree/master/examples) and more advanced scripts can be found in the [case studies directory](https://github.com/firedrakeproject/asQ/tree/master/case_studies), including scripts for the shallow water equations and a model for stratigraphic evolution of the sea floor.

## Help and support

If you would like help setting up and using asQ, please get in touch and [raise an issue](https://github.com/firedrakeproject/asQ/issues).

## Contributors
Werner Bauer (University of Surrey)

Colin Cotter (Imperial College London)

Abdalaziz Hamdan (Imperial College London)

Joshua Hope-Collins (Imperial College London)

Lawrence Mitchell (Nvidia)

## Funding

Development of asQ has been supported by the following funders and grants: 

EPSRC (EP/W015439/1 & EP/R029628/1)

NERC (NE/R008795/1)

ExCALIBUR & UK Met Office (SPF EX20-8 Exposing Parallelism: Parallel-in-Time)
