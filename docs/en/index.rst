PyQCD
=====

PyQCD is a lattice-QCD toolkit in Python with optional native kernels. It
provides building blocks for propagator and operator contractions, gradient
flow, renormalisation, and gluon TMD-PDF extraction, plus a self-contained
contraction engine with an ``opt_einsum``-compatible API.

This site is a short introduction to what the package does and how to use it.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   overview
   usage
   install

Where to start
--------------

* :doc:`overview` — what each subpackage provides and how the contraction
  engine is layered.
* :doc:`usage` — runnable examples for every major module.
* :doc:`install` — building the optional native backends, backend selection,
  MPI, and the test entry points.

Scope
-----

These pages describe the package and its usage. The repository also keeps
research notes and LaTeX reports under ``docs/``; those are separate material
and are deliberately not part of this site.
