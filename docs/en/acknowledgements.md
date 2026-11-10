# Acknowledgements

PyQCD's comparison-test programme and this documentation build on open-source
software from the following projects. They are used as independent
implementations, test targets or reference material; PyQCD does not vendor
their source into its runtime.

* NumPy, SciPy and opt_einsum — array semantics, reference contractions and
  contraction planning.
* CuPy — GPU array backend used by the CUDA comparisons.
* MatX (NVIDIA) — CUDA tensor library. Its `cutensor::einsum` is the subject
  of the `cpp/cuda/matx` bridge. Its 32-bit compute type is recorded as an
  upstream limitation, not as a PyQCD choice.
* Einsums — C++20 tensor algebra library. Its `compute_graph::einsum` is the
  subject of the `cpp/cpu/cpu-einsums` bridge.
* OpenBLAS and LAPACK — CPU BLAS/LAPACK implementations used by the measured
  engines.
* Sphinx, MyST-Parser and Read the Docs — documentation build and hosting.
* The lattice-QCD packages archived under `refer/` in the source repository
  (PyQUDA, EasyDistillation, LQCD_Master, lamet-agent and others) — API and
  workflow references.

Fixed commits and licences for the reference repositories are recorded in
`refer/git-rep/README.md` in the source repository.
