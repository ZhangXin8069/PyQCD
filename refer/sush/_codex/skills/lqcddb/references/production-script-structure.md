# CuPy/MPI Production Contraction Structure

The goal is to make the data contract, time decomposition, registries, contraction loop, gathering, and source estimator explicit. A shorter script is not necessarily correct.

Start a new program from the matching flat example and first read `flat-production-template.md` and `dynamic-contraction-workflow.md`. These documents define the input contract, registry/time/MPI checks, and the distinction between baseline and optimized versions. Hadron content, current multiplication order, and configuration metadata come from `physics.md` and must be expanded into concrete tokens.

## Production data contract

Before allocating the correlator, write the saved-axis table, including absolute/source-relative time, source/sink spin, Lorentz components, momentum, operator, link, and GEVP basis. Keep raw configurations, source sums, and source averages distinguishable.

## Computation stages

First fix the physical operator, source-operator conjugation, flavor balance, boundary sign, and output axes; then set the backend, MPI grid, and time ownership. Construct momentum phases and VDV/VVV using the spatial eigenvector layout, and record time, local index, conjugation, backend, and dtype for every registration. The root reads peram and distributes it along the time axis; the dynamic loop updates only declared registry keys. Check the plan and shapes after the first complete registration, then call `calculate_all`. Finally reconstruct global axes with `TGather`, apply `loop_tsrc` and the wraparound factor once, and save.

## Optimization requirements

Baseline and optimized versions must be numerically identical. Measure plan reuse, host/device residency, MPI transfer, and momentum blocking only without changing physical axes; report wall time, peak memory, transfer time, and residuals. A Roofline estimate cannot replace a measurement. Explicitly state GPU, MPI, or full-ensemble paths that were not run.
