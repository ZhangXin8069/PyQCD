---
name: lqcddb
description: Review and use the lqcddb lattice-QCD distillation package for Wick contractions, perambulators, eigenvector vertices, correlator construction, MPI execution, statistical analysis, effective masses, GEVPs, momentum conversion, and contraction performance. Trigger for hadron correlators, distillation, lattice spectroscopy, or any lqcddb API or source audit.
---

# lqcddb

Treat every task as a chain

`physical definition -> convention and normalization -> tensor formula -> numerical algorithm -> implementation`.

Do not infer correctness from plausible output. State assumptions, distinguish implementation restrictions from physics requirements, and validate important results with identities or small independent calculations.

## Repository policy

- Treat `src/lqcddb/` and `src/lqcddb/test/` as read-only. Do not patch, reformat, or regenerate them.
- Write only user-requested scripts, reports, or documentation outside those directories.
- Before importing local code, set `PYTHONDONTWRITEBYTECODE=1` so review work does not create bytecode.
- Prefer named imports. Do not use `from lqcddb import *`: the current lazy export list resolves optional MPI symbols and can initialize MPI.
- Do not assume `README.md`, type stubs, package metadata, and implementation agree. Check the active source definition.
- Preserve the user's gamma basis, Euclidean conventions, source/sink order, Fourier sign, normalization, boundary conditions, and array-axis meanings.

## Select the process

Read the matching reference completely before doing that process:

- Source/README/API correctness audit: [references/code-audit.md](references/code-audit.md)
- Wick expansion and diagram equivalence: [references/wick-and-diagrams.md](references/wick-and-diagrams.md)
- Perambulators, vertices, and correlator implementation: [references/correlator-implementation.md](references/correlator-implementation.md)
- Eigenvectors, gauge links, smearing, and momentum phases: [references/eigenvectors-and-gauge.md](references/eigenvectors-and-gauge.md)
- Jackknife, Bootstrap, ratios, effective masses, fitting, and GEVP: [references/statistics-and-spectroscopy.md](references/statistics-and-spectroscopy.md)
- MPI, I/O, contraction caching, and performance estimates: [references/mpi-io-performance.md](references/mpi-io-performance.md)

Read multiple references when a task crosses process boundaries.

## Current-source safety gates

These are verified properties of the current repository, not general lattice-QCD claims:

- Keep dynamic diagram equivalence disabled unless tensor identities, time labels, gamma components, transpose signs, and connected topology have all been proved identical.
- Independently check eigenvectors with `max(abs(V.conj().T @ V - I))`; the package checks can accept non-orthogonal vectors.
- Do not rely on the current Stout-smearing routine for scientific data without an independent staple/index and SU(3) validation.
- Preserve complex Hermitian correlator matrices for GEVP work; the current solver projects them to real matrices.
- Treat contraction-adviser bandwidth and runtime numbers as heuristic until its ellipsis accounting and hardware efficiencies are independently corrected.

## Handoff standard

Report:

1. conventions and assumptions;
2. source files and callable signatures inspected;
3. invariant or independent reference used;
4. reproduced failures, including shapes and minimal inputs;
5. untested paths and environment limitations;
6. whether any file outside the read-only code area was changed.
