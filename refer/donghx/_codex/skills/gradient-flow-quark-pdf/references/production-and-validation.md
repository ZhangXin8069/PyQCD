# Production and validation contract

## Current L32x96 record

Read
`/public/home/donghx/Gradient_Flow/Blending_L32x96/CURRENT_PRODUCTION_SETTINGS_20260902.md`
for the dated snapshot. Treat job states in that file as historical; query
Slurm and completion markers for current state.

The original `t/a^2=0.5` products use thin links, `epsilon=0.01`, temporal
window 4, 400 current basis vectors, 200 source/VVV vectors, `dz=0..20`,
`Pz=+3,+4,+5`, and `tsep=6..10`. The remaining-five-flow jobs use
`t/a^2=0.10,0.20,0.30,0.40,0.60` with windows 2,3,3,4,5.

## Pre-production gates

1. Verify PyQUDA/QUDA exposes gauge flow and Dirac-Laplace functionality.
2. Run zero-flow regression against the unflowed VDV/C2/C3 reference.
3. Check `V^dag V`, axes, color order, stochastic weights, and compression.
4. Check flowed-link unitarity/determinant, action/plaquette, and temporal
   center-phase convention.
5. Reproduce the free-field heat kernel and halve `epsilon`.
6. Test fermion anti-periodic wrap and gauge covariance.
7. Compare direct final-time flow with checkpoint restart.
8. Validate discarded temporal norm and compare selected C3 at wider windows.
9. Compare factorized contraction to the explicit double-time sum on small
   deterministic tensors.
10. Run one-/four-rank consistency and validate all output axes and metadata.

## Completion and cleanup

Only clean recoverable `basis/`, `vertex/`, and `flowed_gauge/` caches after
all expected raw and channel-projected outputs pass their completion guard.
Save a cleanup receipt with targets and bytes. Keep incomplete checkpoints for
resume. Node-local flowed perambulators and staging data should be removed by
the job trap unless a diagnostic flag explicitly retains them.

## Analysis gates

- Keep configuration lists and bootstrap indices aligned across all flow
  times and channels.
- The existing single-positive-z ratio is a pilot diagnostic. Use the full
  six-direction VDV estimator for final bare physics.
- Apply fit validity and tau masks; NaN padding is not data.
- Determine `Z_chi` for every ensemble and flow time with documented
  derivative and mass prescriptions.
- At fixed physical flow time/radius, combine the three spacings and monitor
  `a^2/t`; then test residual `t/z^2`, `tPz^2`, and `t Lambda^2` effects.
- Separate statistical, fit-window, flow-time, lattice-spacing, matching-scale,
  perturbative-order, finite-volume, finite-z, and distillation systematics.

## Status language

- `finite-flow connected correlator`: raw C2/C3;
- `bare matrix element`: fitted C3/C2 before ringed normalization/SFTE;
- `MSbar bilocal diagnostic`: after validated `Z_chi` and SFTE, before LaMET;
- `matched nonsinglet PDF`: only after channel-specific LaMET, controlled
  limits, and reconstruction;
- never call connected `u` or `d` a complete singlet PDF without disconnected
  contributions and required mixing.
