---
name: gradient-flow-quark-pdf
description: Build, audit, normalize, or analyze gradient-flowed quark quasi-PDF C2/C3 calculations in the Donghx distillation workflow, including 4D PyQUDA flow, multi-flow checkpoints, ringed fermions, small-flow-time conversion, and the handoff to channel-specific LaMET matching. Do not treat finite-flow connected correlators as physical PDFs.
---

# Gradient-flow quark PDF

Follow the chain

`operator scheme -> 4D gauge/fermion flow -> distillation vertex -> C2/C3 -> bare fit -> ringed normalization -> SFTE -> LaMET`.

Keep these boundaries explicit:

- current production computes finite-flow connected correlators with ordinary
  flowed fermion fields;
- `C3/C2` cancels nucleon overlaps, not flowed-fermion normalization;
- the corrected small-flow coefficient applies to a ringed, thin-link,
  four-dimensional flowed bilocal;
- SFTE converts the flowed bilocal to an MSbar quasi-PDF operator; it does not
  replace the channel-specific LaMET kernel or reconstruct a physical PDF;
- connected `u` and `d` are not complete singlet PDFs; prefer `u-d` for the
  first controlled analysis.

## Route the task

- Read [references/theory-and-matching.md](references/theory-and-matching.md)
  for 3D/4D flow, the kernel, ringed fields, SFTE coefficients, scale windows,
  linear divergences, or matching order.
- Read [references/code-and-data.md](references/code-and-data.md) before
  changing or auditing flowed basis, generalized vertices, perambulator
  factorization, C2/C3 axes, channel projection, or cleanup.
- Read [references/production-and-validation.md](references/production-and-validation.md)
  before submitting, resuming, monitoring, or interpreting L32x96 jobs and
  products.

For bare C3/C2 fitting and bootstrap contracts, also use
`quark-pdf-bare-matrix`. For distillation tensor identities, also use
`lqcddb`. Do not patch submitted-job scripts while queued jobs may still read
them; create a versioned or isolated successor and test it first.

## Non-negotiable analysis gates

1. Require the same thin gauge history for `B_mu(t)`, both flowed endpoints,
   and the Wilson line; reject HYP+flow data for thin-link matching.
2. Require explicit flow-time, step size, temporal-window, axes, gamma,
   proton-operator, boundary-condition, and input-path metadata.
3. Measure `Z_chi(t)` independently. Never insert a guessed value or silently
   label ordinary flowed fields as ringed.
4. Use the corrected Brambilla--Wang 2024 SFTE convention and record the sign
   convention for `delta_m`. Do not combine it with an overlapping hybrid or
   ratio linear-divergence subtraction.
5. Build and save masks for `a/r_flow`, `r_flow/|z|`, `sqrt(t) Pz`, and
   `t Lambda_QCD^2`; vary thresholds globally, never point by point.
6. Match ensembles at fixed physical flow time/radius. A common `t/a^2` is not
   a common physical flow time.
7. Preserve shared configuration lists and bootstrap indices across flow
   times, channels, flavors, directions, and fit methods.
8. Keep transversity at `pending_external_physics_confirmation` until its
   relative operator sign is checked against the external definition.

Report which layers are implemented, validated, diagnostic, or missing.
