---
name: quark-pdf-bare-matrix
description: Construct, fit, validate, or audit connected-quark PDF bare matrix elements from C3/C2 ratios in the Donghx Diagram_GluonPDF workflow, including unpolarized, helicity, transversity, six momentum directions, even/odd projections, bootstrap statistics, Sumratio/Ratio fits, and cut/window selection. Also route audits and handoffs for the local thin-link gradient-flow gluon OPE/ratio products through the dedicated reference. Do not use this skill for propagator generation or renormalized/matched physical PDFs.
---

# Quark PDF bare matrix elements

Use this skill for the local pipeline rooted at
`/public/group/lqcd/donghx/Diagram_GluonPDF`.  It covers the connected quark
three-point function through the unrenormalized matrix-element fit.  Treat
renormalization, LaMET matching, Fourier reconstruction, continuum limits, and
disconnected contributions as later layers unless the user explicitly places
them in scope.

## Route the task

- Read [references/physics-conventions.md](references/physics-conventions.md)
  when interpreting unpolarized/helicity/transversity, `u/d/u-d`, six
  directions, even/odd signals, signs, or what a “bare matrix element” means.
- Read [references/local-pipeline.md](references/local-pipeline.md) before
  running, changing, or auditing C3 construction, ratio bootstrap, file
  schemas, versions, production commands, or provenance.
- Read [references/fit-selection.md](references/fit-selection.md) before
  fitting Sumratio or Ratio, changing cuts/windows, comparing methods, or
  selecting/reporting a result.
- Read
  [references/gradient-flow-gluon-ratio.md](references/gradient-flow-gluon-ratio.md)
  only when auditing, plotting, extending, or handing off the local thin-link
  flowed-gluon OPE and disconnected-ratio workflow.  This is a narrow local
  exception to the connected-quark scope, not a general gluon-PDF theory
  contract.

Read every reference relevant to the requested mode, but do not load the other
references merely because they exist.

## Non-negotiable local contracts

- Current authoritative products are projected C3 `c3_ratio_v3`, ratio
  `ratio_v4`, and bare fits `bare_matrix_fit_v3`.  Older ratio v2 and bare-fit
  v1 products contain a C2 momentum-axis error and are traceability-only.
- Consume `valid_tau_mask` and `ratio_valid_mask`; invalid common-axis padding
  is NaN, never a zero-valued measurement.
- Keep the configuration list and the bootstrap index array aligned across
  directions, even/odd, channels, flavors, `p`, `z`, and both fit methods.
  Method differences require paired bootstrap errors.
- Preserve axes and validate them explicitly.  In particular, fit bootstrap
  arrays are `(component,boot,pabs,delta_z)` while comparison bootstrap arrays
  are `(boot,component,pabs,delta_z)`.
- Distinguish finite diagnostic values from accepted physical outputs.  Do not
  promote a failed quality gate, an allow/diagnostic flag, or a same-sign
  direction check into a validated result.
- Keep transversity at `pending_external_physics_confirmation` until its
  relative operator sign is checked against the external operator definition.
- Never choose a cut, window, or method point-by-point because it has the
  smallest error, largest Q, or best-looking central value.
- For the local flowed-gluon products, preserve the full complex helicity
  numerator, use `pol35` only in that numerator, and normalize both channels
  by `nopol`.  Never use legacy ratio-v1 for physics.
- Call flowed-gluon OPE and C3/C2 products finite-flow bare disconnected
  observables.  Do not call them gluon PDFs before state isolation,
  renormalization/flow-scheme conversion, mixing, matching, continuum and
  large-momentum control, and finite-coordinate reconstruction.

## Working pattern

1. Identify whether the request is construction, statistical fitting,
   convention diagnosis, production, or reporting.
2. Inspect the current code, configs, product metadata, completion JSON, and
   hashes before relying on a dated summary.  Preserve original and legacy
   products unless deletion is explicitly requested.
3. Apply the relevant reference contracts.  For generic correlated-bootstrap
   details, also use `pyqcd-statistics`; for spectral models requiring C2 gaps,
   use `pyqcd-physics-spectrum`.
4. Verify changes proportionally: unit tests and syntax checks for code;
   shapes, masks, shared indices, versions, hashes, and acceptance counts for
   products; direction-resolved plots or reports for sign claims.
5. Report the estimator, ensemble/configuration count, bootstrap seed/count,
   window/cuts, covariance treatment, fit gates, valid/invalid counts,
   systematic-error prescription, and every unresolved physics status.

“Complete” means the requested artifacts exist and pass their declared
contracts.  It does not mean that a bare matrix element is already a physical
quark PDF.
