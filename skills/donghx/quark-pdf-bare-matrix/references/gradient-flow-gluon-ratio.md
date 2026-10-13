# Thin-link gradient-flow gluon OPE and disconnected ratio

Use this reference only for the local `L32x96` gluon workflow.  It records the
verified state as of 3 September 2026.  Reinspect product metadata and
completion receipts before relying on the snapshot.

## Scope and status

The pipeline starts from unsmeared gauge configurations, applies four-
dimensional Wilson flow, constructs nonlocal gluon OPE loops, and correlates
them with stored nucleon two-point functions.  Its endpoint is a
**finite-flow bare disconnected C3/C2 ratio**, not a renormalized or matched
gluon PDF.

Main roots:

```text
OPE code:
  /public/group/lqcd/donghx/Gradient_Flow_GluonPDF
OPE production products:
  /public/group/lqcd/donghx/Gradient_Flow_GluonPDF/output_v5
ratio code:
  /public/group/lqcd/donghx/Diagram_GluonPDF/Gradient_flow_gluon
authoritative ratios:
  /public/group/lqcd/donghx/Diagram_GluonPDF/Gradient_flow_gluon/results_v2
two-point input:
  /public/group/lqcd/donghx/2pt_Result/beta6.41_mu-0.2295_ms-0.2050_L32x96
```

Legacy `Gradient_flow_gluon/results/` ratio-v1 files are provenance-only.
They divided the helicity channel by the near-zero ensemble mean of `pol35`
and must not be used for physics, fits, or plots.

## Flow and field-strength definition

The production implementation is `src/flowed_gluon_ope.cpp`.  Gauge links are
evolved from thin links with a third-order Runge--Kutta Wilson-flow update and
step size `epsilon=0.01`.  There is no HYP or other pre-smearing.

At flow time `tau`, the four-leaf Clover tensor is proportional to

```text
F_mu_nu = -i (Q_mu_nu - Q_mu_nu^dagger) / 8,
```

with the code's lattice normalization.  The production selector is the
color-traceless projection.  The dual tensor is

```text
Fdual_mu_nu = (1/2) epsilon_mu_nu_rho_sigma F_rho_sigma.
```

The gauge-invariant bilocal separated along `z_dir=2` is

```text
M_{mu lambda;nu rho}(z)
 = sum_x Tr[F_{mu lambda}(x+z zhat) W(x+z,x)
            F_{nu rho}(x) W(x,x+z)].
```

With Euclidean time index `3`, transverse indices `i=0`, `j=1`, the saved
component identities are

```text
Mtiti(unpolarized) = M_30;30 + M_31;31
Mijij(unpolarized) = 2 M_01;01
combined(unpolarized) = Mtiti - Mijij

combined(helicity) = Mtiti + Mijij
physical helicity orientation = odd_difference = raw(+z) - raw(-z)
```

The unpolarized form is the Euclidean-index version of
`M_tx;tx + M_ty;ty - 2 M_xy;xy` in arXiv:2510.26425v2.  The helicity form and
odd-in-`z` target follow arXiv:2207.08733v2.  Do not copy the HYP-smearing or
hybrid-renormalization constants of the former paper into the thin-link flow
scheme.

## OPE schema-v2

One OPE file has shape

```text
(field_projection, channel, z_orientation, component, z, t)
= (2, 2, 4, 6, 25, 96)
```

Labels are

```text
field_projection: legacy_untraced, traceless
channel:          unpolarized, helicity
z_orientation:    plus_z_raw, minus_z_raw, even_sum, odd_difference
component:        combined, Mtiti, Mijij, ti, tj, ij_single
```

`even_sum` and `odd_difference` are stored without a factor `1/2`.  Preserve
raw orientations and all components when changing code: they provide exact
linear-identity tests.  Use `traceless` for production analysis.

Completed flow times `tau/a^2` are

```text
0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8.
```

The smoothing radius is `r_flow/a = sqrt(8 tau/a^2)`.  A larger list of flow
times does not by itself establish a valid small-flow window; test lattice-
artifact suppression and oversmearing against `a`, `|z|`, `1/Pz`, and the
hadronic scale.

## Correct disconnected estimator

Use the common, frozen 231-configuration manifest

```text
manifests/common_all14_plusz_P345.tsv
SHA-256 abd94236885925483106b1691094d15a941ac83a48c7e2e57861245b3f4aa2da
```

For every jackknife sample, source time, insertion time, and channel,

```text
C3_unpol = <O_unpol C2_nopol>_cfg
           - <O_unpol>_cfg <C2_nopol>_cfg

C3_hel   = <O_hel C2_pol35>_cfg
           - <O_hel>_cfg <C2_pol35>_cfg

R_channel = mean_tsrc(C3_channel)
            / mean_tsrc(<C2_nopol>_cfg).
```

The helicity numerator uses the **full complex `pol35` two-point function**.
Do not take its imaginary part before covariance construction or jackknife.
Both channels use `nopol` in the denominator.  Only at the final Euclidean
phase projection use

```text
Re[-i R_hel] = Im(R_hel).
```

Delete-one jackknife must recompute the vacuum subtraction and the `nopol`
denominator inside every sample.  Preserve complex central values; store real
and imaginary jackknife errors separately.

## Ratio-v2 production contract

The 14 tasks from Slurm job `3561437` completed with `ExitCode=0:0`, and all
14 products passed strict validation.  Current parameters are

```text
lattice:             32^3 x 96
Nconf:               231
Pz:                  +3, +4, +5
tsep:                5..15
z/a:                 0..24
flow epsilon:        0.01
field projection:    traceless
saved components:    combined, Mtiti, Mijij
input scheme:        thin_link_no_hyp_no_smear
```

Only `+z` momentum is available because the OPE production has only
`z_dir=2`; do not present it as the six-direction average used by the quark
workflow.

The ratio arrays have shape

```text
(channel, z_orientation, component, direction, z, tsep, insertion,
 momentum_abs)
= (2, 4, 3, 1, 25, 11, 16, 3).
```

Honor `valid_insertion_mask`.  The intended physics selectors are

```text
unpolarized: channel=unpolarized, orientation=even_sum, component=combined
helicity:    channel=helicity, orientation=odd_difference, component=combined
```

## Current signal and next analysis layer

The diagnostic

```text
plots/ratio_plateau_tau0p500_z2_P345.{png,pdf,json}
```

uses `tau/a^2=0.5`, `z/a=2`.  The unpolarized interior is approximately
`-0.5`.  The phase-projected helicity signal is approximately `-0.08` to
`-0.10`; `Pz=3,4` are clearer and `Pz=5` becomes noisy at larger `tsep`.
Endpoint points can contain contact effects.  These are diagnostics, not fit
results.

Before a matrix element or PDF claim, perform and document:

1. endpoint cuts and plateau, summed-ratio/Feynman--Hellmann, or multi-state
   state-isolation fits;
2. correlated resampling and fit-window stability;
3. continuum analysis at controlled physical flow radius and a justified
   small-flow or flow-scheme conversion;
4. gluon--singlet-quark mixing and operator renormalization;
5. target-mass and higher-twist control;
6. LaMET or pseudo-PDF matching and finite-coordinate reconstruction.

Do not call a finite-flow bare ratio a physical PDF.
