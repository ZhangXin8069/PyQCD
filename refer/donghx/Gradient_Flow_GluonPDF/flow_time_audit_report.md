# Gradient-flow gluon flow-time analysis

> Finite-flow bare disconnected matrix-element diagnostic; not a
> renormalized, matched, continuum gluon PDF.

## Input and estimator

- Fit products: `/public/group/lqcd/donghx/Diagram_GluonPDF/Gradient_flow_gluon/bare_fit_sumdiff_aic_v1`
- Flow times `tau/a^2`: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1, 1.4, 1.8, 2.2, 2.6, 3, 3.4, 3.8
- Ensemble: `L32x96`, `Nconf=231`, `Nboot=1500`, seed `1115`
- Manifest SHA-256: `abd94236885925483106b1691094d15a941ac83a48c7e2e57861245b3f4aa2da`
- Analysis code SHA-256: `9866a7614bac46598b6dba03046adacbbe4d2962eaff4dac772286ee349f54ed`
- Input fields: accepted (`fit_accepted`) only; `M_filled` and all fallback points are excluded.
- Flow model: correlated GLS `M(tau)=M0+c_tau*tau`; bootstrap error is the replica standard deviation (ddof=1).
- With `a=0.0775 fm`, the primary `tau=0.1--0.6` window spans `r_flow=0.069--0.170 fm`.

## Operators retained

| component | final projection |
|---|---|
| `unpolarized_TH_minus_SH_even_real` | Re[(T_U-S_U)_even] |
| `helicity_TH_odd_imag` | Im[(T_H)_odd] |
| `helicity_TH_plus_SH_odd_imag` | Im[(T_H+S_H)_odd] |

Both helicity constructions are carried through every flow window independently.

## Geometry and flow windows

The default scale-separated mask is `r_flow/a=sqrt(8*tau/a^2) < z/a`.
The local point `z=0` is retained in raw flow-dependence plots but excluded from the tau-to-zero fit.
At `z/a=1` only `tau/a^2=0.1` survives, so no two-parameter line fit is attempted.
For example, `tau/a^2=0.5, z/a=2` is rejected because `r_flow/a=2` exactly.

| window | flow times | fit-computed | quality-pass |
|---|---:|---:|---:|
| `small_0p1_0p3` | 3 | 207 | 195 |
| `small_0p1_0p4` | 4 | 207 | 201 |
| `small_0p1_0p5` | 5 | 207 | 201 |
| `small_0p1_0p6` | 6 | 206 | 202 |
| `extended_0p1_1p0` | 7 | 206 | 203 |
| `extended_0p1_1p8` | 9 | 206 | 205 |
| `extended_0p1_2p6` | 11 | 206 | 205 |
| `extended_0p1_3p8` | 14 | 206 | 206 |

Primary window: `small_0p1_0p6`.  Alternate windows are stability diagnostics, not point-by-point selections.

## Primary-window diagnostics

| component | fits computed | quality pass | max replica fallback fraction | median max cut-spread/error |
|---|---:|---:|---:|---:|
| `unpolarized_TH_minus_SH_even_real` | 69 | 67 | 0.7466666666666667 | 0.5274434364670356 |
| `helicity_TH_odd_imag` | 68 | 68 | 0.83 | 0.6066739282788322 |
| `helicity_TH_plus_SH_odd_imag` | 69 | 67 | 0.686 | 0.7209201732539358 |

### Primary `M0` at `z/a=2`

| component | Pz=3 | Pz=4 | Pz=5 |
|---|---:|---:|---:|
| `unpolarized_TH_minus_SH_even_real` | 0.17163 +/- 0.289 | -0.287374 +/- 0.342 | -0.453624 +/- 0.634 |
| `helicity_TH_odd_imag` | 0.00336021 +/- 0.0726 | 0.174805 +/- 0.0964 | 0.118747 +/- 0.17 |
| `helicity_TH_plus_SH_odd_imag` | 0.0703491 +/- 0.129 | 0.256391 +/- 0.183 | -0.107986 +/- 0.332 |

## Window stability (common quality-pass points)

The following RMS/median shifts are diagnostics only; they are not added to the statistical error.

| alternate window | component | common points | RMS shift | median absolute shift |
|---|---|---:|---:|---:|
| `small_0p1_0p3` | `unpolarized_TH_minus_SH_even_real` | 64 | 0.436158 | 0.274008 |
| `small_0p1_0p3` | `helicity_TH_odd_imag` | 65 | 0.100033 | 0.0582744 |
| `small_0p1_0p3` | `helicity_TH_plus_SH_odd_imag` | 63 | 0.231119 | 0.116809 |
| `small_0p1_0p4` | `unpolarized_TH_minus_SH_even_real` | 66 | 0.261119 | 0.130171 |
| `small_0p1_0p4` | `helicity_TH_odd_imag` | 68 | 0.0553547 | 0.0292592 |
| `small_0p1_0p4` | `helicity_TH_plus_SH_odd_imag` | 63 | 0.127319 | 0.0600671 |
| `small_0p1_0p5` | `unpolarized_TH_minus_SH_even_real` | 65 | 0.0666509 | 0.0533271 |
| `small_0p1_0p5` | `helicity_TH_odd_imag` | 68 | 0.0293653 | 0.0135872 |
| `small_0p1_0p5` | `helicity_TH_plus_SH_odd_imag` | 66 | 0.0583718 | 0.022477 |
| `extended_0p1_1p0` | `unpolarized_TH_minus_SH_even_real` | 67 | 0.0926254 | 0.055926 |
| `extended_0p1_1p0` | `helicity_TH_odd_imag` | 68 | 0.0391882 | 0.0267617 |
| `extended_0p1_1p0` | `helicity_TH_plus_SH_odd_imag` | 67 | 0.048538 | 0.0262409 |
| `extended_0p1_1p8` | `unpolarized_TH_minus_SH_even_real` | 67 | 0.140259 | 0.0715264 |
| `extended_0p1_1p8` | `helicity_TH_odd_imag` | 68 | 0.0787997 | 0.0625781 |
| `extended_0p1_1p8` | `helicity_TH_plus_SH_odd_imag` | 67 | 0.0983621 | 0.0392224 |
| `extended_0p1_2p6` | `unpolarized_TH_minus_SH_even_real` | 67 | 0.128899 | 0.069653 |
| `extended_0p1_2p6` | `helicity_TH_odd_imag` | 68 | 0.0818006 | 0.0591782 |
| `extended_0p1_2p6` | `helicity_TH_plus_SH_odd_imag` | 67 | 0.108619 | 0.0474104 |
| `extended_0p1_3p8` | `unpolarized_TH_minus_SH_even_real` | 67 | 0.13324 | 0.0790349 |
| `extended_0p1_3p8` | `helicity_TH_odd_imag` | 68 | 0.0839274 | 0.0605841 |
| `extended_0p1_3p8` | `helicity_TH_plus_SH_odd_imag` | 67 | 0.116556 | 0.0483789 |

## Interpretation and next gates

- The tau-to-zero intercept is a finite-lattice-spacing, finite-flow-scheme diagnostic.
- Extended windows are oversmearing/stability checks; the small `0.1--0.6` window is the registered primary.
- Before a PDF interpretation, repeat at controlled physical flow radius and lattice spacing, perform operator renormalization and gluon--singlet mixing, then apply the appropriate LaMET/pseudo-PDF matching and finite-coordinate analysis.
