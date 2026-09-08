# Helicity operator comparison

This directory contains a covariance-preserving comparison of

\\[
(T_H-S_H)_{\\rm odd},\\qquad
(T_H+S_H)_{\\rm odd},\\qquad
(T_H)_{\\rm odd}.
\\]

The current diagnostic is fixed to `tau/a^2=0.5`, `z/a=2`,
`Pz={3,4,5}`, and the frozen 231-configuration manifest with SHA-256
`abd94236885925483106b1691094d15a941ac83a48c7e2e57861245b3f4aa2da`.
It uses the full complex `pol35` correlator in the disconnected numerator
and `nopol` in the denominator. Vacuum subtraction and the denominator are
recomputed in every shared delete-one jackknife sample.

Run from the `Gradient_Flow_GluonPDF` repository root:

```bash
python3 analysis/helicity_operator_comparison.py --output-dir comparison
```

The NPZ stores complex central values and all 231 jackknife replicas.
The `TH_plus_SH` result is regression-tested against ratio-v2. The
`TH_minus_SH` selector is the theory target under the convention documented
in the Gradient Flow Note; the other two are diagnostics. These are
finite-flow bare disconnected ratios, not renormalized or matched PDFs.
