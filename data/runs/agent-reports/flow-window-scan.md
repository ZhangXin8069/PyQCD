# Flow-window restricted TMD diagnostic

Date: 2026-10-22 (Asia/Shanghai)

## Purpose

The tau=3 production set has `r_flow/a = sqrt(24) = 4.899`, so every nonzero
`b=1..4` and small `z=1..4` fails the nonlocal SFTE window. This report records
a controlled lower-flow-time scan that explicitly keeps only
`sqrt(8*tau) < z,b`.

## tau=0.5

Inputs:

```text
conf_ids = 4150,6250,6450,6650,6850,7050,7250,7450,7650,7850,8050
z/a = 0,3,4,5,6,7,8,9,10,11,12
b/a = 3,4
tau = 0.5
eps = 0.05
bootstrap replicas = 1000
```

For `tau=0.5`, `r_flow/a=2`; all retained nonlocal z and b satisfy the
window. Results:

```text
P000 quality 17/22, central fallback 5, replica fallback mean 0.78%
P200 quality 18/22, central fallback 4, replica fallback mean 1.06%
P400 quality 22/22, central fallback 0, replica fallback mean 0.064%
```

Failed central cells:

```text
P000: (z,b)=(0,4),(3,3),(5,3),(5,4),(10,4)
P200: (z,b)=(0,4),(3,3),(3,4),(4,3)
P400: none
```

All listed failures are labelled minimum-statistical-error fallback; none are
promoted to quality-accepted values. P400 still has a nonzero replica fallback
fraction, so the strict no-fallback physical gate remains closed.

## tau=1.0

The same 11 configurations, z/b support and bootstrap count were used with
`tau=1.0` (`r_flow/a=sqrt(8)=2.828`). Results:

```text
P000 quality 14/22, central fallback 8, replica fallback mean 6.82%, max 66%
P200 quality 17/22, central fallback 5, replica fallback mean 0.286%, max 1.3%
P400 quality 22/22, central fallback 0, replica fallback mean 0.059%, max 0.2%
```

The P000 degradation at larger flow time (14/22 versus 17/22 at tau=0.5) is a
real flow-time sensitivity diagnostic, not evidence for selecting the
better-looking tau. P400 remains central-quality complete at both flow times
but still has nonzero replica fallback. A two-point flow-time comparison is
not a controlled tau-to-zero extrapolation.

## Reproducible entry

```bash
python -m pyqcd.testing.tmd.flow_window_scan \
  --source-run data/runs/tmd-production-10conf \
  --out data/runs/tmd-flow-window-tau0p5-b3to4 \
  --z-list 0,3,4,5,6,7,8,9,10,11,12 \
  --b-list 3,4 --tau 0.5 --eps 0.05 --backend numpy

python -m pyqcd.testing.tmd.flow_window_scan \
  --source-run data/runs/tmd-production-10conf \
  --out data/runs/tmd-flow-window-tau1p0-b3to4 \
  --z-list 0,3,4,5,6,7,8,9,10,11,12 \
  --b-list 3,4 --tau 1.0 --eps 0.05 --backend numpy
```

The outputs include strict OPE caches under `data/conf*/` and
`analysis/flow_window_scan.json`. They are deliberately marked
`physical_acceptance=false`; soft/rapidity, gluon-quark mixing, Z_R,
multiple lattice spacings and a controlled continuum/flow-time limit are
still missing.
