# Code and data contract

## Authoritative local paths

Production and multi-flow code:

- `/public/group/lqcd/donghx/Gradient_Flow/Blending_L32x96`: original
  single-flow production used by historical/submitted jobs;
- `/public/home/donghx/Gradient_Flow/Blending_L32x96`: isolated multi-flow
  successor and active remaining-five-flow driver;
- `/public/home/donghx/Gradient_Flow/Blending_L32x96/small_flow_time`: SFTE
  post-processing that requires a validated ringed normalization.

Bare matrix analysis:

- `/public/group/lqcd/donghx/Diagram_GluonPDF/Gradient_flow`;
- current products are `projected_c3_flow_v1`, `ratio_flow_v1`, and
  `bare_matrix_fit_flow_v1` pilot diagnostics.

Theory note:

- `/public/home/donghx/Note/Gradient_Flow/gradient_flow_quasi_pdf_note.tex`;
- `/public/home/donghx/Note/Gradient_Flow/reference.bib`.

## Core tensor definitions

```text
flowed basis: (eigen,z,y,x,color) = (400,32,32,32,3)
vertex:       (vdv_sign,delta_z,eigen_left,eigen_right) = (2,21,400,400)
corr_2pt:     (pol,mom,t_source,t_sink)
corr_3pt:     (sign,pol,mom,dz,current_m,current_o,t_source,t_current)
```

The factorized contraction builds

\[
H_{uSLpamb}=\sum_{t,n}J_{tuSLnp}A_{tambn}
\]

once per current/sink group, reuses it for all six Wick terms, and deletes it.
Do not merge Wick topologies without an explicit tensor proof.

## Implemented physics conventions

- initial gauge links are thin/`Nosmear`; HYP pre-smearing is rejected;
- proton diquark operator is `C gamma5 gamma4`;
- C2 source/sink remain unflowed;
- fermion flow uses anti-periodic temporal boundary conditions;
- C2/C3 wrap signs follow the reference connected-diagram script;
- ordinary flowed endpoints are used; ringed `Z_chi` is currently missing;
- current C3 files retain open current-spin indices and can project
  unpolarized, helicity, and two transversity diagnostics without repeating
  the Wick contraction.

## Multi-flow implementation

The home successor supports a checkpoint schedule such as

```text
0.10:2,0.20:3,0.30:3,0.40:4,0.50:4,0.60:5
```

where the second number is the temporal-window half width. One QUDA trajectory
is restarted between checkpoints; a direct-versus-restart GPU gate must pass.
Every checkpoint has a separate cache tag and discarded-norm validation.

The default `pilot_z` profile uses `Pz=+3,+4,+5`, `tsep=6..10`. The
`six_direction` profile uses the six momentum directions and must be used for
the final symmetry estimator after a validated pilot. Keep profiles in
separate roots because historical filenames do not encode the profile.

## SFTE post-processing

`small_flow_time/apply_sfte.py` consumes fitted bare-matrix NPZ products. It:

- refuses a `ringed_z_chi_v1` JSON unless `status=validated`;
- checks `(component,pz,delta_z)` and `(component,boot,pz,delta_z)` axes;
- applies one `Z_chi` and the corrected one-loop coordinate-space factor;
- preserves bootstrap alignment and stores separate input and `Z_chi` errors;
- saves explicit window masks and states that LaMET is not applied.

The template JSON is not data and must never be promoted to production.

## Safe modification rule

Queued Slurm jobs read scripts at execution time. Do not edit their referenced
files merely because already-running processes have imported them. Create an
isolated successor, run CPU tests and GPU smoke gates, then submit a new job
that names the successor explicitly.
