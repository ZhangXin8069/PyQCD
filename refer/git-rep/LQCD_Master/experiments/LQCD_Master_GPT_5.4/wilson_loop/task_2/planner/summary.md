## Revised plan summary

This revision keeps the task in **freeform mode**, which is the correct structure for a **pure-gauge Wilson-loop measurement** rather than a hadron correlator workflow. The core objective is now stated more precisely: the deliverable is a **per-configuration** value
\[
W_{\mathrm{cfg}}(R=1,T=2)=\frac{W_{XT}+W_{YT}+W_{ZT}}{3},
\]
with each plane contribution averaged over **all spacetime origins** on the loaded gauge field.

## What was fixed

- The plan now explicitly separates the **active observable definition** from the inherited ensemble metadata. The ensemble block is kept because the rewrite must remain consistent with the provided configuration, but the plan states clearly that fermion-related entries there are **irrelevant to the gauge-only observable**.
- The previous ambiguity between a single-configuration measurement and an ensemble average is removed. This plan now states unambiguously that it computes **only** the value on **cfg 10000**.
- The reduction recipe is made concrete and executable:
  1. extract per-site color matrices from `gauge.loop()`,
  2. take `ReTr`,
  3. divide by `Nc`,
  4. average over the **global** four-volume,
  5. average the three plane values with equal weight.
- A mandatory validation step is added for the `gauge.loop()` return object and for the **local lattice shape used in MPI gather/reduction**, reducing the risk of silent normalization bugs.
- The fourth `gauge.loop()` group is now explained explicitly as **padding only**, with zero weight, so only XT/YT/ZT contribute.
- The file-write contract is tightened operationally: **rank 0 only**, exact filename, current working directory as the run directory, and file contents equal to **one numeric scalar plus newline and nothing else**.
- The choice of **unsmeared links** is no longer left as an arbitrary default; it is stated as the observable choice for this run.

## Final physics/execution picture

The executable measurement remains simple and stable:
- initialize PyQUDA with the given lattice and process grid,
- load the gauge configuration,
- build the three rectangular paths for \(R=1\), \(T=2\),
- measure the XT, YT, ZT loops,
- perform the verified global reduction,
- output the final averaged scalar to
`wilson_loop_R1_T2_avg_xt_yt_zt_cfg10000.txt`.

So the revised plan preserves the reasonable original structure while making the physics scope, normalization, MPI behavior, and output contract much tighter and safer.