# Bundled reference profiles

Read the actual chosen files before copying. This inventory explains their inputs; it is not a second source of generated code. It supplies defaults **only when the user explicitly inherits the profile**. If a file changes, re-extract its values rather than trusting a stale table. Freeze file hashes in the task log at selection time.

| Field | `nn-2pt` | `njn-3pt` |
| --- | --- | --- |
| Python asset | `assets/contraction.NN.I0.5.2pt.cupy.py` | `assets/contraction.NJNp-.3pt.GEVP.cupy.na800.py` |
| Observable | 2pt, three sink and three source basis entries exactly as written | 3pt, one active sink/source basis entry; retain literal tokens and coefficients |
| Ensemble/geometry | E32P29, `[32,32,32,64]` | Same |
| Backend/grid | CuPy, `[1,1,1,2]`, two MPI ranks, temporal decomposition | Same |
| Eigenvector ports | `Nev_src=100`; `Nev_max=400` is present but unused in the 2pt calculation | Hadron/source 100; current port `Nev_max=400` |
| Stored scheme | Filename encodes exact 100, shells `[100,200,400,700,1100]`, sums `[20]*5`, extracts `[10]*5`, noise 200; check actual stored layout | Same; VDV explicitly multiplied by `create_omega_accelerate` with those parameters |
| Ordered momenta | Zero plus sorted shells based on `[0,0,1]`, `[0,1,1]`, `[1,1,1]` for both VDV/VVV | Same VDV order; each nonzero VVV shell is reversed |
| Source times | All `range(Nt)` | Same |
| Sink/current times | Sink offsets `range(Nt//2)`; declared `tsep=24`, `tgap=6`, `Tn` do not control this loop | `tsep=20`; current offsets `range(5,tsep-3)`, i.e. 5–16 inclusive |
| Current | None | Flavor/gamma tokens `u^d, gamma_1, d` with the existing hadron separators; `gamma_1=gamma(mu) @ gamma(5)`, mu=1,2,3,4 |
| Projection | `False`; preserve existing unused Projector registration | `True`; two `(Ns,2)` spin maps as written |
| Diagram/path flags | `ignore_dis=False`, `use_equivalence=False`, `optimize='auto'` | Same |
| Raw axes | `(t_src, t_sink_local, spin_sink, spin_src, momentum, sink_basis, source_basis)` | `(sink_basis, t_src, t_cur_local, mu, spin_sink_projected, spin_src_projected, momentum_N, momentum_M)` |
| Gather and estimator | `TGather(axis=1)`; `loop_tsrc(indx=[0,1], Boundary_Conditions='Antiperiodic')`, source sum | `TGather(axis=2)`; `loop_tsrc(indx=[1,2], Boundary_Conditions='Antiperiodic', Ctype='3pt', t_sep=tsep)`, source sum |
| Result basename | `corr_3pt_NN_2pt_src100.npy` (retain this existing name in exact mode) | `corr_3pt_NJN_tsep{tsep}.npy` |

The tables describe current code, not an independent proof of physical correctness. In particular, literal source coefficients/order and the three-entry 2pt basis must be read from the file. The API must confirm light-flavor mapping and tensor axes for a new target.

## Paths and shell inheritance

Both assets read under `/nexdata/project/lqcd/sush/eigensystem` and `/nexdata/project/lqcd/sush/perambulators`, with the E32P29 directory `beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64`. Copy the complete filename pattern from the asset. Results go under `/public/home/sush/distillation/0v2b/result_nex/E32P29/Px0Py0Pz0/ENV_{Nev_src}/conf{conf_id}`. These paths are inherited only by explicit reference selection, not by an unrelated request for E32P29.

Shared shell: `assets/contraction.NJNp-.3pt.GEVP.cupy.sh`, one node, two ranks/two GPUs, two CPUs per task, partition `nv100-ins,nv100-sug` on mgmt01, environment `/public/home/sush/envs/sm80`, no active wall-time directive (scheduler default), configurations 1000–2960 inclusive, step 40. `run_dir=/public/home/sush/distillation/0v2b`; the 3pt target basename is `contraction.NJNp-.3pt.GEVP.cupy`. For `nn-2pt`, adapt the three existing `exe/dir_file/output_file` target names to the requested two-point basename; keep the rest unless inputs differ.

The generated-file destination remains a user/workspace input; execution `run_dir` and staging destination need not coincide. Do not guess missing configuration IDs unless the user explicitly inherits the complete shell profile.

## External reference distinction

The previously linked external file `/public/home/sush/distillation/0v2b/contraction.NJNp-.3pt.GEVP.cupy.py` is a different reference: at inspection it used eight ranks, `tsep=14`, three active sink basis entries, host VVV/correlator arrays, `use_equivalence=True`, and a different time window. It cannot be treated as identical to the bundled `njn-3pt`. If the user selects it, read/freeze it and preserve its actual structure/parameters; do not substitute the bundled profile or silently alter physical coefficients.

## Maintained baseline repairs

On 2026-11-17, the bundled 3pt asset gained the missing `current_operator` definition, using the same explicit `u^d gamma_1 d` tokens as the external program and the existing gamma loop. The shared shell's MPI/GPU counts were aligned to the two-rank Python assets. Source-time comments in both Python assets were corrected from mean to sum; the numerical estimator was not changed. No external production program was modified. Subsequent exact-copy claims refer to the selected maintained file/hash, not an earlier revision.

The same day's progress update added flushed stage starts/completions, per-source and second-loop-level progress, a main-loop total, gather timing and a post-save filename/time message. The 2pt second level reports rank-0 local sink-basis work; the 3pt second level reports rank-0 local insertion-time work. No MPI collective or computational expression was changed. See [progress-timing.md](progress-timing.md); exact copies now include these messages.
