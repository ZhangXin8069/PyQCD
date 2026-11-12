# Appendix: specialized paths

Every contraction listed here is dispatched by **topology signature** (label
independent), so `ij,jk->ik` and `ij,jb->ib` take the same path. Counts are
generated from the corpus and the registry, not typed by hand.

* pyqcd version: `0.3.4` (backend `cuda`, array backend `numpy`)
* corpus size: **8638** contractions (2582 base table + 6056 appendix variants)
* topology index: **8638** signatures in **24** buckets

## 1. Corpus coverage by case

| Case | Contractions |
|---|---:|
| 1. | 2 |
| 10. | 120 |
| 13. | 120 |
| 14. | 720 |
| 2. | 6 |
| 3. | 120 |
| 4. | 720 |
| 5. | 6 |
| 6. | 24 |
| 7. | 720 |
| 9. | 24 |
| 动量指标与流分量指标 | 6056 |

## 2. Coverage by operand count

| Operands | Contractions |
|---:|---:|
| 6 | 4 |
| 7 | 12 |
| 9 | 30 |
| 10 | 120 |
| 12 | 72 |
| 13 | 1200 |
| 14 | 1440 |
| 16 | 5760 |

## 3. Literal registry (hand-listed shapes)

These are matched by exact subscript string first, before topology lookup.
`kind` reports what actually runs: `kernel(...)` means a native specialized
kernel, `plan-cache` means only the reusable plan is shared.

| Name | Kind | Subscripts |
|---|---|---|
| gauge-propagator(E4) | `plan-cache` | `zyxcTabn,zyxgTdef,Gae,Tzyxgc->TGdbfn` |
| gamma-sandwich(E5) | `plan-cache` | `ab,...bce,cd->...dae` |
| sush-2pt(S1) | `plan-cache` | `eofp,ambn,cgdh,ce,mo,Nbdf,Mnph,ga->MN` |
| sush-3pt-munu(S2) | `plan-cache` | `gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG` |
| sush-2pt-mu(S3) | `plan-cache` | `manb,cedf,bd,Mfn,ac,MFem->MF` |

## 4. What runs, and how to tell

| Situation | Log line | Speed |
|---|---|---|
| Topology hit with native kernel | `专用路径: topology:<digest> [kernel: kupl_dedicated]` | 2–4× vs generic |
| Topology hit, no kernel | `专用路径: ... [plan-cache]` | ≈ generic |
| No specialization | `通用路径: 未命中特化形态 (...)` | baseline |

Both notices print **once per shape per process**. Disable them with
`PYQCD_EINSUM_NOTICE=0`; make the specialized notice fire on *every* call with
`PYQCD_EINSUM_SPECIALIZE_VERBOSE=1`.

## 5. Verifying the list yourself

```bash
python -m pyqcd.einsum._specialize                 # registered shapes + self-check
python -m pyqcd.testing._topology_full_bench --dim 3  # run the whole corpus
python -m pyqcd.testing._topology_perf_multidim       # multi-size speed-up
```
