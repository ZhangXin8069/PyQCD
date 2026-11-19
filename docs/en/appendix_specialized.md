# Appendix: specialized-path reference

**Strongly recommended: check an equation with
`python -m pyqcd.einsum.verify_specialized "<equation>"` before you tune or
benchmark it.** It answers "does this equation have a specialized route at all?" in
milliseconds, without building operands and without calling `contract()`, so it is
safe in a production start-up self-check.

Coverage of the current corpus:

* pyqcd version: `0.5.4`
* specialized paths: **8638** (2582 base table + 6056 appendix variants)
* distinct topology signatures: **8638**, indexed in **24** buckets

Matching is by **topology signature**, not by string: label names are irrelevant
(`ij,jk->ik` and `ij,jb->ib` take the same path), whitespace and an omitted output do
not change the signature, while operand order and the index order inside an operand
do.

## 1. Querying one equation

```bash
python -m pyqcd.einsum.verify_specialized "agbh,cmdn,eqfr,ospt,ikjl,ce,mo,gi,kq,bdf,np,hj,lrt->as"
python -m pyqcd.einsum.verify_specialized "<eq>" --json
```

Exit codes make the query scriptable: `0` the equation hits a specialized path, `1` it
does not (the generic step-by-step path runs), `2` the equation is malformed or the
arguments are wrong.

The text report answers, field by field:

| Field | Meaning |
|---|---|
| `topology` | the renumbered signature: labels replaced by their first-appearance index |
| `arity/slots` | operand count and total index slots (operands plus output) |
| `digest` | stable hash of the signature; the path name is `topology:<digest[:8]>` |
| `specialise` | whether specialisation is currently enabled (`PYQCD_EINSUM_SPECIALIZE`) |
| `result` | `HIT` with the matched path, or `MISS` with the reason |
| `kernels` | the kernels the match **declares** — not a promise of what will run |
| `corpus` | provenance: `seq`, task id and the corpus case the equation came from |
| `spelling` | the corpus equation that produced that entry |
| `roles` | the role of each operand in the corpus (perambulators, gammas, VDV/VVV) |

On a miss the report says why: the equation is outside the `wick_12cases` corpus, has
no shape-specialised path, and no equivalent spelling reaches a known form. With
`--equivalence true` the miss report also lists the equivalent spelling together with
the input-side tensor adjustment and the output-side result mapping, when one exists.

`kernels` is a **declaration**: dispatch still depends on `out=`, dtype and whether the
native library for the actual array backend is available. For numeric verification use
the test-side entry, which is not shipped by `pip install`:

```bash
python -m pyqcd.testing.einsum.verify_specialized "<eq>"
```

## 2. How to tell which path ran

Both notices print **once per shape per process** and follow the locale (Chinese on
`zh_*`, English otherwise — force it with `PYQCD_EINSUM_LANG=zh|en`).

| Situation | Log line |
|---|---|
| Corpus path, native kernel ran | `[pyqcd.einsum] specialized path: topology:<digest> [match: topology] [kernel: kupl_topology] ...` |
| Corpus path, kernel unavailable | `[pyqcd.einsum] specialized path: topology:<digest> [match: topology] [plan-cache] ...` |
| Outside the corpus | `[pyqcd.einsum] generic path: no specialized shape matched (...)` |

`[match: …]` names the shape family that matched, `[kernel: …]` names the route that
actually executed, and `[plan-cache]` means only the plan was reused. Disable both
notices with `PYQCD_EINSUM_NOTICE=0`; make the specialized one fire on **every** call
with `PYQCD_EINSUM_SPECIALIZE_VERBOSE=1`.

## 3. Coverage by case

| Case | Contractions |
|---|---:|
| 1-介子两点函数 | 2 |
| 10-重子四点函数 | 120 |
| 13-五夸克到三夸克三点函数 | 120 |
| 14-五夸克到三夸克四点函数 | 720 |
| 2-重子两点函数 | 6 |
| 3-五夸克多粒子两点函数 | 120 |
| 4-六夸克多粒子两点函数 | 720 |
| 5-介子三点函数 | 6 |
| 6-重子三点函数 | 24 |
| 7-五夸克多粒子三点函数 | 720 |
| 9-介子四点函数 | 24 |
| 动量指标与流分量指标 | 6056 |

## 4. Coverage by operand count

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

## 5. Regenerating the per-entry list

The per-entry listing (all 8638 equations with their `seq` ids) is generated from the
`wick_12cases` corpus, never typed by hand, so it lives in the repository instead of
this page. Regenerate it with the test-side generator:

```bash
python -m pyqcd.testing._gen_specialized_appendix
```

An earlier release also carried five hand-listed shapes — `gauge-propagator(E4)`,
`gamma-sandwich(E5)`, `sush-2pt(S1)`, `sush-3pt-munu(S2)`, `sush-2pt-mu(S3)`. Those were
leftovers from retired dedicated kernels, were never part of this corpus, and have been
removed: everything above is served by the current topology-specialized kernels.
