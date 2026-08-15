# Changelog — generate_einsum meson 3pt

## 2026-06-06: Meson 3pt interface alignment

### Context
The `generate_einsum` tool's `meson_3pt` type was refactored to use a simplified
4-parameter API `(snk_op, src_op, cur_op, t_sep)`. Multiple files were updated
to ensure consistency across the package.

---

### 1. `utils/generate_einsum/__init__.py`

**Problem:** The old wrapper function `gen_meson_3pt_code(snk, src, current, t_sep)`
used keyword arguments `sink=snk, source=src` that did not match the underlying
codegen's actual parameter names (`snk_op, src_op`). Calling the wrapper would
raise a TypeError.

Additionally, `from .three_pt.codegen_baryon import gen_code` failed because
the function is named `gen_baryon_3pt_code`, not `gen_code`. This broke the
entire `utils.generate_einsum` package import.

**Fix:**
- Deleted the wrapper function entirely. `gen_meson_3pt_code` is now a direct
  re-export of `codegen_meson.gen_meson_3pt_code` with the native 4-parameter
  signature `(snk_op, src_op, cur_op, t_sep="t_sep")`.
- Fixed the baryon import name: `gen_code` → `gen_baryon_3pt_code`.

```python
from .three_pt.codegen_meson import gen_meson_3pt_code
from .three_pt.codegen_baryon import gen_baryon_3pt_code as gen_pyquda_baryon
```

---

### 2. `utils/generate_einsum/three_pt/__init__.py`

**Problem:** Same broken import as above. Blocked the `three_pt` sub-package.

**Fix:**
```python
from .codegen_baryon import gen_baryon_3pt_code as gen_pyquda_baryon
```

---

### 3. `utils/generate_einsum/three_pt/codegen_meson.py`

**Problems:**
1. `_GAMMA_TENSOR_TO_CALL` was missing `"gy"`, `"gz"`, `"gt"`, `"gamma_y"`,
   `"gamma_z"`, `"gamma_t"`. Passing these gamma names to `meson_operator()`
   produced a tensor with an unrecognized name, which `_gamma_call()` could not
   map, resulting in invalid PyQUDA code like `cp.asarray(gx, ...)`.
2. `_GAMMA_DISPLAY` was missing `"gx"`, `"gy"`, `"gz"`, `"gt"` entries.
3. The codegen comment line `"#   sequential = {seq_flavor} ({prop_seq})"`
   was not an f-string, so `{seq_flavor}` and `{prop_seq}` appeared literally
   in the generated code instead of being interpolated.

**Fix:**
```python
_GAMMA_TENSOR_TO_CALL = {
    "G5": "gamma.gamma(15)",
    "g1": "gamma.gamma(1)",
    "gx": "gamma.gamma(1)",
    "gy": "gamma.gamma(2)",
    "gz": "gamma.gamma(4)",
    "gt": "gamma.gamma(8)",
    "gamma_x": "gamma.gamma(1)",
    "gamma_y": "gamma.gamma(2)",
    "gamma_z": "gamma.gamma(4)",
    "gamma_t": "gamma.gamma(8)",
    ...
}

_GAMMA_DISPLAY = {
    ...
    "gx": "g1", "gy": "g2", "gz": "g3", "gt": "g4",
    ...
}

# f-string fix:
f"#   sequential = {seq_flavor} (prop_seq)"
```

---

### 4. `utils/generate_einsum/hadron_operator.py`

**Problem:** The gamma name normalization dict in `gamma()` only handled
`"g5"`, `"g1"`–`"g4"`, `"gtg5"`. Names like `"gx"`, `"gy"`, `"gz"`, `"gt"`
were stored as-is in the Tensor, and no downstream code (codegen's
`_GAMMA_TENSOR_TO_CALL`) recognized them.

**Fix:** Added normalization entries so these names are normalized at creation
time:
```python
def gamma(name, left, right):
    return Tensor("gamma", {
        "g5": "G5", "g1": "g1", "gx": "g1",
        "gy": "g2", "gz": "g3", "gt": "g4",
        "gtg5": "gtg5", "g2": "g2", "g3": "g3", "g4": "g4"
    }.get(name, name), ...)
```

---

### 5. `utils/skill_utils.py`

**Changes:**

**(a) Import path** — `_handle_meson_3pt` now imports from the package entry
point instead of the internal module directly:
```python
from utils.generate_einsum import gen_meson_3pt_code
```

**(b) Schema** — The meson_3pt section of `_generate_einsum_schema()` was
rewritten:

| Action | Fields |
|--------|--------|
| Added | `sink_antiquark`, `current_quark`, `current_antiquark`, `gamma_snk`, `gamma_src`, `tseq` |
| Removed | `src_name`, `snk_name` (no longer used by codegen) |
| Relaxed | `gamma_cur` no longer has a restrictive enum |

Description strings for `current_quark` and `current_antiquark` now
explicitly state the direction convention:
> "IMPORTANT: this is the OPPOSITE of the task description's q̄·Γ·q
> convention — e.g. for c->s transition: current_quark='c',
> current_antiquark='s'."

---

### 6. `skills/pyquda-tool/SKILL.md`

**Problem:** The meson_3pt row in the dispatch table described the interface
via operator constructors (`meson_operator()`, `current_operator()`), not the
actual `tool_args` fields. The LLM could not infer which flat parameters to
pass. No direction convention for the current was mentioned.

**Fix:** Replaced the row with explicit `tool_args` field names, a direction
warning, and accepted gamma value ranges:

> | `meson_3pt` | **tool_args fields**:<br>`src_antiquark`, `src_quark` — source meson flavors (initial state)<br>`sink_antiquark`, `snk_quark` — sink meson flavors (final state)<br>`current_quark`, `current_antiquark` — **⚠ direction**: current = anti(current_antiquark) · Γ · q(current_quark). This is OPPOSITE to the task's q̄·Γ·q convention. E.g. for c→s transition: `current_quark="c"`, `current_antiquark="s"`.<br>`gamma_snk`, `gamma_src`, `gamma_cur` — gamma names. Accepts gamma1-5, gamma_x/y/z/t, gx/gy/gz/gt.<br>`tseq` — time separation | ... |

---

### Remaining issues (meson)

1. `codegen_meson.py`: `prop_seq` variable definition is commented out (line 145).
   Not a runtime bug since it's only referenced in the now-fixed f-string comment,
   but should be cleaned up.
2. `run.py` Executor currently ignores the tool's output and writes its own
   `main.py` from scratch. This is a pipeline behavior issue, not a codegen bug.
3. `skills/pyquda-tool/SKILL.md`: Only the meson_3pt row has been updated with
   flat tool_args. The meson_2pt, baryon_2pt, and baryon_3pt rows still describe
   the old operator-constructor interface — update needed (see proposed sections below).

---

## Proposed: Baryon 3pt alignment (future work)

`codegen_baryon.py` has similar issues to the pre-fix meson codegen. The
following changes are proposed but not yet implemented.

### Rationale

Bring the baryon 3pt interface in line with the meson 3pt pattern:
flat `tool_args` parameters (instead of baryon name lookup), complete
`_GAMMA_PYQUDA` mapping, and updated SKILL.md.

---

### 7. `codegen_baryon.py` — `_GAMMA_PYQUDA` completeness

**Problem:** Only 4 gamma index entries are mapped:
```python
_GAMMA_PYQUDA = {5: "Cg5", 15: "G5", 1: "g1", 7: "gtg5"}
```
Missing `{0: "I4", 2: "g2", 3: "g3", 4: "g4"}`. Unknown gamma indices
fall back to `f"gamma.gamma({idx})"` which is syntactically valid but may
not match the codegen's assumptions about available gamma matrix variables.

**Proposed fix:**
```python
_GAMMA_PYQUDA = {
    0: "I4", 1: "g1", 2: "g2", 3: "g3", 4: "g4",
    5: "Cg5", 7: "gtg5", 15: "G5",
}
```

---

### 8. `utils/skill_utils.py` — `_handle_baryon_3pt` flat parameters

**Problem:** Current interface uses baryon name lookup:
```python
src_name = str(tool_args.get("src_baryon", ""))  # e.g. "proton"
snk_name = str(tool_args.get("snk_baryon", ""))
cur_in   = str(tool_args.get("current_in", ""))
cur_out  = str(tool_args.get("current_out", ""))
cur_gamma = _gmap.get(str(tool_args.get("current_gamma", "gamma1")), "g1")
```

This is inconsistent with the meson_3pt flat-parameter approach. The LLM
cannot specify individual flavors, and the hardcoded `_BARYON_FLAVORS` dict
limits extensibility.

**Proposed change — new flat parameter interface:**

| Old param | New flat param | Description |
|-----------|---------------|-------------|
| `src_baryon` | `src_a`, `src_b`, `src_c` | Source baryon quark flavors |
| `snk_baryon` | `snk_a`, `snk_b`, `snk_c` | Sink baryon quark flavors |
| `current_in` | `current_quark` | Flavor annihilated by the current |
| `current_out` | `current_antiquark` | Flavor created by the current |
| `current_gamma` | `current_gamma` | Gamma matrix (keep as-is) |
| (none) | `projector` | Spin projector: "P_plus", "P_minus" |
| (none) | `diquark_gamma_snk` | Diquark gamma at sink (e.g. "Cg5") |
| (none) | `diquark_gamma_src` | Diquark gamma at source |
| (none) | `c_gamma_snk` | c-quark gamma at sink (e.g. "I4") |
| (none) | `c_gamma_src` | c-quark gamma at source |
| (none) | `tseq` | Time separation |

**Proposed `_handle_baryon_3pt` signature:**
```python
src = baryon_operator(src_a, src_b, src_c, diquark_gamma_src, c_gamma_src)
snk = baryon_operator(snk_a, snk_b, snk_c, diquark_gamma_snk, c_gamma_snk)
cur = current_operator(current_antiquark, current_quark, cur_gamma)
code = gen_pyquda_baryon(snk, src, cur, tseq, projector)
```

---

### 9. `utils/generate_einsum/__init__.py` — baryon wrapper (optional)

If desired, add a `gen_baryon_3pt_code` wrapper to `__init__.py` for
package-level symmetry with meson:
```python
from .three_pt.codegen_baryon import gen_baryon_3pt_code
```
No wrapper function needed since the parameter names already match the
function signature (unlike the meson case where `sink` vs `snk_op` caused
issues).

---

### 10. Schema update for baryon_3pt

Replace the old baryon_3pt Schema section with flat fields:

```python
"src_a":     {"type": "string", "description": "Source baryon quark a flavor."},
"src_b":     {"type": "string", "description": "Source baryon quark b flavor."},
"src_c":     {"type": "string", "description": "Source baryon quark c flavor."},
"snk_a":     {"type": "string", "description": "Sink baryon quark a flavor."},
"snk_b":     {"type": "string", "description": "Sink baryon quark b flavor."},
"snk_c":     {"type": "string", "description": "Sink baryon quark c flavor."},
"current_quark":       {"type": "string", "description": "..."},
"current_antiquark":   {"type": "string", "description": "..."},
"current_gamma":       {"type": "string", "description": "..."},
"projector":           {"type": "string", "description": "P_plus or P_minus."},
"diquark_gamma_snk":   {"type": "string", "description": "Diquark gamma at sink. Default Cg5."},
"diquark_gamma_src":   {"type": "string", "description": "Diquark gamma at source. Default Cg5."},
"c_gamma_snk":         {"type": "string", "description": "c-quark gamma at sink. Default I4."},
"c_gamma_src":         {"type": "string", "description": "c-quark gamma at source. Default I4."},
"tseq":                {"type": "string", "description": "Time separation."},
```

---

### 11. SKILL.md — baryon_3pt row update

Replace the operator-constructor description with flat tool_args fields,
analogous to the meson_3pt fix in section 6.

---

### Implementation order (baryon 3pt)

1. `_GAMMA_PYQUDA` completeness in `codegen_baryon.py` (safe, no API change)
2. Flat parameters in `_handle_baryon_3pt` in `skill_utils.py`
3. Schema update in `_generate_einsum_schema()`
4. SKILL.md dispatch table update
5. (Optional) `__init__.py` wrapper

---

## Proposed: Meson 2pt alignment

### 12. `utils/skill_utils.py` — `meson_2pt` flat parameters

**Problem:** Current interface uses meson name lookup + flavor table:

```python
meson = str(tool_args.get("meson", "pion"))
quark, antiquark = self._MESON_FLAVORS.get(meson, ("u", "d"))
snk = _mk_meson(quark, antiquark, gamma)        # ← order: (quark, antiquark)
```

Issues:
- `_MESON_FLAVORS` stores `(quark, antiquark)`, but `meson_operator(anti_quark_flavor, quark_flavor, ...)` expects `(antiquark, quark)`. The mismatch is hidden for π⁺ (u/d symmetric) but wrong for D⁰ or K⁻.
- LLM cannot specify custom flavor combinations.
- The code already has a TODO comment noting this:
  `### one should directly use str(tool_args.get("quark", "u")) to get the flaovr and gamma`

Code also ignores the gamma_src parameter — both sink and source use the same `gamma_snk`.

**Proposed change — flat parameter interface:**

| Old param | New flat param | Description |
|-----------|---------------|-------------|
| `meson` | `quark` | Quark flavor (second arg of meson_operator) |
| (from table) | `antiquark` | Antiquark flavor (first arg of meson_operator) |
| `gamma_snk` | `gamma_snk` | Gamma at sink (keep as-is) |
| (none) | `gamma_src` | Gamma at source (needed for non-G5 cases) |
| (none) | `point_source` | Bool, keep as-is |

**Proposed `_handle_meson_2pt` signature:**

```python
antiquark = str(tool_args.get("antiquark", "u"))
quark     = str(tool_args.get("quark", "d"))
gamma_snk = _gmap.get(str(tool_args.get("gamma_snk", "gamma5")), "g5")
gamma_src = _gmap.get(str(tool_args.get("gamma_src", "gamma5")), "g5")

snk = _mk_meson(antiquark, quark, gamma_snk)
src = _mk_meson(antiquark, quark, gamma_src)
```

---

### 13. Schema update for meson_2pt

Replace the current `meson` enum field with flat flavor fields:

```python
"antiquark": {"type": "string", "description": "Antiquark flavor (for meson_2pt)."},
"quark":     {"type": "string", "description": "Quark flavor (for meson_2pt)."},
"gamma_snk": {"type": "string", "description": "Gamma at sink (for meson_2pt). Default gamma5."},
"gamma_src": {"type": "string", "description": "Gamma at source (for meson_2pt). Default gamma5. Gets Dirac adjoint."},
"point_source": {"type": "boolean"},
```

Remove the `meson` enum field (currently `["pion", "rho"]`).

---

### 14. SKILL.md — meson_2pt row update

Replace:

> | `meson_2pt` | `(antiquark,quark,gamma)_snk`, `(antiquark,quark,gamma)_src` |

With flat parameter names, e.g.:

> | `meson_2pt` | **tool_args fields**: `antiquark`, `quark` — meson flavor content. `gamma_snk`, `gamma_src` — gamma matrices. `point_source` — optional. |

---

## Proposed: Baryon 2pt alignment

### 15. `utils/skill_utils.py` — `baryon_2pt` flat parameters

**Problem:** Current interface uses baryon name lookup:

```python
baryon = str(tool_args.get("baryon", "proton"))
a, b, c = flavors[baryon]
```

Same issues as meson_2pt: LLM cannot specify custom flavor combinations,
and diquark gamma / c-quark gamma / projector are all hardcoded or
name-dependent.

**Proposed change — flat parameter interface:**

```python
quark_a = str(tool_args.get("quark_a", "u"))
quark_b = str(tool_args.get("quark_b", "d"))
quark_c = str(tool_args.get("quark_c", "u"))
projector = str(tool_args.get("projector", "P_plus"))
diquark_gamma = str(tool_args.get("diquark_gamma", "Cg5"))
c_gamma = str(tool_args.get("c_gamma", "I4"))

snk = _mk_baryon(quark_a, quark_b, quark_c, diquark_gamma, c_gamma)
src = _mk_baryon(quark_a, quark_b, quark_c, diquark_gamma, c_gamma)
```

---

### 16. Schema update for baryon_2pt

Replace the `baryon` enum field with flat flavor fields:

```python
"quark_a":       {"type": "string", "description": "Baryon quark a flavor (for baryon_2pt)."},
"quark_b":       {"type": "string", "description": "Baryon quark b flavor (for baryon_2pt)."},
"quark_c":       {"type": "string", "description": "Baryon quark c flavor (for baryon_2pt)."},
"projector":     {"type": "string", "description": "Spin projector: P_plus or P_minus."},
"diquark_gamma": {"type": "string", "description": "Diquark gamma matrix. Default Cg5."},
"c_gamma":       {"type": "string", "description": "c-quark gamma matrix. Default I4."},
```

---

### 17. SKILL.md — baryon_2pt row update

Replace the operator-constructor description with flat tool_args fields.

---

## Summary: All proposed changes

| type | current interface | proposed interface | status |
|------|------------------|-------------------|--------|
| `meson_3pt` | flat params | flat params | ✅ **done** (this changelog) |
| `meson_2pt` | name lookup (`meson="pion"`) | flat params `antiquark`, `quark` | ⬜ proposed |
| `baryon_2pt` | name lookup (`baryon="proton"`) | flat params `quark_a/b/c` | ⬜ proposed |
| `baryon_3pt` | name lookup (`src_baryon="proton"`) | flat params `src_a/b/c`, `snk_a/b/c` | ⬜ proposed |

### Implementation order (full)

| step | what | files |
|------|------|-------|
| 1 | meson_2pt flat params + schema | `skill_utils.py` |
| 2 | meson_2pt SKILL.md update | `SKILL.md` |
| 3 | baryon_2pt flat params + schema | `skill_utils.py` |
| 4 | baryon_2pt SKILL.md update | `SKILL.md` |
| 5 | baryon_3pt flat params + schema | `skill_utils.py` |
| 6 | baryon_3pt `_GAMMA_PYQUDA` completeness | `codegen_baryon.py` |
| 7 | baryon_3pt SKILL.md update | `SKILL.md` |
| 8 | baryon 3pt `__init__.py` (optional) | `__init__.py` |
