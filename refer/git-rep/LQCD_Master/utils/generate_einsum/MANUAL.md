# generate_einsum Manual

Hadronic correlator contraction engine. Given hadron operators (meson/baryon), produces PyQUDA `contract()` code for 2-point and 3-point functions.

---

## Directory Layout

```
generate_einsum/
├── _wick_translate.py     Shared: Tensor → wicklib → PyQUDA name conversion
├── hadron_operator.py     Operator definitions (meson, baryon, current)
├── wicklib/               Wick contraction engine (MIT-licensed)
│
├── two_pt/
│   ├── contract.py        Wick contraction: wick_contract_2pt()
│   ├── codegen.py         PyQUDA formatter: pyquda_format_contract()
│   └── demo_2pt.py        20+ demo cases (meson + baryon)
│
├── three_pt/
│   ├── contract.py        Wick contraction: contract_baryon_3pt(), contract_meson_3pt()
│   ├── codegen.py         Code generation: gen_seq_source_code(), gen_final_contract_code()
│   └── demo_3pt.py        36 demo cases (21 baryon + 15 meson)
│
└── MANUAL.md              This file
```

### `_wick_translate.py` — shared conversion layer

Three functions used by both 2pt and 3pt:

| Function | Role |
|----------|------|
| `_to_wicklib(op, location)` | Tensor[] → wicklib `Block` (QuarkBilinear / Diquark×Quark) |
| `_rename_op(op_str)` | wicklib operand string → PyQUDA variable name |
| `_simplify_gammas(expr)` | Cancel adjacent identical gammas (γ² = I) |

---

## 2-Point Functions

### Pipeline (5 steps)

```
Step 1:  Define operator
           meson_operator(anti, quark, gamma)
           baryon_operator(a, b, c, [diquark_gamma])

Step 2:  Tensor[] → wicklib Block
           _to_wicklib(op, location)
           → QuarkBilinear (meson) or Diquark×Quark (baryon)

Step 3:  Wick contraction + gamma simplification
           ⟨sink · source†⟩  →  Correlator.simplify()

Step 4:  Extract einsum + rename
           term.to_einsum()  →  (factor, subs, operands)
           _rename_op()  →  PyQUDA names (prop_l, G5, ...)

Step 5:  Format as PyQUDA contract() call
           pyquda_format_contract(term)
           → "contract('wtzyxCBba, wtzyxCBba -> t', prop_l.conj(), prop_l)"
```

### Entry Point

```python
from utils.generate_einsum.two_pt.contract import wick_contract_2pt
from utils.generate_einsum.two_pt.codegen import pyquda_format_contract

# Define sink and source operators
sink = meson_operator("u", "d", "g5")      # ū·γ₅·d  (π⁺)
source = meson_operator("u", "d", "g5")    # same at t=0

# Contract: Note: One should add the projector for the baryon case.
# !!! This should be revised, consult wicklib/operator.py
result = wick_contract_2pt(sink, source)   # → ContractionResult

# Format code
code = pyquda_format_contract(result.terms[0])
```

### Naming Convention

| wicklib operand | `_rename_op()` output | Meaning |
|-----------------|----------------------|---------|
| `gamma(15)` | `G5` | γ₅ |
| `gamma(1)` | `g1` | γ₁ (vector) |
| `gamma(4)` | `g3` | γ₃ |
| `gamma(5)` | `Cg5` | C·γ₅ (diquark, Jᴾ=1/2⁺) |
| `gamma(11)` | `Cg1` | C·γ₁ (diquark, Jᴾ=3/2⁺) |
| `gamma(7)` | `gtg5` | γ₄γ₅ (nonlocal current) |
| `gamma(10)` | `Cmat` | Charge conjugation C |
| `gamma(0)` | `I4` | Identity |
| `propag_u_x_y` → `prop_l` (forward) | `prop_l` | Light propagator |
| `propag_u_y_x` → backward | `G5 @ prop_l.dag() @ G5` | Gamma5-hermiticity wrapped |
| `propag_s_x_y` | `prop_s` | Strange propagator |
| `propag_c_x_y` | `prop_c` | Charm propagator |
| `epsilon` | `epsilon` | Color epsilon tensor |
| `projector` | `Tmat` | Spin projector (1+γ₄)/2 |

### Example Output

**Pion (ū·γ₅·d, distinct flavors → 1 term):**
```
contract('wtzyxCBba, wtzyxCBba -> t', prop_l.conj(), prop_l)
```

**Eta_s (s̄·γ₅·s, identical flavors → 2 terms):**
```
-contract('wtzyxBAaa, wtzyxBAaa -> t', prop_s, prop_s)         # hairpin
contract('wtzyxCBba, wtzyxCBba -> t', prop_s.conj(), prop_s)   # standard
```

**Proton (uud, 2 topologies):**
```
-contract('wtzyxBCaa, wtzyxILjk, wtzyxHMkl -> t', ...)
+contract('wtzyxAIin, wtzyxBHjm, wtzyxGDlk -> t', ...)
```

---

## 3-Point Functions

### Pipeline

```
Step 1:  Define sink, source, and current operators
           baryon_operator(...) / meson_operator(...)
           current_operator(f_out, f_in, gamma)

Step 2:  Topology enumeration + Wick contraction
           contract_baryon_3pt(sink, source, current)
           contract_meson_3pt(sink, source, current)
           → dict with sink_terms, current_gamma, n_topologies

Step 3:  Generate sequential source code
           gen_seq_source_code(src_name, snk_name, result)
           → Python code block (epsilon, gamma matrices, sink block)

Step 4:  Generate final contraction code
           gen_final_contract_code(result)
           → Python code block (G5-dagger + trace)
```

### Entry Point

```python
from utils.generate_einsum.three_pt.contract import contract_baryon_3pt
from utils.generate_einsum.three_pt.codegen import (
    gen_seq_source_code, gen_final_contract_code
)

# Define operators
p = baryon_operator("u", "d", "u")          # proton
cur = current_operator("u", "u", "g1")      # vector current ū·γ₁·u

# Contract
result = contract_baryon_3pt(p, p, cur)     # → dict

# Generate code
seq_code = gen_seq_source_code("p", "p", result)
final_code = gen_final_contract_code(result)
```

### Meson 3pt

**Wick rule:** `src_anti` must equal `snk_anti` (spectator anti-quark must be the same across sink and source). Otherwise wicklib raises `unmatched quark found`.

Pattern: `M₁(a, b) → M₂(a, c)` with current `b → c`.

```python
from hadron_operator import current_operator

D_K = contract_meson_3pt(
    meson_operator("u", "c", "g5"),   # D⁰ = c̄·γ₅·u (sink)
    meson_operator("u", "s", "g5"),   # K⁻ = s̄·γ₅·u (source)
    current_operator("c", "s", "g1"), # J = c̄·γ₁·s
)
```

**Meson sink block:** Single spectator propagator (no epsilon, no Tmat).

Generated sink block einsum:
```
B = contract('wtzyxjiba -> wtzyx', phase * prop_a.conj().data)
```

### Baryon 3pt

**Wick rule:** After the current changes flavors, the two remaining quarks from the source must match the two remaining quarks at the sink. The more identical flavors, the more topologies.

| Process | Topologies | Note |
|---------|:----------:|------|
| Ω⁻ → Ω⁻ | **18** | Three identical sss→sss |
| Ω⁻ → Ξ⁰, Ωccc → Ωcc | 6 | |
| p → p, Σ⁺ → Σ⁺, Ξ⁰ → Ξ⁰ | 4 | Two identical u |
| Ωcc → Ωc | 4 | |
| Λ → p, Λb → p, Ξc → Ξ | 2 | |
| Λ → Λ, Λc → Λ, Λb → Λc | 1 | All distinct |

**Baryon sink block:** 8-operand einsum with `wtzyx` prefix:

```
einsum = wtzyx, ijk, lmn, AB, GH, ID, wtzyx{prop_a}, wtzyx{prop_b} -> wtzyx{output}
```

Where:
- `wtzyx` — phase (sequential source)
- `ijk, lmn` — epsilon tensors (sink, source)
- `AB, GH` — Cg5 insertion matrices
- `ID` — Tmat (spin projector P_plus)

Generated code:
```python
Tmat = cp.asarray((I4 + gamma.gamma(8)) * 0.5, dtype=cp.complex128)  # P_plus
Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)
Cg5 = Cmat @ G5

B.data = (
    + contract('wtzyx, ijk, lmn, AB, GH, ID, wtzyxAGil, wtzyxBHjm -> wtzyxIDnk',
               phase, epsilon, epsilon, Cg5, Cg5, Tmat, prop_a.data, prop_b.data),
    ...
)
```

### Sequential Source + G5-Dagger Pattern

```
# G5-dagger on sink block
B.data = contract('AB, wtzyxCBji, CD -> wtzyxADij', G5, B.data.conj(), G5)

# Sequential source at t_sink
src_seq = source.sequential12(B, t_sink)

# [inversion happens here]

# G5-dagger on sequential propagator
tmp_prop = core.LatticePropagator(latt_info)
tmp_prop.data = contract('AB, wtzyxCBji, CD -> wtzyxADij',
                         G5, prop_seq.data.conj(), G5)

# Final contraction: Tr[G_seq_dag @ Gamma_cur @ S_fwd]
three_pt_site = contract('wtzyxijba, jk, wtzyxkiab -> wtzyx',
                         tmp_prop.data, Gamma_cur, fwd_prop.data)
```

### Codegen Output Structure

`gen_seq_source_code()` outputs:
1. Gamma matrix definitions (I4, G5, Gamma_snk, Gamma_cur)
2. Epsilon tensor (GPU, baryon only)
3. Cmat, Cg5, Tmat (baryon only)
4. Topology summary comments
5. Sink block `B = core.LatticePropagator(latt_info)` with einsum
6. G5-dagger on B
7. Sequential source line

`gen_final_contract_code()` outputs:
1. G5-dagger on sequential propagator
2. Three-point site contraction
3. Local average over spatial sites
4. MPI gather + save

---

## Data Structures

### ContractionTerm (2pt only)

```python
@dataclass
class ContractionTerm:
    coefficient: complex    # ±1, etc.
    einsum_subs: str        # contraction indices, e.g. "AB,CD,BCab,DAba"
    operands: List[str]     # operands, e.g. ['G5', 'G5', 'G5 @ prop_l.dag() @ G5', 'prop_l']
```

### ContractionResult (2pt only)

```python
@dataclass
class ContractionResult:
    sink_desc: str
    source_desc: str
    n_raw_terms: int
    terms: List[ContractionTerm]
    has_epsilon: bool           # True for baryons
    projector: Optional[str]    # "P_plus" / None
```

### ContractionTopology (3pt only)

```python
@dataclass
class ContractionTopology:
    name: str
    fermion_sign: int
    connections: Dict[str, Tuple[str, bool]]  # src_slot → (snk_slot, is_current)
    description: str
```

### 3pt Sink Block Return Dict

```python
{
    "sink_terms": [
        {
            "einsum": str,           # 8-operand einsum string
            "sign": int,             # fermion sign (±1)
            "n_props": 2,            # = 2 for baryon, 1 for meson
            "var_a": "prop_l",       # propagator variable name
            "var_b": "prop_s",       # propagator variable name
            "flavor_a": "u",
            "flavor_b": "s",
            "description": str,      # human-readable topology
        },
        ...
    ],
    "fwd_var": "prop_l",             # forward propagator variable
    "fwd_flavor": "u",               # flavor at the current
    "current_gamma": "g1",           # gamma matrix at the current
    "n_topologies": int,
}
```

---

## Gamma Matrix Conventions

wicklib uses Clifford algebra with 4-bit mask indices.

### Base Matrices

| Name | Index | Binary | Meaning |
|------|-------|--------|---------|
| `I4` | 0 | `0000` | Identity |
| `g1` | 1 | `0001` | γ₁ |
| `g2` | 2 | `0010` | γ₂ |
| `g3` | 4 | `0100` | γ₃ |
| `g4` | 8 | `1000` | γ₄ |

### Composite Matrices

| Name | Index | Binary | Meaning |
|------|-------|--------|---------|
| `Cg5` | 5 | `0101` | C · γ₅ (diquark, Jᴾ=1/2⁺) |
| `gtg5` | 7 | `0111` | γ₄ · γ₅ (nonlocal current) |
| `Cmat` | 10 | `1010` | Charge conjugation C |
| `Cg1` | 11 | `1011` | C · γ₁ (diquark, Jᴾ=3/2⁺) |
| `G5` | 15 | `1111` | γ₅ |

### Clifford Algebra

```
Gamma(i) @ Gamma(j) = Gamma(i ^ j, factor = popsign[i & j])
```

Indices XOR; sign from parity of bitwise AND.

---

## FAQ

### Why do meson 2pt terms have 2 operands or 4?

2 operands: gamma matrices canceled (γ₅·γ₅ → -I₄, simplified away).
4 operands: gamma matrices survive (g₁·g₁ ≠ ±I₄, retained as separate operands).

### Why are baryon 3pt propagators all backward (`G5 @ ... @ G5`)?

Because of the P_plus projector. It reverses some spin index contractions, pushing all propagators through the gamma5-hermiticity wrapper.

### What is `Tmat`?

`Tmat = (I₄ + γ₄) / 2 = P_plus`, the spin projector used in the baryon 3pt sequential source. Generated as a `cp.asarray` in the codegen output.

### Can I add a new 3pt decay process?

Yes. Check the flavor matching rules:

**Meson:** `src_anti` must equal `snk_anti`. The spectator anti-quark never changes.

**Baryon:** After current insertion, the two unchanged quarks must match between source and sink. More identical quarks → more Wick topologies.

### Why does my 3pt contraction return 0 topologies?

Common causes:
- Flavor mismatch at spectator (meson 3pt)
- Current flavor doesn't connect the two remaining quarks (baryon 3pt)
- wicklib `unmatched quark found` error → wrong operator ordering

### 2pt vs 3pt: entry points

| | 2pt | 3pt |
|--|-----|-----|
| Contract | `wick_contract_2pt()` | `contract_baryon_3pt()` / `contract_meson_3pt()` |
| Codegen | `pyquda_format_contract()` | `gen_seq_source_code()` + `gen_final_contract_code()` |
| Topologies | Wick pairing only | Topology enumeration + Wick pairing |
| Meson | γ₅/wrapped contract() | Sequential source + 3pt trace |
| Baryon | 10-operand einsum | 8-operand sink block + Tmat + Cg5 |

### Demo cases

Run `python3 demo_2pt.py` → 20+ meson + baryon 2pt cases with code output.
Run `python3 demo_3pt.py` → 36 cases (21 baryon + 15 meson 3pt) with codegen output.
