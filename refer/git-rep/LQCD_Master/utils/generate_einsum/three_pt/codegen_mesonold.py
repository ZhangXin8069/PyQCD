"""codegen_meson — Meson 3pt PyQUDA code generation.

Entry point:
  gen_meson_3pt_code(sink, source, cur, ...) — from operator objects

Output:
  Gamma_snk / Gamma_src / Gamma_snk_bar / Gamma_src_bar
  Phase tensor + clean sink block einsum
  G5-dagger → sequential solve → final contraction
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from utils.generate_einsum._wick_translate import _to_wicklib
except ImportError:
    from _wick_translate import _to_wicklib
from hadron_operator import meson_operator, current_operator
try:
    from utils.generate_einsum.wicklib.correlator import Correlator
except ImportError:
    from wicklib.correlator import Correlator
LIGHT = frozenset({"u","d"})
GAMMA_PYQ = {15:"G5", 5:"Cg5", 1:"g1", 7:"gtg5"}
# Gamma tensor name → PyQUDA gamma() call
_GAMMA_TENSOR_TO_CALL = {
    'G5':   'gamma.gamma(15)',
    'g1':   'gamma.gamma(1)',
    'g2':   'gamma.gamma(2)',
    'g3':   'gamma.gamma(4)',
    'g4':   'gamma.gamma(8)',
    'gtg5': 'gamma.gamma(7)',
    'I4':   'gamma.gamma(0)',
}
# Gamma tensor name → short display name for comments
_GAMMA_DISPLAY = {
    'G5':   'G5',
    'g1':   'g1',
    'g2':   'g2',
    'g3':   'g3',
    'g4':   'g4',
    'gtg5': 'gtg5',
    'I4':   'I4',
}
def var(f):
    return "prop_l" if f in LIGHT else f"prop_{f}"
def rename_label(l, r):
    return "".join(r.get(c,c) for c in l)
def op_flavor(op):
    prts = op.split("_")
    return prts[1] if len(prts) > 1 else "l"
def op_to_pyq(op):
    if op.startswith("gamma("):
        idx = int(op.replace("gamma(","").replace(")",""))
        return GAMMA_PYQ.get(idx, f"gamma.gamma({idx})")
    if op.startswith("propag_"):
        return f"{var(op_flavor(op))}.data"
    return op
def _gamma_call(name: str) -> str:
    """Convert gamma tensor name to PyQUDA gamma() call string."""
    if name in _GAMMA_TENSOR_TO_CALL:
        return _GAMMA_TENSOR_TO_CALL[name]
    # Composite gamma: parse digits, e.g. 'g1g5' → gamma.gamma(1) @ gamma.gamma(15)
    parts = []
    i = 0
    while i < len(name):
        if name[i] == 'g' and i + 1 < len(name) and name[i+1].isdigit():
            parts.append(f"gamma.gamma({name[i+1]})")
            i += 2
        elif name[i] == 'G' and i + 1 < len(name) and name[i+1] == '5':
            parts.append("gamma.gamma(15)")
            i += 2
        else:
            i += 1
    if len(parts) >= 1:
        return " @ ".join(parts)
    return name  # fallback
def _gamma_display(name: str) -> str:
    """Get short display name for a gamma tensor name."""
    return _GAMMA_DISPLAY.get(name, name)
def _get_tensors(op):
    """Extract tensor list from an operator (handles tuple wrapper)."""
    obj = op[0] if isinstance(op, tuple) else op
    return list(obj.tensors)
def gen_meson_3pt_code(sink, source, current, src_name="src", snk_name="snk",
                         t_sep="t_sep", phase="ones_phase", out_path='"output.npy"'):
    # ════════════════════════════════════════════════════════════════
    #  ── 0. Extract flavor/std info from operator tensors ──
    # ════════════════════════════════════════════════════════════════
    snk_tensors = _get_tensors(sink)
    src_tensors = _get_tensors(source)
    cur_tensors = _get_tensors(current)
    snk_gamma_name = next(t.name for t in snk_tensors if t.type == "gamma")
    src_gamma_name = next(t.name for t in src_tensors if t.type == "gamma")
    cur_gamma_name = next(t.name for t in cur_tensors if t.type == "gamma")
    # spectator = sink antiquark flavor (exists in both sink and source)
    spectator_flavor = next(t.flavor for t in snk_tensors if t.type == "antiquark")
    # forward = current's incoming quark flavor (the heavy flavor being probed)
    forward_flavor = next(t.flavor for t in cur_tensors if t.type == "quark")
    # sequential = current's outgoing antiquark flavor (the lighter flavor created)
    seq_flavor = next(t.flavor for t in cur_tensors if t.type == "antiquark")
    prop_spec = var(spectator_flavor)
    prop_fwd = var(forward_flavor)
    prop_seq = f"prop_{seq_flavor}" if seq_flavor not in LIGHT else f"prop_{seq_flavor}"
    # ════════════════════════════════════════════════════════════════
    #  ── 1. Build wicklib correlator ──
    # ════════════════════════════════════════════════════════════════
    sr = sink[0] if isinstance(sink, tuple) else sink
    sr2 = source[0] if isinstance(source, tuple) else source
    cr = current[0] if isinstance(current, tuple) else current
    w_snk, _ = _to_wicklib(sr, "x")
    w_src, _ = _to_wicklib(sr2, "y")
    w_cur, _ = _to_wicklib(cr, "z")
    corr = Correlator(w_snk * w_cur * w_src.adjoint())
    corr.simplify(degenerate=False)
    # ════════════════════════════════════════════════════════════════
    #  ── 2. Process each term via to_einsum() ──
    #      (kept for verification / raw output)
    # ════════════════════════════════════════════════════════════════
    terms = []
    forward_var = None
    gamma_idx = 1
    gamma_name = "g1"
    for term in corr.terms:
        fac, einsum, ops = term.to_einsum()
        einsum = einsum.replace("...", "wtzyx")
        groups = [g.strip() for g in einsum.split("->")[0].split(",")]
        # Identify P_forward (_z_y) and Gamma_cur (operand 2, index 1)
        fwd_i = fwd_letters = None
        for i, op in enumerate(ops):
            if "_z_y" in op or "_y_z" in op:
                fwd_i = i
                fwd_letters = groups[fwd_i].replace("wtzyx", "")
                break
        gamma_i = 1  # meson: operand 2
        if fwd_letters is None:
            continue
        gamma_letters = groups[gamma_i]
        # ── Rename P_forward and Gamma_cur letters ──
        rename = {}
        pi = pj = gi = 0
        for ch in fwd_letters:
            if ch.isupper():
                rename[ch] = ["J","K"][pi]; pi += 1
            else:
                rename[ch] = ["j","k"][pj]; pj += 1
        for ch in gamma_letters:
            if ch in rename: continue
            rename[ch] = ["L","M"][gi]; gi += 1
        renamed_groups = [rename_label(g, rename) for g in groups]
        # ── Remove Gamma_cur, P_forward, P_after ──
        remove = {gamma_i, fwd_i}
        for i, op in enumerate(ops):
            if "_x_z" in op:
                remove.add(i)
        sink_groups = [renamed_groups[i] for i in range(len(groups)) if i not in remove]
        sink_ops = [ops[i] for i in range(len(ops)) if i not in remove]
        # Filter: skip if any remaining propagator has same-coordinate suffix
        skip = False
        for op in sink_ops:
            if op.startswith("propag_"):
                parts = op.split("_")
                if len(parts) >= 4 and parts[-3] == parts[-2]:
                    skip = True
                    break
        if skip:
            continue
        # ── Free indices → output ──
        letters = {}
        for g in sink_groups:
            for ch in g.replace("wtzyx", ""):
                letters[ch] = letters.get(ch, 0) + 1
        free = [c for c, cnt in letters.items() if cnt == 1]
        spin_free = "".join(c for c in free if c.isupper())
        col_free = "".join(c for c in free if c.islower())
        out = f"wtzyx{spin_free}{col_free}"
        pyq_ops = [op_to_pyq(o) for o in sink_ops]
        terms.append((sink_groups, pyq_ops, fac, out))
        # Forward info from first term
        if forward_var is None and fwd_i is not None:
            forward_var = var(op_flavor(ops[fwd_i]))
        if forward_var is None:
            forward_var = "prop_l"
        if gamma_idx == 1:
            g = ops[gamma_i]
            if g.startswith("gamma("):
                gamma_idx = int(g.replace("gamma(","").replace(")",""))
                gamma_name = GAMMA_PYQ.get(gamma_idx, f"gamma.gamma({gamma_idx})")
    # Filter: only terms with 4 free indices
    valid_terms = [(sg, po, fac, out) for sg, po, fac, out in terms if len(out) - 5 == 4]
    if not valid_terms:
        return "/* No valid terms */"
    n = len(valid_terms)
    # ════════════════════════════════════════════════════════════════
    #  ── 3. Build clean PyQUDA code ──
    #      (codegen_sep style: Gamma_snk/Gamma_src + phase + clean einsum)
    # ════════════════════════════════════════════════════════════════
    # Raw to_einsum info kept in comments for verification
    raw_comment_lines = []
    for i, (sg, po, fac, out) in enumerate(valid_terms):
        sig = "+" if fac >= 0 else "-"
        raw_einsum = f"wtzyx,{','.join(sg)},{out}->{out}"
        raw_comment_lines.append(f"    # raw to_einsum #{i}: {sig} {raw_einsum}")
        raw_comment_lines.append(f"    #   operands: {', '.join(po)}")
    lines = [
        "# ═══════════════════════════════════════════════════════",
        f"# Meson 3pt: {src_name} -> {snk_name}",
        f"#   spectator  = {spectator_flavor}  ({prop_spec})",
        f"#   forward    = {forward_flavor}  ({prop_fwd})",
        f"#   sequential = {seq_flavor} (prop_seq)",
        f"#   sink Gamma = {_gamma_display(snk_gamma_name)}",
        f"#   src Gamma  = {_gamma_display(src_gamma_name)}",
        f"#   cur Gamma  = {_gamma_display(cur_gamma_name)}",
        "# ═══════════════════════════════════════════════════════",
        "",
        "# ------------------------------------------------------------------",
        "#  Gamma matrices",
        "# ------------------------------------------------------------------",
        "I4 = cp.eye(4, dtype=cp.complex128)",
        "G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)",
        "### input your Gamma_snk, Gamma_src, Gamma_cur. Here we call them using the inputs",
        f"Gamma_snk = cp.asarray({_gamma_call(snk_gamma_name)},",
        " dtype=cp.complex128)",
        f"Gamma_src = cp.asarray({_gamma_call(src_gamma_name)},",
        " dtype=cp.complex128)",
        f"Gamma_cur = cp.asarray({_gamma_call(cur_gamma_name)},",
        " dtype=cp.complex128)",
        "",
        "# ------------------------------------------------------------------",
        "#  G5-conjugate gammas:  Γ̄ = γ₅ . Γ . γ₅",
        "#  These appear in the sink block;.",
        "# ------------------------------------------------------------------",
        "Gamma_snk_bar = G5 @ Gamma_snk @ G5",
        "Gamma_src_bar = G5 @ Gamma_src @ G5",
        "",
        "# ------------------------------------------------------------------",
        "#  Step 1 -- Sink block",
        "#  B(x) = phase * Γ̄_snk . S_spectator . Γ̄_src",
        "#  No G5-dagger or conj needed — this is a pure γ₅ sandwich.",
        "# ------------------------------------------------------------------",
        f"# spectator = {spectator_flavor}  ({prop_spec})",
    ]
    # Add raw to_einsum comments for verification
    lines.append("#  Raw to_einsum decomposition (for verification):")
    lines.extend(raw_comment_lines)
    lines.append("")
    # Clean sink block with phase tensor
    lines.append("B = core.LatticePropagator(latt_info)")
    lines.append("B.data = contract(")
    lines.append("    'wtzyx, AB, wtzyxBCab, CD -> wtzyxADab',")
    lines.append(f"    {phase}, Gamma_snk_bar, {prop_spec}.data, Gamma_src_bar)")
    lines.append("")
    lines.append(f"# (equivalently: sum over {n} topology/ies from Wick contraction)")
    lines.append("")
    # ── 4. G5-dagger → sequential solve → final contraction ──
    lines.append("# ------------------------------------------------------------------")
    lines.append("#  Step 2 -- Sequential source + solve")
    lines.append("# ------------------------------------------------------------------")
    lines.append("# Sequential source at t_sink")
    lines.append(f"src_seq = source.sequential12(B, {t_sep})")
    lines.append("")
    lines.append("# Sequential inversion: insert your CG / BiCGstab solver here")
    lines.append("# After inversion, prop_seq = G_seq(x,0). Example:")
    lines.append(f"dirac_{seq_flavor}.loadGauge(gauge_stout)")
    lines.append(f"prop_seq = core.invertPropagator(dirac_{seq_flavor}, src_seq) ")
    lines.append("")
    lines.append("# ------------------------------------------------------------------")
    lines.append("#  Step 3 -- Final contraction")
    lines.append("#  C_3pt(t) = Tr[ γ₅·G†_seq·γ₅ · Γ_cur · S_fwd ]")
    lines.append("#  The outer G5-dagger (γ₅·G†_seq·γ₅) gives back G(0,x).")
    lines.append("# ------------------------------------------------------------------")
    lines.append("")
    lines.append("# 3a. Outer G5-dagger: tmp = γ₅ · S_seq† · γ₅ = G(0,x)")
    lines.append("tmp_prop = core.LatticePropagator(latt_info)")
    lines.append("tmp_prop.data = contract(")
    lines.append("    'AB, wtzyxCBji, CD -> wtzyxADij',")
    lines.append("    G5, prop_seq.data.conj(), G5)")
    lines.append("")
    lines.append(f"# 3b. Trace contraction: Tr[ G(0,x) · Γ_cur · S_fwd ]")
    lines.append(f"# forward = {forward_flavor}  ({prop_fwd})")
    lines.append("three_pt_local = contract(")
    lines.append("    'wtzyxijba, jk, wtzyxkiab -> t',")
    lines.append(f"    tmp_prop.data, Gamma_cur, {prop_fwd}.data)")
    lines.append("")
    lines.append("# 3c. MPI gather")
    lines.append("C3_t = core.gatherLattice(")
    lines.append("     three_pt_local.get(), [0, -1, -1, -1])")
    lines.append("")
    return "\n".join(lines)
# ═══════════════════════════════════════════════════════════════════════
