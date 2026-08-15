#!/usr/bin/env python3
"""baryon_3pt_gen.py — Generate PyQUDA code from to_einsum() output.

Sink block = operands 1-10 minus Gamma_cur (operand 3) minus P_forward (_z_y/_y_z) minus P_after (_x_z).
Remaining = structure + 2 spectators → exactly 4 free indices for physical terms.
Non-physical terms (unpaired current) get filtered automatically.
P_after is removed → 4 free indices guaranteed for physical terms.
"""
import sys
from pathlib import Path
_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from hadron_operator import baryon_operator, current_operator
try:
    from utils.generate_einsum._wick_translate import _to_wicklib
except ImportError:
    from _wick_translate import _to_wicklib
try:
    from utils.generate_einsum.wicklib.correlator import Correlator
    from utils.generate_einsum.wicklib.operator import SpinProjector
except ImportError:
    from wicklib.correlator import Correlator
    from wicklib.operator import SpinProjector

_GAMMA_PYQUDA = {0: "I4", 1: "g1", 2: "g2", 3: "g3", 4: "g4", 5: "Cg5", 7: "gtg5", 15: "G5"}
_LIGHT = frozenset({"u", "d"})


def var(f):
    return "prop_l" if f in _LIGHT else f"prop_{f}"


def op_role(op):
    if "_z_y" in op or "_y_z" in op: return "forward"
    if "_x_z" in op: return "after"
    if "_x_y" in op or "_y_x" in op: return "spectator"
    return "struct"


def op_flavor(op):
    parts = op.split("_")
    return parts[1] if len(parts) > 1 else "l"


def op_to_pyq(op):
    if op.startswith("gamma("):
        idx = int(op.replace("gamma(", "").replace(")", ""))
        return _GAMMA_PYQUDA.get(idx, f"gamma.gamma({idx})")
    if op == "epsilon": return "eps"
    if op == "projector": return "Tmat"
    if op.startswith("propag_"): return f"{var(op_flavor(op))}.data"
    return op


def gamma_cur_name(ops):
    for op in ops:
        if op.startswith("gamma("):
            idx = int(op.replace("gamma(", "").replace(")", ""))
            return _GAMMA_PYQUDA.get(idx, f"gamma.gamma({idx})")
    return "g1"


def gamma_cur_idx_name(ops):
    # Gamma_cur is always the third operand (index 2)
    if len(ops) > 2 and ops[2].startswith("gamma("):
        idx = int(ops[2].replace("gamma(", "").replace(")", ""))
        return idx, _GAMMA_PYQUDA.get(idx)
    return 1, "g1"


def forward_operand(ops):
    for i, op in enumerate(ops):
        if op_role(op) == "forward":
            return i, op
    return None, None

def unused_letters(einsum):
    groups = einsum.split("->")[0].split(",")
    used = set()
    for g in groups:
        for ch in g:
            used.add(ch)
    unused_spin = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in used]
    unused_col  = [c for c in "abcdefghijklmnopqrstuvwxyz" if c not in used]
    return unused_spin, unused_col

def find_replace_letters(groups, ops):
    fwd_i  = [i for i, op in enumerate(ops) if "_z_y" in op or "_y_z" in op][0]
    back_i = [i for i, op in enumerate(ops) if "_x_z" in op][0]
    fwd_l  = groups[fwd_i].replace("wtzyx", "")
    back_l = groups[back_i].replace("wtzyx", "")
    return list(dict.fromkeys(fwd_l + back_l))

#def find_replace_letters(groups, ops):
#    fwd_i  = [i for i, op in enumerate(ops) if "_z_y" in op or "_y_z" in op][0]
#    back_i = [i for i, op in enumerate(ops) if "_x_z" in op][0]
##    gam_i  = [i for i, op in enumerate(ops) if "gamma(" in op and "Gamma_cur" in str(op)][0]
#    key_letters = list(dict.fromkeys(fwd_i+ back_i ))
#    return key_letters


def rename_einsum(einsum, ops):
    groups = einsum.split("->")[0].split(",")

    # 1.  find the unused letters
    unused_spin, unused_col= unused_letters(einsum)

    # 2. find the replace_letters 
    key_letters = find_replace_letters(groups,ops)

#    print(key_letters)
    # 3.give  rename lago
    n_spin = len([c for c in key_letters if c.isupper()])
    n_col  = len([c for c in key_letters if c.islower()])
    si = ci = 0
    rename = {}
    for ch in key_letters:
        if ch.isupper(): rename[ch] = unused_spin[si]; si += 1
        else:            rename[ch] = unused_col[ci];  ci += 1

    # 4. replace
    renamed_groups = []
    for g in groups:
        renamed_groups.append("".join(rename.get(c, c) for c in g))

    return renamed_groups

_PROJECTORS = {
    "P_plus":  SpinProjector.P_plus,
    "P_minus": SpinProjector.P_minus,
    "none":    None,   # meson 
}



def gen_baryon_3pt_code(sink, source, current, 
              t_sep="t_sep", projector="P_plus"):
    # ── 1. Build wicklib correlator ──
    sr = sink[0] if isinstance(sink, tuple) else sink
    sr2 = source[0] if isinstance(source, tuple) else source
    cr = current[0] if isinstance(current, tuple) else current
    w_snk, si_snk = _to_wicklib(sr, "x")
    w_src, si_src = _to_wicklib(sr2, "y")
    w_cur, _ = _to_wicklib(cr, "z")
    proj_fn = _PROJECTORS.get(projector)
    if proj_fn is not None:
        T = proj_fn(si_src, si_snk)
        corr = Correlator(T * w_snk * w_cur * w_src.adjoint())
    else:
        corr = Correlator(w_snk * w_cur * w_src.adjoint())
    
    corr.simplify(degenerate=False)

    # ── 2. Process terms ──
    terms = []
    forward_var = None
    gamma_name = "g1"
    gamma_idx = 1

   ## remove the dicconneted diagram

    ### replace the leters in forward propgators, projector, and the sequential propgators

    def has_zz(op):
          """operator  source  sink  _z_z, _x_x, _y_y)"""
          return any(f"_{p}_{p}" in op for p in "xyz")

    for term in corr.terms:
        fac, einsum, ops = term.to_einsum()

        # remove the disconned digram
        if any(has_zz(op) for op in ops if "propag" in op):
            continue

        ## else make the replacement
        einsum = einsum.replace("...", "wtzyx")
        groups = [g.strip() for g in einsum.split("->")[0].split(",")]
#
#        print("before_repalce",groups)

        renamed_groups= rename_einsum(einsum, ops)
        
#        print("after_replace",renamed_groups)
#        # Identify P_forward index (operand with _z_y)
        fwd_i, fwd_op = forward_operand(ops)
#
#        # Gamma_cur is always operand 3 (index 2 in 0-based)
        gamma_idx_idx = 2
#
#        # ── Rename P_forward and Gamma_cur letters to standard ──
        gamma_letters = groups[gamma_idx_idx]
#
        # Remove Gamma_cur, P_forward, and P_after from sink block
        remove = {gamma_idx_idx}
        if fwd_i is not None:
            remove.add(fwd_i)
        after_i = None
        for i, op in enumerate(ops):
            if op_role(op) == "after":
                after_i = i
                remove.add(i)

        # Remaining groups and ops
        sink_groups = [renamed_groups[i] for i in range(len(groups)) if i not in remove]
        sink_ops = [op for i, op in enumerate(ops) if i not in remove]

        # Construct out from forward and after propagator index positions
        fwd_group = renamed_groups[fwd_i].replace("wtzyx", "")
        after_group = renamed_groups[after_i].replace("wtzyx", "")
        out_label = f"wtzyx{fwd_group[-3]}{after_group[-4]}{fwd_group[-1]}{after_group[-2]}"

        # Save term data
        pyq_ops = [op_to_pyq(op) for op in sink_ops]
        terms.append((sink_groups, pyq_ops, fac, out_label))

        # Forward info from first term
        if forward_var is None and fwd_op is not None:
            forward_var = var(op_flavor(fwd_op))
        if forward_var is None:
            forward_var = "prop_l"
        if gamma_name == "g1":
            g_idx, g_name = gamma_cur_idx_name(ops)
            gamma_idx = g_idx
            gamma_name = g_name if g_name else f"gamma.gamma({g_idx})"

    if not terms:
        return "/* No valid terms */"

    # Filter: only keep terms with exactly 4 free indices (2 spin + 2 color)
    # Output format is wtzyx{spin}{color}, need len(out) - 5 == 4
    valid_terms = [(sg, po, fac, ol) for sg, po, fac, ol in terms
                   if len(ol) - 5 == 4]

    if not valid_terms:
        return "/* No terms with 4 free indices */"

    n = len(valid_terms)

    # ── 3. Build code ──
    lines = [
        "# ═══════════════════════════════════════════════════════",
        f"# Baryon 3pt:",
        f"#   Gamma: {gamma_name}  Forward: {forward_var}",
        f"#   Terms: {n}",
        "# ═══════════════════════════════════════════════════════",
        "",
        "I4 = cp.eye(4, dtype=cp.complex128)",
        "G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)",
        f"Gamma_cur = cp.asarray(gamma.gamma({gamma_idx}), dtype=cp.complex128)",
        "",
        "# Epsilon tensor (GPU)",
        "eps = cp.zeros((3, 3, 3), dtype=cp.complex128)",
        "eps[0,1,2] = eps[1,2,0] = eps[2,0,1] = 1.0",
        "eps[0,2,1] = eps[2,1,0] = eps[1,0,2] = -1.0",
        "",
        "# Dirac gamma matrices (GPU)",
        "Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)",
        "Cg5 = Cmat @ G5",
        "Tmat = cp.asarray((I4 + gamma.gamma(8)) * 0.5, dtype=cp.complex128)  # P_plus",
        "",
    ]

    # ── 4. Sink block ──
    lines.append(f"# Sink block: sum over {n} Wick topologies")
    lines.append("B = core.LatticePropagator(latt_info)")
    lines.append("B.data = (")

    indent = "    "
    indent2 = "        "

    for i, (groups, pyq_ops, fac, out) in enumerate(valid_terms):
        sig = "+" if fac >= 0 else "-"
        # Full einsum with phase + output
#        full_groups = ["wtzyx"] + groups 
        full_groups =  groups 
        einsum_str = ",".join(full_groups) + f"->{out}"
#        print(einsum_str)
        lines.append(f"{indent}{sig} contract('")
        lines.append(f"{indent2}{einsum_str},")
#        lines.append(f"{indent2}{phase},")
        for po in pyq_ops:
            lines.append(f"{indent2}{po},")
        lines.append(f"{indent}),  # topo {i}")

    lines.append(")")
    lines.append("")

    # ── 5. G5-dagger → sequential solve → G5-dagger → final contraction ──
    lines.append("# G5-dagger: B̃ = γ₅ · B† · γ₅")
    lines.append("B.data = contract(")
    lines.append("    'AB, wtzyxCBji, CD -> wtzyxADij',")
    lines.append("    G5, B.data.conj(), G5)")
    lines.append("")

    lines.append("# Sequential source at t_sink")
    lines.append(f"src_seq = source.sequential12(B, {t_sep})")
    lines.append("")
    lines.append("# Sequential solve (insert your inversion code here)")
    lines.append("")

    lines.append("# G5-dagger on sequential propagator")
    lines.append("tmp_prop = core.LatticePropagator(latt_info)")
    lines.append("tmp_prop.data = contract(")
    lines.append("    'AB, wtzyxCBji, CD -> wtzyxADij',")
    lines.append("    G5, prop_seq.data.conj(), G5)")
    lines.append("")

    lines.append(f"# Tr[ tmp_prop · Gamma_cur · {forward_var} ] (P_forward)")
    lines.append("three_pt_local = contract(")
    lines.append("    'wtzyxijba, jk, wtzyxkiab -> t',")
    lines.append(f"    tmp_prop.data, Gamma_cur, {forward_var}.data)")
    lines.append("")
    lines.append("C3_t = core.gatherLattice(")
    lines.append("     three_pt_local.get(), [0, -1, -1, -1])")
    lines.append("")

    return "\n".join(lines)
