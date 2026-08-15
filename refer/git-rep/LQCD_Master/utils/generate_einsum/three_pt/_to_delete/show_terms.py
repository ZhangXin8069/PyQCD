#!/usr/bin/env python3
"""Print Λ→p sequential source for each term individually."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hadron_operator import baryon_operator, current_operator
from three_pt.contract import _to_wicklib
from wicklib.correlator import Correlator
from wicklib.operator import SpinProjector

LIGHT = frozenset({"u","d"})
GAMMA_PYQ = {5:"Cg5",15:"G5",1:"g1",7:"gtg5"}

def rename_label(label, rename):
    return "".join(rename.get(ch, ch) for ch in label)

def op_to_pyq(op):
    if op.startswith("gamma("):
        idx = int(op.replace("gamma(","").replace(")",""))
        return GAMMA_PYQ.get(idx, f"gamma.gamma({idx})")
    if op == "epsilon": return "eps"
    if op == "projector": return "Tmat"
    if op.startswith("propag_"):
        f = op.split("_")[1]
        return "prop_l" if f in LIGHT else f"prop_{f}"
    return op

def var(f):
    return "prop_l" if f in LIGHT else f"prop_{f}"

cur_t = current_operator("u","s","g1")
cur = cur_t[0]
snk_t,_ = baryon_operator("u","d","u")
src_t,_ = baryon_operator("u","d","s")
w_snk,si_snk = _to_wicklib(snk_t,"x")
w_src,si_src = _to_wicklib(src_t,"y")
w_cur = _to_wicklib(cur,"z")
T = SpinProjector.P_plus(si_src, si_snk)
corr = Correlator(T * w_snk * w_cur * w_src.adjoint())
corr.simplify(degenerate=False)

sep = "#" + "=" * 65

for ti, term in enumerate(corr.terms):
    fac, einsum, ops = term.to_einsum()
    einsum = einsum.replace("...","wtzyx")
    groups = [g.strip() for g in einsum.split("->")[0].split(",")]

    # Find P_forward
    fwd_i = fwd_letters = None
    for i, op in enumerate(ops):
        if "_z_y" in op or "_y_z" in op:
            fwd_i = i
            fwd_letters = groups[i].replace("wtzyx","")
            break

    if not fwd_letters:
        continue

    gamma_letters = groups[2]

    # Rename
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

    fwd_op = ops[fwd_i]
    fwd_flav = fwd_op.split("_")[1]
    fwd_var_name = var(fwd_flav)

    # Remove: Gamma_cur(2), P_forward(fwd_i), P_after(_x_z)
    remove = {2, fwd_i}
    for i, op in enumerate(ops):
        if "_x_z" in op:
            remove.add(i)

    sink_groups = [renamed_groups[i] for i in range(len(groups)) if i not in remove]
    sink_ops = [ops[i] for i in range(len(ops)) if i not in remove]

    # Free indices
    letters = {}
    for g in sink_groups:
        for ch in g.replace("wtzyx",""):
            letters[ch] = letters.get(ch, 0) + 1
    free = [ch for ch, cnt in letters.items() if cnt == 1]
    out = "wtzyx" + "".join(free)

    pyq_ops = [op_to_pyq(o) for o in sink_ops]
    sink_einsum = "wtzyx," + ",".join(sink_groups) + f"->{out}"

    print(sep)
    print(f"# Term {ti}: sign={fac:+d}")
    print(f"#   Forward: {fwd_op} -> {fwd_var_name}.data")
    print(sep)

    print(f"B = core.LatticePropagator(latt_info)")
    print(f"B.data = contract('")
    print(f"    {sink_einsum},")
    print(f"    ones_phase,")
    for o in pyq_ops:
        if o.startswith("prop"):
            print(f"    {o}.data,")
        else:
            print(f"    {o},")
    print(f")")
    print()

    print(f"# G5-dagger")
    print(f"B.data = contract('AB, wtzyxCBji, CD -> wtzyxADij', G5, B.data.conj(), G5)")
    print()
    print(f"# Sequential source")
    print(f"src_seq = source.sequential12(B, t_sep)")
    print(f"# Sequential solve")
    print()
    print(f"# G5-dagger on sequential")
    print(f"tmp_prop = core.LatticePropagator(latt_info)")
    print(f"tmp_prop.data = contract('AB, wtzyxCBji, CD -> wtzyxADij', G5, prop_seq.data.conj(), G5)")
    print()
    print(f"# Final contraction")
    print(f"Gamma_cur = cp.asarray(gamma.gamma(1), dtype=cp.complex128)")
    print(f"three_pt_site = contract(")
    print(f"    'wtzyxijba, jk, wtzyxkiab -> wtzyx',")
    print(f"    tmp_prop.data, Gamma_cur, {fwd_var_name}.data)")
    print(f"three_pt_local = contract('wtzyx -> t', three_pt_site)")
    print(f"C3_t = core.gatherLattice(cp.array(three_pt_local), [0, -1, -1, -1])")
    print(f"if core.getMPIRank() == 0:")
    print(f"    np.save('output_term{ti}.npy', cp.asnumpy(C3_t))")
    print()
