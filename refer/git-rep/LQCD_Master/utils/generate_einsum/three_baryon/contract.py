"""contract — pnΛ three-baryon 2pt, full wicklib contraction with Cg1.

Strategy:
  1. Build sink op = Diquark(p)×Quark(p) × Diquark(n)×Quark(n) × Diquark(Λ)×Quark(Λ)
  2. Build source op = same at y, take adjoint
  3. Inject Cg1 (sink) and Cg1.D (source†) between p/n spectators
  4. WickTerm contracts all
  5. Output: codegen-ready topology list
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wicklib.quark import QuarkField
from wicklib.gamma import Gamma, GAMMA_1, GAMMA_5, C
from wicklib.index import SpinIndex, ColorIndex
from wicklib.tensor import SpinGammaTensor, QuarkFieldTensor
from wicklib.operator import Diquark, Quark, Operator
from wicklib.correlator import WickTerm

Cg1 = C @ GAMMA_1
LIGHT = frozenset({"u", "d"})
def _var(f: str) -> str: return "prop_l" if f in LIGHT else f"prop_{f}"
def _gamma_name(idx: int) -> str:
    return {0: "I4", 1: "g1", 2: "g2", 4: "g3", 8: "g4",
            15: "G5", 10: "Cmat", 5: "Cg5", 11: "Cg1"}.get(idx, f"gamma{idx}")


def _make_baryon(flavors, loc):
    f1, f2, f3 = flavors
    ci = ColorIndex.new()
    si = SpinIndex.new()
    diq = Diquark(QuarkField(f1), QuarkField(f2), C @ GAMMA_5)
    spec = Quark(QuarkField(f3), Gamma(0))
    return diq.at(loc, ci) * spec.at(loc, si, ci)


def _build_pnL_op(loc):
    return (_make_baryon(("u","d","u"), loc) *
            _make_baryon(("d","u","d"), loc) *
            _make_baryon(("u","d","s"), loc))


def _extract_specs(tensors, loc, is_antiquark):
    """Find spectator qft positions based on location and type.
    
    Sink: DiquarkBlock + QuarkBlock → q3 at end of each 7-tensor group [6, 13, 20]
    Source† (adjoint): AntiQuarkBlock + AntiDiquarkBlock → q̄3 at start [1, 8, 15]
    """
    if is_antiquark:
        return [(1 + 7*i, tensors[1 + 7*i]) for i in range(3)]
    else:
        return [(6 + 7*i, tensors[6 + 7*i]) for i in range(3)]


def _inject_Cg1(tensors, loc, is_antiquark):
    """Remove p/n I-gammas, insert Cg1 between spectator spins."""
    specs = _extract_specs(tensors, loc, is_antiquark)
    p_spec, n_spec = specs[0], specs[1]
    
    # I-gamma position: -1 for sink (before qft), +1 for source† (after qft)
    ig_idx_p = p_spec[0] + (-1 if not is_antiquark else 1)
    ig_idx_n = n_spec[0] + (-1 if not is_antiquark else 1)
    
    new_tensors = list(tensors)
    del new_tensors[max(ig_idx_p, ig_idx_n)]
    del new_tensors[min(ig_idx_p, ig_idx_n)]
    
    # Re-find spectator qfts after removal
    # Two removals: first removes the LARGER index
    # For sink: p_spec at 6 (I_g at 5), n_spec at 13 (I_g at 12)
    # Remove ig_idx_n (12) first → both shift -1: p=5, n=12
    # Remove ig_idx_p (5) next → both shift -1 more: p=4, n=11
    # Shift per spec: p_spec.(6→5→4 = -2), n_spec.(13→12→11 = -2)
    # Wait: after first removal (ig_n=12), spec_n shifts from 13→12, spec_p stays at 6
    # After second removal (ig_p=5), spec_p shifts 6→5, spec_n shifts 12→11
    # So: p_shift = -1 (only second removal affects it)
    #     n_shift = -2 (both removals affect it)
    # But if ig_p > ig_n, the order reverses.
    ig_ordered = sorted([ig_idx_p, ig_idx_n])
    p_new_idx = p_spec[0] - (1 if p_spec[0] > ig_ordered[0] else 0) - (1 if p_spec[0] > ig_ordered[1] else 0)
    n_new_idx = n_spec[0] - (1 if n_spec[0] > ig_ordered[0] else 0) - (1 if n_spec[0] > ig_ordered[1] else 0)
    p_new = (p_new_idx, new_tensors[p_new_idx])
    n_new = (n_new_idx, new_tensors[n_new_idx])
    
    cg1_gamma = Cg1.D if is_antiquark else Cg1
    new_tensors.append(SpinGammaTensor(cg1_gamma, p_new[1].spin, n_new[1].spin))
    return new_tensors


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def contract_pnL_2pt():
    """Full pnΛ 2pt Wick contraction."""
    SpinIndex._counter = 0
    ColorIndex._counter = 0
    
    sink_op = _build_pnL_op("x")
    sink_factor, sink_tensors = sink_op.terms[0].to_tensor()
    sink_tensors = _inject_Cg1(sink_tensors, "x", is_antiquark=False)
    
    src_op = _build_pnL_op("y")
    src_adj = src_op.adjoint()
    _, src_tensors = src_adj.terms[0].to_tensor()
    src_tensors = _inject_Cg1(src_tensors, "y", is_antiquark=True)
    
    all_tensors = sink_tensors + src_tensors
    wt = WickTerm(1, all_tensors)
    for adj in wt.adjacency_terms:
        adj.simplify(degenerate=False)
    
    return [t for t in wt.adjacency_terms if t.factor != 0]


def to_codegen_list(terms):
    """Convert to codegen-ready list of topology dicts."""
    results = []
    for adj in terms:
        factor, subs, ops = adj.to_einsum()
        
        # Parse einsum: splits by "," and processes each group
        raw_groups = [g.strip() for g in subs.split(",")]
        
        clean_groups = []
        clean_ops = []
        for g, op in zip(raw_groups, ops):
            # Propagator: replace "..." with "wtzyx"
            if g.startswith("..."):
                clean_groups.append(g.replace("...", "wtzyx", 1))
                # Rename propag_* to variable
                if op.startswith("propag_"):
                    parts = op.replace("propag_", "").split("_")
                    flav = parts[0]
                    snk = parts[1] if len(parts) > 1 else "x"
                    src = parts[2] if len(parts) > 2 else "y"
                    vn = _var(flav)
                    if snk > src:
                        clean_ops.append(f"G5 @ {vn}.dag() @ G5")
                    else:
                        clean_ops.append(vn)
                else:
                    clean_ops.append(op)
            else:
                # Structural: epsilon or gamma
                clean_groups.append(g)
                if op == "epsilon":
                    clean_ops.append("eps")
                elif op.startswith("gamma("):
                    idx = int(op.replace("gamma(", "").replace(")", ""))
                    clean_ops.append(_gamma_name(idx))
                else:
                    clean_ops.append(op)
        
        # Remove trailing "->..." (everything after the last ->)
        base = ", ".join(clean_groups)
        if "->" in subs:
            base = base.rsplit("->", 1)[0]
        final_subs = base + " -> t"
        
        sign = 1 if float(factor.real) >= 0 else -1
        
        results.append({
            "einsum": final_subs,
            "sign": sign,
            "factor": float(factor.real),
            "operands": clean_ops,
        })
    
    return results


def get_topologies():
    """Full pipeline: contract → codegen-ready."""
    terms = contract_pnL_2pt()
    return to_codegen_list(terms)


if __name__ == "__main__":
    topos = get_topologies()
    print(f"Total topologies: {len(topos)}")
    t0 = topos[0]
    print(f"Einsum: {t0['einsum'][:200]}...")
    print(f"Ops ({len(t0['operands'])}): {t0['operands']}")
