from __future__ import annotations

"""codegen_multi_hadron — Universal multi-hadron 2pt code generation.

Pattern: hadron specs → wicklib contraction → result dict → PyQUDA code
(following the baryon_3pt_gen.py architecture)
"""
import sys
from pathlib import Path
from typing import Any

_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

try:
    from utils.generate_einsum.hadron_operator import (
        Operator, meson_operator, baryon_operator, current_operator)
    from utils.generate_einsum._wick_translate import _to_wicklib
    from utils.generate_einsum.wicklib.correlator import Correlator
    from utils.generate_einsum.wicklib.operator import SpinProjector
except ImportError:
    from hadron_operator import Operator, meson_operator, baryon_operator, current_operator
    from _wick_translate import _to_wicklib
    from wicklib.correlator import Correlator
    from wicklib.operator import SpinProjector


# Gamma index → PyQUDA variable name
_GAMMA_PYQ = {5: "Cg5", 15: "G5", 1: "g1", 7: "gtg5"}


# ═══════════════════════════════════════════════════════════════════════
#  Hadron spec → Operator
# ═══════════════════════════════════════════════════════════════════════

def build_hadron_op(spec: dict) -> Operator:
    """Convert a hadron spec dict to an Operator object.

    Spec format:
      meson:  {"type": "meson", "flavors": ("u","d"), "gamma": "g5"}
      baryon: {"type": "baryon", "flavors": ("u","d","u")}
      antibaryon: {"type": "antibaryon", "flavors": ("u","d","s")}
    """
    t = spec["type"]
    if t == "meson":
        anti, quark = spec["flavors"]
        op, _ = meson_operator(anti, quark, spec.get("gamma", "g5"))
        return op
    elif t in ("baryon", "antibaryon"):
        flavors = spec["flavors"]
        op, _ = baryon_operator(*flavors)
        return op
    else:
        raise ValueError(f"Unknown hadron type: {t}")


def build_wick_ops(
    sink_specs: list[dict],
    source_specs: list[dict],
    sink_loc: str = "x",
    source_loc: str = "y",
    conj_source: bool = True,
) -> "tuple":
    """Build wicklib operators for sink and source.

    Returns (wick_sink, wick_source, spin_indices, proj_choices).

    proj_choices[i] = projector for baryon pair (sink[i], source[i]).
    """
    spin_indices = []
    proj_choices = []

    sink_parts = []
    for i, spec in enumerate(sink_specs):
        loc = sink_loc
        op = build_hadron_op(spec)  # always O
        w, si = _to_wicklib(op, loc)
        if spec.get("type") == "antibaryon":
            w = w.adjoint()  # O†(x) → ψ̄(x)
        sink_parts.append(w)
        if spec["type"] in ("baryon", "antibaryon") and si is not None:
            spin_indices.append((loc, i, si))
            proj_choices.append(spec.get("projector", "P_plus"))

    wick_sink = 1
    for p in sink_parts:
        wick_sink = wick_sink * p

    source_parts = []
    for i, spec in enumerate(source_specs):
        loc = source_loc
        op = build_hadron_op(spec)  # always O
        w, si = _to_wicklib(op, loc)
        if conj_source:
            if spec.get("type") == "antibaryon":
                pass  # O(y) → ψ(y)
            else:
                w = w.adjoint()  # O†(y) → ψ̄(y)
        else:
            if spec.get("type") == "antibaryon":
                w = w.adjoint()  # O†(y) → ψ̄(y)
            else:
                pass  # O(y) → ψ(y)
        source_parts.append(w)
        if spec["type"] in ("baryon", "antibaryon") and si is not None:
            spin_indices.append((loc, i, si))

    wick_source = 1
    for p in source_parts:
        wick_source = wick_source * p

    return wick_sink, wick_source, spin_indices, proj_choices


# ═══════════════════════════════════════════════════════════════════════
#  Spin connector: pair baryon spin indices
# ═══════════════════════════════════════════════════════════════════════

def build_spin_connector(
    spin_indices: list,
    n_sink: int,
    proj_choices: list[str],
) -> "Any":
    """Build per-baryon-pair projectors.

    spin_indices has sink entries first (length n_sink),
    then source entries.  Pair i: sink[i] ↔ source[i]
    with projector from proj_choices[i].
    """
    n_source = len(spin_indices) - n_sink
    if n_sink == 0 or n_source == 0:
        return 1

    result = 1
    for idx in range(min(n_sink, n_source)):
        _, _, si_snk = spin_indices[idx]
        _, _, si_src = spin_indices[n_sink + idx]
        pname = proj_choices[idx] if idx < len(proj_choices) else "P_plus"
        if pname == "P_plus":
            result = result * SpinProjector.P_plus(si_snk, si_src)
        elif pname == "P_minus":
            result = result * SpinProjector.P_minus(si_snk, si_src)
        else:
            raise ValueError(f"Unknown projector: {pname}")
    return result


# ═══════════════════════════════════════════════════════════════════════
#  Op → PyQUDA variable name
# ═══════════════════════════════════════════════════════════════════════

def _op_to_pyq(op: str) -> str:
    """Map wicklib operand name → PyQUDA variable name."""
    if op.startswith("gamma("):
        idx = int(op.replace("gamma(", "").replace(")", ""))
        return _GAMMA_PYQ.get(idx, f"gamma.gamma({idx})")
    elif op.startswith("propag_"):
        parts = op.split("_")
        flavor = parts[1]
        var = "prop_l" if flavor in ("u", "d") else f"prop_{flavor}"
        return f"{var}.data"
    elif op == "epsilon":
        return "epsilon"
    elif op == "projector":
        return "Tmat"
    return op


# ═══════════════════════════════════════════════════════════════════════
#  Contraction + result (stores raw pyq_ops for dynamic codegen)
# ═══════════════════════════════════════════════════════════════════════

def contract_multi_2pt(
    sink_specs: list[dict],
    source_specs: list[dict],
    conj_source: bool = True,
) -> dict:
    """Contraction entry point. Returns a result dict with pyq_ops."""
    w_snk, w_src, spin_ix, proj = build_wick_ops(sink_specs, source_specs, conj_source=conj_source)
    n_sink_bar = sum(1 for s in sink_specs if s.get("type") in ("baryon", "antibaryon"))
    S = build_spin_connector(spin_ix, n_sink_bar, proj)
    corr = Correlator(S * w_snk * w_src)
    corr.simplify(degenerate=False)

    # Map projector name → PyQUDA variable name
    uproj = list(set(proj))  # unique projector types
    if uproj == ["P_plus"]:
        proj_map = {"P_plus": "Tmat"}
    elif uproj == ["P_minus"]:
        proj_map = {"P_minus": "Tmat"}
    else:
        proj_map = {"P_plus": "Tmat_P", "P_minus": "Tmat_M"}

    sink_terms = []
    for term in corr.terms:
        fac, einsum, ops = term.to_einsum()
        einsum = einsum.replace("...", "wtzyx")
        pyq_ops = [_op_to_pyq(op) for op in ops]

        # Replace "Tmat" by position with the correct projector name
        tmat_idx = 0
        for j, op in enumerate(pyq_ops):
            if op == "Tmat":
                pname = proj[tmat_idx] if tmat_idx < len(proj) else "P_plus"
                pyq_ops[j] = proj_map[pname]
                tmat_idx += 1

        sink_terms.append({
            "sign": round(fac.real),
            "einsum": einsum,
            "pyq_ops": pyq_ops,
        })

    return {
        "ok": True,
        "n_topologies": len(corr.terms),
        "sink_terms": sink_terms,
        "proj_used": uproj,
    }


# ═══════════════════════════════════════════════════════════════════════
#  PyQUDA code generation — purely dynamic, no hardcoded operand layout
# ═══════════════════════════════════════════════════════════════════════

def gen_code_2pt(
    sink_specs: list[dict],
    source_specs: list[dict],
    src_name: str = "source",
    snk_name: str = "sink",
    conj_source: bool = True,
    out_path: str = '"output.npy"',
    sink_block_only: bool = False,
) -> str:
    """Generate PyQUDA code for multi-hadron 2pt.

    When sink_block_only=True, omits I4/G5/epsilon/Cg5/Tmat definitions
    (assumes main script provides them).  Returns only the sink block
    contract code, trace, gather, and save.
    """
    r = contract_multi_2pt(sink_specs, source_specs, conj_source=conj_source)
    terms = r["sink_terms"]
    if not terms:
        return "/* No contraction terms */"

    n_topo = r["n_topologies"]

    if not sink_block_only:
        # Collect definitions needed
        needs_eps = any("epsilon" in t["pyq_ops"] for t in terms)
        needs_tmat = any(t_o.startswith("Tmat") for t in terms for t_o in t["pyq_ops"])
        proj_used = r.get("proj_used", [])

        def_lines = [
            "# ═══════════════════════════════════════════════════════",
            f"# Multi-hadron 2pt: {snk_name} <- {src_name}",
            f"#   Hadrons in sink:   {len(sink_specs)}",
            f"#   Hadrons in source: {len(source_specs)}",
            f"#   Topologies: {n_topo}",
            "# ═══════════════════════════════════════════════════════",
            "",
            "import cupy as cp",
            "from opt_einsum import contract",
        ]

        if needs_eps:
            def_lines += [
                "# Epsilon tensor (3D color anti-symmetric)",
                "epsilon = cp.zeros((3, 3, 3), dtype=cp.float64)",
                "epsilon[0,1,2] = epsilon[1,2,0] = epsilon[2,0,1] = 1.0",
                "epsilon[0,2,1] = epsilon[2,1,0] = epsilon[1,0,2] = -1.0",
                "",
            ]
        if needs_tmat:
            def_lines += [
                "# Gamma matrices for baryon spin/parity structure",
                "from pyquda_utils import core, io, gamma, source",
                "I4 = cp.eye(4, dtype=cp.complex128)",
                "G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)",
                "Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)",
                "Cg5 = Cmat @ G5",
            ]
            proj_map_cg = {}
            if proj_used == ["P_plus"]:
                proj_map_cg = {"P_plus": "Tmat"}
            elif proj_used == ["P_minus"]:
                proj_map_cg = {"P_minus": "Tmat"}
            else:
                proj_map_cg = {"P_plus": "Tmat_P", "P_minus": "Tmat_M"}
            for p in proj_used:
                name = proj_map_cg[p]
                if p == "P_plus":
                    def_lines.append(f"{name} = cp.asarray((I4 + gamma.gamma(8)) * 0.5, dtype=cp.complex128)  # P_plus")
                elif p == "P_minus":
                    def_lines.append(f"{name} = cp.asarray((I4 - gamma.gamma(8)) * 0.5, dtype=cp.complex128)  # P_minus")
            def_lines.append("")
        def_text = "\n".join(def_lines)
    else:
        def_text = ""

    # ── Sink block: separate contract per topology (avoids giant single expression) ──
    #   greedy by default; OOM → fallback to dp for that topology
    lines = []
    lines.append(f"# Sink block: sum over {n_topo} Wick topology/ies")
    lines.append("")

    indent = "    "
    indent2 = "        "
    indent3 = "            "

    for i, t in enumerate(terms):
        ops = t["pyq_ops"]
        einsum_t = t['einsum'].rsplit('->', 1)[0] + '->t'

        # Outer try: if both greedy and dp OOM, skip this topology
        lines.append("try:")

        if i == 0:
            sgn = "-" if t["sign"] < 0 else ""
            # First term: direct assignment, inner try for dp fallback
            lines.append(f"{indent}try:")
            if sgn:
                lines.append(f"{indent2}two_pt_site = {sgn} contract('{einsum_t}',")
            else:
                lines.append(f"{indent2}two_pt_site = contract('{einsum_t}',")
            for j in range(0, len(ops), 3):
                chunk = ops[j:j + 3]
                lines.append(f"{indent3}{', '.join(chunk)},")
            lines.append(f"{indent2})  # topo {i} greedy")
            # dp fallback
            lines.append(f"{indent}except (cp.cuda.memory.OutOfMemoryError, MemoryError):")
            lines.append(f"{indent2}print('topo {i}: OOM, retry dp...')")
            lines.append(f"{indent2}cp.get_default_memory_pool().free_all_blocks()")
            if sgn:
                lines.append(f"{indent2}two_pt_site = {sgn} contract('{einsum_t}',")
            else:
                lines.append(f"{indent2}two_pt_site = contract('{einsum_t}',")
            for j in range(0, len(ops), 3):
                chunk = ops[j:j + 3]
                lines.append(f"{indent3}{', '.join(chunk)},")
            lines.append(f"{indent2}optimize='dp',")
            lines.append(f"{indent2})  # topo {i} dp")
            lines.append(f"{indent}print('topo={i} is finished')")

        else:
            sgn = "+" if t["sign"] >= 0 else "-"
            # Remaining terms: inner try for dp fallback
            lines.append(f"{indent}try:")
            lines.append(f"{indent2}res_gpu = contract('{einsum_t}',")
            for j in range(0, len(ops), 3):
                chunk = ops[j:j + 3]
                lines.append(f"{indent3}{', '.join(chunk)},")
            lines.append(f"{indent2})  # topo {i} greedy")
            # dp fallback
            lines.append(f"{indent}except (cp.cuda.memory.OutOfMemoryError, MemoryError):")
            lines.append(f"{indent2}print('topo {i}: OOM, retry dp...')")
            lines.append(f"{indent2}cp.get_default_memory_pool().free_all_blocks()")
            lines.append(f"{indent2}res_gpu = contract('{einsum_t}',")
            for j in range(0, len(ops), 3):
                chunk = ops[j:j + 3]
                lines.append(f"{indent3}{', '.join(chunk)},")
            lines.append(f"{indent2}optimize='dp',")
            lines.append(f"{indent2})  # topo {i} dp")
            # Accumulate
            lines.append(f"{indent}two_pt_site {sgn}= res_gpu")
            lines.append(f"{indent}del res_gpu")
            lines.append(f"{indent}print('topo={i} is finished')")

        # Outer except: dp also OOM, skip
        lines.append(f"except (cp.cuda.memory.OutOfMemoryError, MemoryError):")
        lines.append(f"{indent}print('topo {i}: both greedy and dp OOM, skipping')")

        # Always synchronize and free
        lines.append("cp.cuda.Stream.null.synchronize()")
        lines.append("cp.get_default_memory_pool().free_all_blocks()")
        lines.append("cp.get_default_pinned_memory_pool().free_all_blocks()")
        lines.append("")

    lines += [
        "",
        "# Trace spatial volume → time-slice",
        "two_pt_local = two_pt_site",
        "",
        "# MPI gather",
        "from pyquda_comm import array",
        "two_pt_result = core.gatherLattice(array.arrayAsNumpy(two_pt_local, backend='cupy'), [0, -1, -1, -1])",
    ]

    sink_text = "\n".join(lines)
    if sink_block_only:
        return sink_text
    # Full output: definitions + sink + save
    save_lines = [
        "",
#        "# Save",
#        "if core.getMPIRank() == 0:",
#        f"    np.savetxt({out_path}, two_pt_result)",
    ]
    return def_text + "\n" + sink_text + "\n" + "\n".join(save_lines)


# ═══════════════════════════════════════════════════════════════════════
#  Demos
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    PN_LAMBDA = [
        ("p+n+Λ 3-baryon 2pt",
         [{"type": "baryon", "flavors": ("u","d","u")},
          {"type": "baryon", "flavors": ("d","u","d")},
          {"type": "baryon", "flavors": ("u","d","s")}],
         True),
    ]

    for label, hadrons, cs in PN_LAMBDA:
        snk = src = hadrons
        print(f"\n{'#' * 70}")
        print(f"#  {label}")
        print(f"{'#' * 70}")
        r = contract_multi_2pt(snk, src, conj_source=cs)
        print(f"  topologies: {r['n_topologies']}")
        for t in r['sink_terms'][:4]:
            print(f"    sign={t['sign']:+d}  ops={t['pyq_ops'][:6]}...")
        print(f"    ... ({r['n_topologies'] - 4} more)" if r['n_topologies'] > 4 else "")
        print()
        code = gen_code_2pt(snk, src, "pnL", "pnL", conj_source=cs)
        # Show just the definition section
        print(code)
#        for line in code.split("\n"):
#            if "# Epsilon" in line or "# Gamma" in line or \
#               line.strip().startswith("epsilon") or \
#               ("Tmat" in line and "cp." in line):
#                print(line)
#        print("  ...")

