"""two_pt.codegen — PyQUDA einsum format for 2-point functions.

Formats wick contraction output into PyQUDA contract() calls.
"""

from pathlib import Path
try:
    from .._wick_translate import _simplify_gammas
    from .contract import ContractionTerm
except ImportError:
    # Allow standalone operation (no parent package)
    import sys
    _p = str(Path(__file__).resolve().parent.parent)
    if _p not in sys.path: sys.path.insert(0, _p)
    from _wick_translate import _simplify_gammas
    from contract import ContractionTerm


# Gamma matrix variable names recognized in PyQUDA code
_GAMMAS = frozenset(("G5", "g1", "g2", "g3", "g4", "gtg5", "Cg5", "I4","Cg1"))


# ======================================================================
# PyQUDA einsum format
# ======================================================================

def pyquda_format_contract(t: ContractionTerm) -> str:
    """Format ContractionTerm -> contract() string.
    meson 2pt and baryon 2pt:
      Meson 2pt: <O_snk . O_src^dag> = Tr[S_bwd . Gamma_snk . S_fwd . Gamma_bar_src]
      Reference einsum:
        contract('wtzyxjiba, jk, wtzyxklba, li -> t',
                 S_bwd,        Gamma_snk,      S_fwd,  Gamma_bar_src)
      where Gamma_snk = G5 @ gamma_op_snk,
            Gamma_bar_src = gamma_op_src_bar @ G5,
            S_bwd = S_fwd.conj()  (gamma5 wrapped into gamma insertions)
      Baryon 2pt: <O_snk . O_src^dag Tmat > -> ContractionTerm 
    """
    parts = [p.strip() for p in t.einsum_subs.split(",") if p.strip()]
    ops = list(t.operands)
    has_eps = any("epsilon" in o for o in ops)

    def _is_gamma(op):
        return op in _GAMMAS

    # ---- Meson 2pt: 4 operands [gamma_snk, gamma_src, prop_bwd, prop_fwd] ----
    if not has_eps and len(ops) == 4 and _is_gamma(ops[0]) and _is_gamma(ops[1]):
        g_snk = ops[0]   # operator gamma at sink
        g_src = ops[1]   # operator gamma at source (Dirac adjoint in coefficient)
        bwd_raw = ops[2]  # backward propagator: "G5 @ S.dag() @ G5"
        fwd = ops[3]     # forward propagator = S

        # Extract S.dag() -> S.conj() from "G5 @ S.dag() @ G5" wrapping
        bwd = bwd_raw
        bwd_sub_transposed = None  # dag-transposed sub, set if dag extracted
        if bwd_raw.startswith("G5 @ ") and " @ G5" in bwd_raw:
            inner = bwd_raw[5:-5].strip()  # remove "G5 @ " prefix and " @ G5" suffix
            if inner and inner.endswith(".dag()"):
                basename = inner[:-6]  # "prop_l.dag()" -> "prop_l"
                bwd = f"{basename}.conj()"  # PyQUDA convention
                # dag transpose: swap spin_snk<->spin_src, color_snk<->color_src
                p1_sub_raw = parts[2]  # e.g. "BCab"
                bwd_sub_transposed = f"{p1_sub_raw[1]}{p1_sub_raw[0]}{p1_sub_raw[3]}{p1_sub_raw[2]}"

        # Add .data suffix for PyQUDA lattice field access
        def _with_data(s):
            return s.replace(".conj()", ".data.conj()") if s.endswith(".conj()") else f"{s}.data"
        bwd = _with_data(bwd)
        fwd = _with_data(fwd)

        # Reference convention (rho_mass.md):
        #   sink gamma insertion:   G5 @ gamma_op  (= gamma5.Gamma)
        #   source gamma insertion: gamma_op @ G5  (= Gamma_bar.gamma5)
        gamma_snk_expr = _simplify_gammas(f"G5 @ {g_snk}") if g_snk != "G5" else "I4"
        gamma_src_expr = _simplify_gammas(f"{g_src} @ G5") if g_src != "G5" else "I4"

        g1_sub, g2_sub = parts[0], parts[1]
        p1_sub = parts[2]  # e.g. "BCab"
        p2_sub = parts[3]  # e.g. "DAba"

        need_gammas = gamma_snk_expr != "I4" or gamma_src_expr != "I4"
        new_ops = []
        new_subs_list = []

        if need_gammas:
            sub_b = bwd_sub_transposed if bwd_sub_transposed else p1_sub
            Jdag, Idag = sub_b[0], sub_b[1]
            clr_a_bwd, clr_b_bwd = sub_b[2], sub_b[3]
            K, L = p2_sub[0], p2_sub[1]
            clr_a_fwd, clr_b_fwd = p2_sub[2], p2_sub[3]
            new_ops.extend([bwd, gamma_snk_expr, fwd, gamma_src_expr])
            new_subs_list.extend([
                f"wtzyx{Jdag}{Idag}{clr_a_bwd}{clr_b_bwd}",
                f"{Jdag}{K}",
                f"wtzyx{K}{L}{clr_a_fwd}{clr_b_fwd}",
                f"{L}{Idag}",
            ])
        else:
            sub_b = bwd_sub_transposed if bwd_sub_transposed else p1_sub
            new_ops.extend([bwd, fwd])
            new_subs_list.extend([
                f"wtzyx{sub_b}",
                f"wtzyx{sub_b}",
            ])

        subs_final = ", ".join(new_subs_list) + " -> t"
        args = ", ".join(new_ops)

        # Disconnected diagram detection for same-flavor mesons (e.g. eta_s, eta_c)
        # A propagator with identical color indices (e.g. "BAaa", "DCbb") traces
        # against itself and requires all-to-all methods. Omit from production code.
        if (len(p1_sub) >= 4 and p1_sub[2] == p1_sub[3]) or \
           (len(p2_sub) >= 4 and p2_sub[2] == p2_sub[3]):
            coeff = t.coefficient
            sign = "-" if coeff == -1 else ("" if coeff == 1 else f"{coeff} * ")
            return f"# [DISCONNECTED] {sign}contract('{subs_final}', {args})"

    else:
        # ---- Baryons / complex: keep as-is; propagators get wtzyx prefix ----
####### add something on the projector
#Tmat = cp.asarray((I4 + gamma.gamma(8)) * 0.5)
        new_subs_list = []
        new_ops = []
        for part, op in zip(parts, ops):
#            print(part, op)
            if _is_gamma(op) or "epsilon" in op or "Tmat" in op:
                new_subs_list.append(part)
                new_ops.append(op)
            else:
                new_subs_list.append(f"wtzyx{part}")
                new_ops.append(f"{op}.data")

        subs_final = ", ".join(new_subs_list) + " -> t"
        args = ", ".join(new_ops)

    c = t.coefficient
    if c == 1:
        return f"contract('{subs_final}', {args})"
    if c == -1:
        return f"-contract('{subs_final}', {args})"
    return f"{c} * contract('{subs_final}', {args})"
