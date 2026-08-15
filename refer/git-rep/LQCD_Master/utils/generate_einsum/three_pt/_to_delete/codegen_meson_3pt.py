"""codegen_meson_3pt — Direct PyQUDA code generation for meson 3pt.

No intermediate contraction step needed: the meson 3pt has exactly one
connected Wick topology, determined entirely by flavor conservation.

Given three meson operators J_snk(x), J_cur(z), J_src(y):
    J_src(y)  =  ψ̄_src_anti · Γ_src · ψ_src_quark
    J_cur(z)  =  ψ̄_cur_anti · Γ_cur · ψ_cur_quark
    J_snk(x)  =  ψ̄_snk_anti · Γ_snk · ψ_snk_quark

The single connected contraction (sequential-source method) gives three
propagators connecting the six fermion fields:

    spectator:  ψ_src_anti(y) · ψ̄_snk_anti(x)   (flavor = src_anti = snk_anti)
    forward:    ψ_cur_quark(z) · ψ̄_src_quark(y)   (flavor = cur_quark = src_quark)
    sequential: ψ_snk_quark(x) · ψ̄_cur_anti(z)    (flavor = snk_quark = cur_anti)

The sequential source approach:
    1. Sink block:  B = S_spectator · Γ_snk (at fixed t_sink)
    2. G5-dagger:   B̃ = G5 · conj(B) · G5
    3. Sequential inversion: use B̃ as source at t_sink → S_seq
    4. Final contraction: Tr[ S̃_seq · Γ_cur · S_fwd · Γ_src_bar ]
"""

from typing import Tuple

# ── Gamma index tables (matched to pyquda's gamma.gamma()) ──

_PYQUDA_GAMMA_IDX: dict = {
    "g1": "1", "g2": "2", "g3": "4", "g4": "8",
    "g5": "15", "G5": "15", "gtg5": "7", "I4": "0", "I": "0",
}

# Dirac adjoint factor:  Γ_bar = gamma_4 · Gamma_dagger · gamma_4 = factor × Gamma
# Derived from the anticommutation relation gamma_4·gamma_mu·gamma_4 = -gamma_mu (mu!=4),
# gamma_4·gamma_4·gamma_4 = gamma_4
_DIRAC_ADJOINT_FACTOR: dict = {
    "I4": 1, "g1": -1, "g2": -1, "g3": -1, "g4": 1,
    "g5": -1, "G5": -1, "gtg5": 1,
}

_LIGHT: frozenset = frozenset({"u", "d"})


# ── Helpers ──

def _extract_flavors(op) -> Tuple[str, str, str]:
    """Extract (antiquark_flavor, quark_flavor, gamma_name) from an operator.

    Handles both tuple-wrapped (Operator, is_baryon) and bare Operator.
    """
    tensors = op[0].tensors if isinstance(op, tuple) else op.tensors
    anti_f = next(t.flavor for t in tensors if t.type == "antiquark")
    quark_f = next(t.flavor for t in tensors if t.type == "quark")
    gamma_n = next(t.name for t in tensors if t.type == "gamma")
    return anti_f, quark_f, gamma_n


def _var(flavor: str) -> str:
    """Propagator variable name for a given flavor."""
    return "prop_l" if flavor in _LIGHT else f"prop_{flavor}"


def _gamma_g_call(gamma_name: str) -> str:
    """Generate a ``gamma.gamma(N)`` call for a gamma name."""
    name = gamma_name.strip().lower()
    if name.startswith("gamma"):
        idx = name[5:]
        name = f"g{idx}" if idx.isdigit() else name
    return f"gamma.gamma({_PYQUDA_GAMMA_IDX.get(name, '0')})"


# ═══════════════════════════════════════════════════════════════════════
# Unified codegen — full meson 3pt program in one function
# ═══════════════════════════════════════════════════════════════════════

def gen_meson_3pt_code(
    src,
    snk,
    current,
    src_name: str = "source",
    snk_name: str = "sink",
    out: str = "out_path",
) -> str:
    """Generate the complete PyQUDA code for a meson 3pt measurement.

    This is the single entry point.  It produces one contiguous code block
    covering the sink block, sequential source setup, inversion placeholder,
    and final contraction with MPI gather and save.

    Parameters
    ----------
    src : tuple  (Operator, False)
        Source meson operator, typically from ``meson_operator()``.
    snk : tuple  (Operator, False)
        Sink meson operator, typically from ``meson_operator()``.
    current : tuple  (Operator, False)
        Current insertion operator, typically from ``current_operator()``.
    src_name : str
        Label for the source in comments (e.g. ``"D0"``).
    snk_name : str
        Label for the sink in comments (e.g. ``"K+"``).
    out : str
        Output path / variable name for saving the C3(t) result.

    Returns
    -------
    str
        Complete PyQUDA code block.

    Raises
    ------
    AssertionError
        If the three operators do not form a valid connected topology
        (flavor mismatch).
    """
    # ── Extract flavors ──
    src_anti, src_quark, src_gamma = _extract_flavors(src)
    snk_anti, snk_quark, snk_gamma = _extract_flavors(snk)
    cur_anti, cur_quark, cur_gamma = _extract_flavors(current)

    # ── Validate the single connected topology ──
    # The three propagators in the connected meson 3pt are:
    #
    #   1. Spectator:  ψ_src_anti(y) · ψ̄_snk_anti(x)   (flavor = src_anti = snk_anti)
    #   2. Forward:    ψ_cur_quark(z) · ψ̄_src_quark(y)  (flavor = cur_quark = src_quark)
    #   3. Sequential: ψ_snk_quark(x) · ψ̄_cur_anti(z)   (flavor = snk_quark = cur_anti)
    assert src_anti == snk_anti, (
        f"Disconnected meson 3pt: src anti-quark flavor '{src_anti}' "
        f"≠ snk anti-quark flavor '{snk_anti}'.\\n"
        "Meson 3pt requires the spectator line (src anti → snk anti) "
        "to carry the same flavor."
    )
    assert src_quark == cur_quark, (
        f"Forward line mismatch: src quark '{src_quark}' "
        f"≠ cur quark '{cur_quark}'.\\n"
        "The forward propagator (cur quark → src antiquark) requires "
        "src_quark = cur_quark."
    )
    assert snk_quark == cur_anti, (
        f"Sequential line mismatch: snk quark '{snk_quark}' "
        f"≠ cur anti '{cur_anti}'.\\n"
        "The sequential propagator (snk quark → cur antiquark) requires "
        "snk_quark = cur_anti."
    )

    # ── Resolve variable names ──
    spectator_flavor = src_anti       # = snk_anti
    forward_flavor = cur_quark        # = src_quark
    prop_spec = _var(spectator_flavor)
    prop_fwd = _var(forward_flavor)
    dag_factor = _DIRAC_ADJOINT_FACTOR.get(src_gamma, 1)

    # Build Gamma_src_dag expression
    dag_expr = (
        f"cp.asarray({_gamma_g_call(src_gamma)}, dtype=cp.complex128)"
    )
    if dag_factor == -1:
        dag_expr = f"-{dag_expr}"

    lines = [
        "# ═══════════════════════════════════════════════════════",
        f"# Meson 3pt: {src_name} -> {snk_name}",
        f"#   spectator  = {spectator_flavor}  ({prop_spec})",
        f"#   forward    = {forward_flavor}  ({prop_fwd})",
        f"#   sink Γ     = {snk_gamma}",
        f"#   source Γ   = {src_gamma}  (Dirac adjoint factor = {dag_factor})",
        f"#   current Γ  = {cur_gamma}",
        "# ═══════════════════════════════════════════════════════",
        "",
        "# ------------------------------------------------------------------",
        "#  Gamma matrices",
        "# ------------------------------------------------------------------",
        "I4 = cp.eye(4, dtype=cp.complex128)",
        "G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)",
        f"Gamma_snk = cp.asarray({_gamma_g_call(snk_gamma)},",
        " dtype=cp.complex128)",
        f"Gamma_cur = cp.asarray({_gamma_g_call(cur_gamma)},",
        " dtype=cp.complex128)",
        f"Gamma_src = {dag_expr}",
        "",
        "# ------------------------------------------------------------------",
        "#  Step 1 — Sink block",
        "#  B(x) = S_spectator · Γ_snk",
        "# ------------------------------------------------------------------",
        f"# spectator = {spectator_flavor}  ({prop_spec})",
        "B = core.LatticePropagator(latt_info)",
        "B.data = contract(",
        "    'wtzyxjiba, jk -> wtzyxkiba',",
        f"    {prop_spec}.data, Gamma_snk)",
        "",
        "# ------------------------------------------------------------------",
        "#  Step 2 — G5-dagger for sequential source",
        "#  B̃ = G5 · conj(B) · G5",
        "# ------------------------------------------------------------------",
        "B.data = contract(",
        "    'AB, wtzyxCBji, CD -> wtzyxADij',",
        "    G5, B.data.conj(), G5)",
        "",
        "# ------------------------------------------------------------------",
        "#  Step 3 — Sequential source + solve",
        "# ------------------------------------------------------------------",
        "src_seq = source.sequential12(B, t_sink)",
        "",
        "# Sequential inversion: insert your CG / BiCGstab solver here",
        "# After inversion, prop_seq is the solution.",
        "#",
        "# ------------------------------------------------------------------",
        "#  Step 4 — Final contraction",
        "#  C_3pt(t) = Tr[ S̃_seq · Γ_cur · S_fwd · Γ̄_src ]",
        "# ------------------------------------------------------------------",
        "",
        "# 4a. G5-dagger on the sequential propagator",
        "#     S̃_seq(x,y) = G5 · conj(S_seq(y,x)) · G5",
        "tmp_prop = core.LatticePropagator(latt_info)",
        "tmp_prop.data = contract(",
        "    'AB, wtzyxCBji, CD -> wtzyxADij',",
        "    G5, prop_seq.data.conj(), G5)",
        "",
        "# 4b. 4-point contraction",
        f"# forward = {forward_flavor}  ({prop_fwd})",
        "three_pt_site = contract(",
        "    'wtzyxijba, jk, wtzyxklba, lm -> wtzyx',",
        f"    tmp_prop.data, Gamma_cur, {prop_fwd}.data, Gamma_src)",
        "",
        "# 4c. Trace spatial volume → time-slice correlator",
        "three_pt_local = contract('wtzyx -> t', three_pt_site)",
        "",
        "# 4d. MPI gather",
        "C3_t = core.gatherLattice(",
        "    cp.array(three_pt_local), [0, -1, -1, -1])",
        "",
        "# 4e. Save",
        "if core.getMPIRank() == 0:",
        f"    np.save({out}, cp.asnumpy(C3_t))",
    ]
    return "\n".join(lines)
