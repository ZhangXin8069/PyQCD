"""codegen — PyQUDA contraction code for pnΛ three-baryon 2pt.

576 topologies, each with:
  6 epsilon tensors + 6 Cg5 + 2 Cg1 + 9 propagators = 23 operands
"""

def gen_code(result: dict, phase: str = "phase_sink") -> str:
    terms = result["terms"]

    lines = [
        "# ═══════════════════════════════════════════════════════",
        "# pnΛ three-baryon 2pt (576 topologies)",
        "# pn spin-1 via Cg1 coupling of extra u+d spectators",
        "# ═══════════════════════════════════════════════════════",
        "",
        "I4 = cp.eye(4, dtype=cp.complex128)",
        "G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)",
        "Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)",
        "Cg5 = Cmat @ G5",
        "Cg1 = cp.asarray(Cmat @ gamma.gamma(1), dtype=cp.complex128)",
        "",
        "# Epsilon tensor (GPU)",
        "eps = cp.zeros((3, 3, 3), dtype=cp.float64)",
        "eps[0,1,2] = eps[1,2,0] = eps[2,0,1] = 1.0",
        "eps[0,2,1] = eps[2,1,0] = eps[1,0,2] = -1.0",
        "",
    ]

    lines.append(f"# {len(terms)} topology(ies)")
    lines.append("")
    lines.append("# Contract all topologies")

    # For each topology, operands = [eps×6, Cg5×6, Cg1×2, propagators×9]
    for i, t in enumerate(terms):
        sgn = "+" if t["sign"] >= 0 else "-"
        desc = t.get("description", "")
        einsum = t["einsum"]
        ops = t["operands"]
        # Prefix propagator data with wtzyx convention
        py_ops = []
        for o in ops:
            if o == "eps":
                py_ops.append("eps")
            elif o in ("Cg5", "Cg1"):
                py_ops.append(o)
            else:
                py_ops.append(f"{o}.data")

        py_ops_str = ", ".join(py_ops)
        if i == 0:
            if sgn == '-':
                lines.append(f"C3_t = -contract('{einsum}',")
            else:
                lines.append(f"C3_t = contract('{einsum}',")
        else:
            if sgn == '-':
                lines.append(f"C3_t = C3_t - contract('{einsum}',")
            else:
                lines.append(f"C3_t = C3_t + contract('{einsum}',")

        for j in range(0, len(py_ops), 4):
            chunk = py_ops[j:j + 4]
            lines.append(f"    {', '.join(chunk)},")
        lines.append(")")
        if i % 100 == 0 and i > 0:
            lines.append(f"")

    lines.append("")
    lines.append("# MPI reduction")
    lines.append("C3_t = core.gatherLattice(")
    lines.append("    cp.array(C3_t), [0, -1, -1, -1])")
    lines.append("")
    lines.append("if core.getMPIRank() == 0:")
    lines.append("    np.save('pnL_2pt.npy', cp.asnumpy(C3_t))")

    return "\n".join(lines)
