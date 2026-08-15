#!/usr/bin/env python3
"""Complete pnΛ three-baryon 2pt: contract → codegen → demo."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from contract import get_topologies


def gen_code(topos, out_file="pnL_2pt.npy"):
    """Generate complete PyQUDA code for pnΛ 2pt."""
    lines = [
        "# ═══════════════════════════════════════════════════════",
        "# pnΛ three-baryon 2pt (576 Wick topologies)",
        "#",
        "# p n Λ system with pn in spin-1 configuration:",
        "#   p = ε(u·Cg5·d)·u,  n = ε(d·Cg5·u)·d,  Λ = ε(u·Cg5·d)·s",
        "#   extra u(p.spec) + extra d(n.spec) coupled via Cg1",
        "#",
        "# Source: p̄·n̄·Λ̄ at t=0, Sink: p·n·Λ at T",
        "# 9 quark lines: 4u + 4d + 1s → 4!×4!×1! = 576 contractions",
        "# ═══════════════════════════════════════════════════════",
        "",
        "import cupy as cp",
        "import numpy as np",
        "from pyquda import core, gamma, lattice as latt_info",
        "",
        "# ── Gamma matrices ──",
        "I4 = cp.eye(4, dtype=cp.complex128)",
        "G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)",
        "Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8),",
        "                 dtype=cp.complex128)",
        "Cg5 = Cmat @ G5",
        "Cg1 = cp.asarray(Cmat @ gamma.gamma(1),",
        "                 dtype=cp.complex128)",
        "",
        "# ── Color epsilon tensor (GPU) ──",
        "eps = cp.zeros((3, 3, 3), dtype=cp.float64)",
        "eps[0,1,2] = eps[1,2,0] = eps[2,0,1] = 1.0",
        "eps[0,2,1] = eps[2,1,0] = eps[1,0,2] = -1.0",
        "",
        "# ── Propagators ──",
        "# Needs: prop_l (u/d), prop_s (s)",
        "# prop_l.data shape: (nt, nz, ny, nx, 4, 4, 3, 3)",
        "# prop_s.data shape: (nt, nz, ny, nx, 4, 4, 3, 3)",
        "",
        f"# ── Summation over {len(topos)} topologies ──",
        "C3_t = 0.0",
        "",
    ]

    for i, t in enumerate(topos):
        sgn = "+" if t["sign"] >= 0 else "-"
        ops = t["operands"]
        einsum = t["einsum"]
        
        # Format operands
        py_ops = []
        for o in ops:
            if o.startswith("prop_"):
                py_ops.append(f"{o}.data")
            elif o.startswith("I4"):
                py_ops.append(o)
            else:
                py_ops.append(o)

        if i == 0:
            prefix = "C3_t =" if sgn == "+" else "C3_t = -"
        else:
            prefix = "C3_t += " if sgn == "+" else "C3_t -= "

        lines.append(f"{prefix}contract('{einsum}',")
        for j in range(0, len(py_ops), 4):
            chunk = ", ".join(py_ops[j:j+4])
            lines.append(f"    {chunk},")
        lines.append(")")

    lines.extend([
        "",
        "# ── MPI reduction (sum over spatial volume) ──",
        "C3_t = core.gatherLattice(",
        "    cp.array(C3_t), [0, -1, -1, -1])",
        "",
        "if core.getMPIRank() == 0:",
        f"    np.save('{out_file}', cp.asnumpy(C3_t))",
        "    print(f'Saved pnΛ 2pt to {out_file}')",
        "",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    print("Computing pnΛ 2pt topologies...")
    topos = get_topologies()
    print(f"Topologies: {len(topos)}")
    
    # Stats
    n_pos = sum(1 for t in topos if t["sign"] > 0)
    n_neg = sum(1 for t in topos if t["sign"] < 0)
    print(f"  positive: {n_pos}, negative: {n_neg}")
    t0 = topos[0]
    print(f"  ops per topology: {len(t0['operands'])}")
    print(f"  propagator types: {set(o for t in topos for o in t['operands'] if o.startswith('prop_'))}")
    print(f"  structural types: {set(o for t in topos for o in t['operands'] if not o.startswith('prop_'))}")
    
    # Write code
    code = gen_code(topos)
    with open("gen_pnL_2pt.py", "w") as f:
        f.write(code)
    lines = code.count("\n")
    contract_calls = code.count("contract(")
    print(f"\nGenerated: gen_pnL_2pt.py")
    print(f"  {lines} lines, {contract_calls} contract() calls")
    print(f"  ~{len(code)/1000:.0f} KB")
