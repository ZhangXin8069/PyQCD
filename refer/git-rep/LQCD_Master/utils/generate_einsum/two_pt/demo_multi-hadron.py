"""Generate multi-baryon 2pt contraction code (pnL + pnpn)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from codegen_multi_hadron import gen_code_2pt

# ═══════════════════════════════════════════════════════════════════
#  Case 1: p + n + Lambda  (3 baryons, 1 strange)
# ═══════════════════════════════════════════════════════════════════

hadrons_pnL = [
    {"type": "baryon", "flavors": ("u", "d", "u")},
    {"type": "baryon", "flavors": ("d", "u", "d")},
    {"type": "baryon", "flavors": ("u", "d", "s")},
]

# Generate full code (definitions + sink block + save)
code = gen_code_2pt(hadrons_pnL, hadrons_pnL, "pnL", "pnL", conj_source=True)
Path("sink_pnlambda_full.py").write_text(code)
print("Written: sink_pnlambda_full.py")

# Generate sink block only (for executor pipeline)
code_sink = gen_code_2pt(hadrons_pnL, hadrons_pnL, "pnL", "pnL",
                          conj_source=True, sink_block_only=True)
Path("sink_pnlambda_block.py").write_text(code_sink)
print("Written: sink_pnlambda_block.py")

# ═══════════════════════════════════════════════════════════════════
#  Case 2: p + n + n  (3 baryons, 4u 5d -> 24x120 = 2,880 terms)
# ═══════════════════════════════════════════════════════════════════

hadrons_pnn = [
    {"type": "baryon", "flavors": ("u", "d", "u")},
    {"type": "baryon", "flavors": ("d", "u", "d")},
    {"type": "baryon", "flavors": ("d", "u", "d")},
]

code = gen_code_2pt(hadrons_pnn, hadrons_pnn, "PNN", "PNN", conj_source=True)
Path("sink_pnn_full.py").write_text(code)
print("Written: sink_pnn_full.py")

code_sink = gen_code_2pt(hadrons_pnn, hadrons_pnn, "PNN", "PNN",
                          conj_source=True, sink_block_only=True)
Path("sink_pnn_block.py").write_text(code_sink)
print("Written: sink_pnn_block.py")

# ═══════════════════════════════════════════════════════════════════
#  Case 3: p + n + p + n  (4 baryons, 6u 6d -> 720x720 = 518k terms)
#  NOTE: This takes ~3 min; run separately if needed.
# ═══════════════════════════════════════════════════════════════════

hadrons_pnpn = [
    {"type": "baryon", "flavors": ("u", "d", "u")},
    {"type": "baryon", "flavors": ("d", "u", "d")},
    {"type": "baryon", "flavors": ("u", "d", "u")},
    {"type": "baryon", "flavors": ("d", "u", "d")},
]

code = gen_code_2pt(hadrons_pnpn, hadrons_pnpn, "PNPN", "PNPN", conj_source=True)
Path("sink_pnpn_full.py").write_text(code)
print("Written: sink_pnpn_full.py")

code_sink = gen_code_2pt(hadrons_pnpn, hadrons_pnpn, "PNPN", "PNPN",
                          conj_source=True, sink_block_only=True)
Path("sink_pnpn_block.py").write_text(code_sink)
print("Written: sink_pnpn_block.py")
