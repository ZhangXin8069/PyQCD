#!/usr/bin/env python3
"""Real-gauge epsilon-halving gates for the two newly added flow times."""
import json
import time

import numpy as np

from ope_components import ROOT, atomic_json, export_product, raw_base, require, validate_raw
from run_task import check_campaign, compute_raw


def main():
    check_campaign()
    requests = json.loads((ROOT / "requests.json").read_text())
    gates = []
    for tau in (0.05, 0.8):
        req = next(r for r in requests if r["conf_id"] == "10000" and r["tau"] == tau)
        started = time.monotonic()
        coarse_base = compute_raw(req)
        fine_req = dict(req, epsilon=0.005)
        fine_base = compute_raw(fine_req, ROOT / "validation" / "fine")
        coarse, cm = validate_raw(coarse_base, req)
        fine, fm = validate_raw(fine_base, fine_req)
        # Primitive T/S sectors only: a cancellation in combined must not hide a failed sector.
        sector_errors = {}
        for channel, prefix in enumerate(("U", "H")):
            for component, label in ((1, "T"), (2, "S")):
                reference = fine[:, channel, :2, component]
                delta = coarse[:, channel, :2, component] - reference
                norm = float(np.linalg.norm(reference))
                sector_errors[prefix + label] = float(np.linalg.norm(delta)) / norm if norm else float(np.linalg.norm(delta))
        plaquette_difference = abs(cm["flowed_gauge_stats"]["plaquette"] - fm["flowed_gauge_stats"]["plaquette"])
        passed = all(v < 5e-3 for v in sector_errors.values()) and plaquette_difference < 5e-5
        gate = dict(conf_id="10000", tau_a2=tau, epsilon_coarse=0.01, epsilon_fine=0.005,
                    relative_l2_primitives=sector_errors, tolerance_relative_l2=5e-3,
                    plaquette_difference=plaquette_difference, tolerance_plaquette=5e-5,
                    passed=passed, elapsed_seconds=time.monotonic() - started,
                    coarse_base=str(coarse_base), fine_base=str(fine_base))
        atomic_json(ROOT / "validation" / f"epsilon_tau{tau:.3f}.json", gate)
        print(json.dumps(gate), flush=True)
        require(passed, f"epsilon gate failed for tau={tau}")
        export_product(coarse_base, req)
        gates.append(gate)
    atomic_json(ROOT / "validation" / "real_gauge_gates.json", {"passed": True, "gates": gates})


if __name__ == "__main__":
    main()
