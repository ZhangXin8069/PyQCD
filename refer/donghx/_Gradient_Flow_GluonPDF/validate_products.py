#!/usr/bin/env python3
"""Shape, finiteness, and shared-bootstrap checks for analysis products."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
ratio_files = sorted((ROOT / "ratios").glob("*.npz"))
fit_files = sorted((ROOT / "fits").glob("*.npz"))
taus = sorted({float(np.load(p, allow_pickle=False)["tau_a2"]) for p in ratio_files})
expected = {"ratio": 9 * len(taus), "fit": 18 * len(taus)}
assert len(ratio_files) == expected["ratio"], (len(ratio_files), expected, taus)
assert len(fit_files) == expected["fit"], (len(fit_files), expected, taus)
ref = np.load(ratio_files[0], allow_pickle=False)["boot_indices"]
for path in ratio_files:
    with np.load(path, allow_pickle=False) as d:
        assert d["ratio_boot"].shape == (100, 2, 3, 21, 5, 11), path
        assert d["ratio_original"].shape == (2, 3, 21, 5, 11), path
        assert d["ratio_valid_mask"].shape == (2, 3, 21, 5, 11), path
        assert np.array_equal(ref, d["boot_indices"]), path
        mask = np.broadcast_to(d["ratio_valid_mask"][None], d["ratio_boot"].shape)
        assert np.all(np.isfinite(d["ratio_boot"][mask])), path
for path in fit_files:
    with np.load(path, allow_pickle=False) as d:
        assert d["bare_matrix_element_raw_diagnostic"].shape == (2, 3, 21), path
        assert d["candidate_boot_values_raw"].shape == (2, 4, 100, 3, 21), path
        assert np.array_equal(ref, d["boot_indices"]), path
print(json.dumps({"ratio_files": len(ratio_files), "fit_files": len(fit_files), "shared_bootstrap_shape": list(ref.shape), "status": "passed"}, indent=2))
