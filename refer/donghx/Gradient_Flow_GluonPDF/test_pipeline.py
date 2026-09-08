#!/usr/bin/env python3
"""End-to-end tiny LIME -> Wilson flow -> OPE -> completion-guard test."""

from __future__ import annotations

import json
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]


def write_identity_lime(path: Path) -> None:
    nt = nx = 2
    gauge = np.zeros((nt, nx, nx, nx, 4, 3, 3, 2), dtype=">f8")
    for color in range(3):
        gauge[..., color, color, 0] = 1.0
    payload = gauge.tobytes(order="C")
    record_type = b"ildg-binary-data".ljust(128, b"\0")
    header = struct.pack(">IHHQ128s", 0x456789AB, 1, 3, len(payload), record_type)
    path.write_bytes(header + payload + b"\0" * ((-len(payload)) % 8))


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="gf_ope_test.") as temporary:
        root = Path(temporary)
        lime = root / "identity.lime"
        base = root / "out" / "conf1_tau0p020_eps0.010_zdir2"
        write_identity_lime(lime)
        subprocess.run(
            [
                str(REPO / "build/flowed_gluon_ope_v5"),
                "--input", str(lime), "--conf-id", "1", "--output", str(base),
                "--nx", "2", "--ny", "2", "--nz", "2", "--nt", "2",
                "--tau", "0.02", "--epsilon", "0.01", "--z-count", "2", "--z-dir", "2",
            ],
            check=True,
        )
        subprocess.run(
            [
                "python", str(REPO / "scripts/validate_output_v5.py"), str(base),
                "--conf-id", "1", "--z-count", "2", "--nt", "2", "--tau", "0.02",
                "--epsilon", "0.01",
            ],
            check=True,
        )
        data = np.load(Path(str(base) + ".npy"), allow_pickle=False)
        metadata = json.loads(Path(str(base) + ".json").read_text())
        assert float(np.max(np.abs(data))) < 1e-12
        assert data.shape == (2, 2, 4, 6, 2, 2)
        assert metadata["schema"] == "gradient_flow_gluon_ope_v2"
        assert metadata["lime_payload_bytes"] == 2**4 * 4 * 3 * 3 * 2 * 8
        assert metadata["raw_gauge_stats"]["plaquette"] == 1
        assert metadata["flowed_gauge_stats"]["plaquette"] == 1
    print("PIPELINE_TEST_OK")


if __name__ == "__main__":
    main()
