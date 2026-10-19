#!/usr/bin/env python3
"""Index the common projected C2/C3 products without rewriting correlators."""

import json
import os
from pathlib import Path
import zipfile

import numpy as np


HERE = Path(__file__).resolve().parent
TAG = "thin_tf0p50000000_eps0p01000000_tw4_Nv400_dz21"
OLD_ROOT = Path(
    "/public/group/lqcd/donghx/Gradient_Flow/Result/"
    "beta6.41_mu-0.2295_ms-0.2050_L32x96/thin_link"
) / TAG / "Nev_200"
NEW_ROOT = HERE / "results_tf0p5_pilot_z" / TAG / "Nev_200"
OUTPUT_ROOT = HERE / "unified_tf0p5_pilot_z"
COMMON_TSEPS = (6, 7, 8, 9, 10)
EXPECTED = {
    "corr_2pt.npy": ((4, 3, 96, 96), np.dtype("complex128")),
    "corr_3pt_u.npy": ((2, 4, 3, 21, 4, 4, 96, 96), np.dtype("complex128")),
    "corr_3pt_d.npy": ((2, 4, 3, 21, 4, 4, 96, 96), np.dtype("complex128")),
}
AXES_C2 = ["pol", "mom", "t_source", "t_sink"]
AXES_C3 = [
    "vdv_sign", "pol", "mom", "delta_z", "current_spin_m",
    "current_spin_o", "t_source", "t_current",
]


def validate_schema(path):
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        if not set(EXPECTED).issubset(names):
            raise ValueError(f"missing correlator arrays in {path}")
        for name, expected in EXPECTED.items():
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                shape, _, dtype = np.lib.format._read_array_header(stream, version)
            if (shape, dtype) != expected:
                raise ValueError(f"{path}:{name}: {(shape, dtype)} != {expected}")
    with np.load(path, allow_pickle=False) as arrays:
        checks = {
            "flow_time": 0.5,
            "flow_epsilon": 0.01,
            "flow_n_steps": 50,
            "flow_temporal_window": 4,
            "flow_gauge_input_scheme": "thin",
            "flow_gauge_pre_smearing": "none",
            "mom_labels": ["Pp3_z", "Pp4_z", "Pp5_z"],
            "pol_labels": ["nopol", "pol15", "pol25", "pol35"],
            "axes_corr_2pt": AXES_C2,
            "axes_corr_3pt": AXES_C3,
        }
        for key, expected in checks.items():
            if key not in arrays or arrays[key].tolist() != expected:
                raise ValueError(f"{path}:{key} differs from {expected}")
        return str(arrays["flowed_peram_storage"].item())


def main():
    inventory = json.loads((HERE / "status" / "inventory20.json").read_text())
    old = [row["conf"] for row in inventory["existing_configurations"]]
    new = [row["conf"] for row in inventory["new_configurations"]]
    if len(old) != 10 or len(new) != 10 or len(set(old + new)) != 20:
        raise ValueError("inventory must contain ten old and ten new unique configurations")
    if old + new != inventory["target_confs"]:
        raise ValueError("inventory configuration order changed")

    entries = []
    for conf in old + new:
        cohort = "original_full_spin" if conf in old else "new_spin2"
        source_dir = (OLD_ROOT if conf in old else NEW_ROOT) / conf
        sources = [source_dir / f"corr_projected_tsep{t}_dz0-20_Nev200.npz"
                   for t in COMMON_TSEPS]
        if any(not p.is_file() or p.stat().st_size <= 0 for p in sources):
            raise FileNotFoundError(f"missing/empty common-tsep product: {conf}")
        storage = validate_schema(sources[0])
        entries.append({
            "conf": conf,
            "cohort": cohort,
            "flowed_peram_storage": storage,
            "source_dir": str(source_dir),
            "files": {str(t): {"source": str(p), "bytes": p.stat().st_size}
                      for t, p in zip(COMMON_TSEPS, sources)},
        })

    destination = OUTPUT_ROOT / TAG / "Nev_200"
    for entry in entries:
        conf_dir = destination / entry["conf"]
        conf_dir.mkdir(parents=True, exist_ok=True)
        for tsep, record in entry["files"].items():
            link = conf_dir / f"corr_projected_tsep{tsep}_dz0-20_Nev200.npz"
            source = Path(record["source"])
            if link.is_symlink():
                if link.resolve(strict=True) != source.resolve(strict=True):
                    raise FileExistsError(f"conflicting link: {link}")
            elif link.exists():
                raise FileExistsError(f"refusing to replace an existing file: {link}")
            else:
                link.symlink_to(source)

    manifest = {
        "schema": "quark_flow_projected_c2c3_unified20_v1",
        "meaning": "common projected outputs; external 4-to-2 spin reduction is an internal contraction optimization",
        "flow_time_lattice_units": 0.5,
        "common_tseps": list(COMMON_TSEPS),
        "axes_corr_2pt": AXES_C2,
        "axes_corr_3pt": AXES_C3,
        "correlator_dtype": "complex128",
        "links_are_copies": False,
        "entries": entries,
    }
    target = OUTPUT_ROOT / "manifest.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, target)
    print(f"unified {len(entries)} configurations, {len(entries)*len(COMMON_TSEPS)} links: {destination}")


if __name__ == "__main__":
    main()
