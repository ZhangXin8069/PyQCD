"""Audit finite-b TMD ratio inputs without modifying the source run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from pyqcd.operator import (
    GRADIENT_FLOW_CHANNELS,
    GRADIENT_FLOW_ORIENTATIONS,
    GRADIENT_FLOW_COMPONENTS,
)
from pyqcd.pipeline._tmd9 import (
    _FULL_OPE_ALGORITHM_VERSION,
    _POLARIZED_2PT_ALGORITHM_VERSION,
    TWO_POINT_POLARIZATIONS,
)


AUDIT_SCHEMA = "pyqcd.finite_b_ratio_input_audit.v2"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _h5_summary(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        if set(handle.keys()) != {"data"}:
            raise ValueError(
                f"HDF5 must contain only the data dataset: {path}")
        data = handle["data"]
        raw_contract = handle.attrs.get("pyqcd_cache_contract_json")
        if raw_contract is None:
            raise ValueError(f"HDF5 cache contract is missing: {path}")
        contract = json.loads(raw_contract)
        return {
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
            "shape": list(data.shape),
            "dtype": str(data.dtype),
            "contract": contract,
        }


def _source_resolved_c2(summary: dict) -> bool:
    shape = summary["shape"]
    contract = summary["contract"]
    return (
        len(shape) == 2
        and shape[0] == shape[1]
        and contract.get("shape") == shape
        and summary["dtype"] == "complex128"
        and contract.get("artifact") == "tmd9_multi_2pt_polarized"
        and contract.get("algorithm_version")
        == _POLARIZED_2PT_ALGORITHM_VERSION
        and contract.get("source_axis_order") == "source,dt"
        and contract.get("time_axis_order") == "source,sink"
        and contract.get("antiperiodic_boundary") is True
        and contract.get("dirac_interpolator") == "Cg5g4"
        and contract.get("polarization") in TWO_POINT_POLARIZATIONS
    )


def _full_channel_tmd_ope(summary: dict) -> bool:
    shape = summary["shape"]
    contract = summary["contract"]
    return (
        len(shape) == 6
        and tuple(shape[:3]) == (2, 4, 6)
        and contract.get("shape") == shape
        and contract.get("artifact") == "tmd_ope_time_full_channel"
        and contract.get("algorithm_version") == _FULL_OPE_ALGORITHM_VERSION
        and contract.get("channel_labels") == list(GRADIENT_FLOW_CHANNELS)
        and contract.get("orientation_labels")
        == list(GRADIENT_FLOW_ORIENTATIONS)
        and contract.get("component_labels") == list(GRADIENT_FLOW_COMPONENTS)
    )


def audit_finite_b_ratio_inputs(run_dir, conf_ids=None) -> dict:
    """Return an availability report for finite-b ratio-v2 inputs."""
    root = Path(run_dir)
    data_root = root / "data"
    if conf_ids is None:
        discovered = []
        if data_root.is_dir():
            for path in data_root.glob("conf*"):
                suffix = path.name[4:]
                if path.is_dir() and suffix.isdigit():
                    discovered.append(int(suffix))
        conf_ids = sorted(discovered)
    else:
        conf_ids = list(conf_ids)
    if any(isinstance(value, (bool, np.bool_))
           or not isinstance(value, (int, np.integer))
           for value in conf_ids):
        raise ValueError("conf_ids must contain non-boolean integers")
    conf_ids = [int(value) for value in conf_ids]
    if len(set(conf_ids)) != len(conf_ids):
        raise ValueError("conf_ids must not contain duplicates")

    per_configuration = {}
    missing = []
    c2_states = []
    ope_states = []
    for conf_id in conf_ids:
        conf_dir = data_root / f"conf{conf_id}"
        if not conf_dir.is_dir():
            missing.append(conf_id)
            per_configuration[str(conf_id)] = {
                "present": False,
                "source_resolved_c2": False,
                "c2_polarizations": [],
                "full_channel_tmd_ope": False,
                "multi_2pt": [],
                "tmd_ope": [],
            }
            c2_states.append(False)
            ope_states.append(False)
            continue
        files = sorted(conf_dir.glob("*.h5"))
        c2 = [_h5_summary(path) for path in files
              if path.name.startswith("corr_pp_")]
        ope = [_h5_summary(path) for path in files
               if path.name.startswith("tmd_ope_")]
        valid_c2 = [item for item in c2 if _source_resolved_c2(item)]
        polarization_coverage = {}
        for item in valid_c2:
            contract = item["contract"]
            signature = (
                contract.get("channel"), tuple(contract.get("momentum", ())))
            polarization_coverage.setdefault(signature, set()).add(
                contract["polarization"])
        source_resolved = any(
            values == set(TWO_POINT_POLARIZATIONS)
            for values in polarization_coverage.values())
        c2_polarizations = sorted({
            contract["polarization"] for contract in (
                item["contract"] for item in valid_c2)
        })
        full_channel_ope = [
            item for item in ope
            if item["contract"].get("artifact")
            == "tmd_ope_time_full_channel"
        ]
        # Legacy and full-channel products may coexist in a production run.
        # Only the explicitly selected artifact participates in this gate.
        full_channel = bool(full_channel_ope) and all(
            _full_channel_tmd_ope(item) for item in full_channel_ope)
        c2_states.append(source_resolved)
        ope_states.append(full_channel)
        per_configuration[str(conf_id)] = {
            "present": True,
            "source_resolved_c2": source_resolved,
            "c2_polarizations": c2_polarizations,
            "full_channel_tmd_ope": full_channel,
            "multi_2pt": c2,
            "tmd_ope": ope,
        }

    reasons = []
    if len(conf_ids) < 2:
        reasons.append("fewer_than_two_configurations")
    if missing:
        reasons.append("missing_configuration_data")
    if not c2_states or not all(c2_states):
        reasons.append("source_resolved_c2_missing")
    if not ope_states or not all(ope_states):
        reasons.append("full_channel_orientation_component_tmd_ope_missing")

    ratio_outputs = []
    ratio_dir = root / "analysis" / "tmd_ratio"
    if ratio_dir.is_dir():
        for path in sorted(ratio_dir.glob("*.npy")):
            value = np.load(path, allow_pickle=False)
            ratio_outputs.append({
                "path": str(path.resolve()),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "finite_count": int(np.isfinite(value).sum()),
                "size": int(value.size),
            })

    return {
        "schema": AUDIT_SCHEMA,
        "run_dir": str(root.resolve()),
        "configurations": conf_ids,
        "nconf": len(conf_ids),
        "missing_configurations": missing,
        "source_resolved_c2": bool(c2_states) and all(c2_states),
        "full_channel_tmd_ope": bool(ope_states) and all(ope_states),
        "per_configuration": per_configuration,
        "existing_ratio_outputs": ratio_outputs,
        "can_run_finite_b_ratio_v2": not reasons,
        "status": "ready" if not reasons else "blocked",
        "blocking_reasons": reasons,
        "scope": (
            "Input structure availability only; ready still requires verified "
            "nopol/pol35 channel coverage, t_sep, covariance and physical "
            "acceptance gates."
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--conf-ids",
        default=None,
        help="comma-separated configuration IDs; default: discover conf* dirs",
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    conf_ids = None
    if args.conf_ids:
        conf_ids = [int(value) for value in args.conf_ids.split(",") if value]
    report = audit_finite_b_ratio_inputs(args.run_dir, conf_ids)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        print(text, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(args.out.resolve())
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
