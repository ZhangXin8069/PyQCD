#!/usr/bin/env python3
"""Freeze a shared configuration intersection across flow times and C2 channels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DEFAULT_OPE_ROOT = Path(
    "/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/output_v5"
)
DEFAULT_TWOPT_ROOT = Path(
    "/public/group/lqcd/donghx/2pt_Result/"
    "beta6.41_mu-0.2295_ms-0.2050_L32x96"
)
DEFAULT_SOURCE_MANIFEST = Path(
    "/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/manifest.tsv"
)
DEFAULT_TAUS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8)


def parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.replace(" ", ",").split(",") if item.strip()]


def tau_tag(tau: float) -> str:
    return f"tau{tau:.3f}".replace(".", "p")


def ope_pair(root: Path, conf: str, tau: float, epsilon: float, z_dir: int) -> bool:
    tag = tau_tag(tau)
    base = root / f"conf{conf}" / tag / f"conf{conf}_{tag}_eps{epsilon:.3f}_zdir{z_dir}"
    npy = Path(str(base) + ".npy")
    metadata_path = Path(str(base) + ".json")
    if not npy.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return (
            metadata.get("schema") == "gradient_flow_gluon_ope_v2"
            and metadata.get("input_scheme") == "thin_link_no_hyp_no_smear"
            and str(metadata.get("conf_id")) == conf
            and abs(float(metadata["flow"]["tau_t_over_a2"]) - tau) < 1e-12
            and abs(float(metadata["flow"]["epsilon"]) - epsilon) < 1e-12
            and int(metadata["operator"]["z_dir"]) == z_dir
            and tuple(metadata.get("shape", ())) == (2, 2, 4, 6, 25, 96)
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def twopt_complete(
    root: Path,
    conf: str,
    direction: str,
    momenta: list[int],
    mom_phase_abs: int,
    element: str,
) -> bool:
    sign = +1 if direction == "+z" else -1
    phase = sign * mom_phase_abs
    for momentum in momenta:
        for polarization in ("nopol", "pol35"):
            path = (
                root
                / f"momsmear{phase}z"
                / conf
                / (
                    f"twopt_slice_pp_Px0Py0Pz{sign * momentum}_eginphase{phase}"
                    f"{element}_{polarization}_ss_conf{conf}.npy"
                )
            )
            if not path.is_file():
                return False
            try:
                data = np.squeeze(np.asarray(np.load(path, allow_pickle=False)))
                if (
                    data.shape != (96, 96)
                    or data.dtype.kind != "c"
                    or not np.isfinite(data).all()
                ):
                    return False
            except (OSError, ValueError):
                return False
    return True


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--ope-root", type=Path, default=DEFAULT_OPE_ROOT)
    parser.add_argument("--twopt-root", type=Path, default=DEFAULT_TWOPT_ROOT)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--taus", default=",".join(map(str, DEFAULT_TAUS)))
    parser.add_argument("--epsilon", default=0.01, type=float)
    parser.add_argument("--z-dir", default=2, type=int)
    parser.add_argument("--directions", default="+z")
    parser.add_argument("--pabs-list", default="3,4,5")
    parser.add_argument("--mom-phase-abs", default=2, type=int)
    parser.add_argument("--element", default="_Cg5g4")
    args = parser.parse_args()

    taus = [float(value) for value in parse_csv(args.taus)]
    directions = parse_csv(args.directions)
    momenta = [int(value) for value in parse_csv(args.pabs_list)]
    if not taus or len(taus) != len(set(taus)):
        raise ValueError("taus must be a non-empty unique list")
    if not directions or any(item not in ("+z", "-z") for item in directions):
        raise ValueError("directions must contain only +z/-z")
    if not momenta or min(momenta) <= 0 or len(momenta) != len(set(momenta)):
        raise ValueError("pabs-list must contain unique positive integers")

    source_bytes = args.source_manifest.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    rows = list(csv.DictReader(source_bytes.decode("utf-8").splitlines(), delimiter="\t"))
    confs = [str(row["conf_id"]) for row in rows]
    usable: list[str] = []
    rejected_ope = 0
    rejected_c2 = 0
    for index, conf in enumerate(confs):
        if index % 100 == 0:
            print(f"CHECK {index}/{len(confs)}", flush=True)
        if not all(
            ope_pair(args.ope_root, conf, tau, args.epsilon, args.z_dir) for tau in taus
        ):
            rejected_ope += 1
            continue
        if not all(
            twopt_complete(
                args.twopt_root, conf, direction, momenta,
                args.mom_phase_abs, args.element,
            )
            for direction in directions
        ):
            rejected_c2 += 1
            continue
        usable.append(conf)
    if len(usable) < 2:
        raise RuntimeError(f"Only {len(usable)} common configurations; refusing manifest")

    lines = ["task_index\tconf_id"]
    lines.extend(f"{index}\t{conf}" for index, conf in enumerate(usable))
    content = ("\n".join(lines) + "\n").encode("utf-8")
    manifest_sha256 = hashlib.sha256(content).hexdigest()
    atomic_write(args.output, content)
    contract = {
        "schema": "gradient_flow_gluon_ratio_manifest_v1",
        "status": "frozen_shared_configuration_intersection",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": source_sha256,
        "ope_root": str(args.ope_root.resolve()),
        "twopt_root": str(args.twopt_root.resolve()),
        "taus_t_over_a2": taus,
        "epsilon": args.epsilon,
        "z_dir": args.z_dir,
        "directions": directions,
        "pabs_list": momenta,
        "polarizations": ["nopol", "pol35"],
        "mom_phase_abs": args.mom_phase_abs,
        "element": args.element,
        "n_source_manifest": len(confs),
        "n_common": len(usable),
        "n_rejected_missing_ope": rejected_ope,
        "n_rejected_missing_c2_after_ope": rejected_c2,
        "manifest_sha256": manifest_sha256,
    }
    sidecar = Path(str(args.output) + ".json")
    atomic_write(sidecar, (json.dumps(contract, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(f"MANIFEST_OK n_common={len(usable)} sha256={manifest_sha256} output={args.output}")


if __name__ == "__main__":
    main()
