#!/usr/bin/env python3
"""Record live OPE/C2 coverage and the frozen common-manifest statistics."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OPE_ROOT = ROOT / "output_v5"
TWOPT_ROOT = Path(
    "/public/group/lqcd/donghx/2pt_Result/"
    "beta6.41_mu-0.2295_ms-0.2050_L32x96"
)
SOURCE = ROOT / "manifest.tsv"
FROZEN = Path(__file__).resolve().parent / "manifests/common_all14_plusz_P345.tsv"
TAUS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def confs(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as stream:
        return [row["conf_id"] for row in csv.DictReader(stream, delimiter="\t")]


def tag(tau: float) -> str:
    return f"tau{tau:.3f}".replace(".", "p")


def ope_pair(conf: str, tau: float) -> bool:
    stem = OPE_ROOT / f"conf{conf}" / tag(tau) / (
        f"conf{conf}_{tag(tau)}_eps0.010_zdir2"
    )
    return Path(str(stem) + ".npy").is_file() and Path(str(stem) + ".json").is_file()


def c2_complete(conf: str) -> bool:
    for momentum in (3, 4, 5):
        for polarization in ("nopol", "pol35"):
            path = TWOPT_ROOT / "momsmear2z" / conf / (
                f"twopt_slice_pp_Px0Py0Pz{momentum}_eginphase2_Cg5g4_"
                f"{polarization}_ss_conf{conf}.npy"
            )
            if not path.is_file():
                return False
    return True


def main() -> None:
    source = confs(SOURCE)
    frozen = confs(FROZEN)
    c2 = {conf for conf in source if c2_complete(conf)}
    flow_sets = [{conf for conf in source if ope_pair(conf, tau)} for tau in TAUS]
    common = set.intersection(*flow_sets) & c2
    if not set(frozen).issubset(common):
        raise ValueError("a frozen-manifest member is no longer complete")
    payload = {
        "schema": "gradient_flow_gluon_live_statistics_audit_v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(SOURCE.resolve()),
        "source_manifest_sha256": digest(SOURCE),
        "n_source": len(source),
        "n_c2_complete_plusz_P345_nopol_pol35": len(c2),
        "per_tau": {
            f"{tau:.1f}": {
                "n_ope_pair_eps0p010": len(flow),
                "n_ope_and_c2_complete": len(flow & c2),
            }
            for tau, flow in zip(TAUS, flow_sets)
        },
        "n_live_common_all14_and_c2": len(common),
        "frozen_manifest": str(FROZEN.resolve()),
        "frozen_manifest_sha256": digest(FROZEN),
        "n_frozen_common": len(frozen),
        "note": "live counts can grow while OPE jobs run; production uses the immutable frozen list",
    }
    output = Path(__file__).resolve().parent / "statistics_snapshot.json"
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
