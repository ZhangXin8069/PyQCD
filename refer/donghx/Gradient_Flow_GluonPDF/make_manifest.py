#!/usr/bin/env python3
"""Create a stable, top-level-only manifest of a thin-link L32xNx ensemble."""

from __future__ import annotations

import argparse
import os
import re
import struct
from pathlib import Path


PATTERN = re.compile(r"_cfg_(\d+)\.lime$")
LIME_HEADER = struct.Struct(">IHHQ128s")
BYTES_PER_TIMESLICE = 32**3 * 4 * 3 * 3 * 2 * 8


def ildg_payload(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        while True:
            raw = stream.read(LIME_HEADER.size)
            if not raw:
                break
            if len(raw) != LIME_HEADER.size:
                raise ValueError("truncated LIME header")
            magic, _version, _flags, size, kind = LIME_HEADER.unpack(raw)
            if magic != 0x456789AB:
                raise ValueError(f"bad LIME magic 0x{magic:08x}")
            kind_text = kind.split(b"\0", 1)[0].decode("ascii", errors="replace")
            offset = stream.tell()
            if kind_text == "ildg-binary-data":
                return offset, size
            stream.seek((size + 7) // 8 * 8, 1)
    raise ValueError("missing ildg-binary-data record")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gauge-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--nt", default=96, type=int,
                        help="temporal extent; defaults to the historical L32x96 value")
    args = parser.parse_args()
    if args.nt <= 0:
        raise SystemExit("--nt must be positive")
    expected_payload_bytes = args.nt * BYTES_PER_TIMESLICE

    rows: list[tuple[int, Path]] = []
    for path in args.gauge_root.glob("*.lime"):
        match = PATTERN.search(path.name)
        if match:
            rows.append((int(match.group(1)), path.resolve()))
    rows.sort()
    if not rows:
        raise SystemExit(f"no top-level *.lime configurations found in {args.gauge_root}")
    if len({conf for conf, _ in rows}) != len(rows):
        raise SystemExit("duplicate configuration IDs in gauge manifest")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write("task_index\tconf_id\tgauge_path\tlime_payload_offset\tlime_payload_bytes\n")
        for index, (conf, path) in enumerate(rows):
            try:
                offset, size = ildg_payload(path)
            except Exception as error:
                raise SystemExit(f"invalid LIME file {path}: {error}") from error
            if size != expected_payload_bytes:
                raise SystemExit(
                    f"wrong L32x{args.nt} payload size for {path}: {size} != {expected_payload_bytes}"
                )
            stream.write(f"{index}\t{conf}\t{path}\t{offset}\t{size}\n")
    temporary.replace(args.output)
    print(f"MANIFEST_OK count={len(rows)} first={rows[0][0]} last={rows[-1][0]}")


if __name__ == "__main__":
    main()
