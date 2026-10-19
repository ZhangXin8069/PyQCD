#!/usr/bin/env python3
"""Delete only reconstructible L32x96 vertex caches; retain basis and gauge."""

import argparse
import os
from pathlib import Path
import re
import shutil


ROOT = Path("/public/group/lqcd/donghx/Gradient_Flow/Blending_L32x96/Result")
CONF = re.compile(r"10[0-9]00")
FLOW = re.compile(r"(?:thin_)?tf[0-9p]+_eps[0-9p]+_tw[0-9]+_Nv400_dz21")


def targets():
    if ROOT.is_symlink() or not ROOT.is_dir():
        raise RuntimeError(f"Unsafe result root: {ROOT}")
    found = []
    for conf in sorted(ROOT.iterdir()):
        if not conf.is_dir() or conf.is_symlink():
            raise RuntimeError(f"Unexpected configuration entry: {conf}")
        if not CONF.fullmatch(conf.name):
            raise RuntimeError(f"Unexpected configuration name: {conf}")
        for flow in sorted(conf.iterdir()):
            if not flow.is_dir() or flow.is_symlink():
                raise RuntimeError(f"Unexpected flow entry: {flow}")
            vertex = flow / "vertex"
            if not vertex.exists() and not vertex.is_symlink():
                continue
            if not FLOW.fullmatch(flow.name) or not vertex.is_dir() or vertex.is_symlink():
                raise RuntimeError(f"Unsafe vertex target: {vertex}")
            basis = flow / "basis"
            gauge = flow / "flowed_gauge"
            if basis.is_symlink() or gauge.is_symlink():
                raise RuntimeError(f"Symlink in reconstruction inputs: {flow}")
            if not basis.is_dir() or not (gauge / "index.json").is_file():
                raise RuntimeError(f"Missing reconstruction inputs: {flow}")
            missing = [tau for tau in range(96) if not (basis / f"tau{tau:03d}" / "done.json").is_file()]
            if missing:
                raise RuntimeError(f"Incomplete basis ({len(missing)} tau): {basis}")
            found.append(vertex)
    return found


def inventory(root):
    size = 0
    files = 0
    for directory, _, names in os.walk(root, followlinks=False):
        for name in names:
            path = os.path.join(directory, name)
            size += os.lstat(path).st_size
            files += 1
    return files, size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="irreversibly delete checked vertex directories")
    args = parser.parse_args()
    found = targets()
    if not found:
        if not args.execute:
            print(f"root={ROOT} vertex_directories=0 execute=False", flush=True)
        return
    if not args.execute:
        print(f"root={ROOT} vertex_directories={len(found)} execute=False", flush=True)
        for vertex in found:
            print(f"TARGET {vertex}", flush=True)
        return
    removed_bytes = 0
    for vertex in found:
        files, size = inventory(vertex)
        shutil.rmtree(vertex)
        if vertex.exists() or not (vertex.parent / "basis").is_dir() or not (vertex.parent / "flowed_gauge" / "index.json").is_file():
            raise RuntimeError(f"Post-delete validation failed: {vertex}")
        removed_bytes += size
        print(f"DELETE_DONE path={vertex} files={files} bytes={size}", flush=True)
    print(f"COMPLETE directories={len(found)} bytes={removed_bytes}", flush=True)


if __name__ == "__main__":
    main()
