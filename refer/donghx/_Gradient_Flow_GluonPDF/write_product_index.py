#!/usr/bin/env python3
from pathlib import Path
import hashlib, json

ROOT = Path(__file__).resolve().parent
def digest(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()

def main() -> int:
    products = []
    for kind, directory in (("ratio", ROOT / "ratios"), ("fit", ROOT / "fits"), ("plot", ROOT / "plots")):
        for p in sorted(directory.glob("*")) if directory.is_dir() else []:
            if p.is_file(): products.append({"kind": kind, "path": str(p), "size": p.stat().st_size, "sha256": digest(p)})
    payload = {"analysis_root": str(ROOT), "product_count": len(products), "products": products, "bootstrap_indices": str(ROOT / "bootstrap_indices_Nboot100_seed1115.npy"), "note": "Current completed flow times are 0.05, 0.10, 0.20, 0.50, 0.80; tau=1.00 waits for conf 12900/13000 production."}
    (ROOT / "product_index.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"product_count": len(products), "output": str(ROOT / "product_index.json")}, indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())
