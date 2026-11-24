"""CPU Poppler PDF readback/preview successor; no PyMuPDF dependency."""

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
PLOT_ROOT = ROOT / "plot_repair_v2_20261108"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("PDF readback/rendering requires a CPU allocation")
    for tool in ("pdfinfo", "pdftoppm"):
        if not shutil.which(tool):
            raise RuntimeError(f"Required system PDF tool unavailable: {tool}")
    completion = ROOT / "complete.json"
    record = json.loads(completion.read_text())
    assert record["status"] == "COMPLETE_THREE_ENSEMBLE_BOOT200_EXPLORATORY_CONTINUUM"
    assert record["repaired_plot_root"] == str(PLOT_ROOT)
    out = PLOT_ROOT / "00_audit/visual_review_v4"
    out.mkdir(exist_ok=True)
    pages = {}
    sources = []
    for task in range(9):
        receipt = PLOT_ROOT / "00_audit" / f"plots_{task}.json"
        sources.append(dict(path=str(receipt), sha256=digest(receipt)))
        entries = json.loads(receipt.read_text())["artifacts"]
        for entry in entries:
            path = Path(entry["path"])
            assert path.is_relative_to(PLOT_ROOT / "plots")
            assert digest(path) == entry["sha256"]
            result = subprocess.run(
                ["pdfinfo", str(path)], capture_output=True, text=True, check=True,
                timeout=60,
            )
            match = re.search(r"^Pages:\s+(\d+)\s*$", result.stdout, re.M)
            if not match or int(match[1]) < 1:
                raise RuntimeError(f"No readable PDF pages: {path}")
            if result.stderr.strip():
                raise RuntimeError(f"PDF parser warnings/errors for {path}: {result.stderr}")
            assert str(path) not in pages, f"Duplicate PDF receipt: {path}"
            pages[str(path)] = int(match[1])
    assert len(pages) == record["pdf_count"]
    previews = [
        "plots/L32x64/01_bare/unpolarized_u_minus_d_native.pdf",
        "plots/L32x96/04_flow_conversion/helicity_u_minus_d_RGR.pdf",
        "plots/L24x72/00_ratio_fit_checks/unpolarized_u_minus_d.pdf",
        "plots/continuum/07_common_kinematics/unpolarized_u_minus_d.pdf",
        "plots/continuum/08_a2_fit_checks/unpolarized_u_minus_d.pdf",
        "plots/continuum/09_flow_fit_checks/unpolarized_u_minus_d.pdf",
        "plots/continuum/09_continuum_tau0/helicity_u_minus_d.pdf",
        "plots/continuum/10_continuum_m0_corrected/transversity_u_minus_d.pdf",
    ]
    rendered = []
    for i, rel in enumerate(previews):
        path = PLOT_ROOT / rel
        count = pages[str(path)]
        for number in sorted({1, min(3, count)}):
            prefix = out / f"{i:02d}_page{number:03d}"
            result = subprocess.run(
                ["pdftoppm", "-f", str(number), "-l", str(number), "-singlefile",
                 "-scale-to", "1800", "-png", str(path), str(prefix)],
                capture_output=True, text=True, check=True, timeout=90,
            )
            if result.stderr.strip():
                raise RuntimeError(f"PDF rendering warnings/errors: {path}: {result.stderr}")
            png = prefix.with_suffix(".png")
            assert png.stat().st_size > 1000
            rendered.append(dict(
                pdf=str(path), page=number, pages_total=count, png=str(png),
                png_sha256=digest(png),
            ))
    artifact = dict(
        status="PASS_POPPLER_ALL_PDF_READBACK_AND_PREVIEWS_RENDERED",
        timestamp=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        job=os.environ["SLURM_JOB_ID"], script_sha256=digest(Path(__file__)),
        complete_sha256=digest(completion), receipt_sha256=sources,
        pdf_count=len(pages), total_pdf_pages=sum(pages.values()), pdf_pages=pages,
        previews=rendered, visual_human_review="pending",
        physics_validated=False, controlled_continuum_limit=False,
        replaces_failed_preview_job="3955055; missing fitz only; numerical inputs unchanged",
    )
    (out / "validation.json").write_text(json.dumps(artifact, indent=2) + "\n")
    print("PDF_READBACK_COMPLETE", len(pages), "PDFS", sum(pages.values()),
          "PAGES", len(rendered), "PREVIEWS", flush=True)


if __name__ == "__main__":
    main()
