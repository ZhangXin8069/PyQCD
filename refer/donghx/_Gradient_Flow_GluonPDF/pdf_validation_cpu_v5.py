"""Read all final PDFs and render representative pages using a private wheel."""

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
PLOT_ROOT = ROOT / "plot_repair_v2_20261108"
TOOLS = ROOT / "tools/pdf_preview_pymupdf_1p24p14"
WHEEL = ROOT / "tools/pdf_preview_wheels/PyMuPDF-1.24.14-cp39-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"
WHEEL_SHA256 = "0de4f5ed903c2be6d0abcccdc796368939b51ce03916eb53292916e3b6ea65d6"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("PDF readback/rendering requires a CPU allocation")
    assert digest(WHEEL) == WHEEL_SHA256
    sys.path.insert(0, str(TOOLS))
    import fitz
    assert fitz.VersionBind == "1.24.14"
    completion = ROOT / "complete.json"
    record = json.loads(completion.read_text())
    assert record["status"] == "COMPLETE_THREE_ENSEMBLE_BOOT200_EXPLORATORY_CONTINUUM"
    assert record["repaired_plot_root"] == str(PLOT_ROOT)
    out = PLOT_ROOT / "00_audit/visual_review_v5"
    out.mkdir(exist_ok=True)
    pages = {}
    sources = []
    for task in range(9):
        receipt = PLOT_ROOT / "00_audit" / f"plots_{task}.json"
        sources.append(dict(path=str(receipt), sha256=digest(receipt)))
        for entry in json.loads(receipt.read_text())["artifacts"]:
            path = Path(entry["path"])
            assert path.is_relative_to(PLOT_ROOT / "plots")
            assert digest(path) == entry["sha256"]
            with fitz.open(path) as document:
                assert document.is_pdf and not document.is_encrypted
                assert document.page_count > 0
                for number in sorted({0, document.page_count - 1}):
                    assert len(document.load_page(number).get_text().strip()) > 10
                assert str(path) not in pages, f"Duplicate PDF receipt: {path}"
                pages[str(path)] = document.page_count
            warnings = fitz.TOOLS.mupdf_warnings(reset=True)
            if warnings.strip():
                raise RuntimeError(f"MuPDF parser warnings for {path}: {warnings}")
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
        with fitz.open(path) as document:
            for page in sorted({0, min(2, document.page_count - 1)}):
                png = out / f"{i:02d}_page{page+1:03d}.png"
                document.load_page(page).get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).save(png)
                assert png.stat().st_size > 1000
                rendered.append(dict(
                    pdf=str(path), page=page + 1, pages_total=document.page_count,
                    png=str(png), png_sha256=digest(png),
                ))
        warnings = fitz.TOOLS.mupdf_warnings(reset=True)
        if warnings.strip():
            raise RuntimeError(f"MuPDF render warnings for {path}: {warnings}")
    artifact = dict(
        status="PASS_ALL_PDF_READBACK_AND_PREVIEWS_RENDERED",
        timestamp=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        job=os.environ["SLURM_JOB_ID"], script_sha256=digest(Path(__file__)),
        wheel_sha256=WHEEL_SHA256, fitz_version=fitz.VersionBind,
        complete_sha256=digest(completion), receipt_sha256=sources,
        pdf_count=len(pages), total_pdf_pages=sum(pages.values()), pdf_pages=pages,
        previews=rendered, visual_human_review="pending",
        physics_validated=False, controlled_continuum_limit=False,
        replaces_failed_preview_jobs=["3955055", "3955091"],
        numerical_products_modified=False,
    )
    (out / "validation.json").write_text(json.dumps(artifact, indent=2) + "\n")
    print("PDF_READBACK_COMPLETE", len(pages), "PDFS", sum(pages.values()),
          "PAGES", len(rendered), "PREVIEWS", flush=True)


if __name__ == "__main__":
    main()
