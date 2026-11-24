"""Render selected real deliverables on CPU for an explicit visual review."""
import hashlib
import json
import os
from pathlib import Path
import fitz

ROOT=Path(__file__).resolve().parent


def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('CPU allocation required for PDF rendering')
    output=ROOT/'00_audit/visual_review';output.mkdir(parents=True,exist_ok=True)
    pdfs=[
        'plots/L32x64/01_bare/unpolarized_u_minus_d_native.pdf',
        'plots/L32x96/04_flow_conversion/helicity_u_minus_d_RGR.pdf',
        'plots/L24x72/00_ratio_fit_checks/unpolarized_u_minus_d.pdf',
        'plots/continuum/07_common_kinematics/unpolarized_u_minus_d.pdf',
        'plots/continuum/08_a2_fit_checks/unpolarized_u_minus_d.pdf',
        'plots/continuum/09_flow_fit_checks/unpolarized_u_minus_d.pdf',
        'plots/continuum/09_continuum_tau0/helicity_u_minus_d.pdf',
        'plots/continuum/10_continuum_m0_corrected/transversity_u_minus_d.pdf']
    rows=[]
    for i,rel in enumerate(pdfs):
        path=ROOT/rel
        with fitz.open(path) as doc:
            pages=sorted(set([0,min(2,doc.page_count-1)]))
            for page in pages:
                out=output/f'{i:02d}_page{page+1:03d}.png'
                doc.load_page(page).get_pixmap(matrix=fitz.Matrix(1.2,1.2),alpha=False).save(out)
                rows.append(dict(pdf=str(path),page=page+1,pages_total=doc.page_count,png=str(out)))
    (output/'rendered.json').write_text(json.dumps(dict(status='RENDERED_FOR_VISUAL_INSPECTION_NOT_A_PHYSICS_GATE',
        job=os.environ['SLURM_JOB_ID'],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),pages=rows),indent=2)+'\n')
    print('REAL_PDF_PREVIEWS_RENDERED',len(rows),flush=True)


if __name__=='__main__':main()
