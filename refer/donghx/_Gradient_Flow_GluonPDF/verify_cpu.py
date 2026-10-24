"""Independent post-completion readback and PDF previews, CPU allocation only."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np

HERE=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Verification and rendering require CPU allocation')
    report=json.loads((HERE/'complete.json').read_text())
    samples=points=0
    for item in report['artifacts']:
        path=Path(item['path'])
        assert sha(path)==item['sha256'],str(path)
        if path.suffix!='.npz':
            continue
        with np.load(path,allow_pickle=False) as d:
            m=d['bare_M_selected']
            b=d['bare_M_boot_selected']
            with np.errstate(invalid='ignore',divide='ignore'):
                oracle=m/m[0,:,0][None,:,None]
                oracle_boot=b/b[0,:,:,0][None,:,:,None]
            oracle[~np.isfinite(oracle)]=np.nan
            oracle_boot[~np.isfinite(oracle_boot)]=np.nan
            np.testing.assert_array_equal(d['H_local'],oracle)
            np.testing.assert_array_equal(d['H_local_boot'],oracle_boot)
            available=np.isfinite(oracle)&np.isfinite(oracle_boot).all(axis=1)
            np.testing.assert_array_equal(d['selected_available'],available)
            errors=np.std(oracle_boot,axis=1,ddof=1)
            errors[~available]=np.nan
            np.testing.assert_array_equal(d['statistical_error'],errors)
            np.testing.assert_array_equal(d['local_anchor_constraint'][0,:,0],True)
            np.testing.assert_array_equal(d['local_anchor_constraint'][1],False)
            assert not d['line_mass_subtraction_applied']
            assert not d['SFTE_applied'] and not d['LaMET_matching_applied']
            samples+=int(oracle_boot.size)
            points+=int(oracle.size)
    assert points==report['points']==14742
    preview=HERE/'previews'
    preview.mkdir(exist_ok=True)
    renderer=shutil.which('pdftoppm')
    if renderer is None:
        raise RuntimeError('pdftoppm is needed for PDF layout review')
    previews=[]
    for rel in ('plots/moving/localnorm_unpolarized_u_minus_d.pdf',
                'plots/moving/localnorm_helicity_u_minus_d.pdf',
                'plots/Pz0/localnorm_transversity_u_minus_d.pdf',
                'plots/local_denominators_helicity.pdf'):
        source=HERE/rel
        output=preview/(source.stem+('_Pz0' if '/Pz0/' in rel else ''))
        subprocess.run([renderer,'-f','1','-singlefile','-scale-to','1500','-png',str(source),str(output)],check=True)
        previews.append(str(output.with_suffix('.png')))
    result=dict(status='PASS_COMPLETE_INDEPENDENT_READBACK_AND_PDF_RENDER',
        generated=datetime.now().astimezone().isoformat(),job=os.environ['SLURM_JOB_ID'],
        node=os.environ.get('SLURMD_NODENAME'),complete_sha256=sha(HERE/'complete.json'),
        normalized_points_checked=points,normalized_replica_values_checked=samples,
        artifacts_checked=len(report['artifacts']),previews=previews,
        visual_review_status='rendered_for_separate_visual_review',
        verification_code_sha256=sha(__file__))
    (HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
