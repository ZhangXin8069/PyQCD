"""Independent gather/parity oracle and final artifact audit (CPU allocation)."""
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import numpy as np
from analysis_common import *


def pdf_pages(path):
    # Matplotlib writes uncompressed Page/Pages dictionaries. Validate this exact
    # producer schema, header, EOF, individual Page objects and root /Count.
    # This is not a general PDF renderer/parser; visual review is separate.
    value = Path(path).read_bytes()
    assert value.startswith(b'%PDF-') and value.rstrip().endswith(b'%%EOF')
    assert b'Matplotlib' in value
    pages = re.findall(rb'\d+ 0 obj\s*<< /Type /Page\b', value)
    counts = re.findall(rb'/Count\s+(\d+)\b', value)
    assert len(counts) == 1 and int(counts[0]) == len(pages) and len(pages)>0
    return len(pages)


def main():
    allocation(); code_check(); ind = indices()
    verification_lock = json.loads((HERE / 'verification_v3_lock.json').read_text())
    for rel, digest in verification_lock['files_sha256'].items():
        assert sha(HERE / rel) == digest, rel
    artifacts, diagnostics, c2_reference = [], [], None
    for tau in FLOWS:
        mark = json.loads((HERE / 'status' / f'ratio_{tag(tau)}.json').read_text())
        assert mark['status'] == 'PASS' and mark['new30_raw_cases_reverified'] == 7560
        assert sha(mark['projected']) == mark['projected_sha256']
        assert sha(mark['raw_ledger']) == mark['raw_ledger_sha256']
        with np.load(mark['projected'], allow_pickle=False) as d:
            c3, c2 = d['c3_source_mean'], d['c2_source_mean']
            np.testing.assert_array_equal(d['confs'], CONFS)
        mask = np.arange(13)[None, :] <= np.asarray(TSEPS)[:, None]
        if c2_reference is None: c2_reference = c2.copy()
        difference = np.linalg.norm((c2-c2_reference)[..., mask]) / np.linalg.norm(c2_reference[..., mask])
        assert difference < 2e-10, (tau, difference)
        saved = {}
        for channel in CHANNELS:
            for flavor in FLAVORS:
                path = ratio_path(tau, channel, flavor)
                expected = next(p['sha256'] for p in mark['products'] if p['path'] == str(path))
                assert sha(path) == expected
                with np.load(path, allow_pickle=False) as d:
                    np.testing.assert_array_equal(d['boot_indices'], ind)
                    r, boot = d['ratio_original'], d['ratio_boot']
                    assert boot.shape == (NBOOT, 2, 3, DZ, 6, 13)
                    assert np.isfinite(boot[..., mask]).all() and np.isnan(boot[..., ~mask]).all()
                    assert str(d['c2_mode']) == 'complex_nopol_dt_matched'
                    assert int(d['weight_applied_count']) == 1
                    err = d['component_error']
                    comp = np.stack((boot[:, 0].real, boot[:, 1].imag), axis=1)
                    np.testing.assert_allclose(np.std(comp, axis=0, ddof=1), err, equal_nan=True)
                    np.testing.assert_allclose(boot[:, 1, :, 0, ..., mask], 0, atol=1e-11)
                    saved[channel, flavor] = boot
                ch = CHANNELS.index(channel)
                for rep in (-1, 0, 17, 199):
                    cs = c3 if rep == -1 else c3[ind[rep]]
                    ds = c2 if rep == -1 else c2[ind[rep]]
                    if flavor == 'u': numerator = cs[:, ch, 0].mean(axis=0)
                    elif flavor == 'd': numerator = cs[:, ch, 1].mean(axis=0)
                    else: numerator = (cs[:, ch, 0]-cs[:, ch, 1]).mean(axis=0)
                    denominator = ds.mean(axis=0)
                    with np.errstate(invalid='ignore'):
                        raw = numerator / denominator[None, :, None]
                    # Deliberately staged rather than production's combined four-term expression.
                    ep, em = .5*(raw[0, 0::2]+raw[1, 0::2]), .5*(raw[0, 1::2]+raw[1, 1::2])
                    op, om = .5*(raw[0, 0::2]-raw[1, 0::2]), .5*(raw[0, 1::2]-raw[1, 1::2])
                    oracle = np.stack((.5*(ep+em), .5*(op-om)))
                    np.testing.assert_allclose(r if rep == -1 else boot[rep], oracle,
                                               rtol=2e-10, atol=2e-10, equal_nan=True)
                artifacts.append(dict(path=str(path), sha256=expected))
        for channel in CHANNELS:
            np.testing.assert_allclose(saved[channel, 'u_minus_d'], saved[channel, 'u']-saved[channel, 'd'],
                                       rtol=2e-10, atol=2e-10, equal_nan=True)
        pmark = json.loads((HERE / 'status' / f'plots_{tag(tau)}.json').read_text())
        assert pmark['status'] == 'PASS'
        for product in pmark['products']:
            assert sha(product['path']) == product['sha256']
            assert pdf_pages(product['path']) == product['pages']
            artifacts.append(product)
        with Path(pmark['comparison_csv']).open() as stream:
            diagnostics.extend(csv.DictReader(stream))
        print('INDEPENDENT_GATHER_AND_MASK_PASS', tau, difference, flush=True)
    cross = json.loads((HERE / 'status/cross_flow_plots.json').read_text())
    for product in cross['products']:
        assert sha(product['path']) == product['sha256']
        assert pdf_pages(product['path']) == product['pages']
        artifacts.append(product)
    # Small-z u-d paired T7/T12 evidence, with low-power counts kept separate.
    evidence = []
    for channel in CHANNELS:
        for p in PZ:
            for comp in ('Re (even-z)', 'Im (odd-z)'):
                rows = [r for r in diagnostics if r['channel']==channel and r['flavor']=='u_minus_d'
                        and int(r['p'])==p and r['component']==comp and int(r['T'])==7
                        and 1 <= int(r['z_a']) <= 6]
                pulls = [abs(float(r['paired_pull'])) for r in rows if r['paired_pull']]
                relative = [float(r['relative_difference_to_T12']) for r in rows if r['relative_difference_to_T12']]
                evidence.append(dict(channel=channel, p=p, component=comp, n=len(rows),
                    paired_T7_vs_T12_ge2sigma=sum(v>=2 for v in pulls),
                    median_absolute_paired_pull=float(np.median(pulls)) if pulls else None,
                    median_relative_change=float(np.median(relative)) if relative else None,
                    low_power_large_relative_change=sum(r['status']=='inconclusive_large_relative_shift_low_power' for r in rows)))
    result = dict(status='COMPLETE_B_SCHEME_RATIO50_BOOT200_PLATEAU_DIAGNOSTICS',
        nconf=NCONF, nboot=NBOOT, seed=SEED, flows=list(FLOWS), pz=list(PZ), tseps=list(TSEPS),
        tsep_fm=[round(t*A_FM, 4) for t in TSEPS], n_ratio_products=72, n_pdf=81,
        raw_new30_cases_reverified=60480, shared_bootstrap_indices_sha256=sha(HERE / 'bootstrap_indices_Nboot200_Nconf50_seed1115.npy'),
        paired_small_z_diagnostic=evidence, artifacts=artifacts,
        independent_gather_oracle_replicas=[0,17,199],
        ground_state_status='not_established_by_visual_plateau_or_paired_comparison_alone',
        transversity_convention_status=TRANS_STATUS, job=os.environ['SLURM_JOB_ID'],
        verification_v3_lock_sha256=sha(HERE / 'verification_v3_lock.json'),
        pdf_validation='Matplotlib header/EOF/Page-object count/root Count; independent representative pdfinfo and visual review separate')
    atomic_json(HERE / 'complete.json', result)
    report = ['# L32x64 B: 50 configurations, 200 bootstrap ratio diagnostics', '',
        '- Configurations: 10000..11960, step 40; signed momenta +/-4,+/-5,+/-6.',
        '- Flow times: tau=t_f/a^2 = 0.1,0.2,...,0.8; epsilon=0.01; original finite windows retained.',
        '- T/a = 7..12, or 0.6279,0.7176,0.8073,0.8970,0.9867,1.0764 fm.',
        '- Four current anchors (0,16,32,63) averaged per configuration first.',
        '- Shared 200x50 configuration bootstrap, seed 1115. Each replica averages C3 and C2 separately,',
        '  divides by complex insertion-matched nopol C2, then applies z-pair and signed-momentum projection.',
        '- Error: std(ddof=1) of bootstrap estimates, with no division by sqrt(200).',
        '- This is a connected finite-flow bare ratio, not a renormalized/matched physical PDF.', '',
        '## Plots', '',
        '`plots/by_flow/`: 72 PDFs (one channel/flavor/flow each). Every PDF has 20 pages:',
        '14 interior pages covering every z/a=0..20 and both components, followed by 6 full-range',
        'pages covering z/a=0,2,4,6,8,12,16,20. Columns are P=4,5,6; colors are T=7..12.',
        'The interior display cut dt>=2 and T-dt>=2 is NOT a fit selection. Full-range contact endpoints are open circles.', '',
        '`plots/flow_comparison/`: 9 PDFs of central-insertion ratio versus T, with all eight flows offset.',
        'All z and insertion times remain available in the 72 NPZ products in `ratios/`.', '',
        '## Plateau evidence (paired shared-bootstrap differences)', '',
        'The following counts are descriptive and correlated, not independent significance tests.',
        'Low significance alone is not evidence of a stable ground state. Large fractional shifts with',
        'low paired significance are explicitly flagged as low-power/inconclusive.', '',
        '| channel | P | component | points (tau,z=1..6) | T7-T12 >=2 sigma | median |pull| | median fractional change | low-power |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in evidence:
        def fmt(v): return 'undefined' if v is None else f'{v:.3g}'
        report.append(f'| {r["channel"]} | {r["p"]} | {r["component"]} | {r["n"]} | {r["paired_T7_vs_T12_ge2sigma"]} | '
                      f'{fmt(r["median_absolute_paired_pull"])} | {fmt(r["median_relative_change"])} | {r["low_power_large_relative_change"]} |')
    report += ['', 'Per-point values and paired differences: `diagnostics/central_T_comparison_tau*.csv`.',
        'No bare matrix-element fit, AIC selection, matching, or ground-state-isolation certification was performed.',
        'Transversity retains pending_external_physics_confirmation for its external relative-sign convention.', '',
        'Generated: 2026-10-30 (Asia/Shanghai).']
    (HERE / 'RESULTS.md').write_text('\n'.join(report)+'\n')
    print('RATIO50_BOOT200_FINAL_PASS', flush=True)


if __name__ == '__main__': main()
