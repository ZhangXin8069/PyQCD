"""Audit exact completed populations and freeze a true configuration bootstrap."""
import csv
from datetime import datetime
from common import *


def verify_parent_sources(parent):
    lock = json.loads((parent/'code_lock.json').read_text())
    for rel, expected in lock['files_sha256'].items():
        assert sha256(parent/rel) == expected, (parent, rel)


def main():
    allocation(); code_check()
    if (HERE/'input_manifest.json').exists():
        manifest(); shared_indices()
        print('VALIDATED_INPUT_GATE_SKIP', KEY, flush=True)
        return
    parent = PARENT64 if KEY == 'L32x64' else PARENT96
    verify_parent_sources(parent)
    parent_complete = json.loads((parent/'complete.json').read_text())
    assert parent_complete['nconf'] == NCONF
    evidence = []
    if KEY == 'L32x64':
        verify_parent_sources(OLD64)
        for root, status, cases in ((RAW64, 'COMPLETE_30_NEW_CONF_8_FLOW_B_P456', 60480),
                                   (P064, 'PASS_50CONF_8FLOW_B_PZ0', 100800)):
            complete = json.loads((root/'summary/complete.json').read_text())
            assert complete['status'] == status and complete['case_count'] == cases
            assert complete['manifest_sha256'] == sha256(root/'manifest.json')
            assert complete['max_C2_cross_flow_relative_L2'] == 0
            evidence.append(dict(path=str(root/'summary/complete.json'), status=status, case_count=cases))
        p0_manifest = json.loads((P064/'manifest.json').read_text())
        rows = p0_manifest['tasks'] + p0_manifest['reused_tasks']
        assert len(rows) == 400
        assert {(r['conf'], r['flow_time_a2']) for r in rows} == {(c, t) for c in CONFS for t, _ in FLOWS}
        old_index = PARENT64/'bootstrap_indices_Nboot200_Nconf50_seed1115.npy'
        indices = np.load(old_index, allow_pickle=False)
        assert sha256(old_index) == parent_complete['shared_bootstrap_indices_sha256']
    else:
        complete = json.loads((RAW96/'summary/complete.json').read_text())
        assert complete['status'] == 'COMPLETE_L32X96_B_MATCHED_RADIUS_C3_C2'
        assert complete['case_count'] == 35280 and complete['nconf'] == 20
        assert complete['manifest_sha256'] == sha256(RAW96/'manifest.json')
        assert complete['C2_cross_flow_relative_L2'] == 0
        raw = json.loads((RAW96/'manifest.json').read_text())
        assert raw['confs'] == list(CONFS) and raw['momenta_z'] == list(MOMENTA)
        for row in raw['tasks']:
            receipt = json.loads((RAW96/'status'/f'task{row["task"]}.json').read_text())
            assert receipt['status'] == 'PASS' and receipt['case_count'] == 252 and not receipt['smoke']
            assert receipt['manifest_sha256'] == complete['manifest_sha256']
        evidence.append(dict(path=str(RAW96/'summary/complete.json'), status=complete['status'], case_count=35280))
        indices = np.random.default_rng(SEED).integers(0, NCONF, (NBOOT, NCONF), dtype=np.int32)
        old_indices = np.load(PARENT96/'bootstrap_indices_Nboot100_seed1115.npy', allow_pickle=False)
        np.testing.assert_array_equal(indices[:100], old_indices)
    assert indices.shape == (NBOOT, NCONF) and indices.dtype == np.int32
    expected = np.random.default_rng(SEED).integers(0, NCONF, (NBOOT, NCONF), dtype=np.int32)
    np.testing.assert_array_equal(indices, expected)
    projected = []
    for tau, window in FLOWS:
        marker = json.loads((parent/'status'/f'ratio_{parent_tag(tau)}.json').read_text())
        assert marker['status'] == 'PASS'
        path = Path(marker['projected'])
        assert sha256(path) == marker['projected_sha256']
        ledger = Path(marker['raw_ledger'] if KEY == 'L32x64' else marker['raw_case_ledger'])
        ledger_sha = marker['raw_ledger_sha256'] if KEY == 'L32x64' else marker['raw_case_ledger_sha256']
        assert sha256(ledger) == ledger_sha
        with np.load(path, allow_pickle=False) as d:
            np.testing.assert_array_equal(d['confs'], CONFS)
            assert int(d['nconf']) == NCONF and float(d['tau_a2']) == tau
            assert str(d['weight_scheme']) == SCHEME and int(d['weight_applied_count']) == 1
            nm = 6 if KEY == 'L32x64' else len(MOMENTA)
            assert d['c3_source_mean'].shape == (NCONF,3,2,2,nm,21,6,13)
            assert d['c2_source_mean'].shape == (NCONF,nm,6,13)
            physical = np.arange(13)[None,:] <= np.asarray(TSEPS)[:,None]
            for a in (d['c3_source_mean'], d['c2_source_mean']):
                assert np.isfinite(a[...,physical]).all() and np.isnan(a[...,~physical]).all()
        projected.append(dict(tau=tau, window=window, path=str(path), sha256=marker['projected_sha256'],
                              ledger=str(ledger), ledger_sha256=ledger_sha))
    if index_path().exists():
        np.testing.assert_array_equal(np.load(index_path(), allow_pickle=False), indices)
    else:
        with index_path().open('xb') as stream:
            np.save(stream, indices, allow_pickle=False)
    assert TMIN*A_FM > .6 and (TMIN-1)*A_FM <= .6
    table = [dict(tau=t, tau64=FLOWS64[i], rf_fm=float(A_FM*np.sqrt(8*t)),
                  a_fm=A_FM, epsilon=EPSILON, nstep=round(t/EPSILON), window=w,
                  radius_window_selected=.0897 < A_FM*np.sqrt(8*t) < .3)
             for i,(t,w) in enumerate(FLOWS)]
    with (HERE/'flow_radius_selection.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=table[0].keys());writer.writeheader();writer.writerows(table)
    atomic_json(HERE/'input_manifest.json', dict(status='PASS_COMPLETED_B_PRODUCTION_AND_PROJECTED_PAYLOADS',
        timestamp=datetime.now().astimezone().isoformat(), job=os.environ['SLURM_JOB_ID'], ensemble=ENSEMBLE,
        confs=list(CONFS), nconf=NCONF, nboot=NBOOT, resample_size=NCONF, seed=SEED,
        bootstrap_kind='iid_configuration_bootstrap', bootstrap_indices_sha256=sha256(index_path()),
        pairing='shared across all flow/channel/flavor/P/z/T/dt within ensemble; NOT paired across ensembles',
        flows=[t for t,_ in FLOWS], flow_radius_selection=table, pz=list(PZ), signed_momenta=list(MOMENTA),
        tseps=list(TSEPS), current_anchors=list(CURRENTS), weight_scheme=SCHEME, estimator_id=ESTIMATOR,
        evidence=evidence, parent_projected=projected, parent_complete=str(parent/'complete.json'),
        Tmin=TMIN, Tmin_fm=TMIN*A_FM, Tmin_threshold_fm=.6, Q_min=Q_MIN, chi2_dof_max=CHI_MAX,
        candidates=[dict(label=l,tseps=list(ts),cut=c) for l,ts,c in CANDIDATES], fit_model='R(T,dt)=M',
        covariance='Schaefer-Strimmer correlation-to-identity + jitter + SVD 1e-12',
        aic='chi2+2*k+2*N_omit; k=1; AIC windows within cuts; equal usable cut mixture; per bootstrap',
        fallback='smallest candidate bootstrap std(ddof=1); diagnostic, NOT quality-accepted',
        audit_scope='existing full raw SHA/Wick/gamma audit + current compact hashes; per-flow live raw existence/oracles; P0 full revalidation',
        limitations=['finite-flow connected-only bare matrix elements, no renormalization/matching',
          'four fixed current anchors and finite temporal support; not a six-direction estimator',
          'bootstrap200 is not 200 independent configurations; autocorrelation not yet assessed',
          'no ground-state-isolation claim; report quality and fallback separately',
          'transversity relative sign pending_external_physics_confirmation']))
    print('INPUT_GATE_PASS',KEY,NCONF,NBOOT,flush=True)


if __name__ == '__main__':
    main()
