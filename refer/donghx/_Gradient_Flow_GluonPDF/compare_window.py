"""Audit the fixed-a cut2 mass comparison at native z in [0.4, 0.6] fm."""

import csv
from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'quark_gf_cut2_P0_RGR_LRR_20261124'
SPEC = importlib.util.spec_from_file_location('p0_analysis', SOURCE / 'code/analyze.py')
PREVIOUS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREVIOUS)


def checked(relative, hashes):
    path = SOURCE / relative
    assert sha256(path.read_bytes()).hexdigest() == hashes[relative], path
    return path


def reference(pv, coupling, branch, distances, variation):
    targets, alpha_h, reason = [], [], []
    for distance in distances:
        try:
            detail = pv.details(distance / PREVIOUS.HBARC, scheme=branch, mu=coupling['mu_GeV'],
                                alpha=coupling['central'], nf=coupling['nf'], c_variation=variation)
        except ValueError as error:
            targets.append(np.nan)
            alpha_h.append(np.nan)
            reason.append(str(error))
        else:
            targets.append(detail['value'])
            alpha_h.append(detail['alpha_h'])
            reason.append('')
    return np.array(targets), np.array(alpha_h), reason


def run():
    receipt = json.loads((SOURCE / 'complete.json').read_text())
    assert receipt['status'] == 'COMPLETE_DIAGNOSTIC'
    for path in (Path(SPEC.origin), PREVIOUS.PV_CODE, PREVIOUS.GF / 'coupling.json'):
        PREVIOUS.audited(path, receipt['inputs_sha256'][str(path)])
    pv = PREVIOUS.load_pv()
    coupling = PREVIOUS.read(PREVIOUS.GF / 'coupling.json')
    PREVIOUS.check_coefficients(pv, coupling)
    with checked('flow_audit.csv', receipt['outputs_sha256']).open() as stream:
        audit = {(row['ensemble'], row['endpoint'], float(row['z_fm'])): row for row in csv.DictReader(stream)
                 if row['coefficient'] == 'primary' and row['window'] == 'all5'}
    results, points = [], []
    for ensemble in PREVIOUS.ENSEMBLES.values():
        for label, branch, endpoint, variation in [('NLO_fixed', 'nlo', 'FO', 1.),
                                                   ('RGR_nominal', 'rgr_nlo', 'RGR', 1.),
                                                   ('LRR_nominal', 'rgr_lrr_nlo', 'RGR', 1.),
                                                   ('LRR_scale_x2', 'rgr_lrr_nlo', 'RGR', 2.)]:
            name = f'data/{ensemble}_primary_{endpoint}.npz'
            with np.load(checked(name, receipt['outputs_sha256']), allow_pickle=False) as data:
                mask = (data['z'] >= .4 - 1e-12) & (data['z'] <= .6 + 1e-12)
                distances, center, replicas = data['z'][mask], data['H0'][mask], data['H0_boot'][:, mask]
                scale_counts = data['scale_mask'][:, mask].sum(axis=0)
            assert len(distances) == 2 and replicas.shape == (500, 2)
            assert np.all(np.isfinite(center)) and np.all(np.isfinite(replicas))
            assert np.all(center > 0) and np.all(replicas > 0)
            targets, alpha_h, errors = reference(pv, coupling, branch, distances, variation)
            fit = None if not np.isfinite(targets).all() else PREVIOUS.gls(
                np.log(center / targets), np.log(replicas / targets), -distances[:, None] / PREVIOUS.HBARC)
            record = dict(ensemble=ensemble, scheme=label, z_min_fm=distances[0], z_max_fm=distances[-1],
                          n_points=len(distances), n_valid_targets=int(np.isfinite(targets).sum()),
                          status='diagnostic_only' if fit is not None else 'target_unavailable')
            if fit is not None:
                record.update(m0_GeV=fit['values'][0], m0_SD_GeV=fit['sd'][0], Q=fit['Q'],
                              chi2_dof=fit['chi2_dof'], dof=fit['dof'])
            results.append(record)
            for idx, distance in enumerate(distances):
                flow = audit[ensemble, endpoint, float(distance)]
                assert int(flow['nscale']) == scale_counts[idx] < 3
                points.append(dict(ensemble=ensemble, scheme=label, z_fm=distance, n_scale_flows=int(scale_counts[idx]),
                                   n_bare_fits_passing=int(flow['source_pass']), flow_Q=float(flow['Q']),
                                   flow_chi2_dof=float(flow['chi2_dof']), H0=center[idx], H0_SD=replicas[:, idx].std(ddof=1),
                                   target=targets[idx] if not errors[idx] else None,
                                   mu_target_GeV=coupling['mu_GeV'] if branch == 'nlo' else variation*pv.C_NATURAL*PREVIOUS.HBARC/distance,
                                   alpha_h=alpha_h[idx] if not errors[idx] else None, target_status=errors[idx] or 'finite'))
    for filename, rows in [('mass_windows.csv', results), ('point_audit.csv', points)]:
        with (ROOT / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
            writer.writeheader()
            writer.writerows(rows)
    assert all(row['n_valid_targets'] == 0 for row in results if row['scheme'].endswith('_nominal'))
    assert all(row.get('dof') == 1 for row in results if row['status'] == 'diagnostic_only')
    print('Nominal RGR and LRR: 0/6 valid targets each; verified two native points per ensemble.')


if __name__ == '__main__':
    run()
