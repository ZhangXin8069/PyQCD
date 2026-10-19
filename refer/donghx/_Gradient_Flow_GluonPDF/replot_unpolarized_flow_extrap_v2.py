#!/usr/bin/env python3
"""Plot frozen flow-extrapolation products and audit the missing P3 points.

No extrapolation, matching, resampling, or quality-threshold changes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from extrapolate_unpolarized_tau_ge1_correlated_v2 import plot
from plot_unpolarized_multiflow_fixed_ylim_v1 import load_and_audit, product_path, DEFAULT_ROOT

DEFAULT_RESULT = DEFAULT_ROOT.parent / 'unpolarized_flow_extrap_tau_ge1_correlated_v2'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_p3(root, products, result):
    rows = []
    ip = list(result['pabs_values']).index(3)
    for it, tau in enumerate(result['flow_times']):
        op = products[float(tau)]['operator_index']
        with np.load(product_path(root, tau), allow_pickle=False) as f:
            fields = {k: f[k][op, ip] for k in (
                'estimate_available', 'fit_accepted', 'central_fallback_used',
                'fit_failure_reason', 'bootstrap_success_fraction')}
            gate = float(f['bootstrap_min_success'])
        for iz, z in enumerate(result['z_values']):
            usable = bool(fields['estimate_available'][iz] and fields['fit_accepted'][iz]
                          and not fields['central_fallback_used'][iz])
            assert usable == bool(result['input_usable'][it, ip, iz])
            rows.append(dict(n_z=3, z_over_a=int(z), tau=float(tau),
                estimate_available=bool(fields['estimate_available'][iz]),
                fit_accepted=bool(fields['fit_accepted'][iz]),
                central_fallback_used=bool(fields['central_fallback_used'][iz]),
                input_usable=usable, reason=str(fields['fit_failure_reason'][iz]),
                bootstrap_success_fraction=float(fields['bootstrap_success_fraction'][iz]),
                bootstrap_min_success=gate))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', type=Path, default=DEFAULT_RESULT)
    parser.add_argument('--matched-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--ymin', type=float, default=-2.5)
    parser.add_argument('--ymax', type=float, default=1.)
    args = parser.parse_args()
    if not np.isfinite([args.ymin, args.ymax]).all() or args.ymin >= args.ymax:
        parser.error('Invalid y-axis bounds')
    out = args.out or args.result_dir / 'plots_ylim_m2p5_1p0'
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a fresh output directory')
    source = args.result_dir / 'unpolarized_tau0_correlated_aic.npz'
    source_hash = digest(source)
    with np.load(source, allow_pickle=False) as f:
        result = {k: f[k] for k in f.files}
    original_manifest = json.loads((args.result_dir / 'manifest.json').read_text())
    products, records, settings, index_hash = load_and_audit(args.matched_root)
    assert index_hash == original_manifest['common_index_sha256']
    assert settings == original_manifest['matching_parameters']
    for tau, row in products.items():
        if tau not in result['flow_times']:
            continue
        it = list(result['flow_times']).index(tau)
        np.testing.assert_array_equal(row['M_filled'][row['operator_index']], result['input_central'][it])
    rows = audit_p3(args.matched_root, products, result)
    out.mkdir(parents=True, exist_ok=True)
    paths = plot(dict(products=products, z=result['z_values'], pz=result['pabs_values']),
                 result, out, args, ylim=(args.ymin, args.ymax))
    with (out / 'p3_input_quality_audit.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    clipped = []
    for ip, pz in enumerate(result['pabs_values']):
        for iz, z in enumerate(result['z_values']):
            m = float(result['beta'][ip, iz, 0])
            if 1 <= z <= 15 and result['valid'][ip, iz] and not args.ymin <= m <= args.ymax:
                clipped.append(dict(n_z=int(pz), z_over_a=int(z), M0=m,
                                    statistical_error=float(result['stat_error'][ip, iz, 0])))
    assert digest(source) == source_hash, 'Frozen fit product changed'
    for record in records:
        s = Path(record['path']).stat()
        assert (s.st_size, s.st_mtime_ns) == (record['bytes'], record['mtime_ns'])
    manifest = dict(status='replot_only_no_refit', generated_utc=datetime.now(timezone.utc).isoformat(),
        source=str(source.resolve()), source_sha256=source_hash, source_unchanged=True,
        ylim=[args.ymin, args.ymax], plots=paths, centers_outside_ylim=clipped,
        valid_counts_z1to15=result['valid'][:, 1:16].sum(axis=1).tolist(),
        missing_p3_z1to15=(np.flatnonzero(~result['valid'][0, 1:16])+1).tolist(),
        quality_audit=str((out/'p3_input_quality_audit.csv').resolve()),
        script=str(Path(__file__).resolve()), script_sha256=digest(Path(__file__)),
        plotting_module_sha256=digest(Path(__file__).with_name('extrapolate_unpolarized_tau_ge1_correlated_v2.py')))
    (out/'plot_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
