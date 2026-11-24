"""Reuse audited old20 projections, hash/read new30 once, then paired boot200."""
import argparse
import csv
import json
import os
import time
import numpy as np
from analysis_common import *


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--flow-index', type=int, required=True)
    args = parser.parse_args(); allocation(); code_check()
    tau = FLOWS[args.flow_index]; ind = indices(); start = time.monotonic()
    assert not (HERE / 'status' / f'ratio_{tag(tau)}.json').exists()
    prod = production(); m = prod.load_manifest()
    oldmark = json.loads((OLD / 'status' / f'ratio_{tag(tau)}.json').read_text())
    assert sha(oldmark['projected']) == oldmark['projected_sha256']
    c3 = np.full((NCONF, 3, 2, 2, 6, DZ, 6, 13), np.nan + 1j*np.nan)
    c2 = np.full((NCONF, 6, 6, 13), np.nan + 1j*np.nan)
    with np.load(oldmark['projected'], allow_pickle=False) as old:
        c3[:20] = np.take(old['c3_source_mean'], range(2, 8), axis=4)
        c2[:20] = np.take(old['c2_source_mean'], range(2, 8), axis=1)
    rows = {r['conf']: r for r in m['tasks'] if r['flow_time_a2'] == tau}
    ledger_path = HERE / 'projected' / f'new30_raw_ledger_{tag(tau)}.csv'
    with ledger_path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('conf', 'tau', 'current', 'tsep', 'dt', 'path', 'sha256'))
        writer.writeheader(); count = 0
        for ic, conf in enumerate(CONFS[20:], 20):
            row = rows[conf]
            for it, sep in enumerate(TSEPS):
                for dt in range(sep+1):
                    numer = np.zeros((3, 2, 2, 6, DZ), complex); denom = np.zeros(6, complex)
                    for cur in CURRENTS:
                        raw, wick, raw2, path, receipt = prod.read_case(row, cur, sep, dt, verify_hash=True)
                        numer += project_spin(raw) / 4; denom += raw2[0] / 4
                        writer.writerow(dict(conf=conf, tau=tau, current=cur, tsep=sep, dt=dt,
                                             path=str(path), sha256=receipt['npz_sha256']))
                        count += 1
                    c3[ic, ..., it, dt] = numer; c2[ic, :, it, dt] = denom
            print('CONF_PROJECTED', tau, conf, ic+1, '/', NCONF, flush=True)
    assert count == 7560
    projected = HERE / 'projected' / f'projected_{tag(tau)}_Nconf50.npz'
    atomic_npz(projected, c3_source_mean=c3, c2_source_mean=c2,
        axes_c3=np.asarray('conf,channel,flavor_u_d,sign,signed_momentum,z,tsep,insertion'),
        axes_c2=np.asarray('conf,signed_momentum,tsep,insertion'), **metadata(tau))
    ro, rb, den, dens = compute_ratios(c3, c2, ind)
    mask = np.arange(13)[None, :] <= np.asarray(TSEPS)[:, None]
    products = []
    for ch, channel in enumerate(CHANNELS):
        for jf, flavor in enumerate(FLAVORS):
            r, b = ro[ch, jf], rb[:, ch, jf]
            valid = np.broadcast_to(mask, r.shape).copy()
            assert np.isfinite(r[valid]).all() and np.isfinite(b[:, valid]).all()
            assert np.isnan(r[~valid]).all() and np.isnan(b[:, ~valid]).all()
            component = np.stack((r[0].real, r[1].imag))
            component_boot = np.stack((b[:, 0].real, b[:, 1].imag), axis=1)
            path = ratio_path(tau, channel, flavor)
            atomic_npz(path, ratio_original=r, ratio_boot=b, ratio_valid_mask=valid, valid_tau_mask=mask,
                component_original=component, component_boot=component_boot,
                component_error=np.std(component_boot, axis=0, ddof=1),
                component_labels=np.asarray(['even_real', 'odd_imag']),
                channel=np.asarray(channel), flavor_combo=np.asarray(flavor),
                axes_original=np.asarray('vdv,p,z,tsep,insertion'),
                axes_boot=np.asarray('boot,vdv,p,z,tsep,insertion'),
                axes_component_boot=np.asarray('boot,component,p,z,tsep,insertion'),
                is_contact=(np.arange(13)[None, :]==0)|(np.arange(13)[None, :]==np.asarray(TSEPS)[:, None]),
                pipeline_version=np.asarray('L32x64_B_ratio50_boot200_plateau_v1'), **metadata(tau))
            products.append(dict(path=str(path), sha256=sha(path)))
    denpath = HERE / 'diagnostics' / f'c2_{tag(tau)}.npz'
    sigma = np.std(dens.real, axis=0, ddof=1)
    snr = np.divide(np.abs(den.real), sigma, out=np.full_like(sigma, np.nan), where=sigma>0)
    crossing = np.mean(np.sign(dens.real) != np.sign(den.real)[None], axis=0)
    atomic_npz(denpath, c2_original=den, c2_boot=dens, real_snr=snr,
        bootstrap_real_sign_cross_fraction=crossing, valid_tau_mask=mask, **metadata(tau))
    atomic_json(HERE / 'status' / f'ratio_{tag(tau)}.json', dict(status='PASS', tau=tau,
        nconf=NCONF, nboot=NBOOT, products=products,
        projected=str(projected), projected_sha256=sha(projected),
        old20_projected_sha256=oldmark['projected_sha256'],
        new30_raw_cases_reverified=count, raw_ledger=str(ledger_path), raw_ledger_sha256=sha(ledger_path),
        c2_product=str(denpath), c2_sha256=sha(denpath),
        minimum_C2_real_snr=float(np.nanmin(snr)),
        maximum_C2_bootstrap_real_sign_cross_fraction=float(np.max(crossing[:, mask])),
        bootstrap_indices_sha256=sha(HERE / 'bootstrap_indices_Nboot200_Nconf50_seed1115.npy'),
        seconds=time.monotonic()-start, job=os.environ['SLURM_JOB_ID']))
    print('RATIO50_FLOW_PASS', tau, flush=True)


if __name__ == '__main__':
    main()
