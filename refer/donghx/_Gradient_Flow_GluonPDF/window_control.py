#!/usr/bin/env python3
"""CPU control: use all measured lambda>=7 and splice at the last data point.

Keep the completed primary products untouched. Only u-d, all three channels
and four momenta are used for this explicit joint window/splice sensitivity.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'code'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from reconstruction import fit_tail, coefficients, basis, fourier_operator, G
from plotting import band, error


def load(p):
    with np.load(p, allow_pickle=False) as f:
        return {k: np.array(f[k]) for k in f.files}


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'CPU Slurm job required'
    base = ROOT/'results'
    assert json.loads((base/'checks/independent_final_review.json').read_text())['status'] == 'passed'
    out = base/'all_measured_tail_control'
    for stage in ['07_lambda', '08_fourier', '09_lightcone']:
        (out/stage/'data').mkdir(parents=True, exist_ok=True)
        (out/stage/'plots').mkdir(parents=True, exist_ok=True)
    records = []
    books = {s: PdfPages(out/s/'plots'/f'{s}_window_control_all_channels.pdf')
             for s in ['07_lambda', '08_fourier', '09_lightcone']}
    try:
        for ch in ['unpolarized', 'helicity', 'transversity']:
            figs = {s: plt.subplots(4, 2, figsize=(11, 12.2), squeeze=False) if s == '07_lambda'
                    else plt.subplots(2, 2, figsize=(11, 8.7), squeeze=False) for s in books}
            for ip, nz in enumerate([3, 4, 5, 6]):
                stem = f'{ch}_u_minus_d_Pz{nz}.npz'
                s7 = load(base/'07_lambda'/'data'/stem)
                s8 = load(base/'08_fourier'/'data'/stem)
                s9 = load(base/'09_lightcone'/'data'/stem)
                op = load(base/'09_lightcone'/'operators'/f'{ch}_Pz{nz}.npz')
                lam, h = s7['lambda_data'], s7['h_data']
                pz = float(s7['Pz_GeV'])
                join = float(lam[-1])
                fit = fit_tail(lam, h, lower=7., upper=join)
                assert np.all(fit['bootstrap_success'])
                cf = coefficients(h, fit)
                Fy = fourier_operator(nz, pz, ch, tuple(op['y']), join=join)
                Fx = fourier_operator(nz, pz, ch, tuple(op['x']), join=join)
                fm = Fx-G*(op['K']@Fy+op['extension'])
                q, matched = cf@Fy.T, cf@fm.T
                assert np.all(np.isfinite(q)) and np.all(np.isfinite(matched))
                lp = s7['lambda_plot']
                curve = cf@basis(lp, lam, pz, ch, join=join).T
                meta = dict(channel=ch, flavor='u_minus_d', nz=nz, Pz_GeV=pz,
                    boot_indices=s7['boot_indices'], confs=s7['confs'],
                    bootstrap_original_indices=np.arange(100),
                    bootstrap_index_sha256=s7['bootstrap_index_sha256'],
                    source_sha256=s7['frozen_source_sha256'],
                    status='conditional fixed-a joint tail-window/splice control',
                    tail_window=np.array([7, join]), splice=np.array([join-1, join]))
                np.savez_compressed(out/'07_lambda'/'data'/stem, **meta,
                    lambda_data=lam, h_data=h, lambda_plot=lp, h_reconstructed=curve,
                    **{'fit_'+k: v for k, v in fit.items()})
                np.savez_compressed(out/'08_fourier'/'data'/stem, **meta,
                    x=op['y'], central=q[0], samples=q[1:], std=np.std(q[1:], axis=0, ddof=1))
                np.savez_compressed(out/'09_lightcone'/'data'/stem, **meta,
                    x=op['x'], central=matched[0], samples=matched[1:],
                    std=np.std(matched[1:], axis=0, ddof=1),
                    covariance=np.cov(matched[1:], rowvar=False, ddof=1),
                    difference_from_primary=matched-np.r_[s9['central'][None, :], s9['samples']])
                for ic, ax in enumerate(figs['07_lambda'][1][ip]):
                    val = h.real if ic == 0 else h.imag
                    ax.errorbar(lam, val[0], yerr=error(val), fmt='o', ms=3,
                        color='#292929', capsize=2, linestyle='none', label='All measured 06 data')
                    primary = s7['h_reconstructed'].real if ic == 0 else s7['h_reconstructed'].imag
                    control = curve.real if ic == 0 else curve.imag
                    band(ax, lp, primary, '#2774a0', 'Main: fit 7..12, splice 9..10')
                    band(ax, lp, control, '#ad6721', 'Control: fit 7..last, splice last-1..last')
                    ax.axvspan(7, join, alpha=.08, color='#ad6721')
                    ax.set(xlim=(0, 36), ylim=(-1, 2) if ic == 0 else (-2, 1),
                        xlabel=r'$\lambda=zP_z$', ylabel=r'$\mathrm{Re}\,H$' if ic == 0 else r'$\mathrm{Im}\,H$',
                        title=f'nP={nz}, Pz={pz:.3f} GeV; control max Q={np.max(fit["Q_nominal"]):.2g}')
                    ax.grid(alpha=.15)
                for stage, grid, mainvals, newvals in [
                    ('08_fourier', op['y'], np.r_[s8['central'][None, :], s8['samples']], q),
                    ('09_lightcone', op['x'], np.r_[s9['central'][None, :], s9['samples']], matched)]:
                    ax = figs[stage][1].flat[ip]
                    band(ax, grid, mainvals, '#2774a0', 'Main')
                    band(ax, grid, newvals, '#ad6721', 'All-measured-tail control')
                    ax.set(xlim=(-2, 2) if stage == '08_fourier' else (-1, 1),
                        xlabel=r'$x$', ylabel=r'$q_\Gamma/g_\Gamma$', title=f'nP={nz}, Pz={pz:.3f} GeV')
                    ax.axhline(0, color='.7', lw=.6)
                    ax.grid(alpha=.15)
                rec = dict(channel=ch, nz=nz, tail_window=[7, join], splice=[join-1, join],
                    tail_max_Q=float(np.max(fit['Q_nominal'])),
                    main_tail_max_Q=float(np.max(s7['tail_Q_nominal'])),
                    sample_rank=int(fit['sample_rank']), effective_rank=int(fit['effective_rank']),
                    valid_bootstrap=100,
                    max_abs_lightcone_shift=float(np.max(abs(matched[0]-s9['central']))))
                records.append(rec)
                print(json.dumps(rec), flush=True)
            for stage, (fig, axes) in figs.items():
                axes.flat[0].legend(fontsize=7, loc='best')
                fig.suptitle(f'{stage} joint window/splice sensitivity: {ch}, u-d', fontsize=13)
                fig.text(.02, .015, 'Conditional diagnostic, 20 cfg / 100 paired bootstrap. Upstream m0/finite-scheme limitations retained.\n'
                    'This compares BOTH fit extent and the splice; no model uncertainty added to statistical bands.', fontsize=7)
                fig.subplots_adjust(left=.10, right=.98, top=.92, bottom=.10, hspace=.42, wspace=.30)
                name = out/stage/'plots'/f'{stage}_{ch}_u_minus_d_window_control'
                fig.savefig(str(name)+'.pdf')
                fig.savefig(str(name)+'.png', dpi=120)
                books[stage].savefig(fig)
                plt.close(fig)
    finally:
        for book in books.values():
            book.close()
    (out/'manifest.json').write_text(json.dumps(dict(status='complete', products=12,
        job_id=os.environ['SLURM_JOB_ID'], code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        records=records), indent=2)+'\n')
    print('COMPLETE: 12 all-measured-tail u-d controls, 3 stages, all original 100 rows.', flush=True)


if __name__ == '__main__':
    main()
