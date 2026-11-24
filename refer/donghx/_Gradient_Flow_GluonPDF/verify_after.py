#!/usr/bin/env python3
"""Independent CPU reread and scalar Fourier validation of completed files."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results'


def load(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: np.array(f[k]) for k in f.files}


def scalar_H(lam, data, channel):
    """Independent scalar version, without importing reconstruction code."""
    ls = data['lambda_data']
    h = data['h_data'][0]
    pz = float(data['Pz_GeV'])
    p, c = {'unpolarized': (1.5, 2.5), 'helicity': (1.5, 3.5), 'transversity': (2., 2.)}[channel]
    g = .3*(4/3)/(2*np.pi)
    def c0(v):
        return 1+g*(p*np.log(v*v*4*np.exp(2*np.euler_gamma)/(4*pz*pz))+c)
    if lam < ls[0]:
        low = (h[0].real+1j*h[0].imag*lam/ls[0])*c0(lam)/c0(ls[0])
    else:
        low = np.interp(lam, ls, h.real)+1j*np.interp(lam, ls, h.imag)
    if lam <= 9:
        return low
    tail = 0j
    for shape, weight, pars in zip(data['tail_registry'], data['tail_weights'], data['tail_parameters'][0]):
        a, b, length = shape
        A, B, cp, cm = pars
        tail += weight*np.exp(-(lam-8)/length)*((A+1j*B)*(lam/8)**(-a)+
            cp*np.exp(-1j*lam)*complex(0, -lam/8)**(-b)+
            cm*np.exp(1j*lam)*complex(0, lam/8)**(-b))
    u = min(1., max(0., lam-9))
    blend = u*u*(3-2*u)
    return (1-blend)*low+blend*tail


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'CPU Slurm job required'
    manifest = json.loads((OUT/'manifest.json').read_text())
    assert manifest['status'] == 'complete' and len(manifest['products']) == 36
    audit = json.loads((OUT/'checks'/'output_audit.json').read_text())
    assert audit['status'] == 'passed'
    for src in manifest['inputs']:
        assert hashlib.sha256(Path(src['source']).read_bytes()).hexdigest() == src['sha256']
        assert hashlib.sha256(Path(src['frozen']).read_bytes()).hexdigest() == src['sha256']
    scalar_checks, clip_records, nlo = [], [], []
    for ch in ['unpolarized', 'helicity', 'transversity']:
        for fl in ['u', 'd', 'u_minus_d']:
            source = load(OUT/'06_input'/'data'/f'{ch}_{fl}.npz')
            for nz in [3, 4, 5, 6]:
                stem = f'{ch}_{fl}_Pz{nz}.npz'
                s7 = load(OUT/'07_lambda'/'data'/stem)
                s8 = load(OUT/'08_fourier'/'data'/stem)
                s9 = load(OUT/'09_lightcone'/'data'/stem)
                op = load(OUT/'09_lightcone'/'operators'/f'{ch}_Pz{nz}.npz')
                original = np.concatenate([source['value_complex'][nz-3][None, :],
                    source['samples_complex'][:, nz-3, :]], axis=0)
                np.testing.assert_array_equal(s7['h_data'], original[:, 1:])
                for s in (s7, s8, s9):
                    np.testing.assert_array_equal(s['boot_indices'], source['boot_indices'])
                    np.testing.assert_array_equal(s['bootstrap_original_indices'], np.arange(100))
                coeff = s7['coefficients']
                expected = coeff@op['Fx'].T-.3*(4/3)/(2*np.pi)*(coeff@op['Fy'].T@op['K'].T+coeff@op['extension'].T)
                np.testing.assert_allclose(s9['central'], expected[0], rtol=1e-9, atol=1e-10)
                np.testing.assert_allclose(s9['samples'], expected[1:], rtol=1e-9, atol=1e-10)
                np.testing.assert_allclose(s8['std'], np.std(s8['samples'], axis=0, ddof=1))
                np.testing.assert_allclose(s9['std'], np.std(s9['samples'], axis=0, ddof=1))
                for comp, arr, lim in [('Re', original.real, (-1, 2)), ('Im', original.imag, (-2, 1))]:
                    clip_records.append(dict(channel=ch, flavor=fl, nz=nz, component=comp,
                        central_points_outside_ylim=int(np.sum((arr[0] < lim[0]) | (arr[0] > lim[1])))))
                bulk = (abs(s9['x']) >= .1) & (abs(s9['x']) <= .8)
                change = abs(s9['central']-s9['quasi'][0])
                denom = np.maximum(abs(s9['quasi'][0]), .05)
                nlo.append(dict(channel=ch, flavor=fl, nz=nz,
                    bulk_max_correction_relative_to_quasi_with_absolute_floor005=float(np.max(change[bulk]/denom[bulk]))))
                if fl == 'u_minus_d' and nz == 3:
                    for xx in [-.4, .2, .6]:
                        def f(v):
                            z = scalar_H(v, s7, ch)
                            return (z.real*np.cos(xx*v)-z.imag*np.sin(xx*v))/np.pi
                        bounds = np.unique(np.r_[0, s7['lambda_data'][s7['lambda_data'] < 10], 9, 10, 160])
                        direct = sum(quad(f, a, b, epsabs=2e-9, epsrel=2e-9, limit=300)[0]
                            for a, b in zip(bounds[:-1], bounds[1:]))
                        j = np.argmin(abs(s8['x_matching']-xx))
                        stored = float(s8['quasi_at_matching_x'][0, j])
                        difference = abs(direct-stored)
                        if difference > 2e-7:
                            raise AssertionError((ch, xx, direct, stored))
                        scalar_checks.append(dict(channel=ch, nz=nz, x=xx,
                            adaptive_scalar=float(direct), stored=stored, absolute_difference=float(difference)))
    report = dict(status='passed', job_id=os.environ['SLURM_JOB_ID'],
        source_hashes_verified=9, paired_products_verified=36,
        all_100_bootstrap_rows_retained=True, original_center_verified=True,
        source_complex_components_verified=True, numerical_matching_products_verified=True,
        independent_scalar_Fourier=scalar_checks, plot_clipping=clip_records,
        perturbative_correction_diagnostics=nlo,
        limitations=manifest['restrictions'])
    (OUT/'checks'/'independent_final_review.json').write_text(json.dumps(report, indent=2)+'\n')
    low = sum(p['all_tail_models_low_Q'] for p in manifest['products'])
    table = ['|channel|flavor|nP|tail max Q|best chi2/dof|', '|---|---|---|---|---|']
    for p in manifest['products']:
        table.append(f"|{p['channel']}|{p['flavor']}|{p['nz']}|{p['tail_max_Q']:.5g}|{p['tail_best_chi2_dof']:.5g}|")
    content = ('# Completed results: 06 -> lambda -> Fourier -> light-cone matching\n\n'
        f'36 products, 20 configurations and all 100 paired bootstrap replicas. All three numerical layers and stage PDFs completed on CPU.\n\n'
        f'{low}/36 products have no tail model with nominal Q >= 0.05; all source residual-mass fits have very poor goodness of fit. '
        'Results remain conditional diagnostics.\n\n'
        '- [Input complex matrix elements](results/06_input/plots/06_input_all_channels_flavors.pdf)\n'
        '- [Lambda continuation](results/07_lambda/plots/07_lambda_all_channels_flavors.pdf)\n'
        '- [Fourier quasi PDFs](results/08_fourier/plots/08_fourier_all_channels_flavors.pdf)\n'
        '- [Before/after light-cone matching](results/09_lightcone/plots/09_lightcone_all_channels_flavors.pdf)\n'
        '- [Matched momentum overlay](results/09_lightcone/plots/matched_momenta_overlay.pdf)\n'
        '- [Model sensitivity](results/07_lambda/plots/model_sensitivity.pdf)\n\n'
        'Original central estimates and bootstrap means are stored separately. Statistical bands exclude model/finite-scheme/continuum systematics.\n\n'
        +'\n'.join(table)+'\n\nGenerated 2026-10-25: L32x64 B/RGR 06 downstream reconstruction.\n')
    (ROOT/'RESULTS.md').write_text(content)
    print(json.dumps(dict(status='passed', products=36, scalar_checks=len(scalar_checks), tail_low_Q_count=low)), flush=True)


if __name__ == '__main__':
    main()
