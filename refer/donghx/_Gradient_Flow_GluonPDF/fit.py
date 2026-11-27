"""Fit the symmetric expanded two-state model to ordinary projected C3/C2."""

from hashlib import sha256
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parent
BARE = ROOT.parent / 'quark_gf_adjacent_ref75_boot500_T0p8_1p1_20261122'
PREVIOUS = ROOT.parent / 'quark_gf_cut2_F32_targets_all_stages_20261125'
ENSEMBLES = {'L24x72': 'C24P29', 'L32x64': 'E32P29', 'L32x96': 'F32P30'}
NBOOT = 500
HBARC = 0.1973269804
GAPS = np.geomspace(.2, 2., 121)
RCOND = 1e-10


def digest(path):
    with path.open('rb') as stream:
        result = sha256()
        for block in iter(lambda: stream.read(2**20), b''):
            result.update(block)
    return result.hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def design(gaps, ts, ti, components):
    gap = np.asarray(gaps)[..., None]
    src = np.exp(-gap * ti)
    snk = np.exp(-gap * (ts-ti))
    diagonal = np.exp(-gap * ts)
    columns = np.stack((np.ones_like(diagonal), src+snk, diagonal), axis=-1)
    derivative = np.stack((np.zeros_like(diagonal), -ti*src-(ts-ti)*snk, -ts*diagonal), axis=-1)
    return np.kron(np.eye(components), columns), np.kron(np.eye(components), derivative)


def two_state(data, ts, ti, spacing, components):
    samples = data[1:]
    cov = np.cov(samples, rowvar=False, ddof=1)
    scale = np.sqrt(cov.diagonal())
    assert np.all(scale > 0)
    corr = cov/scale[:, None]/scale[None, :]
    eigen, vec = np.linalg.eigh(corr)
    assert eigen[0] > -1e-12
    keep = eigen > eigen[-1]*1e-12
    whitener = (vec[:, keep]/np.sqrt(eigen[keep])).T/scale
    obs = (whitener@data.T).T
    dof = int(keep.sum())-3*components-1
    assert dof > 0
    lattice_gaps = GAPS*spacing/HBARC
    matrix, _ = design(lattice_gaps, ts, ti, components)
    qi, ri = np.linalg.qr(whitener@matrix, mode='complete')
    assert np.all(np.abs(ri.diagonal(axis1=-2, axis2=-1)) > 1e-12)
    residual_projection = qi[:, :, 3*components:].transpose(0, 2, 1)@obs.T
    score = np.sum(residual_projection**2, axis=1)
    index = np.argmin(score, axis=0)
    gap = lattice_gaps[index]

    def conditional(candidate):
        matrix, derivative = design(candidate, ts, ti, components)
        weighted = whitener@matrix
        orth, triangular = np.linalg.qr(weighted)
        amplitudes = np.linalg.solve(triangular, orth.transpose(0,2,1)@obs[..., None])[..., 0]
        residuals = obs - (weighted@amplitudes[..., None])[..., 0]
        return np.sum(residuals**2, axis=1), amplitudes, weighted, whitener@derivative

    log_step = np.diff(np.log(lattice_gaps))[0]
    offsets = np.linspace(-1, 1, 5)
    for _ in range(3):
        candidates = np.clip(gap[None,:]*np.exp(log_step*offsets[:,None]), lattice_gaps[0], lattice_gaps[-1])
        trial = np.stack([conditional(candidate)[0] for candidate in candidates])
        gap = candidates[np.argmin(trial, axis=0), np.arange(len(data))]
        log_step *= .5
    best, amplitudes, weighted, derivative = conditional(gap)
    assert np.all(best <= score[index,np.arange(len(data))]+1e-6)
    jac = np.concatenate((weighted, derivative@amplitudes[..., None]), axis=2)
    norms = np.linalg.norm(jac, axis=1)
    singular = np.linalg.svd(jac/np.where(norms > 0, norms, 1)[:,None,:], compute_uv=False)
    jac_rank = np.sum(singular > RCOND*singular[:,:1], axis=1)
    at_bound = np.isclose(gap, lattice_gaps[0], rtol=1e-5, atol=0) | np.isclose(gap, lattice_gaps[-1], rtol=1e-5, atol=0)
    confidence_profile = (score <= best[None,:]+1)
    closed = ~confidence_profile[0] & ~confidence_profile[-1]
    identified = closed & ~at_bound & (jac_rank==3*components+1)
    q = chi2.sf(best, dof)
    residual_ok = (q >= .05) & (best/dof <= 3)
    pars = np.zeros((len(data), 7))
    pars[:,:3*components] = amplitudes
    pars[:,6] = gap*HBARC/spacing
    return dict(pars=pars, chisq=best, q=q, dof=dof, covariance_rank=int(keep.sum()),
                jac_rank=jac_rank, bound=at_bound, closed=closed, identified=identified,
                residual_ok=residual_ok, profile=score[:,0], jac_cond=singular[0,0]/max(singular[0,-1],1e-300))


def self_test():
    ts = np.repeat([8, 9, 10], 3).astype(float)
    ti = np.tile([2,3,4], 3).astype(float)
    spacing, gap = .105, .8
    matrix, _ = design(gap*spacing/HBARC, ts, ti, 2)
    true = np.array([.7, -.2, .25, -.2, .18, -.4])
    y = matrix@true
    rng = np.random.default_rng(67241)
    fit = two_state(np.vstack((y, y+rng.normal(0, 1e-5, (NBOOT,len(y))))), ts, ti, spacing, 2)
    np.testing.assert_allclose(fit['pars'][0], [*true,gap], rtol=.003, atol=1e-4)
    assert fit['identified'][0] and fit['residual_ok'][0]
    flat = matrix@np.array([.7, 0, 0, -.2, 0, 0])
    weak = two_state(np.vstack((flat, flat+rng.normal(0, 1e-5, (NBOOT,len(y))))), ts, ti, spacing, 2)
    assert not weak['identified'][0]


def run():
    out = ROOT/'01_bare_two_state'
    out.mkdir(exist_ok=True)
    previous = json.loads((PREVIOUS/'run_receipt.json').read_text())
    plan_digest = digest(BARE/'input_plan.json')
    hashes, rows, paths = {}, [], []
    for ensemble, name in ENSEMBLES.items():
        previous_name = f'01_bare_local_ratio/{name}.npz'
        previous_path = PREVIOUS/previous_name
        assert digest(previous_path)==previous['outputs_sha256'][previous_name]
        with np.load(previous_path, allow_pickle=False) as old:
            meta = {field:old[field] for field in ('z_fm','pz_GeV','rf_fm','tau_over_a2','confs','boot_indices','seed')}
        blocks = []
        for flow_index, tau in enumerate(meta['tau_over_a2']):
            tag = f'tau{tau:.8f}'.replace('.', 'p')
            marker = BARE/ensemble/tag/'status/bootstrap.json'
            receipt = json.loads(marker.read_text())
            assert receipt['status']=='PASS' and receipt['ensemble']==ensemble and receipt['nboot']==NBOOT
            path = Path(receipt['signals'])
            assert path == marker.parents[1]/'00_inputs/signals.npz'
            hashes[str(path)] = digest(path)
            assert hashes[str(path)]==receipt['signals_sha256']
            with np.load(path, allow_pickle=False) as signals:
                for field in ('confs','boot_indices','seed'):
                    np.testing.assert_array_equal(signals[field],meta[field])
                assert str(signals['configuration_manifest_sha256']) == plan_digest
                assert str(signals['ordinary_estimator_id']) == 'four_current_mean_shared_configuration_bootstrap_directional_complex_ratio_then_zpair_v1'
                spacing = float(signals['a_fm'])
                radius = float(signals['rf_fm'])
                assert np.isclose(radius,meta['rf_fm'][flow_index],atol=1e-12,rtol=0)
                momenta = signals['pz_values']
                z = signals['delta_z_values']
                np.testing.assert_allclose(z*spacing, meta['z_fm'], atol=1e-14)
                np.testing.assert_allclose(2*np.pi*HBARC*momenta/(spacing*signals['lattice_size_xyzt'][2]),meta['pz_GeV'],atol=1e-14)
                valid = signals['valid_time_mask']
                tseps, insertion = np.nonzero(valid)
                t = signals['tseps'][tseps]
                np.testing.assert_array_equal(np.unique(t),signals['available_T'])
                assert np.all((t*spacing>.8)&(t*spacing<1.1)&(insertion>=2)&(insertion<=t-2))
                original = signals['ordinary_original']; bootstrap = signals['ordinary_boot']
                assert np.all(np.isnan(original[..., ~valid])) and np.all(np.isnan(bootstrap[..., ~valid]))
                original, bootstrap = original[..., valid], bootstrap[..., valid]
                assert np.all(np.isfinite(original)) and np.all(np.isfinite(bootstrap))
                denominator = signals['ordinary_denominator_stable'][...,valid]
            shape = (len(momenta),len(z))
            pars = np.zeros(shape+(NBOOT+1,7))
            stat = dict.fromkeys(('q','chisq','bound','closed','jac_rank','identified','residual_ok'))
            stat = {key:np.zeros(shape+(NBOOT+1,)) for key in stat}
            profiles = np.zeros(shape+(len(GAPS),))
            accepted = np.zeros(shape,dtype=bool)
            bootstrap_good = np.zeros(shape+(NBOOT,),dtype=bool)
            for p_idx, momentum in enumerate(momenta):
                for z_idx, length in enumerate(z):
                    key = p_idx,z_idx
                    values = np.concatenate((original[key][None,:],bootstrap[:,p_idx,z_idx]),axis=0)
                    components = 1 if momentum==0 or length==0 else 2
                    obs = values.real if components==1 else np.column_stack((values.real,values.imag))
                    fit = two_state(obs,t,insertion,spacing,components)
                    pars[key] = fit['pars']
                    for column in stat:
                        stat[column][key] = fit[column]
                    profiles[key] = fit['profile']
                    fraction = fit['identified'][1:].mean()
                    denominator_ok = denominator[p_idx].all()
                    accepted[key] = denominator_ok and fit['identified'][0] and fit['residual_ok'][0] and fraction>=.9
                    bootstrap_good[key] = denominator_ok & fit['identified'][1:] & fit['residual_ok'][1:]
                    center = fit['pars'][0]; error = fit['pars'][1:].std(axis=0,ddof=1)
                    rows.append(dict(ensemble=name, rf_fm=radius, n=int(momentum), z_fm=length*spacing,
                                     z_over_a=int(length), T=';'.join(map(str,np.unique(t))), Re=center[0], Im=center[3],
                                     Re_sd=error[0], Im_sd=error[3], gap_GeV=center[6], gap_sd=error[6],
                                     chi2=fit['chisq'][0], dof=fit['dof'], Q=fit['q'][0], covariance_rank=fit['covariance_rank'],
                                     jac_rank=fit['jac_rank'][0], column_scaled_jac_cond=fit['jac_cond'],
                                     denominator_safe_count=int(denominator[p_idx].sum()), denominator_total=len(t),
                                     denominator_ok=denominator_ok, gap_closed=fit['closed'][0], gap_bound=fit['bound'][0],
                                     gap_identified_fraction=fraction, residual_ok=fit['residual_ok'][0], quality_accepted=accepted[key]))
            output = out/f'{name}_{tag}.npz'
            np.savez_compressed(output, parameters=pars, parameter_order='c0Re,c1Re,c2Re,c0Im,c1Im,c2Im,gap_GeV',
                                parameter_axes='p,z,original_plus_paired_bootstrap,parameter', gap_scan_GeV=GAPS,
                                center_profile_chi2=profiles, good=accepted, good_boot=bootstrap_good,
                                T=t, tinsertion=insertion, a_fm=spacing, rf_fm=radius, tau_a2=tau,
                                pz_values=momenta, confs=meta['confs'], boot_indices=meta['boot_indices'],
                                origin_plus_boot_ratio=np.concatenate((original[None,...],bootstrap),axis=0),
                                input_sha256=hashes[str(path)], **stat)
            paths.append(output)
            blocks.append(int(accepted.sum()))
            print(name, tag, 'accepted', blocks[-1], '/', accepted.size, flush=True)
        print(name, 'total accepted', sum(blocks), flush=True)
    table = out/'fit_quality.csv'
    write_csv(table,rows)
    paths.append(table)
    record = dict(status='COMPLETE_TWO_STATE_DIAGNOSTIC', model='symmetric_expanded_two_state_c3c2',
                  channel='unpolarized connected u-d', cut=2, open_Ta_fm=[.8,1.1], window='fixed all available T',
                  joint_c2=False, share_gap_across_z=False, share_gap_real_imag=True, nboot=NBOOT,
                  gap_bounds_GeV=[float(GAPS[0]),float(GAPS[-1])], covariance='full bootstrap, correlation eigenmodes >1e-12 of largest',
                  quality='ordinary C2 denominator, Q>=.05, chi2/dof<=3, identified center and >=90% gap-identified replicas',
                  configuration_manifest_sha256=plan_digest, inputs_sha256=hashes,
                  outputs_sha256={str(path.relative_to(ROOT)):digest(path) for path in [*paths,Path(__file__)]})
    (ROOT/'fit_receipt.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__ == '__main__':
    self_test()
    run()
