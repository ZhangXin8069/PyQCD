"""Read-back audit and single-configuration diagnostics; run in a CPU allocation."""
import csv
import json
import os
from pathlib import Path
import traceback
import core as c

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'numerical audit requires Slurm'
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    req = read(ROOT/'request.json')
    contract = req['contract']
    lock_sha = c.sha(ROOT/'source_lock.json')
    errors, rows, accepted, histories = [], [], {}, []
    for name, digest in read(ROOT/'source_lock.json')['sha256'].items():
        if c.sha(ROOT/name) != digest:
            errors.append('source changed: '+name)
    source_ok = not errors
    for spec in req['tasks']:
        tid = spec['task']
        row = dict(task=tid, ensemble=spec['ensemble'], scenario=spec['scenario'],
                   method=spec['method'], status='MISSING', theta=None, iterations=None,
                   kernel_seconds=None, tight_theta=None)
        rows.append(row)
        try:
            candidates = sorted((ROOT/'runs'/f'task{tid:02d}').glob('job*'))
            assert len(candidates) == 1, f'expected one attempt, found {len(candidates)}'
            out = candidates[0]
            row['directory'] = str(out.relative_to(ROOT))
            if not (out/'result.json').exists():
                row['status'] = 'FAILED' if (out/'failed.json').exists() else 'NO_RESULT'
                row['failure'] = read(out/'failed.json') if (out/'failed.json').exists() else None
                continue
            result = read(out/'result.json')
            assert result['task'] == spec
            assert result['request_sha256'] == c.sha(ROOT/'request.json')
            assert result['source_lock_sha256'] == lock_sha
            assert result['input_and_unfixed_flow_unchanged'] is True
            assert result['d_fitted'] is False
            env, inp = read(out/'environment.json'), read(out/'input_verified.json')
            assert env['source_lock_sha256'] == lock_sha and env['partition'] == 'gpu-debug'
            assert inp['input_identity'] == req['ensembles'][spec['ensemble_index']]['identity']
            assert inp['original_request'] == spec and inp['tau'] == spec['tau']
            tiny = read(out/'tiny_contract.json')
            assert tiny['unit_and_pure_gauge_paths'] == 'PASS'
            for value in tiny['algorithms'].values():
                assert value['theta'] < 1.1e-12 and value['loop_error'] < contract['closed_loop_atol']
            arrays = {}
            for path in sorted(out.glob('*.npz')):
                meta = read(path.with_suffix('.json'))
                assert c.sha(path) == meta['npz_sha256'], path.name
                with np.load(path, allow_pickle=False) as data:
                    f, a, loops = (data[k].copy() for k in ('fundamental', 'adjoint', 'closed_loops'))
                    assert f.shape == a.shape == (3, 2, contract['line_nmax']+1)
                    assert loops.shape == (9, 2)
                    assert f.dtype == np.complex128 and a.dtype == loops.dtype == np.float64
                    assert np.array_equal(data['lengths_a'], np.arange(contract['line_nmax']+1))
                    assert np.array_equal(data['directions'], ['x', 'y', 'z'])
                    assert np.array_equal(data['signs'], [1, -1])
                    assert np.array_equal(data['loop_geometry'], c.LOOPS)
                assert all(np.isfinite(v).all() for v in (f, a, loops))
                assert np.max(abs(f[:, :, 0]-1)) == np.max(abs(a[:, :, 0]-1)) == 0
                reversal = max(np.max(abs(f[:, 1]-f[:, 0].conj())), np.max(abs(a[:, 1]-a[:, 0])))
                assert reversal < contract['line_reversal_atol']
                assert max(meta['unitarity_max'], meta['determinant_max']) < contract['su3_atol']
                assert abs(loops[:6, 0].mean()-meta['quda_plaquette'][0]) < contract['closed_loop_atol']
                arrays[path.stem] = (f, a, loops, meta)
            assert 'flow_unfixed' in arrays
            for label, (_, _, loops, meta) in arrays.items():
                assert np.max(abs(loops-arrays['flow_unfixed'][2])) < contract['closed_loop_atol'], label
            for name in ('primary', 'tight'):
                phase = result.get(name)
                if phase is None:
                    continue
                target = contract['theta_primary' if name == 'primary' else 'theta_tight']
                assert phase['theta_target'] == target
                assert phase['converged'] == (phase['final']['theta'] <= 1.1*target)
                hist = phase['history']
                assert hist[0]['iterations'] == 0 and hist[-1]['iterations'] == phase['iterations']
                assert phase['iterations'] <= contract[name+'_max_steps']
                for before, after in zip(hist, hist[1:]):
                    assert after['iterations']-before['iterations'] == after['chunk_steps'] > 0
                assert all(np.isfinite(h['theta']) and h['theta'] >= 0 for h in hist)
                label = name+('_converged' if phase['converged'] else '_unconverged')
                assert arrays[label][3] == result[name+'_measurement']
                assert np.isclose(arrays[label][3]['theta'], phase['final']['theta'], rtol=1e-8, atol=1e-20)
                histories.append((spec, name, hist))
            primary = result['primary']
            passed = primary['converged'] and (not spec['tight_check'] or result['tight']['converged'])
            expected = 'PASS_GPU_NUMERICAL_VALIDATION' if passed else 'INCOMPLETE_CONVERGENCE'
            assert result['status'] == expected
            receipt = read(out/('complete.json' if passed else 'incomplete.json'))
            assert receipt['result_sha256'] == c.sha(out/'result.json') and receipt['task'] == tid
            if not passed:
                ck = read(out/'checkpoint.json')
                assert ck['source_lock_sha256'] == lock_sha
                assert c.sha(out/ck['file']) == ck['sha256']
                gauge = np.load(out/ck['file'], mmap_mode='r', allow_pickle=False)
                assert list(gauge.shape) == ck['shape'] and str(gauge.dtype) == ck['dtype']
                import hashlib
                assert hashlib.sha256(memoryview(gauge).cast('B')).hexdigest() == ck['data_sha256']
                del gauge
            row.update(status=expected, theta=primary['final']['theta'], iterations=primary['iterations'],
                       kernel_seconds=primary['kernel_seconds'],
                       tight_theta=result['tight']['final']['theta'] if result['tight'] else None,
                       closed_loop_max_change=result['primary_measurement']['closed_loop_max_change'],
                       readback='PASS')
            if passed and source_ok:
                accepted[tid] = dict(spec=spec, inp=inp, arrays=arrays)
        except Exception:
            row['status'] = 'AUDIT_FAILED'
            row['error'] = traceback.format_exc()
            errors.append(f'task {tid}: '+row['error'])
    comparisons, differences = [], []
    pairs = [(i, i+1, 'OVR_vs_FFT') for i in range(0, 12, 2)]
    pairs += [(b+2, b+4, 'original_vs_random_OVR') for b in (0, 6)]
    pairs += [(b+3, b+5, 'original_vs_random_FFT') for b in (0, 6)]
    for left, right, kind in pairs:
        pair = dict(left=left, right=right, kind=kind, status='UNAVAILABLE')
        comparisons.append(pair)
        if left not in accepted or right not in accepted:
            continue
        x, y = accepted[left], accepted[right]
        keys = ['raw_data_sha256', 'flow_data_sha256']
        if kind == 'OVR_vs_FFT':
            keys.append('start_data_sha256')
        if any(x['inp'][k] != y['inp'][k] for k in keys):
            pair['status'] = 'INPUT_HASH_MISMATCH'
            errors.append(f'pair {left},{right}: input hashes differ')
            continue
        a, b = x['arrays']['primary_converged'], y['arrays']['primary_converged']
        pair.update(status='DIAGNOSTIC_ONLY', fundamental_max_abs=float(np.max(abs(b[0]-a[0]))),
                    adjoint_max_abs=float(np.max(abs(b[1]-a[1]))),
                    functional_right_minus_left=b[3]['gauge_functional']-a[3]['gauge_functional'])
        for mu in range(3):
            for si, sign in enumerate((1, -1)):
                for n in range(contract['line_nmax']+1):
                    df, da = b[0][mu, si, n]-a[0][mu, si, n], b[1][mu, si, n]-a[1][mu, si, n]
                    differences.append(dict(left=left, right=right, kind=kind, direction='xyz'[mu],
                                            sign=sign, length_a=n, delta_F_real=float(df.real),
                                            delta_F_imag=float(df.imag), delta_A=float(da)))
    with (ROOT/'paired_differences.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['left','right','kind','direction','sign','length_a',
                                                   'delta_F_real','delta_F_imag','delta_A'])
        writer.writeheader()
        writer.writerows(differences)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for spec, name, hist in histories:
        ax = axes[spec['ensemble_index'], 0 if name == 'primary' else 1]
        ax.semilogy([h['iterations'] for h in hist], [max(h['theta'], 1e-30) for h in hist],
                    '.-', label=f"{spec['task']} {spec['method']} {spec['scenario']}")
    for ei in range(2):
        for pi, phase in enumerate(('primary', 'tight')):
            ax = axes[ei, pi]
            ax.axhline(contract['theta_'+phase], color='black', ls='--', lw=.8)
            ax.set(title=req['ensembles'][ei]['name']+' '+phase, xlabel='Iterations in phase', ylabel='theta')
            ax.grid(alpha=.2)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7)
    fig.savefig(ROOT/'convergence.pdf')
    plt.close(fig)
    all_pass = len(accepted) == 12 and not errors and all(p['status'] == 'DIAGNOSTIC_ONLY' for p in comparisons)
    audit = dict(status='PASS_NUMERICAL_VALIDATION' if all_pass else 'INCOMPLETE_OR_FAILED',
                 source_lock_sha256=lock_sha, rows=rows, comparisons=comparisons, errors=errors,
                 d_fitted=False, statistics='one configuration per ensemble; no ensemble uncertainties')
    c.write_json(ROOT/'result_audit.json', audit)
    report = ['# Landau OVR / FFT 数值验证', '', '状态：`'+audit['status']+'`。', '',
              '每个 ensemble 仅 conf10000；不提供统计误差或线性发散系数。差异可能包含 Gribov copy 依赖。', '',
              '|task|ensemble|scenario|method|status|iterations|theta|tight theta|',
              '|---:|---|---|---|---|---:|---:|---:|']
    for r in rows:
        report.append('|'+ '|'.join(str(r[k]) for k in ('task','ensemble','scenario','method','status','iterations','theta','tight_theta'))+'|')
    report += ['', '主目标 theta=1e-10，收紧目标 1e-12，独立验收容限均为 1.1 倍。', '',
               '配对差定义为右任务减左任务，均使用主目标收敛数据；未通过完整验收的任务不参与比较。', '',
               '|left/right|comparison|status|max abs delta F|max abs delta A|', '|---|---|---|---:|---:|']
    for p in comparisons:
        report.append(f"|{p['left']}/{p['right']}|{p['kind']}|{p['status']}|{p.get('fundamental_max_abs', '')}|{p.get('adjoint_max_abs', '')}|")
    report += ['', '完整审计：`result_audit.json`；收敛曲线：`convergence.pdf`；逐长度差：`paired_differences.csv`。',
               '', f'审计错误数：{len(errors)}。缺失、运行失败和未收敛均不得作为通过。']
    (ROOT/'RESULTS.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(dict(status=audit['status'], accepted=len(accepted), errors=len(errors))))
    return 0 if all_pass else 2


if __name__ == '__main__':
    raise SystemExit(main())
