#!/usr/bin/env python3
"""test0_fh —— 06_FH_bare_matele 功能测试套件（test12 风格单文件 main.py）。

被测功能：pyqcd.analysis.run_fh（独立实现，功能对齐
refer/huangcl/06_FH_bare_matele/code_FH_bare_matele.py：6 方向 ratio 平均 →
FH 变换（多 nex）→ FH 图 → 常数模型逐 z 拟合 → 参数图/对比图/bestfit 图）。

合成数据（makedata）：6 方向 ratio 平坦 R = c0_true(z) + 独立噪声 →
FH(t) = ΣR(t+1) − ΣR(t) = c0_true(z) 精确常数，fit 恢复 c0_true。
布局 {data_root}/{conf}/P{4}/{dir}/ratio.npy（参考约定）。

子命令：env / makedata / run / verify / check / collect。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

WORKDIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(WORKDIR, 'input')

SYNTH = dict(
    conf_short='L24x72', P=4,
    Nsample=40, dt_max=12, Nx=8, nexmax=2,
    ave_dirs=['pos_x', 'pos_y', 'pos_z', 'neg_x', 'neg_y', 'neg_z'],
    c0_coeff=0.5, c0_decay=5.0, ratio_noise=0.001, seed=42,
    fit_windows=[(7, 10)], nex_fit=2,
    z_list=list(range(8)),
)


def dump_env(path):
    git_branch = git_head = 'n/a'
    try:
        git_branch = subprocess.run(
            ['git', '-C', ROOT, 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True, text=True).stdout.strip()
        git_head = subprocess.run(
            ['git', '-C', ROOT, 'log', '-1', '--oneline'],
            capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    import importlib
    def _ver(m):
        try:
            return importlib.import_module(m).__version__
        except Exception:
            return None
    info = {
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'hostname': platform.node(),
        'python': platform.python_version(),
        'numpy': _ver('numpy'), 'matplotlib': _ver('matplotlib'),
        'gvar': _ver('gvar'), 'lsqfit': _ver('lsqfit'),
        'git_branch': git_branch, 'git_head': git_head,
        'cmdline': ' '.join(sys.argv),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(info, f, indent=2)
    return info


def resolve_outdir(args, default_base=WORKDIR):
    outdir = args.outdir or os.environ.get('TEST0_OUTDIR')
    if outdir:
        outdir = os.path.abspath(outdir)
    else:
        vdir = os.path.join(default_base, 'v' + datetime.now().strftime('%Y%m%d%H%M'))
        if os.path.exists(vdir):
            vdir = f"{vdir}-{datetime.now().strftime('%S')}"
        outdir = vdir
    os.makedirs(outdir, exist_ok=True)
    return outdir


def cmd_env(args):
    print("=== test0_fh env check ===")
    checks = []
    for name, mod in [('numpy', 'numpy'), ('matplotlib', 'matplotlib'),
                      ('gvar', 'gvar'), ('lsqfit', 'lsqfit')]:
        try:
            m = __import__(mod)
            checks.append((name, 'OK', getattr(m, '__version__', '?')))
        except ImportError:
            checks.append((name, 'MISSING', ''))
    try:
        from pyqcd.analysis import run_fh
        checks.append(('run_fh', 'OK', 'importable'))
    except Exception as e:
        checks.append(('run_fh', 'MISSING', str(e)))
    for name, st, ver in checks:
        print(f"  [{'OK' if st == 'OK' else 'MISSING'}] {name:8s} {ver}")
    all_ok = all(st == 'OK' for _, st, _ in checks)
    print(f"env check: {'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


# ═══════════════════════════════════════════════════════════════════
# makedata —— 6 方向平坦 ratio
# ═══════════════════════════════════════════════════════════════════

def make_synth_data(data_dir, cfg=None):
    """6 方向 ratio：R(dt,dtau,z) = c0_true(z) + 独立噪声（FH 精确 = c0）。"""
    cfg = dict(SYNTH, **(cfg or {}))
    rng = np.random.default_rng(cfg['seed'])
    Ns, dtmax, Nx = cfg['Nsample'], cfg['dt_max'], cfg['Nx']
    z = np.arange(Nx, dtype=np.float64)
    c0_true = cfg['c0_coeff'] * np.exp(-z / cfg['c0_decay'])

    base = os.path.join(data_dir, cfg['conf_short'], f"P{cfg['P']}")
    for d in cfg['ave_dirs']:
        dd = os.path.join(base, d)
        os.makedirs(dd, exist_ok=True)
        ratio = np.empty((Ns, dtmax, dtmax, Nx))
        for dt in range(dtmax):
            for dtau in range(dt + 1):
                ratio[:, dt, dtau, :] = (
                    c0_true[None, :]
                    + cfg['ratio_noise'] * rng.standard_normal((Ns, Nx)))
        np.save(os.path.join(dd, 'ratio.npy'), ratio)

    truth = {k: v for k, v in cfg.items()}
    truth['c0_true'] = c0_true.tolist()
    with open(os.path.join(data_dir, 'truth.json'), 'w') as f:
        json.dump(truth, f, indent=2)
    return os.path.join(data_dir, 'truth.json')


def cmd_makedata(args):
    data_dir = args.outdir or DEFAULT_DATA_DIR
    tp = make_synth_data(data_dir)
    print(f"合成数据 → {data_dir}")
    print(f"truth     → {tp}")


# ═══════════════════════════════════════════════════════════════════
# run —— 全链（委托 pyqcd.analysis.run_fh）
# ═══════════════════════════════════════════════════════════════════

def cmd_run(args):
    data_root = os.path.abspath(args.data_root)
    truth = json.load(open(os.path.join(data_root, 'truth.json')))
    run_dir = resolve_outdir(args)
    print(f"版本目录: {run_dir}")
    dump_env(os.path.join(run_dir, 'env.json'))

    from pyqcd.analysis import FHParams, FitParams, run_fh
    params = FHParams(
        conf_short=truth['conf_short'], P=truth['P'],
        nexmax=truth['nexmax'], ave_dirs=list(truth['ave_dirs']),
        z_list=list(truth['z_list']), z_step=3, xoffset=0.2,
        fh_xlim=[2.5, truth['dt_max'] - 0.5], fh_ylim=[-0.1, 1.1],
        para_xlim=[-0.5, truth['Nx'] - 0.5],
        param_ylim={'c0': [-0.1, 1.0], 'c1': [-0.5, 0.5],
                    'c2': [-0.1, 0.1], 'dE': [0.0, 1.0]})
    fitpa_list = [FitParams(p0={'c0': 0.4}, prior=None,
                            dt_start=a, dt_end=b, nex=truth['nex_fit'])
                  for a, b in truth['fit_windows']]
    bestfit = {'dt_start': truth['fit_windows'][0][0],
               'dt_end': truth['fit_windows'][0][1],
               'nex': truth['nex_fit']}
    res = run_fh(data_root, run_dir, params, fitpa_list,
                 bestfit_params=bestfit,
                 parts=tuple(int(x) for x in args.parts.split(',')))
    print(f"\nrun complete → {run_dir}")
    print(f"saved {len(res['saved'])} images")


# ═══════════════════════════════════════════════════════════════════
# verify
# ═══════════════════════════════════════════════════════════════════

def _jack_sem(vals):
    return vals.std(0) * np.sqrt(vals.shape[0] - 1)


def verify_run(run_dir, data_root, results, missing, verbose=True):
    truth = json.load(open(os.path.join(data_root, 'truth.json')))
    conf = truth['conf_short']
    P = truth['P']
    out_dir = os.path.join(run_dir, conf, f"P{P}")
    c0_true = np.array(truth['c0_true'])
    Nx = truth['Nx']
    dtmax = truth['dt_max']
    z_list = truth['z_list']
    a, b = truth['fit_windows'][0]
    nex = truth['nex_fit']

    # ---- A. 产物存在性 ----
    fh_dir = os.path.join(out_dir, 'fh')
    fit_dir = os.path.join(out_dir, f'fit_nex{nex}')
    window_dir = os.path.join(fit_dir, f'dt{a}_{b}')
    artifacts = ['env.json']
    for n in range(truth['nexmax'] + 1):
        artifacts.append(os.path.join(conf, f'P{P}', 'fh', f'FH_nex{n}.npy'))
        for z in z_list:
            artifacts.append(os.path.join(conf, f'P{P}', 'fh', f'z{z}.png'))
    artifacts += [os.path.join(conf, f'P{P}', f'fit_nex{nex}',
                               f'dt{a}_{b}', f'report_dt{a}_{b}.txt'),
                  os.path.join(conf, f'P{P}', f'fit_nex{nex}',
                               f'dt{a}_{b}', f'fit_dt{a}_{b}.npz'),
                  os.path.join(conf, f'P{P}', f'fit_nex{nex}',
                               f'dt{a}_{b}', 'c0.png'),
                  os.path.join(conf, f'P{P}', f'fit_nex{nex}',
                               f'dt{a}_{b}', 'chi2.png'),
                  os.path.join(conf, f'P{P}', f'fit_nex{nex}', 'c0.png'),
                  os.path.join(conf, f'P{P}', f'fit_nex{nex}', 'chi2.png'),
                  os.path.join(conf, f'P{P}', 'bestfit', f'z{z_list[0]}.png')]
    for rel in artifacts:
        ok = os.path.exists(os.path.join(run_dir, rel))
        results.append({'item': f'A:{rel}', 'pass': bool(ok)})
        if not ok:
            missing.append(rel)

    # ---- B. FH 解析：fh_mean(t,z) ≈ c0_true(z)（有效区 t ≥ 2·nex）----
    fh = np.load(os.path.join(fh_dir, f'FH_nex{nex}.npy'))   # (Nsample, dt, Nz)
    fm = fh.mean(0)
    t_valid = np.arange(2 * nex, dtmax)
    dev = np.abs(fm[t_valid] - c0_true[None, :]).max()
    sem_max = _jack_sem(fh)[t_valid].max()
    tol = 5.0 * sem_max + 0.05
    results.append({'item': 'B:fh_analytic_const', 'dev': float(dev),
                    'tol': float(tol), 'pass': bool(dev < tol)})

    # ---- C. fit 恢复 c0_true ----
    fit = np.load(os.path.join(window_dir, f'fit_dt{a}_{b}.npz'))
    c0 = fit['c0']                                     # (Nsample, Nz)
    dev = np.abs(c0.mean(0) - c0_true)
    sem_c0 = np.array([_jack_sem(c0[:, zz]) for zz in range(Nx)])
    tol = 5.0 * sem_c0 + 0.05
    results.append({'item': 'C:c0_restore', 'max_dev': float(dev.max()),
                    'max_tol': float(tol.max()),
                    'pass': bool(np.all(dev < tol))})

    # ---- D. 报告完整性 ----
    rp = os.path.join(window_dir, f'report_dt{a}_{b}.txt')
    if os.path.exists(rp):
        txt = open(rp).read()
        ok = ('Summary Table' in txt and 'condition number' in txt)
        results.append({'item': 'D:report', 'pass': bool(ok)})

    if verbose:
        n_pass = sum(1 for r in results if r['pass'])
        print(f"一致项 {n_pass}/{len(results)}，失败 "
              f"{sum(not r['pass'] for r in results)}，缺文件 {len(missing)}")
        for r in results:
            mark = 'PASS' if r['pass'] else 'FAIL'
            if 'dev' in r and 'tol' in r:
                print(f"  {mark} {r['item']:24s} dev={r['dev']:.4f} "
                      f"(tol {r['tol']:.4f})")
            elif 'max_dev' in r:
                print(f"  {mark} {r['item']:24s} max_dev={r['max_dev']:.4f} "
                      f"(max_tol {r['max_tol']:.4f})")
            else:
                print(f"  {mark} {r['item']}")
        for m in missing:
            print(f"  MISSING {m}")
    return sum(not r['pass'] for r in results) == 0 and not missing


def cmd_verify(args):
    run_dir = os.path.abspath(args.run_dir)
    data_root = os.path.abspath(args.data_root)
    if not os.path.isdir(run_dir):
        print(f"[error] 运行目录不存在: {run_dir}")
        sys.exit(2)
    if not os.path.exists(os.path.join(data_root, 'truth.json')):
        print(f"[error] 缺少 truth.json（先运行 makedata）: {data_root}")
        sys.exit(2)
    out = {'results': [], 'missing': []}
    ok = verify_run(run_dir, data_root, out['results'], out['missing'],
                    verbose=True)
    out['summary'] = {'n_pass': sum(1 for r in out['results'] if r['pass']),
                      'n_fail': sum(1 for r in out['results'] if not r['pass']),
                      'n_missing': len(out['missing']),
                      'total': len(out['results'])}
    with open(os.path.join(run_dir, 'test0_verify.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"verify: {'PASS' if ok else 'FAIL'} → {run_dir}/test0_verify.json")
    sys.exit(0 if ok else 1)


def cmd_check(args):
    vpath = os.path.join(os.path.abspath(args.run_dir), 'test0_verify.json')
    if not os.path.exists(vpath):
        print(f"[error] 先运行 verify: {vpath} 不存在")
        sys.exit(2)
    data = json.load(open(vpath))
    s = data['summary']
    print(f"[{args.label}] gate: n_fail=0 且 无缺文件")
    print(f"  n_pass={s['n_pass']}/{s['total']}  n_fail={s['n_fail']}  "
          f"missing={s['n_missing']}")
    if s['n_fail'] or s['n_missing']:
        for r in data['results']:
            if not r['pass']:
                print(f"  FAIL {r['item']}")
        for m in data['missing']:
            print(f"  MISSING {m}")
        sys.exit(1)
    sys.exit(0)


def cmd_collect(args):
    run_dir = os.path.abspath(args.run_dir)
    tree = []
    for root, dirs, files in os.walk(run_dir):
        dirs.sort()
        rel = os.path.relpath(root, run_dir)
        for f in sorted(files):
            p = os.path.join(root, f)
            tree.append({'path': os.path.join(rel, f),
                         'bytes': os.path.getsize(p)})
    out = {'run_dir': run_dir, 'n_files': len(tree),
           'total_mb': round(sum(t['bytes'] for t in tree) / 2**20, 1)}
    out['files'] = tree
    with open(os.path.join(run_dir, 'test0_collect.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"n_files={out['n_files']}, total={out['total_mb']} MB → "
          f"{run_dir}/test0_collect.json")


def main():
    ap = argparse.ArgumentParser(
        description="test0_fh —— 06_FH_bare_matele 功能测试套件（test12 风格）")
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('env')
    p.set_defaults(func=cmd_env)

    p = sub.add_parser('makedata')
    p.add_argument('--outdir', default=None)
    p.set_defaults(func=cmd_makedata)

    p = sub.add_parser('run')
    p.add_argument('--data-root', default=DEFAULT_DATA_DIR)
    p.add_argument('--outdir', default=None)
    p.add_argument('--parts', default='1,3')
    p.set_defaults(func=cmd_run)

    p = sub.add_parser('verify')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--data-root', default=DEFAULT_DATA_DIR)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser('check')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--label', default='test0_fh check')
    p.set_defaults(func=cmd_check)

    p = sub.add_parser('collect')
    p.add_argument('--run-dir', required=True)
    p.set_defaults(func=cmd_collect)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
