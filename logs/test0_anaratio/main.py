#!/usr/bin/env python3
"""test0_anaratio —— 03_ana_ratio 功能测试套件（test12 风格单文件 main.py）。

被测功能：pyqcd.analysis.ana_ratio_plot_all（独立实现，功能对齐
refer/huangcl/03_ana_ratio/code.py：从 02_ratio 输出只读画图）。
纯画图：单次 fit 图（ratio_z{z}/c0/dE/chi2）、多窗口对比图（cmp_*）、
整体 ratio 图（ratio_z{z}_nofit）。

合成数据（makedata）：按 02_ratio 输出布局生成 ratio.npy + 各拟合窗口
0_fit_data.npz（解析形状 + 噪声）；ground-truth 写入 input/truth.json。

子命令：env / makedata / run / verify / check / collect（test12 风格，
版本目录 v<YYYYMMDDHHMM>/，--outdir > $TEST0_OUTDIR > v<ts>/）。
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
    conf_short='L24x72', Pz=2, Nsample=40, dt_max=8, Nx=8, jack=True,
    fit_ranges=[(3, 6, 1), (4, 6, 1)],
    c0_coeff=0.05, c0_decay=5.0, c1=-0.02, dE=0.5, ratio_noise=0.003,
    fit_noise=0.0005, chi2_mean=1.05,
    z_ylim=[(0, [-0.1, 0.8]), (4, [-0.1, 0.6])],
    seed=42,
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
    print("=== test0_anaratio env check ===")
    checks = []
    for name, mod in [('numpy', 'numpy'), ('matplotlib', 'matplotlib')]:
        try:
            m = __import__(mod)
            checks.append((name, 'OK', getattr(m, '__version__', '?')))
        except ImportError:
            checks.append((name, 'MISSING', ''))
    try:
        from pyqcd.analysis import ana_ratio_plot_all
        checks.append(('ana_ratio_plot_all', 'OK', 'importable'))
    except Exception as e:
        checks.append(('ana_ratio_plot_all', 'MISSING', str(e)))
    for name, st, ver in checks:
        print(f"  [{'OK' if st == 'OK' else 'MISSING'}] {name:18s} {ver}")
    all_ok = all(st == 'OK' for _, st, _ in checks)
    print(f"env check: {'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


# ═══════════════════════════════════════════════════════════════════
# makedata —— 02_ratio 输出布局的合成数据
# ═══════════════════════════════════════════════════════════════════

def make_synth_data(data_dir, cfg=None):
    """生成 ratio.npy + 各拟合窗口 0_fit_data.npz（参考 02_ratio 输出布局）。"""
    cfg = dict(SYNTH, **(cfg or {}))
    rng = np.random.default_rng(cfg['seed'])
    Ns, Nx, dtmax = cfg['Nsample'], cfg['Nx'], cfg['dt_max']
    z = np.arange(Nx, dtype=np.float64)
    c0_true = cfg['c0_coeff'] * np.exp(-z / cfg['c0_decay'])

    base = os.path.join(data_dir, cfg['conf_short'])
    os.makedirs(base, exist_ok=True)

    # ratio.npy（参考文件名契约）
    ratio_name = f"ratio_Pz{cfg['Pz']}_Nsam{Ns}_dtmax{dtmax}.npy"
    ratio = np.empty((Ns, dtmax, dtmax, Nx))
    for dt in range(dtmax):
        for dtau in range(dt + 1):
            base_v = (c0_true[None, :]
                      + cfg['c1'] * (np.exp(-cfg['dE'] * dtau)
                                     + np.exp(-cfg['dE'] * (dt - dtau))))
            ratio[:, dt, dtau, :] = (
                base_v + cfg['ratio_noise'] * rng.standard_normal((Ns, Nx)))
    np.save(os.path.join(base, ratio_name), ratio)

    # 各拟合窗口的 0_fit_data.npz（{c0,c1,dE,chi2: (Nsample, Nx)}）
    for a, b, c in cfg['fit_ranges']:
        fd = (f"fit_Pz{cfg['Pz']}_Nsam{Ns}_dtmax{dtmax}"
              f"_tsep{a}_{b}_nex{c}")
        d = os.path.join(base, fd)
        os.makedirs(d, exist_ok=True)
        np.savez(os.path.join(d, '0_fit_data.npz'),
                 c0=c0_true[None, :] + cfg['fit_noise'] * rng.standard_normal((Ns, Nx)),
                 c1=np.full((Ns, Nx), cfg['c1']) + cfg['fit_noise'] * rng.standard_normal((Ns, Nx)),
                 dE=np.full((Ns, Nx), cfg['dE']) + cfg['fit_noise'] * rng.standard_normal((Ns, Nx)),
                 chi2=cfg['chi2_mean'] + 0.1 * rng.standard_normal((Ns, Nx)))

    truth = {k: v for k, v in cfg.items()}
    truth['c0_true'] = c0_true.tolist()
    truth['ratio_name'] = ratio_name
    with open(os.path.join(data_dir, 'truth.json'), 'w') as f:
        json.dump(truth, f, indent=2)
    return os.path.join(data_dir, 'truth.json')


def cmd_makedata(args):
    data_dir = args.outdir or DEFAULT_DATA_DIR
    tp = make_synth_data(data_dir)
    print(f"合成数据 → {data_dir}")
    print(f"truth     → {tp}")


# ═══════════════════════════════════════════════════════════════════
# run —— 纯画图（委托 pyqcd.analysis.ana_ratio_plot_all）
# ═══════════════════════════════════════════════════════════════════

def cmd_run(args):
    data_root = os.path.abspath(args.data_root)
    truth = json.load(open(os.path.join(data_root, 'truth.json')))
    run_dir = resolve_outdir(args)
    print(f"版本目录: {run_dir}")
    dump_env(os.path.join(run_dir, 'env.json'))

    from pyqcd.analysis import AnaRatioParams, ana_ratio_plot_all
    params = AnaRatioParams(
        conf_short=truth['conf_short'], Pz=truth['Pz'],
        Nsample=truth['Nsample'], dt_max=truth['dt_max'], Nx=truth['Nx'],
        jack=truth['jack'], dt_list=list(range(3, 8)),
        ratio_xlim=[-4, 4], ratio_z_ylim=truth['z_ylim'],
        zval_xlim=[-1, truth['Nx']], c0_ylim=[-0.2, 1.0],
        cmp_ylim={'c0': [-0.2, 0.8], 'dE': [0.5, 1.8], 'chi2': [0, 2]},
        z_step=3)
    fitpa_list = [type('FP', (), dict(dt_start=a, dt_end=b, nex=c))()
                  for a, b, c in truth['fit_ranges']]
    pic_dir = os.path.join(run_dir, truth['conf_short'], f"Pz{truth['Pz']}")
    saved = ana_ratio_plot_all(data_root, pic_dir, params, fitpa_list,
                               plot_mode=args.plot_mode)
    print(f"\nrun complete → {run_dir}")
    print(f"saved {len(saved)} images")


# ═══════════════════════════════════════════════════════════════════
# verify —— 断言
# ═══════════════════════════════════════════════════════════════════

def verify_run(run_dir, data_root, results, missing, verbose=True):
    truth = json.load(open(os.path.join(data_root, 'truth.json')))
    conf = truth['conf_short']
    Nx, Ns, Pz, dtmax = truth['Nx'], truth['Nsample'], truth['Pz'], truth['dt_max']
    z_ylim = truth['z_ylim']

    # ---- A. 产物存在性 ----
    pzdir = os.path.join(conf, f"Pz{Pz}")
    for a, b, c in truth['fit_ranges']:
        sd = f"tsep{a}_{b}_nex{c}"
        for z, _ in z_ylim:
            artifacts = [os.path.join(pzdir, sd, f'ratio_z{z}.png')]
            artifacts += [os.path.join(pzdir, sd, f) for f in
                          ('c0.png', 'dE.png', 'chi2.png')]
            for rel in artifacts:
                ok = os.path.exists(os.path.join(run_dir, rel))
                results.append({'item': f'A:{rel}', 'pass': bool(ok)})
                if not ok:
                    missing.append(rel)
    for qty in ('c0', 'dE', 'chi2'):
        ok = os.path.exists(os.path.join(run_dir, pzdir, f'cmp_{qty}.png'))
        results.append({'item': f'A:cmp_{qty}.png', 'pass': bool(ok)})
        if not ok:
            missing.append(f'cmp_{qty}.png')
    for z, _ in z_ylim:
        ok = os.path.exists(os.path.join(run_dir, pzdir, f'ratio_z{z}_nofit.png'))
        results.append({'item': f'A:ratio_z{z}_nofit.png', 'pass': bool(ok)})
        if not ok:
            missing.append(f'ratio_z{z}_nofit.png')
    ok = os.path.exists(os.path.join(run_dir, 'env.json'))
    results.append({'item': 'A:env.json', 'pass': bool(ok)})

    # ---- B. 数据加载/统计自洽（画图数据源 vs 独立 numpy 重算）----
    from pyqcd.analysis import ana_load_ratio, load_fit_result
    ratio = ana_load_ratio(data_root, conf, Pz, Ns, dtmax)
    rm = ratio.mean(0)
    re = ratio.std(0) * np.sqrt(Ns - 1)      # 独立 jackknife sem
    d_mean = np.abs(rm - np.load(os.path.join(
        data_root, conf, f'ratio_Pz{Pz}_Nsam{Ns}_dtmax{dtmax}.npy')).mean(0)).max()
    results.append({'item': 'B:ratio_mean_load', 'max_dev': float(d_mean),
                    'pass': bool(d_mean < 1e-12)})
    for a, b, c in truth['fit_ranges']:
        fp = type('FP', (), dict(dt_start=a, dt_end=b, nex=c))()
        fit = load_fit_result(data_root, conf, Pz, Ns, dtmax, fp)
        c0 = fit['c0']
        c0_true = np.array(truth['c0_true'])
        dev = np.abs(c0.mean(0) - c0_true).max()
        results.append({'item': f'B:fit_load_c0_{a}_{b}', 'max_dev': float(dev),
                        'pass': bool(dev < 5 * c0.std(0).max() * np.sqrt(Ns - 1)
                                     + 0.01)})
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
        description="test0_anaratio —— 03_ana_ratio 功能测试套件（test12 风格）")
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('env')
    p.set_defaults(func=cmd_env)

    p = sub.add_parser('makedata')
    p.add_argument('--outdir', default=None)
    p.set_defaults(func=cmd_makedata)

    p = sub.add_parser('run')
    p.add_argument('--data-root', default=DEFAULT_DATA_DIR)
    p.add_argument('--outdir', default=None)
    p.add_argument('--plot-mode', type=int, default=1, choices=[1, 2])
    p.set_defaults(func=cmd_run)

    p = sub.add_parser('verify')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--data-root', default=DEFAULT_DATA_DIR)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser('check')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--label', default='test0_anaratio check')
    p.set_defaults(func=cmd_check)

    p = sub.add_parser('collect')
    p.add_argument('--run-dir', required=True)
    p.set_defaults(func=cmd_collect)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
