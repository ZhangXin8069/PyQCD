#!/usr/bin/env python3
"""test0_energy —— 04_proton_energy 功能测试套件（test12 风格单文件 main.py）。

被测功能：pyqcd.analysis.run_energy（独立实现，功能对齐
refer/huangcl/04_proton_energy/code.py：读 2pt 切片 → 相对时间 → ti 平均 →
重采样 → corr2 → E0 平台拟合 → eff_mass.png（GeV））。

合成数据（makedata）：2pt 切片平移不变（含第二指数），组态噪声 δ +
逐点加性噪声 w（2pt 为 (sink,src) 矩阵，ti 平均后 w 沿不同对角线独立 →
协方差满秩可拟合）→ corr2 可解析，E0 恢复 m。

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
    conf_short='L24x72', conf_name='beta6.20_mu-0.2770_ms-0.2400_L24x72',
    conf_ids=list(range(6200, 6200 + 40 * 200, 200)),
    Nt=32, Nx=8, Px=0, Py=0, Pz=2, Nsample=40, dt_max=12,
    m=1.10, A=1.0, c1=0.2, dE=1.5,
    sigma_eta=0.05, sigma_xi=0.02, seed=42,
    a_fm=0.1053, fm_to_GeV=0.197,
    dt_start=6, dt_end=11,
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
    print("=== test0_energy env check ===")
    checks = []
    for name, mod in [('numpy', 'numpy'), ('matplotlib', 'matplotlib'),
                      ('gvar', 'gvar'), ('lsqfit', 'lsqfit')]:
        try:
            m = __import__(mod)
            checks.append((name, 'OK', getattr(m, '__version__', '?')))
        except ImportError:
            checks.append((name, 'MISSING', ''))
    try:
        from pyqcd.analysis import run_energy
        checks.append(('run_energy', 'OK', 'importable'))
    except Exception as e:
        checks.append(('run_energy', 'MISSING', str(e)))
    for name, st, ver in checks:
        print(f"  [{'OK' if st == 'OK' else 'MISSING'}] {name:12s} {ver}")
    all_ok = all(st == 'OK' for _, st, _ in checks)
    print(f"env check: {'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


# ═══════════════════════════════════════════════════════════════════
# makedata
# ═══════════════════════════════════════════════════════════════════

def make_synth_data(data_dir, cfg=None):
    """合成 2pt 切片：C(t) = A·e^(−m·t)·(1+c1·e^(−dE·t))。

    噪声：组态 δ（乘性）+ 逐点乘性 η（相对噪声，避免大 t 处负值；
    2pt 为 (sink,src) 矩阵，ti 平均后 η 沿不同对角线独立 → 协方差满秩）。
    """
    cfg = dict(SYNTH, **(cfg or {}))
    rng = np.random.default_rng(cfg['seed'])
    Nt = cfg['Nt']
    t_abs = np.arange(Nt, dtype=np.float64)
    base_t = (cfg['A'] * np.exp(-cfg['m'] * t_abs)
              * (1.0 + cfg['c1'] * np.exp(-cfg['dE'] * t_abs)))

    corr_dir = os.path.join(data_dir, cfg['conf_name'], 'momsmear2z')
    for i, cid in enumerate(cfg['conf_ids']):
        delta = cfg['sigma_eta'] * rng.standard_normal()
        eta = cfg['sigma_xi'] * rng.standard_normal((Nt, Nt))
        corr = np.empty((Nt, Nt), dtype=complex)
        for src in range(Nt):
            corr[:, src] = base_t[(np.arange(Nt) - src) % Nt]
        corr = corr * (1.0 + delta) * (1.0 + eta)

        d = os.path.join(corr_dir, str(cid))
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(
            d, f"twopt_slice_pp_Px{cfg['Px']}Py{cfg['Py']}Pz{cfg['Pz']}"
               f"_eginphase2_Cg5g4_nopol_ss_conf{cid}.npy"), corr)

    truth = {k: v for k, v in cfg.items()}
    truth['base_t'] = base_t.tolist()
    with open(os.path.join(data_dir, 'truth.json'), 'w') as f:
        json.dump(truth, f, indent=2)
    return os.path.join(data_dir, 'truth.json')


def cmd_makedata(args):
    data_dir = args.outdir or DEFAULT_DATA_DIR
    tp = make_synth_data(data_dir)
    print(f"合成数据 → {data_dir}")
    print(f"truth     → {tp}")


# ═══════════════════════════════════════════════════════════════════
# run
# ═══════════════════════════════════════════════════════════════════

def build_params(truth):
    from pyqcd.analysis import EnergyParams
    return EnergyParams(
        conf_short=truth['conf_short'], conf_name=truth['conf_name'],
        conf_ids=list(truth['conf_ids']), Nt=truth['Nt'], Nx=truth['Nx'],
        Px=truth['Px'], Py=truth['Py'], Pz=truth['Pz'],
        Nsample=truth['Nsample'], dt_max=truth['dt_max'],
        a=truth['a_fm'], fm_to_GeV=truth['fm_to_GeV'],
        p0={'c0': 0.6, 'c1': 0.6, 'E0': 1.5, 'dE': 0.4},
        dt_start=truth['dt_start'], dt_end=truth['dt_end'],
        xlim=[2.5, truth['dt_max'] - 0.5], ylim=[0.5, 2.0])


def cmd_run(args):
    data_root = os.path.abspath(args.data_root)
    truth = json.load(open(os.path.join(data_root, 'truth.json')))
    run_dir = resolve_outdir(args)
    print(f"版本目录: {run_dir}")
    dump_env(os.path.join(run_dir, 'env.json'))

    from pyqcd.analysis import run_energy
    params = build_params(truth)
    res = run_energy(data_root, run_dir, params, jack=args.jackknife,
                     parts=tuple(int(x) for x in args.parts.split(',')))
    print(f"\nrun complete → {run_dir}")
    print(f"saved {len(res.get('saved', []))} images")


# ═══════════════════════════════════════════════════════════════════
# verify
# ═══════════════════════════════════════════════════════════════════

def _jack_sem(vals):
    return vals.std(0) * np.sqrt(vals.shape[0] - 1)


def verify_run(run_dir, data_root, results, missing, verbose=True):
    truth = json.load(open(os.path.join(data_root, 'truth.json')))
    conf = truth['conf_short']
    out_dir = os.path.join(run_dir, conf, f"_Pz{truth['Pz']}")

    # ---- A. 产物存在性 ----
    for rel in ['env.json', os.path.join(conf, f"_Pz{truth['Pz']}",
                                         '0_corr2.npy'),
                os.path.join(conf, f"_Pz{truth['Pz']}", '1_fit_data.npz'),
                os.path.join(conf, f"_Pz{truth['Pz']}", '2_fit_report.txt'),
                os.path.join(conf, f"_Pz{truth['Pz']}", 'eff_mass.png')]:
        ok = os.path.exists(os.path.join(run_dir, rel))
        results.append({'item': f'A:{rel}', 'pass': bool(ok)})
        if not ok:
            missing.append(rel)

    corr2_path = os.path.join(out_dir, '0_corr2.npy')
    fit_path = os.path.join(out_dir, '1_fit_data.npz')
    if not (os.path.exists(corr2_path) and os.path.exists(fit_path)):
        return sum(not r['pass'] for r in results) == 0 and not missing

    # ---- B. corr2 解析形状：corr2_mean(t) ≈ A·e^(−m·t)·(1+c1·e^(−dE·t)) ----
    corr2 = np.load(corr2_path)                     # (Nsample, dt_max)
    dtmax = truth['dt_max']
    t = np.arange(dtmax, dtype=np.float64)
    base = np.array(truth['base_t'])[:dtmax]
    cm = corr2.mean(0)
    # 标定振幅（数据 / truth 形状最小二乘）
    s = float(np.sum(cm * base) / np.sum(base ** 2))
    dev = np.abs(cm - s * base).max()
    sem_max = _jack_sem(corr2).max()
    tol = 5.0 * sem_max + 0.02 * np.abs(s * base).max()
    results.append({'item': 'B:corr2_analytic_shape', 'dev': float(dev),
                    'tol': float(tol), 'pass': bool(dev < tol),
                    'scale': float(s)})

    # ---- C. E0 恢复：|E0 − m| < 5·sem + 0.05（格点单位）----
    fit = np.load(fit_path)
    E0 = fit['E0']
    dev = abs(float(E0.mean()) - truth['m'])
    sem_e0 = float(_jack_sem(E0))
    tol = 5.0 * sem_e0 + 0.05
    results.append({'item': 'C:E0_restore', 'dev': float(dev), 'tol': float(tol),
                    'pass': bool(dev < tol), 'sem': float(sem_e0)})

    # ---- D. 报告完整性 ----
    rp = os.path.join(out_dir, '2_fit_report.txt')
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
                print(f"  {mark} {r['item']:28s} dev={r['dev']:.4f} "
                      f"(tol {r['tol']:.4f})")
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
        description="test0_energy —— 04_proton_energy 功能测试套件（test12 风格）")
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('env')
    p.set_defaults(func=cmd_env)

    p = sub.add_parser('makedata')
    p.add_argument('--outdir', default=None)
    p.set_defaults(func=cmd_makedata)

    p = sub.add_parser('run')
    p.add_argument('--data-root', default=DEFAULT_DATA_DIR)
    p.add_argument('--outdir', default=None)
    p.add_argument('--jackknife', action='store_true')
    p.add_argument('--parts', default='1,3')
    p.set_defaults(func=cmd_run)

    p = sub.add_parser('verify')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--data-root', default=DEFAULT_DATA_DIR)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser('check')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--label', default='test0_energy check')
    p.set_defaults(func=cmd_check)

    p = sub.add_parser('collect')
    p.add_argument('--run-dir', required=True)
    p.set_defaults(func=cmd_collect)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
