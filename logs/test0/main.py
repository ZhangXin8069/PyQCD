#!/usr/bin/env python3
"""test0 —— ana_3dir 三方向差异分析测试套件（test12 风格单文件 main.py）。

背景：给 pyqcd 新增「输入数据路径 → 数据分析并作图」功能
（pyqcd.analysis.analyze_3dir，独立实现，功能对齐
refer/huangcl/05_ana_3dir_diff_sem/code_ana_3dir_diff_sem.py）。
本套件参考 /root/PyQCU/logs/test12 与 examples/test0 的形式：
单文件 main.py 多子命令 + 版本目录 v<YYYYMMDDHHMM>/（或 --outdir），
全部产物（json/png/env.json/运行日志）落在版本目录内互不覆盖。

子命令：
    env       环境自检（numpy/matplotlib/pyqcd）
    makedata  生成合成测试数据（含 ground-truth truth.json）→ <outdir>/input/
    run       调用 pyqcd.analysis.analyze_3dir 分析+作图 → 版本目录
    verify    断言验证（产物存在性 + 数值自洽 + 物理自洽）→ test0_verify.json
    check     断言门（verify 全 PASS → exit 0；否则 exit 1）
    collect   汇总版本目录产物清单

运行（仓库根下）：
    python logs/test0/main.py env
    python logs/test0/main.py makedata
    python logs/test0/main.py run --data-root logs/test0/input
    python logs/test0/main.py verify --run-dir <v<ts>>
    python logs/test0/main.py check  --run-dir <v<ts>>
    bash logs/test0/run-local.sh                       # 一键：env→makedata→run→verify→check→collect
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

WORKDIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(WORKDIR, 'input')

# 合成数据配置（与 truth.json 同步，makedata 用）
SYNTH = dict(
    conf_short='L24x72', Px=0, Py=0, Pz=2,
    dt=6, dtau=3, z=0,
    Nsample=500, Ntsep=12, Nz=8,
    corr2_A=1.0, corr2_noise=0.005,
    m=dict(x=1.10, y=1.12, z=1.15),
    ratio_base=dict(x=0.60, y=0.65, z=0.55),
    ratio_sigma=0.02, ratio_peak=3.0, ratio_width=8.0,
    seed=42,
)

TOL_STAT = 1e-8     # 统计量（mean/sem）自洽相对差
TOL_MEFF_ABS = 0.02  # meff 恢复物理值绝对容差（叠加 5·sem）


def dump_env(path):
    """环境快照（test12 env.json 约定）。"""
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
        'numpy': _ver('numpy'), 'scipy': _ver('scipy'),
        'matplotlib': _ver('matplotlib'), 'cupy': _ver('cupy'),
        'pyqcd': _ver('pyqcd'),
        'git_branch': git_branch, 'git_head': git_head,
        'cmdline': ' '.join(sys.argv),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(info, f, indent=2)
    return info


def resolve_outdir(args, default_base=WORKDIR):
    """--outdir > $TEST0_OUTDIR > <default_base>/v<ts>/（test12 约定）。"""
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


# ═══════════════════════════════════════════════════════════════════
# env —— 环境自检
# ═══════════════════════════════════════════════════════════════════

def cmd_env(args):
    print("=== test0 env check ===")
    checks = []
    for name, mod in [('numpy', 'numpy'), ('matplotlib', 'matplotlib'),
                      ('pyqcd', 'pyqcd'), ('scipy', 'scipy')]:
        try:
            m = __import__(mod)
            checks.append((name, 'OK', getattr(m, '__version__', '?')))
        except ImportError:
            checks.append((name, 'MISSING', ''))
    try:
        from pyqcd.analysis import analyze_3dir
        checks.append(('analyze_3dir', 'OK', 'importable'))
    except Exception as e:
        checks.append(('analyze_3dir', 'MISSING', str(e)))
    for name, st, ver in checks:
        print(f"  [{'OK' if st == 'OK' else 'MISSING'}] {name:14s} {ver}")
    all_ok = all(st == 'OK' for _, st, _ in checks)
    print(f"env check: {'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


# ═══════════════════════════════════════════════════════════════════
# makedata —— 合成测试数据（物理合理 + ground-truth）
# ═══════════════════════════════════════════════════════════════════

def make_synth_data(data_dir, cfg=None):
    """生成合成数据：corr2 指数衰减（meff≡m 可解析恢复）、ratio 方向差异。

    目录结构（与 pyqcd.analysis._ana_3dir 约定一致）:
        <data_dir>/<conf>/Pz<Pz>/{x,y,z,ave}_dir/ratio.npy   (Nsample, Ntsep, Ntins, Nz)
        <data_dir>/<conf>/Pz<Pz>/corr2_{x,y,z,ave}.npy       (Nsample, Ntsep)
        <data_dir>/truth.json                                 ground-truth
    """
    cfg = dict(SYNTH, **(cfg or {}))
    rng = np.random.default_rng(cfg['seed'])
    Ns, Nt, Nz = cfg['Nsample'], cfg['Ntsep'], cfg['Nz']
    Ntins = cfg['dt'] + 1
    base = os.path.join(data_dir, cfg['conf_short'], f"Pz{cfg['Pz']}")
    os.makedirs(base, exist_ok=True)

    t = np.arange(Nt, dtype=np.float64)
    corr_parts = []
    for d, m in cfg['m'].items():
        corr = cfg['corr2_A'] * np.exp(-m * t)
        noise = 1.0 + cfg['corr2_noise'] * rng.standard_normal((Ns, Nt))
        c = corr[None, :] * noise
        np.save(os.path.join(base, f"corr2_{d}.npy"), c)
        corr_parts.append(c)
    np.save(os.path.join(base, "corr2_ave.npy"), np.mean(corr_parts, axis=0))

    env_z = np.exp(-((np.arange(Nz) - cfg['ratio_peak']) ** 2) / cfg['ratio_width'])
    ratio_parts = []
    for d, b in cfg['ratio_base'].items():
        ddir = os.path.join(base, f"{d}_dir")
        os.makedirs(ddir, exist_ok=True)
        arr = np.empty((Ns, Nt, Ntins, Nz))
        for s in range(Ns):
            arr[s] = (b * env_z[None, None, :]
                      + cfg['ratio_sigma'] * rng.standard_normal((Nt, Ntins, Nz)))
        np.save(os.path.join(ddir, "ratio.npy"), arr)
        ratio_parts.append(arr)
    os.makedirs(os.path.join(base, "ave_dir"), exist_ok=True)
    np.save(os.path.join(base, "ave_dir", "ratio.npy"),
            np.mean(ratio_parts, axis=0))

    truth = {k: v for k, v in cfg.items()}
    truth['Ntins'] = Ntins
    truth['env_z'] = env_z.tolist()
    truth_path = os.path.join(data_dir, 'truth.json')
    with open(truth_path, 'w') as f:
        json.dump(truth, f, indent=2)
    return truth_path


def cmd_makedata(args):
    data_dir = args.outdir or DEFAULT_DATA_DIR
    truth_path = make_synth_data(data_dir)
    print(f"合成数据 → {data_dir}")
    print(f"truth     → {truth_path}")
    tree = sorted(os.path.relpath(os.path.join(r, f), data_dir)
                  for r, _, fs in os.walk(data_dir) for f in fs)
    for rel in tree:
        print(f"  {rel}")


# ═══════════════════════════════════════════════════════════════════
# run —— 分析 + 作图（调用 pyqcd.analysis.analyze_3dir）
# ═══════════════════════════════════════════════════════════════════

def cmd_run(args):
    data_root = os.path.abspath(args.data_root)
    if not os.path.isdir(data_root):
        print(f"[error] 数据根目录不存在: {data_root}")
        sys.exit(2)
    truth = json.load(open(os.path.join(data_root, 'truth.json')))
    run_dir = resolve_outdir(args)
    print(f"版本目录: {run_dir}")
    dump_env(os.path.join(run_dir, 'env.json'))

    from pyqcd.analysis import AnaParams, analyze_3dir
    params = AnaParams(
        conf_short=truth['conf_short'],
        Px=truth['Px'], Py=truth['Py'], Pz=truth['Pz'],
        dt=truth['dt'], dtau=truth['dtau'], z=truth['z'],
    )
    summary = analyze_3dir(data_root=data_root, out_root=run_dir,
                           params=params, jackknife=args.jackknife)
    print(f"\nrun complete → {run_dir}")
    print(f"time: {summary['time_s']:.2f}s")


# ═══════════════════════════════════════════════════════════════════
# verify —— 断言验证（存在性 + 数值自洽 + 物理自洽）
# ═══════════════════════════════════════════════════════════════════

def _rel_maxdiff(a, b):
    """逐元素相对差最大值（分母为 |b| 的 norm）。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float('inf')
    denom = np.linalg.norm(b)
    if denom == 0:
        return float(np.linalg.norm(a))
    return float(np.linalg.norm(a - b) / denom)


def _jack_sem(vals):
    """独立 jackknife sem（与 pyqcd 实现相互独立验证）。"""
    return float(vals.std(0) * np.sqrt(vals.shape[0] - 1))


def _sem_like(vals, jk):
    """与 pyqcd.analysis.sem 同语义：jackknife 乘 sqrt(N-1)，否则普通 std。"""
    return float(vals.std(0) * (np.sqrt(vals.shape[0] - 1) if jk else 1.0))


def verify_run(run_dir, data_root, results, missing, verbose=True):
    truth = json.load(open(os.path.join(data_root, 'truth.json')))
    conf, Pz = truth['conf_short'], truth['Pz']
    dt, dtau, z = truth['dt'], truth['dtau'], truth['z']
    in_base = os.path.join(data_root, conf, f"Pz{Pz}")
    out_base = os.path.join(run_dir, conf)

    # ── A. 产物存在性 ──
    artifacts = [
        'env.json',
        os.path.join(conf, 'ana_3dir_summary.json'),
        os.path.join(conf, 'ratio', f'hist_ratio_P{Pz}_z{z}_tsep{dt}_tins{dtau}.png'),
        os.path.join(conf, 'corr2', f'hist_corr2_P{Pz}_tsep{dt}.png'),
        os.path.join(conf, 'eff_mass', f'hist_eff_mass_P{Pz}_tsep{dt}.png'),
    ]
    for rel in artifacts:
        ok = os.path.exists(os.path.join(run_dir, rel))
        results.append({'item': f'A:{rel}', 'pass': ok, 'exists': ok})
        if not ok:
            missing.append(rel)

    summary = json.load(open(os.path.join(run_dir, conf, 'ana_3dir_summary.json')))
    jk = bool(summary['jackknife'])

    # ── B. 数值自洽：切片统计量 vs 独立 numpy 重算（rel < 1e-8）──
    ratio = np.load(os.path.join(in_base, 'x_dir', 'ratio.npy'))
    corr2 = np.load(os.path.join(in_base, 'corr2_x.npy'))
    meff = np.log(corr2[:, :-1] / corr2[:, 1:])
    for name, vals, key in [
        ('ratio_x_slice', ratio[:, dt, dtau, z], 'ratio'),
        ('corr2_x_slice', corr2[:, dt], 'corr2'),
        ('eff_mass_x_slice', meff[:, dt], 'eff_mass'),
    ]:
        hist = summary['histograms'][key]['x_dir']
        d_mean = _rel_maxdiff([hist['mean']], [vals.mean()])
        d_sem = _rel_maxdiff([hist['sem']], [_sem_like(vals, jk)])
        results.append({'item': f'B:{name}_mean', 'rel_diff': d_mean,
                        'tol': TOL_STAT, 'pass': d_mean < TOL_STAT,
                        'val': hist['mean'], 'ref': float(vals.mean())})
        results.append({'item': f'B:{name}_sem', 'rel_diff': d_sem,
                        'tol': TOL_STAT, 'pass': d_sem < TOL_STAT,
                        'val': hist['sem'], 'ref': _sem_like(vals, jk)})

    # ── C. 物理自洽：meff 恢复合成 ground-truth m（|dev| < 5·sem_CLT + 0.02）──
    for d in ['x', 'y', 'z']:
        c2 = np.load(os.path.join(in_base, f'corr2_{d}.npy'))
        mf = np.log(c2[:, :-1] / c2[:, 1:])
        m_mean = mf[:, dt].mean()
        m_sem = float(mf[:, dt].std(0) / np.sqrt(mf.shape[0]))   # CLT 标准误
        dev = float(abs(m_mean - truth['m'][d]))
        tol = max(5.0 * m_sem, TOL_MEFF_ABS)
        results.append({'item': f'C:meff_{d}_restore',
                        'dev': dev, 'tol': tol, 'pass': bool(dev < tol),
                        'mean': float(m_mean), 'truth': truth['m'][d],
                        'sem': float(m_sem)})

    # ── D. 相关系数矩阵性质：对称 / 对角 1 / |offdiag| ≤ 1 ──
    corr = np.asarray(summary['correlation']['matrix'])
    sym = _rel_maxdiff(corr, corr.T)
    diag = _rel_maxdiff(np.diag(corr), np.ones(3))
    off = np.abs(corr - np.diag(np.diag(corr))).max()
    results.append({'item': 'D:corr_symmetric', 'rel_diff': sym,
                    'pass': sym < 1e-12})
    results.append({'item': 'D:corr_diag_unit', 'rel_diff': diag,
                    'pass': diag < 1e-12})
    results.append({'item': 'D:corr_offdiag_bounded',
                    'max_offdiag': float(off),
                    'pass': float(off) <= 1.0 + 1e-12})

    if verbose:
        print(f"一致项 {sum(r['pass'] for r in results)}/{len(results)}，"
              f"失败 {sum(not r['pass'] for r in results)}，缺文件 {len(missing)}")
        for r in results:
            mark = 'PASS' if r['pass'] else 'FAIL'
            if 'rel_diff' in r:
                print(f"  {mark} {r['item']:28s} rel={r['rel_diff']:.3e}")
            elif 'dev' in r:
                print(f"  {mark} {r['item']:28s} dev={r['dev']:.4f} "
                      f"(tol {r['tol']:.4f})")
            elif 'max_offdiag' in r:
                print(f"  {mark} {r['item']:28s} max_offdiag={r['max_offdiag']:.4f}")
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
                print(f"  FAIL {r['item']} rel={r.get('rel_diff')}")
        for m in data['missing']:
            print(f"  MISSING {m}")
        sys.exit(1)
    sys.exit(0)


def cmd_collect(args):
    """汇总版本目录产物清单（test12 collect 风格）。"""
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


# ═══════════════════════════════════════════════════════════════════
# 子命令分派
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="test0 —— ana_3dir 三方向差异分析测试套件（test12 风格）")
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('env')
    p.set_defaults(func=cmd_env)

    p = sub.add_parser('makedata')
    p.add_argument('--outdir', default=None, help='合成数据输出目录')
    p.set_defaults(func=cmd_makedata)

    p = sub.add_parser('run')
    p.add_argument('--data-root', default=DEFAULT_DATA_DIR,
                   help='输入数据根目录（含 conf/PzN/ 结构与 truth.json）')
    p.add_argument('--outdir', default=None, help='版本目录（默认 v<ts>/）')
    p.add_argument('--jackknife', action='store_true', help='jackknife SEM')
    p.set_defaults(func=cmd_run)

    p = sub.add_parser('verify')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--data-root', default=DEFAULT_DATA_DIR)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser('check')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--label', default='test0 check')
    p.set_defaults(func=cmd_check)

    p = sub.add_parser('collect')
    p.add_argument('--run-dir', required=True)
    p.set_defaults(func=cmd_collect)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
