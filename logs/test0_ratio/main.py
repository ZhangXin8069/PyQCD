#!/usr/bin/env python3
"""test0_ratio —— 02_ratio 功能测试套件（test12 风格单文件 main.py）。

被测功能：pyqcd.analysis.run_ratio2pt（独立实现，功能对齐
refer/huangcl/02_ratio/code.py：2pt+OPE → ratio → 逐 z 拟合 → 图）。
测试框架参考 /root/PyQCU/logs/test12：单文件多子命令 + 版本目录
v<YYYYMMDDHHMM>/（或 --outdir），产物（json/png/npz/txt/env.json）落版本目录。

合成数据（makedata）：2pt 切片平移不变指数衰减 C(t)=A·e^(−m·t)，OPE 形状
o(z)·g(τ)，两者共享组态噪声 η（正相关）→ 真空扣除后 ratio 呈解析形状
σ²·o(z)·g(dtau)，物理自洽可断言；ground-truth 写入 input/truth.json。

子命令：
    env       环境自检
    makedata  生成合成数据（含 truth.json）→ <outdir>/input/
    run       调用 run_ratio2pt 全链 → 版本目录
    verify    断言（A 产物存在性 / B ratio 物理形状 / C fit 平台一致性 / D 报告）
    check     断言门（exit 0/1）
    collect   产物清单
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
    Nt=32, Nx=8, Px=0, Py=0, Pz=2, Nsample=40, dt_max=8,
    m=1.10, A=1.0, sigma_eta=0.10, sigma_xi=0.15, seed=42,
    o_coeff=0.8, o_decay=4.0,           # o(z) = o_coeff·e^(−z/o_decay)
    g_a=0.20, g_b=0.50,                  # g(τ) = 1 + g_a·e^(−g_b·τ)
    fit_ranges=[(3, 6, 1), (4, 6, 1)],   # (dt_start, dt_end, nex)
    plot_z_list=[0, 4],
    # 直接生成 ratio（fit/plot 测试用）：R_true = c0 + c1·(e^−dE·dtau + e^−dE·(dt−dtau))
    c0_coeff=0.05, c0_decay=5.0,         # c0_true(z) = c0_coeff·e^(−z/c0_decay)
    c1=-0.02, dE=0.5, ratio_noise=0.003,
)

TOL_SHAPE = 0.10   # ratio 形状偏差（×o·g 最大值）
TOL_PLATEAU = 0.05  # c0 与窗口均值偏差


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
        'matplotlib': _ver('matplotlib'), 'gvar': _ver('gvar'),
        'lsqfit': _ver('lsqfit'),
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


def cmd_env(args):
    print("=== test0_ratio env check ===")
    checks = []
    for name, mod in [('numpy', 'numpy'), ('matplotlib', 'matplotlib'),
                      ('gvar', 'gvar'), ('lsqfit', 'lsqfit')]:
        try:
            m = __import__(mod)
            checks.append((name, 'OK', getattr(m, '__version__', '?')))
        except ImportError:
            checks.append((name, 'MISSING', ''))
    try:
        from pyqcd.analysis import run_ratio2pt
        checks.append(('run_ratio2pt', 'OK', 'importable'))
    except Exception as e:
        checks.append(('run_ratio2pt', 'MISSING', str(e)))
    for name, st, ver in checks:
        print(f"  [{'OK' if st == 'OK' else 'MISSING'}] {name:14s} {ver}")
    all_ok = all(st == 'OK' for _, st, _ in checks)
    print(f"env check: {'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


# ═══════════════════════════════════════════════════════════════════
# makedata —— 合成数据（物理可解析）
# ═══════════════════════════════════════════════════════════════════

def make_synth_data(data_dir, cfg=None):
    """生成合成 2pt 切片 + OPE（共享组态噪声 η，真空扣除后 ratio 可解析）。

    2pt:  C[sink, src] = A·e^(−m·((sink−src) mod Nt))·(1+η_c)
    OPE:  O[z, τ] = o(z)·g(τ)·(1+η_c)
    → 真空扣除后 ratio ≈ var(η)·o(z)·g(dtau)（解析预言，验证 compute 管道）。
    """
    cfg = dict(SYNTH, **(cfg or {}))
    rng = np.random.default_rng(cfg['seed'])
    Nt, Nx, Nconf = cfg['Nt'], cfg['Nx'], len(cfg['conf_ids'])
    base_dir = os.path.join(data_dir, cfg['conf_short'], 'zdir')
    corr_dir_root = os.path.join(data_dir, cfg['conf_name'], 'momsmear2z')

    z = np.arange(Nx, dtype=np.float64)
    o_z = cfg['o_coeff'] * np.exp(-z / cfg['o_decay'])
    tau = np.arange(Nt, dtype=np.float64)
    g_tau = 1.0 + cfg['g_a'] * np.exp(-cfg['g_b'] * tau)
    t_abs = np.arange(Nt, dtype=np.float64)
    base_t = cfg['A'] * np.exp(-cfg['m'] * t_abs)

    for i, cid in enumerate(cfg['conf_ids']):
        delta = cfg['sigma_eta'] * rng.standard_normal()

        corr = np.empty((Nt, Nt), dtype=complex)
        for src in range(Nt):
            corr[:, src] = base_t[(np.arange(Nt) - src) % Nt]
        corr = corr * (1.0 + delta)

        # 加性独立逐点噪声 w：真空扣除后 ratio ≈ o·g·var(δ) + 独立涨落，
        # 使样本协方差满秩（乘性噪声一阶项会被真空扣除抵消）
        w = rng.standard_normal((Nx, Nt))
        ope_base = o_z[:, None] * g_tau[None, :]
        ope = ope_base * (1.0 + delta) + cfg['sigma_xi'] * w

        d = os.path.join(corr_dir_root, str(cid))
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(
            d, f"twopt_slice_pp_Px{cfg['Px']}Py{cfg['Py']}Pz{cfg['Pz']}"
               f"_eginphase2_Cg5g4_nopol_ss_conf{cid}.npy"), corr)
        d = os.path.join(base_dir, str(cid))
        os.makedirs(d, exist_ok=True)
        # 三个 OPE 取不同系数（模拟真实数据），组合 O = −O30−O31+2·O01 ≠ 0
        for mu, nu, tag, coef in [(0, 1, '01', 1.0), (3, 0, '30', 0.3),
                                  (3, 1, '31', 0.5)]:
            np.savez(os.path.join(
                d, f"ops_mu{mu}_nu{nu}_dz{Nx}_conf{cid}.npz"),
                ops=ope * coef)

    truth = {k: v for k, v in cfg.items()}
    truth['o_z'] = o_z.tolist()
    truth['g_tau'] = g_tau.tolist()
    truth['base_t'] = base_t.tolist()
    truth['c0_true'] = [cfg['c0_coeff'] * np.exp(-zz / cfg['c0_decay'])
                        for zz in z]

    # 直接生成的解析 ratio（fit/plot 测试）：与 fit 模型精确一致
    dtmax = cfg['dt_max']
    rng2 = np.random.default_rng(cfg['seed'] + 1)
    c0_true = np.array(truth['c0_true'])
    fit_ratio = np.empty((cfg['Nsample'], dtmax, dtmax, Nx))
    for dt in range(dtmax):
        for dtau in range(dt + 1):
            base = (c0_true[None, :]
                    + cfg['c1'] * (np.exp(-cfg['dE'] * dtau)
                                   + np.exp(-cfg['dE'] * (dt - dtau))))
            fit_ratio[:, dt, dtau, :] = (
                base + cfg['ratio_noise'] * rng2.standard_normal(
                    (cfg['Nsample'], Nx)))
    np.save(os.path.join(data_dir, 'fit_ratio.npy'), fit_ratio)

    with open(os.path.join(data_dir, 'truth.json'), 'w') as f:
        json.dump(truth, f, indent=2)
    return os.path.join(data_dir, 'truth.json')


def cmd_makedata(args):
    data_dir = args.outdir or DEFAULT_DATA_DIR
    tp = make_synth_data(data_dir)
    print(f"合成数据 → {data_dir}")
    print(f"truth     → {tp}")


# ═══════════════════════════════════════════════════════════════════
# run —— 全链（委托 pyqcd.analysis.run_ratio2pt）
# ═══════════════════════════════════════════════════════════════════

def build_sampa(truth):
    from pyqcd.analysis import SampleParams2pt, PlotParamsRatio
    sampa = SampleParams2pt(
        conf_short=truth['conf_short'], conf_name=truth['conf_name'],
        conf_ids=list(truth['conf_ids']), Nt=truth['Nt'], Nx=truth['Nx'],
        Px=truth['Px'], Py=truth['Py'], Pz=truth['Pz'],
        Nsample=truth['Nsample'], dt_max=truth['dt_max'])
    plotpa = PlotParamsRatio(
        plot_z=0, dt_list=list(range(3, 8)),
        z_list=truth['plot_z_list'],
        xlim=[-4, 4], ylim=[-0.2, 1.0], c0_ylim=[-0.2, 1.0])
    return sampa, plotpa


def cmd_run(args):
    data_root = os.path.abspath(args.data_root)
    truth = json.load(open(os.path.join(data_root, 'truth.json')))
    run_dir = resolve_outdir(args)
    if args.tag:
        run_dir = os.path.join(run_dir, args.tag)
        os.makedirs(run_dir, exist_ok=True)
    print(f"版本目录: {run_dir}")
    dump_env(os.path.join(run_dir, 'env.json'))

    from pyqcd.analysis import FitParams, run_ratio2pt
    import gvar as gv
    sampa, plotpa = build_sampa(truth)
    _prior = {'c0': gv.gvar(0.05, 0.5), 'c1': gv.gvar(-0.02, 1.0),
              'dE': gv.gvar(0.5, 0.3)}
    fitpa_list = [FitParams(p0={'c0': 0.05, 'c1': -0.02, 'dE': 0.5},
                            prior=_prior,
                            dt_start=a, dt_end=b, nex=c)
                  for a, b, c in truth['fit_ranges']]
    parts = tuple(int(x) for x in args.parts.split(','))
    if args.ratio_source:
        # 用预生成的解析 ratio 做 fit/plot（compute 单独验证）
        dst = os.path.join(run_dir, truth['conf_short'],
                           f"ratio_Pz{truth['Pz']}_Nsam{truth['Nsample']}"
                           f"_dtmax{truth['dt_max']}.npy")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        import shutil
        shutil.copy(args.ratio_source, dst)
        print(f"ratio copied: {args.ratio_source} → {dst}")
    res = run_ratio2pt(data_root, run_dir, sampa, fitpa_list, plotpa,
                       jack=args.jackknife, parts=parts)
    print(f"\nrun complete → {run_dir}")
    print(f"saved {len(res['saved'])} images")


# ═══════════════════════════════════════════════════════════════════
# verify —— 断言
# ═══════════════════════════════════════════════════════════════════

def _rel_maxdiff(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float('inf')
    denom = np.linalg.norm(b)
    if denom == 0:
        return float(np.linalg.norm(a))
    return float(np.linalg.norm(a - b) / denom)


def _jack_sem(vals):
    return vals.std(0) * np.sqrt(vals.shape[0] - 1)


def verify_run(run_dir, data_root, results, missing, verbose=True):
    truth = json.load(open(os.path.join(data_root, 'truth.json')))
    conf = truth['conf_short']
    sampa, _ = build_sampa(truth)
    Nx, dt_max = truth['Nx'], truth['dt_max']
    ratio_name = (f"ratio_Pz{truth['Pz']}_Nsam{truth['Nsample']}"
                  f"_dtmax{truth['dt_max']}.npy")
    fit_dirs = [f"fit_Pz{truth['Pz']}_Nsam{truth['Nsample']}"
                f"_dtmax{truth['dt_max']}_tsep{a}_{b}_nex{c}"
                for a, b, c in truth['fit_ranges']]

    # 阶段子目录：compute/（管道）与 fit/（解析 ratio 拟合）
    subs = [d for d in ('compute', 'fit')
            if os.path.isdir(os.path.join(run_dir, d))]
    targets = subs or [run_dir]
    for sub in targets:
        base = os.path.join(run_dir, sub) if subs else run_dir
        is_compute = (not subs) or sub == 'compute'

        # ---- A. 产物存在性 ----
        if is_compute:
            artifacts = [os.path.join(conf, ratio_name), 'env.json']
        else:
            artifacts = ['env.json']
            for fd in fit_dirs:
                artifacts += [os.path.join(conf, fd, '0_fit_data.npz'),
                              os.path.join(conf, fd, '1_fit_report.txt'),
                              os.path.join(conf, fd, 'ratio.png'),
                              os.path.join(conf, fd, 'c0.png'),
                              os.path.join(conf, fd, 'chi2.png')]
        for rel in artifacts:
            ok = os.path.exists(os.path.join(base, rel))
            results.append({'item': f'A[{sub or "run"}]:{rel}',
                            'pass': bool(ok), 'exists': bool(ok)})
            if not ok:
                missing.append(f"{sub}/{rel}")

        ratio_path = os.path.join(base, conf, ratio_name)
        if not os.path.exists(ratio_path):
            continue

        # ---- B. compute 管道：ratio 平均形状 ≈ σ²_hat·o(z)·g(dtau) ----
        if is_compute:
            ratio = np.load(ratio_path)
            o_z = np.array(truth['o_z'])
            g_tau = np.array(truth['g_tau'])
            rm = ratio.mean(0)
            ref = np.broadcast_to(
                o_z[None, None, :] * g_tau[None, :dt_max, None], rm.shape)
            # 只对合法区域（dtau ≤ dt）标定
            tri = np.broadcast_to(
                np.tril(np.ones((dt_max, dt_max)), 0)[:, :, None].astype(bool),
                rm.shape)
            mask = (ref != 0) & tri
            s2_hat = float(np.sum(rm[mask] * ref[mask])
                           / np.sum(ref[mask] ** 2))
            scale = np.sqrt(max(s2_hat, 0.0))
            dev = np.abs(rm - s2_hat * ref)[mask].max()
            sem_max = np.abs(ratio).std(0)[rm.shape[0] // 2].max()
            tol = max(5.0 * sem_max / scale, TOL_SHAPE) * scale
            results.append({'item': 'B[compute]:ratio_analytic_shape',
                            'dev': float(dev), 'tol': float(tol),
                            'pass': bool(dev < tol),
                            'sigma2_hat': float(s2_hat)})

        # ---- C. fit 参数恢复：c0(z) ≈ c0_true(z) ----
        c0_true = np.array(truth.get('c0_true', []))
        for fd in fit_dirs:
            fp_path = os.path.join(base, conf, fd, '0_fit_data.npz')
            if not os.path.exists(fp_path):
                continue
            fit = np.load(fp_path)
            c0 = fit['c0']                                  # (Nsample, Nx)
            if len(c0_true) == Nx:
                dev = np.abs(c0.mean(0) - c0_true)
                sem_c0 = np.array([_jack_sem(c0[:, zz]) for zz in range(Nx)])
                tol = 5.0 * sem_c0 + TOL_PLATEAU
                results.append({'item': f'C[{sub or "run"}]:c0_restore_{fd[-8:]}',
                                'max_dev': float(dev.max()),
                                'max_tol': float(tol.max()),
                                'pass': bool(np.all(dev < tol))})

        # ---- D. 报告完整性 ----
        for fd in fit_dirs:
            rp = os.path.join(base, conf, fd, '1_fit_report.txt')
            if os.path.exists(rp):
                txt = open(rp).read()
                ok = ('Summary Table' in txt and 'condition number' in txt
                      and 'z = 0' in txt)
                results.append({'item': f'D[{sub or "run"}]:report_{fd[-8:]}',
                                'pass': bool(ok)})

    if verbose:
        n_pass = sum(1 for r in results if r['pass'])
        print(f"一致项 {n_pass}/{len(results)}，失败 "
              f"{sum(not r['pass'] for r in results)}，缺文件 {len(missing)}")
        for r in results:
            mark = 'PASS' if r['pass'] else 'FAIL'
            if 'dev' in r and 'tol' in r:
                print(f"  {mark} {r['item']:52s} dev={r['dev']:.4f} "
                      f"(tol {r['tol']:.4f})")
            elif 'max_dev' in r:
                print(f"  {mark} {r['item']:52s} max_dev={r['max_dev']:.4f} "
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


# ═══════════════════════════════════════════════════════════════════
# 子命令分派
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="test0_ratio —— 02_ratio 功能测试套件（test12 风格）")
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
    p.add_argument('--parts', default='1,3',
                   help='起始,结束 part（1=ratio 计算, 2=fit, 3=plot）')
    p.add_argument('--ratio-source', default=None,
                   help='预生成 ratio.npy 路径（fit/plot 用，跳过 compute）')
    p.add_argument('--tag', default=None,
                   help='版本目录内子目录名（compute/fit 分阶段）')
    p.set_defaults(func=cmd_run)

    p = sub.add_parser('verify')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--data-root', default=DEFAULT_DATA_DIR)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser('check')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--label', default='test0_ratio check')
    p.set_defaults(func=cmd_check)

    p = sub.add_parser('collect')
    p.add_argument('--run-dir', required=True)
    p.set_defaults(func=cmd_collect)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
