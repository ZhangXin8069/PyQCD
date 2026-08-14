#!/usr/bin/env python3
"""
test0 —— docker-v20260805 全量蒸馏 GPU 管线一致性测试（test12 形式）
========================================================================

目标：在 logs/test0 复现 /root/PyQCD/examples/docker-v20260805/output/
output_20260802_120104 的完整管线结果（10 组态蒸馏：vertex → 2pt/3pt/4pt →
OPE → analysis → plots → LaTeX 报告），中间数据与图表与基线一样完整保存。

实现：自包含照抄 docker-v20260805 管线（config/utils/lib/compute_*/analyze/
report 均为本地副本，不 import examples/），总体形式参照 PyQCU/logs/test12：
代码文件位于根目录不入版本目录，运行产物全部进版本目录 v<YYYYMMDDHHMM>/。

子命令（--outdir 为公共参数，位置在子命令前后皆可）：
    env      环境自检（GPU/CuPy/git），并写 env.json
    pipeline 完整管线（--steps 可选；--conf-ids/--precision/--Nev1/--skip-* 透传）
    verify   数值一致性验证 vs 基线 output_20260802_120104（rtol=1e-3）
    collect  汇总 timing/meff/文件清单 → test0_results.json
    report   生成并编译 LaTeX 物理报告（physics_report.tex/.pdf）

--outdir 优先级：命令行 > TEST0_OUTDIR 环境变量 > logs/test0/。
每次调用自动在输出目录写 env.json（test12 约定）。
"""

from __future__ import annotations

import argparse, json, os, subprocess, sys, time, traceback
from datetime import datetime
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import config as C
from utils import setup_logging, print_banner, Timer, dump_config_snapshot, HAS_CUPY

REF_DIR = '/root/PyQCD/examples/docker-v20260805/output/output_20260802_120104'
VERSION = 'docker-v20260805'


# ═══════════════════════════════════════════════════════════════════
# 公共：输出目录 / env.json（test12 约定）
# ═══════════════════════════════════════════════════════════════════

def resolve_outdir(args) -> str:
    d = getattr(args, 'outdir', None) or os.environ.get('TEST0_OUTDIR') \
        or str(_SCRIPT_DIR)
    os.makedirs(d, exist_ok=True)
    return d


def write_env_json(out: str, cmdline: list):
    """环境快照（比对基准）：GPU 型号/显存/驱动、cupy、python、git HEAD、命令。"""
    env = {
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'hostname': os.uname().nodename,
    }
    try:
        if HAS_CUPY:
            import cupy as cp
            dev = cp.cuda.Device(); props = cp.cuda.runtime.getDeviceProperties(dev.id)
            env['gpu_name'] = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
            free, total = cp.cuda.runtime.memGetInfo()
            env['gpu_mem'] = f"{total/2**20:.0f} MiB"
            env['cupy'] = cp.__version__
            env['cuda_runtime'] = cp.cuda.runtime.runtimeGetVersion()
    except Exception as e:
        env['gpu_error'] = str(e)
    try:
        import subprocess
        env['git_branch'] = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True, text=True, cwd=_SCRIPT_DIR).stdout.strip()
        env['git_head'] = subprocess.run(
            ['git', 'log', '-1', '--oneline'],
            capture_output=True, text=True, cwd=_SCRIPT_DIR).stdout.strip()
    except Exception:
        pass
    env['python'] = sys.version.split()[0]
    env['cmdline'] = ' '.join(cmdline)
    with open(os.path.join(out, 'env.json'), 'w') as f:
        json.dump(env, f, indent=2)
    return env


# ═══════════════════════════════════════════════════════════════════
# 子命令 1: env
# ═══════════════════════════════════════════════════════════════════

def cmd_env(args):
    out = resolve_outdir(args)
    print(f"输出目录: {out}")
    print(f"Python {sys.version.split()[0]} | numpy {np.__version__}")
    print(f"基线: {REF_DIR}")
    print(f"配置: {C.CONF_IDS} (Nconf={len(C.CONF_IDS)}), "
          f"Nev={C.NEV}, Nev1={C.NEV1}, precision={C.PRECISION}, "
          f"lattice={C.NT}x{C.NX}^3, a={C.ALttc} fm")
    ok = True
    for cid in C.CONF_IDS:
        e = os.path.isdir(os.path.dirname(C.get_eigen_path(cid, 0)))
        p = os.path.isdir(C.get_peram_dir(cid))
        g = os.path.exists(C.get_gauge_path(cid))
        ok &= e and p and g
        print(f"  conf={cid}: eigvec={'OK' if e else 'MISS'} "
              f"peram={'OK' if p else 'MISS'} gauge={'OK' if g else 'MISS'}")
    if HAS_CUPY:
        import cupy as cp
        print(f"CuPy {cp.__version__} | CUDA {cp.cuda.runtime.runtimeGetVersion()}")
    write_env_json(out, sys.argv)
    print("env OK" if ok else "env DATA MISSING")
    return 0 if ok else 1


# ═══════════════════════════════════════════════════════════════════
# 子命令 2: pipeline（照抄 run_pipeline.py 的步骤与驱动逻辑）
# ═══════════════════════════════════════════════════════════════════

def _step_env(config, logger):
    print_banner("Step 0: Environment Check", logger)
    env = {'ok': True}
    logger.info(f"Python {sys.version.split()[0]}")
    logger.info(f"Configs: {config['conf_ids']} (Nconf={len(config['conf_ids'])})")
    logger.info(f"Precision: {config['precision']}, Nev={C.NEV}, Nev1={config['Nev1']}")
    if HAS_CUPY:
        import cupy as cp
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
        free, total = cp.cuda.runtime.memGetInfo()
        env['gpu'] = name
        logger.info(f"GPU: {name} | free={free/2**30:.1f}/{total/2**30:.1f} GB")
        logger.info(f"CuPy {cp.__version__} | CUDA {cp.cuda.runtime.runtimeGetVersion()}")
    else:
        logger.warning("NO GPU — falling back to CPU (slow)")
    for cid in config['conf_ids']:
        e = os.path.isdir(os.path.dirname(C.get_eigen_path(cid, 0)))
        p = os.path.isdir(C.get_peram_dir(cid))
        g = os.path.exists(C.get_gauge_path(cid))
        logger.info(f"  conf={cid}: eigvec={'OK' if e else 'MISS'} "
                    f"peram={'OK' if p else 'MISS'} gauge={'OK' if g else 'MISS'}")
        env['ok'] &= e and p and g
    return env


def _step_vertex(config, run_dir, logger):
    print_banner("Step 1: Vertex Functions (VdV, VVV)", logger)
    from compute_vertex import compute_all_vertices
    compute_all_vertices(config['conf_ids'], run_dir, logger,
                         config['precision'], recompute=False)
    return None


def _step_2pt(config, run_dir, logger):
    print_banner("Step 2: 2pt Correlators (pp, pn, pion)", logger)
    from compute_contraction import compute_2pt_all
    data = compute_2pt_all(config['conf_ids'], run_dir, logger,
                           vertices=None, precision=config['precision'],
                           channels=config.get('channels', ('pp', 'pn', 'pion')))
    return data


def _step_ope(config, run_dir, logger):
    print_banner("Step 3: OPE (gluon operator)", logger)
    from compute_ope import compute_ope_all
    data = compute_ope_all(config['conf_ids'], run_dir, logger,
                           config['precision'])
    return data


def _step_3pt(config, run_dir, logger):
    print_banner("Step 4: 3pt Correlators (PJN)", logger)
    from compute_contraction import compute_3pt_all
    data = compute_3pt_all(config['conf_ids'], run_dir, logger,
                           vertices=None, precision=config['precision'])
    return data


def _step_4pt(config, run_dir, logger):
    print_banner("Step 5: 4pt Correlators (PJNNJNp)", logger)
    from compute_contraction import compute_4pt_all
    data = compute_4pt_all(
        config['conf_ids'], run_dir, logger,
        vertices=None, precision=config['precision'],
        t_sep=config.get('fourpt_tsep', C.FOURPT_TSEP),
        nev1=config.get('fourpt_nev1', C.FOURPT_NEV1),
        momenta=config.get('fourpt_mom', C.FOURPT_MOM),
        src_step=config.get('fourpt_src_step', C.FOURPT_SRC_STEP))
    return data


def _load_2pt(run_dir, logger):
    corr = {}
    for cid in os.listdir(os.path.join(run_dir, 'data')):
        if not cid.startswith('conf'):
            continue
        cid = cid[4:]
        cdir = os.path.join(run_dir, 'data', f'conf{cid}')
        entry = {}
        for f in os.listdir(cdir):
            if f.startswith('corr_') and f.endswith('.npy'):
                key = f[5:].replace(f'_{cid}.npy', '')
                entry[f'corr_{key}'] = np.load(os.path.join(cdir, f))
        if entry:
            corr[int(cid)] = entry
    logger.info(f"Loaded 2pt correlators for {len(corr)} configs")
    return corr


def _load_3pt(run_dir, logger):
    corr = {}
    for cid in os.listdir(os.path.join(run_dir, 'data')):
        if not cid.startswith('conf'):
            continue
        cid = cid[4:]
        cdir = os.path.join(run_dir, 'data', f'conf{cid}')
        entry = {}
        for f in os.listdir(cdir):
            if '_3pt_' in f and f.endswith('.npy') and 'pjnnjnp' not in f:
                key = f[:-4].replace(f'_{cid}', '')
                entry[key] = np.load(os.path.join(cdir, f))
        if entry:
            corr[int(cid)] = entry
    if corr:
        logger.info(f"Loaded 3pt correlators for {len(corr)} configs")
    return corr


def _load_ope(run_dir, logger):
    ope = {}
    for cid in os.listdir(os.path.join(run_dir, 'data')):
        if not cid.startswith('conf'):
            continue
        cid = cid[4:]
        cdir = os.path.join(run_dir, 'data', f'conf{cid}')
        comb = os.path.join(cdir, f'ope_combined_conf{cid}.npy')
        if os.path.exists(comb):
            ope[int(cid)] = {'combined': np.load(comb)}
    if ope:
        logger.info(f"Loaded combined OPE for {len(ope)} configs")
    return ope


def _step_analysis(config, run_dir, logger):
    print_banner("Step 6: Statistical Analysis (Jackknife/meff/ratio_3p)", logger)
    from analyze import (run_meff_jackknife, run_connected_ratio,
                         run_disconnected_ratio)
    corr2 = _load_2pt(run_dir, logger)
    meff_res = run_meff_jackknife(corr2, config['conf_ids'], run_dir, logger)

    corr3 = _load_3pt(run_dir, logger)
    if corr3:
        ratio_conn = run_connected_ratio(corr2, corr3, config['conf_ids'],
                                         run_dir, logger)
    else:
        ratio_conn = {}
        logger.warning("No 3pt data — skipping connected ratio")

    ope = _load_ope(run_dir, logger)
    if ope:
        ratio_disc = run_disconnected_ratio(corr2, ope, config['conf_ids'],
                                            run_dir, logger)
    else:
        ratio_disc = {}
        logger.warning("No OPE data — skipping disconnected ratio")

    return {'meff': meff_res, 'connected_ratio': ratio_conn,
            'disconnected_ratio': ratio_disc}


def _step_plots(config, run_dir, logger, meff_res=None, ratio_conn=None):
    print_banner("Step 7: Plots", logger)
    from analyze import plot_meff_results, plot_correlators, plot_connected_ratio
    if meff_res is None:
        an_dir = os.path.join(run_dir, 'data', 'analysis')
        from analyze import CHANNELS
        meff_res = {}
        for particle, mom, key in CHANNELS:
            fm = os.path.join(an_dir, f'meff_{particle}_{mom}_mean.npy')
            fe = os.path.join(an_dir, f'meff_{particle}_{mom}_err.npy')
            if os.path.exists(fm):
                m = np.load(fm); e = np.load(fe)
                ps, pe = (4, min(C.NT - 2, 14)) if particle == 'proton' else (5, min(C.NT - 2, 18))
                mask = np.isfinite(m[ps:pe]) & (e[ps:pe] > 0) & (m[ps:pe] > 0.01)
                w = 1.0 / (e[ps:pe][mask] ** 2 + 1e-10)
                meff_res[f'{particle}_{mom}'] = {
                    'meff_mean': m, 'meff_err': e, 'plateau': (ps, pe),
                    'E0': float(np.sum(m[ps:pe][mask] * w) / np.sum(w)),
                    'E0_err': float(1 / np.sqrt(np.sum(w))),
                    'E_exp': 1.0 if particle == 'proton' else 0.30,
                    'corr_mean': np.load(os.path.join(an_dir, f'corr_{particle}_{mom}_mean.npy')),
                    'corr_err': np.load(os.path.join(an_dir, f'corr_{particle}_{mom}_err.npy')),
                }
    plot_meff_results(meff_res, run_dir, logger)
    plot_correlators(meff_res, run_dir, logger)
    if ratio_conn:
        plot_connected_ratio(ratio_conn, run_dir, logger)
    return meff_res


def _step_report(config, run_dir, logger, meff_res, timing):
    print_banner("Step 8: Analysis Summary JSON", logger)
    summary = {
        'version': VERSION,
        'conf_ids': config['conf_ids'],
        'precision': config['precision'],
        'nev': C.NEV, 'nev1': config['Nev1'],
        'lattice': [C.NT, C.NX, C.NX, C.NX], 'alttc': C.ALttc,
        'meff': {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating))
                     else (list(vv) if isinstance(vv, tuple) else None)
                     for kk, vv in v.items() if kk in ('E0', 'E0_err', 'E_exp', 'dev', 'plateau', 'npts')}
                 for k, v in meff_res.items()},
        'timing_s': timing,
        'run_dir': run_dir,
    }
    with open(os.path.join(run_dir, 'analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Analysis summary -> {run_dir}/analysis_summary.json")
    return summary


def cmd_pipeline(args):
    out = resolve_outdir(args)
    config = {
        'precision': args.precision,
        'Nev1': min(args.Nev1, C.NEV),
        'channels': tuple(args.channels.split(',')),
        'verbose': args.verbose,
    }
    config['conf_ids'] = [args.conf_id] if args.conf_id else C.CONF_IDS
    if args.conf_ids:
        config['conf_ids'] = [int(x) for x in args.conf_ids.split(',')]
    config['Nev1'] = min(config['Nev1'], C.NEV)
    if args.fourpt_nev1:
        config['fourpt_nev1'] = args.fourpt_nev1
    if args.fourpt_tsep:
        config['fourpt_tsep'] = args.fourpt_tsep

    run_dir = os.path.join(out, f'output_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    if args.run_dir:
        run_dir = os.path.join(out, args.run_dir)
    for d in ['data', 'analysis', 'plots']:
        os.makedirs(os.path.join(run_dir, d), exist_ok=True)

    logger = setup_logging(C.AGENT_LOGS_DIR, name=f'test0-pipeline-{os.path.basename(run_dir)}',
                           verbose=config['verbose'])
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Config: {config}")
    dump_config_snapshot(config, os.path.join(run_dir, 'run_config.json'), logger)
    write_env_json(out, sys.argv)

    steps = ['env', 'vertex', '2pt', 'ope', '3pt', '4pt',
             'analysis', 'plots', 'report']
    if args.steps != 'all':
        steps = [s.strip() for s in args.steps.split(',')]
    if args.skip_2pt: steps = [s for s in steps if s != '2pt']
    if args.skip_ope: steps = [s for s in steps if s != 'ope']
    if args.skip_3pt: steps = [s for s in steps if s != '3pt']
    if args.skip_4pt: steps = [s for s in steps if s != '4pt']
    if args.skip_analysis: steps = [s for s in steps if s != 'analysis']
    if args.skip_plots: steps = [s for s in steps if s != 'plots']
    if args.skip_report: steps = [s for s in steps if s != 'report']

    timing = {}
    meff_res, ratio_conn, env = None, None, None
    total_start = time.perf_counter()
    try:
        for step in steps:
            tmr = Timer(f"STEP {step}", logger)
            tmr.__enter__()
            try:
                if step == 'env':
                    env = _step_env(config, logger)
                elif step == 'vertex':
                    _step_vertex(config, run_dir, logger)
                elif step == '2pt':
                    _step_2pt(config, run_dir, logger)
                elif step == 'ope':
                    _step_ope(config, run_dir, logger)
                elif step == '3pt':
                    _step_3pt(config, run_dir, logger)
                elif step == '4pt':
                    _step_4pt(config, run_dir, logger)
                elif step == 'analysis':
                    analysis = _step_analysis(config, run_dir, logger)
                    meff_res = analysis['meff']
                    ratio_conn = analysis['connected_ratio']
                elif step == 'plots':
                    meff_res = _step_plots(config, run_dir, logger, meff_res, ratio_conn)
                elif step == 'report':
                    _step_report(config, run_dir, logger, meff_res, timing)
            finally:
                tmr.__exit__(None, None, None)
            timing[step] = round(tmr.elapsed, 1)

        total_t = time.perf_counter() - total_start
        print_banner(f"Pipeline Complete! Total {total_t:.0f}s "
                     f"({total_t/60:.1f} min)", logger)
        logger.info(f"Run directory: {run_dir}")
        return 0
    except Exception as e:
        logger.error(f"PIPELINE FAILED: {e}")
        logger.error(traceback.format_exc())
        return 1


# ═══════════════════════════════════════════════════════════════════
# 子命令 3: verify — 数值一致性验证 vs 基线
# ═══════════════════════════════════════════════════════════════════

def _cmp_array(name, a, b, rtol, atol):
    """比较两个数组，返回 (ok, max_rel, max_abs)。ok=None 表示缺失。"""
    if a is None or b is None:
        return (None, None, None)
    a = np.asarray(a); b = np.asarray(b)
    if a.shape != b.shape:
        return (False, None, None)
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = np.maximum(np.abs(b), 1e-30)
        rel = np.abs(a - b) / denom
    rel = np.where(np.abs(b) < 1e-30, np.abs(a - b), rel)
    return (bool(np.all(rel < rtol) or np.all(np.abs(a - b) < atol)),
            float(np.max(rel)), float(np.max(np.abs(a - b))))


def cmd_verify(args):
    out = resolve_outdir(args)
    ref = args.ref or REF_DIR
    print(f"基线: {ref}")
    print(f"本次: {out}")

    items = []
    for kind in ('analysis', 'configs', 'ope', '3pt', '4pt'):
        items.append((kind, []))

    # 1) data/analysis 统计数组（meff/corr/ratio）
    an_files = ['meff_proton_P0_mean.npy', 'meff_proton_P0_err.npy',
                'meff_proton_P2_mean.npy', 'meff_proton_P2_err.npy',
                'meff_pion_P0_mean.npy', 'meff_pion_P0_err.npy',
                'meff_pion_P2_mean.npy', 'meff_pion_P2_err.npy',
                'corr_proton_P0_mean.npy', 'corr_proton_P0_err.npy',
                'corr_proton_P2_mean.npy', 'corr_proton_P2_err.npy',
                'corr_pion_P0_mean.npy', 'corr_pion_P0_err.npy',
                'corr_pion_P2_mean.npy', 'corr_pion_P2_err.npy',
                'ratio_proton_P0_mean.npy', 'ratio_proton_P0_err.npy',
                'ratio_proton_P2_mean.npy', 'ratio_proton_P2_err.npy',
                'ratio_pion_P0_mean.npy', 'ratio_pion_P0_err.npy',
                'ratio_pion_P2_mean.npy', 'ratio_pion_P2_err.npy']
    rtol = args.rtol
    atol = args.atol
    for f in an_files:
        pa = os.path.join(ref, 'data', 'analysis', f)
        pb = os.path.join(out, 'data', 'analysis', f)
        a = np.load(pa) if os.path.exists(pa) else None
        b = np.load(pb) if os.path.exists(pb) else None
        ok, rel, _ = _cmp_array(f, a, b, rtol, atol)
        items[0][1].append((f, ok, rel))
        print(f"  [{'PASS' if ok else 'FAIL'}] {f}  max_rel={rel:.2e}" if rel is not None
              else f"  [MISS] {f}")

    # 2) 每组态 2pt / 3pt / 4pt / OPE 数组
    for cid in C.CONF_IDS:
        for f in ['corr_pp_P0', 'corr_pp_P2', 'corr_pn_P0', 'corr_pn_P2',
                  'corr_pion_P0', 'corr_pion_P2',
                  'proton_P0_3pt', 'proton_P2_3pt',
                  'pion_P0_3pt', 'pion_P2_3pt', 'pjnnjnp_4pt',
                  'ope_combined']:
            pa = os.path.join(ref, 'data', f'conf{cid}', f'{f}_{cid}.npy')
            pb = os.path.join(out, 'data', f'conf{cid}', f'{f}_{cid}.npy')
            a = np.load(pa) if os.path.exists(pa) else None
            b = np.load(pb) if os.path.exists(pb) else None
            kind = 1
            if f.endswith('3pt'): kind = 3
            elif f == 'pjnnjnp_4pt': kind = 4
            elif f == 'ope_combined': kind = 2
            ok, rel, _ = _cmp_array(f'{f}_conf{cid}', a, b, rtol, atol)
            items[kind][1].append((f'{f}_conf{cid}', ok, rel))
            print(f"  [{'PASS' if ok else 'FAIL'}] conf{cid}/{f}  "
                  f"max_rel={rel:.2e}" if rel is not None
                  else f"  [MISS] conf{cid}/{f}")

    # 3) analysis_summary.json 标量（E0/E0_err 等）
    sa = json.load(open(os.path.join(ref, 'analysis_summary.json')))
    sb_path = os.path.join(out, 'analysis_summary.json')
    sb = json.load(open(sb_path)) if os.path.exists(sb_path) else {}
    scalar_ok = True
    for ch, va in sa.get('meff', {}).items():
        vb = sb.get('meff', {}).get(ch, {})
        for k in ('E0', 'E0_err', 'E_exp'):
            if k in va and k in vb:
                ok = abs(va[k] - vb[k]) <= rtol * abs(va[k]) + atol
                scalar_ok &= ok
                print(f"  [{'PASS' if ok else 'FAIL'}] summary meff[{ch}].{k}: "
                      f"基线={va[k]:.6g} 本次={vb[k]:.6g}")
    items.append(('summary', [(k, scalar_ok, None)]))

    # 4) 汇总
    n_pass = sum(1 for _, lst in items for _, ok, _ in lst if ok and ok is not None)
    n_fail = sum(1 for _, lst in items for _, ok, _ in lst
                 if ok is False and ok is not None)
    n_miss = sum(1 for _, lst in items for _, ok, _ in lst if ok is None)
    total = len([x for _, lst in items for x in lst])
    print(f"\nverify: 共 {total} 项，PASS {n_pass}，FAIL {n_fail}，MISS {n_miss}")

    result = {
        'ref_dir': ref,
        'out_dir': out,
        'rtol': rtol, 'atol': atol,
        'n_total': total, 'n_pass': n_pass, 'n_fail': n_fail, 'n_miss': n_miss,
        'details': {k: [{'name': n, 'ok': ok, 'max_rel': r}
                        for n, ok, r in lst] for k, lst in items},
    }
    vpath = os.path.join(out, 'test0_verify.json')
    with open(vpath, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"verify 结果 -> {vpath}")
    return 0 if n_fail == 0 else 1


# ═══════════════════════════════════════════════════════════════════
# 子命令 4: collect — 汇总 test0_results.json（表/图输入）
# ═══════════════════════════════════════════════════════════════════

def cmd_collect(args):
    out = resolve_outdir(args)
    run_dir = args.run_dir
    result = {
        'version': VERSION,
        'run_dir': run_dir,
        'env': json.load(open(os.path.join(out, 'env.json'))) if os.path.exists(
            os.path.join(out, 'env.json')) else {},
        'summary': json.load(open(os.path.join(run_dir, 'analysis_summary.json')))
        if os.path.exists(os.path.join(run_dir, 'analysis_summary.json')) else {},
        'files': {},
    }
    # 产物清单
    for root, _dirs, files in os.walk(run_dir):
        for f in sorted(files):
            full = os.path.join(root, f)
            key = os.path.relpath(full, run_dir)
            result['files'][key] = os.path.getsize(full)
    result['verify'] = json.load(open(os.path.join(out, 'test0_verify.json'))) \
        if os.path.exists(os.path.join(out, 'test0_verify.json')) else {}
    rpath = os.path.join(out, 'test0_results.json')
    with open(rpath, 'w') as f:
        json.dump(result, f, indent=2)
    nfiles = len(result['files'])
    print(f"collect: {nfiles} 个产物 -> {rpath}")
    return 0


# ═══════════════════════════════════════════════════════════════════
# 子命令 5: report — LaTeX 物理报告
# ═══════════════════════════════════════════════════════════════════

def cmd_report(args):
    out = resolve_outdir(args)
    run_dir = args.run_dir
    print(f"生成报告: {run_dir}")
    r = subprocess.run([sys.executable, str(_SCRIPT_DIR / 'report.py'),
                        '--run-dir', run_dir, '--out', out],
                       capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
    return r.returncode


# ═══════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='test0 — docker-v20260805 管线一致性测试')
    p.add_argument('--outdir', default=None, help='产物目录（默认 TEST0_OUTDIR / logs/test0/）')
    sub = p.add_subparsers(dest='cmd', required=True)

    pe = sub.add_parser('env', help='环境自检 + env.json')
    pe.add_argument('--outdir', default=None)

    pp = sub.add_parser('pipeline', help='完整管线（vertex→2pt/3pt/4pt/OPE→analysis→plots→report）')
    pp.add_argument('--outdir', default=None)
    pp.add_argument('--conf-id', type=int, default=None)
    pp.add_argument('--conf-ids', type=str, default=None)
    pp.add_argument('--precision', choices=['complex64', 'complex128'], default=C.PRECISION)
    pp.add_argument('--Nev1', type=int, default=C.NEV1)
    pp.add_argument('--steps', type=str, default='all')
    pp.add_argument('--skip-2pt', action='store_true')
    pp.add_argument('--skip-ope', action='store_true')
    pp.add_argument('--skip-3pt', action='store_true')
    pp.add_argument('--skip-4pt', action='store_true')
    pp.add_argument('--skip-analysis', action='store_true')
    pp.add_argument('--skip-plots', action='store_true')
    pp.add_argument('--skip-report', action='store_true')
    pp.add_argument('--channels', type=str, default='pp,pn,pion')
    pp.add_argument('--fourpt-nev1', type=int, default=None)
    pp.add_argument('--fourpt-tsep', type=int, default=None)
    pp.add_argument('--run-dir', type=str, default=None)
    pp.add_argument('--verbose', '-v', action='store_true')

    pv = sub.add_parser('verify', help='数值一致性验证 vs 基线 output_20260802_120104')
    pv.add_argument('--outdir', default=None)
    pv.add_argument('--ref', default=None, help='基线目录（默认 examples/docker-v20260805/...）')
    pv.add_argument('--rtol', type=float, default=1e-3)
    pv.add_argument('--atol', type=float, default=1e-8)

    pc = sub.add_parser('collect', help='汇总 test0_results.json')
    pc.add_argument('--outdir', default=None)
    pc.add_argument('--run-dir', required=True, help='管线输出子目录名')

    pr = sub.add_parser('report', help='生成并编译 LaTeX 物理报告')
    pr.add_argument('--outdir', default=None)
    pr.add_argument('--run-dir', required=True, help='管线输出子目录名')

    return p.parse_args()


def main():
    args = parse_args()
    write_env_json(resolve_outdir(args), sys.argv)
    fn = {'env': cmd_env, 'pipeline': cmd_pipeline, 'verify': cmd_verify,
          'collect': cmd_collect, 'report': cmd_report}[args.cmd]
    try:
        return fn(args)
    except Exception as e:
        print(f"[test0] {args.cmd} FAILED: {e}")
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
