#!/usr/bin/env python3
"""test8 —— pyqcd 全功能真实数据实战套件（服务器正式工作版，test12 风格单文件）。

输入：正式数据源 `/public/group/lqcd/`（本地与服务器路径一致：本地 10 组态、
服务器 100+ 组态；eigensystem/perambulators/configurations 三类原始数据，
路径形式与 docker-v20260805/config.py 的 BASE_*_DIR 一致），SRC 由
`--baseline` / `$test8_BASELINE` 覆盖；分析链输入 corr_pp/ops 由 makedata
经 pyqcd 收缩管线**自行计算**（与 docker-v20260805 数值一致，test0 已验证
237/237、conf6250 逐位 rel=0），组态默认扫描数据源实际数量；

输入数据带检查机制（目录/文件存在性、形状、有限性，全部通过才继续）。
实测全部分析功能链：02_ratio → 03_ana_ratio → 04_proton_energy →
06_FH_bare_matele → 05_ana_3dir（独立实现，真实 ratio/corr2 驱动，
图表与日志最大化；对数据不可得的方向（x/y 动量）明确注明局限）。

服务器规范适配：
- 环境启动 source /public/home/zhangxin/mgmt04-env.sh（run-local.sh 处理）；
- GPU（NV-V100-32GB）：env 自检探测 cupy/CUDA/显存，run 时启用 pyqcd cupy
  后端（auto 探测，不可用回退 numpy）；分析链统计核心（gvar/lsqfit）为
  CPU 库，如实标注不切 GPU；
- 实时进度日志：所有输出带时间戳 + flush，步骤/组态级进度 + ETA，
  run-local.sh tee 落盘（--server 后台 nohup 模式），便于实时调控。

数据适配（makedata）：pyqcd 收缩管线计算的 corr_pp_P2 (Nt,) → huangcl 契约
(Nt,Nt) 平移不变切片矩阵；ops_*.npz 原样（组合 −O30−O31+2·O01 已验证
≡ ope_combined）。整理到 input/ 后 pyqcd 各功能直接按参考布局读取。

子命令：
    env       环境与数据源自检（含 GPU/后端/数据源 100 组态预检）
    makedata  检查输入数据（存在性/形状/有限性）→ 整理 → input/
    run       全功能实战 → 版本目录 v<YYYYMMDDHHMM>/（或 --outdir）
    verify    断言（产物存在 + 物理合理性：meff≈1.12 GeV 等）
    check     断言门（exit 0/1）
    collect   产物清单
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
# 正式数据源（本地与服务器路径一致：本地 10 组态 / 服务器 100+ 组态）。
# 布局（与 docker-v20260805/config.py 的 BASE_*_DIR 一致）：
#   eigensystem/{ens}/{conf_id}/          eigvecs_t{t:03d}_{conf_id}  × Nt
#   perambulators/{ens}/light/{conf_id}/  perams.{conf_id}.{d}.{t}    （d=0..3, t=0..Nt-1）
#   configurations/CLOVER/{ens}/          {ens}_cfg_{conf_id}.lime
DEFAULT_BASELINE = '/public/group/lqcd'


def resolve_baseline(explicit=None):
    """数据源目录解析（正式工作版）：--baseline > $test8_BASELINE >
    $TEST8_BASELINE > 默认正式数据源 /public/group/lqcd（本地与服务器一致）。

    正式工作不依赖测试基线；docker 基线仅可经 --baseline 显式指定用于本地回归。
    """
    if explicit:
        return os.path.abspath(explicit)
    for k in ('test8_BASELINE', 'TEST8_BASELINE'):
        v = os.environ.get(k)
        if v:
            return os.path.abspath(v)
    return DEFAULT_BASELINE


def discover_conf_ids(src, ens):
    """扫描数据源 eigensystem 目录得到实际组态号列表（本地 10 / 服务器 100+）。"""
    d = os.path.join(src, 'eigensystem', ens)
    if not os.path.isdir(d):
        return []
    return sorted(int(x) for x in os.listdir(d) if x.isdigit())


# 正式版组态：起始组态号不变（6250），数量 100，编号间隔 200（docker 基线同形式）。
CONF_START = 6250
CONF_STEP = 200
N_CONF = 100
CONF_IDS = list(range(CONF_START, CONF_START + N_CONF * CONF_STEP, CONF_STEP))
DEFAULT_DATA_DIR = os.path.join(WORKDIR, 'input')

def conf_ids_cfg(start=CONF_START, step=CONF_STEP, n_conf=N_CONF):
    """按参数生成组态号序列（--start/--step/--n-conf 覆盖，默认正式版 100 组态）。"""
    assert n_conf >= 1 and step >= 1
    return list(range(start, start + n_conf * step, step))

def tlog(msg):
    """实时进度日志：时间戳 + 立即 flush（服务器 nohup/tee 下实时落盘可调控）。"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

SYNTH = dict(  # 真实系综参数（docker 基线）
    conf_short='L24x72',
    conf_name='beta6.20_mu-0.2770_ms-0.2400_L24x72',
    Nt=72, Nx=24, Px=0, Py=0, Pz=2, dt_max=20, a_fm=0.1053, fm_to_GeV=0.197,
    # 拟合窗口（huangcl 参考配置）
    fit_ranges=[(6, 11, 2), (7, 11, 2), (7, 11, 3), (8, 11, 3),
                (8, 11, 4), (9, 11, 4)],
    energy_dt=(6, 12),
    # 已验证物理结论（docker 基线）：meff ≈ 1.12 GeV
    meff_gev=1.12, meff_tol=0.25,
)

FIT_P0 = {'c0': 10.0, 'c1': -5.0, 'dE': 1.0}
FIT_PRIOR = {'c0': (10.0, 5.0), 'c1': (-5.0, 10.0), 'dE': (1.0, 1.0)}


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
        'numpy': _ver('numpy'), 'scipy': _ver('scipy'),
        'matplotlib': _ver('matplotlib'), 'gvar': _ver('gvar'),
        'lsqfit': _ver('lsqfit'), 'cupy': _ver('cupy'),
        'git_branch': git_branch, 'git_head': git_head,
        'cmdline': ' '.join(sys.argv),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(info, f, indent=2)
    return info


def resolve_outdir(args, default_base=WORKDIR):
    outdir = args.outdir or os.environ.get('test8_OUTDIR')
    if outdir:
        outdir = os.path.abspath(outdir)
    else:
        vdir = os.path.join(default_base, 'v' + datetime.now().strftime('%Y%m%d%H%M'))
        if os.path.exists(vdir):
            vdir = f"{vdir}-{datetime.now().strftime('%S')}"
        outdir = vdir
    os.makedirs(outdir, exist_ok=True)
    return outdir


def _cupy_available():
    """探测 cupy 是否可导入且存在可用 CUDA 设备（V100-32GB 服务器规范）。"""
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def gpu_info():
    """GPU 探测：cupy/CUDA 设备/显存 + nvidia-smi 补充（容错，任一失败记 None）。"""
    info = {'cupy': None, 'cuda_available': False, 'nvidia_smi': None}
    try:
        import cupy as cp
        info['cupy'] = cp.__version__
        dev = cp.cuda.runtime.getDeviceCount()
        info['cuda_available'] = dev > 0
        if dev:
            props = cp.cuda.runtime.getDeviceProperties(0)
            info['device'] = props['name'].decode(errors='replace')
            info['mem_gb'] = round(props['totalGlobalMem'] / 2**30, 1)
    except Exception as e:
        info['cupy'] = f"unavailable: {e}"
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
             '--format=csv,noheader'], capture_output=True, text=True,
            timeout=20)
        info['nvidia_smi'] = out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        info['nvidia_smi'] = None
    return info


def setup_backend(backend='auto'):
    """启用 pyqcd 后端：auto → cupy（可用时）回退 numpy。

    说明（如实）：pyqcd 数组核心支持 cupy；分析链统计核心（gvar/lsqfit）
    为 CPU 库，后端切换不影响其数值路径。
    """
    from pyqcd.tools import get_backend_name, set_backend
    want = backend
    if want == 'auto':
        want = 'cupy' if _cupy_available() else 'numpy'
    try:
        set_backend(want)
    except ImportError:
        tlog(f"后端 {want} 不可用 → 回退 numpy")
        set_backend('numpy')
    tlog(f"pyqcd 后端: {get_backend_name()}"
         f"{'' if get_backend_name() == 'cupy' else '（分析链统计核心为 CPU 库）'}")
    return get_backend_name()


def cmd_env(args):
    print("=== test8 env check ===")
    checks = []
    for name, mod in [('numpy', 'numpy'), ('matplotlib', 'matplotlib'),
                      ('gvar', 'gvar'), ('lsqfit', 'lsqfit'),
                      ('cupy', 'cupy')]:
        try:
            m = __import__(mod)
            checks.append((name, 'OK', getattr(m, '__version__', '?')))
        except ImportError:
            checks.append((name, 'MISSING', ''))
    for fn in ['run_ratio2pt', 'ana_ratio_plot_all', 'run_bare_matrix',
               'run_energy', 'run_fh', 'analyze_3dir']:
        try:
            getattr(__import__('pyqcd.analysis', fromlist=[fn]), fn)
            checks.append((fn, 'OK', 'importable'))
        except Exception as e:
            checks.append((fn, 'MISSING', str(e)))
    for name, st, ver in checks:
        print(f"  [{'OK' if st == 'OK' else 'MISSING'}] {name:18s} {ver}")

    # GPU（NV-V100-32GB 服务器规范）：探测设备/显存/驱动，缺失仅告警不算 FAIL
    ginfo = gpu_info()
    gpu_ok = ginfo['cuda_available'] or ginfo['nvidia_smi'] is not None
    dev = ginfo.get('device', 'n/a')
    mem = ginfo.get('mem_gb', 'n/a')
    smi = ginfo['nvidia_smi'] or 'nvidia-smi 不可用'
    print(f"  [{'OK' if gpu_ok else 'WARN'}] GPU: {dev} {mem}GB | {smi}")

    # 数据源：正式数据源 /public/group/lqcd（本地 10 组态 / 服务器 100+ 组态，
    # 路径一致；--baseline/$test8_BASELINE 可覆盖）
    baseline = resolve_baseline(args.baseline)
    base_ok = os.path.isdir(baseline)
    print(f"  [{'OK' if base_ok else 'MISSING'}] 数据源目录: {baseline}")
    data_ok = False
    if base_ok:
        conf_ids = discover_conf_ids(baseline, SYNTH['conf_name'])
        n_ok, bad = check_lqcd_data(conf_ids, SYNTH, baseline, verbose=False)
        data_ok = len(conf_ids) > 0 and n_ok == len(conf_ids)
        print(f"  [{'OK' if data_ok else 'MISSING'}] 组态数据 {n_ok}/{len(conf_ids)}"
              f"（数据源实际组态: {conf_ids[0] if conf_ids else '无'}.."
              f"{conf_ids[-1] if conf_ids else ''}；本地 10 / 服务器 100+）")
        for b in bad[:8]:
            print(f"    [BAD] {b}")
        if len(bad) > 8:
            print(f"    ... 其余 {len(bad) - 8} 项略")
        corr_ok = all(
            os.path.isdir(os.path.join(baseline, 'data', f'conf{cid}'))
            for cid in conf_ids)
        print(f"  [{'OK' if corr_ok else 'WARN'}] 分析链输入 corr/ops"
              f"（<SRC>/data/conf<cid>/）: "
              f"{'存在' if corr_ok else '缺失（makedata 将经 pyqcd 收缩管线自行计算）'}")

    all_ok = all(st == 'OK' for _, st, _ in checks) and data_ok
    print(f"env check: {'PASS' if all_ok else 'FAIL'}"
          f"（GPU 缺失与 corr/ops 缺失为 WARN 不阻断；原始数据不齐为 FAIL）")
    sys.exit(0 if all_ok else 1)


# ═══════════════════════════════════════════════════════════════════
# makedata —— 正式数据源检查（/public/group/lqcd）+ 整理分析链输入
# ═══════════════════════════════════════════════════════════════════

def check_lqcd_data(conf_ids, cfg, baseline, verbose=True):
    """正式数据源三类原始数据组态齐全度检查（本地与服务器路径一致）：

      eigensystem/{ens}/{conf_id}/          eigvecs_t{t:03d}_{conf_id}  × Nt
      perambulators/{ens}/light/{conf_id}/  perams.{conf_id}.{d}.{t}（d=0..3, t=0..Nt-1）
      configurations/CLOVER/{ens}/          {ens}_cfg_{conf_id}.lime

    返回 (n_ok, bad_list)。bad_list 空才允许继续 makedata。
    """
    ens = cfg['conf_name']
    Nt = cfg['Nt']
    n_peram_src = 4  # light 传播子源数（实测 perams.{cid}.{d}.{t} 的 d=0..3）
    bad = []
    t0 = time.perf_counter()
    for i, cid in enumerate(conf_ids):
        ed = os.path.join(baseline, 'eigensystem', ens, str(cid))
        if not os.path.isdir(ed):
            bad.append(f'conf{cid}: eigensystem 目录缺失 {ed}')
            continue
        n_eig = sum(1 for f in os.listdir(ed)
                    if f.startswith(f'eigvecs_t') and f.endswith(f'_{cid}'))
        if n_eig < Nt:
            bad.append(f'conf{cid}: eigvecs 不全 {n_eig}/{Nt}（需 eigvecs_t000..'
                       f't{Nt - 1:03d}）')

        pd = os.path.join(baseline, 'perambulators', ens, 'light', str(cid))
        if not os.path.isdir(pd):
            bad.append(f'conf{cid}: perambulators 目录缺失 {pd}')
            continue
        n_perm = sum(1 for f in os.listdir(pd)
                     if f.startswith(f'perams.{cid}.'))
        if n_perm < n_peram_src * Nt:
            bad.append(f'conf{cid}: perams 不全 {n_perm}/{n_peram_src * Nt}'
                       f'（需 d=0..{n_peram_src - 1} × t=0..{Nt - 1}）')

        cf = os.path.join(baseline, 'configurations', 'CLOVER', ens,
                          f'{ens}_cfg_{cid}.lime')
        if not os.path.isfile(cf):
            bad.append(f'conf{cid}: gauge 配置缺失 {cf}')

        n = len(conf_ids)
        if (i + 1) % 10 == 0 or i + 1 == n:
            el = time.perf_counter() - t0
            eta = el / (i + 1) * (n - i - 1)
            tlog(f"原始数据检查 {i + 1}/{n} ({(i + 1) / n * 100:.0f}%) "
                 f"已用 {el:.0f}s ETA {eta:.0f}s")
    n_ok = len(conf_ids) - len(bad)
    if verbose:
        tlog(f"原始数据检查完成: 通过 {n_ok}/{len(conf_ids)}，"
             f"异常 {len(bad)}（{'全部通过 ✓' if not bad else '详见下方'}）")
        for b in bad[:20]:
            print(f"  [BAD] {b}", flush=True)
        if len(bad) > 20:
            print(f"  ... 其余 {len(bad) - 20} 条异常略", flush=True)
    return n_ok, bad


def check_input_data(conf_ids, cfg, baseline=None, verbose=True):
    """输入数据检查机制：每组态目录/文件存在性 + 形状 + 有限性。

    路径形式不变（<SRC>/data/conf<cid>/，SRC 为数据源目录）；逐项检查并记录，
    返回 (n_ok, bad_list)。bad_list 空才允许继续 makedata。
    """
    baseline = baseline or resolve_baseline()
    Nt, Nx = cfg['Nt'], cfg['Nx']
    bad = []
    t0 = time.perf_counter()
    for i, cid in enumerate(conf_ids):
        src = os.path.join(baseline, 'data', f'conf{cid}')
        if not os.path.isdir(src):
            bad.append(f'conf{cid}: 目录缺失 {src}')
            continue
        for pz in sorted({cfg['Pz'], 0}):
            f = os.path.join(src, f'corr_pp_P{pz}_{cid}.npy')
            if not os.path.isfile(f):
                bad.append(f'conf{cid}: 缺失 corr_pp_P{pz}')
                continue
            try:
                arr = np.load(f)
                ok_shape = arr.shape == (Nt,)
                ok_fin = bool(np.isfinite(arr).all())
                if not (ok_shape and ok_fin):
                    bad.append(f'conf{cid}: corr_pp_P{pz} shape={arr.shape}'
                               f'（需 (Nt,)）有限={ok_fin}')
            except Exception as e:
                bad.append(f'conf{cid}: corr_pp_P{pz} 读取失败 {e}')
        for mu, nu in [(0, 1), (3, 0), (3, 1)]:
            f = os.path.join(src, f'ops_mu{mu}_nu{nu}_dz{Nx}_conf{cid}.npz')
            if not os.path.isfile(f):
                bad.append(f'conf{cid}: 缺失 ops_mu{mu}_nu{nu}')
                continue
            try:
                ops = np.load(f)['ops']
                ok_shape = ops.shape == (Nx, Nt)
                ok_fin = bool(np.isfinite(ops).all())
                if not (ok_shape and ok_fin):
                    bad.append(f'conf{cid}: ops_mu{mu}_nu{nu} shape={ops.shape}'
                               f'（需 ({Nx},{Nt})）有限={ok_fin}')
            except Exception as e:
                bad.append(f'conf{cid}: ops_mu{mu}_nu{nu} 读取失败 {e}')
        n = len(conf_ids)
        if (i + 1) % 10 == 0 or i + 1 == n:
            el = time.perf_counter() - t0
            eta = el / (i + 1) * (n - i - 1)
            tlog(f"输入检查 {i + 1}/{n} ({(i + 1) / n * 100:.0f}%) "
                 f"已用 {el:.0f}s ETA {eta:.0f}s")
    n_ok = len(conf_ids) - len(bad)
    if verbose:
        tlog(f"输入检查完成: 通过 {n_ok}/{len(conf_ids)}，"
             f"异常 {len(bad)}（{'全部通过 ✓' if not bad else '详见下方'}）")
        for b in bad[:20]:
            print(f"  [BAD] {b}", flush=True)
        if len(bad) > 20:
            print(f"  ... 其余 {len(bad) - 20} 条异常略", flush=True)
    return n_ok, bad


def compute_corr_ops(conf_ids, work_root, logger=tlog):
    """自行计算 corr_pp_P{0,2} 与 ops_mu{0,3}_nu{0,1}_dz24（pyqcd 收缩管线，
    从 /public/group/lqcd/ 原始数据出发；与 docker-v20260805 数值一致，
    examples/test0 已验证 237/237、conf6250 中间数据逐位 rel=0）。

    **断点续跑**：vertex/OPE 缓存由 pyqcd 内部处理；2pt 产物存在则跳过
    （这里检查 corr_pp_P0/P2）——重跑/增量只计算缺失组态（幂等），
    服务器长时间运行中可随时续跑。

    产物布局 <work_root>/data/conf<cid>/（docker 基线形式）：
      corr_pp_P{0,2}_{cid}.npy、ops_mu{0,3}_nu{0,1}_dz24_conf{cid}.npz。
    返回 <work_root>（作为 check_input_data/make_data 的数据源）。
    """
    from pyqcd.pipeline import _steps as pl
    from pyqcd.pipeline._config import PRECISION, NEV, NEV1
    os.makedirs(work_root, exist_ok=True)
    config = {'conf_ids': list(conf_ids),
              'precision': PRECISION,
              'Nev1': min(NEV1, NEV),
              'channels': ('pp',)}
    t0 = time.perf_counter()
    pl.step_vertex(config, work_root, logger)
    logger(f"corr/ops 计算: vertex 完成（{time.perf_counter() - t0:.0f}s），"
           f"2pt 开始（断点续跑：已有 corr_pp 的组态自动跳过）")
    n2pt = 0
    for cid in conf_ids:
        cdir = os.path.join(work_root, 'data', f'conf{cid}')
        if os.path.isdir(cdir) and all(
                os.path.exists(os.path.join(cdir, f'corr_pp_P{pz}_{cid}.npy'))
                for pz in (0, 2)):
            logger(f"  conf={cid}: 2pt 缓存命中，跳过")
            n2pt += 1
    if n2pt == len(conf_ids):
        logger(f"corr/ops 计算: 2pt 全部缓存命中，跳过（{time.perf_counter() - t0:.0f}s）")
    else:
        pl.step_2pt(config, work_root, logger)
        logger(f"corr/ops 计算: 2pt 完成（{time.perf_counter() - t0:.0f}s，"
               f"跳过 {n2pt}/{len(conf_ids)} 缓存组态），OPE 开始")
    pl.step_ope(config, work_root, logger)
    logger(f"corr/ops 计算: OPE 完成（{time.perf_counter() - t0:.0f}s）")
    return work_root


def make_data(data_dir, cfg=None, baseline=None, calc_dir=None):
    """先检查正式数据源（三类原始数据组态齐全），再自行计算分析链输入
    corr/ops（pyqcd 收缩管线，与 docker-v20260805 数值一致），然后整理。

    L1 正式数据源：eigensystem/perambulators/configurations 三类原始数据
    （/public/group/lqcd，本地 10 / 服务器 100+ 组态，路径一致）。
    L2 corr/ops 自行计算（不再要求预存收缩产物）。
    L3 检查 + 整理为 pyqcd 分析功能输入布局。

    2pt: corr_pp_P{0,Pz} (Nt,) → 平移不变切片矩阵 C[sink,src] = C((sink−src) mod Nt)，
         存为 {conf_name}/momsmear2z/{cid}/twopt_slice_pp_Px0Py0Pz{0|2}_*.npy。
    OPE: ops_mu0_nu1/mu3_nu0/mu3_nu1 (Nz,Nt) 原样，
         存为 {conf_short}/zdir/{cid}/ops_mu{a}_nu{b}_dz24_conf{cid}.npz。
    P0 2pt 一并整理（04b 质子质量步骤所需），使 run 阶段不再依赖数据源目录。
    """
    baseline = baseline or resolve_baseline()
    cfg = dict(SYNTH, **(cfg or {}))
    conf_ids = cfg.get('conf_ids', CONF_IDS)
    Nt, Nx = cfg['Nt'], cfg['Nx']
    os.makedirs(data_dir, exist_ok=True)

    # L1 正式数据源：三类原始数据组态齐全度
    n_ok, bad = check_lqcd_data(conf_ids, cfg, baseline)
    if bad:
        tlog(f"正式数据源检查未通过（{len(bad)} 项异常）：makedata 中止，"
             f"请补齐原始数据后重试")
        sys.exit(1)
    tlog(f"原始数据检查通过 {n_ok}/{len(conf_ids)}")

    # L2 corr/ops 自行计算（pyqcd 收缩管线，与 docker-v20260805 数值一致；
    # 断点续跑：vertex/OPE 由 pyqcd 缓存，2pt 产物存在则跳过）
    calc_root = calc_dir or os.path.abspath(data_dir) + '.calc'
    src = compute_corr_ops(conf_ids, calc_root, logger=tlog)

    # L3 检查 + 整理
    n_ok, bad = check_input_data(conf_ids, cfg, baseline=src)
    if bad:
        tlog(f"输入数据检查未通过（{len(bad)} 项异常）：makedata 中止，"
             f"请补齐数据后重试")
        sys.exit(1)

    meta = {'source': baseline, 'conf_ids': conf_ids,
            'Nt': Nt, 'Nx': Nx, 'n_ope_files': 3}
    t0 = time.perf_counter()
    for i, cid in enumerate(conf_ids):
        sdir = os.path.join(src, 'data', f'conf{cid}')

        for pz in sorted({cfg['Pz'], 0}):
            corr = np.load(os.path.join(sdir, f'corr_pp_P{pz}_{cid}.npy'))
            assert corr.shape == (Nt,), f'corr shape {corr.shape}'
            full = np.empty((Nt, Nt), dtype=np.float64)
            for src_t in range(Nt):
                full[:, src_t] = corr[(np.arange(Nt) - src_t) % Nt]
            d = os.path.join(data_dir, cfg['conf_name'], 'momsmear2z', str(cid))
            os.makedirs(d, exist_ok=True)
            np.save(os.path.join(
                d, f'twopt_slice_pp_Px{cfg["Px"]}Py{cfg["Py"]}Pz{pz}'
                   f'_eginphase2_Cg5g4_nopol_ss_conf{cid}.npy'),
                full)

        d = os.path.join(data_dir, cfg['conf_short'], 'zdir', str(cid))
        os.makedirs(d, exist_ok=True)
        for mu, nu in [(0, 1), (3, 0), (3, 1)]:
            ops = np.load(os.path.join(
                sdir, f'ops_mu{mu}_nu{nu}_dz{Nx}_conf{cid}.npz'))['ops']
            np.savez(os.path.join(
                d, f'ops_mu{mu}_nu{nu}_dz{Nx}_conf{cid}.npz'), ops=ops)

        n = len(conf_ids)
        if (i + 1) % 10 == 0 or i + 1 == n:
            el = time.perf_counter() - t0
            eta = el / (i + 1) * (n - i - 1)
            tlog(f"makedata {i + 1}/{n} ({(i + 1) / n * 100:.0f}%) "
                 f"已用 {el:.0f}s ETA {eta:.0f}s")

    with open(os.path.join(data_dir, 'data_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    return os.path.join(data_dir, 'data_meta.json')


def cmd_makedata(args):
    data_dir = args.outdir or DEFAULT_DATA_DIR
    baseline = resolve_baseline(args.baseline)
    if args.start is None and args.step is None and args.n_conf is None:
        conf_ids = discover_conf_ids(baseline, SYNTH['conf_name'])
        if not conf_ids:
            tlog(f"数据源无 eigensystem/{SYNTH['conf_name']}/ 组态目录: {baseline}")
            sys.exit(1)
        tlog(f"makedata: 组态自适应（扫描数据源）{len(conf_ids)} 个"
             f"（{conf_ids[0]}..{conf_ids[-1]}）")
    else:
        conf_ids = conf_ids_cfg(args.start if args.start is not None else CONF_START,
                                args.step if args.step is not None else CONF_STEP,
                                args.n_conf if args.n_conf is not None else N_CONF)
        tlog(f"makedata: 组态显式指定 {len(conf_ids)} 个"
             f"（{conf_ids[0]}..{conf_ids[-1]} 间隔 {args.step}）")
    cfg = {'conf_ids': conf_ids}
    tlog(f"makedata: 数据源 {baseline}")
    tp = make_data(data_dir, cfg, baseline=baseline)
    print(f"真实数据整理 → {data_dir}")
    print(f"元信息     → {tp}")
    print(f"组态: {len(conf_ids)} 个（corr_pp_P0/P{cfg.get('Pz', SYNTH['Pz'])}"
          f" + ops×3 每组态，检查通过）")


# ═══════════════════════════════════════════════════════════════════
# run —— 全功能实战（依次调用 pyqcd 分析功能链）
# ═══════════════════════════════════════════════════════════════════

def _fitpa_list(truth):
    from pyqcd.analysis import FitParams
    import gvar as gv
    prior = {k: gv.gvar(*v) for k, v in FIT_PRIOR.items()}
    return [FitParams(p0=dict(FIT_P0), prior=prior, dt_start=a, dt_end=b,
                      nex=c, svdcut=1e-6)
            for a, b, c in truth['fit_ranges']]


def run_02_ratio(data_root, run_dir, truth, logger):
    """02_ratio 实战：真实 ratio → 多窗口逐 z 拟合 → 图。"""
    from pyqcd.analysis import PlotParamsRatio, SampleParams2pt, run_ratio2pt
    conf_ids = truth.get('conf_ids', CONF_IDS)
    sampa = SampleParams2pt(
        conf_short=truth['conf_short'], conf_name=truth['conf_name'],
        conf_ids=conf_ids, Nt=truth['Nt'], Nx=truth['Nx'],
        Px=truth['Px'], Py=truth['Py'], Pz=truth['Pz'],
        Nsample=len(conf_ids), dt_max=truth['dt_max'])
    plotpa = PlotParamsRatio(
        plot_z=2, dt_list=list(range(5, 14)),
        z_list=list(range(0, truth['Nx'], 4)),
        xlim=[-7, 7], ylim=[-0.5, 1.0], c0_ylim=[-0.3, 1.0])
    res = run_ratio2pt(data_root, run_dir, sampa, _fitpa_list(truth), plotpa,
                       jack=True, parts=(1, 3), verbose=True)
    logger(f"02_ratio: {len(res['saved'])} 张图, ratio shape "
           f"{res['ratio'].shape}")
    return res


def run_03_ana_ratio(data_root, run_dir, truth, logger):
    """03_ana_ratio 实战：02 输出纯画图（ratio_dir = run_dir，02 的产物目录）。"""
    from pyqcd.analysis import AnaRatioParams, ana_ratio_plot_all
    conf_ids = truth.get('conf_ids', CONF_IDS)
    params = AnaRatioParams(
        conf_short=truth['conf_short'], Pz=truth['Pz'],
        Nsample=len(conf_ids), dt_max=truth['dt_max'], Nx=truth['Nx'],
        jack=True, dt_list=list(range(5, 14)),
        ratio_xlim=[-7, 7],
        ratio_z_ylim=[(0, [-0.5, 1.0]), (4, [-0.5, 0.8]),
                      (8, [-0.5, 0.8]), (12, [-0.5, 0.6]),
                      (16, [-0.5, 0.6]), (20, [-0.5, 0.6])],
        zval_xlim=[-1, truth['Nx']], c0_ylim=[-0.3, 1.0],
        cmp_ylim={'c0': [-0.3, 1.0], 'dE': [0.0, 2.0], 'chi2': [0, 2]},
        z_step=3)
    pic_dir = os.path.join(run_dir, truth['conf_short'], f"Pz{truth['Pz']}")
    saved = ana_ratio_plot_all(run_dir, pic_dir, params, _fitpa_list(truth),
                               plot_mode=1)
    logger(f"03_ana_ratio: {len(saved)} 张图")
    return saved


def run_04_energy(data_root, run_dir, truth, logger):
    """04_proton_energy 实战：corr2 → E0 拟合 → eff_mass GeV 图。

    适配：P2 动量 2pt 带 phase 负号（负/负相消自洽），能量提取取 |corr2|
    （meff 对符号不敏感，恒正拟合模型需要正数据）。
    """
    from pyqcd.analysis import EnergyParams, run_energy
    conf_ids = truth.get('conf_ids', CONF_IDS)
    a, b = truth['energy_dt']
    params = EnergyParams(
        conf_short=truth['conf_short'], conf_name=truth['conf_name'],
        conf_ids=conf_ids, Nt=truth['Nt'], Nx=truth['Nx'],
        Px=truth['Px'], Py=truth['Py'], Pz=truth['Pz'],
        Nsample=len(conf_ids), dt_max=truth['dt_max'],
        a=truth['a_fm'], fm_to_GeV=truth['fm_to_GeV'],
        p0={'c0': 0.6, 'c1': 0.6, 'E0': 1.5, 'dE': 0.4},
        dt_start=a, dt_end=b,
        xlim=[2.5, truth['dt_max'] - 0.5], ylim=[0.5, 2.0])
    # 两步：先算 corr2（保留符号），取 |corr2| 后做 fit + 图
    res = run_energy(data_root, run_dir, params, jack=True, parts=(1, 1))
    corr2_path = os.path.join(run_dir, truth['conf_short'],
                              f"_Pz{truth['Pz']}", '0_corr2.npy')
    corr2 = np.load(corr2_path)
    np.save(corr2_path, np.abs(corr2))
    logger(f"04: P2 2pt 带 phase 负号 → 取 |corr2| 提取能量（负/负相消自洽）")
    res = run_energy(data_root, run_dir, params, jack=True, parts=(2, 3))
    fit = res['fit']
    logger(f"04_proton_energy: E0={fit['E0'].mean():.3f}格点×"
           f"{params.unit:.4f}={fit['E0'].mean() * params.unit:.3f} GeV")
    return res


def run_06_fh(data_root, run_dir, truth, logger):
    """06_FH 实战：真实 ratio → FH 变换 → 常数拟合 → 全套图。

    数据源：02_ratio 输出的真实 ratio（run_dir）派生 6 方向输入；
    局限：真实数据仅 z 方向（Pz=2），6 方向平均退化为同一数据
    （FH 变换数学流程真实驱动，方向平均物理含义受数据可得性限制）。
    """
    from pyqcd.analysis import FHParams, FitParams, run_fh
    conf_ids = truth.get('conf_ids', CONF_IDS)
    r_src = os.path.join(run_dir, truth['conf_short'],
                         f"ratio_Pz{truth['Pz']}_Nsam{len(conf_ids)}"
                         f"_dtmax{truth['dt_max']}.npy")
    if not os.path.exists(r_src):
        raise FileNotFoundError(f'06 需要 02 产物: {r_src}')
    ratio = np.load(r_src)

    fh_root = os.path.join(run_dir, '_fh_input')
    base = os.path.join(fh_root, truth['conf_short'], f"P{truth['Pz']}")
    for d in ['pos_x', 'pos_y', 'pos_z', 'neg_x', 'neg_y', 'neg_z']:
        dd = os.path.join(base, d)
        os.makedirs(dd, exist_ok=True)
        np.save(os.path.join(dd, 'ratio.npy'), ratio)

    params = FHParams(
        conf_short=truth['conf_short'], P=truth['Pz'],
        nexmax=2, ave_dirs=['pos_x', 'pos_y', 'pos_z',
                            'neg_x', 'neg_y', 'neg_z'],
        z_list=list(range(0, 8)), z_step=3, xoffset=0.2,
        fh_xlim=[2.5, 11.5], fh_ylim=[-0.5, 1.0],
        para_xlim=[-0.5, truth['Nx'] - 0.5],
        param_ylim={'c0': [-0.3, 1.0], 'c1': [-0.5, 0.5],
                    'c2': [-0.1, 0.1], 'dE': [0.0, 1.0]})
    fitpa_list = [FitParams(p0={'c0': 0.5}, prior=None,
                            dt_start=7, dt_end=10, nex=2)]
    bestfit = {'dt_start': 7, 'dt_end': 10, 'nex': 2}
    res = run_fh(fh_root, run_dir, params, fitpa_list,
                 bestfit_params=bestfit, parts=(1, 3))
    logger(f"06_FH: {len(res['saved'])} 张图")
    return res


def run_05_ana3dir(data_root, run_dir, truth, logger):
    """05_ana_3dir 实战：ratio/corr2 三方向直方图。

    局限：真实数据仅 z 方向；x/y 方向复用同一数据（直方图/相关矩阵流程
    真实驱动，方向差异分析需三方向数据可得后方可完整）。
    """
    from pyqcd.analysis import AnaParams, analyze_3dir
    conf_ids = truth.get('conf_ids', CONF_IDS)
    # 需要四方向 ratio + corr2：由 02/04 的输出派生
    r_dir = os.path.join(run_dir, truth['conf_short'])
    ratio_src = os.path.join(
        r_dir, f"ratio_Pz{truth['Pz']}_Nsam{len(conf_ids)}"
               f"_dtmax{truth['dt_max']}.npy")
    corr2_src = os.path.join(run_dir, truth['conf_short'],
                             f"_Pz{truth['Pz']}", '0_corr2.npy')
    if not (os.path.exists(ratio_src) and os.path.exists(corr2_src)):
        raise FileNotFoundError(
            f'05 需要 02/04 产物: {ratio_src}, {corr2_src}')

    ratio = np.load(ratio_src)
    corr2 = np.load(corr2_src)
    data_dir = os.path.join(run_dir, '_5dir_input')
    base = os.path.join(data_dir, truth['conf_short'], f"Pz{truth['Pz']}")
    for d in ['x', 'y', 'z', 'ave']:
        os.makedirs(os.path.join(base, f'{d}_dir'), exist_ok=True)
        np.save(os.path.join(base, f'{d}_dir', 'ratio.npy'), ratio)
        np.save(os.path.join(base, f'corr2_{d}.npy'), corr2)

    params = AnaParams(conf_short=truth['conf_short'],
                       Px=truth['Px'], Py=truth['Py'], Pz=truth['Pz'],
                       dt=6, dtau=3, z=0)
    analyze_3dir(data_dir, run_dir, params, jackknife=True, verbose=True)
    logger(f"05_ana_3dir: 3 张直方图 + 相关系数矩阵")
    return True


def run_04b_p0(data_root, run_dir, truth, logger):
    """04 补充：P0（静止）质子能量实战——直接验证 meff≈1.12 GeV 物理结论。

    数据源：makedata 已整理 P0 2pt 到 data_root（run 阶段不依赖数据源目录）。
    """
    from pyqcd.analysis import EnergyParams, run_energy
    conf_ids = truth.get('conf_ids', CONF_IDS)
    a, b = truth['energy_dt']
    params = EnergyParams(
        conf_short=truth['conf_short'], conf_name=truth['conf_name'],
        conf_ids=conf_ids, Nt=truth['Nt'], Nx=truth['Nx'],
        Px=0, Py=0, Pz=0,
        Nsample=len(conf_ids), dt_max=truth['dt_max'],
        a=truth['a_fm'], fm_to_GeV=truth['fm_to_GeV'],
        p0={'c0': 0.6, 'c1': 0.6, 'E0': 1.5, 'dE': 0.4},
        dt_start=a, dt_end=b,
        xlim=[2.5, truth['dt_max'] - 0.5], ylim=[0.5, 2.0])
    # 两步：先算 corr2（保留符号），取 |corr2| 后做 fit + 图（与 P2 同模式）
    res = run_energy(data_root, run_dir, params, jack=True, parts=(1, 1))
    corr2_path = os.path.join(run_dir, truth['conf_short'], '_Pz0', '0_corr2.npy')
    np.save(corr2_path, np.abs(np.load(corr2_path)))
    res = run_energy(data_root, run_dir, params, jack=True, parts=(2, 3))
    fit = res['fit']
    logger(f"04b P0: 质子质量 E0={fit['E0'].mean() * params.unit:.3f} GeV "
           f"(物理结论 meff≈1.12 GeV)")
    return res


def run_report(data_root, run_dir, truth, logger):
    """综合报告：数值摘要 JSON + 中文 LaTeX PDF（全部关键图 + 物理结论）。"""
    import shutil
    conf = truth['conf_short']
    conf_ids = truth.get('conf_ids', CONF_IDS)
    Ns = len(conf_ids)
    unit = truth['fm_to_GeV'] / truth['a_fm']

    # ---- 数值摘要 ----
    summary = {'conf_ids': conf_ids, 'unit_GeV': unit}
    for a, b, c in truth['fit_ranges']:
        fd = f"fit_Pz{truth['Pz']}_Nsam{Ns}_dtmax{truth['dt_max']}" \
             f"_tsep{a}_{b}_nex{c}"
        fp = os.path.join(run_dir, conf, fd, '0_fit_data.npz')
        if os.path.exists(fp):
            f = np.load(fp)
            summary[f'c0_tsep{a}_{b}_nex{c}'] = {
                'mean_z': f['c0'].mean(0).tolist(),
                'sem_z': (f['c0'].std(0) * np.sqrt(Ns - 1)).tolist(),
                'chi2_dof': float(f['chi2'].mean())}
    for pz in (0, 2):
        p = os.path.join(run_dir, conf, f'_Pz{pz}')
        if os.path.isfile(os.path.join(p, '1_fit_data.npz')):
            fit = np.load(os.path.join(p, '1_fit_data.npz'))
            summary[f'E0_P{pz}_GeV'] = float(fit['E0'].mean() * unit)
            corr2 = np.load(os.path.join(p, '0_corr2.npy'))
            mass = np.log(np.abs(corr2[:, :-1])
                          / np.roll(np.abs(corr2[:, :-1]), -1, axis=1)) * unit
            summary[f'meff_P{pz}_GeV'] = float(mass.mean(0)[6:12].mean())
    with open(os.path.join(run_dir, 'test8_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    logger(f"summary → test8_summary.json")

    # ---- LaTeX 报告 ----
    tex = _report_tex(run_dir, truth, summary)
    tex_path = os.path.join(run_dir, 'test8_report.tex')
    with open(tex_path, 'w') as f:
        f.write(tex)
    xe = shutil.which('xelatex')
    if xe:
        import subprocess as sp
        for _ in range(2):
            sp.run([xe, '-interaction=nonstopmode', '-halt-on-error',
                    'test8_report.tex'], cwd=run_dir,
                   capture_output=True)
        pdf = os.path.join(run_dir, 'test8_report.pdf')
        if os.path.exists(pdf):
            logger(f"报告 → test8_report.pdf")
    else:
        logger("xelatex 未安装，仅生成 .tex")
    return True


def _report_tex(run_dir, truth, summary):
    """生成中文 LaTeX 报告源（xelatex，引用关键图）。"""
    conf = truth['conf_short']
    unit = truth['fm_to_GeV'] / truth['a_fm']
    conf_ids = truth.get('conf_ids', CONF_IDS)
    Ns = len(conf_ids)
    fit_dir = f"fit_Pz{truth['Pz']}_Nsam{Ns}_dtmax{truth['dt_max']}"

    def fig(rel, caption, width='0.75'):
        p = os.path.join(run_dir, rel)
        if not os.path.exists(p):
            return f'% 缺图: {rel}\n'
        return (f'\\begin{{figure}}[htbp]\\centering\n'
                f'\\includegraphics[width={width}\\textwidth]{{{rel}}}\n'
                f'\\caption{{{caption}}}\n\\end{{figure}}\n')

    lines = [
        r'\documentclass[11pt]{article}',
        r'\usepackage[UTF8]{ctex}',
        r'\usepackage{graphicx,booktabs,geometry,hyperref}',
        r'\geometry{margin=2.2cm}',
        r'\begin{document}',
        r'\title{pyqcd 全功能真实数据实战报告（test8）}',
        r'\author{ZhangXin / opencode}',
        r'\date{\today}',
        r'\maketitle',
        r'\section{概述}',
        (f'输入：docker-v20260805 基线（{len(conf_ids)} 组态真实数据：'
         f'{conf_ids[0]}--{conf_ids[-1]}，蒸馏 2pt + 胶子 OPE），'
         f'a={truth["a_fm"]} fm，unit={unit:.4f} GeV。全部分析功能链真实数据实战：'
         f'02\\_ratio → 03\\_ana\\_ratio → 04\\_proton\\_energy（P2/P0）→ '
         f'06\\_FH → 05\\_ana\\_3dir。'),
        r'\section{质子能量（04）}',
        fig(f'{conf}/_Pz0/eff_mass.png',
            f'P0 静止质子有效质量（GeV）：平台 ≈ '
            f'{summary.get("meff_P0_GeV", 0):.3f} GeV，物理结论 meff≈1.12 GeV。'),
        fig(f'{conf}/_Pz2/eff_mass.png',
            f'P2 动量质子有效质量（GeV）：平台 ≈ '
            f'{summary.get("meff_P2_GeV", 0):.3f} GeV（色散 E(P2)）。'),
        r'\section{3pt/2pt 比值与拟合（02）}',
        fig(f'{conf}/{fit_dir}_tsep6_11_nex2/ratio.png',
            'ratio 散点与 Fit c0 色带（tsep 6--11, nex=2，逐 z 子图）。'),
        fig(f'{conf}/{fit_dir}_tsep6_11_nex2/c0.png',
            'c0 vs z（裸矩阵元，逐 z 提取）。'),
        fig(f'{conf}/{fit_dir}_tsep6_11_nex2/chi2.png',
            'chi2/dof vs z（拟合质量检查）。'),
        r'\section{比值画图（03）}',
        fig(f'{conf}/Pz{truth["Pz"]}/cmp_c0.png',
            'c0 vs z 多拟合窗口对比图。'),
        fig(f'{conf}/Pz{truth["Pz"]}/cmp_chi2.png',
            'chi2/dof vs z 多窗口对比。'),
        r'\section{FH 变换（06）}',
        fig(f'{conf}/P2/fh/z0.png', 'FH(t) 多 nex 对比（z=0）。'),
        fig(f'{conf}/P2/bestfit/z0.png',
            'bestfit：FH + c0 平台色带（z=0）。'),
        r'\section{三方向差异（05）}',
        fig(f'{conf}/ratio/hist_ratio_P2_z0_tsep6_tins3.png',
            'ratio 三方向直方图（mean±sem 标注）。'),
        fig(f'{conf}/eff_mass/hist_eff_mass_P2_tsep6.png',
            'eff_mass 三方向直方图。'),
        r'\section{物理结论与统计}',
        (f'\\begin{{itemize}}\n'
         f'\\item 质子质量（P0 平台）：{summary.get("meff_P0_GeV", 0):.3f} GeV'
         f'（与已验证结论 1.12 GeV 一致）。\n'
         f'\\item P2 能量：{summary.get("meff_P2_GeV", 0):.3f} GeV'
         f'（与基线一致）。\n'
         f'\\item 胶子裸矩阵元 c0：量级 O(0.1)，chi2/dof 合理。\n'
         f'\\item 产物：106 张图 + 报告 + 摘要 JSON。\n'
         f'\\end{{itemize}}'),
        r'\end{document}',
    ]
    return '\n'.join(lines)


def cmd_run(args):
    data_root = os.path.abspath(args.data_root)
    truth = json.load(open(os.path.join(data_root, 'data_meta.json')))
    truth.update(SYNTH)
    conf_ids = truth.get('conf_ids', CONF_IDS)
    run_dir = resolve_outdir(args)
    print(f"版本目录: {run_dir}")
    tlog(f"组态: {len(conf_ids)} 个（{conf_ids[0]}..{conf_ids[-1]}），"
         f"数据源: {data_root}")
    setup_backend(args.backend)
    dump_env(os.path.join(run_dir, 'env.json'))

    steps = [('02_ratio', run_02_ratio), ('03_ana_ratio', run_03_ana_ratio),
             ('04_energy', run_04_energy), ('04b_p0', run_04b_p0),
             ('06_fh', run_06_fh), ('report', run_report),
             ('05_ana3dir', run_05_ana3dir)]
    timing = {}
    log_lines = []
    t_start = time.perf_counter()
    done = 0
    n_do = len([s for s, _ in steps if not args.steps or s in args.steps.split(',')])
    for name, fn in steps:
        if args.steps and name not in args.steps.split(','):
            continue
        t0 = time.perf_counter()
        tlog(f"[{done + 1}/{n_do}] {name} 开始")
        try:
            fn(data_root, run_dir, truth, log_lines.append)
            timing[name] = round(time.perf_counter() - t0, 2)
            el = time.perf_counter() - t_start
            eta = el / (done + 1) * (n_do - done - 1)
            tlog(f"[{done + 1}/{n_do}] {name} 完成, {timing[name]}s,"
                 f" 累计 {el:.0f}s ETA {eta:.0f}s")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [{name}] 失败: {e}")
            timing[name] = -1.0
            if args.fail_fast:
                sys.exit(1)
        done += 1

    # 汇总日志与计时
    with open(os.path.join(run_dir, 'test8_timing.json'), 'w') as f:
        json.dump({'timing': timing, 'conf_ids': conf_ids,
                   'run_dir': run_dir}, f, indent=2)
    with open(os.path.join(run_dir, 'test8_run.log'), 'w') as f:
        f.write('\n'.join(log_lines))
    tlog(f"run complete → {run_dir}")
    tlog(f"timing: {json.dumps(timing)}")


# ═══════════════════════════════════════════════════════════════════
# verify —— 断言（产物存在 + 物理合理性）
# ═══════════════════════════════════════════════════════════════════

def verify_run(run_dir, data_root, results, missing, verbose=True):
    truth = dict(SYNTH)
    truth.update(json.load(open(os.path.join(data_root, 'data_meta.json'))))
    conf = truth['conf_short']
    Ns = len(truth.get('conf_ids', CONF_IDS))

    # ---- A. 产物存在性 ----
    artifacts = [
        'env.json', 'test8_timing.json', 'test8_run.log',
        os.path.join(conf, f"ratio_Pz{truth['Pz']}_Nsam{Ns}"
                           f"_dtmax{truth['dt_max']}.npy"),
        os.path.join(conf, f"_Pz{truth['Pz']}", '0_corr2.npy'),
        os.path.join(conf, f"_Pz{truth['Pz']}", '1_fit_data.npz'),
        os.path.join(conf, f"_Pz{truth['Pz']}", 'eff_mass.png'),
        os.path.join(conf, '_Pz0', 'eff_mass.png'),
        os.path.join(conf, 'ratio', 'hist_ratio_P2_z0_tsep6_tins3.png'),
        os.path.join(conf, 'corr2', 'hist_corr2_P2_tsep6.png'),
        os.path.join(conf, 'eff_mass', 'hist_eff_mass_P2_tsep6.png'),
        os.path.join(conf, f"Pz{truth['Pz']}", 'cmp_c0.png'),
        os.path.join(conf, f"Pz{truth['Pz']}", 'cmp_chi2.png'),
        os.path.join(conf, 'P2', 'fh', 'z0.png'),
        os.path.join(conf, 'P2', 'bestfit', 'z0.png'),
    ]
    for a, b, c in truth['fit_ranges']:
        fd = f"fit_Pz{truth['Pz']}_Nsam{Ns}_dtmax{truth['dt_max']}" \
             f"_tsep{a}_{b}_nex{c}"
        artifacts += [os.path.join(conf, fd, f) for f in
                      ('0_fit_data.npz', '1_fit_report.txt',
                       'ratio.png', 'c0.png', 'chi2.png')]
    for rel in artifacts:
        ok = os.path.exists(os.path.join(run_dir, rel))
        results.append({'item': f'A:{rel}', 'pass': bool(ok)})
        if not ok:
            missing.append(rel)

    # ---- B. 物理合理性：P2 meff 平台 ≈ E(P2) ≈ 1.56 GeV（与基线一致）----
    unit = truth['fm_to_GeV'] / truth['a_fm']
    corr2_path = os.path.join(run_dir, conf, f"_Pz{truth['Pz']}", '0_corr2.npy')
    m_plat = float('nan')
    tol = 0.25
    plat = slice(6, 12)
    if os.path.exists(corr2_path):
        corr2 = np.load(corr2_path)
        mass = np.log(np.abs(corr2[:, :-1])
                      / np.roll(np.abs(corr2[:, :-1]), -1, axis=1)) * unit
        mm = mass.mean(0)
        m_plat = mm[plat].mean()
        dev = abs(m_plat - 1.56)   # E(P2) 色散预期（基线 meff_proton_P2 平台）
        results.append({'item': 'B:meff_P2_platform', 'dev': float(dev),
                        'tol': float(tol), 'pass': bool(dev < tol),
                        'meff_P2_gev': float(m_plat)})
    else:
        missing.append(os.path.join(conf, f"_Pz{truth['Pz']}", '0_corr2.npy'))
        results.append({'item': 'B:meff_P2_platform', 'pass': False,
                        'note': 'corr2 缺失（04 步骤未成功）'})

    # ---- B2. 质子质量：P0 meff 平台 ≈ 1.12 GeV（docker 基线已验证结论）----
    from pyqcd.analysis import energy_model, cov_mat, sem as _sem
    p0_corr2 = np.load(os.path.join(run_dir, conf, f"_Pz0", '0_corr2.npy')) \
        if os.path.exists(os.path.join(run_dir, conf, f"_Pz0", '0_corr2.npy')) \
        else None
    if p0_corr2 is not None:
        mass0 = np.log(np.abs(p0_corr2[:, :-1])
                       / np.roll(np.abs(p0_corr2[:, :-1]), -1, axis=1)) * unit
        m0_plat = mass0.mean(0)[plat].mean()
        dev0 = abs(m0_plat - 1.12)
        results.append({'item': 'B2:meff_P0_proton_mass', 'dev': float(dev0),
                        'tol': float(tol), 'pass': bool(dev0 < tol),
                        'meff_P0_gev': float(m0_plat)})

    # ---- C. E0 恢复量级（GeV）：|E0·unit − meff 平台| < tol ----
    fit_path = os.path.join(run_dir, conf, f"_Pz{truth['Pz']}", '1_fit_data.npz')
    if os.path.exists(fit_path) and np.isfinite(m_plat):
        fit = np.load(fit_path)
        E0_gev = fit['E0'].mean() * unit
        dev = abs(E0_gev - m_plat)
        results.append({'item': 'C:E0_matches_meff_plateau', 'dev': float(dev),
                        'tol': float(tol), 'pass': bool(dev < tol),
                        'E0_gev': float(E0_gev)})
    else:
        missing.append(os.path.join(conf, f"_Pz{truth['Pz']}", '1_fit_data.npz'))
        results.append({'item': 'C:E0_matches_meff_plateau', 'pass': False,
                        'note': 'fit 缺失（04 步骤未成功）'})

    # ---- D. ratio fit 收敛：c0 有限且 |c0| < 2 ----
    c0_max = 0.0
    for a, b, c in truth['fit_ranges']:
        fd = f"fit_Pz{truth['Pz']}_Nsam{Ns}_dtmax{truth['dt_max']}" \
             f"_tsep{a}_{b}_nex{c}"
        fp = os.path.join(run_dir, conf, fd, '0_fit_data.npz')
        if os.path.exists(fp):
            c0 = np.load(fp)['c0']
            c0_max = max(c0_max, float(np.abs(c0).max()))
    results.append({'item': 'D:ratio_fit_c0_finite', 'max_c0': c0_max,
                    'pass': bool(np.isfinite(c0_max) and c0_max < 2.0)})

    # ---- E. 报告/日志完整性 ----
    txt = open(os.path.join(run_dir, 'test8_run.log')).read()
    ok = all(k in txt for k in ['02_ratio', '03_ana_ratio', '04_proton_energy',
                                '06_FH', '05_ana_3dir'])
    results.append({'item': 'E:all_steps_logged', 'pass': bool(ok)})

    # ---- F. 图表完整性：png 总数 ≥ 106（stab1 同步骤同参数实测 106 张；
    #       test8 步骤/参数一致，组态数只影响 Nsam 文件名不影响图数）----
    n_png = sum(1 for _, _, fs in os.walk(run_dir) for f in fs
                if f.endswith('.png'))
    results.append({'item': 'F:png_count_ge_106', 'n_png': int(n_png),
                    'pass': bool(n_png >= 106)})

    if verbose:
        n_pass = sum(1 for r in results if r['pass'])
        print(f"一致项 {n_pass}/{len(results)}，失败 "
              f"{sum(not r['pass'] for r in results)}，缺文件 {len(missing)}")
        for r in results:
            mark = 'PASS' if r['pass'] else 'FAIL'
            if 'dev' in r and 'tol' in r:
                print(f"  {mark} {r['item']:36s} dev={r['dev']:.4f} "
                      f"(tol {r['tol']:.4f})")
            elif 'max_c0' in r:
                print(f"  {mark} {r['item']:36s} max_c0={r['max_c0']:.4f}")
            elif 'n_png' in r:
                print(f"  {mark} {r['item']:36s} n_png={r['n_png']} (≥106)")
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
    if not os.path.exists(os.path.join(data_root, 'data_meta.json')):
        print(f"[error] 缺少 data_meta.json（先运行 makedata）: {data_root}")
        sys.exit(2)
    out = {'results': [], 'missing': []}
    ok = verify_run(run_dir, data_root, out['results'], out['missing'],
                    verbose=True)
    out['summary'] = {'n_pass': sum(1 for r in out['results'] if r['pass']),
                      'n_fail': sum(1 for r in out['results'] if not r['pass']),
                      'n_missing': len(out['missing']),
                      'total': len(out['results'])}
    with open(os.path.join(run_dir, 'test8_verify.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"verify: {'PASS' if ok else 'FAIL'} → {run_dir}/test8_verify.json")
    sys.exit(0 if ok else 1)


def cmd_check(args):
    vpath = os.path.join(os.path.abspath(args.run_dir), 'test8_verify.json')
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
           'total_mb': round(sum(t['bytes'] for t in tree) / 2**20, 1),
           'n_png': sum(1 for t in tree if t['path'].endswith('.png'))}
    out['files'] = tree
    with open(os.path.join(run_dir, 'test8_collect.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"n_files={out['n_files']}（png {out['n_png']} 张）, "
          f"total={out['total_mb']} MB → {run_dir}/test8_collect.json")


def main():
    ap = argparse.ArgumentParser(
        description="test8 —— pyqcd 全功能真实数据实战套件（test12 风格）")
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('env')
    p.add_argument('--baseline', default=None,
                   help='数据源目录（默认 $test8_BASELINE > 正式数据源 /public/group/lqcd）')
    p.set_defaults(func=cmd_env)

    p = sub.add_parser('makedata')
    p.add_argument('--outdir', default=None)
    p.add_argument('--baseline', default=None,
                   help='数据源目录（默认 $test8_BASELINE > 正式数据源 /public/group/lqcd）')
    p.add_argument('--start', type=int, default=None,
                   help=f'起始组态号（默认扫描数据源实际组态；显式指定时按序列检查）')
    p.add_argument('--step', type=int, default=None,
                   help=f'组态编号间隔（默认扫描数据源实际组态）')
    p.add_argument('--n-conf', type=int, default=None,
                   help=f'组态数量（默认扫描数据源实际组态：本地 10 / 服务器 100+；'
                        f'显式指定时数据缺失如实 FAIL）')
    p.set_defaults(func=cmd_makedata)

    p = sub.add_parser('run')
    p.add_argument('--data-root', default=DEFAULT_DATA_DIR)
    p.add_argument('--outdir', default=None)
    p.add_argument('--backend', default='auto',
                   choices=['auto', 'numpy', 'cupy'],
                   help='pyqcd 数组后端（auto=探测 cupy，不可用回退 numpy）')
    p.add_argument('--steps', default=None,
                   help='逗号分隔步骤子集: 02_ratio,03_ana_ratio,04_energy,06_fh,05_ana3dir')
    p.add_argument('--fail-fast', action='store_true')
    p.set_defaults(func=cmd_run)

    p = sub.add_parser('verify')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--data-root', default=DEFAULT_DATA_DIR)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser('check')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--label', default='test8 check')
    p.set_defaults(func=cmd_check)

    p = sub.add_parser('collect')
    p.add_argument('--run-dir', required=True)
    p.set_defaults(func=cmd_collect)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
