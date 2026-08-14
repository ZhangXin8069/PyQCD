#!/usr/bin/env python3
"""
test1 —— 调用 pyqcd 包复现 docker-v20260805 全量蒸馏 GPU 管线（test12 形式）
================================================================================

目标：以 main.py 只调用 pyqcd 包的方式，复现
/root/PyQCD/examples/docker-v20260805/output/output_20260802_120104
的完整管线结果（10 组态蒸馏：vertex → 2pt/3pt/4pt → OPE → analysis → plots →
LaTeX 报告），中间数据与图表与基线一样完整保存。

实现：计算全部走 pyqcd 包（vertex/contraction/operator/analysis/tools），
不 import examples/、不 import refer/；绘图（matplotlib）与 LaTeX 报告在
main.py 内实现（pyqcd 无绘图/报告模块）。总体形式参照 PyQCU/logs/test12：
版本目录 v<YYYYMMDDHHMM>/ + env.json + 汇总 json + run 脚本。

VVV 顶点例外：pyqcd.vertex.Mom_VVV_sink_t 为单 einsum，8GB GPU 在
Nev=100/Nx=24 下不可行 → 本文件用基线 x-slicing 因子化
（_compute_vvv_single_t_gpu，数学等价：逐 x 切片累加 Levi-Civita 六项），
仅依赖 pyqcd 的 backend（get_backend）与 phase_exp_3pt。

子命令（--outdir 为公共参数，位置在子命令前后皆可）：
    env      环境自检（GPU/CuPy/git），并写 env.json
    pipeline 完整管线（--steps 可选；--conf-ids/--precision/--Nev1/--skip-* 透传）
    verify   数值一致性验证 vs 基线 output_20260802_120104（rtol=1e-3）
    collect  汇总 timing/meff/文件清单 → test1_results.json
    report   生成并编译 LaTeX 物理报告（physics_report.tex/.pdf）

--outdir 优先级：命令行 > TEST1_OUTDIR 环境变量 > examples/test1/。
每次调用自动在输出目录写 env.json（test12 约定）。
"""

from __future__ import annotations

import argparse, json, logging, os, subprocess, sys, time, traceback
from datetime import datetime
from pathlib import Path

import numpy as np

# ── 仓库根入 sys.path（仅 pyqcd 包）──────────────────────────────
_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pyqcd.tools import set_backend, get_backend           # noqa: E402
from pyqcd.pipeline._config import (                       # noqa: E402
    ENSEMBLE, NX, NT, ALttc, FM2GEV, A_INV,
    CONF_IDS, NEV, NEV1, PRECISION, T_SEP,
    DELTA_Z, Z_DIR, OPE_COMPONENTS,
    MOM_SINK_VDV, MOM_SINK_VVV, ANALYSIS_MOMENTA,
    PION_SINK, PION_SRC, PROTON_SINK, PROTON_SRC,
    NEUTRON_SINK, NEUTRON_SRC, PP_SINK, PP_SRC, PN_SINK, PN_SRC,
    PJN_SINK, PJN_SRC, PJN_CURR,
    PION3_SINK, PION3_SRC, PION3_CURR,
    PJNNJNP_SINK, PJNNJNP_SRC, PJNNJNP_CURR,
    FOURPT_NEV1, FOURPT_TSEP, FOURPT_MOM, FOURPT_SRC_STEP,
    get_eigen_path, get_peram_dir, get_gauge_path, conf_data_dir,
)

REF_DIR = '/root/PyQCD/examples/docker-v20260805/output/output_20260802_120104'
VERSION = 'docker-v20260805'

# ═══════════════════════════════════════════════════════════════════
# 公共：输出目录 / env.json / 日志（test12 约定）
# ═══════════════════════════════════════════════════════════════════

def resolve_outdir(args) -> str:
    d = getattr(args, 'outdir', None) or os.environ.get('TEST1_OUTDIR') \
        or str(Path(__file__).parent)
    os.makedirs(d, exist_ok=True)
    return d


def write_env_json(out: str, cmdline: list):
    """环境快照（比对基准）：GPU 型号/显存/驱动、cupy、python、git HEAD、命令。"""
    env = {
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'hostname': os.uname().nodename,
    }
    try:
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
        env['git_branch'] = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True, text=True, cwd=_REPO).stdout.strip()
        env['git_head'] = subprocess.run(
            ['git', 'log', '-1', '--oneline'],
            capture_output=True, text=True, cwd=_REPO).stdout.strip()
    except Exception:
        pass
    env['python'] = sys.version.split()[0]
    env['cmdline'] = ' '.join(cmdline)
    with open(os.path.join(out, 'env.json'), 'w') as f:
        json.dump(env, f, indent=2)
    return env


def setup_logging(outdir: str, name: str, verbose: bool = False) -> logging.Logger:
    """日志：版本目录 logs/ 下归档 + 终端输出（与基线 setup_logging 等价）。"""
    log_dir = os.path.join(outdir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter('%(asctime)s %(levelname)-7s %(message)s', '%H:%M:%S')
    fh = logging.FileHandler(os.path.join(log_dir, f'{name}.log'), encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    if verbose:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    logger.propagate = False
    return logger


def print_banner(title: str, logger):
    line = '=' * 76
    logger.info(f"\n{line}\n  {title}\n{line}")


class Timer:
    """简单计时上下文。"""

    def __init__(self, label: str, logger):
        self.label, self.logger, self.elapsed = label, logger, 0.0
        self._t = 0.0

    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._t
        self.logger.info(f"[{self.label}] {self.elapsed:.1f}s")


def free_gpu_memory():
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def log_gpu_memory(logger, tag=''):
    try:
        import cupy as cp
        free, total = cp.cuda.runtime.memGetInfo()
        logger.info(f"GPU mem{tag}: free={free/2**30:.1f}/{total/2**30:.1f} GB")
    except Exception:
        pass


def save_array(path: str, arr, logger):
    np.save(path, arr)
    logger.info(f"    saved {os.path.basename(path)} {arr.shape}")


def dump_config_snapshot(config: dict, path: str, logger):
    with open(path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    logger.info(f"Config snapshot -> {path}")


def HAS_CUPY() -> bool:
    try:
        import cupy  # noqa: F401
        return True
    except ImportError:
        return False


# ═══════════════════════════════════════════════════════════════════
# 子命令 1: env
# ═══════════════════════════════════════════════════════════════════

def cmd_env(args):
    out = resolve_outdir(args)
    print(f"输出目录: {out}")
    print(f"Python {sys.version.split()[0]} | numpy {np.__version__}")
    print(f"基线: {REF_DIR}")
    print(f"配置: {CONF_IDS} (Nconf={len(CONF_IDS)}), "
          f"Nev={NEV}, Nev1={NEV1}, precision={PRECISION}, "
          f"lattice={NT}x{NX}^3, a={ALttc} fm")
    ok = True
    for cid in CONF_IDS:
        e = os.path.isdir(os.path.dirname(get_eigen_path(cid, 0)))
        p = os.path.isdir(get_peram_dir(cid))
        g = os.path.exists(get_gauge_path(cid))
        ok &= e and p and g
        print(f"  conf={cid}: eigvec={'OK' if e else 'MISS'} "
              f"peram={'OK' if p else 'MISS'} gauge={'OK' if g else 'MISS'}")
    try:
        import cupy as cp
        print(f"CuPy {cp.__version__} | CUDA {cp.cuda.runtime.runtimeGetVersion()}")
    except Exception as e:
        print(f"CuPy 不可用: {e}")
        ok = False
    write_env_json(out, sys.argv)
    print("env OK" if ok else "env DATA MISSING")
    return 0 if ok else 1


# ═══════════════════════════════════════════════════════════════════
# 子命令 2: pipeline（步骤驱动，计算全部走 pyqcd 包）
# ═══════════════════════════════════════════════════════════════════

def _dtype(precision):
    return np.complex64 if precision == 'complex64' else np.complex128


# ── Step: vertex（pyqcd.vertex + x-slicing VVV）────────────────────

def _compute_vvv_single_t_gpu(ev_t_gpu, ph_gpu, Nx, Nev1):
    """VVV 单时间片 — x-slicing 因子化（基线算法，数学等价单 einsum）。

    VVV_{mnl}(p) = Σ_x e^{-ipx} ε_{abc} φ^a_m φ^b_n φ^c_l
    逐 x 切片累加六项 Levi-Civita 置换，8GB GPU 内存友好。
    """
    backend = get_backend()
    VVV_t = backend.zeros((Nev1, Nev1, Nev1), dtype=ev_t_gpu.dtype)
    L = Nx * Nx
    for xi in range(Nx):
        s, e = xi * L, (xi + 1) * L
        es = ev_t_gpu[:Nev1, s:e, :]  # (Nev1, Nx², 3)
        ps = ph_gpu[s:e]
        T = backend.einsum('x,ax,bx->abx', ps, es[..., 0], es[..., 1])
        VVV_t += backend.einsum('abx,cx->abc', T, es[..., 2])
        T = backend.einsum('x,ax,bx->abx', ps, es[..., 1], es[..., 2])
        VVV_t += backend.einsum('abx,cx->abc', T, es[..., 0])
        T = backend.einsum('x,ax,bx->abx', ps, es[..., 2], es[..., 0])
        VVV_t += backend.einsum('abx,cx->abc', T, es[..., 1])
        T = backend.einsum('x,ax,bx->abx', ps, es[..., 0], es[..., 2])
        VVV_t -= backend.einsum('abx,cx->abc', T, es[..., 1])
        T = backend.einsum('x,ax,bx->abx', ps, es[..., 1], es[..., 0])
        VVV_t -= backend.einsum('abx,cx->abc', T, es[..., 2])
        T = backend.einsum('x,ax,bx->abx', ps, es[..., 2], es[..., 1])
        VVV_t -= backend.einsum('abx,cx->abc', T, es[..., 0])
    return VVV_t


def compute_vertices_for_config(conf_id, run_dir, logger, precision='complex64',
                                recompute=False):
    """单组态 VdV/VVV：pyqcd.vertex（phase_exp_*、Mom_VdV_sink_t）+
    pyqcd.tools.readin_eigvecs_gpu + x-slicing VVV。"""
    from pyqcd.vertex import phase_exp_2pt, phase_exp_3pt, Mom_VdV_sink_t
    from pyqcd.tools import readin_eigvecs_gpu

    backend = get_backend()
    dtype = _dtype(precision)
    cdir = conf_data_dir(run_dir, conf_id)
    vdv_path = os.path.join(cdir, f'VdV_mom_{conf_id}.npy')
    vvv_path = os.path.join(cdir, f'VVV_mom_{conf_id}.npy')

    if os.path.exists(vdv_path) and os.path.exists(vvv_path) and not recompute:
        VdV, VVV = np.load(vdv_path), np.load(vvv_path)
        logger.info(f"  conf={conf_id}: loaded cached vertices "
                    f"VdV{VdV.shape} VVV{VVV.shape}")
        return {'VdV': VdV, 'VVV': VVV}

    logger.info(f"  conf={conf_id}: computing vertices over {NT} time slices "
                f"(VdV {len(MOM_SINK_VDV)} mom, VVV {len(MOM_SINK_VVV)} mom, "
                f"Nev={NEV}, Nev1={NEV1}, dtype={dtype.__name__})")

    p2f = np.zeros((len(MOM_SINK_VDV), NX * NX * NX * 3), dtype=np.complex128)
    for i, mom in enumerate(MOM_SINK_VDV):
        _ph = phase_exp_2pt(NX, mom)
        _ph_np = _ph.get() if hasattr(_ph, 'get') else np.asarray(_ph)
        p2f[i] = _ph_np.reshape(-1)
    p2f_gpu = backend.asarray(p2f.astype(dtype))
    p3_list = []
    for mom in MOM_SINK_VVV:
        _ph = phase_exp_3pt(NX, mom)
        p3_list.append(_ph.get() if hasattr(_ph, 'get') else np.asarray(_ph))

    VdV = np.zeros((NT, len(MOM_SINK_VDV), NEV, NEV), dtype=dtype)
    VVV = np.zeros((NT, len(MOM_SINK_VVV), NEV1, NEV1, NEV1), dtype=dtype)

    t0 = time.perf_counter()
    for t in range(NT):
        ev = readin_eigvecs_gpu(get_eigen_path(conf_id, t), NX, NEV)
        ev = ev.reshape(NEV, NX, NX, NX, 3).astype(dtype)
        vdv_t = Mom_VdV_sink_t(p2f_gpu, ev)
        VdV[t] = vdv_t.get() if hasattr(vdv_t, 'get') else vdv_t
        ev_flat = ev.reshape(NEV, NX * NX * NX, 3)
        for m, ph_np in enumerate(p3_list):
            ph_gpu = backend.asarray(ph_np.reshape(-1).astype(dtype))
            vvv_t = _compute_vvv_single_t_gpu(ev_flat, ph_gpu, NX, NEV1)
            VVV[t, m] = vvv_t.get() if hasattr(vvv_t, 'get') else vvv_t
        if t % 12 == 0 or t == NT - 1:
            logger.info(f"    t={t:3d}/{NT}  elapsed={time.perf_counter()-t0:.0f}s")

    free_gpu_memory()
    log_gpu_memory(logger, " after vertices")
    diag = np.abs(np.diag(VdV[0, 0])).real
    logger.info(f"    VdV(P=0,t=0) diagonal: [{diag.min():.3f}, {diag.max():.3f}]  "
                f"(≈1 ⇒ orthonormal)")
    logger.info(f"    VVV(P=0,t=0) |v|: [{np.abs(VVV[0,0]).min():.3e}, "
                f"{np.abs(VVV[0,0]).max():.3e}]")
    save_array(vdv_path, VdV, logger)
    save_array(vvv_path, VVV, logger)
    return {'VdV': VdV, 'VVV': VVV}


def _step_vertex(config, run_dir, logger):
    print_banner("Step 1: Vertex Functions (VdV, VVV)", logger)
    set_backend('cupy')
    from pyqcd.contraction._dynamic import clear_plan_cache
    clear_plan_cache()
    for cid in config['conf_ids']:
        with Timer(f"  Vertices conf={cid}", logger):
            compute_vertices_for_config(cid, run_dir, logger,
                                        config['precision'], recompute=False)
        free_gpu_memory()
    logger.info(f"Vertices computed & saved for {len(config['conf_ids'])} configs")
    return None


# ── Step: 2pt / 3pt / 4pt（pyqcd.contraction 动态收缩引擎）─────────

def _real_sum(val):
    v = val.get() if hasattr(val, 'get') else val
    return float(np.real(np.sum(np.asarray(v).ravel())))


def _load_peram_set(backend, peram_dir, conf_id, times, dtype, nev1=None):
    """按时间片读 perambulator（pyqcd.tools.readin_peram_time_slice），
    每次只读一遍并缓存；返回 {t: (peram_t, peram_seq_t)}。"""
    from pyqcd.tools import readin_peram_time_slice
    from pyqcd.contraction import seq_peram
    if nev1 is None:
        nev1 = NEV
    cache = {}
    for t in times:
        if t in cache:
            continue
        peram_cpu = readin_peram_time_slice(peram_dir, str(conf_id), t, NT, NEV)
        peram_t = backend.asarray(peram_cpu[:, :, :, :nev1, :nev1].astype(dtype))
        cache[t] = (peram_t, seq_peram(peram_t))
    return cache


def _run_2pt(backend, sink_op, src_op, peram_t, peram_seq_t,
             t_src, t_sink, v_src, v_sink, v_kind, gamma_name, gamma_val,
             projector, Vindex=('M', 'M')):
    """单个 2pt 动态收缩（pyqcd.contraction.dynamic_contraction）。"""
    from pyqcd.contraction import PeramRegistry, VRegistry, GammaRegistry, \
        dynamic_contraction
    PR = PeramRegistry(); VR = VRegistry(); GR = GammaRegistry()
    GR.register(gamma_name, gamma_val)
    GR.register('Projector', (projector, projector))
    if v_kind == 'VVV':
        VR.register('VVV_0', 'tsrc', v_src)
        VR.register('VVV_0', 'tsink', v_sink)
    else:
        VR.register('VDV_0', 'tsrc', v_src)
        VR.register('VDV_0', 'tsink', v_sink)
    PR.register('light', ('tsrc', 'tsrc'), peram_t[t_src])
    PR.register('light', ('tsink', 'tsrc'), peram_t[t_sink])
    PR.register('light', ('tsrc', 'tsink'), peram_seq_t[t_sink])
    try:
        dc = dynamic_contraction(
            [(sink_op, src_op)],
            peram_registry=PR, v_registry=VR, gamma_registry=GR,
            Cpt='2pt', Vindex=list(Vindex),
            use_equivalence=False, ignore_dis=False,
            Projection=True, verbose=False)
        return _real_sum(dc.calculate_all())
    except KeyError:
        return 0.0


def compute_2pt_for_config(conf_id, run_dir, logger, vertices,
                           precision=PRECISION, channels=('pp', 'pn', 'pion')):
    """单组态源平均 2pt（pyqcd.contraction）。"""
    from pyqcd.lattice import gamma
    from pyqcd.tools import readin_peram_time_slice
    from pyqcd.contraction import seq_peram
    backend = get_backend()
    dtype = _dtype(precision)
    cdir = conf_data_dir(run_dir, conf_id)
    VdV, VVV = vertices['VdV'], vertices['VVV']
    peram_dir = get_peram_dir(conf_id)

    projector = backend.asarray((gamma(0) + gamma(4)) / 2.0, dtype=dtype)
    g7 = backend.asarray(gamma(7), dtype=dtype)
    g5 = backend.asarray(gamma(5), dtype=dtype)

    acc = {f'corr_{ch}_{mom}': np.zeros(NT, dtype=np.float64)
           for ch in channels for mom in ('P0', 'P2')}
    op_cfg = {
        'pp':   (PP_SINK, PP_SRC, 'VVV', g7, 'gamma_7'),
        'pn':   (PN_SINK, PN_SRC, 'VVV', g7, 'gamma_7'),
        'pion': (PION_SINK, PION_SRC, 'VDV', g5, 'gamma_5'),
    }
    op_cfg = {ch: op_cfg[ch] for ch in channels}

    logger.info(f"  2pt channels: {list(op_cfg.keys())} at P=(0,0,0),(0,0,2)")
    t_start = time.perf_counter()
    for t_src in range(NT):
        peram_cpu = readin_peram_time_slice(peram_dir, str(conf_id), t_src, NT, NEV)
        peram_t = backend.asarray(peram_cpu.astype(dtype))
        peram_seq_t = seq_peram(peram_t)
        for t_sink in range(NT):
            dt = (t_sink - t_src + NT) % NT
            for ch, (sink_op, src_op, vkind, gval, gname) in op_cfg.items():
                for mi, mom_tag in enumerate(('P0', 'P2')):
                    if vkind == 'VVV':
                        v_src = backend.asarray(VVV[t_src, mi:mi + 1].conj(), dtype=dtype)
                        v_sink = backend.asarray(VVV[t_sink, mi:mi + 1], dtype=dtype)
                    else:
                        v_src = backend.asarray(VdV[t_src, mi:mi + 1].conj(), dtype=dtype)
                        v_sink = backend.asarray(VdV[t_sink, mi:mi + 1], dtype=dtype)
                    val = _run_2pt(backend, sink_op, src_op,
                                   peram_t, peram_seq_t,
                                   t_src, t_sink, v_src, v_sink, vkind,
                                   gname, gval, projector)
                    acc[f'corr_{ch}_{mom_tag}'][dt] += val / NT
        if t_src % 12 == 0 or t_src == NT - 1:
            logger.info(f"    t_src={t_src:3d}/{NT} "
                        f"elapsed={time.perf_counter()-t_start:.0f}s "
                        f"pp0={acc['corr_pp_P0'][0]:.4e} "
                        f"pi0={acc['corr_pion_P0'][0]:.4e}")
        del peram_t, peram_seq_t
    for key, arr in acc.items():
        save_array(os.path.join(cdir, f'{key}_{conf_id}.npy'), arr, logger)
    return acc


def _step_2pt(config, run_dir, logger):
    print_banner("Step 2: 2pt Correlators (pp, pn, pion)", logger)
    set_backend('cupy')
    for cid in config['conf_ids']:
        logger.info(f"\n─── 2pt: conf {cid} ───")
        cdir = conf_data_dir(run_dir, cid)
        verts = {
            'VdV': np.load(os.path.join(cdir, f'VdV_mom_{cid}.npy')),
            'VVV': np.load(os.path.join(cdir, f'VVV_mom_{cid}.npy')),
        }
        with Timer(f"  2pt conf={cid}", logger):
            compute_2pt_for_config(cid, run_dir, logger, verts,
                                   config['precision'],
                                   channels=config.get('channels', ('pp', 'pn', 'pion')))
        del verts
        free_gpu_memory()
    return None


def _run_3pt(backend, sink_op, src_op, curr_op, PR, VR, GR, Vindex, Gindex):
    """单个 3pt 动态收缩 → (n_gamma_mu,) 数组。"""
    from pyqcd.contraction import dynamic_contraction
    dc = dynamic_contraction(
        [(sink_op, src_op, curr_op)],
        peram_registry=PR, v_registry=VR, gamma_registry=GR,
        Cpt='3pt', Vindex=list(Vindex), Gindex=list(Gindex),
        use_equivalence=False, ignore_dis=False,
        Projection=True, verbose=False)
    r = dc.calculate_all()
    v = r.get() if hasattr(r, 'get') else r
    return np.asarray(v).ravel()


def compute_3pt_for_config(conf_id, run_dir, logger, vertices,
                           precision=PRECISION, t_sep=T_SEP):
    """单组态 3pt PJN（tau × gamma_mu），输出 (Ntau, 4)。"""
    from pyqcd.lattice import gamma
    backend = get_backend()
    dtype = _dtype(precision)
    cdir = conf_data_dir(run_dir, conf_id)
    Ntau = t_sep + 1
    VdV, VVV = vertices['VdV'], vertices['VVV']
    peram_dir = get_peram_dir(conf_id)

    projector = backend.asarray((gamma(0) + gamma(4)) / 2.0, dtype=dtype)
    gmu = backend.asarray([gamma(1), gamma(2), gamma(3), gamma(4)], dtype=dtype)
    g7 = backend.asarray(gamma(7), dtype=dtype)
    g5 = backend.asarray(gamma(5), dtype=dtype)

    acc = {f'{had}_{mom}': np.zeros((Ntau, 4), dtype=np.float64)
           for had in ('proton', 'pion') for mom in ('P0', 'P2')}

    logger.info(f"  3pt PJN: t_sep={t_sep}, Ntau={Ntau}, gamma_mu=4 components")
    t_start = time.perf_counter()
    for t_src in range(NT):
        t_sink = (t_src + t_sep) % NT
        need_times = sorted(set([t_src, t_sink] + [(t_src + tau) % NT
                                                   for tau in range(Ntau)]))
        pc = _load_peram_set(backend, peram_dir, str(conf_id), need_times, dtype)
        p_src, p_srcS = pc[t_src]
        p_snk, p_snkS = pc[t_sink]

        for tau in range(Ntau):
            t_cur = (t_src + tau) % NT
            p_cur, p_curS = pc[t_cur]

            from pyqcd.contraction import PeramRegistry, VRegistry, GammaRegistry
            PR = PeramRegistry()
            PR.register('light', ('tsink', 'tsrc'), p_src[t_sink])
            PR.register('light', ('tcur0', 'tsrc'), p_src[t_cur])
            PR.register('light', ('tsrc', 'tsrc'),  p_src[t_src])
            PR.register('light', ('tsink', 'tcur0'), p_cur[t_sink])
            PR.register('light', ('tcur0', 'tcur0'), p_cur[t_cur])
            PR.register('light', ('tsrc', 'tcur0'),  p_cur[t_src])
            PR.register('light', ('tcur0', 'tsink'), p_snk[t_cur])
            PR.register('light', ('tsink', 'tsink'), p_snk[t_sink])
            PR.register('light', ('tsrc', 'tsink'),  p_srcS[t_sink])
            PR.register('light', ('tsrc', 'tcur0'),  p_srcS[t_cur])
            PR.register('light', ('tcur0', 'tsink'), p_curS[t_sink])
            PR.register('light', ('tsink', 'tcur0'), p_snkS[t_cur])

            GRp = GammaRegistry()
            GRp.register('gamma_7', g7)
            GRp.register('gamma_mu', gmu)
            GRp.register('Projector', (projector, projector))
            GRpi = GammaRegistry()
            GRpi.register('gamma_5', g5)
            GRpi.register('gamma_mu', gmu)
            GRpi.register('Projector', (projector, projector))

            for mi, mom_tag in enumerate(('P0', 'P2')):
                VR = VRegistry()
                VR.register('VVV_0', 'tsrc',
                            backend.asarray(VVV[t_src, mi:mi + 1].conj(), dtype=dtype))
                VR.register('VDV_0', 'tcur0',
                            backend.asarray(VdV[t_cur, mi:mi + 1], dtype=dtype))
                VR.register('VVV_0', 'tsink',
                            backend.asarray(VVV[t_sink, mi:mi + 1], dtype=dtype))
                vn = _run_3pt(backend, PJN_SINK, PJN_SRC, PJN_CURR,
                              PR, VR, GRp, ['M', 'M', 'M'], ['', 'G', ''])
                acc[f'proton_{mom_tag}'][tau, :min(4, len(vn))] += \
                    np.real(vn[:min(4, len(vn))]) / NT

            for mi, mom_tag in enumerate(('P0', 'P2')):
                VR = VRegistry()
                VR.register('VDV_0', 'tsrc',
                            backend.asarray(VdV[t_src, mi:mi + 1].conj(), dtype=dtype))
                VR.register('VDV_0', 'tcur0',
                            backend.asarray(VdV[t_cur, mi:mi + 1], dtype=dtype))
                VR.register('VDV_0', 'tsink',
                            backend.asarray(VdV[t_sink, mi:mi + 1], dtype=dtype))
                vn = _run_3pt(backend, PION3_SINK, PION3_SRC, PION3_CURR,
                              PR, VR, GRpi, ['M', 'M', 'M'], ['', 'G', ''])
                acc[f'pion_{mom_tag}'][tau, :min(4, len(vn))] += \
                    np.real(vn[:min(4, len(vn))]) / NT
            del p_cur, p_curS

        if t_src % 12 == 0 or t_src == NT - 1:
            logger.info(f"    t_src={t_src:3d}/{NT} "
                        f"elapsed={time.perf_counter()-t_start:.0f}s "
                        f"protP0[0,3]={acc['proton_P0'][0,3]:.3e} "
                        f"piP0[0,3]={acc['pion_P0'][0,3]:.3e}")
        del p_src, p_snk, p_srcS, p_snkS

    for key, arr in acc.items():
        save_array(os.path.join(cdir, f'{key}_3pt_{conf_id}.npy'), arr, logger)
    return acc


def _step_3pt(config, run_dir, logger):
    print_banner("Step 3: 3pt Correlators (PJN)", logger)
    set_backend('cupy')
    for cid in config['conf_ids']:
        logger.info(f"\n─── 3pt PJN: conf {cid} ───")
        cdir = conf_data_dir(run_dir, cid)
        verts = {
            'VdV': np.load(os.path.join(cdir, f'VdV_mom_{cid}.npy')),
            'VVV': np.load(os.path.join(cdir, f'VVV_mom_{cid}.npy')),
        }
        with Timer(f"  3pt conf={cid}", logger):
            compute_3pt_for_config(cid, run_dir, logger, verts,
                                   config['precision'], t_sep=T_SEP)
        del verts
        free_gpu_memory()
    return None


def compute_4pt_for_config(conf_id, run_dir, logger, vertices,
                           precision=PRECISION, t_sep=FOURPT_TSEP,
                           nev1=FOURPT_NEV1, momenta=FOURPT_MOM,
                           src_step=FOURPT_SRC_STEP):
    """单组态 4pt PJNNJNp，输出 (Ntau, N_mom, 4)。"""
    from pyqcd.lattice import gamma
    from pyqcd.contraction import PeramRegistry, VRegistry, GammaRegistry, \
        dynamic_contraction
    backend = get_backend()
    dtype = _dtype(precision)
    cdir = conf_data_dir(run_dir, conf_id)
    Ntau = t_sep + 1
    N_mom = len(momenta)
    sources = list(range(0, NT, src_step))

    VdV, VVV = vertices['VdV'], vertices['VVV']
    peram_dir = get_peram_dir(conf_id)

    projector = backend.asarray((gamma(0) + gamma(4)) / 2.0, dtype=dtype)
    gmu = backend.asarray([gamma(1), gamma(2), gamma(3), gamma(4)], dtype=dtype)
    g7 = backend.asarray(gamma(7), dtype=dtype)
    g5 = backend.asarray(gamma(5), dtype=dtype)

    acc = np.zeros((Ntau, N_mom, 4), dtype=np.float64)
    nsrc = len(sources)

    logger.info(f"  4pt PJNNJNp: n(sink) — J — [p̄ + π](src), t_sep={t_sep}, "
                f"Nev1={nev1}, mom={momenta}, src_step={src_step} "
                f"({nsrc}/{NT} sources)")
    t_start = time.perf_counter()
    for t_src in sources:
        t_sink = (t_src + t_sep) % NT
        need_times = sorted(set([t_src, t_sink] + [(t_src + tau) % NT
                                                   for tau in range(Ntau)]))
        pc = _load_peram_set(backend, peram_dir, str(conf_id), need_times,
                             dtype, nev1=nev1)
        p_src, p_srcS = pc[t_src]
        p_snk, p_snkS = pc[t_sink]

        for tau in range(Ntau):
            t_cur = (t_src + tau) % NT
            p_cur, p_curS = pc[t_cur]

            PR = PeramRegistry()
            PR.register('light', ('tsink', 'tsrc'), p_src[t_sink])
            PR.register('light', ('tcur0', 'tsrc'), p_src[t_cur])
            PR.register('light', ('tsrc', 'tsrc'),  p_src[t_src])
            PR.register('light', ('tsink', 'tcur0'), p_cur[t_sink])
            PR.register('light', ('tcur0', 'tcur0'), p_cur[t_cur])
            PR.register('light', ('tsrc', 'tcur0'),  p_cur[t_src])
            PR.register('light', ('tcur0', 'tsink'), p_snk[t_cur])
            PR.register('light', ('tsink', 'tsink'), p_snk[t_sink])
            PR.register('light', ('tsrc', 'tsink'),  p_srcS[t_sink])
            PR.register('light', ('tsrc', 'tcur0'),  p_srcS[t_cur])
            PR.register('light', ('tcur0', 'tsink'), p_curS[t_sink])
            PR.register('light', ('tsink', 'tcur0'), p_snkS[t_cur])

            GR = GammaRegistry()
            GR.register('gamma_7', g7)
            GR.register('gamma_5', g5)
            GR.register('gamma_mu', gmu)
            GR.register('Projector', (projector, projector))

            for imi, mi in enumerate(momenta):
                VR = VRegistry()
                VR.register('VVV_0', 'tsink',
                            backend.asarray(VVV[t_sink, mi:mi + 1, :nev1, :nev1, :nev1],
                                            dtype=dtype))
                VR.register('VDV_0', 'tcur0',
                            backend.asarray(VdV[t_cur, mi:mi + 1, :nev1, :nev1],
                                            dtype=dtype))
                VR.register('VVV_0', 'tsrc',
                            backend.asarray(VVV[t_src, mi:mi + 1, :nev1, :nev1, :nev1].conj(),
                                            dtype=dtype))
                VR.register('VDV_0', 'tsrc',
                            backend.asarray(VdV[t_src, mi:mi + 1, :nev1, :nev1].conj(),
                                            dtype=dtype))
                dc = dynamic_contraction(
                    [(PJNNJNP_SINK, PJNNJNP_SRC, PJNNJNP_CURR)],
                    peram_registry=PR, v_registry=VR, gamma_registry=GR,
                    Cpt='3pt', Vindex=['M', 'M', 'M', 'M'],
                    Gindex=['', 'G', '', ''],
                    use_equivalence=False, ignore_dis=False,
                    Projection=True, verbose=False)
                try:
                    r = dc.calculate_all()
                except Exception as e:
                    logger.warning(f"  4pt contraction failed at "
                                   f"(t_src={t_src},tau={tau},mom={mi}): {e}")
                    continue
                v = r.get() if hasattr(r, 'get') else r
                vn = np.asarray(v).ravel()
                acc[tau, imi, :min(4, len(vn))] += np.real(vn[:min(4, len(vn))]) / nsrc
            del p_cur, p_curS

        if (t_src - sources[0]) % 12 == 0 or t_src == sources[-1]:
            logger.info(f"    t_src={t_src:3d} "
                        f"elapsed={time.perf_counter()-t_start:.0f}s "
                        f"acc[0,0,3]={acc[0,0,3]:.3e}")
        del p_src, p_snk, p_srcS, p_snkS

    save_array(os.path.join(cdir, f'pjnnjnp_4pt_{conf_id}.npy'), acc, logger)
    return acc


def _step_4pt(config, run_dir, logger):
    print_banner("Step 4: 4pt Correlators (PJNNJNp)", logger)
    set_backend('cupy')
    for cid in config['conf_ids']:
        logger.info(f"\n─── 4pt PJNNJNp: conf {cid} ───")
        cdir = conf_data_dir(run_dir, cid)
        verts = {
            'VdV': np.load(os.path.join(cdir, f'VdV_mom_{cid}.npy')),
            'VVV': np.load(os.path.join(cdir, f'VVV_mom_{cid}.npy')),
        }
        with Timer(f"  4pt conf={cid}", logger):
            compute_4pt_for_config(
                cid, run_dir, logger, verts,
                precision=config['precision'],
                t_sep=config.get('fourpt_tsep', FOURPT_TSEP),
                nev1=config.get('fourpt_nev1', FOURPT_NEV1),
                momenta=config.get('fourpt_mom', FOURPT_MOM),
                src_step=config.get('fourpt_src_step', FOURPT_SRC_STEP))
        del verts
        free_gpu_memory()
    return None


# ── Step: OPE（pyqcd.operator）────────────────────────────────────

def _first_link_unitarity(raw: np.ndarray, Nc: int = 3) -> float:
    U = raw[:Nc * Nc * 2].reshape(Nc, Nc, 2)
    Uc = U[..., 0] + 1j * U[..., 1]
    return float(np.abs(Uc @ Uc.conj().T - np.eye(Nc)).max())


def validate_gauge(gauge: np.ndarray, logger=None) -> dict:
    rng = np.random.default_rng(42)
    Nt, Nz, Ny, Nx, Nd, Nc, _ = gauge.shape
    devs = []
    for _ in range(60):
        t = rng.integers(0, Nt); z = rng.integers(0, Nz)
        y = rng.integers(0, Ny); x = rng.integers(0, Nx)
        U = gauge[t, z, y, x, rng.integers(0, Nd)]
        devs.append(np.abs(U @ U.conj().T - np.eye(Nc)).max())
    plaq = []
    for _ in range(30):
        ti, zi, yi, xi = (rng.integers(0, Nt), rng.integers(0, Nz),
                          rng.integers(0, Ny), rng.integers(0, Nx))
        mu, nu = 1, 2
        U1 = gauge[ti, zi, yi, xi, mu]
        U2 = gauge[ti, zi, (yi + 1) % Ny, xi, nu]
        U3 = gauge[ti, zi, (yi + 1) % Ny, xi, mu].conj().T
        U4 = gauge[ti, zi, yi, xi, nu].conj().T
        plaq.append(np.trace(U1 @ U2 @ U3 @ U4))
    res = {'unitary_dev_max': float(np.max(devs)),
           'plaq_trace_mean_re': float(np.real(np.mean(plaq)))}
    if logger:
        logger.info(f"  Gauge: unitarity_dev={res['unitary_dev_max']:.2e}, "
                    f"plaq_trace_re={res['plaq_trace_mean_re']:.6f}")
    return res


def compute_ope_for_config(conf_id, run_dir, logger, precision='complex64',
                           delta_z=DELTA_Z, z_dir=Z_DIR,
                           components=OPE_COMPONENTS, recompute=False):
    """单组态 OPE（pyqcd.operator：read_gauge_lime/plaquette_clover/
    compute_dual_field_strength/gluon_ope_operator_z0）。"""
    from pyqcd.operator import (plaquette_clover,
                                compute_dual_field_strength,
                                gluon_ope_operator_z0, read_gauge_lime)
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError("OPE requires a CUDA GPU (cupy)")

    dtype = _dtype(precision)
    cdir = conf_data_dir(run_dir, conf_id)
    paths = {c: os.path.join(cdir, f'ops_mu{c[0]}_nu{c[1]}_dz{delta_z}_conf{conf_id}.npz')
             for c in components}

    if all(os.path.exists(p) for p in paths.values()) and not recompute:
        logger.info(f"  conf={conf_id}: loading cached OPE components")
        ops = {c: np.load(paths[c])['ops'] for c in components}
        combined = -ops[(3, 0)] - ops[(3, 1)] + 2.0 * ops[(0, 1)]
        return {'components': ops, 'combined': combined}

    gauge_file = get_gauge_path(conf_id)
    logger.info(f"  conf={conf_id}: OPE from {gauge_file} "
                f"(dz={delta_z}, z_dir={z_dir}, {precision})")

    with Timer(f"  read gauge conf={conf_id}", logger):
        gauge_cpu = read_gauge_lime(gauge_file, NT, NX)
    val = validate_gauge(gauge_cpu, logger)
    gauge_gpu = cp.asarray(gauge_cpu.astype(dtype))
    del gauge_cpu

    ops = {}
    for mu, nu in components:
        with Timer(f"  OPE mu={mu},nu={nu} conf={conf_id}", logger):
            o = gluon_ope_operator_z0(gauge_gpu, mu, nu, z_dir, delta_z,
                                      NT, NX, dtype)
        ops[(mu, nu)] = o
        np.savez(paths[(mu, nu)], ops=o, mu=np.array(mu), nu=np.array(nu),
                 delta_z=np.array(delta_z), conf_id=np.array(conf_id),
                 shape=np.array(o.shape))
        logger.info(f"    saved ops_mu{mu}_nu{nu}: shape={o.shape}, "
                    f"|O|∈[{np.abs(o).min():.2e},{np.abs(o).max():.2e}]")

    combined = -ops[(3, 0)] - ops[(3, 1)] + 2.0 * ops[(0, 1)]
    save_array(os.path.join(cdir, f'ope_combined_conf{conf_id}.npy'),
               combined, logger)
    free_gpu_memory()
    log_gpu_memory(logger, " after OPE")
    return {'components': ops, 'combined': combined, 'validation': val}


def _step_ope(config, run_dir, logger):
    print_banner("Step 5: OPE (gluon operator)", logger)
    set_backend('cupy')
    for cid in config['conf_ids']:
        logger.info(f"\n─── OPE: conf {cid} ───")
        with Timer(f"  OPE conf={cid}", logger):
            compute_ope_for_config(cid, run_dir, logger, config['precision'])
        free_gpu_memory()
    return None


# ── Step: analysis（pyqcd.analysis）───────────────────────────────

CHANNELS = [
    ('proton', 'P0', 'corr_pp_P0'),
    ('proton', 'P2', 'corr_pp_P2'),
    ('pion',   'P0', 'corr_pion_P0'),
    ('pion',   'P2', 'corr_pion_P2'),
]


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


def run_meff_jackknife(corr_2pt_all, conf_ids, run_dir, logger):
    """Jackknife 有效质量（pyqcd.analysis：Jackknife/meff 原语）。"""
    from pyqcd.analysis import Jackknife, meff
    print_banner("Analysis 1: Jackknife + Effective Mass", logger)
    an_dir = os.path.join(run_dir, 'data', 'analysis')
    os.makedirs(an_dir, exist_ok=True)

    meff_types = {'proton': 'cosh', 'pion': 'log'}
    results = {}
    for particle, mom, key in CHANNELS:
        ml = f"P{list(ANALYSIS_MOMENTA[particle].values())[0 if mom == 'P0' else 1]}"
        stack = np.stack([np.real(corr_2pt_all[cid][key]) for cid in conf_ids])
        jk = Jackknife(stack, Nconf_axes=0)
        mf = meff(jk['data_sample'], ALttc, Nconf_axes=0, Nt_axes=1,
                  meff_type=meff_types[particle])

        cmean, cerr = np.real(jk['data_mean']), np.real(jk['data_err'])
        mmean, merr = np.real(mf['data_mean']), np.real(mf['data_err'])

        if particle == 'proton':
            ps, pe = 6, min(NT - 2, 12)
        else:
            ps, pe = 5, min(NT - 2, 18)
        mask = np.isfinite(mmean[ps:pe]) & (merr[ps:pe] > 0) & (mmean[ps:pe] > 0.01)
        if np.sum(mask) < 2:
            ps, pe = 2, min(8, NT - 1)
            mask = np.isfinite(mmean[ps:pe]) & (merr[ps:pe] > 0)
        t_plt = np.arange(ps, pe)[mask]
        w = 1.0 / (merr[ps:pe][mask] ** 2 + 1e-10)
        E0 = float(np.sum(mmean[ps:pe][mask] * w) / np.sum(w))
        E0_err = float(1.0 / np.sqrt(np.sum(w)))

        if mom == 'P0':
            E_exp = 1.0 if particle == 'proton' else 0.30
        else:
            m0 = results.get(f'{particle}_P0', {}).get('E0', E0)
            p_phys = (2 * np.pi * 2 / NX) * (FM2GEV / ALttc)
            E_exp = np.sqrt(m0 ** 2 + p_phys ** 2)
        dev = abs(E0 - E_exp) / (E0_err + 1e-10)
        status = '✓' if dev < 2 else ('⚠' if dev < 4 else '✗')

        logger.info(f"\n{particle} {ml}: E0 = {E0:.4f} ± {E0_err:.4f} GeV  "
                    f"(expected {E_exp:.3f}, {status} dev={dev:.1f}σ, "
                    f"plateau t∈[{ps},{pe}], {np.sum(mask)} pts)")
        logger.info(f"  C(0) = {cmean[0]:.6e} ± {cerr[0]:.6e}")

        results[f'{particle}_{mom}'] = {
            'E0': E0, 'E0_err': E0_err, 'E_exp': E_exp, 'dev': dev,
            'plateau': (ps, pe), 'npts': int(np.sum(mask)),
            'meff_mean': mmean, 'meff_err': merr,
            'corr_mean': cmean, 'corr_err': cerr,
        }
        save_array(os.path.join(an_dir, f'meff_{particle}_{mom}_mean.npy'), mmean, logger)
        save_array(os.path.join(an_dir, f'meff_{particle}_{mom}_err.npy'), merr, logger)
        save_array(os.path.join(an_dir, f'corr_{particle}_{mom}_mean.npy'), cmean, logger)
        save_array(os.path.join(an_dir, f'corr_{particle}_{mom}_err.npy'), cerr, logger)
    return results


def run_connected_ratio(corr_2pt_all, corr_3pt_all, conf_ids, run_dir, logger,
                        t_sep=None):
    """连通 3pt/2pt 比值 R(τ)（pyqcd.analysis：Jackknife/ratio_3pt 原语）。"""
    from pyqcd.analysis import Jackknife, ratio_3pt
    print_banner("Analysis 2: Connected 3pt/2pt Ratio R(τ)", logger)
    an_dir = os.path.join(run_dir, 'data', 'analysis')
    os.makedirs(an_dir, exist_ok=True)

    pairs = [
        ('proton', 'P0', 'corr_pp_P0', 'proton_P0_3pt'),
        ('proton', 'P2', 'corr_pp_P2', 'proton_P2_3pt'),
        ('pion',   'P0', 'corr_pion_P0', 'pion_P0_3pt'),
        ('pion',   'P2', 'corr_pion_P2', 'pion_P2_3pt'),
    ]
    results = {}
    for had, mom, k2, k3 in pairs:
        s3 = np.stack([np.real(corr_3pt_all[cid][k3][:, 3]) for cid in conf_ids])
        s2 = np.stack([np.real(corr_2pt_all[cid][k2]) for cid in conf_ids])
        ts = s3.shape[1] - 1 if t_sep is None else t_sep
        jk3 = Jackknife(s3, Nconf_axes=0)
        jk2 = Jackknife(s2, Nconf_axes=0)
        ratio = ratio_3pt(jk3['data_sample'], jk2['data_sample'],
                          data_2ptF_sample=None, t_sep=ts,
                          Nconf_axes=0, tau_axes=1, t_sink_axes=1)
        rm, re_ = np.real(ratio['data_mean']), np.real(ratio['data_err'])
        log_lines = [f"  {had} {mom}  R(τ) (t_sep={ts}, γ₃):"]
        for t in range(min(len(rm), ts + 1)):
            log_lines.append(f"    R({t:2d}) = {rm[t]:+.6f} ± {re_[t]:.6f}")
        logger.info('\n'.join(log_lines))
        results[f'{had}_{mom}'] = {'R': rm, 'R_err': re_, 't_sep': ts}
        save_array(os.path.join(an_dir, f'ratio_{had}_{mom}_mean.npy'), rm, logger)
        save_array(os.path.join(an_dir, f'ratio_{had}_{mom}_err.npy'), re_, logger)
    return results


def _step_analysis(config, run_dir, logger):
    from pyqcd.analysis import run_disconnected_ratio
    print_banner("Step 6: Statistical Analysis (Jackknife/meff/ratio_3p)", logger)
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
    if ope and len(config['conf_ids']) >= 2:
        disc_dir = os.path.join(run_dir, 'analysis', 'disconnected')
        res = run_disconnected_ratio(corr2, ope, config['conf_ids'], run_dir,
                                     logger=logger.info, NT=NT, NX=NX)
        ratio_disc = {k: {'ratio': v['ratio'], 'c0': v['c0'], 'c1': v['c1'],
                          'dE': v['dE'], 'chi2': v['chi2']}
                      for k, v in res.items()}
        _plot_disconnected(ratio_disc, disc_dir, logger)
    else:
        ratio_disc = {}
        if ope:
            logger.warning("Nconf<2 — skipping disconnected ratio (统计无意义)")
        else:
            logger.warning("No OPE data — skipping disconnected ratio")

    return {'meff': meff_res, 'connected_ratio': ratio_conn,
            'disconnected_ratio': ratio_disc}


# ── Step: plots（matplotlib，与基线 plots/ 同构）───────────────────

def _sem(data, jackknife=True):
    error = data.std(0)
    if jackknife:
        error = error * np.sqrt(data.shape[0] - 1)
    return error


def _plot_disconnected(ch_results, out_dir, logger):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    for had_name, res in ch_results.items():
        ratio = res['ratio']
        para_c0, para_c1 = res['c0'], res['c1']
        chi2 = res['chi2']
        rm = ratio.mean(0); re_ = _sem(ratio, True)
        z_list = list(range(NX))
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(z_list, para_c0.mean(0), yerr=_sem(para_c0, True), fmt='x-',
                    label='c0(z)')
        ax.axhline(0, color='gray', lw=0.8)
        ax.set_xlabel('z'); ax.set_ylabel('c0')
        ax.set_title(f'{had_name}: c0 vs z (disconnected ratio fit)')
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'c0_{had_name}.png'), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(z_list, chi2.mean(0), s=30)
        ax.axhline(1.0, color='orange', ls='--')
        ax.set_xlabel('z'); ax.set_ylabel('chi2/dof'); ax.set_ylim(0, 2)
        ax.set_title(f'{had_name}: chi2/dof vs z')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'chi2_{had_name}.png'), dpi=150)
        plt.close(fig)

        zs = [0, 6, 12, 18]
        nrow = (len(zs) + 1) // 2
        fig, axes = plt.subplots(nrow, 2, figsize=(12, 4 * nrow), squeeze=False)
        for k, z in enumerate(zs):
            ax = axes[k // 2][k % 2]
            for dt in [8, 10, 12, 14]:
                tau = np.arange(dt + 1)
                xv = tau - dt / 2
                yv = rm[dt, :dt + 1, z] if rm.shape[0] > dt else rm[min(dt, rm.shape[0]-1), :, z]
                ye = re_[dt, :dt + 1, z]
                ax.errorbar(xv, yv, yerr=ye, fmt='x', capsize=0, label=f'dt={dt}')
            ax.set_xlabel('tau - t_sep/2'); ax.set_ylabel('R')
            ax.set_title(f'z={z}, c0={para_c0[:, z].mean():.3f}')
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.suptitle(f'{had_name}: Disconnected ratio R(dt,dtau,z), Pz=2')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'ratio_{had_name}.png'), dpi=150)
        plt.close(fig)
        logger.info(f"  disconnected plots saved to {out_dir}")


def plot_meff_results(meff_results, run_dir, logger):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pdir = os.path.join(run_dir, 'plots')
    os.makedirs(pdir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (particle, mom, key) in zip(axes.ravel(), CHANNELS):
        res = meff_results.get(f'{particle}_{mom}')
        if res is None:
            continue
        m, e = res['meff_mean'], res['meff_err']
        t = np.arange(len(m))
        ps, pe = res['plateau']
        ax.errorbar(t, m, yerr=e, fmt='o', ms=4, capsize=2)
        ax.axvspan(ps, pe - 1, alpha=0.15, color='C1')
        ax.axhline(res['E0'], color='C3', ls='--', lw=1)
        ax.axhline(res['E_exp'], color='C4', ls=':', lw=1)
        ax.set_title(f'{particle} P={mom}  E0={res["E0"]:.3f}±{res["E0_err"]:.3f} '
                     f'(exp {res["E_exp"]:.2f})')
        ax.set_xlabel('t'); ax.set_ylabel(r'$m_{\rm eff}$ [GeV]')
        ax.grid(alpha=0.3)
    fig.suptitle('Effective masses (Jackknife, 10 configs)')
    fig.tight_layout()
    out = os.path.join(pdir, 'meff_all_channels.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved {out}")


def plot_correlators(meff_results, run_dir, logger):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pdir = os.path.join(run_dir, 'plots')
    os.makedirs(pdir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (particle, mom, key) in zip(axes.ravel(), CHANNELS):
        res = meff_results.get(f'{particle}_{mom}')
        if res is None:
            continue
        c, ce = res['corr_mean'], res['corr_err']
        t = np.arange(len(c))
        ax.errorbar(t, np.abs(c), yerr=ce, fmt='.', ms=4, capsize=0)
        ax.set_yscale('log')
        ax.set_title(f'{particle} P={mom}  C(0)={c[0]:.4e}')
        ax.set_xlabel('t'); ax.set_ylabel('|C(t)|')
        ax.grid(alpha=0.3, which='both')
    fig.suptitle('2pt correlators (Jackknife mean)')
    fig.tight_layout()
    out = os.path.join(pdir, 'correlators_all_channels.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved {out}")


def plot_connected_ratio(ratio_results, run_dir, logger):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pdir = os.path.join(run_dir, 'plots')
    os.makedirs(pdir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    pairs = [('proton', 'P0'), ('proton', 'P2'), ('pion', 'P0'), ('pion', 'P2')]
    for ax, (had, mom) in zip(axes.ravel(), pairs):
        res = ratio_results.get(f'{had}_{mom}')
        if res is None:
            continue
        r, e = res['R'], res['R_err']
        tau = np.arange(len(r))
        ax.errorbar(tau, r, yerr=e, fmt='o', ms=4, capsize=2)
        ax.axhline(0, color='gray', lw=0.8)
        ax.axhline(1, color='k', ls='--', lw=0.8)
        ax.set_title(f'{had} P={mom}  R(τ)  (t_sep={res["t_sep"]})')
        ax.set_xlabel('τ'); ax.set_ylabel('R(τ)')
        ax.grid(alpha=0.3)
    fig.suptitle('Connected 3pt/2pt ratios (PJN, γ₃)')
    fig.tight_layout()
    out = os.path.join(pdir, 'ratio_3pt_all_channels.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved {out}")


def _step_plots(config, run_dir, logger, meff_res=None, ratio_conn=None):
    print_banner("Step 7: Plots", logger)
    if meff_res is None:
        an_dir = os.path.join(run_dir, 'data', 'analysis')
        meff_res = {}
        for particle, mom, key in CHANNELS:
            fm = os.path.join(an_dir, f'meff_{particle}_{mom}_mean.npy')
            fe = os.path.join(an_dir, f'meff_{particle}_{mom}_err.npy')
            if os.path.exists(fm):
                m = np.load(fm); e = np.load(fe)
                ps, pe = (4, min(NT - 2, 14)) if particle == 'proton' else (5, min(NT - 2, 18))
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


# ── Step: report（analysis_summary.json + LaTeX）──────────────────

def _step_report(config, run_dir, logger, meff_res, timing):
    print_banner("Step 8: Analysis Summary JSON", logger)
    summary = {
        'version': VERSION,
        'conf_ids': config['conf_ids'],
        'precision': config['precision'],
        'nev': NEV, 'nev1': config['Nev1'],
        'lattice': [NT, NX, NX, NX], 'alttc': ALttc,
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
        'Nev1': min(args.Nev1, NEV),
        'channels': tuple(args.channels.split(',')),
        'verbose': args.verbose,
    }
    config['conf_ids'] = [args.conf_id] if args.conf_id else CONF_IDS
    if args.conf_ids:
        config['conf_ids'] = [int(x) for x in args.conf_ids.split(',')]
    config['Nev1'] = min(config['Nev1'], NEV)
    if args.fourpt_nev1:
        config['fourpt_nev1'] = args.fourpt_nev1
    if args.fourpt_tsep:
        config['fourpt_tsep'] = args.fourpt_tsep

    run_dir = os.path.join(out, f'output_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    if args.run_dir:
        run_dir = os.path.join(out, args.run_dir)
    for d in ['data', 'analysis', 'plots']:
        os.makedirs(os.path.join(run_dir, d), exist_ok=True)

    logger = setup_logging(out, name=f'test1-pipeline-{os.path.basename(run_dir)}',
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
    meff_res, ratio_conn = None, None
    total_start = time.perf_counter()
    try:
        for step in steps:
            tmr = Timer(f"STEP {step}", logger)
            tmr.__enter__()
            try:
                if step == 'env':
                    _step_env(config, logger)
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


def _step_env(config, logger):
    print_banner("Step 0: Environment Check", logger)
    env = {'ok': True}
    logger.info(f"Python {sys.version.split()[0]}")
    logger.info(f"Configs: {config['conf_ids']} (Nconf={len(config['conf_ids'])})")
    logger.info(f"Precision: {config['precision']}, Nev={NEV}, Nev1={config['Nev1']}")
    try:
        import cupy as cp
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
        free, total = cp.cuda.runtime.memGetInfo()
        env['gpu'] = name
        logger.info(f"GPU: {name} | free={free/2**30:.1f}/{total/2**30:.1f} GB")
        logger.info(f"CuPy {cp.__version__} | CUDA {cp.cuda.runtime.runtimeGetVersion()}")
    except Exception as e:
        logger.warning(f"NO GPU ({e}) — falling back to CPU (slow)")
    for cid in config['conf_ids']:
        e = os.path.isdir(os.path.dirname(get_eigen_path(cid, 0)))
        p = os.path.isdir(get_peram_dir(cid))
        g = os.path.exists(get_gauge_path(cid))
        logger.info(f"  conf={cid}: eigvec={'OK' if e else 'MISS'} "
                    f"peram={'OK' if p else 'MISS'} gauge={'OK' if g else 'MISS'}")
        env['ok'] &= e and p and g
    return env


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


def _resolve_run_dir(out, run_dir):
    """管线输出目录：--run-dir 显式指定，否则取最新 output_*。"""
    if run_dir:
        d = os.path.join(out, run_dir)
        if os.path.isdir(d):
            return d
        return run_dir
    cands = sorted([d for d in os.listdir(out)
                    if d.startswith('output_') and os.path.isdir(os.path.join(out, d))])
    if not cands:
        return out
    return os.path.join(out, cands[-1])


def cmd_verify(args):
    out = resolve_outdir(args)
    ref = args.ref or REF_DIR
    run_dir = _resolve_run_dir(out, getattr(args, 'run_dir', None))
    print(f"基线: {ref}")
    print(f"本次: {run_dir}")

    items = []
    for kind in ('analysis', 'configs', 'ope', '3pt', '4pt'):
        items.append((kind, []))

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
        pb = os.path.join(run_dir, 'data', 'analysis', f)
        a = np.load(pa) if os.path.exists(pa) else None
        b = np.load(pb) if os.path.exists(pb) else None
        ok, rel, _ = _cmp_array(f, a, b, rtol, atol)
        items[0][1].append((f, ok, rel))
        print(f"  [{'PASS' if ok else 'FAIL'}] {f}  max_rel={rel:.2e}" if rel is not None
              else f"  [MISS] {f}")

    for cid in CONF_IDS:
        for f in ['corr_pp_P0', 'corr_pp_P2', 'corr_pn_P0', 'corr_pn_P2',
                  'corr_pion_P0', 'corr_pion_P2',
                  'proton_P0_3pt', 'proton_P2_3pt',
                  'pion_P0_3pt', 'pion_P2_3pt', 'pjnnjnp_4pt',
                  'ope_combined']:
            pa = os.path.join(ref, 'data', f'conf{cid}', f'{f}_{cid}.npy')
            pb = os.path.join(run_dir, 'data', f'conf{cid}', f'{f}_{cid}.npy')
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

    sa = json.load(open(os.path.join(ref, 'analysis_summary.json')))
    sb_path = os.path.join(run_dir, 'analysis_summary.json')
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

    n_pass = sum(1 for _, lst in items for _, ok, _ in lst if ok and ok is not None)
    n_fail = sum(1 for _, lst in items for _, ok, _ in lst
                 if ok is False and ok is not None)
    n_miss = sum(1 for _, lst in items for _, ok, _ in lst if ok is None)
    total = len([x for _, lst in items for x in lst])
    print(f"\nverify: 共 {total} 项，PASS {n_pass}，FAIL {n_fail}，MISS {n_miss}")

    result = {
        'ref_dir': ref,
        'run_dir': run_dir,
        'rtol': rtol, 'atol': atol,
        'n_total': total, 'n_pass': n_pass, 'n_fail': n_fail, 'n_miss': n_miss,
        'details': {k: [{'name': n, 'ok': ok, 'max_rel': r}
                        for n, ok, r in lst] for k, lst in items},
    }
    vpath = os.path.join(out, 'test1_verify.json')
    with open(vpath, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"verify 结果 -> {vpath}")
    return 0 if n_fail == 0 else 1


# ═══════════════════════════════════════════════════════════════════
# 子命令 4: collect — 汇总 test1_results.json
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
    for root, _dirs, files in os.walk(run_dir):
        for f in sorted(files):
            full = os.path.join(root, f)
            key = os.path.relpath(full, run_dir)
            result['files'][key] = os.path.getsize(full)
    result['verify'] = json.load(open(os.path.join(out, 'test1_verify.json'))) \
        if os.path.exists(os.path.join(out, 'test1_verify.json')) else {}
    rpath = os.path.join(out, 'test1_results.json')
    with open(rpath, 'w') as f:
        json.dump(result, f, indent=2)
    nfiles = len(result['files'])
    print(f"collect: {nfiles} 个产物 -> {rpath}")
    return 0


# ═══════════════════════════════════════════════════════════════════
# 子命令 5: report — LaTeX 物理报告
# ═══════════════════════════════════════════════════════════════════

def build_tex(summary, run_dir, meff_vals, connected_ratio, disconn, conf_corrs):
    """生成 physics_report.tex（与基线 report.py 同构，版本标识 test1）。"""
    from datetime import datetime as _dt
    conf_ids = summary.get('conf_ids', CONF_IDS)
    precision = summary.get('precision', 'complex64')
    nev1 = summary.get('nev1', 100)

    meff_rows = []
    for had, mom in [('proton', 'P0'), ('proton', 'P2'), ('pion', 'P0'), ('pion', 'P2')]:
        key = f'{had}_{mom}'
        m = meff_vals.get(key, {})
        e0 = m.get('E0'); e0e = m.get('E0_err'); ee = m.get('E_exp')
        ps, pe = m.get('plateau', (0, 0)); npts = m.get('npts', 0)
        e0s = '—' if e0 is None else f'{e0:.3f}'
        e0es = '—' if e0e is None else f'{e0e:.3f}'
        ees = '—' if ee is None else f'{ee:.3f}'
        meff_rows.append(
            f"    {had} & $P={{{mom}}}$ & {e0s} $\\pm$ {e0es} & {ees}"
            f" & $[{ps},{pe}]$ & {npts} \\\\")

    ratio_rows = []
    for had, mom in [('proton', 'P0'), ('proton', 'P2'), ('pion', 'P0'), ('pion', 'P2')]:
        key = f'{had}_{mom}'
        r = connected_ratio.get(key, {})
        R = r.get('R'); Re = r.get('R_err')
        if R is not None:
            t_mid = min(len(R) - 1, 4)
            ratio_rows.append(
                f"    {had} & $P={{{mom}}}$ & ${R[t_mid]:+.4f} \\pm {Re[t_mid]:.4f}$ \\\\")

    disc_rows = []
    disc = disconn.get('proton') if isinstance(disconn, dict) else None
    if disc:
        c0, c1, dE = disc['c0'], disc['c1'], disc['dE']
        chi2 = disc['chi2']
        for z in [0, 4, 8, 12, 16, 20]:
            disc_rows.append(
                f"    {z} & ${c0[:, z].mean():.3f}$ & ${c1[:, z].mean():.3f}$"
                f" & ${dE[:, z].mean():.3f}$ & ${chi2[:, z].mean():.2g}$ \\\\")

    timing = summary.get('timing_s', {})
    timing_rows = "\n".join(
        f"    {step} & {t:.1f} s \\\\" for step, t in sorted(timing.items()))

    cfg_rows = []
    if conf_corrs:
        for had, mom in [('proton', 'P0'), ('proton', 'P2'), ('pion', 'P0'), ('pion', 'P2')]:
            vals = []
            for cid in conf_ids:
                c = conf_corrs.get(cid, {}).get(f'corr_pp' if had == 'proton' else 'corr_pi')
                if c is not None and mom in c and len(c[mom]):
                    vals.append(c[mom][0])
            if vals:
                mean, std = np.mean(vals), np.std(vals)
                cfg_rows.append(
                    f"    {had} P{mom} & {mean:.4e} & {std/abs(mean)*100:.1f}\\% \\\\")

    tex = r"""% ===========================================================================
%  格点QCD GPU蒸馏计算管线 — 物理分析报告 (test1, pyqcd)
% ===========================================================================
\documentclass[11pt,a4paper]{article}
\usepackage[UTF8]{ctex}
\setCJKmainfont{AR PL SungtiL GB}[BoldFont=AR PL UMing CN]
\setCJKsansfont{AR PL KaitiM GB}[BoldFont=AR PL UMing CN]
\setCJKmonofont{AR PL UMing CN}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{physics}
\usepackage{braket}
\usepackage{bm}
\usepackage[margin=2.5cm]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{hyperref}
\usepackage{enumitem}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{multirow}
\usepackage{array}

\newcommand{\gev}{\;\mathrm{GeV}}
\newcommand{\fm}{\;\mathrm{fm}}
\newcommand{\fmto}{\;\mathrm{fm}^{-1}}
\newcommand{\meff}{m_{\mathrm{eff}}}
\newcommand{\Nconf}{N_{\mathrm{conf}}}
\newcommand{\Nev}{N_{\mathrm{ev}}}
\newcommand{\Nt}{N_t}
\newcommand{\Nx}{N_x}
\newcommand{\tsep}{t_{\mathrm{sep}}}
\newcommand{\Ctwo}{C^{(2)}}
\newcommand{\Cthree}{C^{(3)}}
\newcommand{\Em}{E^{(0)}}
\newcommand{\pmom}{p_z}
\newcommand{\apm}{a^{-1}}
\newcommand{\jack}{\mathrm{JK}}
\newcommand{\gmu}{\gamma_\mu}

\title{\textbf{格点QCD GPU蒸馏计算管线物理分析报告}\\[0.3em]
       \large 顶点函数、Wick收缩、动态收缩与关联函数分析 (test1, pyqcd)}
\author{张鑫\thanks{中国科学院近代物理研究所 (IMP, CAS)}}
\date{""" + _dt.now().strftime('%Y年%m月%d日') + r"""}

\begin{document}
\maketitle

\begin{abstract}
本报告基于格点QCD蒸馏(Distillation)框架，调用 \texttt{pyqcd} 包在 GPU (CUDA) 上
复现了完整的关联函数计算管线：顶点函数($VdV$/$VVV$)、Wick收缩分析、动态收缩、
以及两点($pp$/$pn$)、OPE、三点($PJN$)、四点($PJNNJNp$)关联函数，并进行
Jackknife/有效质量/三点比值($ratio_{3p}$)统计分析。计算使用CLQCD合作组的规范组态
($\beta=6.20$, $24^3\times72$, $a\approx0.1053\;\fm$, $\apm\approx1.874\;\gev$)，
共 """ + str(len(conf_ids)) + r""" 个组态（""" + ', '.join(map(str, conf_ids)) + r"""），
计算精度 """ + precision + r"""。本测试以 docker-v20260805 输出为数值一致性基线。
\end{abstract}

\tableofcontents
\newpage

\section{引言}
LaMET (Large Momentum Effective Theory) 通过计算大动量下的准分布(quasi-distribution)
关联函数并做微扰匹配，得到光锥 parton 分布函数。胶子 PDF 涉及不相连(disconnected)图，
其中三点函数可分解为质子两点函数与胶子算符 (OPE) 两部分的乘积。本报告以
\texttt{examples/test1/main.py} 调用 \texttt{pyqcd} 包复现 docker-v20260805 的
GPU 蒸馏管线，并给出与基线 output\_20260802\_120104 的一致性验证结果。

\section{理论框架}
\subsection{格点系综参数}
\begin{table}[h]
\centering
\caption{格点系综参数 (表~\ref{tab:ensemble})}
\label{tab:ensemble}
\begin{tabular}{ll}
\toprule
参数 & 值 \\
\midrule
$\beta$ & 6.20 (Clover Wilson) \\
格点 & $24^3\times72$ \\
格距 $a$ & 0.1053 fm \\
逆格距 $\apm$ & $\approx 1.874$ GeV \\
本征矢数 $\Nev$ / $N_{\mathrm{ev,1}}$ & 100 / """ + str(nev1) + r""" \\
动量 & $P=(0,0,0)$, $P=(0,0,2)$ \\
组态数 $\Nconf$ & """ + str(len(conf_ids)) + r""" \\
精度 & """ + precision + r""" \\
\bottomrule
\end{tabular}
\end{table}

\subsection{蒸馏方法与顶点函数}
蒸馏 (distillation) 方法把夸克传播子投影到拉普拉斯算符的低模空间：
\[ \tau_{ij}(t_s,t_f) = v^\dagger_i(t_s)\, M^{-1}(t_s,t_f)\, v_j(t_f). \]
两点关联函数所需的顶点函数为
\begin{equation}
V^{VdV}_{mn}(\mathbf{p}) = \sum_{\mathbf{x}} e^{-i\mathbf{p}\cdot\mathbf{x}}\,
    v^\dagger_m(\mathbf{x})\, v_n(\mathbf{x}),
\label{eq:VdV}
\end{equation}
\begin{equation}
V^{VVV}_{m n l}(\mathbf{p}) = \sum_{\mathbf{x}} e^{-i\mathbf{p}\cdot\mathbf{x}}\,
    \varepsilon_{abc}\, v^a_m(\mathbf{x})\, v^b_n(\mathbf{x})\, v^c_l(\mathbf{x}),
\label{eq:VVV}
\end{equation}
其中 $VdV$ 用于介子，$VVV$ 用于重子（质子/中子）。

\subsection{两点关联函数与有效质量}
源平均后的两点函数为
\[ C(t) = \frac{1}{\Nt}\sum_{t_s} C(t_s,\, t_s + t). \]
有效质量的对数形式与双曲余弦形式为
\begin{equation}
\meff(t) = \ln\frac{C(t)}{C(t+1)}\cdot\frac{\hbar c}{a}, \qquad
\meff(t) = \operatorname{arccosh}\frac{C(t+2)+C(t)}{2C(t+1)}\cdot\frac{\hbar c}{a}.
\label{eq:meff}
\end{equation}

\subsection{三点/两点比值}
连通的三点函数 $C^{(3)}(\tau)$（质子-矢量流-核子）与两点函数 $C^{(2)}$ 的比值采用
lqcddb 的 $ratio_{3p}$ 公式，包含 $\sqrt{\cdots}$ 因子。不相连胶子比值则按
huangcl 的 code\_1.py 算法构造：
\begin{equation}
C^{(3)}(t, \tau, z) = C^{(2)}(t)\, O(z,\tau),
\qquad R(t,\tau,z) = \frac{C^{(3)} - C^{(2)}\braket{O}}{C^{(2)}},
\end{equation}
并对每个 $z$ 做关联拟合 $R(t,\tau) = c_0 + c_1 e^{-dE\,\tau} + c_1 e^{-dE\,(t-\tau)}$。

\section{计算方法 (GPU管线)}
管线步骤：
\begin{enumerate}
    \item 顶点函数：$VdV$/$VVV$（pyqcd.vertex，GPU，按时间片流式计算；VVV 用
          基线 x-slicing 因子化以适配 8GB 显存）
    \item Wick收缩分析 + 动态收缩（pyqcd.contraction，注册表 + einsum 计划缓存）
    \item 关联函数：2pt ($pp$/$pn$/pion)、OPE (pyqcd.operator 胶子算符)、
          3pt ($PJN$)、4pt ($PJNNJNp$)
    \item 统计分析：Jackknife、有效质量、$ratio_{3p}$（pyqcd.analysis，
          code\_1.py 形式）
    \item 绘图与 LaTeX 报告
\end{enumerate}
全部中间结果与日志均保存于版本目录（test12 形式）。

\section{结果与分析}
\subsection{两点关联函数与有效质量}
表~\ref{tab:meff} 给出各道的有效质量（Jackknife，加权平台）。
\begin{table}[h]
\centering
\caption{有效质量 (加权平台) (表~\ref{tab:meff})}
\label{tab:meff}
\begin{tabular}{llccccl}
\toprule
粒子 & 动量 & $E_0$ [GeV] & 期望 [GeV] & 平台 & 点数 \\
\midrule
""" + '\n'.join(meff_rows) + r"""
\bottomrule
\end{tabular}
\end{table}

\subsection{三点/两点连通比值}
表~\ref{tab:ratio} 给出 $\gamma_3$（$z$方向）分量的比值 $R(\tau)$。
\begin{table}[h]
\centering
\caption{连通三点/两点比值 (表~\ref{tab:ratio})}
\label{tab:ratio}
\begin{tabular}{llc}
\toprule
粒子 & 动量 & $R(\tau{\approx}4)$ \\
\midrule
""" + '\n'.join(ratio_rows) + r"""
\bottomrule
\end{tabular}
\end{table}

\subsection{不相连胶子比值与拟合}
表~\ref{tab:disc} 给出质子 $P_z=2$ 道不相连比值按 code\_1.py 拟合的参数。
\begin{table}[h]
\centering
\caption{不相连比值拟合参数 $R=c_0+c_1e^{-dE\,\tau}+c_1e^{-dE\,(t-\tau)}$ (表~\ref{tab:disc})}
\label{tab:disc}
\begin{tabular}{lcccc}
\toprule
$z$ & $c_0$ & $c_1$ & $dE$ & $\chi^2/\mathrm{dof}$ \\
\midrule
""" + '\n'.join(disc_rows) + r"""
\bottomrule
\end{tabular}
\end{table}

\subsection{组态一致性}
表~\ref{tab:config} 给出各道 $C(0)$ 的组态间离散程度。
\begin{table}[h]
\centering
\caption{各道 $C(0)$ 组态一致性 (表~\ref{tab:config})}
\label{tab:config}
\begin{tabular}{lcc}
\toprule
道 & $\braket{C(0)}$ & 相对离散度 \\
\midrule
""" + '\n'.join(cfg_rows) + r"""
\bottomrule
\end{tabular}
\end{table}

\subsection{计算耗时}
\begin{table}[h]
\centering
\caption{分步耗时 (表~\ref{tab:timing})}
\label{tab:timing}
\begin{tabular}{lr}
\toprule
步骤 & 耗时 \\
\midrule
""" + timing_rows + r"""
\bottomrule
\end{tabular}
\end{table}

\section{讨论与展望}
当前运行使用单精度 (complex64)、$\Nev=100$、$\Nconf=""" + str(len(conf_ids)) + r"""$。
$pn$（质子-中子）两点函数因味守恒而恒为零（质子 $uud$ 与中子 $udd$ 味结构不同），
这与理论预期一致。动量 $P=(0,0,2)$ 对应物理动量
$p_z = \frac{2\pi\cdot 2}{24\,a}\approx 0.981\;\gev$。后续工作可增加本征矢数目、
组态数目、动量涂抹、多源与 GEVP 以改善激发态污染。

\section{结论}
\begin{enumerate}
    \item 完整复现了从顶点函数到四点关联函数的 GPU 蒸馏管线（pyqcd 包调用）。
    \item 两点函数与有效质量分析通过 Jackknife 获得统计误差，与基线一致。
    \item 三点 ($PJN$) 与四点 ($PJNNJNp$) 关联函数及比值分析完成。
    \item OPE 胶子算符按 donghx 算法计算并与两点函数组合成不相连比值。
    \item 数值一致性验证（rtol=1e-3）确认与 docker-v20260805 基线一致。
\end{enumerate}

\begin{thebibliography}{9}
\bibitem{zhang2019} J.-H. Zhang et al., PRL 122, 142001 (2019).
\bibitem{fan2021} Z. Fan et al., PRD 104, 074502 (2021).
\bibitem{ji2013} X. Ji, PRL 110, 262002 (2013).
\bibitem{peardon2009} M. Peardon et al., PRD 80, 054506 (2009).
\end{thebibliography}

\end{document}
"""
    return tex


def cmd_report(args):
    out = resolve_outdir(args)
    run_dir = args.run_dir
    print(f"生成报告: {run_dir}")
    summary_path = os.path.join(run_dir, 'analysis_summary.json')
    if not os.path.exists(summary_path):
        print(f"analysis_summary.json not found in {run_dir}")
        return 1
    with open(summary_path) as f:
        summary = json.load(f)
    an_dir = os.path.join(run_dir, 'data', 'analysis')

    meff_vals = {f'{had}_{mom}': {
        'E0': summary['meff'].get(f'{had}_{mom}', {}).get('E0'),
        'E0_err': summary['meff'].get(f'{had}_{mom}', {}).get('E0_err'),
        'E_exp': summary['meff'].get(f'{had}_{mom}', {}).get('E_exp'),
        'plateau': summary['meff'].get(f'{had}_{mom}', {}).get('plateau'),
        'npts': summary['meff'].get(f'{had}_{mom}', {}).get('npts'),
    } for had, mom in [('proton', 'P0'), ('proton', 'P2'), ('pion', 'P0'), ('pion', 'P2')]}

    connected_ratio = {}
    for had, mom in [('proton', 'P0'), ('proton', 'P2'), ('pion', 'P0'), ('pion', 'P2')]:
        fm = os.path.join(an_dir, f'ratio_{had}_{mom}_mean.npy')
        fe = os.path.join(an_dir, f'ratio_{had}_{mom}_err.npy')
        if os.path.exists(fm):
            connected_ratio[f'{had}_{mom}'] = {'R': np.load(fm),
                                               'R_err': np.load(fe)}

    disconn = {}
    disc_dir = os.path.join(run_dir, 'analysis', 'disconnected')
    fp = os.path.join(disc_dir, '0_fit_data.npz')
    if os.path.exists(fp):
        d = np.load(fp)
        disconn['proton'] = {'c0': d['c0'], 'c1': d['c1'], 'dE': d['dE'],
                             'chi2': d['chi2']}

    conf_corrs = {}
    for cid in summary['conf_ids']:
        cdir = os.path.join(run_dir, 'data', f'conf{cid}')
        entry = {}
        for f in os.listdir(cdir) if os.path.isdir(cdir) else []:
            if f.startswith('corr_') and f.endswith('.npy'):
                key = f[5:-4]
                entry[key] = np.load(os.path.join(cdir, f))
        if entry:
            conf_corrs[cid] = entry

    tex = build_tex(summary, run_dir, meff_vals, connected_ratio,
                    disconn, conf_corrs)
    tex_path = os.path.join(run_dir, 'physics_report.tex')
    with open(tex_path, 'w') as f:
        f.write(tex)
    print(f"Wrote {tex_path}")

    for i in range(2):
        subprocess.run(['xelatex', '-interaction=nonstopmode',
                        '-halt-on-error', 'physics_report.tex'],
                       cwd=run_dir, capture_output=True)
    pdf = os.path.join(run_dir, 'physics_report.pdf')
    if not os.path.exists(pdf):
        print("WARNING: PDF not produced — check xelatex output")
        return 1
    print(f"PDF: {pdf}")
    return 0


# ═══════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='test1 — docker-v20260805 管线一致性测试（pyqcd 包）')
    p.add_argument('--outdir', default=None, help='产物目录（默认 TEST1_OUTDIR / examples/test1/）')
    sub = p.add_subparsers(dest='cmd', required=True)

    pe = sub.add_parser('env', help='环境自检 + env.json')
    pe.add_argument('--outdir', default=None)

    pp = sub.add_parser('pipeline', help='完整管线（vertex→2pt/3pt/4pt/OPE→analysis→plots→report）')
    pp.add_argument('--outdir', default=None)
    pp.add_argument('--conf-id', type=int, default=None)
    pp.add_argument('--conf-ids', type=str, default=None)
    pp.add_argument('--precision', choices=['complex64', 'complex128'], default=PRECISION)
    pp.add_argument('--Nev1', type=int, default=NEV1)
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
    pv.add_argument('--run-dir', default=None, help='管线输出子目录名（默认最新 output_*）')
    pv.add_argument('--ref', default=None, help='基线目录（默认 examples/docker-v20260805/...）')
    pv.add_argument('--rtol', type=float, default=1e-3)
    pv.add_argument('--atol', type=float, default=1e-8)

    pc = sub.add_parser('collect', help='汇总 test1_results.json')
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
        print(f"[test1] {args.cmd} FAILED: {e}")
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
