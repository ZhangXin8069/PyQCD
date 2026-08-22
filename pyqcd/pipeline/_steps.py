"""
9 步管线计算编排（pyqcd/pipeline/_steps）
==========================================

照抄 examples/docker-v20260805 的 compute_vertex/compute_contraction/
compute_ope/analyze/report/run_pipeline 编排逻辑（成功实例基线），
但所有计算调用 pyqcd 子包（lattice/tools/vertex/contraction/operator/
analysis），自包含、不 import examples/。

步骤与输出与基线完全一致：

    data/conf{id}/VdV_mom_{id}.npy, VVV_mom_{id}.npy
    data/conf{id}/corr_{ch}_{P0|P2}_{id}.npy
    data/conf{id}/ops_mu{mu}_nu{nu}_dz{dz}_conf{id}.npz, ope_combined_conf{id}.npy
    data/conf{id}/{proton|pion}_{P0|P2}_3pt_{id}.npy, pjnnjnp_4pt_{id}.npy
    data/analysis/{meff|corr}_{had}_{mom}_{mean|err}.npy
    data/analysis/ratio_{had}_{mom}_{mean|err}.npy
    analysis/disconnected/{ratio,0_fit_data,1_fit_report,c0/chi2/ratio png}
    plots/{meff_all_channels,correlators_all_channels,ratio_3pt_all_channels}.png
    physics_report.tex/pdf（xelatex 两遍）
    analysis_summary.json, run_config.json
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np

from ..tools import (
    set_backend, get_backend, get_backend_name, set_precision,
)
from ..tools._io import (
    readin_eigvecs_gpu, readin_peram_time_slice,
    save_tensor_h5, load_tensor_h5,
)
from ..lattice import gamma
from ..vertex import phase_exp_2pt, phase_exp_3pt, Mom_VdV_sink_t
from ..contraction import (
    PeramRegistry, VRegistry, GammaRegistry, dynamic_contraction,
    clear_plan_cache, seq_peram,
)
from ..operator import (
    read_gauge_lime, gluon_ope_operator_z0, plaquette_clover,
)
from ._config import (
    NT, NX, NEV, NEV1, ALttc, FM2GEV, CONF_IDS, PRECISION,
    MOM_SINK_VDV, MOM_SINK_VVV, ANALYSIS_MOMENTA,
    PP_SINK, PP_SRC, PN_SINK, PN_SRC, PION_SINK, PION_SRC,
    PJN_SINK, PJN_SRC, PJN_CURR, PION3_SINK, PION3_SRC, PION3_CURR,
    PJNNJNP_SINK, PJNNJNP_SRC, PJNNJNP_CURR,
    FOURPT_NEV1, FOURPT_TSEP, FOURPT_MOM, FOURPT_SRC_STEP,
    T_SEP, T_SEP_3PT, DELTA_Z, Z_DIR, OPE_COMPONENTS,
    get_eigen_path, get_peram_dir, get_gauge_path,
)

try:  # GPU 内存探测（仅 try/except，遵守 pyqcd 反模式约定）
    import cupy as _cp
    HAS_CUPY = True
except ImportError:
    _cp = None
    HAS_CUPY = False


# ═══════════════════════════════════════════════════════════════════
# 通用工具（照抄 docker-v20260805/utils.py 的用时部分）
# ═══════════════════════════════════════════════════════════════════

def _info(logger, msg):
    """兼容 print 与 logging.Logger。"""
    if logger is None:
        return
    if hasattr(logger, 'info'):
        logger.info(msg)
    else:
        logger(msg)


def _warn(logger, msg):
    if logger is None:
        return
    if hasattr(logger, 'warning'):
        logger.warning(msg)
    else:
        logger(f"[warn] {msg}")


def _timer(name, logger, fn, *args, **kw):
    """带计时地执行 fn(*args, **kw)，返回 (结果, 秒数)。"""
    if get_backend_name() == 'cupy':
        _cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    try:
        res = fn(*args, **kw)
    finally:
        if get_backend_name() == 'cupy':
            _cp.cuda.Stream.null.synchronize()
    el = time.perf_counter() - t0
    _info(logger, f"{name}: {el:.3f} s")
    return res, el


def free_gpu_memory():
    """释放 GPU 内存（cupy 内存池 / torch 缓存，按当前后端）。"""
    if get_backend_name() == 'torch':
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    if get_backend_name() == 'cupy' and HAS_CUPY:
        try:
            _cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass


def log_gpu_memory(logger, label: str = ''):
    if get_backend_name() == 'torch':
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                _info(logger, f"GPU memory{label}: used={(total - free)/2**20:.0f} MB, "
                              f"free={free/2**20:.0f} MB, total={total/2**20:.0f} MB")
                return
        except Exception:
            pass
    if get_backend_name() == 'cupy' and HAS_CUPY:
        try:
            total, used = _cp.cuda.runtime.memGetInfo()
            _info(logger, f"GPU memory{label}: used={(total - used)/2**20:.0f} MB, "
                          f"free={used/2**20:.0f} MB, total={total/2**20:.0f} MB")
        except Exception:
            pass
    else:
        _info(logger, f"GPU memory{label}: N/A")


def save_array(filepath, arr, logger=None):
    """保存数组（GPU → CPU 转换后 .h5；h5py 为唯一读写工具）。

    兼容旧调用：传入 .npy 路径时自动改存 .h5；旧 .npy 产物的
    读取由 ``_load_any`` 回退支持。
    """
    if filepath.endswith('.npy'):
        filepath = filepath[:-4] + '.h5'
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    save_tensor_h5(arr, filepath)
    arr_np = arr.get() if hasattr(arr, 'get') else np.asarray(arr)
    _info(logger, f"Saved {os.path.basename(filepath)} "
                  f"shape={np.shape(arr)} dtype={getattr(arr_np, 'dtype', '?')} "
                  f"({os.path.getsize(filepath)/1024:.1f} KB)")


def _load_any(path_without_ext, dataset='data'):
    """读取数组：优先 .h5（新格式），回退 .npy/.npz（旧产物兼容）。"""
    h5p = path_without_ext + '.h5'
    if os.path.exists(h5p):
        return load_tensor_h5(h5p, dataset=dataset)
    for ext, ds in (('.npy', None), ('.npz', 'ops')):
        p = path_without_ext + ext
        if os.path.exists(p):
            if ext == '.npy':
                return np.load(p)
            return np.load(p)[ds]
    raise FileNotFoundError(f"no data file for {path_without_ext}")


def conf_data_dir(run_dir, conf_id):
    d = os.path.join(run_dir, 'data', f'conf{conf_id}')
    os.makedirs(d, exist_ok=True)
    return d


def dump_config_snapshot(config: dict, filepath: str, logger=None):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    _info(logger, f"Config snapshot -> {filepath}")


# ═══════════════════════════════════════════════════════════════════
# Step 1 — 顶点函数 VdV / VVV（照抄 compute_vertex.py）
# ═══════════════════════════════════════════════════════════════════

def _compute_vvv_single_t_gpu(ev_t_gpu, ph_gpu, Nx, Nev1):
    """单时间片 VVV —— x-slice 分解（比单 einsum 快 ~20×）。"""
    backend = get_backend()
    VVV_t = backend.zeros((Nev1, Nev1, Nev1), dtype=ev_t_gpu.dtype)
    L = Nx * Nx
    for xi in range(Nx):
        s, e = xi * L, (xi + 1) * L
        es = ev_t_gpu[:Nev1, s:e, :]
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


def compute_vertices_for_config(conf_id, run_dir, logger,
                                precision='complex64', recompute=False,
                                mom_sink_vdv=None, mom_sink_vvv=None):
    """一个组态的 VdV/VVV（缓存命中则直接读取）。

    动量列表可自定义（默认用全局 MOM_SINK_VDV/MOM_SINK_VVV），
    供 test9 等多动量物理链复用；缓存路径带动量指纹避免串数据。
    """
    backend = get_backend()
    if get_backend_name() == 'torch':
        set_precision(precision)
    dtype = np.complex64 if precision == 'complex64' else np.complex128

    mom_vdv = list(mom_sink_vdv) if mom_sink_vdv is not None else list(MOM_SINK_VDV)
    mom_vvv = list(mom_sink_vvv) if mom_sink_vvv is not None else list(MOM_SINK_VVV)

    cdir = conf_data_dir(run_dir, conf_id)
    mom_fp = f"mom{''.join(str(m[0])+str(m[1])+str(m[2]) for m in mom_vdv)}" \
             if mom_sink_vdv is not None else 'mom'
    vdv_path = os.path.join(cdir, f'VdV_{mom_fp}_{conf_id}.npy')
    vvv_path = os.path.join(cdir, f'VVV_{mom_fp}_{conf_id}.npy')

    if os.path.exists(vdv_path) and os.path.exists(vvv_path) and not recompute:
        VdV = np.load(vdv_path)
        VVV = np.load(vvv_path)
        _info(logger, f"  conf={conf_id}: loaded cached vertices "
                      f"VdV{VdV.shape} VVV{VVV.shape}")
        return {'VdV': VdV, 'VVV': VVV}

    _info(logger, f"  conf={conf_id}: computing vertices over {NT} time slices "
                  f"(VdV {len(mom_vdv)} mom, VVV {len(mom_vvv)} mom, "
                  f"Nev={NEV}, Nev1={NEV1}, dtype={dtype.__name__})")

    p2f = np.zeros((len(mom_vdv), NX * NX * NX * 3), dtype=np.complex128)
    for i, mom in enumerate(mom_vdv):
        _ph = phase_exp_2pt(NX, mom)
        _ph_np = _ph.get() if hasattr(_ph, 'get') else np.asarray(_ph)
        p2f[i] = _ph_np.reshape(-1)
    p2f_gpu = backend.asarray(p2f.astype(dtype))
    p3_list = []
    for mom in mom_vvv:
        _ph = phase_exp_3pt(NX, mom)
        p3_list.append(_ph.get() if hasattr(_ph, 'get') else np.asarray(_ph))

    VdV = np.zeros((NT, len(mom_vdv), NEV, NEV), dtype=dtype)
    VVV = np.zeros((NT, len(mom_vvv), NEV1, NEV1, NEV1), dtype=dtype)

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
            _info(logger, f"    t={t:3d}/{NT}  elapsed={time.perf_counter()-t0:.0f}s")

    free_gpu_memory()
    log_gpu_memory(logger, " after vertices")

    diag = np.abs(np.diag(VdV[0, 0])).real
    _info(logger, f"    VdV(P=0,t=0) diagonal: [{diag.min():.3f}, {diag.max():.3f}]  "
                  f"(≈1 ⇒ orthonormal)")
    _info(logger, f"    VVV(P=0,t=0) |v|: [{np.abs(VVV[0,0]).min():.3e}, "
                  f"{np.abs(VVV[0,0]).max():.3e}]")

    save_array(vdv_path, VdV, logger)
    save_array(vvv_path, VVV, logger)
    _info(logger, f"    Saved VdV{VdV.shape} VVV{VVV.shape} for conf={conf_id}")
    return {'VdV': VdV, 'VVV': VVV}


def step_vertex(config, run_dir, logger):
    set_backend(config.get('backend', 'cupy'),
                device=config.get('device'))
    for cid in config['conf_ids']:
        _timer(f"  Vertices conf={cid}", logger,
               compute_vertices_for_config, cid, run_dir, logger,
               config['precision'], False)
        free_gpu_memory()
    _info(logger, f"Vertices computed & saved for {len(config['conf_ids'])} configs")


# ═══════════════════════════════════════════════════════════════════
# Step 2/4/5 — 2pt / 3pt / 4pt 关联函数（照抄 compute_contraction.py）
# ═══════════════════════════════════════════════════════════════════

def _real_sum(val):
    v = val.get() if hasattr(val, 'get') else val
    return float(np.real(np.sum(np.asarray(v).ravel())))


def _load_peram_set(backend, peram_dir, conf_id, times, dtype, nev1=None,
                    cache=None, max_size=None):
    """读取 peram 时间片集合（可选外部 cache 增量复用：滑动窗口只补新 t）。

    cache 为 OrderedDict；超过 ``max_size`` 时淘汰最旧时间片，防止
    全 NT 时间片驻留 GPU 显存（72×2×88MB ≈ 12.7GB 不可接受）。
    """
    from collections import OrderedDict
    if nev1 is None:
        nev1 = NEV
    if cache is None:
        cache = OrderedDict()
    if max_size is None:
        max_size = 2 * (T_SEP + 1) + 1
    for t in times:
        if t in cache:
            cache.move_to_end(t)
            continue
        peram_cpu = readin_peram_time_slice(peram_dir, str(conf_id), t, NT, NEV)
        peram_t = backend.asarray(peram_cpu[:, :, :, :nev1, :nev1].astype(dtype))
        cache[t] = (peram_t, seq_peram(peram_t))
        while len(cache) > max_size:
            cache.popitem(last=False)
    return cache


def _run_2pt(backend, sink_op, src_op, peram_t, peram_seq_t,
             t_src, t_sink, v_src, v_sink, v_kind, gamma_name, gamma_val,
             projector, Vindex=('M', 'M')):
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
        # 味禁戒通道（pn）：无有效 Wick 图，恒为零（物理正确）。
        return 0.0


def compute_2pt_for_config(conf_id, run_dir, logger, vertices,
                           precision=PRECISION, channels=('pp', 'pn', 'pion')):
    backend = get_backend()
    if get_backend_name() == 'torch':
        set_precision(precision)
    dtype = np.complex64 if precision == 'complex64' else np.complex128
    cdir = conf_data_dir(run_dir, conf_id)

    VdV = vertices['VdV']
    VVV = vertices['VVV']
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

    _info(logger, f"  2pt channels: {list(op_cfg.keys())} at P=(0,0,0),(0,0,2)")
    t_start = time.perf_counter()

    for t_src in range(NT):
        peram_cpu = readin_peram_time_slice(peram_dir, str(conf_id), t_src,
                                            NT, NEV)
        peram_t = backend.asarray(peram_cpu.astype(dtype))
        peram_seq_t = seq_peram(peram_t)

        for t_sink in range(NT):
            dt = (t_sink - t_src + NT) % NT
            for ch, (sink_op, src_op, vkind, gval, gname) in op_cfg.items():
                for mi, mom_tag in enumerate(('P0', 'P2')):
                    if vkind == 'VVV':
                        v_src = backend.asarray(
                            VVV[t_src, mi:mi + 1].conj(), dtype=dtype)
                        v_sink = backend.asarray(
                            VVV[t_sink, mi:mi + 1], dtype=dtype)
                    else:
                        v_src = backend.asarray(
                            VdV[t_src, mi:mi + 1].conj(), dtype=dtype)
                        v_sink = backend.asarray(
                            VdV[t_sink, mi:mi + 1], dtype=dtype)
                    val = _run_2pt(backend, sink_op, src_op,
                                   peram_t, peram_seq_t,
                                   t_src, t_sink, v_src, v_sink, vkind,
                                   gname, gval, projector)
                    acc[f'corr_{ch}_{mom_tag}'][dt] += val / NT

        if t_src % 12 == 0 or t_src == NT - 1:
            _info(logger, f"    t_src={t_src:3d}/{NT} "
                          f"elapsed={time.perf_counter()-t_start:.0f}s "
                          + " ".join(
                              f"{ch}0={acc[f'corr_{ch}_P0'][0]:.4e}"
                              for ch in channels))
        del peram_t, peram_seq_t

    for key, arr in acc.items():
        save_array(os.path.join(cdir, f'{key}_{conf_id}.npy'), arr, logger)
    _info(logger, f"  2pt saved: " + ", ".join(
        f"{k}={v[0]:.3e}" for k, v in acc.items()))
    return acc


def compute_2pt_for_config_multi(conf_id, run_dir, logger, vertices,
                                 momenta=((0, 0, 0), (0, 0, 2), (0, 0, 4)),
                                 precision=PRECISION,
                                 channels=('pp', 'pn', 'pion'),
                                 v_kind='VVV'):
    """多动量 2pt（test9 胶子 TMD 物理链用）：按 (pz,py,px) 列表逐动量计算。

    momenta: 形如 [(pz,py,px), ...] 的动量列表（格点单位 2π/L）。
    输出键：corr_{ch}_P{pz}{py}{px}（如 corr_pp_P000 / P200 / P400）。
    顶点由 compute_vertices_for_config 的 mom_sink_vdv/vvv 提供对应索引。
    """
    backend = get_backend()
    if get_backend_name() == 'torch':
        set_precision(precision)
    dtype = np.complex64 if precision == 'complex64' else np.complex128
    cdir = conf_data_dir(run_dir, conf_id)

    VdV = vertices['VdV']
    VVV = vertices['VVV']
    peram_dir = get_peram_dir(conf_id)
    n_mom = len(momenta)
    tags = [f'P{m[0]}{m[1]}{m[2]}' for m in momenta]

    projector = backend.asarray((gamma(0) + gamma(4)) / 2.0, dtype=dtype)
    g7 = backend.asarray(gamma(7), dtype=dtype)
    g5 = backend.asarray(gamma(5), dtype=dtype)

    op_cfg = {
        'pp':   (PP_SINK, PP_SRC, g7, 'gamma_7'),
        'pn':   (PN_SINK, PN_SRC, g7, 'gamma_7'),
        'pion': (PION_SINK, PION_SRC, g5, 'gamma_5'),
    }
    op_cfg = {ch: op_cfg[ch] for ch in channels}

    acc = {f'corr_{ch}_{tag}': np.zeros(NT, dtype=np.float64)
           for ch in channels for tag in tags}
    _info(logger, f"  2pt multi: channels={list(op_cfg.keys())}, "
                  f"momenta={list(momenta)}, v_kind={v_kind}")
    t_start = time.perf_counter()

    for t_src in range(NT):
        peram_cpu = readin_peram_time_slice(peram_dir, str(conf_id), t_src,
                                            NT, NEV)
        peram_t = backend.asarray(peram_cpu.astype(dtype))
        peram_seq_t = seq_peram(peram_t)

        for t_sink in range(NT):
            dt = (t_sink - t_src + NT) % NT
            for ch, (sink_op, src_op, gval, gname) in op_cfg.items():
                for mi in range(n_mom):
                    tag = tags[mi]
                    if v_kind == 'VVV':
                        v_src = backend.asarray(
                            VVV[t_src, mi:mi + 1].conj(), dtype=dtype)
                        v_sink = backend.asarray(
                            VVV[t_sink, mi:mi + 1], dtype=dtype)
                    else:
                        v_src = backend.asarray(
                            VdV[t_src, mi:mi + 1].conj(), dtype=dtype)
                        v_sink = backend.asarray(
                            VdV[t_sink, mi:mi + 1], dtype=dtype)
                    val = _run_2pt(backend, sink_op, src_op,
                                   peram_t, peram_seq_t,
                                   t_src, t_sink, v_src, v_sink, v_kind,
                                   gname, gval, projector)
                    acc[f'corr_{ch}_{tag}'][dt] += val / NT

        if t_src % 12 == 0 or t_src == NT - 1:
            _info(logger, f"    t_src={t_src:3d}/{NT} "
                          f"elapsed={time.perf_counter()-t_start:.0f}s "
                          + " ".join(
                              f"{ch}{tags[0]}={acc[f'corr_{ch}_{tags[0]}'][0]:.4e}"
                              for ch in channels))
        del peram_t, peram_seq_t

    for key, arr in acc.items():
        save_array(os.path.join(cdir, f'{key}_{conf_id}.npy'), arr, logger)
    _info(logger, f"  2pt multi saved: " + ", ".join(
        f"{k}={v[0]:.3e}" for k, v in acc.items()))
    return acc


def _2pt_all_present(cdir, conf_id, channels):
    """组态 2pt 产物齐全性检查（.h5 优先，回退 .npy）——断点续跑判据。"""
    for ch in channels:
        for mom in ('P0', 'P2'):
            base = os.path.join(cdir, f'corr_{ch}_{mom}_{conf_id}')
            if not (os.path.exists(base + '.h5') or os.path.exists(base + '.npy')):
                return False
    return True


def step_2pt(config, run_dir, logger):
    set_backend(config.get('backend', 'cupy'),
                device=config.get('device'))
    channels = config.get('channels', ('pp', 'pn', 'pion'))
    recompute = config.get('recompute_2pt', False)
    n_hit = 0
    for cid in config['conf_ids']:
        _info(logger, f"\n─── 2pt: conf {cid} ───")
        # 断点续跑（整合 logs/test8）：该组态 corr_{ch}_{P0,P2} 全存在则跳过
        # （vertex/OPE 缓存由 pyqcd 内部处理，2pt 级此前缺失——服务器长跑
        #   中断后重跑可跳过已完成组态，节省数小时）
        if not recompute and _2pt_all_present(
                conf_data_dir(run_dir, cid), cid, channels):
            _info(logger, f"  conf={cid}: 2pt 缓存命中，跳过"
                          "（recompute_2pt=True 强制重算）")
            n_hit += 1
            continue
        verts = _load_vertices_one(run_dir, cid)
        _timer(f"  2pt conf={cid}", logger, compute_2pt_for_config,
               cid, run_dir, logger, verts, config['precision'],
               channels)
        del verts
        free_gpu_memory()
    if n_hit == len(config['conf_ids']) and n_hit > 0:
        _info(logger, f"2pt 全部缓存命中（{n_hit}/{n_hit}），无需重算")


def _run_3pt(backend, sink_op, src_op, curr_op, PR, VR, GR, Vindex, Gindex):
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
    backend = get_backend()
    if get_backend_name() == 'torch':
        set_precision(precision)
    dtype = np.complex64 if precision == 'complex64' else np.complex128
    cdir = conf_data_dir(run_dir, conf_id)
    Ntau = t_sep + 1

    VdV = vertices['VdV']; VVV = vertices['VVV']
    peram_dir = get_peram_dir(conf_id)

    projector = backend.asarray((gamma(0) + gamma(4)) / 2.0, dtype=dtype)
    gmu = backend.asarray([gamma(1), gamma(2), gamma(3), gamma(4)],
                          dtype=dtype)
    g7 = backend.asarray(gamma(7), dtype=dtype)
    g5 = backend.asarray(gamma(5), dtype=dtype)

    acc = {f'{had}_{mom}': np.zeros((Ntau, 4), dtype=np.float64)
           for had in ('proton', 'pion') for mom in ('P0', 'P2')}

    _info(logger, f"  3pt PJN: t_sep={t_sep}, Ntau={Ntau}, gamma_mu=4 components")
    t_start = time.perf_counter()

    peram_cache = OrderedDict()
    for t_src in range(NT):
        t_sink = (t_src + t_sep) % NT
        need_times = sorted(set([t_src, t_sink] + [(t_src + tau) % NT
                                                   for tau in range(Ntau)]))
        pc = _load_peram_set(backend, peram_dir, str(conf_id), need_times,
                             dtype, cache=peram_cache)
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
            _info(logger, f"    t_src={t_src:3d}/{NT} "
                          f"elapsed={time.perf_counter()-t_start:.0f}s "
                          f"protP0[0,3]={acc['proton_P0'][0,3]:.3e} "
                          f"piP0[0,3]={acc['pion_P0'][0,3]:.3e}")
        del p_src, p_snk, p_srcS, p_snkS

    for key, arr in acc.items():
        save_array(os.path.join(cdir, f'{key}_3pt_{conf_id}.npy'), arr, logger)
    _info(logger, f"  3pt saved: " + ", ".join(
        f"{k}={v[0,3]:.3e}" for k, v in acc.items()))
    return acc


def step_3pt(config, run_dir, logger):
    set_backend(config.get('backend', 'cupy'),
                device=config.get('device'))
    for cid in config['conf_ids']:
        _info(logger, f"\n─── 3pt PJN: conf {cid} ───")
        verts = _load_vertices_one(run_dir, cid)
        _timer(f"  3pt conf={cid}", logger, compute_3pt_for_config,
               cid, run_dir, logger, verts, config['precision'],
               config.get('t_sep', T_SEP))
        del verts
        free_gpu_memory()


def compute_4pt_for_config(conf_id, run_dir, logger, vertices,
                           precision=PRECISION, t_sep=FOURPT_TSEP,
                           nev1=FOURPT_NEV1, momenta=FOURPT_MOM,
                           src_step=FOURPT_SRC_STEP):
    backend = get_backend()
    if get_backend_name() == 'torch':
        set_precision(precision)
    dtype = np.complex64 if precision == 'complex64' else np.complex128
    cdir = conf_data_dir(run_dir, conf_id)
    Ntau = t_sep + 1
    N_mom = len(momenta)
    sources = list(range(0, NT, src_step))

    VdV = vertices['VdV']; VVV = vertices['VVV']
    peram_dir = get_peram_dir(conf_id)

    projector = backend.asarray((gamma(0) + gamma(4)) / 2.0, dtype=dtype)
    gmu = backend.asarray([gamma(1), gamma(2), gamma(3), gamma(4)], dtype=dtype)
    g7 = backend.asarray(gamma(7), dtype=dtype)
    g5 = backend.asarray(gamma(5), dtype=dtype)

    acc = np.zeros((Ntau, N_mom, 4), dtype=np.float64)
    nsrc = len(sources)

    _info(logger, f"  4pt PJNNJNp: n(sink) — J — [p̄ + π](src), t_sep={t_sep}, "
                  f"Nev1={nev1}, mom={momenta}, src_step={src_step} "
                  f"({nsrc}/{NT} sources)")
    t_start = time.perf_counter()
    peram_cache = OrderedDict()

    for t_src in sources:
        t_sink = (t_src + t_sep) % NT
        need_times = sorted(set([t_src, t_sink] + [(t_src + tau) % NT
                                                   for tau in range(Ntau)]))
        pc = _load_peram_set(backend, peram_dir, str(conf_id), need_times,
                             dtype, nev1=nev1, cache=peram_cache)
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
                    _warn(logger, f"  4pt contraction failed at "
                                  f"(t_src={t_src},tau={tau},mom={mi}): {e}")
                    continue
                v = r.get() if hasattr(r, 'get') else r
                vn = np.asarray(v).ravel()
                acc[tau, imi, :min(4, len(vn))] += np.real(vn[:min(4, len(vn))]) / nsrc

            del p_cur, p_curS

        if (t_src - sources[0]) % 12 == 0 or t_src == sources[-1]:
            _info(logger, f"    t_src={t_src:3d} "
                          f"elapsed={time.perf_counter()-t_start:.0f}s "
                          f"acc[0,0,3]={acc[0,0,3]:.3e}")
        del p_src, p_snk, p_srcS, p_snkS

    save_array(os.path.join(cdir, f'pjnnjnp_4pt_{conf_id}.npy'), acc, logger)
    _info(logger, f"  4pt PJNNJNp saved: shape={acc.shape}")
    return acc

def step_4pt(config, run_dir, logger):
    set_backend(config.get('backend', 'cupy'),
                device=config.get('device'))
    for cid in config['conf_ids']:
        _info(logger, f"\n─── 4pt PJNNJNp: conf {cid} ───")
        verts = _load_vertices_one(run_dir, cid)
        _timer(f"  4pt conf={cid}", logger, compute_4pt_for_config,
               cid, run_dir, logger, verts, config['precision'],
               config.get('fourpt_tsep', FOURPT_TSEP),
               config.get('fourpt_nev1', FOURPT_NEV1),
               config.get('fourpt_mom', FOURPT_MOM),
               config.get('fourpt_src_step', FOURPT_SRC_STEP))
        del verts
        free_gpu_memory()


# ═══════════════════════════════════════════════════════════════════
# Step 3 — OPE 胶子算符（照抄 compute_ope.py，调用 pyqcd.operator）
# ═══════════════════════════════════════════════════════════════════

def _validate_gauge(gauge, logger):
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
    _info(logger, f"  Gauge: unitarity_dev={np.max(devs):.2e}, "
                  f"plaq_trace_re={np.real(np.mean(plaq)):.6f}")


def compute_ope_for_config(conf_id, run_dir, logger, precision='complex64',
                           delta_z=DELTA_Z, z_dir=Z_DIR,
                           components=OPE_COMPONENTS, recompute=False):
    if not HAS_CUPY and get_backend_name() != 'torch':
        raise RuntimeError("OPE requires a GPU backend (torch/cupy)")

    if get_backend_name() == 'torch':
        set_precision(precision)
    dtype = np.complex64 if precision == 'complex64' else np.complex128
    cdir = conf_data_dir(run_dir, conf_id)
    paths = {c: os.path.join(cdir, f'ops_mu{c[0]}_nu{c[1]}_dz{delta_z}_conf{conf_id}')
             for c in components}

    if all(any(os.path.exists(p + e) for e in ('.h5', '.npz'))
           for p in paths.values()) and not recompute:
        _info(logger, f"  conf={conf_id}: loading cached OPE components")
        ops = {c: _load_any(p, dataset='data') for c, p in paths.items()}
        combined = -ops[(3, 0)] - ops[(3, 1)] + 2.0 * ops[(0, 1)]
        return {'components': ops, 'combined': combined}

    gauge_file = get_gauge_path(conf_id)
    _info(logger, f"  conf={conf_id}: OPE from {gauge_file} "
                  f"(dz={delta_z}, z_dir={z_dir}, {precision})")

    gauge_cpu, _t = _timer(f"  read gauge conf={conf_id}", logger,
                           read_gauge_lime, gauge_file, NT, NX)
    _validate_gauge(gauge_cpu, logger)
    backend = get_backend()
    gauge_gpu = backend.asarray(gauge_cpu.astype(dtype))
    del gauge_cpu

    ops = {}
    for mu, nu in components:
        o, _t2 = _timer(f"  OPE mu={mu},nu={nu} conf={conf_id}", logger,
                        gluon_ope_operator_z0, gauge_gpu, mu, nu, z_dir,
                        delta_z, NT, NX, dtype)
        ops[(mu, nu)] = o
        save_tensor_h5(o, paths[(mu, nu)])
        _info(logger, f"    saved ops_mu{mu}_nu{nu}: shape={o.shape}, "
                      f"|O|∈[{np.abs(o).min():.2e},{np.abs(o).max():.2e}]")

    combined = -ops[(3, 0)] - ops[(3, 1)] + 2.0 * ops[(0, 1)]
    save_array(os.path.join(cdir, f'ope_combined_conf{conf_id}.npy'),
               combined, logger)

    free_gpu_memory()
    log_gpu_memory(logger, " after OPE")
    return {'components': ops, 'combined': combined}


def step_ope(config, run_dir, logger):
    set_backend(config.get('backend', 'cupy'), device=config.get('device'))
    for cid in config['conf_ids']:
        _info(logger, f"\n─── OPE: conf {cid} ───")
        compute_ope_for_config(cid, run_dir, logger, config['precision'])


# ═══════════════════════════════════════════════════════════════════
# 分析层（照抄 run_pipeline.py step_analysis + analyze.py 保存逻辑）
# ═══════════════════════════════════════════════════════════════════

def _load_vertices_one(run_dir, cid):
    cdir = conf_data_dir(run_dir, cid)
    return {
        'VdV': _load_any(os.path.join(cdir, f'VdV_mom_{cid}')),
        'VVV': _load_any(os.path.join(cdir, f'VVV_mom_{cid}')),
    }


def load_2pt(run_dir, logger=None):
    corr = {}
    data_dir = os.path.join(run_dir, 'data')
    if not os.path.isdir(data_dir):
        return corr
    for name in os.listdir(data_dir):
        if not name.startswith('conf'):
            continue
        cid = name[4:]
        cdir = os.path.join(data_dir, name)
        entry = {}
        for f in os.listdir(cdir):
            if f.startswith('corr_') and (f.endswith('.h5') or f.endswith('.npy')):
                base = f[:-3] if f.endswith('.h5') else f[:-4]
                key = base[5:].replace(f'_{cid}', '')
                entry[f'corr_{key}'] = _load_any(os.path.join(cdir, base))
        if entry:
            corr[int(cid)] = entry
    _info(logger, f"Loaded 2pt correlators for {len(corr)} configs")
    return corr


def load_3pt(run_dir, logger=None):
    corr = {}
    data_dir = os.path.join(run_dir, 'data')
    if not os.path.isdir(data_dir):
        return corr
    for name in os.listdir(data_dir):
        if not name.startswith('conf'):
            continue
        cid = name[4:]
        cdir = os.path.join(data_dir, name)
        entry = {}
        for f in os.listdir(cdir):
            if '_3pt_' in f and (f.endswith('.h5') or f.endswith('.npy')) \
                    and 'pjnnjnp' not in f:
                base = f[:-3] if f.endswith('.h5') else f[:-4]
                key = base.replace(f'_{cid}', '')
                entry[key] = _load_any(os.path.join(cdir, base))
        if entry:
            corr[int(cid)] = entry
    if corr:
        _info(logger, f"Loaded 3pt correlators for {len(corr)} configs")
    return corr


def load_ope(run_dir, logger=None):
    ope = {}
    data_dir = os.path.join(run_dir, 'data')
    if not os.path.isdir(data_dir):
        return ope
    for name in os.listdir(data_dir):
        if not name.startswith('conf'):
            continue
        cid = name[4:]
        cdir = os.path.join(data_dir, name)
        comb = os.path.join(cdir, f'ope_combined_conf{cid}')
        if os.path.exists(comb + '.h5') or os.path.exists(comb + '.npy'):
            ope[int(cid)] = {'combined': _load_any(comb)}
    if ope:
        _info(logger, f"Loaded combined OPE for {len(ope)} configs")
    return ope


def step_analysis(config, run_dir, logger):
    an_dir = os.path.join(run_dir, 'data', 'analysis')
    os.makedirs(an_dir, exist_ok=True)

    from ..analysis import (
        run_meff_jackknife as _run_meff_jackknife,
        run_3pt_ratio as _run_3pt_ratio,
        run_disconnected_ratio as _run_disconnected_ratio,
    )

    corr2 = load_2pt(run_dir, logger)
    meff_res = _run_meff_jackknife(corr2, config['conf_ids'], NT=NT,
                                   ALttc=ALttc, logger=logger)
    for particle, mom, key in [
        ('proton', 'P0', 'corr_pp_P0'), ('proton', 'P2', 'corr_pp_P2'),
        ('pion', 'P0', 'corr_pion_P0'), ('pion', 'P2', 'corr_pion_P2'),
    ]:
        res = meff_res[f'{particle}_{mom}']
        for q, base in [('meff_mean', 'meff'), ('meff_err', 'meff'),
                        ('corr_mean', 'corr'), ('corr_err', 'corr')]:
            tag = 'mean' if q.endswith('mean') else 'err'
            save_array(os.path.join(an_dir, f'{base}_{particle}_{mom}_{tag}.npy'),
                       res[q], logger)

    ratio_conn = {}
    corr3 = load_3pt(run_dir, logger)
    if corr3:
        ratio_conn = _run_3pt_ratio(corr2, corr3, config['conf_ids'],
                                    logger=logger)
        for had, mom in [('proton', 'P0'), ('proton', 'P2'),
                         ('pion', 'P0'), ('pion', 'P2')]:
            res = ratio_conn.get(f'{had}_{mom}')
            if res is None:
                continue
            save_array(os.path.join(an_dir, f'ratio_{had}_{mom}_mean.npy'),
                       res['R'], logger)
            save_array(os.path.join(an_dir, f'ratio_{had}_{mom}_err.npy'),
                       res['R_err'], logger)
    else:
        _warn(logger, "No 3pt data — skipping connected ratio")

    ratio_disc = {}
    ope = load_ope(run_dir, logger)
    if ope:
        if len(config['conf_ids']) >= 2:
            ratio_disc = _run_disconnected_ratio(corr2, ope, config['conf_ids'],
                                                  run_dir, logger=logger, NT=NT,
                                                  NX=NX)
        else:
            _warn(logger, "Nconf<2 —— 不相连 ratio 拟合统计上无意义，跳过"
                          "（全量 ≥2 组态时启用，与基线一致）")
    else:
        _warn(logger, "No OPE data — skipping disconnected ratio")

    return {'meff': meff_res, 'connected_ratio': ratio_conn,
            'disconnected_ratio': ratio_disc}


# ═══════════════════════════════════════════════════════════════════
# 绘图（照抄 analyze.py plot_meff_results / plot_correlators /
#       plot_connected_ratio）
# ═══════════════════════════════════════════════════════════════════

def _plot_style_common(fig):
    pass


def plot_meff_results(meff_results, run_dir, logger):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pdir = os.path.join(run_dir, 'plots')
    os.makedirs(pdir, exist_ok=True)
    channels = [('proton', 'P0'), ('proton', 'P2'),
                ('pion', 'P0'), ('pion', 'P2')]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (particle, mom) in zip(axes.ravel(), channels):
        res = meff_results.get(f'{particle}_{mom}')
        if res is None:
            continue
        m, e = res['meff_mean'], res['meff_err']
        t = np.arange(len(m))
        ps, pe = res['plateau']
        ax.errorbar(t, m, yerr=e, fmt='o', ms=4, capsize=2)
        ax.axvspan(ps, pe - 1, alpha=0.15, color='C1')
        ax.axhline(res['E0'], color='C3', ls='--', lw=1)
        ax.axhline(res.get('E_exp', 0), color='C4', ls=':', lw=1)
        ax.set_title(f'{particle} P={mom}  E0={res["E0"]:.3f}±{res["E0_err"]:.3f} '
                     f'(exp {res.get("E_exp", 0):.2f})')
        ax.set_xlabel('t'); ax.set_ylabel(r'$m_{\rm eff}$ [GeV]')
        ax.grid(alpha=0.3)
    fig.suptitle('Effective masses (Jackknife, 10 configs)')
    fig.tight_layout()
    out = os.path.join(pdir, 'meff_all_channels.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    _info(logger, f"  Saved {out}")


def plot_correlators(meff_results, run_dir, logger):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pdir = os.path.join(run_dir, 'plots')
    os.makedirs(pdir, exist_ok=True)
    channels = [('proton', 'P0'), ('proton', 'P2'),
                ('pion', 'P0'), ('pion', 'P2')]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (particle, mom) in zip(axes.ravel(), channels):
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
    _info(logger, f"  Saved {out}")


def plot_connected_ratio(ratio_results, run_dir, logger):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pdir = os.path.join(run_dir, 'plots')
    os.makedirs(pdir, exist_ok=True)
    pairs = [('proton', 'P0'), ('proton', 'P2'), ('pion', 'P0'), ('pion', 'P2')]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
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
    _info(logger, f"  Saved {out}")


def step_plots(config, run_dir, logger, meff_res=None, ratio_conn=None):
    if meff_res is None:
        an_dir = os.path.join(run_dir, 'data', 'analysis')
        meff_res = {}
        for particle, mom in [('proton', 'P0'), ('proton', 'P2'),
                              ('pion', 'P0'), ('pion', 'P2')]:
            fm = os.path.join(an_dir, f'meff_{particle}_{mom}_mean.npy')
            fe = os.path.join(an_dir, f'meff_{particle}_{mom}_err.npy')
            if os.path.exists(fm):
                m = np.load(fm); e = np.load(fe)
                ps, pe = (4, min(NT - 2, 14)) if particle == 'proton' \
                    else (5, min(NT - 2, 18))
                mask = np.isfinite(m[ps:pe]) & (e[ps:pe] > 0) & (m[ps:pe] > 0.01)
                w = 1.0 / (e[ps:pe][mask] ** 2 + 1e-10)
                meff_res[f'{particle}_{mom}'] = {
                    'meff_mean': m, 'meff_err': e, 'plateau': (ps, pe),
                    'E0': float(np.sum(m[ps:pe][mask] * w) / np.sum(w)),
                    'E0_err': float(1 / np.sqrt(np.sum(w))),
                    'E_exp': 1.0 if particle == 'proton' else 0.30,
                    'corr_mean': np.load(os.path.join(
                        an_dir, f'corr_{particle}_{mom}_mean.npy')),
                    'corr_err': np.load(os.path.join(
                        an_dir, f'corr_{particle}_{mom}_err.npy')),
                }
    plot_meff_results(meff_res, run_dir, logger)
    plot_correlators(meff_res, run_dir, logger)
    if ratio_conn:
        plot_connected_ratio(ratio_conn, run_dir, logger)
    return meff_res


# ═══════════════════════════════════════════════════════════════════
# Step 8 — LaTeX 报告（照抄 report.py build_tex + step_report）
# ═══════════════════════════════════════════════════════════════════

_CHANNELS = [('proton', 'P0'), ('proton', 'P2'), ('pion', 'P0'), ('pion', 'P2')]


def _fmt(v, e):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    if e is None or (isinstance(e, float) and np.isnan(e)):
        return f'{v:.3f}'
    return f'{v:.3f} ± {e:.3f}'


def build_tex(summary, run_dir, meff_vals, connected_ratio, disconn,
              conf_corrs):
    conf_ids = summary.get('conf_ids', CONF_IDS)
    precision = summary.get('precision', 'complex64')
    nev1 = summary.get('nev1', 100)

    meff_rows = []
    for had, mom in _CHANNELS:
        m = meff_vals.get(f'{had}_{mom}', {})
        e0 = m.get('E0'); e0e = m.get('E0_err'); ee = m.get('E_exp')
        ps, pe = m.get('plateau', (0, 0)); npts = m.get('npts', 0)
        meff_rows.append(
            f"    {had} & $P={{{mom}}}$ & {_fmt(e0, e0e)} & {_fmt(ee, None)}"
            f" & $[{ps},{pe}]$ & {npts} \\\\")

    ratio_rows = []
    for had, mom in _CHANNELS:
        r = connected_ratio.get(f'{had}_{mom}', {})
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
            if z >= c0.shape[1]:
                continue

            def s(arr):
                return f"{arr[:, z].mean():.3f}"
            disc_rows.append(
                f"    {z} & ${s(c0)}$ & ${s(c1)}$ & ${s(dE)}$ & ${chi2[:, z].mean():.2g}$ \\\\")

    timing = summary.get('timing_s', {})
    timing_rows = "\n".join(
        f"    {step} & {t:.1f} s \\\\" for step, t in sorted(timing.items()))

    cfg_rows = []
    if conf_corrs:
        for had, mom in _CHANNELS:
            vals = []
            for cid in conf_ids:
                c = conf_corrs.get(cid, {}).get(
                    'corr_pp' if had == 'proton' else 'corr_pion')
                if c is not None and mom in c and len(c[mom]):
                    vals.append(c[mom][0])
            if vals:
                mean, std = np.mean(vals), np.std(vals)
                cfg_rows.append(
                    f"    {had} P{mom} & {mean:.4e} & {std/abs(mean)*100:.1f}\\% \\\\")

    tex = r"""% ===========================================================================
%  格点QCD GPU蒸馏计算管线 — 物理分析报告 (test0 / pyqcd)
%  Physical Analysis Report — test0 (pyqcd 调用)
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
       \large 顶点函数、Wick收缩、动态收缩与关联函数分析 (test0 — pyqcd)}
\author{张鑫\thanks{中国科学院近代物理研究所 (IMP, CAS)}}
\date{""" + datetime.now().strftime('%Y年%m月%d日') + r"""}

\begin{document}
\maketitle

\begin{abstract}
本报告基于格点QCD蒸馏(Distillation)框架，在GPU (CUDA) 上实现了完整的关联函数计算管线：
顶点函数($VdV$/$VVV$)、Wick收缩分析、动态收缩、以及两点($pp$/$pn$)、OPE、三点($PJN$)、
四点($PJNNJNp$)关联函数，并进行Jackknife/有效质量/三点比值($ratio_{3p}$)统计分析。
本运行由 examples/test0 调用 pyqcd 包完成（与成功实例 docker-v20260805 逐项一致）。
计算使用CLQCD合作组的规范组态
($\beta=6.20$, $24^3\times72$, $a\approx0.1053\;\fm$, $\apm\approx1.874\;\gev$)，
共 """ + str(len(conf_ids)) + r""" 个组态（""" + ', '.join(map(str, conf_ids)) + r"""），
计算精度 """ + precision + r"""。
\end{abstract}

\tableofcontents
\newpage

\section{引言}
LaMET (Large Momentum Effective Theory) 通过计算大动量下的准分布(quasi-distribution)
关联函数并做微扰匹配，得到光锥 parton 分布函数。胶子 PDF 涉及不相连(disconnected)图，
其中三点函数可分解为质子两点函数与胶子算符 (OPE) 两部分的乘积。本报告由 test0 套件
调用 pyqcd 包实现（逻辑照抄成功实例 docker-v20260805，自包含）。

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
    \item 顶点函数：$VdV$/$VVV$（GPU，按时间片流式计算）
    \item Wick收缩分析 + 动态收缩（lqcddb 引擎，注册表 + einsum 计划缓存）
    \item 关联函数：2pt ($pp$/$pn$/pion)、OPE (胶子算符)、3pt ($PJN$)、4pt ($PJNNJNp$)
    \item 统计分析：Jackknife、有效质量、$ratio_{3p}$（code\_1.py 形式）
    \item 绘图与 LaTeX 报告
\end{enumerate}
全部中间结果与日志均保存在版本目录中。

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
    \item 完整实现了从顶点函数到四点关联函数的 GPU 蒸馏管线（pyqcd 调用）。
    \item 两点函数与有效质量分析通过 Jackknife 获得统计误差。
    \item 三点 ($PJN$) 与四点 ($PJNNJNp$) 关联函数及比值分析完成。
    \item OPE 胶子算符按 donghx 算法计算并与两点函数组合成不相连比值。
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


def step_report(config, run_dir, logger, meff_res, timing, env=None):
    # 汇总 JSON（照抄 run_pipeline.py step_report）
    summary = {
        'version': 'test0',
        'conf_ids': config['conf_ids'],
        'precision': config['precision'],
        'nev': NEV, 'nev1': config.get('Nev1', NEV1),
        'lattice': [NT, NX, NX, NX], 'alttc': ALttc,
        'meff': {k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating))
                          else (list(vv) if isinstance(vv, tuple) else None))
                     for kk, vv in v.items()
                     if kk in ('E0', 'E0_err', 'E_exp', 'dev', 'plateau', 'npts')}
                 for k, v in meff_res.items()},
        'timing_s': timing,
        'run_dir': run_dir,
    }
    if env:
        summary['env'] = env
    with open(os.path.join(run_dir, 'analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    _info(logger, f"Analysis summary -> {run_dir}/analysis_summary.json")

    an_dir = os.path.join(run_dir, 'data', 'analysis')
    meff_vals = {f'{had}_{mom}': {
        'E0': summary['meff'].get(f'{had}_{mom}', {}).get('E0'),
        'E0_err': summary['meff'].get(f'{had}_{mom}', {}).get('E0_err'),
        'E_exp': summary['meff'].get(f'{had}_{mom}', {}).get('E_exp'),
        'plateau': summary['meff'].get(f'{had}_{mom}', {}).get('plateau'),
        'npts': summary['meff'].get(f'{had}_{mom}', {}).get('npts'),
    } for had, mom in _CHANNELS}

    connected_ratio = {}
    for had, mom in _CHANNELS:
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
    _info(logger, f"Wrote {tex_path}")

    for i in range(2):
        subprocess.run(['xelatex', '-interaction=nonstopmode',
                        '-halt-on-error', 'physics_report.tex'],
                       cwd=run_dir, capture_output=True)
    pdf = os.path.join(run_dir, 'physics_report.pdf')
    if not os.path.exists(pdf):
        _warn(logger, "WARNING: PDF not produced — check xelatex output")
    else:
        _info(logger, f"PDF: {pdf}")
    return summary


# ═══════════════════════════════════════════════════════════════════
# 调度（照抄 run_pipeline.py main 循环）
# ═══════════════════════════════════════════════════════════════════

def run_pipeline(steps=('env', 'vertex', '2pt', 'ope', '3pt', '4pt',
                        'analysis', 'plots', 'report'),
                 conf_ids=None, run_dir=None, logger=print,
                 precision=PRECISION, nev1=None, channels=('pp', 'pn', 'pion'),
                 fourpt_nev1=None, fourpt_tsep=None, fourpt_mom=None,
                 fourpt_src_step=None, t_sep=None, skip_missing=False,
                 backend='cupy', device=None):
    """9 步管线调度（pyqcd 自包含实现，与 docker-v20260805 输出一致）。

    backend: 'cupy'（默认，旧行为）或 'torch'（PyTorch，device='cuda' 走 GPU）。
    返回 dict: {'run_dir', 'timing', 'summary', 'meff', 'ratio_conn'}
    """
    conf_ids = list(conf_ids or CONF_IDS)
    config = {
        'precision': precision,
        'Nev1': min(nev1, NEV) if nev1 else NEV1,
        'channels': tuple(channels),
        'conf_ids': conf_ids,
        'backend': backend,
        'device': device,
    }
    if fourpt_nev1:
        config['fourpt_nev1'] = fourpt_nev1
    if fourpt_tsep:
        config['fourpt_tsep'] = fourpt_tsep
    if fourpt_mom:
        config['fourpt_mom'] = fourpt_mom
    if fourpt_src_step:
        config['fourpt_src_step'] = fourpt_src_step
    if t_sep:
        config['t_sep'] = t_sep

    if run_dir is None:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'output', f'output_{stamp}')
    for d in ['data', 'analysis', 'plots']:
        os.makedirs(os.path.join(run_dir, d), exist_ok=True)
    _info(logger, f"Run directory: {run_dir}")
    _info(logger, f"Config: {config}")
    dump_config_snapshot(config, os.path.join(run_dir, 'run_config.json'), logger)

    timing = {}
    meff_res, ratio_conn, env = None, None, None
    summary = None
    total_start = time.perf_counter()
    try:
        for step in steps:
            t0 = time.perf_counter()
            if step == 'env':
                env = {
                    'conf_ids': conf_ids, 'precision': config['precision'],
                    'nx': NX, 'nt': NT,
                    'gauge_dir': os.path.dirname(get_gauge_path(conf_ids[0])),
                }
                _info(logger, f"env: {env}")
            elif step == 'vertex':
                step_vertex(config, run_dir, logger)
            elif step == '2pt':
                step_2pt(config, run_dir, logger)
            elif step == 'ope':
                step_ope(config, run_dir, logger)
            elif step == '3pt':
                step_3pt(config, run_dir, logger)
            elif step == '4pt':
                step_4pt(config, run_dir, logger)
            elif step == 'analysis':
                analysis = step_analysis(config, run_dir, logger)
                meff_res = analysis['meff']
                ratio_conn = analysis['connected_ratio']
            elif step == 'plots':
                meff_res = step_plots(config, run_dir, logger,
                                      meff_res, ratio_conn)
            elif step == 'report':
                summary = step_report(config, run_dir, logger,
                                      meff_res, timing, env)
            elif step == 'tmd':
                from ..renorm._tmd import gradient_flow_renormalized_tmd
                from ..operator import read_gauge_lime as _rgl
                gauge = _rgl(get_gauge_path(conf_ids[0]), NT, NX)
                if gauge is not None:
                    tau = 3.0 * (ALttc * FM2GEV) ** 2
                    O = gradient_flow_renormalized_tmd(
                        gauge, tau, list(range(DELTA_Z)), list(range(0, 6)))
                    with open(os.path.join(run_dir, 'tmd_gluon_flow.json'),
                              'w') as f:
                        json.dump({'tau': tau,
                                   'O_shape': list(np.shape(O))}, f)
            elif skip_missing:
                _warn(logger, f"step '{step}' unknown — skipped")
            else:
                raise ValueError(f"unknown step '{step}'")
            timing[step] = round(time.perf_counter() - t0, 1)
            _info(logger, f"STEP {step} done in {timing[step]}s")
        total_t = time.perf_counter() - total_start
        _info(logger, f"Pipeline Complete! Total {total_t:.0f}s "
                      f"({total_t/60:.1f} min)")
    except Exception as e:
        import traceback
        _info(logger, f"PIPELINE FAILED: {e}")
        _info(logger, traceback.format_exc())
        raise
    return {'run_dir': run_dir, 'timing': timing, 'summary': summary,
            'meff': meff_res, 'ratio_conn': ratio_conn}
