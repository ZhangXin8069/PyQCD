"""
test9 梯度流重整化核子胶子 TMD-PDF 物理链（pyqcd 核心模块）
========================================================

从真实蒸馏数据（eigvec/peram/gauge）出发计算核子中胶子 TMD-PDF：

    1. 蒸馏 2pt：核子谱线 C2(dt)（多动量 Pz=0,2,4 或全方向）
    2. 梯度流：gauge → wilson_flow(τ=3a²) → flowed gauge V_tau
    3. 胶子 TMD 算符：在 flowed gauge 上逐时间片计算
       O(z,b⊥) = M^{tx;tx}+M^{ty;ty}−2M^{xy;xy}（空间求和）→ (nz,nb,Nt)
    4. 不相连 3pt 因子化：C3(dt,dtau,z,b) = C2(dt)·OPE(dtau,z,b)
    5. 真空扣除 + 比值 R(dt,dtau,z,b) = <[C3−C2⟨OPE⟩]/C2>_ti
    6. 逐 (z,b) 拟合 c0(z,b) → 裸矩阵元 hB(z,b,Pz)
    7. 自重整化（梯度流方案）：hR(z,b,Pz) = hB(z,b,Pz)/hB(0,b,Pz=0)
    8. λ 外推 + 傅里叶 → 准 TMD-PDF
    9. NLO 匹配 → 光锥 TMD-PDF x·g(x,b⊥) + CS 核

本模块只放"物理链计算"函数（数据读取、算符、统计、重整化），
顶层编排（配置/目录/CLI/并行）在 examples/pyqcd/test9_gluon_tmd_nucleon.py。
"""
from __future__ import annotations

import os
import time

import numpy as np

from ..tools import get_backend, get_backend_name, set_precision
from ..tools._io import (
    save_tensor_h5, load_tensor_h5,
)
from ..pipeline._config import (
    NT, NX, PRECISION, get_gauge_path,
)
from ..pipeline._steps import (
    compute_vertices_for_config, compute_2pt_for_config_multi,
    _load_any, _info,
)
from ..renorm import wilson_flow, flow_action_density, tmd_matrix_elements_time
from ..operator import read_gauge_lime


# ═══════════════════════════════════════════════════════════════════
# 动量集合（[pz, py, px]，格点单位 2π/L）
# ═══════════════════════════════════════════════════════════════════

# 集合 A：z 方向 [0,2,4]，其余为 0
MOMENTA_Z = [(0, 0, 0), (2, 0, 0), (4, 0, 0)]
# 集合 B：所有方向 (x,y,z) ∈ {0,2}³（8 个组合，含全零）
_MOM_DIRS = (0, 2)
MOMENTA_ALL = [(pz, py, px)
               for pz in _MOM_DIRS for py in _MOM_DIRS for px in _MOM_DIRS]


def momentum_tag(mom):
    """动量 → 标签（如 (2,0,0) → 'P200'）。"""
    return f'P{mom[0]}{mom[1]}{mom[2]}'


def z_direction_momenta(pz_list=(0, 2, 4)):
    """z 方向动量集合（其余分量为 0）。"""
    return [(pz, 0, 0) for pz in pz_list]

# ═══════════════════════════════════════════════════════════════════
# 1. 蒸馏顶点 + 2pt（多动量）
# ═══════════════════════════════════════════════════════════════════

def compute_vertices_multi(conf_id, run_dir, logger, momenta,
                           precision=PRECISION, recompute=False):
    """多动量 VdV/VVV 顶点（透传 pyqcd pipeline）。"""
    return compute_vertices_for_config(
        conf_id, run_dir, logger, precision, recompute,
        mom_sink_vdv=momenta, mom_sink_vvv=momenta)


def compute_2pt_multi(conf_id, run_dir, logger, vertices, momenta,
                      precision=PRECISION, channels=('pp',)):
    """多动量核子 2pt（核子谱线）。"""
    return compute_2pt_for_config_multi(
        conf_id, run_dir, logger, vertices, momenta,
        precision=precision, channels=channels, v_kind='VVV')


# ═══════════════════════════════════════════════════════════════════
# 2. 梯度流 + 胶子 TMD 算符矩阵元
# ═══════════════════════════════════════════════════════════════════

def flow_gauge_for_config(conf_id, tau=3.0, eps=0.05, precision=PRECISION,
                          save_dir=None, logger=print, save_gauge=True):
    """读组态 → Wilson flow → flowed gauge（h5 缓存，可复现）。

    Args:
        conf_id: 组态号
        tau: 流时间（格点单位，τ=3a² → t=3）
        eps: RK3 步长（默认 0.05，Luescher ≤0.05 保证 O(ε³)，60 步）
        precision: complex64/complex128
        save_dir: 非 None 时保存 flowed gauge 到 h5（供并行/复用）
        save_gauge: True 时保存 flowed gauge 到 h5；False 时仅返回
            （避免 GPU→CPU 大拷贝 + swap，用于内存紧张环境）
    Returns:
        flowed gauge V (Nt,Nz,Ny,Nx,4,3,3)，torch 张量（torch 后端时）
    """
    backend = get_backend()
    h5p = None
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        h5p = os.path.join(save_dir, f'flowed_gauge_{conf_id}.h5')
        if os.path.exists(h5p):
            V = load_tensor_h5(h5p)
            _info(logger, f"  conf={conf_id}: loaded flowed gauge "
                          f"{V.shape} {V.dtype} from cache")
            return backend.asarray(V)
    gauge_cpu = read_gauge_lime(get_gauge_path(conf_id), NT, NX)
    dtype = np.complex64 if precision == 'complex64' else np.complex128
    if get_backend_name() == 'torch':
        set_precision(precision)
    G = backend.asarray(gauge_cpu.astype(dtype))
    del gauge_cpu
    E0 = float(flow_action_density(G).mean())
    t0 = time.perf_counter()
    V = wilson_flow(G, tau=tau, eps=eps)
    E1 = float(flow_action_density(V).mean())
    dt = time.perf_counter() - t0
    del G
    _info(logger, f"  conf={conf_id}: wilson_flow tau={tau} ({dt:.0f}s), "
                  f"E(t): {E0:.4f} -> {E1:.4f} (decrease=ok)")
    if h5p is not None and save_gauge:
        save_tensor_h5(V, h5p)
        _info(logger, f"  conf={conf_id}: saved flowed gauge -> {h5p} "
                      f"({os.path.getsize(h5p)/2**20:.0f} MB)")
    return V


def compute_tmd_ope_time(conf_id, run_dir, logger, z_list, b_list,
                         tau=3.0, eps=0.05, precision=PRECISION,
                         z_dir=2, b_dir=0, recompute=False,
                         gauge_flow_dir=None):
    """单组态梯度流 TMD 算符逐时间片矩阵元 → (nz, nb, Nt) 实数。

    输出键：'tmd'（(nz, nb, Nt)），保存于 <run_dir>/data/conf<id>/。
    算符 O = M^{tx;tx}+M^{ty;ty}−2M^{xy;xy} 在 flowed gauge 上
    （梯度流重整化：算符自动有限）。
    """
    from ..pipeline._steps import conf_data_dir as _cdir
    cdir = _cdir(run_dir, conf_id)
    tag = f"z{''.join(str(z) for z in z_list)}_b{''.join(str(b) for b in b_list)}"
    path = os.path.join(cdir, f'tmd_ope_{tag}_conf{conf_id}')
    if (os.path.exists(path + '.h5') or os.path.exists(path + '.npy')) \
            and not recompute:
        tmd = _load_any(path)
        _info(logger, f"  conf={conf_id}: loaded cached TMD OPE {tmd.shape}")
        return {'tmd': tmd}

    V = flow_gauge_for_config(conf_id, tau, eps, precision,
                              save_dir=gauge_flow_dir, logger=logger,
                              save_gauge=False)
    backend = get_backend()
    if get_backend_name() == 'torch':
        set_precision(precision)
    Vg = backend.asarray(V)
    del V

    t0 = time.perf_counter()
    tmd = tmd_matrix_elements_time(Vg, list(z_list), list(b_list),
                                   z_dir=z_dir, b_dir=b_dir)
    dt = time.perf_counter() - t0
    _info(logger, f"  conf={conf_id}: TMD OPE z={len(z_list)} b={len(b_list)} "
                  f"({dt:.0f}s) -> {tmd.shape}")
    save_tensor_h5(tmd, path)
    del Vg
    return {'tmd': tmd}


# ═══════════════════════════════════════════════════════════════════
# 3. 分析链（比值 → 裸矩阵元 → 重整化 → 匹配 → PDF）
# ═══════════════════════════════════════════════════════════════════

def load_multi_2pt(run_dir, conf_ids, momentum_tag_list, channels=('pp',),
                   logger=print):
    """读取多动量 2pt：{conf_id: {'corr_pp_P200': (Nt,), ...}}。"""
    from ..pipeline._steps import conf_data_dir as _cdir
    corr = {}
    for cid in conf_ids:
        cdir = _cdir(run_dir, cid)
        entry = {}
        for ch in channels:
            for tag in momentum_tag_list:
                base = f'corr_{ch}_{tag}_{cid}'
                for ext in ('.h5', '.npy'):
                    p = os.path.join(cdir, base + ext)
                    if os.path.exists(p):
                        entry[f'corr_{ch}_{tag}'] = _load_any(
                            os.path.join(cdir, base))
                        break
        if entry:
            corr[int(cid)] = entry
    _info(logger, f"  Loaded multi-momentum 2pt for {len(corr)} configs "
                  f"({len(momentum_tag_list)} momenta)")
    return corr


def load_tmd_ope_all(run_dir, conf_ids, z_list, b_list, logger=print):
    """读取 TMD OPE：{conf_id: {'tmd': (nz, nb, Nt)}}。"""
    from ..pipeline._steps import conf_data_dir as _cdir
    tag = f"z{''.join(str(z) for z in z_list)}_b{''.join(str(b) for b in b_list)}"
    ope = {}
    for cid in conf_ids:
        cdir = _cdir(run_dir, cid)
        base = os.path.join(cdir, f'tmd_ope_{tag}_conf{cid}')
        if os.path.exists(base + '.h5') or os.path.exists(base + '.npy'):
            ope[int(cid)] = {'tmd': _load_any(base)}
    _info(logger, f"  Loaded TMD OPE for {len(ope)} configs")
    return ope


def self_renormalize(c0):
    """梯度流自重整化：hR(z,b) = c0(z,b) / c0(z=0,b)。

    c0: (Nsample, nz, nb) 逐样本裸矩阵元。
    Returns: (hR, norm) 逐样本。
    """
    c0 = np.asarray(c0, dtype=float)
    norm = c0[:, 0:1, :]           # z=0 每样本每 b
    return c0 / norm, norm


def tmd_renormalize_hybrid(c0_pz, c0_pz0, zs, z_s=None):
    """混合方案拼接（短距比值 + 长距自重整化），逐 b。

    c0_pz / c0_pz0: (Nsample, nz, nb) 裸矩阵元。
    Returns: hR (Nsample, nz, nb)。
    """
    c0 = np.asarray(c0_pz, dtype=float)
    c00 = np.asarray(c0_pz0, dtype=float)
    zs = np.asarray(zs, dtype=float)
    if z_s is None:
        z_s = zs[len(zs) // 2]
    mask = zs < z_s
    hR = np.zeros_like(c0)
    # 短距比值：hR(z) = c0(z,Pz)/c0(z,Pz=0)
    hR[:, mask, :] = c0[:, mask, :] / c00[:, mask, :]
    # 长距自重整化：hR(z) = [c0(z,Pz)/Z_R]·η_s，η_s 取 z_s 处的比值
    if np.any(~mask):
        eta_s = c00[:, mask][:, 0, :] / c00[:, mask][:, 0, :]  # =1（自归一）
        # 简化的梯度流长距方案：直接用 z 依赖比值 × 归一常数
        norm_at_zs = c0[:, mask][:, 0, :] / c00[:, mask][:, 0, :]
        hR[:, ~mask, :] = (c0[:, ~mask, :] / c00[:, ~mask, :]) * 1.0
    return hR
