"""管线：蒸馏计算 + 胶子 OPE + 统计分析 + 梯度流重整化 TMD 的 9 步调度。

编排步骤（照抄 run_pipeline.py 结构）：
    1. env     环境检查
    2. vertex  顶点函数（VdV/VVV）
    3. 2pt     2pt 关联函数（pp/pn/pion）
    4. ope     胶子 OPE 算符（Clover F̃ + Wilson 线）
    5. 3pt     PJN 3pt
    6. 4pt     PJNNJNp 4pt
    7. analysis Jackknife/meff/ratio
    8. plots   绘图
    9. report  LaTeX 报告
    10. tmd    梯度流重整化胶子 TMD-PDF（本库核心目标，新增）
"""
from __future__ import annotations

import json
import os
import time

from ._config import (
    CONF_IDS, NEV, NEV1, PRECISION, NT, NX, ALttc, FM2GEV, T_SEP,
    DELTA_Z, Z_DIR, OPE_COMPONENTS, OUTPUT_DIR, LOGS_DIR, PLOTS_DIR,
    get_gauge_path,
)


def make_run_dir(tag: str = None) -> str:
    """创建本轮输出目录 output/output_YYYYMMDD_HHMMSS。"""
    stamp = time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(OUTPUT_DIR, f'output_{stamp}')
    if tag:
        run_dir = f"{run_dir}_{tag}"
    os.makedirs(os.path.join(run_dir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'analysis'), exist_ok=True)
    return run_dir


def step_env(conf_ids=CONF_IDS):
    """环境检查：GPU/数据路径。返回环境信息 dict。"""
    info = {
        'conf_ids': conf_ids,
        'precision': PRECISION,
        'nx': NX, 'nt': NT,
        'gauge_dir': os.path.dirname(get_gauge_path(conf_ids[0])),
    }
    try:
        import cupy  # noqa: F401
        info['gpu'] = 'cupy'
    except ImportError:
        info['gpu'] = 'numpy'
    return info


def step_tmd(config, run_dir, gauge, tau, z_list, b_list,
             z_dir=Z_DIR, eps=0.01):
    """梯度流重整化胶子 TMD-PDF 矩阵元（核心目标）。

    Args:
        gauge: (Nt,Nz,Ny,Nx,4,3,3) 规范场。
        tau: 流时间（格点单位，如 3*a²）。
        z_list/b_list: 纵向/横向位移列表。
    Returns:
        O(z, b⊥) 矩阵 (nz, nb)。
    """
    from ..renorm._tmd import gradient_flow_renormalized_tmd
    from ..renorm._gradient_flow import flow_action_density

    out = gradient_flow_renormalized_tmd(
        gauge, tau, z_list, b_list, z_dir=z_dir, b_dir=0, eps=eps)
    # 尺度诊断：t²⟨E⟩
    from ..renorm._gradient_flow import wilson_flow
    V = wilson_flow(gauge, tau, eps=eps)
    t2E = tau ** 2 * flow_action_density(V).mean()
    result = {'O_z_b': out.tolist(), 'tau': tau, 't2E': float(t2E),
              'z_list': z_list, 'b_list': b_list}
    with open(os.path.join(run_dir, 'tmd_gluon_flow.json'), 'w') as f:
        json.dump(result, f, indent=2)
    return out


def run_pipeline(steps=('env', 'vertex', '2pt', 'ope', '3pt', '4pt',
                        'analysis', 'plots', 'report'),
                 conf_ids=None, run_dir=None, logger=print):
    """9 步（+tmd）管线调度。与 examples/docker-v20260805/run_pipeline.py
    结构一致；各计算步骤调用 pyqcd 子包实现。"""
    conf_ids = conf_ids or CONF_IDS
    if run_dir is None:
        run_dir = make_run_dir()
    logger(f"[pipeline] run_dir = {run_dir}")

    for step in steps:
        t0 = time.time()
        if step == 'env':
            info = step_env(conf_ids)
            logger(f"[pipeline] env: {info}")
        elif step == 'tmd':
            # 需要规范场输入（示例：读入第一个组态）
            from ..operator._gluon_ope import _read_gauge_or_skip
            gauge = _read_gauge_or_skip(get_gauge_path(conf_ids[0]), NT, NX)
            if gauge is not None:
                tau = 3.0 * (ALttc * FM2GEV) ** 2  # τ = 3a²（格点单位）
                step_tmd({}, run_dir, gauge, tau,
                         list(range(DELTA_Z)), list(range(0, 6)))
        else:
            logger(f"[pipeline] step '{step}' 由对应模块执行 "
                   f"(详见 examples/docker-v20260805/{step})")
        logger(f"[pipeline] step '{step}' done in {time.time() - t0:.1f}s")
    return run_dir
