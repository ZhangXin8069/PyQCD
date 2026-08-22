"""
自重整化因子 Z_R：全局参数化与拟合（移植 zengch fit_zr_new.py 核心逻辑）。

对应 arXiv:2510.17758 Eq.(3)-(8) 的混合方案自重整化：

    hB(z, Pz=0) 在短距离 (z ≤ z₁) 由 NLO MS-bar 因子 Z_MS 控制；
    长距离 (z > z₁) 由非微扰节点参数 g₁..g₁₄ 描述（th_hB 的分段 B(z)）。

    Z_R(z, a, μ) = exp[ k·z/(a·ln(aΛ)) + ½ln(1 + d/ln(aΛ))²
                       + 5C_A/(3b₀)·ln(ln(1/(aΛ))/ln(μ/Λ))
                       + (m₀ + m₂a²)·z + f·a + f₂·a² ]

其中线性发散项 k·z/a 为梯度流（或 Wilson 线）重整化所需的发散抵消，
对数重求和项对应 MS-bar 窗口。本模块把参数化函数与拟合代价函数做成
纯函数库（不依赖集群数据路径），数据由调用方以 dict 形式传入。
"""
from __future__ import annotations

import numpy as np

from ._const import CA, gammaE, b0
from ._ensembles import fm_to_GeV


def Z_MS(z_gev, mu):
    """NLO MS-bar 重整化因子（z 单位 GeV⁻¹，μ 单位 GeV）。

    Z_MS = 1 + α_s·C_A/(4π)·[ (5/3)·ln( z²μ² / (4e^{−2γ_E}) ) + 3 ]
    """
    alpha_s = A_s_run(mu)
    return 1.0 + alpha_s * CA / (4.0 * np.pi) * (
        5.0 / 3.0 * np.log((z_gev ** 2 * mu ** 2) / (4.0 * np.exp(-2.0 * gammaE))) + 3
    )


def A_s_run(mu, Lambda_QCD=0.23, nf=3.0):
    """α_s/(4π) 1 圈运行耦合（zengch constant.py 的 A_s 语义）。"""
    from ._const import A_s
    return A_s(mu, Lambda_QCD, nf)


def th_hB(z_, a_, mu_, par_g_set, f_set, z1_fm=0.301):
    """对数 hB(z) 的理论参数化（Eq.7）：分段——z ≤ z₁ 用 Z_MS + 质量项，
    z > z₁ 用非微扰节点 g₁..g₁₄。

    Args:
        z_:   z 数组（fm）
        a_:   格距（GeV⁻¹）
        mu_:  重整化标度（GeV）
        par_g_set: (k, d, m0, m2, Lambda_QCD, g1..g14)
        f_set: (f1, f2) 离散化修正系数
    Returns:
        log hB，形状与 z_ 相同。
    """
    z_set_new_gev = np.asarray(z_, dtype=float) / fm_to_GeV
    k, d, m0, m2, lambda_qcd, *g_params = par_g_set
    g_params = np.asarray(g_params, dtype=float)

    nf = 3.0
    b0_ = b0(nf)

    def B(z_i):
        z1 = z1_fm / fm_to_GeV  # GeV⁻¹
        result = np.zeros_like(np.asarray(z_i, dtype=float))
        mask = z_i <= z1
        z_masked = z_i[mask]
        result[mask] = np.log(Z_MS(z_masked, mu_)) + (m0 + m2 * a_ ** 2) * z_masked
        result[~mask] = g_params[: len(result[~mask])]
        return result

    a_set = np.array([a_, a_ ** 2.0])
    f_set_arr = np.asarray(f_set, dtype=float)

    log_hb = (k * z_set_new_gev) / (a_ * np.log(a_ * lambda_qcd))
    log_hb += 5.0 * CA / (3.0 * b0_) * np.log(
        np.log(1.0 / (a_ * lambda_qcd)) / np.log(mu_ / lambda_qcd)
    )
    log_hb += np.log((1.0 + d / np.log(a_ * lambda_qcd)) ** 2.0) / 2.0
    log_hb += f_set_arr @ a_set
    log_hb += B(z_set_new_gev)
    return log_hb


def th_ZR(z_, a_, mu_, k, d, m0, m2, lambda_qcd, f_set):
    """Z_R 的指数参数化（Eq.5）：exp(线性发散 + 质量平移 + 对数重求和 + 离散化)。

    Args:
        z_:   z 数组（fm）
        a_:   格距（GeV⁻¹）
        mu_:  重整化标度（GeV）
        k, d, m0, m2, lambda_qcd: 拟合参数
        f_set: (f1, f2)
    Returns:
        Z_R(z)，形状与 z_ 相同。
    """
    z_set_new_gev = np.asarray(z_, dtype=float) / fm_to_GeV
    nf = 3.0
    b0_ = b0(nf)
    a_set = np.array([a_, a_ ** 2.0])
    f_set_arr = np.asarray(f_set, dtype=float)

    log_hb = (k * z_set_new_gev) / (a_ * np.log(a_ * lambda_qcd))
    log_hb += 5.0 * CA / (3.0 * b0_) * np.log(
        np.log(1.0 / (a_ * lambda_qcd)) / np.log(mu_ / lambda_qcd)
    )
    log_hb += np.log((1.0 + d / np.log(a_ * lambda_qcd)) ** 2.0) / 2.0
    log_hb += (m0 + m2 * a_ ** 2) * z_set_new_gev
    if f_set_arr.ndim == 1:
        log_hb += f_set_arr @ a_set
    else:
        log_hb += np.sum(f_set_arr * a_set[:, None], axis=0)
    return np.exp(log_hb)


def cost_function(z_set_, hb_data, c_inv, a_, mu_, par_set, z1_fm=0.301):
    """单系综 χ²（平均到每数据点）。

    Args:
        z_set_: z 数组（fm）
        hb_data: log hB 数据数组
        c_inv: 协方差逆矩阵
        a_: 格距（GeV⁻¹）
        mu_: 重整化标度（GeV）
        par_set: 前 19 个为 (k,d,m0,m2,Λ,g1..g14)，其余为 (f1,f2)
    """
    data_num = len(z_set_)
    par_g_set = par_set[:19]
    f_set = par_set[19:]
    hb_th = th_hB(z_set_, a_, mu_, par_g_set, f_set, z1_fm)
    del_hb = hb_th - np.asarray(hb_data, dtype=float)
    chi2 = del_hb.T @ c_inv @ del_hb
    return chi2 / data_num


def cost_function_all(par_set, datasets, mu_):
    """多系综联合 χ²/dof。

    Args:
        par_set: 全部拟合参数（前 19 + f1,f2）
        datasets: [dict(z=..., loghB=..., c_inv=..., a=..., n=...), ...]
        mu_: 重整化标度（GeV）
    """
    chi2_sum = 0.0
    n_sum = 0
    for ds in datasets:
        n = len(ds['z'])
        chi2 = cost_function(ds['z'], ds['loghB'], ds['c_inv'],
                             ds['a'], mu_, par_set)
        chi2_sum += chi2 * n
        n_sum += n
    dof = n_sum - len(par_set)
    return chi2_sum / dof


# ═══════════════════════════════════════════════════════════════════
# hB/FH 数据预处理 loader（整合 zengch hB_data_FeynmenHellman_new.py）
# ═══════════════════════════════════════════════════════════════════

def build_hB_dataset(c0_zx, z_fm, z_grid_new=None):
    """ratio c0(z) 数据 → z₀ 归一化 + 线性插值 hB 数据集。

    照抄 zengch hB_data_FeynmenHellman_new.py 的语义（去集群路径依赖）：
        1) hB_o_zn = c0(z) 原始（未归一化）
        2) hB_o    = c0 / c0(z=0)                     （z₀ 归一化）
        3) 线性插值到目标 z 网格 → hB、loghB

    Args:
        c0_zx: (nz, nsample) 或 (nz,)——FH 比值拟合的 c0(z)（逐样本）。
        z_fm:  与 c0_zx 第一维对应的 z 网格（fm）。
        z_grid_new: 目标插值网格（fm）；默认 arange(0.15, 1.05, 0.05)
                    （L48x144 类小体积系综可传至 0.95，同原版约定）。
    Returns:
        dict(z, loghB, hB, z_o, hB_o, hB_o_zn)——形状 (nz_new, nsample)
        或 (nz_new,)（跟随输入维数）。
    """
    from scipy.interpolate import interp1d

    hb_zn = np.atleast_2d(np.asarray(c0_zx, dtype=float))
    if hb_zn.shape[0] == 1 and np.ndim(c0_zx) == 1:
        hb_zn = hb_zn.T                      # (1, nz) 边角：按 (nz,) 处理
    z_o = np.asarray(z_fm, dtype=float)
    if hb_zn.shape[0] != len(z_o):
        raise ValueError(f"c0 首维 {hb_zn.shape[0]} 与 z 网格 {len(z_o)} 不一致")

    hb_o = hb_zn / hb_zn[0:1, :]             # z₀ 归一化
    if z_grid_new is None:
        z_grid_new = np.arange(0.15, 1.0 + 0.05, 0.05)
    interp = interp1d(z_o, hb_o, kind='linear', axis=0,
                      bounds_error=False, fill_value='extrapolate')
    hb_new = interp(np.asarray(z_grid_new, dtype=float))

    squeeze = np.ndim(c0_zx) == 1
    if squeeze:
        hb_new = hb_new[:, 0]
        return {'z': np.asarray(z_grid_new), 'loghB': np.log(hb_new),
                'hB': hb_new, 'z_o': z_o, 'hB_o': hb_o[:, 0],
                'hB_o_zn': hb_zn[:, 0]}
    return {'z': np.asarray(z_grid_new), 'loghB': np.log(hb_new),
            'hB': hb_new, 'z_o': z_o, 'hB_o': hb_o, 'hB_o_zn': hb_zn}


def boot_covariance(samples, n_rep=200, seed=0):
    """自助重采样协方差（照抄 zengch tool.covariance_matrix(·,'boot') 语义）。

    Args:
        samples: (n_point, n_sample)——逐 bootstrap/jackknife 样本。
        n_rep: 重采样次数（对样本轴有放回抽取）。
        seed: 可复现种子。
    Returns:
        (n_point, n_point) 协方差矩阵。
    """
    s = np.asarray(samples, dtype=float)
    rng = np.random.default_rng(seed)
    n_pt, n_sam = s.shape
    idx = rng.integers(0, n_sam, size=(n_rep, n_sam))
    replicates = s[:, idx].mean(axis=1).T          # (n_rep, n_point)
    return np.cov(replicates, rowvar=False)


def make_zr_dataset(loghB_samples, z_fm, a_gev_inv, kind='boot',
                    n_rep=200, seed=0):
    """组装 fit_ZR / cost_function_all 所需数据集 dict。

    Args:
        loghB_samples: (nz, nsample)——归一化 log hB 的逐样本数组
                       （build_hB_dataset 输出的 loghB）。
        z_fm: z 网格（fm，与第一维对应）。
        a_gev_inv: 格距（GeV⁻¹，a_fm/fm_to_GeV）。
        kind: 'boot' 自助协方差（pinv 防奇异）/ 'diag' 对角方差。
    Returns:
        dict(z, loghB(均值), c_inv, a)——直接可入 datasets 列表。
    """
    s = np.atleast_2d(np.asarray(loghB_samples, dtype=float))
    z_ = np.asarray(z_fm, dtype=float)
    if kind == 'boot':
        cov = boot_covariance(s, n_rep=n_rep, seed=seed)
        c_inv = np.linalg.pinv(cov)
    elif kind == 'diag':
        var = s.var(axis=1)
        c_inv = np.diag(1.0 / np.maximum(var, 1e-30))
    else:
        raise ValueError(f"未知 kind: {kind}")
    return {'z': z_, 'loghB': s.mean(axis=1), 'c_inv': c_inv, 'a': a_gev_inv}


def fit_ZR(par_ini, datasets, mu_, use_iminuit=True):
    """全局拟合 Z_R 参数（iminuit 或 scipy 回退）。

    Args:
        par_ini: 参数初值（长度 ≥ 21：k,d,m0,m2,Λ,g1..g14,f1,f2）
        datasets: [dict(z, loghB, c_inv, a), ...]
        mu_: 重整化标度（GeV）
    Returns:
        拟合参数数组（m.values）。
    """
    par_name = ('k', 'd', 'm0', 'm2', 'Lambda_QCD',
                'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8',
                'g9', 'g10', 'g11', 'g12', 'g13', 'g14', 'f1', 'f2')

    def cost(par):
        return cost_function_all(par, datasets, mu_)

    if use_iminuit:
        try:
            from iminuit import Minuit
            m = Minuit(cost, par_ini, name=par_name)
            m.limits['k'] = (None, None)
            m.limits['d'] = (None, None)
            m.limits['m0'] = (None, None)
            m.limits['m2'] = (None, None)
            m.limits['Lambda_QCD'] = (0, None)
            m.fixed['f1', 'f2', 'm2'] = True
            m.migrad()
            return np.array(m.values)
        except ImportError:
            pass  # 回退到 scipy

    from scipy.optimize import minimize
    res = minimize(cost, par_ini, method='Nelder-Mead',
                   options={'maxiter': 5000, 'xatol': 1e-6})
    return res.x


# ═══════════════════════════════════════════════════════════════════
# 逐样本重拟合环（整合 zengch fit_zr_new.fit_ZR 的 bootstrap 样本循环）
# ═══════════════════════════════════════════════════════════════════

_ZR_PAR_NAMES = ('k', 'd', 'm0', 'm2', 'Lambda_QCD',
                 'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8',
                 'g9', 'g10', 'g11', 'g12', 'g13', 'g14', 'f1', 'f2')


def fit_ZR_samples(par_ini, dataset_samples, mu_, use_iminuit=True):
    """Z_R 参数误差的逐样本重拟合环（照抄 zengch fit_zr_new.fit_ZR 样本循环）。

    对每个 bootstrap/jackknife 样本 i：以各系综 loghB 矩阵的第 i 列为
    "均值"数据重跑全局拟合，收集参数分布 → 均值±std（原版逐行写 CSV）。
    协方差 c_inv 与原版一致地跨样本固定（由全体样本的 std 构建后传入）。

    Args:
        par_ini: 参数初值（长度 ≥ 21）。
        dataset_samples: [dict(z, loghB=(nz,n_rep) 样本矩阵, c_inv, a), ...]
            n_rep 取各系综的最小列数（原版取第一系综列数，此处更稳健）。
        mu_: 重整化标度（GeV）。
    Returns:
        list[dict]：每样本一行（sample_i, k, d, m0, m2, Lambda_QCD,
        g1..g14, f1, f2, chi2）；单个样本拟合失败记 NaN 并告警继续
        （对原版行为的唯一偏离，防单坏样本中断整环）。
    """
    n_rep = min(ds['loghB'].shape[1] for ds in dataset_samples)
    rows = []
    for i in range(n_rep):
        datasets_i = [dict(z=ds['z'], loghB=ds['loghB'][:, i],
                           c_inv=ds['c_inv'], a=ds['a'])
                      for ds in dataset_samples]
        try:
            par_fit = fit_ZR(par_ini, datasets_i, mu_,
                             use_iminuit=use_iminuit)
            chi2 = cost_function_all(par_fit, datasets_i, mu_)
        except Exception as exc:  # noqa: BLE001 —— 单坏样本不中断整环
            print(f"[fit_ZR_samples] sample {i} failed: {exc}")
            par_fit = np.full(len(_ZR_PAR_NAMES), np.nan)
            chi2 = np.nan
        row = {"sample_i": i}
        row.update({name: float(val)
                    for name, val in zip(_ZR_PAR_NAMES, par_fit)})
        row["chi2"] = float(chi2) if np.isfinite(chi2) else np.nan
        rows.append(row)
    return rows


def summarize_ZR_samples(rows):
    """逐样本拟合结果的参数分布汇总（mean/std；原版以 CSV 供人工统计）。"""
    keys = [k for k in rows[0].keys() if k != "sample_i"]
    summary = {}
    for key in keys:
        vals = np.array([r[key] for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        summary[key] = (float(np.mean(vals)) if vals.size else np.nan,
                        float(np.std(vals)) if vals.size else np.nan)
    return summary
