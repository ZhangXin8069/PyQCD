"""
核子胶子 TMD-PDF 裸矩阵元提取：不相连 ratio（code_1.py 算法，b⊥ 扩展）
====================================================================

梯度流重整化方案中，核子中胶子 TMD 算符矩阵元 <N|O(z,b⊥)|N> 通过
不相连因子化三点函数计算（照抄 docker-v20260805 analyze.py 的
code_1.py 算法，扩展 b⊥ 维度）：

    C3(dt, dtau, z, b) = C2(dt) · OPE(dtau, z, b)        （不相连因子化）
    C3_disc = C3 − C2·⟨OPE⟩                               （真空扣除）
    R(dt, dtau, z, b) = ⟨C3_disc / C2⟩_ti
    逐 (z, b) 相关拟合：R = c0 + c1·e^{−dE·dtau} + c1·e^{−dE·(dt−dtau)}

其中 OPE(dtau, z, b) 为梯度流涂抹后的胶子 TMD staple 算符
O(z,b⊥) = M^{tx;tx} + M^{ty;ty} − 2M^{xy;xy} 在时间片 dtau 的
空间求和值（形状 (nz, nb, Nt)），C2 为核子 2pt 关联函数。

输出：裸矩阵元 c0(z,b)（逐样本：jackknife）→ 供 Z_R / 混合方案
自重整化链使用。
"""
from __future__ import annotations

import os
import time

import numpy as np

from ._disconnected import sem, resample, cov_mat, model_ratio


def run_disconnected_tmd_ratio(corr_2pt_all, ope_all, conf_ids,
                               run_dir, logger=print,
                               NT=72, nz=24, nb=1,
                               dt_max=20, dt_start=7, dt_end=10,
                               cut=6, p0=None, momentum='P200',
                               z_s=0):
    """不相连胶子 TMD ratio + 逐 (z,b) 拟合（code_1.py 算法，b⊥ 扩展）。

    Args:
        corr_2pt_all: {conf_id: {f'corr_pp_{mom}': (Nt,)}} 核子 2pt。
        ope_all:      {conf_id: {'tmd': (nz, nb, Nt)}} 梯度流 TMD 算符
                      逐时间片空间求和（实数）。
        conf_ids:     组态列表（jackknife 重采样轴）。
        run_dir:      输出目录（analysis/tmd_ratio/）。
        nz:           z 方向格点数（OPE 的 z 维长度）。
        nb:           b⊥ 维长度。
        momentum:     动量标签（'P200' 等，用于命名输出文件）。
        z_s:          比值参考 z 索引（保留参数，备用）。
    Returns:
        ch_results: {hadron: {'ratio', 'c0', 'c1', 'dE', 'chi2', 'x_coor'}}
            c0/c1/dE/chi2 形状 (Nsample, nz, nb)。
    """
    import lsqfit
    import gvar as gv

    if p0 is None:
        p0 = {"c0": 0.6, "c1": -2, "dE": 1}

    Nconf = len(conf_ids)
    Nsample = max(Nconf, 1)
    # 重采样方式选择：delete-one jackknife 需样本数显著大于数据点数；
    # 小样本用 bootstrap（Nsample=200 满秩协方差，svdcut 下拟合稳定）。
    # 守卫：Nconf<2 禁用 jackknife（n−1=0 除零 → 全 NaN）
    front_remove = cut // 2
    back_remove = cut - front_remove
    # 真实数据点计数（与下方 x_coor 同式）：Σ_dt max(dt−front−back+1, 0)
    Ndata_est = sum(max(dt - front_remove - back_remove + 1, 0)
                    for dt in range(dt_start, dt_end + 1))
    jack = (Nconf >= 2) and (Nconf > Ndata_est)
    if not jack:
        Nsample = 200   # bootstrap 重采样 200 次（协方差满秩）
    if Nconf < 2:
        Nsample = 1     # 单组态：重采样无统计意义，走下方直通分支
    out_dir = os.path.join(run_dir, 'analysis', 'tmd_ratio')
    os.makedirs(out_dir, exist_ok=True)

    channels = [('proton', 'corr_pp', 'proton')]
    ch_results = {}

    for ch_key, k2, had_name in channels:
        logger(f"\n  Channel: {had_name} at {momentum} (TMD, b⊥={nb} vals)")

        key2 = f'{k2}_{momentum}'
        _corr = np.stack([np.real(corr_2pt_all[cid][key2]) for cid in conf_ids])
        full = np.zeros((Nconf, NT, NT), dtype=np.float64)
        for ti in range(NT):
            full[:, :, ti] = np.roll(_corr, -ti, axis=1)

        # OPE tmd: (Nconf, nz, nb, Nt) → (Nconf, tau, z, b)
        _ope = np.stack([np.real(ope_all[cid]['tmd']) for cid in conf_ids])
        _ope = _ope.transpose(0, 3, 1, 2)   # (Nconf, Nt, nz, nb)

        # 相对时间构造
        _corr2_rel = np.zeros((Nconf, NT, dt_max), dtype=np.float64)
        _ope_rel = np.zeros((Nconf, NT, dt_max, nz, nb), dtype=np.float64)
        for ti in range(NT):
            corr2_shift = np.roll(full[:, :, ti], -ti, axis=1)
            _corr2_rel[:, ti, :] = corr2_shift[:, :dt_max]
            ope_shift = np.roll(_ope, -ti, axis=1)
            _ope_rel[:, ti, :, :, :] = ope_shift[:, :dt_max, :, :]

        # 不相连 3pt = C2 × OPE（因子化）
        _corr3 = np.zeros((Nconf, NT, dt_max, dt_max, nz, nb), dtype=np.float64)
        for _dt in range(dt_max):
            for _dtau in range(_dt + 1):
                _corr3[:, :, _dt, _dtau, :, :] = (
                    _ope_rel[:, :, _dtau, :, :]
                    * _corr2_rel[:, :, _dt][:, :, None, None])

        if Nconf < 2:
            # 单组态直通：不重采样（bootstrap 复制亦无意义），保形状 (1, ...)
            corr2, ope, corr3 = _corr2_rel, _ope_rel, _corr3
        else:
            corr2 = resample(_corr2_rel, jack, Nsample)
            ope = resample(_ope_rel, jack, Nsample)
            corr3 = resample(_corr3, jack, Nsample)

        # 真空扣除 + ratio
        corr3_disc = corr3 - corr2[:, :, :, None, None, None] * ope[:, :, None, :, :, :]
        eps = 1e-30
        ratio = np.mean(corr3_disc / (corr2[:, :, :, None, None, None] + eps), axis=1)
        ratio = ratio.real   # (Nsample, dt, dtau, nz, nb)
        np.save(os.path.join(out_dir, f'ratio_{had_name}_{momentum}.npy'), ratio)

        # 逐 (z,b) 相关拟合（Nconf<2 时跳过——协方差奇异，统计无意义）
        if Nconf < 2:
            logger(f"  Nconf={Nconf} < 2：跳过 (z,b) 拟合（统计无意义），"
                   f"仅保存 ratio")
            ch_results[had_name] = {
                'ratio': ratio, 'c0': np.zeros((1, nz, nb)),
                'c1': np.zeros((1, nz, nb)),
                'dE': np.zeros((1, nz, nb)),
                'chi2': np.zeros((1, nz, nb)), 'x_coor': [],
                'c0_plateau': plateau_c0(ratio, dt_max=dt_max,
                                         dt_start=dt_start,
                                         dt_end=dt_end, cut=cut),
            }
            continue

        # 逐 (z,b) 相关拟合（front/back_remove 已在重采样选择处计算）
        x_coor = [(dt, dtau)
                  for dt in range(dt_start, dt_end + 1)
                  for dtau in range(front_remove, dt - back_remove + 1)]
        Ndata = len(x_coor)

        para_c0 = np.zeros((Nsample, nz, nb))
        para_c1 = np.zeros((Nsample, nz, nb))
        para_dE = np.zeros((Nsample, nz, nb))
        chi2 = np.zeros((Nsample, nz, nb))

        report_lines = [
            "=" * 70,
            f"  TMD Fit Report: {had_name}, {momentum}, Nconf={Nconf}",
            "=" * 70,
            f"  t_sep range : [{dt_start}, {dt_end}]",
            f"  cut         : {cut}",
            f"  Nsample     : {Nsample}",
            f"  jackknife   : {jack}",
            f"  nz x nb     : {nz} x {nb}",
            "=" * 70, "",
        ]

        t0_fit = time.perf_counter()
        for _z in range(nz):
            for _b in range(nb):
                sub_sample = np.zeros((Nsample, Ndata))
                for i, (dt, dtau) in enumerate(x_coor):
                    sub_sample[:, i] = ratio[:, dt, dtau, _z, _b]
                cov, cond = cov_mat(sub_sample, jack)
                report_lines += [f"z={_z} b={_b} cond={cond:.3g}", "-" * 56, ""]

                for _id in range(Nsample):
                    y_coor = gv.gvar(sub_sample[_id], cov)
                    # 条件数过大时回退对角近似（协方差奇异，参考代码预留选项）
                    if cond > 1e8 or not np.isfinite(cond):
                        y_coor = gv.gvar(sub_sample[_id], np.diag(np.diag(cov)))
                        svd = 1e-6
                    else:
                        svd = 1e-6
                    _fit = lsqfit.nonlinear_fit(data=(x_coor, y_coor), p0=p0,
                                                fcn=model_ratio, svdcut=svd)
                    para_c0[_id, _z, _b] = _fit.pmean["c0"]
                    para_c1[_id, _z, _b] = _fit.pmean["c1"]
                    para_dE[_id, _z, _b] = _fit.pmean["dE"]
                    chi2[_id, _z, _b] = _fit.chi2 / _fit.dof
                if _id == Nsample - 1:
                    report_lines.append(_fit.format(maxline=True))
                    report_lines += ["", ""]

                logger(f"  z={_z} b={_b}: c0={para_c0[:, _z, _b].mean():.4g} ± "
                       f"{sem(para_c0[:, _z, _b], jack):.4g}  "
                       f"chi2/dof={chi2[:, _z, _b].mean():.2g}")

        report = "\n".join(report_lines)
        with open(os.path.join(out_dir, f'1_fit_report_{momentum}.txt'), 'w') as f:
            f.write(report)
        np.savez(os.path.join(out_dir, f'0_fit_data_{momentum}.npz'),
                 c0=para_c0, c1=para_c1, dE=para_dE, chi2=chi2)
        # plateau 均值版（fit 窗口内直接平均，抗奇异协方差；test9 整合项）
        c0_plateau = plateau_c0(ratio, dt_max=dt_max, dt_start=dt_start,
                                dt_end=dt_end, cut=cut)
        np.save(os.path.join(out_dir, f'c0_plateau_{momentum}.npy'),
                c0_plateau)
        logger(f"  Saved ratio + fit + c0_plateau to {out_dir} "
               f"({time.perf_counter()-t0_fit:.1f}s)")

        ch_results[had_name] = {
            'ratio': ratio, 'c0': para_c0, 'c1': para_c1,
            'dE': para_dE, 'chi2': chi2, 'x_coor': x_coor,
            'c0_plateau': c0_plateau,
        }

    return ch_results


def plot_tmd_c0(ch_results, run_dir, momentum, logger=print,
                nz=24, nb=1, out_name='c0_tmd'):
    """c0(z,b) 图表：逐 b 面板（z 横轴，errorbar）+ 3D 热图。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_dir = os.path.join(run_dir, 'analysis', 'tmd_ratio')
    os.makedirs(out_dir, exist_ok=True)

    for had_name, res in ch_results.items():
        c0 = res['c0']               # (Nsample, nz, nb)
        c0m = c0.mean(0); c0e = sem(c0, True)
        z = np.arange(nz)

        # 逐 b 面板
        ncol = 2
        nrow = (nb + 1) // 2
        fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3.5 * nrow),
                                 squeeze=False)
        for _b in range(nb):
            ax = axes[_b // ncol][_b % ncol]
            ax.errorbar(z, c0m[:, _b], yerr=c0e[:, _b], fmt='x-', capsize=3)
            ax.axhline(0, color='gray', lw=0.8)
            ax.set_xlabel('z'); ax.set_ylabel('c0')
            ax.set_title(f'{had_name} {momentum}: c0(z) b⊥={_b}')
            ax.grid(alpha=0.3)
        for _b in range(nb, nrow * ncol):
            axes[_b // ncol][_b % ncol].axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'{out_name}_{had_name}_{momentum}.png'),
                    dpi=150)
        plt.close(fig)

        # 热图（nb≥2 时）
        if nb >= 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(c0m.T, aspect='auto', origin='lower',
                           extent=[0, nz - 1, 0, nb - 1])
            ax.set_xlabel('z'); ax.set_ylabel('b⊥')
            ax.set_title(f'{had_name} {momentum}: c0(z, b⊥)')
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir,
                                     f'{out_name}_heatmap_{had_name}_{momentum}.png'),
                        dpi=150)
            plt.close(fig)

    logger(f"  TMD c0 plots saved to {out_dir}")


def plateau_c0(ratio, dt_max=20, dt_start=7, dt_end=10, cut=6):
    """fit 窗口内 ratio 的 plateau 均值 → c0(z,b)（抗奇异协方差）。

    整合自 examples/pyqcd/test9_gluon_tmd_nucleon.py::_plateau_c0：
    与 run_disconnected_tmd_ratio 的 x_coor 窗口一致
    （dt∈[dt_start,dt_end]，dtau∈[front,dt-back]），窗口内直接平均。
    10 组态统计下比 lsqfit 逐样本拟合更稳健（协方差奇异的绕行方案）。

    Args:
        ratio: (Nsample, dt_max, dtau_max, nz, nb)——不相连比值数组。
    Returns:
        (Nsample, nz, nb) plateau 均值。
    """
    r = np.asarray(ratio)
    Nsample, _, _, nz, nb = r.shape
    front = cut // 2
    back = cut - front
    out = np.zeros((Nsample, nz, nb))
    npts = 0
    for dt in range(dt_start, min(dt_end, r.shape[1] - 1) + 1):
        for dtau in range(front, dt - back + 1):
            out += r[:, dt, dtau, :, :]
            npts += 1
    return out / max(npts, 1)


def plot_tmd_pdf(x_grid, xg_quasi, xg_matched, b_grid_fm, cs_kernel,
                 tag, out_dir, logger=print):
    """TMD-PDF 链成图（整合自 test9 示例 plot_pdf）。

    产出：quasi_tmd_pdf.png / matched_tmd_pdf.png / tmd_pdf_vs_b.png /
    cs_kernel.png（cs_kernel 非 None 时）。

    Args:
        x_grid: x 网格。
        xg_quasi: 准 TMD-PDF x·g̃(x, b⊥)，(nx, nb) 或 (nx,)。
        xg_matched: NLO 匹配后 x·g(x, b⊥)。
        b_grid_fm: b⊥ 网格（fm，标记用）。
        cs_kernel: K(b⊥) 数组或 None。
        tag: 动量标签（图题）。
        out_dir: 输出目录。
    Returns:
        生成的 png 路径列表。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    xg = np.asarray(xg_quasi)
    xgm = np.asarray(xg_matched)
    if xg.ndim == 1:
        xg = xg[:, None]
    if xgm.ndim == 1:
        xgm = xgm[:, None]
    nb = xg.shape[1]
    ncol = 2
    nrow = (nb + 1) // 2
    saved = []

    # 准 TMD-PDF x·g̃(x, b⊥)
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3.5 * nrow),
                             squeeze=False)
    for b in range(nb):
        ax = axes[b // ncol][b % ncol]
        ax.plot(x_grid, xg[:, b], 'o-', ms=3)
        ax.axhline(0, color='gray', lw=0.8)
        ax.set_xlabel('x'); ax.set_ylabel(r'$x\tilde{g}(x,b_\perp)$')
        ax.set_title(f'quasi TMD-PDF, b={b_grid_fm[b]:.3f} fm')
        ax.grid(alpha=0.3)
    for b in range(nb, nrow * ncol):
        axes[b // ncol][b % ncol].axis('off')
    fig.suptitle(f'{tag}: gradient-flow quasi gluon TMD-PDF')
    fig.tight_layout()
    p = os.path.join(out_dir, 'quasi_tmd_pdf.png')
    fig.savefig(p, dpi=150); plt.close(fig); saved.append(p)

    # 匹配前后对比
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3.5 * nrow),
                             squeeze=False)
    for b in range(nb):
        ax = axes[b // ncol][b % ncol]
        ax.plot(x_grid, xg[:, b], 'o-', ms=3, label='quasi')
        ax.plot(x_grid, xgm[:, b], 's-', ms=3, label='NLO matched')
        ax.axhline(0, color='gray', lw=0.8)
        ax.set_xlabel('x'); ax.set_ylabel(r'$xg(x,b_\perp)$')
        ax.set_title(f'b={b_grid_fm[b]:.3f} fm')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    for b in range(nb, nrow * ncol):
        axes[b // ncol][b % ncol].axis('off')
    fig.suptitle(f'{tag}: NLO-matched gluon TMD-PDF')
    fig.tight_layout()
    p = os.path.join(out_dir, 'matched_tmd_pdf.png')
    fig.savefig(p, dpi=150); plt.close(fig); saved.append(p)

    # 所有 b 叠加
    fig, ax = plt.subplots(figsize=(8, 5))
    for b in range(nb):
        ax.plot(x_grid, xgm[:, b], '-', lw=1.2,
                label=f'b={b_grid_fm[b]:.3f} fm')
    ax.axhline(0, color='gray', lw=0.8)
    ax.set_xlabel('x'); ax.set_ylabel(r'$xg(x,b_\perp)$')
    ax.set_title(f'{tag}: gluon TMD-PDF vs b-perp (NLO matched)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, 'tmd_pdf_vs_b.png')
    fig.savefig(p, dpi=150); plt.close(fig); saved.append(p)

    # CS 核
    if cs_kernel is not None:
        K = np.asarray(cs_kernel, dtype=float).ravel()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(b_grid_fm[:len(K)], K, fmt='o-', capsize=3)
        ax.set_xlabel(r'$b_\perp$ [fm]')
        ax.set_ylabel(r'$K(b_\perp)$')
        ax.set_title('Collins-Soper kernel (two-momentum ratio)')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = os.path.join(out_dir, 'cs_kernel.png')
        fig.savefig(p, dpi=150); plt.close(fig); saved.append(p)

    logger(f"  TMD-PDF plots -> {out_dir} ({len(saved)} files)")
    return saved


def plot_tmd_ratio(ch_results, run_dir, momentum, logger=print,
                   nz=24, nb=1):
    """R(dt,dtau,z,b) 若干 (z,b) 点的比值曲线。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_dir = os.path.join(run_dir, 'analysis', 'tmd_ratio')
    for had_name, res in ch_results.items():
        ratio = res['ratio']     # (Nsample, dt, dtau, nz, nb)
        rm = ratio.mean(0); re_ = sem(ratio, True)
        zs = [0, max(0, nz // 4), max(0, nz // 2), max(0, nz - 1)]
        bs = list(range(min(nb, 2)))
        fig, axes = plt.subplots(len(bs), len(zs),
                                 figsize=(4 * len(zs), 3.5 * len(bs)),
                                 squeeze=False)
        for _b in range(len(bs)):
            for _z in range(len(zs)):
                zz, bb = zs[_z], bs[_b]
                ax = axes[_b][_z]
                for dt in [8, 10, 12, 14]:
                    if dt >= rm.shape[0]:
                        continue
                    tau = np.arange(dt + 1)
                    xv = tau - dt / 2
                    yv = rm[dt, :dt + 1, zz, bb]
                    ye = re_[dt, :dt + 1, zz, bb]
                    ax.errorbar(xv, yv, yerr=ye, fmt='x', capsize=0, label=f'dt={dt}')
                ax.set_xlabel('tau - t_sep/2'); ax.set_ylabel('R')
                ax.set_title(f'z={zz} b⊥={bb}')
                ax.legend(fontsize=7); ax.grid(alpha=0.3)
        fig.suptitle(f'{had_name} {momentum}: Disconnected TMD ratio')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'ratio_{had_name}_{momentum}.png'), dpi=150)
        plt.close(fig)
    logger(f"  TMD ratio plots saved to {out_dir}")