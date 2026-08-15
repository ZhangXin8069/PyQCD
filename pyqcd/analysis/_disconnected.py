"""
不相连胶子 ratio 分析（code_1.py 算法移植，自包含）
=====================================================

移植 examples/docker-v20260805/analyze.py 的 code_1.py 风格分析：

    C3(dt, dtau, z) = C2(dt) · OPE(dtau, z)         （不相连因子化）
    C3_disc = C3 − C2·⟨OPE⟩                         （真空扣除）
    R(dt, dtau, z) = ⟨C3_disc / C2⟩_ti
    逐 z 相关拟合：R = c0 + c1·e^{−dE·dtau} + c1·e^{−dE·(dt−dtau)}
    （lsqfit.nonlinear_fit，svdcut=1e-6，jackknife 协方差）

输出：ratio_{ch}.npy、(c0, c1, dE, chi2) 参数、拟合报告。
"""
from __future__ import annotations

import os
import time

import numpy as np


def sem(data, jackknife=True):
    """样本轴（axis 0）均值的标准误。"""
    error = data.std(0)
    if jackknife:
        error = error * np.sqrt(data.shape[0] - 1)
    return error


def resample(corr, jackknife=True, Nsample=None, seed=0):
    """Delete-one jackknife（或 bootstrap）重采样：样本轴移到 axis 0。"""
    n_conf = corr.shape[0]
    if jackknife:
        return (n_conf * corr.mean(0) - corr) / (n_conf - 1)
    rng = np.random.default_rng(seed=seed)
    idx = rng.integers(0, n_conf, size=(Nsample, n_conf))
    return corr[idx].mean(1)


def cov_mat(arr, jackknife=True):
    """Jackknife 协方差（均值）与特征值条件数。"""
    diff = arr - arr.mean(0)
    n = arr.shape[0]
    if jackknife:
        cov = np.matmul(diff.T, diff) / n * (n - 1)
    else:
        cov = np.matmul(diff.T, diff) / n
    eig = np.linalg.eigvalsh(cov)
    cond = eig[-1] / eig[0] if eig[0] > 0 else np.inf
    return cov, cond


def model_ratio(x, p):
    """R(dt, dtau) = c0 + c1·e^{−dE·dtau} + c1·e^{−dE·(dt−dtau)}。"""
    dt = np.array([_x[0] for _x in x])
    dtau = np.array([_x[1] for _x in x])
    return (np.ones(len(x)) * p["c0"]
            + p["c1"] * np.exp(-p["dE"] * dtau)
            + p["c1"] * np.exp(-p["dE"] * (dt - dtau)))


def run_disconnected_ratio(corr_2pt_all, ope_all, conf_ids, run_dir, logger=print,
                           NT=72, NX=24, dt_max=20, dt_start=7, dt_end=10,
                           cut=6, p0=None, target_momentum='P2'):
    """不相连胶子 ratio + 逐 z 拟合（code_1.py 算法）。

    Args:
        corr_2pt_all: {conf_id: {'corr_pp_P2': (Nt,), 'corr_pion_P2': (Nt,), ...}}
        ope_all:      {conf_id: {'combined': (Nz, Nt)}}
        conf_ids:     组态列表
        run_dir:      输出目录（analysis/disconnected/）
        dt_max/dt_start/dt_end/cut: 拟合窗参数
        p0:           拟合初值（默认 {'c0':0.6,'c1':-2,'dE':1}）
        target_momentum: 'P2'
    Returns:
        ch_results: {hadron: {'ratio': ..., 'c0': ..., 'c1': ..., 'dE': ..., 'chi2': ...}}
    """
    import lsqfit
    import gvar as gv

    if p0 is None:
        p0 = {"c0": 0.6, "c1": -2, "dE": 1}

    Nconf = len(conf_ids)
    Nsample = Nconf
    jack = True
    out_dir = os.path.join(run_dir, 'analysis', 'disconnected')
    os.makedirs(out_dir, exist_ok=True)

    channels = [('proton', 'corr_pp', 'proton'), ('pion', 'corr_pion', 'pion')]
    ch_results = {}

    for ch_key, k2, had_name in channels:
        logger(f"\n  Channel: {had_name} at Pz=2")

        # 平移不变 2pt：C(t_sink, t_src) = C((t_sink − t_src) mod Nt)
        key2 = f'{k2}_P{target_momentum[-1]}'
        _corr = np.stack([np.real(corr_2pt_all[cid][key2]) for cid in conf_ids])
        full = np.zeros((Nconf, NT, NT), dtype=np.float64)
        for ti in range(NT):
            full[:, :, ti] = np.roll(_corr, -ti, axis=1)

        # OPE combined：(Nconf, Nz, Nt) → (Nconf, tau, z)
        _ope = np.stack([np.real(ope_all[cid]['combined']) for cid in conf_ids])
        _ope = _ope.transpose(0, 2, 1)

        # 相对时间构造
        _corr2_rel = np.zeros((Nconf, NT, dt_max), dtype=np.float64)
        _ope_rel = np.zeros((Nconf, NT, dt_max, NX), dtype=np.float64)
        for ti in range(NT):
            corr2_shift = np.roll(full[:, :, ti], -ti, axis=1)
            _corr2_rel[:, ti, :] = corr2_shift[:, :dt_max]
            ope_shift = np.roll(_ope, -ti, axis=1)
            _ope_rel[:, ti, :, :] = ope_shift[:, :dt_max, :]

        # 不相连 3pt = C2 × OPE（因子化）
        _corr3 = np.zeros((Nconf, NT, dt_max, dt_max, NX), dtype=np.float64)
        for _dt in range(dt_max):
            for _dtau in range(_dt + 1):
                _corr3[:, :, _dt, _dtau, :] = (
                    _ope_rel[:, :, _dtau, :] * _corr2_rel[:, :, _dt][:, :, None])

        corr2 = resample(_corr2_rel, jack, Nsample)
        ope = resample(_ope_rel, jack, Nsample)
        corr3 = resample(_corr3, jack, Nsample)

        # 真空扣除 + ratio
        corr3_disc = corr3 - corr2[:, :, :, None, None] * ope[:, :, None, :, :]
        eps = 1e-30
        ratio = np.mean(corr3_disc / (corr2[:, :, :, None, None] + eps), axis=1)
        ratio = ratio.real   # (Nsample, dt_max, dt_max, Nz)
        np.save(os.path.join(out_dir, f'ratio_{had_name}_P{target_momentum[-1]}.npy'),
                ratio)

        # 逐 z 相关拟合
        front_remove = cut // 2
        back_remove = cut - front_remove
        x_coor = [(dt, dtau)
                  for dt in range(dt_start, dt_end + 1)
                  for dtau in range(front_remove, dt - back_remove + 1)]
        Ndata = len(x_coor)

        para_c0 = np.zeros((Nsample, NX))
        para_c1 = np.zeros((Nsample, NX))
        para_dE = np.zeros((Nsample, NX))
        chi2 = np.zeros((Nsample, NX))

        report_lines = [
            "=" * 70,
            f"  Fit Report: {had_name}, Pz={target_momentum[-1]}, Nconf={Nconf}",
            "=" * 70,
            f"  t_sep range : [{dt_start}, {dt_end}]",
            f"  cut         : {cut}",
            f"  Nsample     : {Nsample}",
            f"  jackknife   : {jack}",
            "=" * 70, "",
        ]

        for _z in range(NX):
            sub_sample = np.zeros((Nsample, Ndata))
            for i, (dt, dtau) in enumerate(x_coor):
                sub_sample[:, i] = ratio[:, dt, dtau, _z]
            cov, cond = cov_mat(sub_sample, jack)
            report_lines += [f"z = {_z}", "-" * 56,
                             f"condition number = {cond:.3g}", ""]

            for _id in range(Nsample):
                y_coor = gv.gvar(sub_sample[_id], cov)
                _fit = lsqfit.nonlinear_fit(data=(x_coor, y_coor), p0=p0,
                                            fcn=model_ratio, svdcut=1e-6)
                para_c0[_id, _z] = _fit.pmean["c0"]
                para_c1[_id, _z] = _fit.pmean["c1"]
                para_dE[_id, _z] = _fit.pmean["dE"]
                chi2[_id, _z] = _fit.chi2 / _fit.dof
            if _id == Nsample - 1:
                report_lines.append(_fit.format(maxline=True))

            report_lines += ["", ""]

        report = "\n".join(report_lines)
        with open(os.path.join(out_dir, '1_fit_report.txt'), 'w') as f:
            f.write(report)
        np.savez(os.path.join(out_dir, '0_fit_data.npz'),
                 c0=para_c0, c1=para_c1, dE=para_dE, chi2=chi2)
        logger(f"  Saved ratio + fit to {out_dir}")

        ch_results[had_name] = {
            'ratio': ratio, 'c0': para_c0, 'c1': para_c1,
            'dE': para_dE, 'chi2': chi2,
        }

    # ── 绘图（code_1.py 风格：ratio/c0/chi2，与 analyze.py 一致）──
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for had_name, res in ch_results.items():
        _plot_disconnected(had_name, res, out_dir, logger)
    return ch_results


def _plot_disconnected(had_name, res, out_dir, logger=print):
    """code_1.py 风格图：ratio.png（逐 z）、c0.png、chi2.png。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ratio = res['ratio']              # (Nsample, dt, dtau, z)
    para_c0, para_c1 = res['c0'], res['c1']
    chi2 = res['chi2']
    rm = ratio.mean(0); re_ = sem(ratio, True)

    # c0 vs z
    z_list = list(range(rm.shape[-1]))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(z_list, para_c0.mean(0), yerr=sem(para_c0, True), fmt='x-',
                label='c0(z)')
    ax.axhline(0, color='gray', lw=0.8)
    ax.set_xlabel('z')
    ax.set_ylabel('c0')
    ax.set_title(f'{had_name}: c0 vs z (disconnected ratio fit)')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'c0_{had_name}.png'), dpi=150)
    plt.close(fig)

    # chi2/dof vs z
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(z_list, chi2.mean(0), s=30)
    ax.axhline(1.0, color='orange', ls='--')
    ax.set_xlabel('z'); ax.set_ylabel('chi2/dof'); ax.set_ylim(0, 2)
    ax.set_title(f'{had_name}: chi2/dof vs z')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'chi2_{had_name}.png'), dpi=150)
    plt.close(fig)

    # ratio(dt,dtau,z) 若干 z
    zs = [0, 6, 12, 18]
    zs = [z for z in zs if z < rm.shape[-1]]
    nrow = (len(zs) + 1) // 2
    fig, axes = plt.subplots(nrow, 2, figsize=(12, 4 * nrow), squeeze=False)
    for k, z in enumerate(zs):
        ax = axes[k // 2][k % 2]
        for dt in [8, 10, 12, 14]:
            if dt >= rm.shape[0]:
                continue
            tau = np.arange(dt + 1)
            xv = tau - dt / 2
            yv = rm[dt, :dt + 1, z]
            ye = re_[dt, :dt + 1, z]
            ax.errorbar(xv, yv, yerr=ye, fmt='x', capsize=0, label=f'dt={dt}')
        ax.set_xlabel('tau - t_sep/2'); ax.set_ylabel('R')
        ax.set_title(f'z={z}, c0={para_c0[:, z].mean():.3f}')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f'{had_name}: Disconnected ratio R(dt,dtau,z), Pz=2')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'ratio_{had_name}.png'), dpi=150)
    plt.close(fig)
    logger(f"  Plots saved to {out_dir}")
