#!/public/home/huangcl/.venv/bin/python
import numpy as np
import matplotlib.pyplot as plt
import argparse
import gc
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ---- 从上级目录 98_tools 导入通用函数 ----
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "98_tools"))
from analysis_tools import (  # noqa: E402
    sem, resample, get_peak_memory_gb, DEFAULT_PLOT_COLORS,
)


# ===== 独立开关，方便调试时修改 =====
debug = False  # 在登录节点跑, 方便排除错误, 结果输出到 0_debug 文件夹
jack = False  # debug == False
# ===================================


# ===== 定义 dataclass =====

@dataclass
class SampleParams:
    conf_short: str
    conf_name: str
    conf_ids: list[int]
    Nt: int
    Nx: int
    P: int          # 动量大小 (正值)
    momP: int       # 动量符号, ±2
    Nsample: int
    dt_max: int
    # 需要平均的 dir 列表, 如 ["pos_x", "pos_y", "pos_z", "neg_x", "neg_y", "neg_z"]
    ave_dirs: list[str] = field(default_factory=lambda: [
                                "pos_x", "pos_y", "pos_z",
                                "neg_x", "neg_y", "neg_z"])

    @property
    def Nconf(self):
        return len(self.conf_ids)


@dataclass
class PlotParams:
    dt_list: list[int]
    z_list: list[int]
    xlim: list[float]
    ylim: list[float]
    dx_offset = 0.05
    colors: list[str] = field(default_factory=lambda: DEFAULT_PLOT_COLORS)


@dataclass
class OutputParams:
    """路径管理"""
    base_dir: str = "1_result"
    conf_short: str = ""
    P: int = 0

    @property
    def result_dir(self):
        return os.path.join(os.getcwd(), self.base_dir,
                            self.conf_short, f"P{self.P}")

    # def get_sub_dir(self, name):
    #     """返回子目录路径并创建"""
    #     d = os.path.join(self.result_dir, name)
    #     os.makedirs(d, exist_ok=True)
    #     return d

    # def sub_dir_path(self, name):
    #     """仅返回子目录路径, 不创建"""
    #     return os.path.join(self.result_dir, name)


# ===== 画图参数配置（按 |Pz| 索引）=====
# ylim 固定为 [0, 0.8], dt_list 只与动量绝对值有关
_plotpa_config = {
    # |Pz| : dt_list
    2: list(range(5, 14)),
    3: list(range(5, 13)),
    4: list(range(5, 12)),
    5: list(range(5, 11)),
    6: list(range(2, 10)),
}

# 固定参数
_fixed_z_list = list(range(0, 3))  # 前 3 个 z
_fixed_xlim = [-7, 7]
# =====================================


def init_config(conf_short: str):
    """初始化配置, 根据 conf_short 构造 dataclass 实例并创建输出目录

    输入:
        conf_short: 配置简称, 如 "L24x72"

    输出:
        sampa: SampleParams 实例
        plotpa: PlotParams 实例
        outpa: OutputParams 实例
    """
    if conf_short == "L24x72":
        # 12300 是空文件夹, 0 动量没有 14950
        _conf_ids = [x for x in range(
            4050, 48001, 50) if x not in (12300, 14950)]
        sampa = SampleParams(
            conf_short="L24x72",
            conf_name="beta6.20_mu-0.2770_ms-0.2400_L24x72",
            conf_ids=_conf_ids,
            Nt=72,
            Nx=24,
            P=6,
            momP=2,
            Nsample=3000,
            dt_max=20,
            ave_dirs=[
                "pos_x", "pos_y", "pos_z",
                "neg_x", "neg_y", "neg_z"
            ],
        )

        # 根据动量大小 P 构造画图参数 (ylim 固定 [0, 0.8])
        plotpa = PlotParams(
            dt_list=_plotpa_config[sampa.P],
            z_list=_fixed_z_list,
            xlim=_fixed_xlim,
            ylim=[0, 0.8],
        )

    else:
        print(f"conf {conf_short} not exist.")
        sys.exit()

    # 确保 dt_list 中的最大值不超过 dt_max - 1
    if plotpa.dt_list[-1] > sampa.dt_max - 1:
        print(
            f"Warning: dt_list max {plotpa.dt_list[-1]} exceeds dt_max-1={sampa.dt_max - 1}, truncating")
        plotpa.dt_list = [
            dt for dt in plotpa.dt_list if dt <= sampa.dt_max - 1]

    # debug / jack 调整
    if debug:
        print("debug")
        # 清空 debug 目录
        _debug_dir = os.path.join(os.getcwd(), "0_debug")
        if os.path.exists(_debug_dir):
            print(f"removing old debug dir: {_debug_dir}")
            shutil.rmtree(_debug_dir)
        sampa.conf_ids = sampa.conf_ids[:5]
        sampa.Nsample = sampa.Nconf + 5

    if jack:
        sampa.Nsample = sampa.Nconf

    # ===== OutputParams =====
    _base_dir = "0_debug" if debug else "1_result"
    outpa = OutputParams(
        base_dir=_base_dir,
        conf_short=conf_short,
        P=sampa.P,
    )
    # 只保存 ave ratio, 只创建 ave 目录
    os.makedirs(outpa.result_dir, exist_ok=True)
    # =========================

    return sampa, plotpa, outpa


########################################################################################
def load_ope(sampa: SampleParams, axis: str, jack: bool):
    """加载指定轴 (x/y/z) 的 OPE 数据, 返回原始 _ope

    输入:
        sampa: SampleParams 实例
        axis: 轴方向, "x"/"y"/"z"
        jack: 是否使用 jackknife

    输出:
        _ope: 原始 OPE 数据, shape (Nconf, Nt, Nx)
    """
    print("==================== load ope start ====================")
    print(f"axis = {axis}")

    # mu, nu = 0(x), 1(y), 2(z), 3(t)
    if axis == 'x':
        tdir1, tdir2 = 1, 2
    elif axis == 'y':
        tdir1, tdir2 = 2, 0
    elif axis == 'z':
        tdir1, tdir2 = 0, 1
    else:
        print(f'unknown axis: {axis}')
        sys.exit(1)

    _ope_ij = np.zeros((sampa.Nconf, sampa.Nx, sampa.Nt), dtype=complex)
    _ope_ti = np.zeros((sampa.Nconf, sampa.Nx, sampa.Nt), dtype=complex)
    _ope_tj = np.zeros((sampa.Nconf, sampa.Nx, sampa.Nt), dtype=complex)

    for i, conf_id in enumerate(sampa.conf_ids):
        _ope_ij_path = (
            f"/public/group/lqcd/donghx/Ope_Gluon/Result_hpy_4D_10times/"
            f"{sampa.conf_short}/{axis}dir/{conf_id}/"
            f"ops_mu{tdir1}_nu{tdir2}_dz{sampa.Nx}_conf{conf_id}.npz"
        )
        _ope_ti_path = (
            f"/public/group/lqcd/donghx/Ope_Gluon/Result_hpy_4D_10times/"
            f"{sampa.conf_short}/{axis}dir/{conf_id}/"
            f"ops_mu3_nu{tdir1}_dz{sampa.Nx}_conf{conf_id}.npz"
        )
        _ope_tj_path = (
            f"/public/group/lqcd/donghx/Ope_Gluon/Result_hpy_4D_10times/"
            f"{sampa.conf_short}/{axis}dir/{conf_id}/"
            f"ops_mu3_nu{tdir2}_dz{sampa.Nx}_conf{conf_id}.npz"
        )
        _ope_ij[i] = np.load(_ope_ij_path)["ops"]
        _ope_ti[i] = np.load(_ope_ti_path)["ops"]
        _ope_tj[i] = np.load(_ope_tj_path)["ops"]

    print("ope load finish")
    print(f"  example ope_ij:  {_ope_ij_path}")
    print(f"  example ope_ti:  {_ope_ti_path}")
    print(f"  example ope_tj:  {_ope_tj_path}")

    _ope = -_ope_ti - _ope_tj + 2 * _ope_ij
    print(f"the loaded _ope's shape: {_ope.shape}")
    _ope = _ope.transpose(0, 2, 1)  # (Nconf, tau, z)

    print("==================== load ope end ====================")
    return _ope


########################################################################################
def load_2pt(sampa: SampleParams, Px: int, Py: int, Pz: int, momP: int):
    """加载指定动量的 2pt 数据, 返回 _corr (Nconf, Nt, Nt)

    输入:
        sampa: SampleParams 实例
        Px, Py, Pz: 动量分量
        momP: 动量符号, ±2

    输出:
        _corr: 2pt 数据, shape (Nconf, Nt, Nt)
    """
    print("==================== load 2pt start ====================")
    print(f"P=({Px},{Py},{Pz}), momP={momP}")

    # 确定轴名 (用于 corr 文件路径)
    if Px != 0:
        _axis = 'x'
    elif Py != 0:
        _axis = 'y'
    elif Pz != 0:
        _axis = 'z'
    else:
        print('all P=0, invalid')
        sys.exit(1)

    _corr = np.zeros((sampa.Nconf, sampa.Nt, sampa.Nt), dtype=complex)

    for i, conf_id in enumerate(sampa.conf_ids):
        _corr_path = (
            f"/public/group/lqcd/donghx/2pt_Result/{sampa.conf_name}/"
            f"momsmear{momP}{_axis}/{conf_id}/"
            f"twopt_slice_pp_Px{Px}Py{Py}Pz{Pz}_eginphase{momP}_Cg5g4_nopol_ss_conf{conf_id}.npy"
        )
        _corr[i] = np.load(_corr_path)

    print("corr load finish")
    print(f"  example corr:    {_corr_path}")

    print("==================== load 2pt end ====================")
    return _corr


########################################################################################
def compute_3pt(sampa: SampleParams, _ope: np.ndarray, _corr: np.ndarray):
    """独立计算 3pt, 返回 Nconf 级别的原始数据 (未做时间平均, 未做 resample)

    输入:
        sampa: SampleParams 实例
        _ope: OPE 数据, para: conf, tau, z
        _corr: 2pt 数据, para: conf, tf, ti

    输出:
        corr2_rel: 2pt 数据, para: conf, ti(loop), dt
        ope_rel: OPE 数据, para: conf, ti(loop), dtau, z
        corr3: 3pt 数据, para: conf, ti(loop), dt, dtau, z
    """
    print("==================== compute 3pt start ====================")

    _corr2_rel = np.zeros(
        (sampa.Nconf, sampa.Nt, sampa.dt_max), dtype=complex
    )  # para: conf, ti(loop), dt
    _ope_rel = np.zeros(
        (sampa.Nconf, sampa.Nt, sampa.dt_max, sampa.Nx), dtype=complex
    )  # para: conf, ti(loop), dtau, z

    # loop ti
    for ti in range(sampa.Nt):
        corr2_shift = np.roll(_corr[:, :, ti], shift=-ti, axis=1)
        _corr2_rel[:, ti, :] = corr2_shift[:, :sampa.dt_max]

        ope_shift = np.roll(_ope, shift=-ti, axis=1)
        _ope_rel[:, ti, :, :] = ope_shift[:, :sampa.dt_max, :]

    _corr3 = np.zeros(
        (sampa.Nconf, sampa.Nt, sampa.dt_max,
         sampa.dt_max, sampa.Nx), dtype=complex
    )  # para: conf, ti(loop), dt, dtau, z
    for _dt in range(sampa.dt_max):
        for _dtau in range(_dt + 1):
            c2_slice = _corr2_rel[:, :, _dt]  # para: conf, ti(loop)
            ope_slice = _ope_rel[:, :, _dtau, :]  # para: conf, ti(loop), z
            _corr3[:, :, _dt, _dtau, :] = ope_slice * \
                c2_slice[:, :, np.newaxis]

    print("loop ti finish")

    print("corr2_rel shape:", _corr2_rel.shape)
    print("ope_rel shape:", _ope_rel.shape)
    print("corr3 shape:", _corr3.shape)
    print("==================== compute 3pt end ====================")
    return _corr2_rel, _ope_rel, _corr3


########################################################################################
def plot_ratio(ratio: np.ndarray, plotpa: PlotParams,
               save_dir: str, jack: bool, dir_name: str,
               Nconf: int, Nsample: int,
               Px: int = 0, Py: int = 0, Pz: int = 0):
    """绘制 ratio 大图，每张图画一个 z

    不同 tsep 在横轴上的偏移量由 plotpa.dx_offset 控制：
    - 中间的 tsep 不偏移
    - 较小的 tsep 向左偏移（负方向）
    - 较大的 tsep 向右偏移（正方向）
    - 相邻两个 tsep 的偏移量差值为 plotpa.dx_offset
    - 设为 0 则无偏移

    输入:
        ratio: ratio 数据, shape (Nsample, dt_max, dt_max, Nx)
        plotpa: PlotParams 实例
        save_dir: 图片保存目录
        jack: 是否使用 jackknife
        dir_name: 方向名称, 用于标题和文件名
        Nconf: 组态数
        Nsample: 样本数
        Px, Py, Pz: 动量分量
    """
    print(
        f"==================== plot ratio ({dir_name}) start ====================")

    ratio_mean = ratio.mean(0)  # para: dt, dtau, z
    ratio_err = sem(ratio, jack)

    # 计算每个 tsep 的横轴偏移量
    n_dt = len(plotpa.dt_list)
    mid_idx = (n_dt - 1) / 2.0  # 中间索引（可能为半整数）

    for _z in plotpa.z_list:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

        for i, _dt in enumerate(plotpa.dt_list):
            tau_vals = np.arange(0, _dt + 1)

            # 横坐标: tau - dt / 2 + 偏移量
            # 中间 tsep 偏移为 0，小的左偏（负），大的右偏（正）
            offset_i = (i - mid_idx) * plotpa.dx_offset
            x_vals = tau_vals - _dt / 2.0 + offset_i

            y_vals = ratio_mean[_dt, tau_vals, _z]
            y_errs = ratio_err[_dt, tau_vals, _z]

            color = plotpa.colors[i % len(plotpa.colors)]
            ax.errorbar(
                x_vals, y_vals, yerr=y_errs,
                fmt="x", color=color, ecolor=color,
                capsize=0, markersize=7, markeredgewidth=1.8,
                linewidth=1.2, zorder=3, label=f"tsep={_dt}",
            )

        ax.set_xlim(plotpa.xlim[0], plotpa.xlim[1])
        ax.set_ylim(plotpa.ylim[0], plotpa.ylim[1])
        ax.set_xlabel(r"$t_{\mathrm{ins}} - t_{\mathrm{sep}}/2$", fontsize=16)
        ax.set_ylabel(r"$C_3 / C_2$", fontsize=16)
        ax.set_title(
            f"dir={dir_name}, z={_z}, "
            f"P({Px},{Py},{Pz}), Nconf={Nconf}, Nsample={Nsample}",
            fontsize=13,
        )
        ax.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"ratio_{dir_name}_z{_z}.png")
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  saved: {save_path}")
        plt.close(fig)

    print(
        f"==================== plot ratio ({dir_name}) end ====================")


if __name__ == "__main__":

    # ===== 解析命令行参数 =====
    parser = argparse.ArgumentParser(
        description="Gluon unpolarized PDF ratio calculation and plot")
    parser.add_argument("-c", type=str, default="L24x72",
                        help="conf_short, e.g. L24x72 (default: L24x72)")
    parser.add_argument("-s", type=int, default=1,
                        choices=[1, 2], help="start part: 1=ratio, 2=plot (default: 1)")
    parser.add_argument("-e", type=int, default=2,
                        choices=[1, 2], help="end part: 1=ratio, 2=plot (default: 2)")
    args = parser.parse_args()
    conf_short = args.c
    part_start = args.s
    part_end = args.e
    # ==========================

    # ===== 初始化配置 =====
    sampa, plotpa, outpa = init_config(conf_short)
    # ======================

    print("jackknife:", jack)
    print("Nconf:", sampa.Nconf)
    print("Nsample:", sampa.Nsample)
    print("conf_short:", sampa.conf_short)
    print("result base:", outpa.result_dir)

    # ---- Part 1: compute ratio (按轴循环, 每个轴算一次 OPE + 正负方向 3pt,
    #      每个方向算完 3pt 后先做时间平均, 6 个方向全部算完后一次做方向平均,
    #      再 resample, 最后算 ratio, 只保存 ave ratio) ----
    if part_start <= 1:
        time0 = time.perf_counter()

        # 预分配 6 个方向、时间平均后的 Nconf 级别数组
        # (corr3 未做时间平均时内存过大, 故时间平均在每个方向算完后立即做)
        # para: dir, conf, dt
        corr2_all = np.zeros((6, sampa.Nconf, sampa.dt_max), dtype=complex)
        # para: dir, conf, dtau, z
        ope_all = np.zeros(
            (6, sampa.Nconf, sampa.dt_max, sampa.Nx), dtype=complex)
        # para: dir, conf, dt, dtau, z
        corr3_all = np.zeros(
            (6, sampa.Nconf, sampa.dt_max, sampa.dt_max, sampa.Nx),
            dtype=complex)

        _dir_idx = 0  # 方向索引, 0..5
        for _axis in ["x", "y", "z"]:
            # 1. 加载该轴的 OPE (只算一次, 正负方向共享)
            _ope = load_ope(sampa, _axis, jack)

            # 2. 正方向: Px,Py,Pz = +P, momP = +momP
            if _axis == 'x':
                _Px_pos, _Py_pos, _Pz_pos = sampa.P, 0, 0
                _Px_neg, _Py_neg, _Pz_neg = -sampa.P, 0, 0
            elif _axis == 'y':
                _Px_pos, _Py_pos, _Pz_pos = 0, sampa.P, 0
                _Px_neg, _Py_neg, _Pz_neg = 0, -sampa.P, 0
            else:  # 'z'
                _Px_pos, _Py_pos, _Pz_pos = 0, 0, sampa.P
                _Px_neg, _Py_neg, _Pz_neg = 0, 0, -sampa.P

            # pos 方向: 独立函数算 3pt -> 时间平均 (ti 平均) -> 存入预分配数组
            _corr_pos = load_2pt(sampa, _Px_pos, _Py_pos, _Pz_pos, sampa.momP)
            corr2_rel, ope_rel, corr3 = compute_3pt(sampa, _ope, _corr_pos)
            corr2_all[_dir_idx] = corr2_rel.mean(axis=1)  # para: conf, dt
            ope_all[_dir_idx] = ope_rel.mean(axis=1)      # para: conf, dtau, z
            corr3_all[_dir_idx] = corr3.mean(
                axis=1)      # para: conf, dt, dtau, z
            _dir_idx += 1

            # neg 方向: 独立函数算 3pt -> 时间平均 (ti 平均) -> 存入预分配数组
            _corr_neg = load_2pt(sampa, _Px_neg, _Py_neg, _Pz_neg, -sampa.momP)
            corr2_rel, ope_rel, corr3 = compute_3pt(sampa, _ope, _corr_neg)
            corr2_all[_dir_idx] = corr2_rel.mean(axis=1)  # para: conf, dt
            ope_all[_dir_idx] = ope_rel.mean(axis=1)      # para: conf, dtau, z
            corr3_all[_dir_idx] = corr3.mean(
                axis=1)      # para: conf, dt, dtau, z
            _dir_idx += 1

            time1 = time.perf_counter()
            print(f"Peak Memory: {get_peak_memory_gb():.3f} GB")
            print(f"spend time: {(time1 - time0):.2f}s\n")

        # 循环结束后统一回收内存
        del _ope, _corr_pos, _corr_neg, corr2_rel, ope_rel, corr3
        gc.collect()

        # 3. 6 个方向全部算完后, 一次做方向平均 (Nconf 级别), 并立刻取实部
        print("===== average corr2/ope/corr3 over 6 dirs =====")
        corr2_ave = corr2_all.mean(axis=0).real  # para: conf, dt
        ope_ave = ope_all.mean(axis=0).real      # para: conf, dtau, z
        corr3_ave = corr3_all.mean(axis=0).real  # para: conf, dt, dtau, z
        del corr2_all, ope_all, corr3_all
        gc.collect()
        print(f"corr2_ave shape: {corr2_ave.shape}")
        print(f"ope_ave shape: {ope_ave.shape}")
        print(f"corr3_ave shape: {corr3_ave.shape}")

        # 4. resample
        print("===== resample =====")
        corr2 = resample(corr2_ave, sampa.Nsample, jack)  # para: sample, dt
        # para: sample, dtau, z
        ope = resample(ope_ave, sampa.Nsample, jack)
        del corr2_ave, ope_ave
        gc.collect()
        # para: sample, dt, dtau, z
        corr3 = resample(corr3_ave, sampa.Nsample, jack)
        del corr3_ave
        gc.collect()
        print(f"corr2 shape: {corr2.shape}")
        print(f"ope shape: {ope.shape}")
        print(f"corr3 shape: {corr3.shape}")

        # 5. resample 后再算 ratio (disconnected 部分: <C3>/<C2> - <ope>)
        ratio_ave = corr3 / corr2[:, :, np.newaxis,
                                  np.newaxis] - ope[:, np.newaxis, :, :]  # para: sample, dt, dtau, z
        del corr2, ope, corr3
        gc.collect()

        # 6. 保存 ave ratio
        np.save(os.path.join(outpa.result_dir, "ratio.npy"), ratio_ave)
        print("ratio (ave) saved")
        del ratio_ave
        gc.collect()

        if part_end == 1:
            print("job finish")
            sys.exit(0)
    else:
        print("===== skip compute ratio, loading ratio from file =====")

    # ---- Part 2: plot (只画 ave ratio) ----
    if part_start <= 2:
        time0 = time.perf_counter()

        # 加载 ave ratio
        load_path = os.path.join(outpa.result_dir, "ratio.npy")
        print(f"===== loading ratio (ave) from {load_path} =====")
        ratio_ave = np.load(load_path)
        print(f"  loaded, shape: {ratio_ave.shape}")

        # 画 ave 图
        plot_ratio(ratio_ave, plotpa, outpa.result_dir, jack, "ave",
                   Nconf=sampa.Nconf, Nsample=sampa.Nsample,
                   Px=0, Py=0, Pz=sampa.P)

        time1 = time.perf_counter()
        print(f"spend time: {(time1 - time0):.2f}s\n")
    print("job finish")
