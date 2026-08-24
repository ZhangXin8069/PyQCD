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
    # 需要平均的 dir 列表 (与参考一致: 只对正方向 x, y, z 做平均)
    ave_dirs: list[str] = field(default_factory=lambda: ["x", "y", "z"])

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

    def get_sub_dir(self, name):
        """返回子目录路径并创建"""
        d = os.path.join(self.result_dir, name)
        os.makedirs(d, exist_ok=True)
        return d

    def sub_dir_path(self, name):
        """仅返回子目录路径, 不创建"""
        return os.path.join(self.result_dir, name)


# ===== 从命令行参数读取 conf_short =====
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
# =========================================


# ===== 画图参数配置（按 |Pz| 索引）=====
# ylim 固定为 [0, 0.8], dt_list 只与动量绝对值有关
_plotpa_config = {
    # |Pz| : dt_list
    2: list(range(5, 14)),
    3: list(range(5, 13)),
    4: list(range(5, 12)),
    5: list(range(5, 11)),
    6: list(range(5, 10)),
}

# 固定参数
_fixed_z_list = list(range(0, 10))  # 前 10 个 z
_fixed_xlim = [-7, 7]

# =====================================


if conf_short == "L24x72":
    # 12300 是空文件夹
    _conf_ids = [x for x in range(4050, 48001, 50) if x not in (12300, 14950)]
    sampa = SampleParams(
        conf_short="L24x72",
        conf_name="beta6.20_mu-0.2770_ms-0.2400_L24x72",
        conf_ids=_conf_ids,
        Nt=72,
        Nx=24,
        P=4,
        momP=2,
        Nsample=3000,
        dt_max=20,
        ave_dirs=["x", "y", "z"],
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
    plotpa.dt_list = [dt for dt in plotpa.dt_list if dt <= sampa.dt_max - 1]


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
# 创建 ratio 子目录 (存放方向平均后的 ratio 数据和图片)
outpa.get_sub_dir("ratio")
# =========================


########################################################################################
def compute_ope(sampa: SampleParams, axis: str, jack: bool):
    """加载指定轴 (x/y/z) 的预组合 OPE 数据 (ops_dz*.npz, 已做好 -O_ti-O_tj+2*O_ij 组合)

    返回:
        _ope:    (Nconf, Nt, Nx), 中心化后, 未 resample
        ope_raw: (Nsample, Nt, Nx), resample 后
    """
    print("==================== compute ope start ====================")
    print(f"axis = {axis}")

    _ope = np.zeros((sampa.Nconf, sampa.Nx, sampa.Nt), dtype=complex)

    for i, conf_id in enumerate(sampa.conf_ids):
        _ope_path = (
            f"/public/group/imp/zengch/LQCD/gluon_operator/output/"
            f"{sampa.conf_short}/{axis}dir/{conf_id}/"
            f"ops_dz{sampa.Nx}_conf{conf_id}.npz"
        )
        _ope[i] = np.load(_ope_path)["ops"]  # (z, tau)

    print("ope load finish")
    print(f"  example ope: {_ope_path}")
    print(f"  loaded _ope shape: {_ope.shape}")

    _ope = _ope.transpose(0, 2, 1)  # (Nconf, tau, z)

    # ---- 在组态层面中心化（减真空期望），期望与 tau 和 z 有关 ----
    _ope_mean = _ope.mean(0)  # (1, Nt, Nx)
    _ope = _ope - _ope_mean[np.newaxis]  # (Nconf, Nt, Nx)，中心化后的 OPE
    # ---------------------------------------------------------

    ope_raw = resample(_ope, sampa.Nsample, jack)  # (Nsample, Nt, Nx)

    print("==================== compute ope end ====================")
    return _ope, ope_raw


########################################################################################
def resample_and_ratio(corr2_avg: np.ndarray, corr2_avg_orig: np.ndarray,
                       ope_avg: np.ndarray, corr3_avg: np.ndarray,
                       sampa: SampleParams, jack: bool):
    """对 Nconf 级别的平均数据做 resample，然后计算 ratio

    参数:
        corr2_avg:      (Nconf, dt) 中心化后的 2pt，对 ti 平均
        corr2_avg_orig: (Nconf, dt) 原始 2pt，对 ti 平均（用于分母）
        ope_avg:        (Nconf, dtau, z) 中心化后的 OPE，对 ti 平均
        corr3_avg:      (Nconf, dt, dtau, z) 连接部分 3pt，对 ti 平均
        sampa:          SampleParams
        jack:           bool, 是否使用 jackknife

    返回:
        ratio: (Nsample, dt, dtau, z)
    """
    print("==================== resample and ratio start ====================")

    corr2 = resample(corr2_avg, sampa.Nsample, jack)        # (Nsample, dt)
    corr2_orig = resample(corr2_avg_orig, sampa.Nsample, jack)  # (Nsample, dt)
    # (Nsample, dtau, z)
    ope = resample(ope_avg, sampa.Nsample, jack)
    # (Nsample, dt, dtau, z)
    corr3 = resample(corr3_avg, sampa.Nsample, jack)

    print("start to compute ratio")
    # ratio: (Nsample, dt, dtau, z)
    ratio = corr3 / corr2_orig[:, :, np.newaxis, np.newaxis]

    print("ratio shape:", ratio.shape)
    print("==================== resample and ratio end ====================")
    return ratio, ope


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

    print("jackknife:", jack)
    print("Nconf:", sampa.Nconf)
    print("Nsample:", sampa.Nsample)
    print("conf_short:", sampa.conf_short)
    print("result base:", outpa.result_dir)

    # ---- Part 1: compute ratio ----
    #     流程: 对每个轴, 先对正负方向 2pt 做平均 → 用平均后的 2pt 算 3pt → 对 xyz 做平均 → resample + ratio
    if part_start <= 1:
        time0 = time.perf_counter()

        # 1. 对每个轴计算 OPE (中心化后的 Nconf 级别数据)
        _ope_by_axis = {}
        for _axis in ["x", "y", "z"]:
            _ope, _ = compute_ope(sampa, _axis, jack)
            _ope_by_axis[_axis] = _ope  # (Nconf, Nt, Nx), 中心化后, 未 resample
            print(f"  _ope ({_axis}) shape: {_ope.shape}")

        # 2. 对每个轴, 先对正负方向 2pt 做平均, 再用平均后的 2pt 算 3pt
        _axis_config = {
            "x": (sampa.P, 0, 0,  sampa.momP, -sampa.momP),
            "y": (0, sampa.P, 0,  sampa.momP, -sampa.momP),
            "z": (0, 0, sampa.P,  sampa.momP, -sampa.momP),
        }

        # key: axis, value: (corr2_avg, corr2_avg_orig, ope_avg, corr3_avg)
        _nconf_data = {}
        for _axis, (_Px, _Py, _Pz, _momP_pos, _momP_neg) in _axis_config.items():
            _ope = _ope_by_axis[_axis]

            # 加载正负方向的 2pt 原始数据 (Nconf, Nt, Nt)
            _corr_pos = np.zeros(
                (sampa.Nconf, sampa.Nt, sampa.Nt), dtype=complex)
            _corr_neg = np.zeros(
                (sampa.Nconf, sampa.Nt, sampa.Nt), dtype=complex)
            for i, conf_id in enumerate(sampa.conf_ids):
                _corr_pos_path = (
                    f"/public/group/lqcd/donghx/2pt_Result/{sampa.conf_name}/"
                    f"momsmear{_momP_pos}{_axis}/{conf_id}/"
                    f"twopt_slice_pp_Px{_Px}Py{_Py}Pz{_Pz}_eginphase{_momP_pos}_Cg5g4_nopol_ss_conf{conf_id}.npy"
                )
                _corr_neg_path = (
                    f"/public/group/lqcd/donghx/2pt_Result/{sampa.conf_name}/"
                    f"momsmear{_momP_neg}{_axis}/{conf_id}/"
                    f"twopt_slice_pp_Px{-_Px}Py{-_Py}Pz{-_Pz}_eginphase{_momP_neg}_Cg5g4_nopol_ss_conf{conf_id}.npy"
                )
                _corr_pos[i] = np.load(_corr_pos_path)
                _corr_neg[i] = np.load(_corr_neg_path)

            # 对正负方向 2pt 做平均 (在原始数据层面)
            _corr_ave = (_corr_pos + _corr_neg) / 2.0  # (Nconf, Nt, Nt)
            print(f"  {_axis}: pos+neg 2pt averaged")

            # for i, conf_id in enumerate(sampa.conf_ids):
            #     # _corr_pos_path = (
            #     #     f"/public/group/lqcd/donghx/2pt_Result/{sampa.conf_name}/"
            #     #     f"momsmear{_momP_pos}{_axis}/{conf_id}/"
            #     #     f"twopt_slice_pp_Px{_Px}Py{_Py}Pz{_Pz}_eginphase{_momP_pos}_Cg5g4_nopol_ss_conf{conf_id}.npy"
            #     # )
            #     _corr_neg_path = (
            #         f"/public/group/lqcd/donghx/2pt_Result/{sampa.conf_name}/"
            #         f"momsmear{_momP_neg}{_axis}/{conf_id}/"
            #         f"twopt_slice_pp_Px{-_Px}Py{-_Py}Pz{-_Pz}_eginphase{_momP_neg}_Cg5g4_nopol_ss_conf{conf_id}.npy"
            #     )
            #     # _corr_pos[i] = np.load(_corr_pos_path)
            #     _corr_neg[i] = np.load(_corr_neg_path)

            # 对正负方向 2pt 做平均 (在原始数据层面)
            # _corr_ave = _corr_neg  # (Nconf, Nt, Nt)
            # print(f"  {_axis}: pos+neg 2pt averaged")

            # 中心化
            _corr_orig = _corr_ave.copy()
            _corr = _corr_ave.transpose(0, 2, 1)  # (conf, ti, tf)
            _corr_mean = _corr.mean(0)
            _corr = _corr - _corr_mean[np.newaxis]
            _corr_orig = _corr_orig.transpose(0, 2, 1)

            # ti-loop
            _corr2_rel = np.zeros(
                (sampa.Nconf, sampa.Nt, sampa.dt_max), dtype=complex)
            _corr2_rel_orig = np.zeros(
                (sampa.Nconf, sampa.Nt, sampa.dt_max), dtype=complex)
            _ope_rel = np.zeros(
                (sampa.Nconf, sampa.Nt, sampa.dt_max, sampa.Nx), dtype=complex)
            for ti in range(sampa.Nt):
                corr2_shift = np.roll(_corr[:, ti], shift=-ti, axis=1)
                _corr2_rel[:, ti] = corr2_shift[:, :sampa.dt_max]
                corr2_shift_orig = np.roll(
                    _corr_orig[:, ti], shift=-ti, axis=1)
                _corr2_rel_orig[:, ti] = corr2_shift_orig[:, :sampa.dt_max]
                ope_shift = np.roll(_ope, shift=-ti, axis=1)
                _ope_rel[:, ti] = ope_shift[:, :sampa.dt_max]

            # corr3
            _corr3 = np.zeros(
                (sampa.Nconf, sampa.Nt, sampa.dt_max,
                 sampa.dt_max, sampa.Nx), dtype=complex
            )
            for _dt in range(sampa.dt_max):
                for _dtau in range(_dt + 1):
                    c2_slice = _corr2_rel[:, :, _dt]
                    ope_slice = _ope_rel[:, :, _dtau, :]
                    _corr3[:, :, _dt, _dtau, :] = ope_slice * \
                        c2_slice[:, :, np.newaxis]

            del _corr, _corr_orig, _corr_pos, _corr_neg
            gc.collect()

            # 对 ti 求平均
            corr2_avg = _corr2_rel.mean(axis=1).real  # (Nconf, dt)
            corr2_avg_orig = _corr2_rel_orig.mean(axis=1).real  # (Nconf, dt)
            ope_avg = _ope_rel.mean(axis=1).real  # (Nconf, dtau, z)
            corr3_avg = _corr3.mean(axis=1).real  # (Nconf, dt, dtau, z)

            del _corr2_rel, _corr2_rel_orig, _ope_rel, _corr3
            gc.collect()

            _nconf_data[_axis] = (
                corr2_avg, corr2_avg_orig, ope_avg, corr3_avg)
            print(f"  {_axis}: nconf data computed (2pt pos+neg averaged first)")

        # 3. 对 xyz 做平均
        print("===== averaging over xyz directions =====")
        _dirs = ["x", "y", "z"]
        corr2_avg_ave = np.mean([_nconf_data[d][0] for d in _dirs], axis=0)
        corr2_avg_orig_ave = np.mean(
            [_nconf_data[d][1] for d in _dirs], axis=0)
        # OPE 不参与方向平均, 直接用 x 方向的
        ope_avg_ave = _nconf_data["x"][2]
        corr3_avg_ave = np.mean([_nconf_data[d][3] for d in _dirs], axis=0)
        print(f"  corr2_avg_ave:  {corr2_avg_ave.shape}")
        print(f"  corr2_avg_orig_ave: {corr2_avg_orig_ave.shape}")
        print(f"  ope_avg_ave:    {ope_avg_ave.shape}")
        print(f"  corr3_avg_ave:  {corr3_avg_ave.shape}")

        del _nconf_data, _ope_by_axis
        gc.collect()

        # 4. resample + ratio
        ratio_ave, ope_loopT_ave = resample_and_ratio(
            corr2_avg_ave, corr2_avg_orig_ave, ope_avg_ave, corr3_avg_ave,
            sampa, jack)
        print(f"ratio_ave shape: {ratio_ave.shape}")

        # 5. 保存
        np.save(os.path.join(outpa.get_sub_dir(
            "ratio"), "ratio_ave.npy"), ratio_ave)
        print("ratio_ave saved to ratio/ratio_ave.npy")

        del corr2_avg_ave, corr2_avg_orig_ave, ope_avg_ave, corr3_avg_ave, ratio_ave
        gc.collect()

        time1 = time.perf_counter()
        print(f"Peak Memory: {get_peak_memory_gb():.3f} GB")
        print(f"spend time: {(time1 - time0):.2f}s\n")

        if part_end == 1:
            print("job finish")
            sys.exit(0)
    else:
        print("===== skip compute ratio, loading ratio from file =====")

    # ---- Part 2: plot ----
    if part_start <= 2:
        time0 = time.perf_counter()

        ratio_ave = np.load(os.path.join(
            outpa.sub_dir_path("ratio"), "ratio_ave.npy"))
        print(f"loaded ratio_ave, shape: {ratio_ave.shape}")

        plot_ratio(ratio_ave, plotpa, outpa.sub_dir_path("ratio"), jack, "ave",
                   Nconf=sampa.Nconf, Nsample=sampa.Nsample,
                   Px=0, Py=0, Pz=sampa.P)

        time1 = time.perf_counter()
        print(f"spend time: {(time1 - time0):.2f}s\n")
    print("job finish")
