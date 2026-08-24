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
    plot_errbar, plot_scatter,
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
    # 需要平均的 dir 列表, 如 ["x", "y", "z"]
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

    @property
    def ope_dir(self):
        """ope 数据与图片存放目录 (包含 raw 和 loopT)"""
        d = os.path.join(os.getcwd(), self.base_dir,
                         self.conf_short, "ope")
        os.makedirs(d, exist_ok=True)
        return d

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
parser.add_argument("--test2", action="store_true", default=False,
                    help="run test2: average 6-dir corr2 first, then compute ratio with each ope dir (default: False)")
args = parser.parse_args()
conf_short = args.c
part_start = args.s
part_end = args.e
run_test2 = args.test2
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
_fixed_z_list = list(range(0, 3))  # 前 3 个 z
_fixed_xlim = [-7, 7]

# ope 对比图画图参数 (与 Pz 无关)
_ope_ylim = None       # ope 误差棒图 y 轴范围, None 为自动
_ope_sem_ylim = [-0.1, 0.6]   # ope SEM 图 y 轴范围, None 为自动
# =====================================


if conf_short == "L24x72":
    # 12300 是空文件夹
    _conf_ids = [x for x in range(4050, 48001, 50) if x not in (7350, 12300)]
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
# 创建子目录: 6 个方向 (pos_x, neg_x, pos_y, neg_y, pos_z, neg_z) + ave + test2
for _d in ["pos_x", "neg_x", "pos_y", "neg_y", "pos_z", "neg_z", "ave", "test2"]:
    outpa.get_sub_dir(_d)
# =========================


########################################################################################
def load_ope(sampa: SampleParams, axis: str, jack: bool):
    """加载指定轴 (x/y/z) 的 OPE 数据, 返回原始 _ope 和 resample 后的 ope_raw"""
    # 旧版函数名为 compute_ope, 现拆分为 load_ope (仅负责加载 OPE)
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

    ope_raw = resample(_ope, sampa.Nsample, jack)  # (Nsample, Nt, Nx)

    print("==================== load ope end ====================")
    return _ope, ope_raw


########################################################################################
def load_2pt(sampa: SampleParams, Px: int, Py: int, Pz: int, momP: int):
    """加载指定动量的 2pt 数据, 返回 _corr (Nconf, Nt, Nt)"""
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
def compute_ratio(sampa: SampleParams, _ope: np.ndarray, _corr: np.ndarray, jack: bool):
    """用传入的 _ope 和 _corr 计算 3pt 与 ratio, 返回 (ratio, ope_loopT)"""
    print("==================== compute ratio start ====================")

    # # 旧版: 在函数内部加载 corr 数据 (2pt 加载已拆分到 load_2pt)
    # def compute_ratio(sampa: SampleParams, Px: int, Py: int, Pz: int,
    #                   momP: int, _ope: np.ndarray, jack: bool):
    #     """加载 corr 数据并用传入的 _ope 计算 ratio, 返回 (ratio, ope_loopT)"""
    #     print(f"P=({Px},{Py},{Pz}), momP={momP}")
    #     # 确定轴名 (用于 corr 文件路径)
    #     if Px != 0:
    #         _axis = 'x'
    #     elif Py != 0:
    #         _axis = 'y'
    #     elif Pz != 0:
    #         _axis = 'z'
    #     else:
    #         print('all P=0, invalid')
    #         sys.exit(1)
    #     _corr = np.zeros((sampa.Nconf, sampa.Nt, sampa.Nt), dtype=complex)
    #     for i, conf_id in enumerate(sampa.conf_ids):
    #         _corr_path = (
    #             f"/public/group/lqcd/donghx/2pt_Result/{sampa.conf_name}/"
    #             f"momsmear{momP}{_axis}/{conf_id}/"
    #             f"twopt_slice_pp_Px{Px}Py{Py}Pz{Pz}_eginphase{momP}_Cg5g4_nopol_ss_conf{conf_id}.npy"
    #         )
    #         _corr[i] = np.load(_corr_path)
    #     print("corr load finish")
    #     print(f"  example corr:    {_corr_path}")

    # _corr2: conf, tf, ti
    # para: conf, ti(loop), dt
    _corr2_rel = np.zeros((sampa.Nconf, sampa.Nt, sampa.dt_max), dtype=complex)
    _ope_rel = np.zeros(
        (sampa.Nconf, sampa.Nt, sampa.dt_max, sampa.Nx), dtype=complex
    )  # para: conf, ti(loop), dtau, z

    # loop ti
    for ti in range(sampa.Nt):
        corr2_shift = np.roll(_corr[:, :, ti], shift=-ti, axis=1)
        _corr2_rel[:, ti, :] = corr2_shift[:, :sampa.dt_max]

        ope_shift = np.roll(_ope, shift=-ti, axis=1)
        _ope_rel[:, ti, :, :] = ope_shift[:, :sampa.dt_max, :]

    # para: conf, ti(loop), dt, dtau, z
    _corr3 = np.zeros(
        (sampa.Nconf, sampa.Nt, sampa.dt_max,
         sampa.dt_max, sampa.Nx), dtype=complex
    )
    for _dt in range(sampa.dt_max):
        for _dtau in range(_dt + 1):
            c2_slice = _corr2_rel[:, :, _dt]  # para: conf, ti(loop)
            ope_slice = _ope_rel[:, :, _dtau, :]  # para: conf, ti(loop), z
            _corr3[:, :, _dt, _dtau, :] = ope_slice * \
                c2_slice[:, :, np.newaxis]

    print("loop ti finish")
    del _corr
    gc.collect()

    # # 对 ti 求平均, 消除源时间依赖
    # corr2_avg = _corr2_rel.mean(axis=1)  # (Nconf, dt)
    # ope_avg = _ope_rel.mean(axis=1)      # (Nconf, dtau, z)
    # corr3_avg = _corr3.mean(axis=1)      # (Nconf, dt, dtau, z)

    # del _corr2_rel, _ope_rel, _corr3
    # gc.collect()

    # # resample
    # corr2 = resample(corr2_avg, sampa.Nsample)  # (Nsample, dt)
    # ope = resample(ope_avg, sampa.Nsample)       # (Nsample, dtau, z)

    # del corr2_avg, ope_avg
    # gc.collect()

    # # corr3 按 dt 切片分别做 resample, 避免爆内存
    # corr3 = np.zeros((sampa.Nsample, sampa.dt_max,
    #                   sampa.dt_max, sampa.Nx), dtype=complex)
    # for _dt in range(sampa.dt_max):
    #     corr3[:, _dt, :, :] = resample(
    #         corr3_avg[:, _dt, :, :], jack, sampa.Nsample)

    # del corr3_avg
    # gc.collect()

    # print("start to compute ratio")
    # # ratio: (Nsample, dt, dtau, z)
    # ratio = np.zeros((sampa.Nsample, sampa.dt_max,
    #                  sampa.dt_max, sampa.Nx), dtype=float)
    # for _dt in range(sampa.dt_max):
    #     corr3_dt = corr3[:, _dt, :, :]  # (Nsample, dtau, z)
    #     corr2_dt = corr2[:, _dt]        # (Nsample,)

    #     # disconnected 部分: <C3> - <C2> * <ope>
    #     corr3_disc = corr3_dt - corr2_dt[:, np.newaxis, np.newaxis] * ope

    #     # ratio = C3_disc / C2
    #     ratio[:, _dt, :, :] = (
    #         corr3_disc / corr2_dt[:, np.newaxis, np.newaxis]).real

    # print("ratio shape:", ratio.shape)

    # 对 ti 求平均, 消除源时间依赖
    corr2_avg = _corr2_rel.mean(axis=1)  # (Nconf, dt)
    ope_avg = _ope_rel.mean(axis=1)      # (Nconf, dtau, z)
    corr3_avg = _corr3.mean(axis=1)      # (Nconf, dt, dtau, z)

    del _corr2_rel, _ope_rel, _corr3
    gc.collect()

    # resample
    corr2 = resample(corr2_avg, sampa.Nsample, jack)  # (Nsample, dt)
    ope = resample(ope_avg, sampa.Nsample, jack)       # (Nsample, dtau, z)

    del corr2_avg, ope_avg
    gc.collect()

    # corr3 按 dt 切片分别做 resample, 避免爆内存
    corr3 = np.zeros((sampa.Nsample, sampa.dt_max,
                      sampa.dt_max, sampa.Nx), dtype=complex)
    for _dt in range(sampa.dt_max):
        corr3[:, _dt, :, :] = resample(
            corr3_avg[:, _dt, :, :], sampa.Nsample, jack)

    del corr3_avg
    gc.collect()

    print("start to compute ratio")
    # ratio: (Nsample, dt, dtau, z)
    ratio = np.zeros((sampa.Nsample, sampa.dt_max,
                     sampa.dt_max, sampa.Nx), dtype=float)
    for _dt in range(sampa.dt_max):
        corr3_dt = corr3[:, _dt, :, :]  # (Nsample, dtau, z)
        corr2_dt = corr2[:, _dt]        # (Nsample,)

        # disconnected 部分: <C3> - <C2> * <ope>
        corr3_disc = corr3_dt - corr2_dt[:, np.newaxis, np.newaxis] * ope

        # ratio = C3_disc / C2
        ratio[:, _dt, :, :] = (
            corr3_disc / corr2_dt[:, np.newaxis, np.newaxis]).real

    print("ratio shape:", ratio.shape)
    print("==================== compute ratio end ====================")
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

    # ---- Part 1: compute ratio (OPE 3 方向平均, 6 方向 2pt 分别算 ratio, 最后 6 方向平均) ----
    if not run_test2:
        time0 = time.perf_counter()

        # 1. 加载 OPE 对 x, y, z 三个方向做平均
        print("===== loading OPE for x, y, z and averaging =====")
        _ope_list = []
        for _axis in sampa.ave_dirs:
            _ope, _ = load_ope(sampa, _axis, jack)
            _ope_list.append(_ope)
        _ope_avg = np.mean(_ope_list, axis=0)  # (Nconf, tau, z)
        del _ope_list
        gc.collect()
        print(f"  OPE averaged, shape: {_ope_avg.shape}")

        # 2. 分别加载 6 个方向的 2pt, 每个方向单独算 ratio 并保存
        #    6 个方向: pos_x, neg_x, pos_y, neg_y, pos_z, neg_z
        _dir_config = [
            ("pos_x",  sampa.P,  0,        0,        sampa.momP),
            ("neg_x", -sampa.P,  0,        0,       -sampa.momP),
            ("pos_y",  0,        sampa.P,  0,        sampa.momP),
            ("neg_y",  0,       -sampa.P,  0,       -sampa.momP),
            ("pos_z",  0,        0,        sampa.P,  sampa.momP),
            ("neg_z",  0,        0,       -sampa.P, -sampa.momP),
        ]
        _ratio_list = []
        for _dir_name, _Px, _Py, _Pz, _momP in _dir_config:
            print(f"===== loading 2pt ({_dir_name}) =====")
            _corr = load_2pt(sampa, _Px, _Py, _Pz, _momP)
            print(f"===== computing ratio ({_dir_name}) =====")
            _ratio, _ = compute_ratio(sampa, _ope_avg, _corr, jack)
            np.save(os.path.join(outpa.get_sub_dir(
                _dir_name), "ratio.npy"), _ratio)
            print(f"  ratio ({_dir_name}) saved, shape: {_ratio.shape}")
            _ratio_list.append(_ratio)
            del _corr
            gc.collect()

        # 3. 对 6 个方向的 ratio 做平均
        print("===== averaging 6-direction ratios =====")
        ratio_ave = np.mean(_ratio_list, axis=0)  # (Nsample, dt, dtau, z)
        np.save(os.path.join(outpa.get_sub_dir("ave"), "ratio.npy"), ratio_ave)
        print(f"  ratio_ave saved, shape: {ratio_ave.shape}")

        del _ope_avg, _ratio_list, ratio_ave
        gc.collect()
        time1 = time.perf_counter()
        print(f"Peak Memory: {get_peak_memory_gb():.3f} GB")
        print(f"spend time: {(time1 - time0):.2f}s\n")

    # ---- Part 2: plot (6 个方向 + ave) ----
        time0 = time.perf_counter()

        # 加载 6 个方向 + ave 的 ratio
        _dir_mom_map = {
            "pos_x": (sampa.P,  0,        0),
            "neg_x": (-sampa.P,  0,        0),
            "pos_y": (0,        sampa.P,  0),
            "neg_y": (0,       -sampa.P,  0),
            "pos_z": (0,        0,        sampa.P),
            "neg_z": (0,        0,       -sampa.P),
            "ave":   (0,        0,        sampa.P),
        }
        _all_ratios = {}
        for _dir in ["pos_x", "neg_x", "pos_y", "neg_y", "pos_z", "neg_z", "ave"]:
            load_path = os.path.join(outpa.sub_dir_path(_dir), "ratio.npy")
            print(f"===== loading ratio ({_dir}) from {load_path} =====")
            _all_ratios[_dir] = np.load(load_path)
            print(f"  loaded, shape: {_all_ratios[_dir].shape}")

        # 统一画图: 6 个方向 + ave
        for _dir in ["pos_x", "neg_x", "pos_y", "neg_y", "pos_z", "neg_z", "ave"]:
            Px, Py, Pz = _dir_mom_map[_dir]
            plot_ratio(_all_ratios[_dir], plotpa, outpa.sub_dir_path(_dir), jack, _dir,
                       Nconf=sampa.Nconf, Nsample=sampa.Nsample,
                       Px=Px, Py=Py, Pz=Pz)

        time1 = time.perf_counter()
        print(f"spend time: {(time1 - time0):.2f}s\n")

    # ---- Test 2: 先平均 6 方向 corr2, 再分别和 3 方向 ope 算 ratio, 再平均 3 方向 ratio ----
    if run_test2:
        time0 = time.perf_counter()
        print("===== Test 2: average corr2 over 6 dirs, then ratio with each ope dir =====")

        # 1. 加载 6 个方向的 2pt 并平均
        print("===== loading 2pt for 6 dirs and averaging =====")
        _dir_config = [
            ("pos_x",  sampa.P,  0,        0,        sampa.momP),
            ("neg_x", -sampa.P,  0,        0,       -sampa.momP),
            ("pos_y",  0,        sampa.P,  0,        sampa.momP),
            ("neg_y",  0,       -sampa.P,  0,       -sampa.momP),
            ("pos_z",  0,        0,        sampa.P,  sampa.momP),
            ("neg_z",  0,        0,       -sampa.P, -sampa.momP),
        ]
        _corr_list = []
        for _dir_name, _Px, _Py, _Pz, _momP in _dir_config:
            _corr = load_2pt(sampa, _Px, _Py, _Pz, _momP)
            _corr_list.append(_corr)
        _corr_avg = np.mean(_corr_list, axis=0)  # (Nconf, Nt, Nt)
        del _corr_list
        gc.collect()
        print(f"  2pt averaged over 6 dirs, shape: {_corr_avg.shape}")

        # 2. 分别加载 3 个方向 (x, y, z) 的 OPE, 各自与平均 corr2 算 ratio
        _test2_ratio_list = []
        for _axis in ["x", "y", "z"]:
            print(f"===== loading OPE ({_axis}) =====")
            _ope, _ = load_ope(sampa, _axis, jack)
            print(f"===== computing ratio ({_axis}) with averaged corr2 =====")
            _ratio, _ = compute_ratio(sampa, _ope, _corr_avg, jack)
            np.save(os.path.join(outpa.get_sub_dir(
                "test2"), f"ratio_{_axis}.npy"), _ratio)
            print(f"  ratio ({_axis}) saved, shape: {_ratio.shape}")
            _test2_ratio_list.append(_ratio)
            del _ope
            gc.collect()

        # 3. 对 3 个方向的 ratio 做平均
        print("===== averaging 3-dir ratios (test2) =====")
        _test2_ratio_ave = np.mean(_test2_ratio_list, axis=0)
        np.save(os.path.join(outpa.get_sub_dir("test2"),
                "ratio_ave.npy"), _test2_ratio_ave)
        print(f"  test2 ratio_ave saved, shape: {_test2_ratio_ave.shape}")

        # 4. 画 test2 的图 (3 个方向 + ave), 存入 test2 目录
        print("===== plotting test2 ratios =====")
        _test2_save_dir = outpa.sub_dir_path("test2")
        for _axis, _ratio in zip(["x", "y", "z"], _test2_ratio_list):
            plot_ratio(_ratio, plotpa, _test2_save_dir, jack, f"test2_{_axis}",
                       Nconf=sampa.Nconf, Nsample=sampa.Nsample,
                       Px=0, Py=0, Pz=0)
        plot_ratio(_test2_ratio_ave, plotpa, _test2_save_dir, jack, "test2_ave",
                   Nconf=sampa.Nconf, Nsample=sampa.Nsample,
                   Px=0, Py=0, Pz=0)
        print("===== plotting test2 ratios end =====")

        del _corr_avg, _test2_ratio_list, _test2_ratio_ave
        gc.collect()
        time1 = time.perf_counter()
        print(f"Peak Memory: {get_peak_memory_gb():.3f} GB")
        print(f"spend time: {(time1 - time0):.2f}s\n")

    print('job finish')
