#!/public/home/huangcl/.venv/bin/python
"""
plot_FH.py

对给定 conf_short 和 P, 自动扫描 fit/ 目录, 做以下工作:

  Part 1: 扫描 fit/ 下所有子文件夹, 对每个子文件夹:
            - 合并 per-z 文件到 merged/{tag}/
            - 生成 report.txt
            - 画每个参数 vs z 图 + chi2 散点图 (保存在 merged/{tag}/ 下)

  Part 2: 对比图 + bestfit 图:
            - 对 CmpConfig 中指定的 dt_start 列表, 遍历每个 dt_start,
              从 merged/ 读取数据, 多组数据叠加在同一张图上
            - 画 bestfit FH + c0 色带图

用法:
    python plot_FH.py -c L24x72 -p 4

输出:
    01_result/{conf_short}/P{P}/fit/merged/{tag}/
        fit.npz              # 合并后的拟合数据
        report.txt           # 报告
        pic_{param}.png      # 参数 vs z 图
        pic_chi2.png         # chi2/dof 散点图
    01_result/{conf_short}/P{P}/pic/
        pic_{param}.png      # 各窗口参数对比图 (多组数据叠加)
        pic_chi2.png         # chi2/dof 对比散点图
        pic_bestfit_z{Z}.png # bestfit FH + c0 色带图
"""

import numpy as np
import argparse
import os
import re
import sys
import time
from pathlib import Path
from prettytable import PrettyTable
from dataclasses import dataclass, field

# ---- 从上级目录 98_tools 导入通用函数 ----
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "98_tools"))
from analysis_tools import sem, plot_errbar, plot_scatter  # noqa: E402


# ===== 定义 dataclass =====

@dataclass
class PathConfig:
    """路径配置"""
    merged_base_dir: str         # fit/merged/ 目录 (输出合并文件)
    pic_dir: str                 # pic/ 目录 (输出对比图 + bestfit 图)
    fh_path: str                 # FH 数据文件路径 (读取, bestfit 用)


@dataclass
class BestfitConfig:
    """bestfit 配置 (指定哪个窗口是 bestfit)"""
    dt_start: int
    dt_end: int
    nex: int
    model: int


@dataclass
class CmpConfig:
    """对比图画图配置"""
    # 要画对比图的 dt_start 列表, 如 [6, 7, 8, 9]
    cmp_dt_start_list: list[int] = field(default_factory=list)

    # 横纵坐标范围
    para_xlim: list[float] = None   # 参数图横轴 (z 范围)
    param_ylim: dict[str, list[float]] = field(default_factory=dict)

    chi2_ylim: list[float] = None   # chi2 图纵轴, 默认 [0, 2]

    xoffset: float = 0.2            # 多条曲线横坐标错开量
    z_step: int = 3                 # z 步长 (取点间隔)


@dataclass
class FHPlotConfig:
    """FH 图画图配置"""
    xlim: list[float] = None        # FH 图横轴范围
    ylim: list[float] = None        # FH 图纵轴范围


@dataclass
class C0CmpConfig:
    """不同 P 的 bestfit c0 对比图画图配置"""
    P_list: list[int] = field(default_factory=list)  # 要对比的 P 列表
    z_step: int = 2                 # z 步长 (取点间隔)
    xlim: list[float] = None        # 横轴范围 (z 范围)
    ylim: list[float] = None        # 纵轴范围 (c0 范围)


# ===== 初始化配置 =====

def init_config(conf_short: str, P: int):
    """
    根据 conf_short 和 P 初始化配置, 创建输出目录.

    返回:
        pcfg: PathConfig
        bcfg: BestfitConfig
        ccfg: CmpConfig
        fcfg: FHPlotConfig
        z_list: list[int]  # 要画的 z 列表
    """
    # ===================================================
    # ===== 配置参数 (P 直接在下面修改) =====
    # ===================================================
    if conf_short == "L24x72":
        # bestfit 参数: [model, dt_start, dt_end, nex]
        _bestfit_by_P = {
            2: [1, 7, 11, 2],
            3: [1, 8, 10, 2],
            4: [1, 8, 10, 2],
            5: [1, 7, 9, 2],
            6: [3, 2, 7, 1],
        }

        # ---- 对比图画图配置 ----
        # 只填 dt_start, 其他参数 (model, dt_end, nex) 从 _bestfit_by_P[P] 读取
        _cmp_dt_start_list = [6, 7, 8]   # 只填 tsep 左端点
        _bp = _bestfit_by_P[P]

        _cmp_cfg = CmpConfig(
            cmp_dt_start_list=_cmp_dt_start_list,
            para_xlim=[-0.5, 17.5],
            param_ylim={
                'c0': [-0.1, 1.0],
                'c2': [-0.5, 0.5],
                'c1': [-0.5, 0.5],
                'dE': [0.0, 1.0],
            },
            z_step=1,
        )

        # ---- FH 图画图配置 ----
        _fh_xlim_by_P = {
            2: [1.5, 12.5],
            3: [1.5, 12.5],
            4: [1.5, 12.5],
            5: [1.5, 12.5],
            6: [1.5, 12.5],
        }
        _fh_cfg = FHPlotConfig(
            xlim=_fh_xlim_by_P.get(P),
            ylim=[-0.1, 1.1],
        )

        # ---- z 列表 ----
        _z_list = list(range(3))

        # ---- 不同 P 的 bestfit c0 对比图配置 ----
        _c0_cmp_cfg = C0CmpConfig(
            P_list=[2, 3, 4, 5, 6],   # 要对比的 P 列表
            z_step=1,                  # z 步长
            xlim=[-0.5, 14.5],         # 横轴 (z 范围)
            ylim=[-0.05, 0.8],          # 纵轴 (c0 范围)
        )

    else:
        print(f"conf {conf_short} not exist.")
        sys.exit()
    # ===================================================

    # ---- bestfit ----
    _bp = _bestfit_by_P[P]
    bcfg = BestfitConfig(
        dt_start=_bp[1],    # 第1位: dt_start
        dt_end=_bp[2],      # 第2位: dt_end
        nex=_bp[3],         # 第3位: nex
        model=_bp[0],       # 第0位: model
    )

    # ---- 路径 ----
    _base_dir = os.path.join(os.getcwd(), "01_result", conf_short, f"P{P}")
    pcfg = PathConfig(
        merged_base_dir=os.path.join(_base_dir, "fit", "merged"),
        pic_dir=os.path.join(_base_dir, "pic"),
        fh_path=os.path.join(_base_dir, "fh", f"FH_nex{bcfg.nex}.npy"),
    )
    os.makedirs(pcfg.merged_base_dir, exist_ok=True)
    os.makedirs(pcfg.pic_dir, exist_ok=True)

    return pcfg, bcfg, _cmp_cfg, _fh_cfg, _z_list, _bestfit_by_P, _c0_cmp_cfg


# ===== Part 1: 合并 per-z 文件到 merged/{tag}/ =====

def merge_per_z(tag: str, src_dir: str, dst_dir: str):
    """
    读取 src_dir 下的所有 fit_z{Z}.npz, 按 z 合并,
    保存 fit.npz 和 report.txt 到 dst_dir.

    返回:
        merged_path: 合并后的文件路径
    """
    # 收集所有 fit_z*.npz
    z_files = sorted([
        f for f in os.listdir(src_dir)
        if f.startswith("fit_z") and f.endswith(".npz")
    ])
    Nz = len(z_files)

    # 从第一个文件推断参数名和样本数
    first = np.load(os.path.join(src_dir, z_files[0]))
    param_names = list(first.keys())
    Nsample = first[param_names[0]].shape[0]

    # 合并
    merged = {name: np.zeros((Nsample, Nz)) for name in param_names}
    for _iz in range(Nz):
        _path = os.path.join(src_dir, f"fit_z{_iz}.npz")
        _npz = np.load(_path)
        for name in param_names:
            merged[name][:, _iz] = np.asarray(_npz[name]).squeeze()
    # 输出最后一次加载的路径
    print(f"    last loaded: {_path}")

    # 创建输出目录
    os.makedirs(dst_dir, exist_ok=True)

    # 保存 fit.npz
    merged_path = os.path.join(dst_dir, "fit.npz")
    np.savez(merged_path, **merged)
    print(f"    merged -> {merged_path}")

    # 生成 report.txt
    _param_names_plot = [k for k in param_names if k != "chi2"]
    lines = []
    sep_line = "=" * 72
    lines.append(sep_line)
    lines.append(f"  Summary Table: {tag}")
    lines.append(
        f"  # para[model]_n[nex]_tsep[dt_start]_[dt_end]: "
        f"model=参数个数, nex=两端移除数据的个数, "
        f"dt_start/dt_end=拟合窗口")
    lines.append(sep_line)

    tbl = PrettyTable()
    tbl.field_names = ["z"] + _param_names_plot + ["chi2/dof"]
    for name in tbl.field_names:
        tbl.align[name] = "c"

    for _iz in range(Nz):
        row = [f"{_iz}"]
        for name in _param_names_plot:
            mean = merged[name][:, _iz].mean()
            err = sem(merged[name][:, _iz], False)
            row.append(f"{mean:.3f}({err * 1e3:.0f})")
        row.append(f"{merged['chi2'][:, _iz].mean():.2g}")
        tbl.add_row(row)

    lines.append(str(tbl))
    lines.append("")

    report_path = os.path.join(dst_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"    report -> {report_path}")

    return merged_path


# ===== Part 1: 单窗口画图 (每个参数 vs z + chi2) =====

def plot_single_window(tag: str, merged_dir: str, ccfg: CmpConfig,
                       conf_short: str, P: int):
    """
    对单个拟合窗口, 画每个参数 vs z 图 + chi2 散点图,
    保存在 merged/{tag}/ 目录下.

    图题格式:
        参数图: "{conf_short}, P={P}, {param}, model={model}, nex={nex}, tsep=[{start},{end}]"
        chi2图: "{conf_short}, P={P}, chi2/dof, model={model}, nex={nex}, tsep=[{start},{end}]"
    """
    merged_path = os.path.join(merged_dir, "fit.npz")
    if not os.path.exists(merged_path):
        print(f"    Warning: {merged_path} not exist, skip.")
        return

    # 从 tag 解析 model, nex, dt_start, dt_end
    # tag 格式: para{model}_n{nex}_tsep{start}_{end}
    _m = re.match(r"^para(\d+)_n(\d+)_tsep(\d+)_(\d+)$", tag)
    if _m:
        _model = _m.group(1)
        _nex = _m.group(2)
        _dt_start = _m.group(3)
        _dt_end = _m.group(4)
    else:
        _model = _nex = _dt_start = _dt_end = "?"

    data = np.load(merged_path)
    param_names = [k for k in data.keys() if k != "chi2"]
    Nz = data[param_names[0]].shape[1]
    z_vals = np.arange(Nz)

    # ---- 每个参数画一张图 ----
    for _name in param_names:
        _arr = data[_name]
        _mean = _arr.mean(0)
        _err = sem(_arr, jackknife=False)
        _plot_data = {_name: (_mean, _err)}

        save_path = os.path.join(merged_dir, f"pic_{_name}.png")
        plot_errbar(
            z_vals, _plot_data, save_path,
            xlabel="z", ylabel=_name,
            xlim=ccfg.para_xlim,
            ylim=ccfg.param_ylim.get(_name),
            title=f"{conf_short}, P={P}, {_name}, model={_model}, nex={_nex}, tsep=[{_dt_start},{_dt_end}]",
            figsize=(10, 6), dpi=150,
        )
        print(f"    pic saved: {save_path}")

    # ---- chi2 散点图 ----
    _chi2_data = {"chi2/dof": data["chi2"].mean(0)}
    chi2_path = os.path.join(merged_dir, "pic_chi2.png")
    plot_scatter(
        z_vals, _chi2_data, chi2_path,
        xlabel="z", ylabel="chi2/dof",
        xlim=ccfg.para_xlim,
        ylim=[0, 2],
        title=f"{conf_short}, P={P}, chi2/dof, model={_model}, nex={_nex}, tsep=[{_dt_start},{_dt_end}]",
        figsize=(10, 6), dpi=150,
        show_hline=True, hline_y=1.0, hline_label="chi2/dof=1",
    )
    print(f"    pic saved: {chi2_path}")


# ===== Part 2: 对比图 (多组数据叠加) =====

def plot_cmp(pcfg: PathConfig, ccfg: CmpConfig,
             conf_short: str, P: int,
             model: int, nex: int, dt_end: int):
    """
    对每个参数, 遍历 cmp_dt_start_list, 从 merged/ 读取数据,
    多组数据叠加在同一张图上.

    图题格式:
        参数图: "{conf_short}, P={P}, {param}, model={model}, nex={nex}"
        chi2图: "{conf_short}, P={P}, chi2/dof, model={model}, nex={nex}"
    """
    # 先读第一个 dt_start 确定参数名和 Nz
    _first_tag = f"para{model}_n{nex}_tsep{ccfg.cmp_dt_start_list[0]}_{dt_end}"
    _first_path = os.path.join(pcfg.merged_base_dir, _first_tag, "fit.npz")
    if not os.path.exists(_first_path):
        print(f"    Warning: {_first_path} not exist, skip comparison.")
        return

    _first_data = np.load(_first_path)
    param_names = [k for k in _first_data.keys() if k != "chi2"]
    Nz = _first_data[param_names[0]].shape[1]
    z_vals = np.arange(Nz)[::ccfg.z_step]

    # ---- 每个参数画一张对比图 ----
    for _name in param_names:
        _plot_data = {}
        for _dt_start in ccfg.cmp_dt_start_list:
            _tag = f"para{model}_n{nex}_tsep{_dt_start}_{dt_end}"
            _path = os.path.join(pcfg.merged_base_dir, _tag, "fit.npz")
            if not os.path.exists(_path):
                print(f"    Warning: {_path} not exist, skip this window.")
                continue
            _data = np.load(_path)
            _arr = _data[_name]
            _mean = _arr.mean(0)[::ccfg.z_step]
            _err = sem(_arr, jackknife=False)[::ccfg.z_step]
            _plot_data[f"dt: {_dt_start}~{dt_end}"] = (_mean, _err)

        if not _plot_data:
            continue

        save_path = os.path.join(pcfg.pic_dir, f"pic_{_name}.png")
        plot_errbar(
            z_vals, _plot_data, save_path,
            xlabel="z", ylabel=_name,
            xlim=ccfg.para_xlim,
            ylim=ccfg.param_ylim.get(_name),
            title=f"{conf_short}, P={P}, {_name}, model={model}, nex={nex}",
            figsize=(10, 6), dpi=150,
        )
        print(f"    pic saved: {save_path}")

    # ---- chi2 散点图 ----
    _chi2_data = {}
    for _dt_start in ccfg.cmp_dt_start_list:
        _tag = f"para{model}_n{nex}_tsep{_dt_start}_{dt_end}"
        _path = os.path.join(pcfg.merged_base_dir, _tag, "fit.npz")
        if not os.path.exists(_path):
            continue
        _data = np.load(_path)
        _chi2_data[f"dt: {_dt_start}~{dt_end}"] = \
            _data["chi2"].mean(0)[::ccfg.z_step]

    if _chi2_data:
        chi2_path = os.path.join(pcfg.pic_dir, "pic_chi2.png")
        plot_scatter(
            z_vals, _chi2_data, chi2_path,
            xlabel="z", ylabel="chi2/dof",
            xlim=ccfg.para_xlim,
            ylim=[0, 2],
            title=f"{conf_short}, P={P}, chi2/dof, model={model}, nex={nex}",
            figsize=(10, 6), dpi=150,
            show_hline=True, hline_y=1.0, hline_label="chi2/dof=1",
        )
        print(f"    pic saved: {chi2_path}")


# ===== Part 2: bestfit FH + c0 色带图 =====

def plot_bestfit(pcfg: PathConfig, bcfg: BestfitConfig,
                 fcfg: FHPlotConfig,
                 conf_short: str, P: int, z_list: list[int]):
    """
    画 bestfit 的 FH + c0 色带图.

    图题格式:
        "{conf_short}, P={P}, z={z}, model={model}-para, chi2/dof={val}"
    """
    # 加载 FH 数据
    print(f"\n  loading bestfit FH (nex={bcfg.nex}) from: {pcfg.fh_path}")
    fh = np.load(pcfg.fh_path)
    _, dt_max, Nz_fh = fh.shape
    t_vals = np.arange(dt_max)

    # 加载 bestfit 的合并数据
    bf_tag = f"para{bcfg.model}_n{bcfg.nex}_tsep{bcfg.dt_start}_{bcfg.dt_end}"
    merged_path = os.path.join(pcfg.merged_base_dir, bf_tag, "fit.npz")
    if not os.path.exists(merged_path):
        print(f"    Warning: bestfit merged file not found: {merged_path}")
        return

    bf_data = np.load(merged_path)
    _c0 = bf_data["c0"]
    _chi2 = bf_data["chi2"]

    # 构建 c0_data 和 chi2_info
    _c0_data = {}
    _chi2_info = {}
    for _iz in z_list:
        if _iz < _c0.shape[1]:
            _c0_data[_iz] = (
                _c0[:, _iz].mean(),
                sem(_c0[:, _iz], jackknife=False),
            )
            _chi2_info[_iz] = _chi2[:, _iz].mean()

    # 画 FH + c0 色带
    _fh_dict = {bcfg.nex: fh}
    for _iz in z_list:
        if _iz >= Nz_fh:
            print(f"    Warning: z={_iz} exceeds Nz_fh={Nz_fh}, skip.")
            continue

        # 收集 FH 数据
        _data = {}
        for _nex, _fh in sorted(_fh_dict.items()):
            _mean = _fh.mean(0)[:, _iz]
            _err = sem(_fh, jackknife=False)[:, _iz]
            _data[f"nex={_nex}"] = (_mean, _err)

        # c0 色带
        c0_mean, c0_err = _c0_data[_iz]
        _band_x = np.arange(bcfg.dt_start, bcfg.dt_end + 1, dtype=float)
        _band_y_down = np.full_like(_band_x, c0_mean - c0_err, dtype=float)
        _band_y_up = np.full_like(_band_x, c0_mean + c0_err, dtype=float)
        _band_label = f"c0 = {c0_mean:.3f} ± {c0_err:.3f}"

        # 标题
        _chi2_val = _chi2_info.get(_iz, 0)
        _title = (f"{conf_short}, P={P}, z={_iz}, "
                  f"model={bcfg.model}-para, chi2/dof={_chi2_val:.2g}")

        save_path = os.path.join(pcfg.pic_dir, f"pic_bestfit_z{_iz}.png")
        plot_errbar(
            t_vals, _data, save_path,
            xlabel="t", ylabel="FH",
            xlim=fcfg.xlim, ylim=fcfg.ylim,
            x_offset=0.2,
            title=_title,
            show_band=True,
            band_x=_band_x,
            band_y_down=_band_y_down,
            band_y_up=_band_y_up,
            band_label=_band_label,
        )
        print(f"    pic saved: {save_path}")


# ===== Part 3: 不同 P 的 bestfit c0 对比图 =====

def plot_c0_cmp(conf_short: str, P: int,
                bestfit_by_P: dict, c0_cmp_cfg: C0CmpConfig):
    """
    读取不同 P 的 bestfit 数据的 c0, 画在同一张图上.

    对 c0_cmp_cfg.P_list 中的每个 P:
        - 从 bestfit_by_P[P] 获取 bestfit 参数 [model, dt_start, dt_end, nex]
        - 构造 tag: para{model}_n{nex}_tsep{dt_start}_{dt_end}
        - 读取 01_result/{conf_short}/P{P}/fit/merged/{tag}/fit.npz 中的 c0
        - 用 try 包裹, 若某个 P 数据缺失, 提示并退出

    图题: "bare matrix elements"
    图例: "P={P}"
    z 步长: c0_cmp_cfg.z_step (默认 2)
    横纵轴范围: c0_cmp_cfg.xlim / c0_cmp_cfg.ylim
    """
    print(f"\n--- plotting c0 comparison across P (pic/) ---")

    _plot_data = {}
    _z_vals = None

    for _P in c0_cmp_cfg.P_list:
        try:
            # 从 bestfit_by_P 获取该 P 的 bestfit 参数
            _bp = bestfit_by_P[_P]
            _model = _bp[0]
            _dt_start = _bp[1]
            _dt_end = _bp[2]
            _nex = _bp[3]

            # 构造 tag 和路径
            _tag = f"para{_model}_n{_nex}_tsep{_dt_start}_{_dt_end}"
            _merged_path = os.path.join(
                os.getcwd(), "01_result", conf_short, f"P{_P}",
                "fit", "merged", _tag, "fit.npz")

            # 读取 c0
            _data = np.load(_merged_path)
            _c0 = _data["c0"]  # (Nsample, Nz)

            # 按 z_step 取点
            _mean = _c0.mean(0)[::c0_cmp_cfg.z_step]
            _err = sem(_c0, jackknife=False)[::c0_cmp_cfg.z_step]
            _plot_data[f"P={_P}"] = (_mean, _err)

            # 记录 z 坐标 (所有 P 的 Nz 应一致)
            if _z_vals is None:
                Nz = _c0.shape[1]
                _z_vals = np.arange(Nz)[::c0_cmp_cfg.z_step]

            print(f"    loaded c0 from P={_P}: {_merged_path}")
        except Exception as _e:
            print(f"    Please check if P={_P} fit results exist.")
            sys.exit(1)

    if not _plot_data:
        print("    No c0 data loaded, skip.")
        return

    # 画图 (保存到 01_result/{conf_short}/ 下)
    save_path = os.path.join(
        os.getcwd(), "01_result", conf_short,
        "pic_c0_cmp.png")
    plot_errbar(
        _z_vals, _plot_data, save_path,
        xlabel="z", ylabel="c0",
        xlim=c0_cmp_cfg.xlim,
        ylim=c0_cmp_cfg.ylim,
        title="bare matrix elements",
        x_offset=0.15,
        figsize=(10, 6), dpi=150,
    )
    print(f"    pic saved: {save_path}")


# ===== 主函数 =====

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot FH fit results: merge per-z fits, generate reports, "
                    "plot per-window params, comparison plots, and bestfit.")
    parser.add_argument("-c", type=str, default="L24x72",
                        help="conf_short (default: L24x72)")
    parser.add_argument("-p", type=int, default=2,
                        help="momentum P (default: 2)")
    args = parser.parse_args()

    conf_short = args.c
    P = args.p

    # 初始化配置
    pcfg, bcfg, ccfg, fcfg, z_list, bestfit_by_P, c0_cmp_cfg = \
        init_config(conf_short, P)

    print(f"conf_short: {conf_short}, P: {P}")
    _t0 = time.perf_counter()

    # ===== Part 1: 扫描 fit/ 目录, 合并 + 报告 + 单窗口画图 =====
    print(
        f"\n--- scanning fit dir: {os.path.join(os.getcwd(), '01_result', conf_short, f'P{P}', 'fit')} ---")
    fit_dir = os.path.join(os.getcwd(), "01_result",
                           conf_short, f"P{P}", "fit")
    fit_subdirs = sorted([
        d for d in os.listdir(fit_dir)
        if os.path.isdir(os.path.join(fit_dir, d)) and d != "merged"
    ])
    print(f"  found {len(fit_subdirs)} fit subdirs:")
    for _d in fit_subdirs:
        print(f"    {_d}")

    print(f"\n--- merging per-z files to merged/ ---")
    for _tag in fit_subdirs:
        src_dir = os.path.join(fit_dir, _tag)
        dst_dir = os.path.join(pcfg.merged_base_dir, _tag)
        merge_per_z(_tag, src_dir, dst_dir)

    print(f"\n--- plotting per-window parameters ---")
    for _tag in fit_subdirs:
        _merged_dir = os.path.join(pcfg.merged_base_dir, _tag)
        if not os.path.exists(os.path.join(_merged_dir, "fit.npz")):
            print(f"  Warning: {_tag}/fit.npz not exist, skip.")
            continue
        print(f"\n  window: {_tag}")
        plot_single_window(_tag, _merged_dir, ccfg, conf_short, P)

    # ===== Part 2: 对比图 + bestfit =====
    print(f"\n--- plotting comparison (pic/) ---")
    plot_cmp(pcfg, ccfg, conf_short, P,
             bcfg.model, bcfg.nex, bcfg.dt_end)

    print(f"\n--- plotting bestfit FH + c0 band ---")
    plot_bestfit(pcfg, bcfg, fcfg, conf_short, P, z_list)

    # ===== Part 3: 不同 P 的 bestfit c0 对比图 =====
    plot_c0_cmp(conf_short, P, bestfit_by_P, c0_cmp_cfg)

    _t1 = time.perf_counter()
    print(f"\ntotal time: {(_t1 - _t0):.2f}s")
    print("plot_FH done.")
