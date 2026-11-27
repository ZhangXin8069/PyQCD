#!/public/home/huangcl/.venv/bin/python
"""
功能: 统计 cg5/cg5gt 下 NPcorr 作业各 log 的完成状态,
      done = 对应结果数据文件存在,
      problem = 结果数据文件不存在, 但 log 文件存在,
      undo = log 文件不存在.
输入: init() 中的组态清单, 每个 conf_short 给出方向列表和总动量列表.
输出: 03_fig_rep/check_2pt.txt, 各类状态的组态号区间和个数.
"""

from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

# ---- 从项目目录 98_tools 导入通用函数 ----
sys.path.insert(0, "/public/home/huangcl/06_deuteron_gluon_pdf/98_tools")
from output_tools import save_txt  # noqa: E402  # type: ignore
from snsc_params import get_conf  # noqa: E402  # type: ignore
from distillation_tools import get_momsmear_info  # noqa: E402  # type: ignore


data_root = Path("/public/home/huangcl/00_data/00_NP_2pt")


# ==================== 初始化 ====================
def init() -> Tuple[Dict[str, tuple], Path]:
    """
    初始化组态清单和输出文件.

    输出:
    -----
    CHECK_LIST: conf_short -> (方向列表, 动量列表).
    out_file: 输出文件路径.
    """
    CHECK_LIST = {
        "c24p29": (["x", "y", "z"], [0, 2, 4, 6, 8, -2]),
        # "c32p29": (["x"], [0, 8]),
    }

    script_dir = Path(__file__).resolve().parent
    out_file = script_dir / "03_fig_rep" / "check_2pt.txt"
    return CHECK_LIST, out_file


# ==================== 状态检查 ====================
def get_mom_label(mom_dir: str, mom: int) -> Tuple[str, str]:
    """
    构造 log 标签和结果数据文件中的完整动量标签.

    参数:
    -----
    mom_dir: 动量方向, 如 "x".
    mom: 总体系动量, 负值表示反方向.

    输出:
    -----
    log_mom_label: log 标签, 零动量时为 "P0", 否则如 "Px8".
    result_mom_label: 结果数据标签, 如 "Px8Py0Pz0".
    """
    if mom == 0:
        return "P0", "Px0Py0Pz0"

    momentum = {"x": 0, "y": 0, "z": 0}
    momentum[mom_dir] = mom
    result_mom_label = "".join(
        f"P{direction}{value}"
        for direction, value in momentum.items()
    )
    return f"P{mom_dir}{mom}", result_mom_label


def check_conf(conf_short: str, conf_id: int, mom_dir: str, mom: int) -> str:
    """
    根据结果数据和 log 检查给定组态和动量的状态.

    参数:
    -----
    conf_short: 组态短名.
    conf_id: 组态编号.
    mom_dir: 动量方向.
    mom: 总动量, 零动量表示 cg5, 非零表示 cg5gt.

    输出:
    -----
    state: 结果数据存在时为 "done", 仅 log 存在时为 "problem", log 不存在时为 "undo".
    """
    mom_label, result_mom_label = get_mom_label(mom_dir, mom)
    if mom == 0:
        result_dir = data_root / "cg5" / conf_short / str(conf_id)
        log_dir = data_root / "cg5" / "log" / conf_short / str(conf_id)
    else:
        _, result_subdir, _ = get_momsmear_info(
            conf_short, conf_id, mom_dir, 1 if mom > 0 else -1
        )
        result_dir = (
            data_root / "cg5gt" / result_subdir / conf_short / str(conf_id)
        )
        log_dir = data_root / "cg5gt" / "log" / conf_short / str(conf_id)

    result_file = result_dir / f"NPcorr_{result_mom_label}_conf{conf_id}.npy"
    if result_file.is_file():
        return "done"

    log_file = log_dir / f"{mom_label}_conf{conf_id}.log"
    return "problem" if log_file.is_file() else "undo"


# ==================== 输出格式 ====================
def format_ranges(id_list: List[int], id_step: int) -> List[str]:
    """
    把组态号按步长压缩成区间文本.

    参数:
    -----
    id_list: 有序的组态编号.
    id_step: 组态编号步长.

    输出:
    -----
    range_lines: 每个区间一行的文本, 如 "    11000-11100".
    """
    if not id_list:
        return ["    none"]

    range_list = []
    start = id_list[0]
    prev = id_list[0]
    for conf_id in id_list[1:]:
        if conf_id == prev + id_step:
            prev = conf_id
        else:
            range_list.append((start, prev))
            start = conf_id
            prev = conf_id
    range_list.append((start, prev))

    range_lines = []
    for start, prev in range_list:
        if start == prev:
            range_lines.append(f"    {start}")
        else:
            range_lines.append(f"    {start}-{prev}")
    return range_lines


def format_status(name: str, id_list: List[int], id_step: int) -> List[str]:
    """
    格式化一类状态的组态号和个数.

    参数:
    -----
    name: 状态名称.
    id_list: 该类状态的组态编号.
    id_step: 组态编号步长.

    输出:
    -----
    status_lines: 状态统计文本.
    """
    status_lines = [f"{name}:", "  ID:"]
    status_lines += format_ranges(id_list, id_step)
    status_lines.append(f"  count: {len(id_list)}")
    return status_lines


if __name__ == "__main__":
    t_total_begin = time.perf_counter()

    check_list, out_file = init()

    # 输出中的分隔字符串.
    sep_conf = "=" * 20
    sep_mom = "-" * 10

    report = []
    for conf_short, (mom_dirs, momenta) in check_list.items():
        conf = get_conf(conf_short)
        conf_id_first = conf["conf_id_first"]
        conf_id_step = conf["conf_id_step"]
        conf_id_num = conf["conf_id_num"]

        report.append(sep_conf)
        report.append(conf_short)
        for mom in momenta:
            current_dirs = [""] if mom == 0 else mom_dirs

            for mom_dir in current_dirs:
                mom_label, _ = get_mom_label(mom_dir, mom)

                status = {"done": [], "problem": [], "undo": []}
                for index in range(conf_id_num):
                    conf_id = conf_id_first + index * conf_id_step
                    state = check_conf(conf_short, conf_id, mom_dir, mom)
                    status[state].append(conf_id)

                report.append(sep_mom)
                report.append(f"mom = {mom_label}")
                report.append("")
                # 只写出 count 不为 0 的状态.
                for name, id_list in status.items():
                    if not id_list:
                        continue
                    report += format_status(name, id_list, conf_id_step)
                    report.append("")

    report.append(sep_conf)
    report.append(f"last update: {time.strftime('%F %T')}")

    save_txt(out_file, "\n".join(report) + "\n", name="check_2pt")
    print(f"job done, time={time.perf_counter() - t_total_begin:.0f}s")
