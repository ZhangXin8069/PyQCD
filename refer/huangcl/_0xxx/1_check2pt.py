#!/public/home/huangcl/.venv/bin/python
"""
功能: 统计 cg5/cg5gt 下 NPcorr 作业各 log 的完成状态,
      done = 组态目录存在, 且 log 倒数第二行以 "job done" 开头,
      problem = 组态目录存在, 但 log 不存在或不是正常结束,
      undo = 组态目录不存在.
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


project_dir = Path("/public/home/huangcl/06_deuteron_gluon_pdf")


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
        # "c24p29": (["x", "y", "z"], [0, 8, -8]),
        "c24p29": (["x", "y", "z"], [0, 2, 4, 6, 8, -2]),
        # "c32p29": (["x"], [0, 8]),
    }

    script_dir = Path(__file__).resolve().parent
    out_file = script_dir / "03_fig_rep" / "check_2pt.txt"
    return CHECK_LIST, out_file


# ==================== 状态检查 ====================
def get_mom_label(mom_dir: str, mom: int) -> str:
    """
    由方向和总动量构造 log 的动量标签.

    参数:
    -----
    mom_dir: 动量方向, 如 "x".
    mom: 总体系动量, 负值表示反方向.

    输出:
    -----
    mom_label: 动量标签, 零动量时为 "P0", 否则如 "Px8" 或 "Px-8".
    """
    if mom == 0:
        return "P0"
    return f"P{mom_dir}{mom}"


def check_conf(conf_root: Path, conf_id: int, mom_label: str) -> str:
    """
    根据组态目录和给定动量标签下的 log 检查状态.

    参数:
    -----
    conf_root: 组态目录 02_conf_info/<conf_short>.
    conf_id: 组态编号.
    mom_label: 动量标签.

    输出:
    -----
    state: 组态目录不存在时为 "undo", 日志正常结束时为 "done", 其他情况为 "problem".
    """
    conf_dir = conf_root / str(conf_id)
    if not conf_dir.is_dir():
        return "undo"

    log_file = conf_dir / f"{mom_label}_conf{conf_id}.log"
    if not log_file.is_file():
        return "problem"

    # 与 tail -n 2 | head -n 1 一致, 只有一行时取该行.
    log_lines = log_file.read_text(errors="ignore").splitlines()
    job_line = log_lines[-2] if len(log_lines) >= 2 else (
        log_lines[0] if log_lines else "")
    return "done" if job_line.startswith("job done") else "problem"


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
            if mom == 0:
                func_name = "06_NPcorr_cg5"
                current_dirs = [""]
            else:
                func_name = "05_NPcorr_cg5gt"
                current_dirs = mom_dirs

            conf_root = project_dir / func_name / "02_conf_info" / conf_short
            for mom_dir in current_dirs:
                mom_label = get_mom_label(mom_dir, mom)

                status = {"done": [], "problem": [], "undo": []}
                for index in range(conf_id_num):
                    conf_id = conf_id_first + index * conf_id_step
                    state = check_conf(conf_root, conf_id, mom_label)
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
