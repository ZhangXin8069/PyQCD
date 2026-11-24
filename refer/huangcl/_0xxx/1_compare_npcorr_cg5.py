#!/public/home/huangcl/.venv/bin/python
"""
功能: 读取 cg5/c32p29 的 NP 2pt 数据, 对 t_src 求和后与 gengyq 参考结果比较.
输入:
- 本地: LOCAL_ROOT/<conf_id>/NPcorr_Px0Py0Pz0_conf<conf_id>.npy,
  para: dt, t_src, spin*4.
- 参考: REF_ROOT/NPcorr_spin_mom4-8_conf<conf_id>.npy,
  para: p_snk, p_src, dt, spin*4.
输出: dt 截断到前 N_T 个时间片后, 用 98_tools 的 compare_data 打印逐组态和整体的差异统计及 all_match 判定.
"""

import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np

# ---- 从项目目录 98_tools 导入通用函数 ----
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "98_tools"))
from comparison_data_tools import compare_data  # noqa: E402  # type: ignore


# ==================== 初始化 ====================
def init() -> Tuple[Path, Path, tuple, tuple, int]:
    """
    初始化输入路径和比较范围.

    输出:
    -----
    local_root, ref_root: 本地和参考数据根目录.
    conf_ids: 需要比较的组态编号.
    ref_index: 参考数据中取出的 p_snk, p_src 索引.
    n_t: dt 时间轴截断长度.
    """
    LOCAL_ROOT = Path("/public/home/huangcl/00_data/00_NP_2pt/cg5/c24p29")
    REF_ROOT = Path(
        "/public/home/gengyq/NP/result/spin/"
        "beta6.20_mu-0.2770_ms-0.2400_L24x72"
    )
    CONF_IDS = tuple(range(4050, 6950, 50))
    REF_INDEX = (0, 0)
    N_T = 24
    return LOCAL_ROOT, REF_ROOT, CONF_IDS, REF_INDEX, N_T


# ==================== 数据读取与比较 ====================
def load_pair(
        conf_id: int, local_root: Path, ref_root: Path,
        ref_index: tuple, n_t: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取一个组态的本地和参考数据, 并对本地 t_src 轴求和.

    参数:
    -----
    conf_id: 组态编号.
    local_root, ref_root: 本地和参考数据根目录.
    ref_index: 参考数据中取出的 p_snk, p_src 索引.
    n_t: dt 时间轴截断长度.

    输出:
    -----
    local, ref: 形状相同的复数组, para: dt, spin*4.
    """
    local_file = (
        local_root / str(conf_id)
        / f"NPcorr_Px0Py0Pz0_conf{conf_id}.npy"
    )
    # para: dt, t_src, spin*4
    local_all = np.load(local_file, allow_pickle=False)
    # para: dt, spin*4
    local = local_all.sum(axis=1)[:n_t]

    # ref_file = ref_root / f"NPcorr_spin_mom4-8_conf{conf_id}.npy"
    ref_file = ref_root / f"NPcorr_spin_{conf_id}.npy"
    # para: p_snk, p_src, dt, spin*4
    ref_all = np.load(ref_file, allow_pickle=False)
    # para: dt, spin*4
    ref = ref_all[ref_index][:n_t]
    return local, ref


if __name__ == "__main__":
    t_total_begin = time.perf_counter()

    local_root, ref_root, conf_ids, ref_index, n_t = init()
    print(f"local_root = {local_root}")
    print(f"ref_root = {ref_root}")
    print(f"ref_index = {ref_index}, confs = {len(conf_ids)}, N_T = {n_t}")

    local_sum = []
    ref_sum = []
    for conf_id in conf_ids:
        local, ref = load_pair(
            conf_id, local_root, ref_root, ref_index, n_t)
        print(f"\nconf {conf_id}: local={local.shape}, ref={ref.shape}")
        compare_data(ref, local, tag=f"conf_{conf_id}")
        local_sum.append(local)
        ref_sum.append(ref)

    # para: conf, dt, spin*4
    local_all = np.stack(local_sum)
    # para: conf, dt, spin*4
    ref_all = np.stack(ref_sum)
    print(f"\noverall: local={local_all.shape}, ref={ref_all.shape}")
    compare_data(ref_all, local_all, tag="overall")

    print(f"job done, time={time.perf_counter() - t_total_begin:.0f}s")
