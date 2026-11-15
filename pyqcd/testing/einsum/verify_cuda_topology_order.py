"""在真实 GPU 上核对：「表内序病态即 fail closed」守卫是否按量化预测动作。

被验证的行为（``cpp/cuda/einsum/src/apply_contract.cu`` 的
:func:`topology_order_acceptable`）：

* 表内序在**本描述符的真实尺寸**下相对「最小并集乘加优先」贪心参考序劣化
  超过 ``kTopologyOrderTolerance``（= 8）倍时，原生入口返回 status 4，由上层
  回落带真实 path 搜索的通用路径；
* 其余情况逐位保持原行为（同一条 ``kQcdTopoOrder*`` 步序列），数值不变。

判据来源：:mod:`pyqcd.testing.einsum.analyze_cuda_topology_order` 落盘的
``result.json`` 里每条语料的 ``ratio_vs_greedy``（Python 侧同一算法）。因此本
脚本同时是 **C++ 贪心与 Python 贪心的一致性检查**：预测与实测的任何分歧都会
被逐条列出。

用法::

    python -m pyqcd.testing.einsum.verify_cuda_topology_order
    python -m pyqcd.testing.einsum.verify_cuda_topology_order \
        --dtypes complex128,complex64,float64,float32 --sample 6
    python -m pyqcd.testing.einsum.verify_cuda_topology_order --limit 64 --quiet

需要 cupy 与已构建的 ``libpyqcdeinsum_cuda.so``；缺任一项按 SKIP 退出（返回 0）。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any, Sequence

import numpy as np

from pyqcd.einsum import _kernel_cuda
from pyqcd.einsum import _kernel_kupl
from pyqcd.einsum import _wick_tasklist as tasklist
from pyqcd.einsum._bench_scene import dims as scene_dims
from pyqcd.einsum.bench_specialized import _label_sizes
from pyqcd.einsum.cuda._program import topology_binding
from pyqcd.testing._topology_kernel_gen import parse

ARTIFACTS = pathlib.Path(__file__).resolve().parent / "artifacts"

#: 与 C++ ``kTopologyOrderTolerance`` 同一个数；不一致即为守卫阈值漂移。
TOLERANCE = 8

#: 数值容差（双精度 1e-9 / 单精度 1e-4），与仓库其它 einsum 契约同一把尺子。
NUMERIC_TOLERANCE = {"complex128": 1e-9, "float64": 1e-9,
                     "complex64": 1e-4, "float32": 1e-4}
DEFAULT_DTYPES = ("complex128", "complex64", "float64", "float32")


def latest_artifact() -> pathlib.Path:
    candidates = sorted(ARTIFACTS.glob("cuda_topology_order_*"))
    if not candidates:
        raise SystemExit("找不到量化产物；先运行 "
                         "python -m pyqcd.testing.einsum.analyze_cuda_topology_order")
    return candidates[-1]


def predictions(directory: pathlib.Path) -> dict[int, dict[str, Any]]:
    payload = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    return {int(row["seq"]): row for row in payload["rows"]
            if row.get("ratio_vs_greedy") is not None}


def _operands(equation: str, roles: tuple[str, ...], dtype: str, seed: int,
              dims: dict[str, int]) -> list[np.ndarray]:
    terms, _output = parse(equation)
    sizes = _label_sizes(terms, roles, dims)
    shapes = [tuple(sizes[label] for label in term) for term in terms]
    rng = np.random.default_rng(seed)
    complex_dtype = np.dtype(dtype).kind == "c"
    out: list[np.ndarray] = []
    for shape in shapes:
        if complex_dtype:
            out.append((rng.standard_normal(shape)
                        + 1j * rng.standard_normal(shape)).astype(dtype))
        else:
            out.append(rng.standard_normal(shape).astype(dtype))
    return out


def _error(actual: np.ndarray, expected: np.ndarray) -> float:
    scale = float(np.max(np.abs(expected))) or 1.0
    return float(np.max(np.abs(actual - expected))) / scale


def run(directory: pathlib.Path, dtypes: Sequence[str], sample: int | None,
        limit: int | None, quiet: bool) -> int:
    try:
        import cupy
    except ImportError:
        print("SKIP: cupy 不可用")
        return 0
    if topology_binding() is None:
        print("SKIP: libpyqcdeinsum_cuda.so 未导出 qcd_cuda_topology_apply"
              "（先 bash cpp/cuda/einsum/make-nv.sh）")
        return 0

    table = predictions(directory)
    dims = scene_dims("distill")
    tasks = tasklist.by_seq()
    seqs = sorted(seq for seq in table if seq in tasks)
    if limit is not None:
        seqs = seqs[:limit]
    keep = [seq for seq in seqs if table[seq]["ratio_vs_greedy"] <= TOLERANCE]
    drop = [seq for seq in seqs if table[seq]["ratio_vs_greedy"] > TOLERANCE]
    if sample is not None:
        def pick(pool: list[int]) -> list[int]:
            if len(pool) <= sample:
                return pool
            step = len(pool) / sample
            return [pool[min(len(pool) - 1, int(index * step))] for index in range(sample)]
        selected = sorted(set(pick(keep)) | set(pick(drop)))
    else:
        selected = seqs

    report: dict[str, Any] = {
        "artifact": str(directory),
        "stamp": time.strftime("%Y%m%d%H%M%S"),
        "tolerance": TOLERANCE,
        "dims": {key: dims[key] for key in ("Nev", "Ns", "Mom", "Lt", "Nx")},
        "predicted_keep": len(keep),
        "predicted_decline": len(drop),
        "runs": [],
    }
    exit_code = 0
    for dtype in dtypes:
        numeric = NUMERIC_TOLERANCE[dtype]
        accepted = rejected = mismatch = failed = 0
        worst = 0.0
        worst_seq = None
        failures: list[str] = []
        started = time.time()
        for seq in selected:
            row = table[seq]
            task = tasks[seq]
            predicted_hit = row["ratio_vs_greedy"] <= TOLERANCE
            try:
                operands = _operands(task.equation, task.roles, dtype, seq, dims)
                devices = [cupy.asarray(item) for item in operands]
                _kernel_kupl._NEGATIVE_CACHE.clear()
                _kernel_cuda.LAST_INFO.clear()
                _kernel_cuda.LAST_FAILURE.update({"reason": "", "status": None})
                result = _kernel_cuda.try_topology_apply(task.equation, devices)
                if result is None:
                    rejected += 1
                    if predicted_hit:
                        mismatch += 1
                        failures.append(
                            f"seq={seq}: 预测放行却被拒 ({_kernel_cuda.LAST_FAILURE.get('reason')})")
                else:
                    accepted += 1
                    if not predicted_hit:
                        mismatch += 1
                        failures.append(
                            f"seq={seq}: 预测拒绝却被执行 (ratio_vs_greedy="
                            f"{row['ratio_vs_greedy']:.4g}, tolerance={TOLERANCE})")
                    expected = np.einsum(task.equation, *operands, optimize="greedy")
                    error = _error(cupy.asnumpy(result), expected)
                    if error > worst:
                        worst, worst_seq = error, seq
                    if error > numeric:
                        failed += 1
                        failures.append(f"seq={seq}: rel={error:.3e} > {numeric:g}")
                del operands, devices, result
                cupy.get_default_memory_pool().free_all_blocks()
            except Exception as error:                     # noqa: BLE001
                failed += 1
                failures.append(f"seq={seq}: {type(error).__name__}: {error}")
        entry = {
            "dtype": dtype,
            "runs": len(selected),
            "accepted": accepted,
            "rejected": rejected,
            "prediction_mismatch": mismatch,
            "numeric_failures": failed,
            "worst_relative_error": worst,
            "worst_seq": worst_seq,
            "tolerance": numeric,
            "elapsed_seconds": round(time.time() - started, 3),
            "failures": failures[:20],
        }
        report["runs"].append(entry)
        if mismatch or failed:
            exit_code = 1
        if not quiet:
            print(f"{dtype:11s} runs={len(selected):5d} accepted={accepted:5d} "
                  f"rejected={rejected:5d} mismatch={mismatch:3d} "
                  f"numeric_fail={failed:3d} worst_rel={worst:.3e} "
                  f"({time.time() - started:.1f}s)")
            for line in failures[:10]:
                print("   ", line)
    report["selected"] = len(selected)
    report["seqs"] = selected if len(selected) <= 64 else []
    target = directory / f"verify_{report['stamp']}.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not quiet:
        print(f"report: {target}")
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default=None, help="量化产物目录（默认最新）")
    parser.add_argument("--dtypes", default=",".join(DEFAULT_DTYPES),
                        help="逗号分隔的 dtype 列表")
    parser.add_argument("--sample", type=int, default=None,
                        help="每个 dtype 只取 N 条放行 + N 条拒绝（默认全量）")
    parser.add_argument("--limit", type=int, default=None, help="只看前 N 条语料")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    directory = (pathlib.Path(args.artifact) if args.artifact else latest_artifact())
    dtypes = tuple(item.strip() for item in args.dtypes.split(",") if item.strip())
    unknown = [item for item in dtypes if item not in NUMERIC_TOLERANCE]
    if unknown:
        parser.error(f"unknown dtype(s): {unknown}")
    return run(directory, dtypes, args.sample, args.limit, args.quiet)


if __name__ == "__main__":
    sys.exit(main())
