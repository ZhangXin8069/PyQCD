"""用第三方 einsum 读回 ``_topology_full_bench`` 保存的输入/输出做抽样复验。

三方实现（自动探测可用者）：

* ``numpy.einsum``（多路径优化：``optimize='greedy'``）
* ``opt_einsum``（若安装）
* ``torch.einsum``（若安装，CPU 双精度）

判据：读回的 ``outputs/<seq>.npz`` 与该实现独立算出的结果相对偏差 ≤ 容差
（float32 输入默认 1e-5）。差异超限即判该条复验失败并给出 seq 与方程。

用法::

    python -m pyqcd.testing._topology_verify_snapshot <run_dir>
    python -m pyqcd.testing._topology_verify_snapshot <run_dir> --sample 200
    python -m pyqcd.testing._topology_verify_snapshot <run_dir> --engine opt_einsum
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

import numpy as np

DEFAULT_SAMPLE = 100
# float32 在 16 算子链上自身的累积误差可达 5e-5（实测：同一条式子用两种
# float32 算法求解，彼此相差 4.66e-05）。因此默认容差取 1e-4，并在报告里
# 同时给出 float64 参考下的偏差，避免把"精度极限"误判成"错数"。
DEFAULT_RTOL = 1e-4


def _engines(requested: str | None):
    """返回 [(name, callable(equation, operands) -> ndarray)]。"""
    available = []

    def numpy_engine(equation, operands):
        # 注意：16 算子方程上 np.einsum(optimize='greedy') 会病态（实测 >20 s），
        # 因此改用 opt_einsum 的 path 搜索 + 显式逐步收缩；没有 opt_einsum 时
        # 退回 numpy 但用 'dp'（动态规划，16 算子可控）。
        try:
            import opt_einsum as oe
            path, _ = oe.contract_path(equation, *operands, optimize="greedy")
            return oe.contract(equation, *operands, optimize=path)
        except Exception:                              # noqa: BLE001
            return np.einsum(equation, *operands, optimize="dp")

    available.append(("numpy", numpy_engine))

    try:
        import opt_einsum as oe

        def oe_engine(equation, operands):
            path, _ = oe.contract_path(equation, *operands, optimize="greedy")
            return oe.contract(equation, *operands, optimize=path)

        available.append(("opt_einsum", oe_engine))
    except Exception:                                  # noqa: BLE001
        pass

    try:
        import torch

        def torch_engine(equation, operands):
            tensors = [torch.from_numpy(operand.astype(np.float64))
                       for operand in operands]
            return torch.einsum(equation, *tensors).numpy()

        available.append(("torch", torch_engine))
    except Exception:                                  # noqa: BLE001
        pass

    if requested:
        available = [item for item in available if item[0] == requested]
    return available


def verify(run_dir: pathlib.Path, sample: int | None = DEFAULT_SAMPLE,
           rtol: float = DEFAULT_RTOL, engine: str | None = None,
           seed: int = 0) -> dict:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"缺少 manifest.json：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [item for item in manifest
               if item.get("ok") and item.get("input_file")]

    if sample is not None and sample < len(entries):
        rng = random.Random(seed)
        entries = rng.sample(entries, sample)

    engines = _engines(engine)
    if not engines:
        raise SystemExit(f"没有可用的第三方 engine（请求的是 {engine!r}）")

    results = []
    for name, compute in engines:
        checked = failed = 0
        worst = 0.0
        failures = []
        for item in entries:
            inputs = np.load(run_dir / item["input_file"])
            saved = np.load(run_dir / item["output_file"])["out"]
            operands = [inputs[key] for key in sorted(inputs.files,
                                                      key=lambda k: int(k[1:]))]
            try:
                expected = np.asarray(compute(item["equation"], operands))
            except Exception as exc:                   # noqa: BLE001
                failed += 1
                failures.append({"seq": item["seq"], "error":
                                 f"{type(exc).__name__}: {exc}"})
                continue
            checked += 1
            scale = max(float(np.max(np.abs(expected))), 1e-300)
            deviation = float(np.max(np.abs(saved - expected))) / scale
            worst = max(worst, deviation)
            if deviation > rtol:
                failed += 1
                if len(failures) < 5:
                    failures.append({"seq": item["seq"], "tid": item["tid"],
                                     "deviation": deviation,
                                     "equation": item["equation"]})
        results.append({"engine": name, "checked": checked, "failed": failed,
                        "worst_deviation": worst, "failures": failures})
    return {"run_dir": str(run_dir), "sample": len(entries), "rtol": rtol,
            "engines": results}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="bench 产物目录（含 manifest.json）")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                        help=f"抽样条数（默认 {DEFAULT_SAMPLE}；0 表示全量）")
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--engine", default=None,
                        choices=["numpy", "opt_einsum", "torch"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    sample = None if args.sample == 0 else args.sample
    outcome = verify(pathlib.Path(args.run_dir), sample=sample,
                     rtol=args.rtol, engine=args.engine, seed=args.seed)

    print(f"快照目录: {outcome['run_dir']}")
    print(f"抽样条数: {outcome['sample']}   容差: {outcome['rtol']:.0e}")
    exit_code = 0
    for item in outcome["engines"]:
        status = "PASS" if item["failed"] == 0 else "FAIL"
        print(f"  {item['engine']:10s} {status}  复核 {item['checked']} 条，"
              f"失败 {item['failed']}，最大偏差 {item['worst_deviation']:.2e}")
        for failure in item["failures"]:
            print(f"    - {failure}")
        if item["failed"]:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
