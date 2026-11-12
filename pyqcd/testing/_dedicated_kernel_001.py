"""专用内核第 1 条：图纸 ``refer/sush/wick_12cases.md:9142``。

方程（逐字）::

    cwdx,eufv,msnt,okpl,igjh,qarb,ac,mo,Ggi,Hkq,su,bd,fnp,Mhj,Mlr,tvx->MGHew

内核不做形状推断：维度全部来自传入数组的 ``.shape``。收缩顺序是写死的常量。
自检：``python -m pyqcd.testing._dedicated_kernel_001``
"""

from __future__ import annotations

import time

import numpy as np

EQUATION = ("cwdx,eufv,msnt,okpl,igjh,qarb,ac,mo,Ggi,Hkq,"
            "su,bd,fnp,Mhj,Mlr,tvx->MGHew")
TERMS = EQUATION.split("->")[0].split(",")
OUTPUT = EQUATION.split("->")[1]


def kernel(*operands: np.ndarray) -> np.ndarray:
    """按固定顺序逐对收缩：每步只保留"输出指标 + 后续还会用到的指标"。

    数学依据（einsum 定义）：
      某指标若不再出现在任何剩余操作数、也不在输出里，就在本步求和掉；
      否则必须保留（后续还要用，或它是输出）。
    这样中间张量的阶数不会爆炸——之前保留全部自由轴会退化成巨型张量。
    """
    if len(operands) != len(TERMS):
        raise ValueError(f"expected {len(TERMS)} operands, got {len(operands)}")

    pending = list(zip(TERMS, operands))
    while len(pending) > 1:
        left_labels, left_data = pending.pop(0)
        right_labels, right_data = pending.pop(0)
        later_labels = "".join(labels for labels, _ in pending)
        keep: list[str] = []
        for label in left_labels + right_labels:
            if label in keep:
                continue
            if label in OUTPUT or label in later_labels:
                # 还要用（输出 或 后续操作数）→ 保留
                keep.append(label)
            # 否则本步求和掉
        merged = np.einsum(f"{left_labels},{right_labels}->{''.join(keep)}",
                           left_data, right_data, optimize=False)
        pending.insert(0, ("".join(keep), merged))

    labels, data = pending[0]
    keep = [ch for ch in OUTPUT if ch in labels]
    if len(keep) != len(labels):
        data = np.einsum(f"{labels}->{''.join(keep)}", data, optimize=False)
    elif keep != list(labels):
        data = np.transpose(data, [labels.index(ch) for ch in keep])
    return data


def reference(*operands: np.ndarray) -> np.ndarray:
    return np.einsum(EQUATION, *operands, optimize="greedy")


def make_operands(dim: int = 3, external: int = 2, seed: int = 0,
                  dtype=np.float32) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(tuple(external if ch.isupper() else dim
                                     for ch in term)).astype(dtype)
            for term in TERMS]


def timeit(function, rounds: int = 20) -> float:
    best = float("inf")
    for _ in range(rounds):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def verify(dim: int = 3, external: int = 2) -> dict:
    operands = make_operands(dim, external)
    got = kernel(*operands)
    expected = reference(*operands)
    scale = max(float(np.max(np.abs(expected))), 1e-30)
    return {
        "shape": tuple(got.shape),
        "expected_shape": tuple(expected.shape),
        "deviation": float(np.max(np.abs(got - expected))) / scale,
        "kernel_seconds": timeit(lambda: kernel(*operands)),
        "reference_seconds": timeit(lambda: reference(*operands)),
    }


def main() -> int:
    print(f"专用内核：{EQUATION}")
    ok_all = True
    for dim in (2, 3, 4):
        r = verify(dim=dim)
        ok = r["shape"] == r["expected_shape"] and r["deviation"] <= 1e-5
        ok_all = ok_all and ok
        ratio = r["kernel_seconds"] / r["reference_seconds"]
        print(f"  dim={dim} 输出={r['shape']} 偏差={r['deviation']:.2e} "
              f"专用={r['kernel_seconds']*1e3:7.3f}ms 通用={r['reference_seconds']*1e3:7.3f}ms "
              f"比值={ratio:5.2f}x {'ok' if ok else 'FAIL'}")
    print(f"  结论: {'PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
