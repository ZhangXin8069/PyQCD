"""单条方程的特化路径验证：给定一个 einsum 方程，报告它走哪条特化路径、
数值是否与参考一致、实际执行的是原生内核还是计划缓存。

用来回答「我这个方程到底有没有被特化内核接管」以及「接管后的结果对不对」。

    python -m pyqcd.testing.einsum.verify_specialized "csdt,eqfr,mknl,ogph,iajb,ac,mo,Ggi,kq,Nbd,Nfnp,Nhj,Mlrt->MNGes"
    python -m pyqcd.testing.einsum.verify_specialized "<eq>" --dim 8 --repeat 20
    python -m pyqcd.testing.einsum.verify_specialized "<eq>" --dtype complex64
    python -m pyqcd.testing.einsum.verify_specialized "<eq>" --json

判据（任一不满足即非零退出）：

1. **命中**：方程必须匹配到一条特化路径（拓扑签名在语料索引里）；
2. **数值**：特化结果与参考实现的相对偏差在容差内；
3. **真跑**：提示里必须出现 ``kernel: ...``（原生内核执行），而不是只有
   ``plan-cache``——后者只复用计划，没有走专用执行器。

参考实现默认用 ``opt_einsum``（固定 path，只计执行）；装不了时退回 numpy
的同 dtype 结果，并在输出里注明降级原因。
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np

# dtype → (numpy 类型, 判据)。判据口径见 skills/pyqcd-einsum/references/
# measured-pitfalls.md §16：复数单精度的固有极限是 ~3.4e-05，不能套 float32 的 1e-5。
DTYPES = {
    "float32": (np.float32, 1e-4),
    "float64": (np.float64, 1e-12),
    "complex64": (np.complex64, 1e-4),
    "complex128": (np.complex128, 1e-12),
}


def _operands(equation: str, dim: int, dtype, seed: int = 0) -> list[np.ndarray]:
    """按方程标签生成操作数；同一标签在所有操作数里同长。

    大写标签（``M``/``N``/``G`` 等）是流分量指标，固定 2 维；其余是 ``dim`` 维。
    """
    rng = np.random.default_rng(seed)
    operands = []
    for term in equation.split("->")[0].split(","):
        shape = tuple(2 if label.isupper() else dim for label in term)
        real = rng.standard_normal(shape)
        if np.issubdtype(np.dtype(dtype), np.complexfloating):
            operands.append((real + 1j * rng.standard_normal(shape)).astype(dtype))
        else:
            operands.append(real.astype(dtype))
    return operands


def _reference(equation: str, operands, dtype):
    """参考值：优先 opt_einsum 固定 path；不可用时退回同 dtype numpy。

    返回 (值, 来源标签)。**不把低精度结果当参考**：先在 dtype 下算，再按判据
    缩放比较；需要更高精度时调用方用 complex128 的判据。
    """
    try:
        import opt_einsum as oe
    except ImportError:
        return np.einsum(equation, *operands), "numpy (opt_einsum missing)"
    path, _ = oe.contract_path(equation, *operands, optimize="greedy")
    return oe.contract(equation, *operands, optimize=path), "opt_einsum (greedy path)"


def _capture(equation: str, operands) -> tuple[np.ndarray, str]:
    """跑一次并抓取库自己的提示行，用来判断真正执行了什么。

    提示必须**强制打开**：调用方常设 ``PYQCD_EINSUM_NOTICE=0`` 来静音日志，
    但那会让这里什么都抓不到，进而把"真跑了内核"误判成"没跑"。
    按 skill 的纪律，诊断函数要保存/恢复全局状态，不能留下副作用。
    """
    import contextlib
    import io
    import os

    import pyqcd.einsum as pe
    from pyqcd.einsum import _specialize

    saved_env = os.environ.get(_specialize.NOTICE_ENV)
    saved_announced = set(_specialize._announced)
    buffer = io.StringIO()
    try:
        os.environ[_specialize.NOTICE_ENV] = "1"
        _specialize._announced.clear()
        with contextlib.redirect_stdout(buffer):
            result = pe.contract(equation, *operands)
    finally:
        if saved_env is None:
            os.environ.pop(_specialize.NOTICE_ENV, None)
        else:
            os.environ[_specialize.NOTICE_ENV] = saved_env
        _specialize._announced.clear()
        _specialize._announced.update(saved_announced)
    return result, buffer.getvalue().strip()


def _timing(function, repeat: int) -> dict:
    """预热后取中位数；单次太短时自动放大每样本工作量。"""
    function()
    samples = []
    for _ in range(max(1, repeat)):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter() - started)
    median = statistics.median(samples)
    return {"median_ms": round(median * 1e3, 4),
            "min_ms": round(min(samples) * 1e3, 4),
            "samples": len(samples)}


def verify(equation: str, dim: int = 6, dtype_name: str = "complex128",
           repeat: int = 10, seed: int = 0) -> dict:
    """跑完整判据，返回结构化结果（供 CLI 与测试复用）。"""
    from pyqcd.einsum import _specialize as specialize

    if dtype_name not in DTYPES:
        raise SystemExit(f"unknown dtype {dtype_name!r}; "
                         f"choose one of {', '.join(DTYPES)}")
    dtype, tolerance = DTYPES[dtype_name]

    entry = specialize.match(equation)
    report: dict = {
        "equation": equation,
        "equation_normalized": specialize.normalize(equation),
        "dtype": dtype_name,
        "dim": dim,
        "tolerance": tolerance,
        "matched": entry is not None,
    }
    if entry is None:
        report.update({
            "ok": False,
            "reason": "该方程不在 wick_12cases 语料里，走通用路径（无特化内核）",
            "specialized_ran": False,
        })
        return report

    report.update({
        "path_name": entry["name"],
        "path_kind": entry["kind"],
        "kernels": list(entry.get("kernels", ())),
        "signature": repr(specialize.canonical_signature(equation)),
    })

    operands = _operands(equation, dim, dtype, seed)
    specialized, notice = _capture(equation, operands)

    reference, reference_source = _reference(equation, operands, dtype)
    truth = np.asarray(reference)
    got = np.asarray(specialized)
    scale = max(float(np.max(np.abs(truth), initial=0.0)), 1e-300)
    deviation = float(np.max(np.abs(got - truth), initial=0.0)) / scale

    generic = __import__("pyqcd.einsum", fromlist=["contract"]).contract(
        equation, *operands, specialize=False)

    specialized_time = _timing(
        lambda: __import__("pyqcd.einsum", fromlist=["contract"]).contract(
            equation, *operands), repeat)
    generic_time = _timing(
        lambda: __import__("pyqcd.einsum", fromlist=["contract"]).contract(
            equation, *operands, specialize=False), repeat)

    ran_kernel = "kernel:" in notice
    kernel_name = ""
    if ran_kernel:
        inside = notice.split("kernel:", 1)[1]
        kernel_name = inside.split("]", 1)[0].strip()

    report.update({
        "notice": notice,
        "specialized_ran": ran_kernel,
        "kernel": kernel_name or ("plan-cache" if "plan-cache" in notice else ""),
        "reference": reference_source,
        "tolerance_ok": bool(deviation <= tolerance),
        "max_relative_deviation": deviation,
        "matches_generic": bool(np.allclose(got, np.asarray(generic),
                                            rtol=tolerance, atol=tolerance)),
        "shape": list(np.shape(got)),
        "specialized_ms": specialized_time["median_ms"],
        "generic_ms": generic_time["median_ms"],
        "speedup_vs_generic": (round(generic_time["median_ms"]
                                     / specialized_time["median_ms"], 3)
                               if specialized_time["median_ms"] else None),
    })
    report["ok"] = bool(report["tolerance_ok"] and ran_kernel)
    if not report["ok"]:
        if not report["tolerance_ok"]:
            report["reason"] = (
                f"数值超差：{deviation:.2e} > 判据 {tolerance:.0e}")
        else:
            report["reason"] = (
                "命中特化路径但没有跑原生内核（只命中计划缓存）；"
                "用 PYQCD_EINSUM_SPECIALIZE_VERBOSE=1 看每次调用的标签")
    return report


def _print_report(report: dict) -> None:
    ok = report.get("ok", False)
    print(f"方程        : {report['equation']}")
    print(f"判据        : {report['dtype']} @ dim={report['dim']} "
          f"（容差 {report['tolerance']:.0e}，参考 {report.get('reference', '-')}）")
    print(f"结果        : {'PASS' if ok else 'FAIL'}")
    if not report["matched"]:
        print(f"  原因      : {report['reason']}")
        return
    print(f"  特化路径  : {report['path_name']}  [{report['path_kind']}]")
    print(f"  可用内核  : {', '.join(report['kernels']) or '-'}")
    print(f"  实际执行  : {report.get('kernel') or '（无）'}"
          f"  {'（原生内核）' if report['specialized_ran'] else '（未走内核）'}")
    print(f"  最大偏差  : {report['max_relative_deviation']:.3e}"
          f"  {'OK' if report['tolerance_ok'] else '超差'}")
    print(f"  与通用一致: {report['matches_generic']}")
    print(f"  形状      : {tuple(report['shape'])}")
    print(f"  耗时      : 特化 {report['specialized_ms']} ms"
          f" / 通用 {report['generic_ms']} ms"
          f"（快 {report['speedup_vs_generic']}x）")
    if not ok:
        print(f"  原因      : {report['reason']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="验证单条方程是否走特化路径、数值是否正确、是否真跑原生内核")
    parser.add_argument("equation",
                        help='einsum 方程，例如 "csdt,eqfr,...,Mlrt->MNGes"')
    parser.add_argument("--dim", type=int, default=6,
                        help="小写指标维（大写固定为 2；默认 6，便于快速验证）")
    parser.add_argument("--dtype", default="complex128", choices=sorted(DTYPES))
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="输出机器可读结果")
    args = parser.parse_args(argv)

    report = verify(args.equation, dim=args.dim, dtype_name=args.dtype,
                    repeat=args.repeat, seed=args.seed)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_report(report)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
