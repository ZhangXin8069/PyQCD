"""Tables and the markdown report for ``bench_public_ab.py``.

Reads raw timings produced by ``bench_public_ab.py`` and writes
``docs/report_einsum_benchmark_20261109.md``; the same tables can be printed
standalone for any JSON file. The per-case record of the published run lives in
``logs/einsum-benchmark-20261109.txt`` (``logs/`` tracks ``*.txt`` only).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

SIZE_ORDER = {"small": 0, "medium": 1, "large": 2}


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def fmt_seconds(value):
    if value >= 1.0:
        return f"{value:.2f} s"
    if value >= 0.01:
        return f"{value * 1e3:.1f} ms"
    return f"{value * 1e6:.0f} µs"


def table(payload, families=None):
    rows = payload["records"]
    grouped = defaultdict(dict)
    for record in rows:
        grouped[record["family"]][record["size"]] = record
    families = families or sorted(grouped, key=lambda name: min(
        SIZE_ORDER[size] for size in grouped[name]))
    lines = ["| Family | Size | `pyqcd.einsum` | `opt_einsum` (`dp`) | Rate | Round spread |",
             "|---|---|---|---|---|---|"]
    for family in families:
        for size in sorted(grouped[family], key=lambda s: SIZE_ORDER.get(s, 9)):
            record = grouped[family][size]
            spread = (record["ratio_max"] - record["ratio_min"]) / record["ratio"]
            lines.append(
                f"| {family} | {size} | {fmt_seconds(record['pyqcd_s'])} | "
                f"{fmt_seconds(record['opt_s'])} | "
                f"{'**' if record['ratio'] >= 1.5 else ''}{record['ratio']:.2f}×"
                f"{'**' if record['ratio'] >= 1.5 else ''} | {spread:.0%} |")
    return "\n".join(lines)


def summary(payload):
    rows = payload["records"]
    ratios = [record["ratio"] for record in rows]
    below_one = [record for record in rows if record["ratio"] < 0.95]
    returns = [record for record in rows if 0.95 <= record["ratio"] < 1.05]
    wins = [record for record in rows if record["ratio"] >= 1.05]
    fastest = max(rows, key=lambda record: record["ratio"])
    return {
        "cases": len(rows),
        "held": len(returns),
        "wins": len(wins),
        "below": len(below_one),
        "median": sorted(ratios)[len(ratios) // 2],
        "lowest": min(rows, key=lambda record: record["ratio"]),
        "highest": fastest,
        "below_list": below_one,
        "max_error": max(record["rel_error"] for record in rows),
    }


def _data_path(name: str) -> pathlib.Path:
    """Raw JSON location: the local run archive, or ``PYQCD_BENCH_DIR``."""
    override = os.environ.get("PYQCD_BENCH_DIR")
    if override:
        return pathlib.Path(override) / name
    return pathlib.Path(__file__).resolve().parents[3] / "data" / "runs" \
        / "einsum-benchmark-20261109" / name


def write_document() -> pathlib.Path:
    """Render the full report from the two JSON files and write it to docs/."""
    return _write_document()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json", nargs="*")
    parser.add_argument("--write", action="store_true",
                        help="write docs/report_einsum_benchmark_20261109.md")
    args = parser.parse_args()
    if args.write:
        print(f"wrote {write_document()}")
        return 0
    for path in args.json:
        payload = load(path)
        stats = summary(payload)
        print(f"### {payload['device'].upper()} — {payload['device_name']}")
        print()
        print(f"pyqcd {payload['pyqcd']}, opt_einsum {payload['opt_einsum']}, "
              f"rounds {payload['rounds']}, cases {stats['cases']}")
        print()
        print(table(payload))
        print()
        print(f"median rate {stats['median']:.2f}×; "
              f"faster in {stats['wins']}, level (0.95–1.05) in {stats['held']}, "
              f"slower in {stats['below']}")
        if stats["below_list"]:
            print("\nslower than the reference:")
            for record in stats["below_list"]:
                print(f"- {record['family']} / {record['size']}: {record['ratio']:.2f}×")
        print(f"\nmax relative error across all cases: {stats['max_error']:.2e}")
        print()


# ------------------------------------------------------------------ report text

import pathlib  # noqa: E402


def slower_lines(stats):
    if not stats["below_list"]:
        return "- none"
    return "\n".join(
        f"- {record['family']} / {record['size']}: {record['ratio']:.2f}× "
        f"({fmt_seconds(record['pyqcd_s'])} vs {fmt_seconds(record['opt_s'])})"
        for record in sorted(stats["below_list"], key=lambda r: r["ratio"]))


def family_lines(payload):
    rows = payload["records"]
    families = {}
    for record in rows:
        families.setdefault(record["family"], []).append(record["ratio"])
    lines = ["| Family | Best | Worst | Geometric mean |", "|---|---|---|---|"]
    for family, ratios in sorted(families.items(), key=lambda item: -max(item[1])):
        geo = 1.0
        for ratio in ratios:
            geo *= ratio
        geo **= 1.0 / len(ratios)
        lines.append(f"| {family} | {max(ratios):.2f}× | {min(ratios):.2f}× | {geo:.2f}× |")
    return "\n".join(lines)


def _write_document() -> pathlib.Path:
    base = pathlib.Path(__file__).resolve().parents[3]
    docs_dir = base / "docs"
    cpu = load(_data_path("cpu.json"))
    gpu = load(_data_path("gpu.json"))
    cpu_stats, gpu_stats = summary(cpu), summary(gpu)

    text = f"""# qcd-einsum 基准报告（0.2.4）

本报告只做测量，不改动任何库代码。脚本与原始数据见
`pyqcd/testing/einsum/bench_public_ab.py`（测量）与
`pyqcd/testing/einsum/bench_public_ab_tables.py`（本报告生成器）；本次运行的逐 case 原始记录
在 `logs/einsum-benchmark-20261109.txt`（`logs/` 只跟踪 `*.txt`），完整 JSON 可用脚本重跑生成。

## 口径

| 项 | 设置 |
|---|---|
| 被测版本 | pyqcd {cpu['pyqcd']}（PyPI 安装）/ opt_einsum {cpu['opt_einsum']}（`dp` 策略） |
| 轮数 | 每 case **{cpu['rounds']} 轮**，每轮交替"谁先跑"（后跑方会白拿 10–20% 的假优势） |
| 采样 | 单次不足 0.2 s 的 case 在轮内重复取最好值，避免被计时粒度主导 |
| 尺寸 | 每个家族三档：small / medium / large |
| 数据 | **物理结构数据**而非 iid 高斯：传播子式时间指数衰减、低秩 perambulator 块、近幺正的 SU(3) link；通用 case 另配纯随机输入 |
| 正确性 | 每个 case 在计时前先与 opt_einsum 对齐（按相对最大范数判定，避免近零元素被 `atol` 误判） |
| 覆盖 | CPU {cpu_stats['cases']} 个 case、GPU {gpu_stats['cases']} 个 case |
| 机器 | 4 物理核（8 线程）+ RTX 4060 Laptop 8 GB |

轮间波动（同一 case 内 ratio 的 max/min 相对差）中位数意义上 CPU 约 12%、GPU 约 15%；
个别 GPU case 因显存池与时钟波动可达 30–70%，这类数字只能看数量级。

## CPU（42 个 case）

{table(cpu)}

**汇总**：中位倍率 {cpu_stats['median']:.2f}×；比参考快 {cpu_stats['wins']} 项、持平
（0.95–1.05）{cpu_stats['held']} 项、更慢 {cpu_stats['below']} 项。全部 {cpu_stats['cases']}
项的相对误差上限 {cpu_stats['max_error']:.1e}。

按家族聚合：

{family_lines(cpu)}

比参考慢的项：

{slower_lines(cpu_stats)}

## GPU（42 个 case）

{table(gpu)}

**汇总**：中位倍率 {gpu_stats['median']:.2f}×；比参考快 {gpu_stats['wins']} 项、持平
{gpu_stats['held']} 项、更慢 {gpu_stats['below']} 项。全部 {gpu_stats['cases']} 项的
相对误差上限 {gpu_stats['max_error']:.1e}。

按家族聚合：

{family_lines(gpu)}

比参考慢的项：

{slower_lines(gpu_stats)}

## 结论

1. **纯 GEMM 与链式收缩在三档尺寸上都持平**：CPU 0.92–1.06×（多数落在 0.95–1.01×）、
   GPU 0.98–1.07×。大尺寸（4096² 实数、2048² 复数、3×3072 链）两边都在 BLAS 上限，
   差异在噪声内。
2. **批量轴是最大的结构性优势**：CPU 上批量 GEMM 三档 11–12×、ellipsis 批量 8.6–9.3×、
   大量小批量 2.3–2.7×；GPU 上同族只有 0.84–1.06×，因为 cupy 自己已经会批量调用 cuBLAS。
3. **两个物理 kernel 的收益随尺寸放大**：CPU 核子 2pt 6.8×/6.0×/16.8×、γ 夹心
   10.5×/9.4×/20.7×；GPU 核子 2pt 2.1×/2.6×/2.7×、γ 夹心 9.5×/10.2×，大尺寸回落到
   3.7×（8192×1024 时受 2.1 GB 工作区带宽限制）。CPU 侧倍数随尺寸增长，说明收益来自
   "省掉中间体"而非常数开销。
4. **逐元素与全约简**：CPU 逐元素 1.25–1.63×（无求和标签直接交 backend），GPU 全约简
   1.6–2.1×；两者都不涉及 BLAS，收益来自不落地中间体。
5. **仍比参考慢的项**（本报告只记录，未改代码）：
   - `hyper-edge`（`ij,jk,ik->i`）small/medium CPU 0.79×/0.90×，大尺寸回到 0.95×；
   - GPU `many small batches` 0.74–0.91×（1024–8192 个 32–96² 小矩阵），是唯一在生产
     尺寸仍落后的一族；
   - GPU `matmul complex` small 0.59×，但该 case 轮间波动 70%，为噪声；
   - `chain 4 ops` large 0.95×、`batch × matrix` small 0.92×，均在噪声边界。
6. **诊断口径**：这些"慢"的绝对值都在 0.3–11 ms，属于固定开销与路径选择区间；
   生产形状（≥100 ms）上没有一项落后。

## 复现

```bash
python -m pyqcd.testing.einsum.bench_public_ab cpu --rounds 7 --out cpu.json
python -m pyqcd.testing.einsum.bench_public_ab gpu --rounds 7 --out gpu.json
python -m pyqcd.testing.einsum.bench_public_ab_tables --write
```

脚本会在计时前打印 `pyqcd.__version__` 与 `pyqcd.__file__`，避免把 `site-packages` 里的
旧版本当成被测对象。
"""
    out = docs_dir / "report_einsum_benchmark_20261109.md"
    out.write_text(text, encoding="utf-8")
    return out


if __name__ == "__main__":
    raise SystemExit(main())


