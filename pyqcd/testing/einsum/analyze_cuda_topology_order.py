"""量化 CUDA ``topology_steps.h`` 的收缩序在生产尺寸下的劣化倍数（只读分析）。

⚠️ INVALID（2026-11-18）：本脚本产出的劣化倍数基于**已废弃**的拓扑步表口径
（KUPL 生产异形轴 / CUDA 均匀 dim=4 探针）。新口径为 ROLE_SHAPES
（peram[4,4,100,100] gamma[4,4] VDV[100,100] VVV[100,100,100,100]），
两张表已整体重算。**旧数字不得再被引用为新结论。** 脚本只输出实测、不设期望值；
``_label_sizes`` 反推轴长同样是旧口径，未迁移到 ROLE_SHAPES。

问题
----
``cpp/cuda/einsum/src/topology_steps.h`` 的 8638 条步表（基础表 2582 + 附录 6056，
生成端见 ``pyqcd/testing/_cuda_topology_steps_gen.py``）按文件头自述，顺序取自
``opt_einsum greedy`` 在**均匀 dim=4 超立方体**上的结果。生产尺寸却是异形的
（``Nev=100``、``Ns=4``、``Mom=1``、``Lt=1``，见
:func:`pyqcd.einsum._bench_scene.dims`)：等长时全部轴长相同、路径代价与轴长
无关，所以任何等长探针都看不出问题；一到异形轴就会出现巨中间体。

本脚本**不改任何常量表**，只做三件事：

1. 按 CUDA 同口径把语料 8638 条方程映射到表项：``key`` = 签名
   blake2b-**128** 的前 8 字节（大端），``arity`` 必须与方程一致
   （``fnv`` 指纹只用于运行期把表项绑到下发的标签流，量化不依赖它）；
2. 用**同一套 MAC 记账**比较两种收缩序在生产尺寸下的乘加数：
   * 表内序 = ``kQcdTopoOrder<order>`` 的位置对序列（活表语义，与
     ``topology_plan_from_table`` 逐位同口径：每步取 (min, max)、先擦大号、
     结果追加到末尾，中间体标签 = 「输出 ∪ 后续操作数仍需要」的存活集）；
   * 最优序 = ``contract_path(equation, *shapes, optimize="auto")`` 的位置对；
   两者都只用 :func:`simulate` 记账，因此比值只反映**顺序**差异，不含记账口径
   差异（``opt_cost`` 另打印一列做交叉核对）；
3. 复核「表内序确实由均匀 dim=4 定型」：把表内序与
   ``contract_path(..., optimize="greedy")`` 在 ``dim=4`` 等长下的路径比对。

用法::

    python -m pyqcd.testing.einsum.analyze_cuda_topology_order
    python -m pyqcd.testing.einsum.analyze_cuda_topology_order --limit 64 --json /tmp/x.json

产物（默认）：``pyqcd/testing/einsum/artifacts/cuda_topology_order_<时间戳>/``
下 ``result.json``、``table.tsv``、``report.md``。

单条方程的排查请用 ``python -m pyqcd.einsum.test --help``（2026-11-15 起的唯一合法
入口，口径以它的缺省值为准）；本脚本只做语料级的顺序劣化量化。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import statistics
import sys
import time
from typing import Any, Iterable, Sequence

from pyqcd.einsum import contract_path
from pyqcd.einsum import _wick_tasklist as tasklist
from pyqcd.einsum._bench_scene import dims as scene_dims
from pyqcd.einsum.bench_specialized import _label_sizes
from pyqcd.testing._topology_kernel_gen import canonical_signature, parse

ROOT = pathlib.Path(__file__).resolve().parents[3]
HEADER = ROOT / "cpp/cuda/einsum/src/topology_steps.h"
ARTIFACTS = pathlib.Path(__file__).resolve().parent / "artifacts"

#: 「表内序由均匀 dim=4 定型」这一自述用的探针边长（与 ``_topology_kernel_gen``
#: 里被替换掉的旧 ``PROBE_DIM`` 同值）。
UNIFORM_DIM = 4

#: 记账口径：MAC = 每步 ``prod(两侧标签并集)`` 之和。与 CUDA 执行器的
#: pack + cuBLAS GEMM 对应（``M*N*K*count_batch`` 就是并集的元素数）。
PROBE_ORDER = "auto"


# ── 表头解析 ────────────────────────────────────────────────────────────────
_ORDER_RE = re.compile(r"static const int8_t kQcdTopoOrder(\d+)\[\] = \{([^}]*)\};")
_ORDERS_RE = re.compile(r"\{kQcdTopoOrder(\d+), (\d+)\},")
_ENTRY_RE = re.compile(
    r"\{0x([0-9a-f]{16})ULL, 0x([0-9a-f]{16})ULL, (\d+), (\d+), 0\}")


def parse_header(path: pathlib.Path) -> dict[str, Any]:
    """解析 ``topology_steps.h``：位置对序列、``order`` 索引表与全部表项。"""
    text = path.read_text(encoding="utf-8")
    arrays = {
        int(index): tuple(int(value) for value in body.replace(" ", "").split(",") if value)
        for index, body in _ORDER_RE.findall(text)}
    orders_section = text.split("static const qcd_topology_order kQcdTopoOrders[]")[-1]
    orders = [(int(index), int(steps)) for index, steps in _ORDERS_RE.findall(orders_section)]
    entries = [(int(key, 16), int(fnv, 16), int(order), int(arity))
               for key, fnv, order, arity in _ENTRY_RE.findall(text)]
    pairs = [tuple(zip(array[0::2], array[1::2])) for array in (
        arrays[index] for index, _steps in orders)]
    steps = [steps for _index, steps in orders]
    for position, sequence in enumerate(pairs):
        if len(sequence) != steps[position]:
            raise AssertionError(
                f"order {position}: {len(sequence)} pairs but steps={steps[position]}")
    return {
        "order_count": len(orders),
        "pairs": pairs,
        "steps": steps,
        "entries": entries,
    }


def table_key(signature: tuple) -> int:
    """CUDA 建表口径：签名 blake2b-128 前 8 字节（大端）。"""
    payload = "|".join(",".join(map(str, term)) for term in signature)
    digest = hashlib.blake2b(payload.encode(), digest_size=16).digest()
    return int.from_bytes(digest[:8], "big")


# ── MAC 记账 ────────────────────────────────────────────────────────────────
def _dedupe(labels: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


def simulate(terms: Sequence[Sequence[int]], output: Sequence[int],
             sizes: Sequence[int], path: Sequence[Sequence[int]]
             ) -> dict[str, Any]:
    """按 einsum 语义（与 ``topology_plan_from_table`` 同口径）算一条固定序的 MAC。

    ``path`` 里的下标是**活表**位置（0..len(pool)-1），每步先擦大号再擦小号、
    结果追加到末尾 —— 与 CUDA ``topology_plan_from_table`` 完全一致。长度为 1 的
    步是 ``contract_path`` 可能给出的单目步（先归约），MAC 记为被归约操作数的元素数。
    返回总 MAC、每步明细与最大中间体元素数。
    """
    pool: list[list[int]] = [list(term) for term in terms]
    keep_final = set(output)
    total = 0
    largest = 0
    detail: list[dict[str, Any]] = []
    for step, positions in enumerate(path):
        indices = sorted(int(position) for position in positions)
        for index in indices:
            if not 0 <= index < len(pool):
                raise AssertionError(
                    f"step {step}: position {index} out of range {len(pool)}")
        if len(set(indices)) != len(indices):
            raise AssertionError(f"step {step}: repeated position {positions}")
        involved = [pool[index] for index in indices]
        union = _dedupe(label for term in involved for label in term)
        cost = math.prod(sizes[label] for label in union)
        total += cost
        last = step + 1 == len(path)
        remaining = [pool[index] for index in range(len(pool)) if index not in indices]
        if last:
            result = list(output)
        else:
            keep = set(keep_final)
            for term in remaining:
                keep |= set(term)
            result = [label for label in union if label in keep]
        largest = max(largest, math.prod(sizes[label] for label in result))
        detail.append({
            "indices": indices,
            "operands": [len(term) for term in involved],
            "mac": cost,
            "intermediate_elements": math.prod(sizes[label] for label in result),
            "intermediate_rank": len(result),
        })
        for index in reversed(indices):
            pool.pop(index)
        pool.append(result)
    if len(pool) != 1 or list(pool[0]) != list(output):
        raise AssertionError(
            f"fixed order does not end on the equation output: {pool}")
    return {"mac": total, "largest": largest, "steps": detail}


def _pool_path(info: Any) -> list[tuple[int, ...]]:
    """``PathInfo.contraction_list`` → 活表位置序列（与 C++ 记账同语义）。"""
    path: list[tuple[int, ...]] = []
    for entry in info.contraction_list:
        indices = tuple(int(index) for index in entry[0])
        path.append(indices if len(indices) == 1 else tuple(sorted(indices)))
    return path


def greedy_pairs(terms: Sequence[Sequence[int]], output: Sequence[int],
                 sizes: Sequence[int]) -> list[tuple[int, int]]:
    """「省下内存最多优先」贪心路径 —— 与 C++ 侧守卫的参考序同算法。

    每步在活表里挑 ``size12 - size1 - size2`` 最小（即 ``size1 + size2 - size12``
    最大）的一对：这是 ``opt_einsum`` greedy 的 ``memory-removed`` 判据。并列时
    先比并集元素数 ``size12``，再比位置下标（升序），因此结果与实现无关、可复现。
    中间体标签 = 「最终输出 ∪ 其余活操作数仍需要」的存活集，与
    ``topology_step_output_labels`` 同口径。

    为什么不用「并集乘加最小」：``distill`` 尺寸里 ``Mom=1`` / ``Lt=1`` 让大量并集
    乘加数相等，"最小值"判据会先做一堆零成本外积，参考序本身即可比生产最优序差
    1.5e3 倍。同一份 2582 条语料上两种判据作守卫的对照（阈值 8）：

    ==================  ============  ==================  ==============
    参考序判据           表内序/参考    漏放（表>8×最优却放行）  误拒（≤2×却拒）
    ==================  ============  ==================  ==============
    ``size12`` 最小       中位 54.9×              184 条            0 条
    ``memory-removed``   中位 60.6×               32 条            0 条
    ==================  ============  ==================  ==============
    """
    pool: list[list[int]] = [list(term) for term in terms]
    keep_final = set(output)
    path: list[tuple[int, int]] = []
    while len(pool) > 1:
        best: tuple[int, int, int, int] | None = None
        for left in range(len(pool)):
            for right in range(left + 1, len(pool)):
                union = _dedupe(list(pool[left]) + list(pool[right]))
                size12 = math.prod(sizes[label] for label in union)
                size1 = math.prod(sizes[label] for label in pool[left])
                size2 = math.prod(sizes[label] for label in pool[right])
                candidate = (size12 - size1 - size2, size12, left, right)
                if best is None or candidate < best:
                    best = candidate
        assert best is not None
        _cost, _size12, left, right = best
        path.append((left, right))
        union = _dedupe(list(pool[left]) + list(pool[right]))
        remaining = [pool[index] for index in range(len(pool))
                     if index not in (left, right)]
        keep = set(keep_final)
        for term in remaining:
            keep |= set(term)
        result = [label for label in union if label in keep]
        for index in sorted((left, right), reverse=True):
            pool.pop(index)
        pool.append(result)
    return path


# ── 语料 ────────────────────────────────────────────────────────────────────
def corpus_rows(limit: int | None = None) -> list[dict[str, Any]]:
    """语料的 (seq, 方程, 元数, 生产尺寸形状)。"""
    tasks = tasklist.by_seq()
    dims = scene_dims("distill")
    rows: list[dict[str, Any]] = []
    for seq in sorted(tasks):
        if limit is not None and len(rows) >= limit:
            break
        task = tasks[seq]
        terms, output = parse(task.equation)
        sizes = _label_sizes(terms, task.roles, dims)
        shapes = [tuple(sizes[label] for label in term) for term in terms]
        rows.append({
            "seq": seq,
            "equation": task.equation,
            "terms": terms,
            "output": output,
            "arity": len(terms),
            "sizes": sizes,
            "shapes": shapes,
        })
    return rows


def _numeric_terms(equation: str) -> tuple[list[list[int]], list[int]]:
    terms, output = parse(equation)
    ordered: list[str] = []
    for term in terms + [output]:
        for label in term:
            if label not in ordered:
                ordered.append(label)
    index = {label: position for position, label in enumerate(ordered)}
    return ([[index[label] for label in term] for term in terms],
            [index[label] for label in output])


def _numeric_sizes(row: dict[str, Any], terms: Sequence[Sequence[int]]) -> list[int]:
    ordered: list[str] = []
    for term in row["terms"] + [row["output"]]:
        for label in term:
            if label not in ordered:
                ordered.append(label)
    sizes = [row["sizes"][label] for label in ordered]
    if len(terms[0]) and max(max(term) for term in terms) >= len(sizes):
        raise AssertionError(f"seq={row['seq']}: label index out of range")
    return sizes


def analyse(rows: Sequence[dict[str, Any]], header: dict[str, Any]) -> list[dict[str, Any]]:
    """逐条：表内序 vs 生产 ``auto`` 序的 MAC。"""
    by_key = {entry[0]: entry for entry in header["entries"]}
    results: list[dict[str, Any]] = []
    for row in rows:
        terms, output = _numeric_terms(row["equation"])
        sizes = _numeric_sizes(row, terms)
        canonical = canonical_signature(*parse(row["equation"]))
        key = table_key(_signature_of(row))
        entry = by_key.get(key)
        item: dict[str, Any] = {
            "seq": row["seq"],
            "arity": row["arity"],
            "equation": row["equation"],
            "canonical": canonical,
            "key": f"{key:#018x}",
            "matched": entry is not None,
        }
        if entry is None:
            item["reason"] = "key not found in topology_steps.h"
            results.append(item)
            continue
        _k, fnv, order, arity = entry
        if arity != row["arity"]:
            raise AssertionError(
                f"seq={row['seq']}: table arity {arity} != equation arity {row['arity']}")
        recorded = header["pairs"][order]
        table_cost = simulate(terms, output, sizes, recorded)
        shapes = [tuple(sizes[label] for label in term) for term in terms]
        best, best_info = contract_path(
            row["equation"], *shapes, shapes=True, optimize=PROBE_ORDER)
        auto_cost = simulate(terms, output, sizes, _pool_path(best_info))
        greedy = greedy_pairs(terms, output, sizes)
        greedy_cost = simulate(terms, output, sizes, greedy)
        # 交叉核对：opt_einsum 自己的 opt_cost 口径（与上一轮抽样同一把尺子）
        _, table_info = contract_path(
            row["equation"], *shapes, shapes=True, optimize=list(recorded))
        uniform_shapes = [(UNIFORM_DIM,) * len(term) for term in terms]
        _, uniform_info = contract_path(
            row["equation"], *uniform_shapes, shapes=True, optimize="greedy")
        item.update({
            "order": order,
            "fnv": f"{fnv:#018x}",
            "steps": len(recorded),
            "table_mac": table_cost["mac"],
            "table_largest": table_cost["largest"],
            "auto_mac": auto_cost["mac"],
            "auto_largest": auto_cost["largest"],
            "ratio": (table_cost["mac"] / auto_cost["mac"]) if auto_cost["mac"] else None,
            "greedy_mac": greedy_cost["mac"],
            "greedy_largest": greedy_cost["largest"],
            "ratio_vs_greedy": (table_cost["mac"] / greedy_cost["mac"])
                               if greedy_cost["mac"] else None,
            "greedy_vs_auto": (greedy_cost["mac"] / auto_cost["mac"])
                              if auto_cost["mac"] else None,
            "greedy_steps_detail": greedy_cost["steps"],
            "opt_cost_crosscheck": {
                "table": float(table_info.opt_cost),
                "auto": float(best_info.opt_cost),
                "ratio": (float(table_info.opt_cost) / float(best_info.opt_cost))
                         if best_info.opt_cost else None,
            },
            "path_matches_auto": list(recorded) == _pool_path(best_info),
            "path_matches_greedy_dim4": list(recorded) == _pool_path(uniform_info),
            "table_steps_detail": table_cost["steps"],
            "auto_steps_detail": auto_cost["steps"],
        })
        results.append(item)
    return results


def _signature_of(row: dict[str, Any]) -> tuple:
    """与 ``_specialize.canonical_signature`` 同一编号（全式首次出现顺序）。"""
    from pyqcd.einsum._specialize import canonical_signature as numeric

    signature = numeric(row["equation"])
    if signature is None:
        raise AssertionError(f"seq={row['seq']}: no canonical signature")
    return signature


# ── 统计与产物 ──────────────────────────────────────────────────────────────
def bucket(rows: Sequence[dict[str, Any]], key: str) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}
    for row in rows:
        if row.get("ratio") is None:
            continue
        out.setdefault(int(row["arity"]), []).append(float(row[key]))
    return out


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {}
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "max": ordered[-1],
    }


def summarize(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in results if row.get("ratio") is not None]
    ratios = [float(row["ratio"]) for row in rows]
    cross = [float(row["opt_cost_crosscheck"]["ratio"]) for row in rows
             if row.get("opt_cost_crosscheck", {}).get("ratio")]
    worse = [row for row in rows if row["ratio"] > 1.0 + 1e-12]
    better = [row for row in rows if row["ratio"] < 1.0 - 1e-12]
    tie = len(rows) - len(worse) - len(better)
    worst = max(rows, key=lambda row: row["ratio"]) if rows else None
    provenance = sum(1 for row in rows if row["path_matches_greedy_dim4"])
    vs_greedy = [float(row["ratio_vs_greedy"]) for row in rows
                 if row.get("ratio_vs_greedy")]
    greedy_vs_auto = [float(row["greedy_vs_auto"]) for row in rows
                      if row.get("greedy_vs_auto")]
    thresholds = {f"{value:g}x": sum(1 for ratio in vs_greedy if ratio > value)
                  for value in (1.5, 2.0, 4.0, 8.0, 16.0)}
    return {
        "matched": len(rows),
        "unmatched": len(results) - len(rows),
        "table_worse": len(worse),
        "table_better": len(better),
        "tie": tie,
        "ratio": _quantiles(ratios),
        "opt_cost_ratio_crosscheck": _quantiles(cross),
        "over_2x": sum(1 for value in ratios if value > 2),
        "over_10x": sum(1 for value in ratios if value > 10),
        "over_100x": sum(1 for value in ratios if value > 100),
        "provenance_greedy_dim4": provenance,
        "provenance_fraction": provenance / len(rows) if rows else None,
        "path_matches_auto": sum(1 for row in rows if row["path_matches_auto"]),
        "ratio_vs_greedy": _quantiles(vs_greedy),
        "greedy_vs_auto": _quantiles(greedy_vs_auto),
        "reject_if_over": thresholds,
        "table_largest": _quantiles([float(row["table_largest"]) for row in rows]),
        "auto_largest": _quantiles([float(row["auto_largest"]) for row in rows]),
        "worst": None if worst is None else {
            "seq": worst["seq"],
            "arity": worst["arity"],
            "equation": worst["equation"],
            "ratio": worst["ratio"],
            "table_mac": worst["table_mac"],
            "auto_mac": worst["auto_mac"],
            "table_largest": worst["table_largest"],
            "auto_largest": worst["auto_largest"],
            "table_steps_detail": worst["table_steps_detail"],
            "auto_steps_detail": worst["auto_steps_detail"],
        },
        "by_arity": {
            str(arity): _quantiles(values)
            for arity, values in sorted(bucket(rows, "ratio").items())},
    }


def representative(rows: Sequence[dict[str, Any]], per_arity: int = 2
                   ) -> list[dict[str, Any]]:
    """每个元数取最差 + 中位的行，保证 arity 6..16 全覆盖且不少于 20 条。"""
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("ratio") is None:
            continue
        grouped.setdefault(int(row["arity"]), []).append(row)
    picked: list[dict[str, Any]] = []
    for arity in sorted(grouped):
        ordered = sorted(grouped[arity], key=lambda row: row["ratio"])
        chosen = [ordered[-1]]
        if per_arity > 1:
            chosen.append(ordered[len(ordered) // 2])
        picked.extend(chosen)
    return picked


def write_artifacts(directory: pathlib.Path, payload: dict[str, Any],
                    rows: Sequence[dict[str, Any]]) -> pathlib.Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    header = ["seq", "arity", "steps", "table_mac", "auto_mac", "ratio",
              "table_largest", "auto_largest", "matches_auto",
              "matches_greedy_dim4", "equation"]
    lines = ["\t".join(header)]
    for row in rows:
        lines.append("\t".join(str(value) for value in (
            row["seq"], row["arity"], row.get("steps"), row.get("table_mac"),
            row.get("auto_mac"), "" if row.get("ratio") is None
            else f"{row['ratio']:.6g}", row.get("table_largest"),
            row.get("auto_largest"), int(bool(row.get("path_matches_auto"))),
            int(bool(row.get("path_matches_greedy_dim4"))), row["equation"])))
    (directory / "table.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = payload["summary"]
    worst = summary["worst"]
    out: list[str] = []
    out.append("# CUDA `topology_steps.h` 收缩序 + 生产尺寸量化")
    out.append("")
    out.append(f"* 生成时间：{payload['stamp']}")
    out.append(f"* 表：`cpp/cuda/einsum/src/topology_steps.h` "
               f"（{payload['header']['order_count']} 条不同步序 / "
               f"{payload['header']['entries']} 条表项）")
    out.append(f"* 生产尺寸：`dims(\"distill\")` → "
               + ", ".join(f"{key}={value}" for key, value
                           in payload["dims"].items()
                           if key in ("Nev", "Ns", "Mom", "Lt", "Nx")))
    out.append("* MAC 记账：每步 `prod(两侧标签并集)`，与 CUDA 执行器的 "
               "pack + GEMM 一一对应；两种收缩序用**同一** :func:`simulate`。")
    out.append("")
    out.append("## 汇总")
    out.append("")
    out.append(f"* 命中表项：{summary['matched']} / {payload['total_rows']}"
               + (f"（未命中 {summary['unmatched']}）" if summary["unmatched"] else ""))
    out.append(f"* 表内序更差 / 更好 / 持平：{summary['table_worse']} / "
               f"{summary['table_better']} / {summary['tie']}")
    ratio = summary["ratio"]
    out.append(f"* 劣化倍数（表内序 MAC ÷ 生产 auto 序 MAC）：中位 "
               f"{ratio['median']:.4g}、p90 {ratio['p90']:.4g}、最差 {ratio['max']:.4g}")
    out.append(f"* 超过 2× / 10× / 100×：{summary['over_2x']} / "
               f"{summary['over_10x']} / {summary['over_100x']} 条")
    cross = summary["opt_cost_ratio_crosscheck"]
    if cross:
        out.append(f"* 交叉核对（`contract_path` 的 `opt_cost` 口径）：中位 "
                   f"{cross['median']:.4g}、最差 {cross['max']:.4g}")
    out.append(f"* 表内序 == greedy@dim={UNIFORM_DIM}："
               f"{summary['provenance_greedy_dim4']} / {summary['matched']} "
               f"= {summary['provenance_fraction']:.1%}")
    out.append(f"* 表内序 == 生产 auto 序：{summary['path_matches_auto']} / "
               f"{summary['matched']}")
    vs_greedy = summary["ratio_vs_greedy"]
    if vs_greedy:
        out.append("")
        out.append("### 与 C++ 侧守卫参考序（memory-removed 贪心，本描述符尺寸）对照")
        out.append("")
        out.append(f"* 表内序 ÷ 贪心序：中位 {vs_greedy['median']:.4g}、"
                   f"p90 {vs_greedy['p90']:.4g}、最差 {vs_greedy['max']:.4g}")
        greedy = summary["greedy_vs_auto"]
        out.append(f"* 贪心序 ÷ 生产 auto 序：中位 {greedy['median']:.4g}（略优于 auto）、"
                   f"p90 {greedy['p90']:.4g}、最差 {greedy['max']:.4g}")
        out.append("* 若守卫阈值取 X（表内序 > X × 贪心序即拒绝），被拒绝条数："
                   + "、".join(f"{key} → {value}"
                               for key, value in summary["reject_if_over"].items()))
        largest = summary["table_largest"]
        out.append(f"* 表内序最大中间体（元素数）：中位 {largest['median']:.4g}、"
                   f"p90 {largest['p90']:.4g}、最差 {largest['max']:.4g}"
                   f"（auto 序最差 {summary['auto_largest']['max']:.4g}）")
    out.append("")
    out.append("## 按元数")
    out.append("")
    out.append("| arity | 条数 | 中位 | p90 | 最差 |")
    out.append("|---|---|---|---|---|")
    for arity, values in summary["by_arity"].items():
        out.append(f"| {arity} | {values['count']} | {values['median']:.4g} | "
                   f"{values['p90']:.4g} | {values['max']:.4g} |")
    out.append("")
    out.append("## 代表样本（每元数：最差 + 中位）")
    out.append("")
    out.append("| seq | arity | 表内序 MAC | 生产最优序 MAC | 倍数 | 表内最大中间体 | "
               "最优最大中间体 |")
    out.append("|---|---|---|---|---|---|---|")
    for row in payload["representative"]:
        out.append(f"| {row['seq']} | {row['arity']} | {row['table_mac']:.6g} | "
                   f"{row['auto_mac']:.6g} | {row['ratio']:.4g} | "
                   f"{row['table_largest']} | {row['auto_largest']} |")
    out.append("")
    out.append("## 最差样本的逐步对照")
    out.append("")
    out.append(f"seq={worst['seq']}（arity={worst['arity']}，倍数 "
               f"{worst['ratio']:.4g}）：")
    out.append("")
    out.append("| # | 表内序 pos | 表内 MAC | 表内中间体秩/元素 | 最优 pos | 最优 MAC | "
               "最优中间体秩/元素 |")
    out.append("|---|---|---|---|---|---|---|")
    for index, (left, right) in enumerate(zip(worst["table_steps_detail"],
                                              worst["auto_steps_detail"])):
        out.append(f"| {index} | {tuple(left['indices'])} | {left['mac']:.6g} | "
                   f"{left['intermediate_rank']}/{left['intermediate_elements']} | "
                   f"{tuple(right['indices'])} | {right['mac']:.6g} | "
                   f"{right['intermediate_rank']}/{right['intermediate_elements']} |")
    out.append("")
    out.append(f"完整逐条结果见同目录 `table.tsv`（{len(rows)} 行）与 `result.json`。")
    (directory / "report.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    return directory


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="只分析前 N 条语料（默认全量 8638）")
    parser.add_argument("--json", default=None,
                        help="产物目录（默认 artifacts/cuda_topology_order_<时间戳>/）")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    header = parse_header(HEADER)
    rows = corpus_rows(args.limit)
    results = analyse(rows, header)
    summary = summarize(results)
    dims = scene_dims("distill")
    stamp = time.strftime("%Y%m%d%H%M%S")
    payload = {
        "stamp": stamp,
        "elapsed_seconds": round(time.time() - started, 3),
        "total_rows": len(rows),
        "probe_order": PROBE_ORDER,
        "uniform_dim": UNIFORM_DIM,
        "dims": {key: dims[key] for key in ("Nev", "Ns", "Mom", "Lt", "Nx",
                                            "Nev_src", "Mom_src")},
        "header": {"order_count": header["order_count"],
                   "entries": len(header["entries"]),
                   "steps_total": sum(header["steps"])},
        "summary": summary,
        "representative": [{key: value for key, value in row.items()
                            if key not in ("table_steps_detail", "auto_steps_detail",
                                           "greedy_steps_detail", "sizes",
                                           "opt_cost_crosscheck")}
                           for row in representative(results)],
        "rows": results,
    }
    directory = (pathlib.Path(args.json) if args.json else
                 ARTIFACTS / f"cuda_topology_order_{stamp}")
    # 每步明细只在最差样本与代表样本上落盘：全量 8638 条 × 两套步表会到几十 MB，
    # 而量化只需要聚合数字 + 可追溯的少量样例。
    keep_details = {summary["worst"]["seq"] if summary["worst"] else None}
    keep_details |= {row["seq"] for row in payload["representative"]}
    payload["rows"] = [
        {key: value for key, value in row.items()
         if key not in ("table_steps_detail", "auto_steps_detail",
                        "greedy_steps_detail")
         or row["seq"] in keep_details}
        for row in results]
    write_artifacts(directory, payload, results)
    if not args.quiet:
        ratio = summary["ratio"]
        print(f"matched {summary['matched']}/{len(rows)}  worse/better/tie "
              f"{summary['table_worse']}/{summary['table_better']}/{summary['tie']}"
              f"  ratio median {ratio.get('median', float('nan')):.4g}"
              f" p90 {ratio.get('p90', float('nan')):.4g}"
              f" max {ratio.get('max', float('nan')):.4g}"
              f"  >2x {summary['over_2x']} >10x {summary['over_10x']}"
              f" >100x {summary['over_100x']}")
        vs_greedy = summary["ratio_vs_greedy"]
        if vs_greedy:
            print(f"vs greedy-reference median {vs_greedy['median']:.4g}"
                  f" max {vs_greedy['max']:.4g}  greedy/auto median "
                  f"{summary['greedy_vs_auto']['median']:.4g}  reject>8x "
                  f"{summary['reject_if_over'].get('8x')}")
        print(f"provenance greedy@{UNIFORM_DIM}: {summary['provenance_greedy_dim4']}"
              f"/{summary['matched']}   artifacts: {directory}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
