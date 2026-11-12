#!/usr/bin/env python3
"""生成「所有支持的特化路径」附录（中英双语）。

输出 ``docs/{en,zh}/appendix_specialized.md``，内容全部由代码与语料**自动统计**，
不手写数字，保证与实现同步。

用法：python pyqcd/testing/_gen_specialized_appendix.py
"""
from __future__ import annotations

import pathlib
from collections import Counter

ROOT = pathlib.Path("/home/zhangxin/PyQCD")


def collect() -> dict:
    import pyqcd.einsum as pe
    import pyqcd.einsum._wick_tasklist as tasks
    from pyqcd.einsum import _specialize, _topology_registry

    info = pe.build_info()
    by_case = Counter(task.case.split(" ")[0] for task in tasks.TASKS)
    by_arity = Counter(task.arity for task in tasks.TASKS)
    registered = [(entry["name"], entry["kind"], key)
                  for key, entry in _specialize.REGISTRY.items()]
    return {
        "version": __import__("pyqcd").__version__,
        "backend": info.get("backend"),
        "array_backend": info.get("array_backend"),
        "total": tasks.N_TASKS,
        "digests": _topology_registry.N_ENTRIES,
        "buckets": len(_topology_registry._BUCKETS),
        "by_case": dict(sorted(by_case.items())),
        "by_arity": dict(sorted(by_arity.items())),
        "registered": registered,
        "base": tasks.base_table_count(),
    }


EN = """# Appendix: specialized paths

Every contraction listed here is dispatched by **topology signature** (label
independent), so `ij,jk->ik` and `ij,jb->ib` take the same path. Counts are
generated from the corpus and the registry, not typed by hand.

* pyqcd version: `{version}` (backend `{backend}`, array backend `{array_backend}`)
* corpus size: **{total}** contractions ({base} base table + {total_delta} appendix variants)
* topology index: **{digests}** signatures in **{buckets}** buckets

## 1. Corpus coverage by case

| Case | Contractions |
|---|---:|
{case_rows}

## 2. Coverage by operand count

| Operands | Contractions |
|---:|---:|
{arity_rows}

## 3. Literal registry (hand-listed shapes)

These are matched by exact subscript string first, before topology lookup.
`kind` reports what actually runs: `kernel(...)` means a native specialized
kernel, `plan-cache` means only the reusable plan is shared.

| Name | Kind | Subscripts |
|---|---|---|
{registry_rows}

## 4. What runs, and how to tell

| Situation | Log line | Speed |
|---|---|---|
| Topology hit with native kernel | `专用路径: topology:<digest> [kernel: kupl_dedicated]` | 2–4× vs generic |
| Topology hit, no kernel | `专用路径: ... [plan-cache]` | ≈ generic |
| No specialization | `通用路径: 未命中特化形态 (...)` | baseline |

Both notices print **once per shape per process**. Disable them with
`PYQCD_EINSUM_NOTICE=0`; make the specialized notice fire on *every* call with
`PYQCD_EINSUM_SPECIALIZE_VERBOSE=1`.

## 5. Verifying the list yourself

```bash
python -m pyqcd.einsum._specialize                 # registered shapes + self-check
python -m pyqcd.testing._topology_full_bench --dim 3  # run the whole corpus
python -m pyqcd.testing._topology_perf_multidim       # multi-size speed-up
```
"""

ZH = """# 附录：所有支持的特化路径

本附录列出的收缩全部按**拓扑签名**分派（与字母无关），因此 `ij,jk->ik` 与
`ij,jb->ib` 走同一条路径。数字由语料与注册表自动生成，不是手写。

* pyqcd 版本：`{version}`（后端 `{backend}`，数组后端 `{array_backend}`）
* 语料规模：**{total}** 条收缩（基础表 {base} + 附录变体 {total_delta}）
* 拓扑索引：**{digests}** 个签名，分 **{buckets}** 个桶

## 1. 语料覆盖（按情形）

| 情形 | 收缩条数 |
|---|---:|
{case_rows}

## 2. 覆盖（按算子元数）

| 算子数 | 收缩条数 |
|---:|---:|
{arity_rows}

## 3. 字面注册表（手工列出的形态）

这些先按**精确字符串**匹配，再走拓扑查表。`kind` 表示实际执行什么：
`kernel(...)` 是原生专用内核，`plan-cache` 只复用计划。

| 名称 | 类型 | 方程 |
|---|---|---|
{registry_rows}

## 4. 怎么判断走了哪条路

| 情形 | 日志 | 速度 |
|---|---|---|
| 拓扑命中且有原生内核 | `专用路径: topology:<摘要> [kernel: kupl_dedicated]` | 比通用快 2–4× |
| 拓扑命中但无内核 | `专用路径: ... [plan-cache]` | 约等于通用 |
| 未命中特化 | `通用路径: 未命中特化形态 (...)` | 基线 |

两种提示都是**每形态每进程一次**。用 `PYQCD_EINSUM_NOTICE=0` 关闭；
用 `PYQCD_EINSUM_SPECIALIZE_VERBOSE=1` 让特化提示**每次命中**都打印。

## 5. 自行核验

```bash
python -m pyqcd.einsum._specialize                 # 列出注册形态并自检
python -m pyqcd.testing._topology_full_bench --dim 3  # 全语料跑一遍
python -m pyqcd.testing._topology_perf_multidim       # 多尺寸加速比
```
"""


def render(template: str, data: dict) -> str:
    case_rows = "\n".join(f"| {case} | {count} |"
                          for case, count in data["by_case"].items())
    arity_rows = "\n".join(f"| {arity} | {count} |"
                           for arity, count in data["by_arity"].items())
    registry_rows = "\n".join(
        f"| {name} | `{kind}` | `{subscripts}` |"
        for name, kind, subscripts in data["registered"])
    return template.format(
        **data, case_rows=case_rows, arity_rows=arity_rows,
        registry_rows=registry_rows,
        total_delta=data["total"] - data["base"])


def main() -> int:
    data = collect()
    for language, template in (("en", EN), ("zh", ZH)):
        target = ROOT / "docs" / language / "appendix_specialized.md"
        target.write_text(render(template, data), encoding="utf-8")
        print(f"wrote {target} ({target.stat().st_size} bytes)")

    # 两份 index.rst 的 toctree 各加一行
    for language in ("en", "zh"):
        index = ROOT / "docs" / language / "index.rst"
        text = index.read_text(encoding="utf-8")
        if "appendix_specialized" in text:
            continue
        text = text.replace("   appendix\n", "   appendix\n   appendix_specialized\n", 1)
        index.write_text(text, encoding="utf-8")
        print(f"toctree updated: {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
