#!/usr/bin/env python3
"""生成「所有支持的特化路径」附录（中英双语）。

输出 ``docs/{en,zh}/appendix_specialized.md``。清单**逐条列出** wick_12cases
语料的全部收缩（当前 8638 条），内容由代码自动统计，不手写，保证与实现同步。

用法：python pyqcd/testing/_gen_specialized_appendix.py
"""
from __future__ import annotations

import pathlib
from collections import Counter, defaultdict

ROOT = pathlib.Path("/home/zhangxin/PyQCD")


def collect() -> dict:
    import pyqcd.einsum as pe
    import pyqcd.einsum._wick_tasklist as tasks
    from pyqcd.einsum import _topology_registry

    info = pe.build_info()
    by_case = Counter(task.case.split(" ")[0] for task in tasks.TASKS)
    by_arity = Counter(task.arity for task in tasks.TASKS)

    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for task in tasks.TASKS:
        grouped[task.case.split(" ")[0]].append((task.seq, task.equation))

    return {
        "version": __import__("pyqcd").__version__,
        "backend": info.get("backend"),
        "array_backend": info.get("array_backend"),
        "total": tasks.N_TASKS,
        "digests": _topology_registry.N_ENTRIES,
        "buckets": len(_topology_registry._BUCKETS),
        "by_case": dict(sorted(by_case.items())),
        "by_arity": dict(sorted(by_arity.items())),
        "grouped": dict(sorted(grouped.items())),
        "base": tasks.base_table_count(),
    }


EN = """# Appendix: every supported specialized path

This appendix lists **all {total} contractions** that dispatch to a specialized
kernel. Matching is by **topology signature**, not by string: the label names
are irrelevant, so `ij,jk->ik` and `ij,jb->ib` take the same path, and
whitespace or output-order differences do not matter.

The list is generated from the `wick_12cases` corpus, not typed by hand. Run
`python -m pyqcd.testing._gen_specialized_appendix` to regenerate it.

* pyqcd version: `{version}` (backend `{backend}`, array backend `{array_backend}`)
* specialized paths: **{total}** ({base} base table + {base_delta} appendix variants)
* distinct topology signatures: **{digests}**, indexed in **{buckets}** buckets

There is no separate literal registry any more. An earlier release carried five
hand-listed shapes — `gauge-propagator(E4)`, `gamma-sandwich(E5)`,
`sush-2pt(S1)`, `sush-3pt-munu(S2)`, `sush-2pt-mu(S3)`. Those were leftovers
from the retired dedicated kernels, were not part of this corpus, and have been
removed. Everything below is served by the current topology-specialized kernels.

## 1. Coverage by case

| Case | Contractions |
|---|---:|
{case_rows}

## 2. Coverage by operand count

| Operands | Contractions |
|---:|---:|
{arity_rows}

## 3. How to tell which path ran

Both notices print **once per shape per process**, and follow the locale (Chinese
on `zh_*`, English otherwise — force it with `PYQCD_EINSUM_LANG=zh|en`).

| Situation | Log line | Speed |
|---|---|---|
| Corpus path, native kernel ran | `[pyqcd.einsum] specialized path: topology:<digest> [kernel: kupl_dedicated] ...` | 2–4× vs generic |
| Corpus path, kernel unavailable | `[pyqcd.einsum] specialized path: topology:<digest> [plan-cache] ...` | ≈ generic |
| Outside the corpus | `[pyqcd.einsum] generic path: no specialized shape matched (...)` | baseline |

Disable both with `PYQCD_EINSUM_NOTICE=0`; make the specialized notice fire on
**every** call with `PYQCD_EINSUM_SPECIALIZE_VERBOSE=1`.

## 4. The complete list

Grouped by corpus section. The number before each entry is its stable sequence
id (`seq`, 1–{total}); the entry itself is the exact equation from
`refer/sush/wick_12cases.md`.

{path_sections}

## 5. Verifying the list yourself

```bash
python -m pyqcd.einsum._specialize                    # self-check: all must hit
python -m pyqcd.testing._topology_full_bench --dim 3  # run the whole corpus
python -m pyqcd.testing._topology_perf_multidim       # multi-size speed-up
```
"""

ZH = """# 附录：所有支持的特化路径

本附录**逐条列出**全部 {total} 条会走特化内核的收缩。匹配依据是**拓扑签名**，
不是字符串：字母叫什么无关紧要，因此 `ij,jk->ik` 与 `ij,jb->ib` 走同一条路径，
空白与输出顺序的差异同样不影响命中。

清单由 `wick_12cases` 语料自动生成，不是手写。重新生成：
`python -m pyqcd.testing._gen_specialized_appendix`。

* pyqcd 版本：`{version}`（后端 `{backend}`，数组后端 `{array_backend}`）
* 特化路径：**{total}** 条（基础表 {base} + 附录变体 {base_delta}）
* 不同拓扑签名：**{digests}** 个，分 **{buckets}** 个桶

旧的 5 条手工字面表（`gauge-propagator(E4)`、`gamma-sandwich(E5)`、
`sush-2pt(S1)`、`sush-3pt-munu(S2)`、`sush-2pt-mu(S3)`）**已删除**——它们是退役
专用内核的遗留登记项，既不在本语料内，也会让人误以为支持的特化路径只有 5 条。
下面全部由当前拓扑特化内核服务。

## 1. 语料覆盖（按情形）

| 情形 | 收缩条数 |
|---|---:|
{case_rows}

## 2. 覆盖（按算子元数）

| 算子数 | 收缩条数 |
|---:|---:|
{arity_rows}

## 3. 怎么判断走了哪条路

两种提示都是**每形态每进程一次**，且跟随 locale（`zh_*` 说中文，其余说英文；
可用 `PYQCD_EINSUM_LANG=zh|en` 强制指定）。

| 情形 | 日志 | 速度 |
|---|---|---|
| 语料路径，原生内核跑成 | `[pyqcd.einsum] 专用路径: topology:<摘要> [kernel: kupl_dedicated] ...` | 比通用快 2–4× |
| 语料路径，内核不可用 | `[pyqcd.einsum] 专用路径: topology:<摘要> [plan-cache] ...` | 约等于通用 |
| 语料之外 | `[pyqcd.einsum] 通用路径: 未命中特化形态 (...)` | 基线 |

用 `PYQCD_EINSUM_NOTICE=0` 关闭全部提示；
用 `PYQCD_EINSUM_SPECIALIZE_VERBOSE=1` 让特化提示**每次命中**都打印。

## 4. 完整清单

按语料小节分组。每条前面的数字是稳定序号（`seq`，1–{total}），
后面是从 `refer/sush/wick_12cases.md` 取出的原始方程。

{path_sections}

## 5. 自行核验

```bash
python -m pyqcd.einsum._specialize                    # 自检：8638 条必须全部命中
python -m pyqcd.testing._topology_full_bench --dim 3  # 全语料跑一遍
python -m pyqcd.testing._topology_perf_multidim       # 多尺寸加速比
```
"""


def render(template: str, data: dict, heading: str) -> str:
    case_rows = "\n".join(f"| {case} | {count} |"
                          for case, count in data["by_case"].items())
    arity_rows = "\n".join(f"| {arity} | {count} |"
                           for arity, count in data["by_arity"].items())

    sections = []
    for case, rows in data["grouped"].items():
        lines = "\n".join(f"{seq:5d}  {equation}" for seq, equation in rows)
        sections.append(f"### {heading} {case} — {len(rows)}\n\n"
                        f"```text\n{lines}\n```")
    path_sections = "\n\n".join(sections)

    return template.format(
        **data, case_rows=case_rows, arity_rows=arity_rows,
        path_sections=path_sections,
        base_delta=data["total"] - data["base"])


def main() -> int:
    data = collect()
    for language, template, heading in (("en", EN, "Case"), ("zh", ZH, "情形")):
        target = ROOT / "docs" / language / "appendix_specialized.md"
        target.write_text(render(template, data, heading), encoding="utf-8")
        print(f"wrote {target} ({target.stat().st_size} bytes, "
              f"{data['total']} paths)")

    for language in ("en", "zh"):
        index = ROOT / "docs" / language / "index.rst"
        text = index.read_text(encoding="utf-8")
        if "appendix_specialized" in text:
            continue
        index.write_text(
            text.replace("   appendix\n", "   appendix\n   appendix_specialized\n", 1),
            encoding="utf-8")
        print(f"toctree updated: {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
