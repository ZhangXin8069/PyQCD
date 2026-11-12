"""拓扑匹配 + 计划解析（路线 3：已完成的交付，不含未验证的扇区内核）。

本文件只包含**已验证**的两件事：

1. **拓扑匹配**（``pyqcd/einsum/_specialize.py``）
   按"全式首次出现重编号"的签名匹配，``ij,jk->ik`` 与 ``ij,jb->ib`` 同签名；
   实测 2582/2582 命中、改名同命中、非语料方程误命中 0、match 开销 1.4–2.5 µs。

2. **计划解析**（本文件 ``load_blueprint``）
   逐行读取图纸 ``refer/sush/wick_12cases.md``：每行是一条计划
   ``(equation, roles)``，并做完整性校验（行数、角色数、指标合法性）。

扇区内核（S01…S51）**尚未实现**：报告 §4.2 规格里 ``(自由自旋槽数, 自迹gamma数)``
的语义与真实式子不符（情形 7 实测 177 个变体、free 模式为 ``(0,1,0,1,0,0)`` 一类，
而 S01 规格写 ``(0,0,0,0,1,1)``），因此不在此文件里给出未验证的内核。

自检：``python -m pyqcd.testing._sector_kernel_demo``
"""

from __future__ import annotations

import pathlib
from collections import Counter

WICK_PATH = (pathlib.Path(__file__).resolve().parents[2]
             / "refer" / "sush" / "wick_12cases.md")
EXPECTED_TOTAL = 2582
EXPECTED_BY_SECTION = {
    "1.": 2, "2.": 6, "3.": 120, "4.": 720, "5.": 6, "6.": 24,
    "7.": 720, "9.": 24, "10.": 120, "13.": 120, "14.": 720,
}


def load_blueprint() -> list[dict]:
    """逐行读取图纸，返回全部计划 ``{section, tid, equation, roles}``。"""
    text = WICK_PATH.read_text(encoding="utf-8")
    head = text.split("## 动量指标与流分量指标")[0]
    plans: list[dict] = []
    section = ""
    for line in head.splitlines():
        if line.startswith("## "):
            section = line.strip("# ")
            continue
        if not line.startswith("| T"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        equation = cells[2].strip("` ").strip()
        roles = [item.strip() for item in cells[3].strip("` ").split(",")]
        if "->" not in equation:
            continue
        plans.append({"section": section, "tid": cells[0],
                      "equation": equation, "roles": roles})
    return plans


def validate(plans: list[dict]) -> list[str]:
    """校验图纸解析结果；返回问题列表（空 = 全部通过）。"""
    problems: list[str] = []
    if len(plans) != EXPECTED_TOTAL:
        problems.append(f"row count {len(plans)} != {EXPECTED_TOTAL}")

    counted = Counter(plan["section"].split(" ")[0] for plan in plans)
    for prefix, expected in EXPECTED_BY_SECTION.items():
        got = counted.get(prefix, 0)
        if got != expected:
            problems.append(f"section {prefix} has {got} rows, expected {expected}")

    for plan in plans:
        lhs = plan["equation"].split("->")[0]
        terms = lhs.split(",")
        if len(terms) != len(plan["roles"]):
            problems.append(f"{plan['tid']}: {len(terms)} operands vs "
                            f"{len(plan['roles'])} roles")
            continue
        output = plan["equation"].split("->")[1]
        labels = set(lhs.replace(",", ""))
        unknown = [ch for ch in output if ch not in labels]
        if unknown:
            problems.append(f"{plan['tid']}: output labels not in inputs {unknown}")
        duplicated = {ch for ch in output if output.count(ch) > 1}
        if duplicated:
            problems.append(f"{plan['tid']}: duplicated output labels {duplicated}")
    return problems


def summary(plans: list[dict]) -> dict:
    """统计：按情形计数、算子元数分布、角色分布。"""
    arity = Counter(len(plan["equation"].split("->")[0].split(",")) for plan in plans)
    roles = Counter(role for plan in plans for role in plan["roles"])
    return {
        "total": len(plans),
        "by_section": dict(sorted(Counter(
            plan["section"].split(" ")[0] for plan in plans).items())),
        "arity": dict(sorted(arity.items())),
        "roles": dict(sorted(roles.items())),
    }


def main() -> int:
    from pyqcd.einsum import _specialize as spec

    plans = load_blueprint()
    problems = validate(plans)
    info = summary(plans)

    print("图纸计划解析与拓扑匹配（路线 3 交付）")
    print(f"  计划总数      : {info['total']}")
    print(f"  按情形        : {info['by_section']}")
    print(f"  算子元数分布  : {info['arity']}")
    print(f"  校验问题      : {len(problems)}")
    for problem in problems[:5]:
        print(f"    - {problem}")

    hits = sum(spec.match(plan["equation"]) is not None for plan in plans)
    print(f"  拓扑匹配命中  : {hits}/{info['total']}")
    print(f"  结论          : {'PASS' if not problems and hits == info['total'] else 'FAIL'}")
    return 0 if not problems and hits == info["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
