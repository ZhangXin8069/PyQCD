"""契约：`optimize` 的**对外取值恰好 6 个**（`auto / none / greedy / optimal / dp / branch`）。

为什么值得单独立契约
--------------------
``paths.PATH_OPTIONS`` 目前装着 14 个名字，其中 9 个**不是**对外约定：6 个是退化别名
（与某个对外名字是**同一个函数对象**），3 个是待清除的历史名字（``branch-*``）。
名字一多，症状都很安静：``contract(optimize="eager")`` 能跑、数值也对，于是"对外只有
6 个"就变成一句没人能验证的宣言。数值测试抓不到"名字面"是否收窄，所以这里钉四条能
各自独立失败的断言：

1. 5 个注册表名字都能被 ``paths.get_path_fn`` 解析；``none`` **不在**注册表，靠
   ``_api._normalize_optimize("none") -> False`` 与 ``optimize=False`` 共用同一条
   单步路径（现状，数值等价由 ``_einsum_optimize_none_contract`` 单独钉住）；
2. 6 个名字都能被 ``contract(..., optimize=<v>)`` 成功执行，数值 ``rel <= 1e-12``；
3. **反向**：``paths.PATH_OPTIONS`` 去掉这 6 个之后，剩下的名字**恰好**是契约里登记的
   那 9 个，且每个都在三份文档里被逐条标注为 ``退化别名`` 或 ``待清除``——多一个、
   少一个、改了标注都失败，不许沉默；
4. 三份文档（``pyqcd/einsum/AGENTS.md``、``docs/einsum_user_guide_20261116.md``、
   ``skills/pyqcd-einsum/SKILL.md``）里那段约定**逐字相同**。

清理那 9 个名字时先改 ``DEGENERATE`` / ``TO_BE_REMOVED`` 与本契约、再改实现；
否则第 3 条立刻红——这正是它存在的理由。

可直接运行::

    python -B -m pyqcd.testing._einsum_optimize_surface_contract
"""
from __future__ import annotations

import inspect
import re
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from ..einsum import _api, _plan_py, contract, contract_path, paths

#: 对外取值（顺序即文档里的写法，**逐字**比对）。
PUBLIC_OPTIMIZE = ("auto", "none", "greedy", "optimal", "dp", "branch")

#: 必须在 ``paths.PATH_OPTIONS`` 里的那几个（``none`` 不在注册表，见模块 docstring）。
REGISTRY_OPTIMIZE = tuple(name for name in PUBLIC_OPTIMIZE if name != "none")

#: 退化别名：值与目标是**同一个函数对象**（``get_path_fn(name) is get_path_fn(target)``）。
DEGENERATE: dict[str, str] = {
    "auto-hq": "auto",
    "dynamic-programming": "dp",
    "eager": "greedy",
    "opportunistic": "greedy",
    "random-greedy": "greedy",
    "random-greedy-128": "greedy",
}

#: 待清除的历史名字：有独立实现（窄宽分支或全宽的 partial 包装），但不在对外约定里。
TO_BE_REMOVED: frozenset[str] = frozenset({"branch-all", "branch-1", "branch-2"})

#: 每个多余名字在文档里的**唯一**标注行（三份文档共用，逐字一致；表格竖线保留）。
EXTRA_LABEL_ROWS: dict[str, str] = {
    "auto-hq": "| `auto-hq` | 退化别名 ≡ `auto`（同一函数对象） |",
    "dynamic-programming":
        "| `dynamic-programming` | 退化别名 ≡ `dp`（同一函数对象） |",
    "eager": "| `eager` | 退化别名 ≡ `greedy`（同一函数对象） |",
    "opportunistic": "| `opportunistic` | 退化别名 ≡ `greedy`（同一函数对象） |",
    "random-greedy": "| `random-greedy` | 退化别名 ≡ `greedy`（同一函数对象） |",
    "random-greedy-128":
        "| `random-greedy-128` | 退化别名 ≡ `greedy`（同一函数对象） |",
    "branch-all": "| `branch-all` | 待清除（≡ `branch`，全宽；注册表里仍是独立名字） |",
    "branch-1": "| `branch-1` | 待清除（窄宽变体 `nbranch=1`，仅供 `auto`） |",
    "branch-2": "| `branch-2` | 待清除（窄宽变体 `nbranch=2`，仅供 `auto`） |",
}

#: 那段约定的**逐字**文本（六行；文档里允许带 ``> `` 引用前缀与不同标题层级）。
SURFACE_BLOCK: tuple[str, ...] = (
    "**qcd-einsum 对外提供的 `optimize` 取值只有 6 个："
    "`auto / none / greedy / optimal / dp / branch`。**",
    "`auto` 为默认值（选择器：按算子数查表，≤8 算子且 opt_einsum 可用时改用其 DP 结果）。",
    "`none` = 按方程原序（亦可用 `optimize=False` 表示）。"
    "`branch` = 全宽分支定界（≡ 内部 `branch_all`）。",
    "内部另有窄宽变体 `branch_1` / `branch_2`，**仅供 `auto` 使用，不属于对外取值**。",
    "`auto-hq`（≡`auto`）、"
    "`eager`/`opportunistic`/`random-greedy`/`random-greedy-128`（≡`greedy`）",
    "为**退化别名**，不属于对外约定；"
    "`branch-1`/`branch-2`/`branch-all` 属**待清除的历史名字**。",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: 必须同时带上这段约定的三份文档。
DOCS: tuple[Path, ...] = (
    _REPO_ROOT / "pyqcd" / "einsum" / "AGENTS.md",
    _REPO_ROOT / "docs" / "einsum_user_guide_20261116.md",
    _REPO_ROOT / "skills" / "pyqcd-einsum" / "SKILL.md",
)

#: 小方程：3 个操作数才能绕过 ``_api._find_path`` 里 ``num_ops <= 2`` 的短路。
EQUATION = "ij,jk,kl->il"
SHAPES = ((3, 4), (4, 5), (5, 6))
RTOL = 1e-12


# -- 张量 -----------------------------------------------------------------------
def _operands(dtype: Any = np.complex128, seed: int = 20261117) -> list[np.ndarray]:
    """按方程左侧生成 C 连续操作数（轴长见 :data:`SHAPES`）。"""
    rng = np.random.default_rng(seed)
    return [np.ascontiguousarray(rng.standard_normal(shape).astype(dtype))
            for shape in SHAPES]


def _relative_error(actual: Any, expected: Any) -> float:
    """相对最大偏差（全零答案用 1 作尺度，与库内同口径）。"""
    expected_array = np.asarray(expected)
    scale = float(np.max(np.abs(expected_array))) or 1.0
    return float(np.max(np.abs(np.asarray(actual) - expected_array))) / scale


# -- 文档归一化 -----------------------------------------------------------------
def _squash(line: str) -> str:
    """去首尾空白并把连续空白压成一个空格（只抵消排版差异，不放宽用词）。"""
    return " ".join(line.split())


def _normalized_lines(path: Path) -> list[str]:
    """文档的逐行归一化结果：去掉 markdown 引用前缀 ``> ``，其余逐字保留。"""
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        while text.startswith(">"):
            text = text[1:].lstrip()
        lines.append(_squash(text))
    return lines


def _declared_names(block_line: str) -> tuple[str, ...]:
    """从约定的第一行里取出对外名字清单（唯一含 ``/`` 的那个代码段）。"""
    spans = [span for span in re.findall(r"`([^`]+)`", block_line) if "/" in span]
    assert len(spans) == 1, ("约定的首行必须只含一个用 / 分隔的名字清单", spans)
    return tuple(part.strip() for part in spans[0].split("/"))


def _find_block(lines: list[str]) -> int:
    """约定的六行在文档里的起始下标（必须连续出现）。"""
    head = SURFACE_BLOCK[0]
    for start in range(len(lines) - len(SURFACE_BLOCK) + 1):
        if lines[start] != head:
            continue
        if list(SURFACE_BLOCK) == lines[start:start + len(SURFACE_BLOCK)]:
            return start
    raise AssertionError(
        "文档里找不到那段不可拆分的六行约定（逐字比对失败）：请把 "
        "SURFACE_BLOCK 原样贴回，不要改写措辞")


# ── 1. get_path_fn：5 个走注册表，none 走 False 归一化 ────────────────────────
def test_get_path_fn_resolves_every_public_name() -> None:
    """6 个对外名字都能落到"一条可执行的规划路径"上（``none`` 经 False 归一化）。"""
    for name in REGISTRY_OPTIMIZE:
        resolved = paths.get_path_fn(name)
        assert callable(resolved), ("注册表名字必须解析成可调用对象", name, resolved)

    assert _api._normalize_optimize("none") is False, (
        "现状：none 不进注册表，由 _api._normalize_optimize 归一成 False",)
    assert _api._normalize_optimize("  NoNe  ") is False, "大小写/空白也要归一"
    assert _api._normalize_optimize("greedy") == "greedy", "别的名字不许被这条归一吞掉"

    try:
        resolved_none = paths.get_path_fn("none")
    except KeyError as exc:
        # 现状的合法形态：注册表里没有 none，只有归一化那一条路。
        assert "none" in str(exc), ("KeyError 必须点名 none，别静默变成别的名字", str(exc))
    else:                                   # 将来把 none 注册进表也必须继续可用
        assert callable(resolved_none), ("none 若进了注册表，也必须可调用", resolved_none)


# ── 2. contract()：6 个名字都跑得通且数值一致 ────────────────────────────────
def test_contract_runs_every_public_name() -> None:
    """``contract(..., optimize=<v>)`` 对 6 个取值都成功，数值 ``rel <= 1e-12``。"""
    operands = _operands()
    expected = np.einsum(EQUATION, *operands)
    assert _relative_error(expected, expected) == 0.0, "参照本身不可信"

    for name in PUBLIC_OPTIMIZE:
        got = contract(EQUATION, *operands, optimize=name,
                       specialize=False, cache=False, profile=False)
        error = _relative_error(got, expected)
        assert error <= RTOL, ("contract(optimize=%r) 的数值超出容差" % name, error)
        assert np.asarray(got).dtype == operands[0].dtype, (
            "dtype 不许被优化器名字改掉", name, np.asarray(got).dtype)
        assert np.asarray(got).shape == (SHAPES[0][0], SHAPES[-1][-1]), (
            "输出形状不对", name, np.asarray(got).shape)

    # "none = 按方程原序"必须是单步整链收缩，且与 greedy 真的不同（否则上面的
    # 名字区分断言没有意义）。
    off_path, _ = contract_path(EQUATION, *operands, optimize="none", cache=False)
    greedy_path, _ = contract_path(EQUATION, *operands, optimize="greedy", cache=False)
    assert [tuple(step) for step in off_path] == [(0, 1, 2)], (
        "optimize='none' 应当是整条链一次 n 元收缩", off_path)
    assert [tuple(step) for step in greedy_path] != [tuple(step) for step in off_path], (
        "该方程上 greedy 与 none 的 path 相同，这条断言区分不出名字是否生效",
        greedy_path)


# ── 3. 反向：对外集合恰好这 6 个，多余的逐个有标注 ──────────────────────────
def test_public_surface_is_exactly_the_six() -> None:
    """``PATH_OPTIONS`` 去掉 6 个对外名字后，剩下的**恰好**是登记的那 9 个。"""
    extra_expected = set(DEGENERATE) | set(TO_BE_REMOVED)
    extras = set(paths.PATH_OPTIONS) - set(PUBLIC_OPTIMIZE)

    assert set(REGISTRY_OPTIMIZE) <= set(paths.PATH_OPTIONS), (
        "对外名字缺在注册表里", sorted(set(REGISTRY_OPTIMIZE) - set(paths.PATH_OPTIONS)))
    assert extras == extra_expected, (
        "PATH_OPTIONS 的名字面与约定不符",
        "多出（未登记，不许沉默）", sorted(extras - extra_expected),
        "缺登记", sorted(extra_expected - extras))

    # 退化别名必须真的是**同一个函数对象**，不是"另一份实现碰巧等价"。
    for name, target in sorted(DEGENERATE.items()):
        assert paths.get_path_fn(name) is paths.get_path_fn(target), (
            "退化别名必须与目标解析到同一个函数对象", name, target,
            paths.get_path_fn(name), paths.get_path_fn(target))

    # branch = 全宽分支定界：默认 nbranch=None，等价于内部的 branch_all。
    branch_sig = inspect.signature(_plan_py.branch)
    assert branch_sig.parameters["nbranch"].default is None, (
        "对外 branch 必须是全宽（nbranch=None）", branch_sig)
    assert _plan_py.branch_all.func is _plan_py.branch, "内部 branch_all 必须是 branch 的包装"
    assert dict(_plan_py.branch_all.keywords) == {"nbranch": None}, (
        "内部 branch_all 必须是 branch 的全宽 partial", dict(_plan_py.branch_all.keywords))
    assert paths.get_path_fn("branch") is _plan_py.branch, (
        "对外 branch 必须直接落到 _plan_py.branch")

    # 待清除的两个窄宽变体：仅供 auto 查表用，取值与对外名字不同。
    for name, width in (("branch-1", 1), ("branch-2", 2)):
        assert name in TO_BE_REMOVED, name
        resolved = paths.get_path_fn(name)
        assert callable(resolved), (name, resolved)
        assert dict(resolved.keywords) == {"nbranch": width}, (
            "窄宽变体的 nbranch 不对", name, dict(resolved.keywords))
        assert resolved is not paths.get_path_fn("branch"), (
            "窄宽变体不该与全宽 branch 是同一个对象", name)

    auto_values = list(_plan_py._AUTO_CHOICES.values())
    for narrow in (_plan_py.branch_1, _plan_py.branch_2, _plan_py.branch_all):
        assert any(value is narrow for value in auto_values), (
            "窄宽/全宽 branch 变体只应由 auto 的选择器使用", narrow)

    # 文档侧：三份文档的清单必须就是这 6 个（顺序也一致），且 9 个多余名字逐个有标注。
    for doc in DOCS:
        assert doc.exists(), ("约定的文档不见了", str(doc))
        lines = _normalized_lines(doc)
        start = _find_block(lines)
        declared = _declared_names(lines[start])
        assert declared == PUBLIC_OPTIMIZE, (
            "文档里的对外清单与代码不一致", str(doc), declared, PUBLIC_OPTIMIZE)
        known = set(lines)
        for name in sorted(extras):
            row = _squash(EXTRA_LABEL_ROWS[name])
            assert row in known, (
                "文档没有逐条标注这个多余的 optimize 名字", str(doc), name, row)


# ── 4. 三份文档逐字一致 ─────────────────────────────────────────────────────
def test_three_docs_agree_verbatim() -> None:
    """同一段约定在三份文档里**逐字**相同（只允许引用前缀与标题层级不同）。"""
    found: dict[str, tuple[str, ...]] = {}
    for doc in DOCS:
        lines = _normalized_lines(doc)
        start = _find_block(lines)
        found[doc.name] = tuple(lines[start:start + len(SURFACE_BLOCK)])

    reference = SURFACE_BLOCK
    for name, block in found.items():
        assert block == reference, (
            "这段约定在三份文档里必须逐字一致", name, block, reference)

    # 标题层级允许不同，所以只要求每份文档在约定之前出现一个标题行。
    for doc in DOCS:
        lines = _normalized_lines(doc)
        start = _find_block(lines)
        assert any(line.startswith("#") for line in lines[:start]), (
            "约定必须挂在文档标题下（标题层级不限）", str(doc))


TESTS = (
    test_get_path_fn_resolves_every_public_name,
    test_contract_runs_every_public_name,
    test_public_surface_is_exactly_the_six,
    test_three_docs_agree_verbatim,
)


def main() -> int:
    passed = skipped = failed = 0
    for test in TESTS:
        try:
            test()
        except unittest.SkipTest as exc:
            skipped += 1
            print(f"SKIP {test.__name__}: {exc}", flush=True)
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {test.__name__}: {exc}", flush=True)
        except Exception as exc:                          # noqa: BLE001 - 报告用
            failed += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}", flush=True)
        else:
            passed += 1
            print(f"PASS {test.__name__}", flush=True)
    print(f"{passed} passed, {skipped} skipped, {failed} failed", flush=True)
    return 0 if failed == 0 else 1


def test_einsum_optimize_surface_contracts() -> None:
    """套件入口：跑完 :data:`TESTS`，任何一条失败就抛出汇总。"""
    failures: list[str] = []
    passed = 0
    for test in TESTS:
        try:
            test()
        except unittest.SkipTest:
            continue
        except Exception as exc:                          # noqa: BLE001 - 报告用
            failures.append(f"{test.__name__}: {type(exc).__name__}: {exc}")
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}", flush=True)
        else:
            passed += 1
            print(f"PASS {test.__name__}", flush=True)
    if failures:
        raise AssertionError(
            f"{len(failures)} 条 optimize 对外取值契约失败：" + "; ".join(failures))
    if passed == 0:
        raise unittest.SkipTest("optimize 对外取值契约全部跳过")


if __name__ == "__main__":
    raise SystemExit(main())
