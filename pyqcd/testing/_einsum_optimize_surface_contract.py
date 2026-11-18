"""契约：``optimize`` 的**对外取值面**——7 个叶子，三份文档逐字同一段约定。

为什么值得单独立契约
-------------------
名字面是数值测试永远抓不到的缺口：``contract(optimize="eager")`` 能跑、数值也对，
于是"对外只有哪几个名字"就变成一句没人能验证的宣言；反过来 ``branch`` 这种
"看起来合理"的拼写会直接 ``KeyError``，用户只能靠试。所以这里同时钉住**两侧**：

1. 文档侧——:data:`SURFACE_BLOCK` 那段约定在三份文档（``pyqcd/einsum/AGENTS.md``、
   ``docs/einsum_user_guide_20261116.md``、``skills/pyqcd-einsum/SKILL.md``）里
   **逐字相同**，首行声明的名字清单就是那 7 个（顺序也一致）；
2. 代码侧——``paths.PATH_OPTIONS`` 去掉运行时注册的名字与兼容别名之后**恰好**是
   这 7 个；多一个、少一个都失败，不许沉默。

7 个叶子（:data:`PUBLIC_OPTIMIZE`）：``greedy`` / ``optimal`` / ``dp`` /
``branch_1`` / ``branch_2`` / ``branch_all`` / ``none``。

``auto`` **不在此列**：它是 ``contract(optimize=...)`` 的签名缺省值、也是
``paths._IMPLICIT_DEFAULT_NAMES`` 里的内部归一结果，``paths.get_path_fn("auto")``
照旧抛 ``KeyError``；CLI 的 ``--optimize`` 也因此不接受它（缺省值是 ``None`` =
"未指定"，内部再归一回 ``auto``）。

与 :mod:`._einsum_optimize_leaves_contract` 的分工：那个契约钉**运行时**名字面
（解析结果 ``is`` 同一个对象、``PATH_OPTIONS_MAIN``、已删拼写必须抛错、``none``
的内核语义）；本契约钉**对外文档**那一面（逐字一致 + 文档清单与注册表一致），
外加"7 个名字都解析得出来、都跑得通"这两条端到端保险。

可直接运行::

    python -B -m pyqcd.testing._einsum_optimize_surface_contract
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from ..einsum import _api, contract, contract_path, paths

#: 对外取值（顺序即文档里的写法，**逐字**比对）。
PUBLIC_OPTIMIZE: tuple[str, ...] = (
    "greedy", "optimal", "dp", "branch_1", "branch_2", "branch_all", "none",
)

#: 兼容别名 → 主名：必须解析到**同一个**函数对象。
#: 2026-11-18 用户裁定"拼写统一下划线"后，连字符 ``branch-*`` 与
#: ``dynamic-programming`` **全部删除**——保留两条拼写会让 README、配置表与
#: ``--help`` 各写一套。此表因此为空，保留常量只为让"不允许别名"可被断言。
ALIASES: dict[str, str] = {}

#: 每个别名在三份文档里的**唯一**标注行（逐字一致；表格竖线保留）。
EXTRA_LABEL_ROWS: dict[str, str] = {}

#: 那段约定的**逐字**文本（六行；文档里允许带 ``> `` 引用前缀与不同标题层级）。
SURFACE_BLOCK: tuple[str, ...] = (
    "**qcd-einsum 对外提供的 `optimize` 取值只有 7 个："
    "`greedy / optimal / dp / branch_1 / branch_2 / branch_all / none`。**",
    "不写 `optimize` 时走内置选择器 `auto`（按算子数查表，≤8 算子且 opt_einsum 可用时"
    "改用其 DP 结果）；`auto` 是**缺省值而不是可写取值**，CLI 与配置表都不接受它。",
    "`none` = 按方程原序（亦可用 `optimize=False` 表示）。"
    "`branch_1` / `branch_2` / `branch_all` = 分支定界的窄/全宽变体，三者都是对外取值。",
    "`branch-1` / `branch-2` / `branch-all`（下划线主名的连字符拼写）与 "
    "`dynamic-programming`（≡ `dp`）是**兼容别名**，与主名解析到同一个函数对象，"
    "不算独立名字。",
    "`auto-hq` / `eager` / `opportunistic` / `random-greedy` / `random-greedy-128` 已"
    "**从名字面删除**，`paths.get_path_fn` 一律抛 `KeyError`。",
    "契约：`pyqcd/testing/_einsum_optimize_surface_contract.py`（本节逐字一致）与 "
    "`_einsum_optimize_leaves_contract.py`（运行时名字面）。",
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


# ── 1. get_path_fn：7 个名字都解析得出来 ──────────────────────────────────────
def test_get_path_fn_resolves_every_public_name() -> None:
    """7 个对外名字都能落到"一条可执行的规划路径"上（含 ``none`` 的哨兵）。"""
    for name in PUBLIC_OPTIMIZE:
        resolved = paths.get_path_fn(name)
        assert callable(resolved), ("optimize 名字必须解析成可调用对象", name, resolved)

    # ``none`` 是"不规划"的哨兵，不是某个真的优化器 —— 否则上面那条会因为
    # "反正能调用"而失去意义。
    assert paths.get_path_fn("none") is paths._NO_PATH, (
        "optimize='none' 必须解析成 _NO_PATH 哨兵", paths.get_path_fn("none"))
    for name in PUBLIC_OPTIMIZE:
        if name != "none":
            assert paths.get_path_fn(name) is not paths._NO_PATH, (
                "none 不该与真的优化器共用同一个对象", name)


# ── 2. contract()：7 个名字都跑得通且数值一致 ────────────────────────────────
def test_contract_runs_every_public_name() -> None:
    """``contract(..., optimize=<v>)`` 对 7 个取值都成功，数值 ``rel <= 1e-12``。"""
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

    # ``branch_all`` 与 ``branch_1`` 必须是**不同**的计划，否则"窄/全宽是三个名字"
    # 只是文档上的说法（``branch_all`` = ``nbranch=None``，``branch_1`` = 1）。
    branch_paths = {name: [tuple(step) for step in
                           contract_path(EQUATION, *operands, optimize=name,
                                         cache=False)[0]]
                    for name in ("branch_1", "branch_2", "branch_all")}
    assert len({tuple(value) for value in branch_paths.values()}) >= 2, (
        "三个 branch 变体解析成了同一条计划，名字区分没有意义", branch_paths)


# ── 3. 反向：对外集合恰好这 7 个（别名不算），文档清单逐份一致 ───────────────
def test_public_surface_is_exactly_the_seven() -> None:
    """``PATH_OPTIONS`` 去掉运行时注册与别名后**恰好**是这 7 个。"""
    registered = set(paths._REGISTERED)      # 调用方可以注册自己的名字
    declared = set(paths.PATH_OPTIONS) - registered
    aliases = set(ALIASES)
    leaves = set(PUBLIC_OPTIMIZE)

    assert leaves <= declared, (
        "对外名字缺在注册表里", sorted(leaves - declared))
    assert declared - aliases == leaves, (
        "PATH_OPTIONS 的名字面与约定不符",
        "多出（未登记，不许沉默）", sorted(declared - aliases - leaves),
        "缺少", sorted(leaves - (declared - aliases)))
    assert not (set(ALIASES) & leaves), ("别名不许混进叶子清单", sorted(set(ALIASES) & leaves))

    assert set(paths.PATH_OPTIONS_MAIN) == leaves, (
        "主清单必须恰好是这 7 个叶子",
        sorted(set(paths.PATH_OPTIONS_MAIN) - leaves),
        sorted(leaves - set(paths.PATH_OPTIONS_MAIN)))

    # ``auto`` 不在名字面，但规划器必须仍认得它（否则每次缺省调用都 KeyError）。
    assert "auto" not in paths.PATH_OPTIONS, "auto 是缺省值，不是可写的名字"
    for spelling in ("auto", "AUTO"):
        assert paths.is_builtin_path_name(spelling), (
            "auto 必须仍被规划器承接，否则缺省 contract() 会 KeyError", spelling)
    assert _api._normalize_optimize("auto") == "auto", (
        "显式 'auto' 只被规划器承接，不该被归一成某个叶子",
        _api._normalize_optimize("auto"))

    # 缺省档（``None`` / ``True``）与显式 ``auto`` 走同一个选择器：三者都必须能收缩。
    operands = _operands()
    expected = np.einsum(EQUATION, *operands)
    for default in (None, True, "auto"):
        got = contract(EQUATION, *operands, optimize=default, specialize=False,
                       cache=False, profile=False)
        error = _relative_error(got, expected)
        assert error <= RTOL, ("缺省 optimize 必须仍能收缩", default, error)

    # 别名必须是**同一个函数对象**，不是"另一份实现碰巧等价"。
    for alias, target in sorted(ALIASES.items()):
        assert paths.get_path_fn(alias) is paths.get_path_fn(target), (
            "兼容别名必须与主名解析到同一个函数对象", alias, target,
            paths.get_path_fn(alias), paths.get_path_fn(target))

    # 文档侧：三份文档的清单必须就是这 7 个（顺序也一致），别名逐个有标注。
    for doc in DOCS:
        assert doc.exists(), ("约定的文档不见了", str(doc))
        lines = _normalized_lines(doc)
        start = _find_block(lines)
        names = _declared_names(lines[start])
        assert names == PUBLIC_OPTIMIZE, (
            "文档里的对外清单与代码不一致", str(doc), names, PUBLIC_OPTIMIZE)
        known = set(lines)
        for name in sorted(ALIASES):
            row = _squash(EXTRA_LABEL_ROWS[name])
            assert row in known, (
                "文档没有逐条标注这个兼容别名", str(doc), name, row)


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
    test_public_surface_is_exactly_the_seven,
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
