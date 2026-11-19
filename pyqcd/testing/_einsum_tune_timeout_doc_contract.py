"""契约：``--tune-timeout`` 的**真实能力**必须如实写在 help / 报告 / docstring 里。

为什么值得单独立契约
--------------------
``--tune-timeout`` 用 ``signal.setitimer`` + ``SIGALRM`` 实现，而信号处理器只在**回到
解释器**时才派发：

* 纯 Python 长循环（``optimal`` 的穷举规划）**能**被打断（实测 6.5 s 的那次当场中止）；
* 原生 C / CUDA 代码**打不断**——收缩内核不返回解释器，处理器永远不执行。实测
  ``--tune`` 在 ``STAT=Rl+``、``%CPU=100``、``GPU 0%`` 状态下挂了 3 分钟，
  ``--tune-timeout`` 完全没生效。
* ``fork`` 子进程隔离试过并**已回退**：CUDA 上下文不跨 fork 复用，子进程必然
  ``cudaErrorInitializationError``（实测 22 组全 error）。

这类限制**只影响文档，不影响任何数值**，所以普通数值测试永远抓不到它；而旧文案
"可能打断原生内核 —— 建议在新进程里跑 tune"既把能力说反了，又等于没给建议（用户
原话："等于没说"）。四条断言分别钉住：

1. ``--help`` 的 ``--tune-timeout`` 段落写明"**打不断原生** C/CUDA 代码"，且**不含**
   "可能打断原生内核 / 建议在新进程里跑 tune"这类含糊句；
2. ``tune note :`` 行同一口径（不是只在 help 里说一次）；
3. 源码里 ``尽力而为 | best-effort`` 至少出现两行（help + note 各一处）；
4. 三处文案（help / note / docstring）**措辞一致**：同一组关键短语都在，含糊句都不在。

断言只读源码与 ``translate``/``print_tune`` 的字符串，不跑收缩、不需要 GPU。

可直接运行::

    python -B -m pyqcd.testing._einsum_tune_timeout_doc_contract
"""
from __future__ import annotations

import contextlib
import io
import os
import pathlib
import re
import unittest
from collections.abc import Iterator

from ..einsum import test as cli

#: "这是尽力而为的上限"的两个标记（中文 / 英文），至少各出现一次才算说明白了。
BEST_EFFORT_MARKERS = ("尽力而为", "best-effort")

#: 能力边界：中文写"打不断原生"，英文写 "cannot interrupt native"（大小写不敏感）。
NATIVE_LIMIT_ZH = "打不断原生"
NATIVE_LIMIT_EN = "cannot interrupt native"

#: 可操作建议（缺了它，读者只剩"别用"这一条）。
ADVICE_ZH = "--tune pass"
ADVICE_EN = "--tune pass"

#: 必须消失的含糊说法：能力说反 + 把问题推给用户。
VAGUE_PHRASES = (
    "可能打断原生内核",
    "建议在新进程里跑 tune",
    "may interrupt a native kernel",
    "prefer running tune in a fresh process",
)

#: 三处文案必须共同含有的中文短语（"措辞一致"的可判定形式）。
CONSISTENT_ZH = ("尽力而为", "打不断原生", "取足够大", "快速通过", "--tune pass")

#: 英文侧同一组（大小写不敏感）。
CONSISTENT_EN = ("best-effort", "cannot interrupt native", "large enough",
                 "--tune pass first")

#: ``--help`` 里这一段的边界：从 ``--tune-timeout`` 自己的选项行到下一个选项行。
_OPTION = "--tune-timeout"
_NEXT_OPTION = "--tune-budget"


def _flatten(text: str) -> str:
    """把折行拉平：argparse 按列宽插的换行 + 缩进一律折成单个空格。"""
    return " ".join(str(text).split())


def _nospace(text: str) -> str:
    """再进一步去掉全部空白：中文短语被折行切开后仍能逐字比对。"""
    return re.sub(r"\s+", "", str(text))


@contextlib.contextmanager
def _width(columns: int = 120) -> Iterator[None]:
    """固定 ``COLUMNS``，让 ``--help`` 的折行位置与本机终端宽度无关。"""
    saved = os.environ.get("COLUMNS")
    os.environ["COLUMNS"] = str(columns)
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("COLUMNS", None)
        else:
            os.environ["COLUMNS"] = saved


def _help_text(language: str) -> str:
    with _width():
        return cli.build_parser(language=language).format_help()


def _help_section(language: str = "zh") -> str:
    """``--help`` 中 ``--tune-timeout`` 的**完整段落**（含它自己的多行说明）。"""
    lines = _help_text(language).splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.startswith("  " + _OPTION) and not line.startswith("    "):
            start = index
            break
    assert start is not None, f"--help({language}) 里找不到 {_OPTION} 选项"
    body = [lines[start]]
    for line in lines[start + 1:]:
        # 下一个选项行 / 位置参数行以两个空格开头；说明的折行缩进至少 4 格。
        if line.strip() and not line.startswith("    "):
            break
        body.append(line)
    section = "\n".join(body)
    assert _NEXT_OPTION not in section, (
        f"--help({language}) 的 {_OPTION} 段落把 {_NEXT_OPTION} 吞了：{section!r}")
    return section


def _docstring_section() -> str:
    """模块 docstring 里讲 tune 超时的那一条（``* **超时与预算**`` 起到下一条止）。"""
    doc = cli.__doc__ or ""
    marker = "超时与预算"
    start = doc.find(marker)
    assert start >= 0, "模块 docstring 里找不到「超时与预算」那一段"
    rest = doc[start:]
    following = rest.find("* **", len(marker))
    return rest if following < 0 else rest[:following]


def _tune_note_line(language: str) -> str:
    """真跑一次 ``print_tune``，取出 ``tune note :`` 那一行（唯一来源，不抄文案）。"""
    tune = {"rows": [], "best": None, "timer": "sigalrm", "write": {}, "axes": {},
            "branch": {}, "profile": {}, "profile_enabled": None, "combos": 0,
            "warmup": 1, "rounds": 2, "compare": "none", "reference": "-"}
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        cli.print_tune(tune, language)
    lines = [line for line in buffer.getvalue().splitlines()
             if line.startswith("tune note : ")]
    assert len(lines) == 1, f"tune note 行应恰好一条，实得 {lines!r}"
    return lines[0]


def _assert_no_vague(text: str, where: str) -> None:
    for phrase in VAGUE_PHRASES:
        assert phrase.lower() not in _flatten(text).lower(), (
            f"{where} 仍有含糊说法 {phrase!r}：{_flatten(text)!r}")


def test_help_section_states_the_limitation_and_drops_vague_wording() -> None:
    """契约 1/4：``--help`` 段落写明"打不断原生"，且不含含糊句。"""
    zh = _help_section("zh")
    en = _help_section("en")
    assert NATIVE_LIMIT_ZH in _nospace(zh), f"中文 --help 未写明限制：{_flatten(zh)!r}"
    assert NATIVE_LIMIT_EN in _flatten(en).lower(), (
        f"英文 --help 未写明限制：{_flatten(en)!r}")
    assert "尽力而为" in _nospace(zh), f"中文 --help 未标明 best-effort：{_flatten(zh)!r}"
    assert "best-effort" in _flatten(en).lower(), (
        f"英文 --help 未标明 best-effort：{_flatten(en)!r}")
    assert ADVICE_ZH in _flatten(zh), f"中文 --help 缺可操作建议：{_flatten(zh)!r}"
    assert ADVICE_EN in _flatten(en), f"英文 --help 缺可操作建议：{_flatten(en)!r}"
    _assert_no_vague(zh, "--help(zh)")
    _assert_no_vague(en, "--help(en)")
    # 含糊说法在**整份源码**里也不许残留（旧文案曾同时写在 note 与 docstring）。
    source = pathlib.Path(cli.__file__).read_text(encoding="utf-8")
    _assert_no_vague(source, "pyqcd/einsum/test.py")


def test_tune_note_line_states_the_same_limitation() -> None:
    """契约 2/4：``tune note :`` 行同一口径（不是只在 help 里说一次）。"""
    zh = _tune_note_line("zh")
    en = _tune_note_line("en")
    assert NATIVE_LIMIT_ZH in _nospace(zh), f"中文 tune note 未写明限制：{zh!r}"
    assert NATIVE_LIMIT_EN in _flatten(en).lower(), f"英文 tune note 未写明限制：{en!r}"
    assert "尽力而为" in _nospace(zh), f"中文 tune note 未标明 best-effort：{zh!r}"
    assert "best-effort" in _flatten(en).lower(), (
        f"英文 tune note 未标明 best-effort：{en!r}")
    assert ADVICE_ZH in _flatten(zh), f"中文 tune note 缺可操作建议：{zh!r}"
    assert ADVICE_EN in _flatten(en), f"英文 tune note 缺可操作建议：{en!r}"
    _assert_no_vague(zh, "tune note(zh)")
    _assert_no_vague(en, "tune note(en)")


def test_source_mentions_best_effort_at_least_twice() -> None:
    """契约 3/4：``尽力而为 | best-effort`` 至少两行（help + note 各一处）。"""
    path = pathlib.Path(cli.__file__)
    pattern = re.compile(r"尽力而为|best-effort")
    hits = [number for number, line in
            enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if pattern.search(line)]
    assert len(hits) >= 2, f"只有 {len(hits)} 行提到 best-effort：{hits}"
    # 两处**必须分别**落在 help 文案与 tune note 文案里，不能都挤在 docstring。
    assert "best-effort" in _flatten(_help_section("en")).lower(), \
        "help 文案里没有 best-effort 标记"
    assert "best-effort" in _flatten(_tune_note_line("en")).lower(), \
        "tune note 文案里没有 best-effort 标记"


def test_three_places_agree_on_the_wording() -> None:
    """契约 4/4：help / note / docstring 三处措辞一致（同一组关键短语都在）。"""
    places = {
        "--help(zh)": _help_section("zh"),
        "tune note(zh)": _tune_note_line("zh"),
        "docstring": _docstring_section(),
    }
    for where, text in places.items():
        flat, packed = _flatten(text), _nospace(text)
        for phrase in CONSISTENT_ZH:
            assert phrase in flat or phrase in packed, (
                f"{where} 缺关键短语 {phrase!r}：{flat!r}")
        _assert_no_vague(text, where)
    en_places = {
        "--help(en)": _help_section("en"),
        "tune note(en)": _tune_note_line("en"),
    }
    for where, text in en_places.items():
        flat = _flatten(text).lower()
        for phrase in CONSISTENT_EN:
            assert phrase in flat, f"{where} 缺关键短语 {phrase!r}：{flat!r}"


TESTS = (
    test_help_section_states_the_limitation_and_drops_vague_wording,
    test_tune_note_line_states_the_same_limitation,
    test_source_mentions_best_effort_at_least_twice,
    test_three_places_agree_on_the_wording,
)


def main() -> int:
    passed = skipped = failed = 0
    for test in TESTS:
        try:
            test()
        except unittest.SkipTest as exc:
            skipped += 1
            print(f"SKIP {test.__name__}: {exc}", flush=True)
        except Exception as exc:                          # noqa: BLE001 - 报告用
            failed += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}", flush=True)
        else:
            passed += 1
            print(f"PASS {test.__name__}", flush=True)
    print(f"{passed} passed, {skipped} skipped, {failed} failed", flush=True)
    return 0 if failed == 0 else 1


def test_einsum_tune_timeout_doc_contracts() -> None:
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
            f"{len(failures)} 条 tune-timeout 文案契约失败：" + "; ".join(failures))
    if passed == 0:
        raise unittest.SkipTest("tune-timeout 文案契约全部跳过")


if __name__ == "__main__":
    raise SystemExit(main())
