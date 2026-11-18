"""契约：``python -m pyqcd.einsum.test`` 的两个默认值口径（2026-11-18 用户实测缺陷）。

缺陷 1：``--kernel-force`` 的 argparse 缺省曾是 ``DEFAULT_KERNEL``（``generic``），
于是"用户没指定"与"用户显式写了 generic"在 ``_api`` 眼里是同一个值
（``caller_force_given = kernel_force is not None``）。配置表里 ``kernel=backend``
这类**已 tune 好**的行因此永远套不上：每次不写该选项的调用都被判成显式冲突
（``ProfileConflict``）。修法是把 argparse 缺省改成 ``None``（未指定），帮助文案
仍显示最终回落值 ``generic``，而解析出的名字只用于报告。

缺陷 2：``_profile_cache._resolve_dir_text`` 曾是 ``os.path.abspath(text)``，
``--profile-dir ~/.cache`` 被当成字面量目录 ``<cwd>/~/.cache``，与 ``_api`` 侧
（``DIR_ENV`` / ``./.cache``）不是同一张表——CLI 的目录参数**静默失效**。修法是
先 ``os.path.expanduser`` 再取绝对路径（``~`` 与 ``~user`` 都成立，环境变量同口径）。

本模块把下面五件事分别钉成能独立失败的断言：

1. ``argparse`` 解析结果：不传 ``--kernel-force`` 时 ``args.kernel_force is None``，
   而 ``DEFAULT_KERNEL`` 仍是 ``generic``、帮助文案也仍显示它（回落值不是显式值）。
2. 不传 ``--kernel-force`` + 命中表内 ``kernel=backend`` 的行 → **套用成功**：
   ``main()`` 退出码 0，标准输出含 ``profile 命中：…:行号 kernel=backend …（已套用）``；
   ``run()`` 的报告 ``kernel_force`` 仍报 ``generic``（日志口径）且不抛冲突。
3. 显式 ``--kernel-force generic`` + 命中 ``kernel=backend`` → ``ProfileConflict``
   （``run()`` 抛出，``main()`` 打印 ``error:`` 并以 2 退出）——显式冲突仍要报。
4. ``_resolve_dir_text("~/.cache") == _resolve_dir_text(os.path.expanduser("~/.cache"))``，
   且不创建字面 ``~`` 目录；``HOME`` 临时改写后，CLI 的 ``--profile-dir ~/.cache``
   钉到 ``$HOME/.cache``（不碰用户真实缓存）。
5. ``--profile-dir`` 指向真实目录时，写/读都落在该目录（模块级写口在 CLI 之后
   沿用它钉住的目录），临时目录里能看到表文件与读回的行。

口径与边界
----------
* 每条用例都在临时目录里跑，``HOME`` / ``PYQCD_EINSUM_PROFILE_DIR`` /
  ``PYQCD_EINSUM_KERNEL_FORCE`` / ``PYQCD_EINSUM_NOTICE`` 用完即还原；
  ``_profile_cache`` 与 ``_api`` 的进程内状态一并快照还原，**不碰**用户缓存。
* 配置表只有 tune 能写：这里用 :func:`_profile_cache.write_winner` 摆"已 tune"的
  现场（它就是 tune 的写口），普通调用只读那一路仍由被测的 ``run()/main()`` 走。
* 不做计时断言：``wall_ms`` 只是行里的载荷，与结论无关。

可直接运行::

    python -B -m pyqcd.testing._einsum_cli_defaults_contract
"""
from __future__ import annotations

import contextlib
import io
import os
import re
import tempfile
import unittest
from typing import Any, Iterator, Mapping
from unittest.mock import patch

from ..einsum import _api, _profile_cache, _specialize
from ..einsum import test as cli

#: 方程与形状：小到毫秒级，又足以覆盖 numpy 后端的画像键。
EQUATION = "ij,jk->ik"
SHAPES = [(4, 4), (4, 4)]
OUT_SHAPE = (4, 4)

#: 摆进配置表的"已 tune"结论：``backend`` 比专用内核快。
ROW_PAYLOAD = {"kernel": "backend", "optimize": "greedy", "wall_ms": 1.25}

#: 本模块会临时清掉的环境变量（目录/总开关/强制内核/提示开关）。
_ENV_NAMES = (
    _profile_cache.DIR_ENV,
    _profile_cache.PROFILE_ENV,
    cli.KERNEL_FORCE_ENV,
    "PYQCD_EINSUM_NOTICE",
)

#: ``_profile_cache`` 的进程内状态（锁与常量不碰）。
_PROFILE_STATE = ("_TABLE", "_TUNING", "_CONFIGURED_DIR", "_LAST_ERROR",
                  "_PENDING_WARNINGS", "_WARNED_CELLS")

#: ``_api`` 的进程内状态：命中提示去重、最近一次画像/内核记录。
_API_STATE = ("_PROFILE_NOTICED", "_PROFILE_MISSING_NOTICED", "_LAST_PROFILE",
              "_LAST_KERNEL")

#: ``_specialize`` 的语言/详解覆盖（``run()`` 会把它们推进库内）。
_SPECIALIZE_STATE = ("_DETAIL_OVERRIDE", "_LANGUAGE_OVERRIDE")


def _copy(value: Any) -> Any:
    """集合/列表/元组按内容拷贝，其余原样（画像缓存的行列表要连行一起拷）。"""
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, set):
        return set(value)
    if isinstance(value, list):
        return list(value)
    if value is not None and isinstance(value, tuple) and len(value) == 4 \
            and isinstance(value[3], list):                  # ``_TABLE`` 的解析缓存
        return (value[0], value[1], value[2], [_copy(row) for row in value[3]])
    return value


@contextlib.contextmanager
def _isolated() -> Iterator[str]:
    """临时目录 + 干净的环境/进程内状态；退出时逐字还原。"""
    saved_env = {name: os.environ.pop(name, None) for name in _ENV_NAMES}
    saved_state = [(module, name, _copy(getattr(module, name)))
                   for module, names in ((_profile_cache, _PROFILE_STATE),
                                         (_api, _API_STATE),
                                         (_specialize, _SPECIALIZE_STATE))
                   for name in names]
    with tempfile.TemporaryDirectory() as tmp:
        try:
            yield tmp
        finally:
            for module, name, value in saved_state:
                setattr(module, name, value)
            for name, value in saved_env.items():
                if value is not None:
                    os.environ[name] = value


@contextlib.contextmanager
def _home(directory: str) -> Iterator[None]:
    """把 ``HOME`` 临时指向 ``directory``（``~`` 展开的唯一来源，POSIX 口径）。"""
    with patch.dict(os.environ, {"HOME": directory}):
        yield


def _argv(directory: str, *extra: str) -> list[str]:
    """一条完整可跑的 CLI 命令（``--profile-dir`` 就是被测的那一格）。"""
    return ["--equation", EQUATION, "--shape", "[4,4],[4,4]",
            "--compare", "none", "--rounds", "1", "--warmup", "1",
            "--profile", "true", "--profile-dir", directory,
            "--language", "zh-cn", *extra]


def _run_cli(directory: str, *, kernel_force: str | None = None,
             language: str = "zh") -> dict[str, Any]:
    """在临时画像目录上跑一次 ``run()``（不经过 argparse，用于取键与断言报告）。"""
    _api._LAST_PROFILE = None
    return cli.run(equation=EQUATION, shapes=SHAPES, out_shape=OUT_SHAPE,
                   compare="none", rounds=1, warmup=1,
                   profile=True, profile_dir=directory,
                   kernel_force=kernel_force, language=language)


def _profile_key(directory: str) -> dict[str, Any]:
    """CLI 自己算出来的画像键（表里那一行必须与它逐字相同才会命中）。"""
    _run_cli(directory)
    info = _api.last_profile() or {}
    key = info.get("key")
    assert isinstance(key, dict) and key, f"CLI 没给出画像键：{info!r}"
    return dict(key)


def _seed_backend_row(directory: str) -> dict[str, Any]:
    """在该目录写一行 ``kernel=backend``（走 tune 的唯一写口），返回它的键。"""
    key = _profile_key(directory)
    _profile_cache.configure(directory)
    _profile_cache.write_winner(key, dict(ROW_PAYLOAD))
    _api._PROFILE_NOTICED.clear()        # 让本次运行的"已套用"提示一定打出来
    return key


def _main(directory: str, *extra: str) -> tuple[int, str, str]:
    """跑一次真实 CLI；返回 ``(退出码, stdout, stderr)``。"""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(_argv(directory, *extra))
    return code, out.getvalue(), err.getvalue()


# ── 1. argparse 缺省 = 未指定 ────────────────────────────────────────────────
def test_parser_leaves_kernel_force_unset():
    """不传 ``--kernel-force`` 时是 ``None``（未指定），回落值仍由 ``DEFAULT_KERNEL`` 表达。"""
    assert cli.DEFAULT_KERNEL == cli.GENERIC_LABEL, cli.DEFAULT_KERNEL
    parser = cli.build_parser(language="en")
    args = parser.parse_args(["--equation", EQUATION])
    # ``None`` 是唯一能让 `_api` 区分"没给"与"显式 generic"的取值。
    assert args.kernel_force is None, args.kernel_force
    help_text = parser.format_help()
    block = re.search(r"--kernel-force[^\n]*(?:\n(?!\s{4}\S).*)*", help_text)
    assert block is not None, help_text
    # 帮助文案仍要显示最终回落值（"未指定时按 generic 执行"），否则用户不知道该期待什么。
    assert f"Default {cli.DEFAULT_KERNEL}" in block.group(0), block.group(0)
    assert "not given" in block.group(0) or "not the same" in block.group(0), \
        block.group(0)


# ── 2. 不传该选项 → 命中行套用成功 ──────────────────────────────────────────
def test_omitted_kernel_force_applies_tuned_backend_row():
    """缺省调用 + 表内 ``kernel=backend`` → ``（已套用）``，不得抛 ``ProfileConflict``。"""
    with _isolated() as tmp:
        key = _seed_backend_row(tmp)
        assert _profile_cache.read_winner(key)["kernel"] == "backend"
        code, stdout, stderr = _main(tmp)
        assert code == 0, (code, stdout, stderr)
        # 命中提示钉到具体记录（``<csv 路径>:<数据行号>``），并写明已套用。
        assert re.search(r"profile 命中：.*\.csv:\d+\s+kernel=backend", stdout), stdout
        assert "已套用" in stdout, stdout
        assert "冲突" not in stdout + stderr, (stdout, stderr)
        # 日志/报告口径仍是回落值 generic：API 侧看到的是"未指定"。
        report = _run_cli(tmp)
        assert report.get("ok"), report
        fields = report["kernel_force"]
        assert isinstance(fields, Mapping), fields
        assert fields["requested"] == cli.DEFAULT_KERNEL, fields
        # 表里那行真的生效了（`_api.last_kernel()` 记的是本次观测）。
        assert fields["ran"] == "backend", fields


# ── 3. 显式 generic → 冲突仍要报 ────────────────────────────────────────────
def test_explicit_generic_still_conflicts_with_backend_row():
    """``--kernel-force generic`` 是**显式**要求：与表里 ``backend`` 冲突必须报错。"""
    with _isolated() as tmp:
        _seed_backend_row(tmp)
        try:
            _run_cli(tmp, kernel_force="generic")
        except _profile_cache.ProfileConflict as exc:
            assert exc.field == "kernel_force", exc.field
            assert exc.recorded == "backend", exc.recorded
            assert exc.requested == "generic", exc.requested
            assert "冲突" in str(exc), str(exc)
        else:
            raise AssertionError("显式 kernel_force=generic 撞上表里 backend 本应抛冲突")
        # CLI 层：正文原样打出，退出码 2（脚本据此区分"改参数/改表"与"结果不对"）。
        code, stdout, stderr = _main(tmp, "--kernel-force", "generic")
        assert code == 2, (code, stdout, stderr)
        assert "error:" in stderr and "冲突" in stderr, stderr
        assert "已套用" not in stdout, stdout


# ── 4. `~` 展开 ─────────────────────────────────────────────────────────────
def test_tilde_profile_dir_is_expanded():
    """``~/.cache`` 与展开后的绝对路径是同一处；CLI 也钉到 ``$HOME/.cache``。"""
    with _isolated() as tmp:
        expected = os.path.join(tmp, ".cache")
        with _home(tmp):
            assert _profile_cache._resolve_dir_text("~/.cache") == expected
            assert _profile_cache._resolve_dir_text(
                os.path.expanduser("~/.cache")) == expected
            # 环境变量那一路同口径（CLI 不传参时走它）。
            with patch.dict(os.environ, {_profile_cache.DIR_ENV: "~/.cache"}):
                assert _profile_cache._resolve_dir_text(None) == expected
            # CLI 侧：`--profile-dir ~/.cache` 必须钉到 $HOME/.cache（只读，不建目录）。
            _run_cli("~/.cache")
            pinned = _profile_cache.profile_path()
        assert pinned == os.path.join(expected, _profile_cache.PROFILE_FILENAME), pinned
        assert not os.path.exists(os.path.join(os.getcwd(), "~")), "不得创建字面 ~ 目录"
        assert not os.path.exists(expected), "读路径不建目录"


# ── 5. --profile-dir 真实目录：写/读都在这里 ────────────────────────────────
def test_profile_dir_from_cli_is_used_for_write_and_read():
    """CLI 钉住的目录就是后续写/读的目录（不是 cwd 下的默认 ``./.cache``）。"""
    with _isolated() as tmp:
        target = os.path.join(tmp, "cache")
        os.makedirs(target)
        key = _profile_key(target)
        # CLI 把目录钉进了 `_profile_cache`：模块级写口沿用它。
        assert _profile_cache.profile_path() == os.path.join(
            target, _profile_cache.PROFILE_FILENAME), _profile_cache.profile_path()
        _profile_cache.write_winner(key, dict(ROW_PAYLOAD))
        path = _profile_cache.profile_path()
        assert os.path.isfile(path), path
        assert os.path.dirname(path) == target, path
        row = _profile_cache.read_winner(key)
        assert row is not None and row["kernel"] == "backend", row
        # 只有该目录里有表；仓库根/cwd 不该出现产物。
        assert sorted(os.listdir(target)) == [_profile_cache.PROFILE_FILENAME], \
            sorted(os.listdir(target))


TESTS = (
    test_parser_leaves_kernel_force_unset,
    test_omitted_kernel_force_applies_tuned_backend_row,
    test_explicit_generic_still_conflicts_with_backend_row,
    test_tilde_profile_dir_is_expanded,
    test_profile_dir_from_cli_is_used_for_write_and_read,
)


def test_einsum_cli_defaults_contracts():
    """注册入口：逐条运行 ``TESTS``；全跳过时也 SkipTest（不许静默通过）。"""
    passed = skipped = 0
    failures = []
    for test in TESTS:
        try:
            test()
        except unittest.SkipTest as exc:
            skipped += 1
            print(f"SKIP {test.__name__}: {exc}", flush=True)
        except Exception as exc:                          # noqa: BLE001 - 报告用
            failures.append(f"{test.__name__}: {type(exc).__name__}: {exc}")
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}", flush=True)
        else:
            passed += 1
            print(f"PASS {test.__name__}", flush=True)
    if failures:
        raise AssertionError(
            f"{len(failures)} 条 CLI 默认值契约失败：" + "; ".join(failures))
    if passed == 0:
        raise unittest.SkipTest(
            f"CLI defaults contracts did not run ({skipped} skipped)")


def main() -> int:
    """直接运行本模块：返回失败条数（0 = 全通过）。"""
    passed = skipped = failed = 0
    for test in TESTS:
        try:
            test()
        except unittest.SkipTest as exc:
            skipped += 1
            print(f"SKIP {test.__name__}: {exc}")
        except Exception:                                 # noqa: BLE001
            failed += 1
            print(f"FAIL {test.__name__}")
            import traceback

            traceback.print_exc()
        else:
            passed += 1
            print(f"PASS {test.__name__}")
    print(f"\n{passed} passed, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
