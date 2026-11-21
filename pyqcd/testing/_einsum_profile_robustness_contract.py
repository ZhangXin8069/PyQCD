"""契约：配置表缓存的**并发/健壮性**口径（``pyqcd.einsum._profile_cache``）。

``_einsum_profile_cache_contract`` 钉住的是"写了什么、读了什么"（冻结名字、单写者、
同键一行、``wall_ms``、``tune_session``、目录优先级）。本模块钉住的是另一半——
**坏环境下还能不能活着回来**。这六条都不会让数值结果变一位，所以普通数值测试永远
碰不到：

1. **``configure`` 只解析路径（不建目录）**：钉住 A 之后 ``ensure_configured()`` 必须
   仍是 A（哪怕 ``PYQCD_EINSUM_PROFILE_DIR`` 指向别处）；只有 ``configure(None)`` 才按
   "env > 默认"重算，清空环境后回到 ``./.cache``——默认那档只在临时 CWD 下验证，
   **不在仓库根留下 ``.cache``**。
2. **坏行只丢那一行**：往主表尾部追加"非法 UTF-8（含 NUL）"与"列数不足"两行，
   读表不抛、好行照常 :func:`read_winner` 命中、``summary()["rows"]`` 仍只数好行，
   且 :func:`enabled` 仍为真、``last_error`` 仍为空（一行坏数据不许把整张表判死）。
3. **超长 ``note`` 写盘前截断**：5000 字符的备注落盘恰好 ``_CLIP_LIMITS["note"]``
   个字符，读回来也是截断后的值（读写同一口径）。
4. **同键只留一行**：同一个键连写 10 次，磁盘上永远只有 1 行（最后一次胜出），
   别的键不受牵动——"每键多条 + 选优"已废除（§2.2/§7）。
5. **``fork`` 安全**：父进程某线程持着 ``_LOCK`` 时 ``fork``，子进程的 ``summary()``
   必须**返回**（父进程 1 s 内等到子进程退出即通过）。这是一条**活性**判定，不是
   性能结论：``threading.Lock`` 不随 ``fork`` 重置，缺 ``os.register_at_fork`` 钩子时
   子进程会永久阻塞，超时即判失败。
6. **关闭零副作用**：``PYQCD_EINSUM_PROFILE=0`` 时开关为假、写静默、读返回 ``None``、
   ``summary()["enabled"]`` 为假，且临时 CWD 里连目录都不出现。

不做任何性能断言：第 5 条只问"是不是 1 s 内**返回**"，不问快慢；其余用例不测时间。
每个用例都用 ``tempfile`` + 临时 CWD，退出时还原环境变量与模块状态，
所以既不在仓库根留 ``.cache``，也不让上一条用例的目录/解析缓存漏进下一条。

可直接运行::

    python -B -m pyqcd.testing._einsum_profile_robustness_contract
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import signal
import tempfile
import threading
import time
import traceback
import unittest
import warnings
from typing import Any

from ..einsum import _profile_cache

#: 第 5 条的**活性**闸门（不是性能指标）：子进程拿到 CPU 后跑完一次 ``summary()``
#: 只需要一次目录探测 + 一次顺序读，1 s 是"绝不该卡死"的量级。
_LIVENESS_S = 1.0

#: 持锁线程就位/释放、以及父进程等持锁线程退出的兜底上限（防止契约自己卡住）。
_HANDOFF_S = 30.0

#: 模块里需要按用例隔离的进程内状态（``_TABLE`` 是解析缓存，按内容拷贝）。
_STATE_NAMES = ("_CONFIGURED_DIR", "_LAST_ERROR", "_HOST", "_TABLE", "_TUNING")


# ── 状态隔离 ────────────────────────────────────────────────────────────────

def _snapshot(module: Any) -> dict[str, Any]:
    """把模块的进程内状态抄一份（``_TABLE`` 的行列表也拷一层）。"""
    saved: dict[str, Any] = {}
    for name in _STATE_NAMES:
        value = getattr(module, name)
        if name == "_TABLE" and value is not None:
            value = (value[0], value[1], value[2], [dict(row) for row in value[3]])
        saved[name] = value
    return saved


def _restore(module: Any, saved: dict[str, Any]) -> None:
    """把 :func:`_snapshot` 抄下来的状态放回去。"""
    for name, value in saved.items():
        setattr(module, name, value)


def _reset(module: Any) -> None:
    """把模块清到"没配置、没解析缓存、不处于 tune 窗口"的初始态。"""
    module._TABLE = None
    module._TUNING = 0
    module._CONFIGURED_DIR = ""
    module._LAST_ERROR = None


@contextlib.contextmanager
def _isolated(disabled: bool = False):
    """临时 CWD + 干净的环境变量与进程内状态；退出时全部还原。

    配置目录可以是相对路径（默认 ``./.cache`` 就是），所以"不在仓库根留垃圾"靠的
    是把 CWD 换进临时树；两个环境变量与 :data:`_STATE_NAMES` 一律还原，
    避免污染同进程里其它契约（它们也在读同一份模块级状态）。
    """
    module = _profile_cache
    saved_cwd = os.getcwd()
    saved_env = {name: os.environ.get(name)
                 for name in (module.PROFILE_ENV, module.DIR_ENV)}
    saved = _snapshot(module)
    with tempfile.TemporaryDirectory(prefix="pyqcd-profile-robust-") as root:
        work = os.path.join(root, "work")
        os.makedirs(work)
        try:
            os.chdir(work)
            os.environ.pop(module.DIR_ENV, None)
            if disabled:
                os.environ[module.PROFILE_ENV] = "0"
            else:
                os.environ.pop(module.PROFILE_ENV, None)
            _reset(module)
            yield work
        finally:
            os.chdir(saved_cwd)
            for name, value in saved_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            _restore(module, saved)


# ── 小工具 ──────────────────────────────────────────────────────────────────

def _key(**overrides: Any) -> dict[str, Any]:
    """主表键的稀疏写法（``shape`` 用列表写法，与实现同口径归一化）。"""
    key = {
        "backend": "numpy",
        "dtype": "complex128",
        "equation": "ij,jk->ik",
        "shape": [(2, 3), (3, 4)],
        "device": "cpu",
    }
    key.update(overrides)
    return key


def _forget() -> None:
    """丢掉解析缓存：本模块会直接往文件尾部追加坏字节。"""
    _profile_cache._TABLE = None


def _read_rows(path: str) -> list[list[str]]:
    """整份 CSV（含表头行）；用真实 ``open``，本模块不打补丁。"""
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def _fork_summary_while_lock_held(module: Any, deadline_s: float):
    """父进程有线程持着 ``module._LOCK`` 时 ``fork``，等子进程跑完 ``summary()``。

    返回 ``(退出状态, 子进程写回的字节)``；子进程**没有**在 ``deadline_s`` 内退出
    （即卡在继承来的锁上）时杀掉它并抛 ``AssertionError``——这就是"缺
    ``os.register_at_fork`` 钩子"的失败面。
    """
    read_fd, write_fd = os.pipe()
    held = threading.Event()
    release = threading.Event()

    def _hold() -> None:
        # 只做一件事：拿到锁、报信、等放行。父进程任何时刻都不直接调本模块。
        with module._LOCK:
            held.set()
            release.wait(_HANDOFF_S)

    holder = threading.Thread(target=_hold, name="pyqcd-profile-lock-holder",
                              daemon=True)
    holder.start()
    payload = b""
    try:
        assert held.wait(_HANDOFF_S), "持锁线程未能在限期内就位，契约前提不成立"
        # 3.12 起多线程进程 fork 会发 DeprecationWarning：那正是本用例要钉的危险，
        # 我们用一个可判定的活性闸门来验证它——因此把这条告警就地闭掉，别刷屏。
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            pid = os.fork()
        if pid == 0:                                   # 子进程：只跑 summary() 后 _exit
            status = 0
            try:
                state = module.summary()
                os.write(write_fd, json.dumps({
                    "enabled": state["enabled"],
                    "rows": state["rows"],
                    "keys": state["keys"],
                    "dir": state["dir"],
                }).encode("utf-8"))
            except BaseException:                      # noqa: BLE001 - 带回父进程判定
                status = 97
                try:
                    os.write(write_fd,
                             b"ERROR:" + traceback.format_exc().encode()[:2000])
                except BaseException:                  # noqa: BLE001 - 尽力而为
                    pass
            finally:
                try:
                    os.close(write_fd)
                except OSError:
                    pass
                os._exit(status)                       # 绝不回到测试体里
        os.close(write_fd)                             # 父进程：只留读端
        deadline = time.monotonic() + deadline_s
        status = None
        while True:
            done, raw_status = os.waitpid(pid, os.WNOHANG)
            if done == pid:
                status = raw_status
                break
            if time.monotonic() >= deadline:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                raise AssertionError(
                    f"子进程在 {deadline_s:g} s 内没有返回：父进程持锁时 fork 出的"
                    " 子进程必须在 os.register_at_fork(after_in_child=...) 钩子里"
                    " 重建 _LOCK，否则 summary() 永久阻塞")
            time.sleep(0.005)
        payload = os.read(read_fd, 65536)               # 子进程已退出 → 必有 EOF
    finally:
        try:
            os.close(read_fd)
        except OSError:
            pass
        release.set()
        holder.join(_HANDOFF_S)
    return status, payload


# ── 1. configure 只解析路径、ensure_configured 不改写 ────────────────────────

def test_ensure_configured_keeps_pinned_dir():
    """``configure(A)`` 后 ``ensure_configured()`` 仍是 A；只有 ``configure`` 会重算。"""
    with _isolated() as work:
        module = _profile_cache
        pinned = os.path.join(work, "pinned")
        env_dir = os.path.join(work, "env_cache")
        # ``DEFAULT_DIR`` 自 2026-11-19 起是**绝对路径**（用户家目录下的
        # ``~/.cache/pyqcd.einsum``），不再是相对 CWD 的 ``./.cache``。
        # ``os.path.join(work, 绝对路径)`` 会把 work 直接丢掉，所以这里必须按
        # "已经是绝对路径就原样用"来取。
        default = (module.DEFAULT_DIR if os.path.isabs(module.DEFAULT_DIR)
                   else os.path.join(os.path.abspath(work), module.DEFAULT_DIR))
        default_existed = os.path.exists(default)

        assert module.configure(pinned) == os.path.abspath(pinned)
        assert not os.path.exists(pinned), (
            "configure 只解析路径：读路径不建目录，建目录只发生在 write_winner")

        # 1) 已钉住的目录不被改写——哪怕环境变量指向别处。
        os.environ[module.DIR_ENV] = env_dir
        assert module.ensure_configured() == os.path.abspath(pinned), (
            "ensure_configured 不得用环境变量改写已钉住的目录",
            module.ensure_configured())
        state = module.summary()
        assert state["dir"] == os.path.abspath(pinned), state
        assert state["file"] == os.path.join(os.path.abspath(pinned),
                                            module.PROFILE_FILENAME), state
        assert not os.path.exists(env_dir), (
            "ensure_configured 只管补齐，不得按环境变量另建目录")

        # 2) 清空环境后 configure(None) 才按"env > 默认"重算 → 默认目录（用户家目录）。
        os.environ.pop(module.DIR_ENV, None)
        assert module.configure(None) == default, (
            "configure(None) 必须回到默认目录", module.summary())
        state = module.summary()
        assert state["dir"] == default, state
        assert state["file"] == os.path.join(default, module.PROFILE_FILENAME), state
        # 默认目录**不再**落在 CWD 下（那是这次改动的目的），但绝不能是仓库根：
        # 仓库根的 `.cache` 会被 git 忽略、又会被下一次测试读到，正是偶发失败的来源。
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(_profile_cache.__file__))))
        assert not state["dir"].startswith(os.path.join(repo_root, ".cache")), (
            "默认目录不许落在仓库根的 .cache", state, repo_root)
        assert state["dir"] == module.DEFAULT_DIR, (
            "默认目录必须就是 DEFAULT_DIR", state)
        # 默认目录可能在跑测试之前就存在（用户在别的项目里真的 tune 过）。
        # 能断言的是"configure 没有把它新建出来"：与进入时的状态比较。
        assert os.path.exists(default) == default_existed, (
            "只解析路径，不得顺手建出默认目录", default_existed)

        # 3) 再调 ensure：沿用当前目录，不重算。
        assert module.ensure_configured() == default, module.summary()


# ── 2. 坏行只丢那一行 ───────────────────────────────────────────────────────

def test_corrupt_rows_are_skipped_not_fatal():
    """追加非法 UTF-8（含 NUL）与列数不足两行：不抛、好行照常命中、表不被判死。"""
    with _isolated() as work:
        module = _profile_cache
        directory = os.path.join(work, "cache")
        module.configure(directory)
        module.write_winner(_key(equation="ij,jk->ik"),
                            {"kernel": "good_fast", "wall_ms": 1.0})
        module.write_winner(_key(equation="ab,bc->ac", shape=[(4, 5), (5, 6)]),
                            {"kernel": "good_slow", "wall_ms": 9.0})
        path = os.path.join(directory, module.PROFILE_FILENAME)
        assert len(_read_rows(path)) == 3, ("表头 + 2 行", _read_rows(path))

        # 两行坏数据：一行非法 UTF-8（首字节是 NUL，还带 0xFF/0xFE），一行列数不足。
        with open(path, "ab") as handle:
            handle.write(b"\x00\xff\xfe garbage row after NUL\n")
            handle.write(b"too,few,fields\n")
        _forget()

        state = module.summary()
        assert state["rows"] == 2, ("好行不能被坏行带走", state)
        assert state["keys"] == 2, state
        # 关键：坏行只降级"那一行"，不许把整张表判死。
        assert state["enabled"] is True, ("坏行不得关掉缓存", state)
        assert module.enabled() is True, state
        assert state["last_error"] is None, ("读侧的坏行不是目录不可用", state)

        picked = module.read_winner(_key(equation="ij,jk->ik"))
        assert picked is not None, "坏行之后仍必须命中好行"
        assert picked["kernel"] == "good_fast", picked
        assert picked["wall_ms"] == "1.000", picked
        assert module.read_winner(
            _key(equation="ab,bc->ac", shape=[(4, 5), (5, 6)]))["kernel"] == "good_slow"


def test_nul_in_quoted_multiline_record_does_not_swallow_later_rows():
    """NUL filtering must preserve CSV logical-record boundaries."""
    with _isolated() as work:
        module = _profile_cache
        directory = os.path.join(work, "cache")
        module.configure(directory)
        module.write_winner(_key(equation="ij,jk->ik"),
                            {"kernel": "first", "wall_ms": 1.0})
        module.write_winner(_key(equation="ab,bc->ac"),
                            {"kernel": "second", "wall_ms": 2.0})
        path = os.path.join(directory, module.PROFILE_FILENAME)
        header, first = _read_rows(path)[:2]

        bad = dict(zip(header, first))
        bad["equation"] = "bad->x"
        bad["note"] = "before\x00after\ncontinued"
        after = dict(zip(header, first))
        after["equation"] = "xy,yz->xz"
        after["kernel"] = "after"
        after["note"] = "valid row after corrupt multiline record"

        with open(path, "ab") as handle:
            for row in (bad, after):
                stream = io.StringIO()
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow([row[name] for name in header])
                handle.write(stream.getvalue().encode("utf-8"))
        _forget()

        state = module.summary()
        picked = module.read_winner(_key(equation="xy,yz->xz"))

    assert state["rows"] == 3, state
    assert state["keys"] == 3, state
    assert state["last_error"] is None, state
    assert picked is not None and picked["kernel"] == "after", picked


# ── 3. 超长字段写盘前截断 ───────────────────────────────────────────────────

def test_overlong_text_fields_are_clipped():
    """5000 字符 note → 落盘恰好 ``_CLIP_LIMITS["note"]`` 个字符（读回同一口径）。"""
    with _isolated() as work:
        module = _profile_cache
        directory = os.path.join(work, "cache")
        module.configure(directory)
        limit = module._CLIP_LIMITS["note"]
        assert limit == 200, ("备注的写盘上限就是 200", module._CLIP_LIMITS)

        long_note = "N" * 5000
        module.write_winner(_key(), equivalence={
            "target_equation": "cadb,ac,bd->", "operand_order": [1, 0]},
            note=long_note)
        _forget()

        table = _read_rows(os.path.join(directory, module.PROFILE_FILENAME))
        assert len(table) == 2, ("表头 + 1 行", table)
        note_cell = table[1][module.PROFILE_COLUMNS.index("note")]
        assert len(note_cell) == limit, ("落盘 note 必须是 200 字符", len(note_cell))
        assert note_cell == "N" * limit, note_cell[-16:]
        # 内存与落盘同一口径（读回来也是截断后的值）。
        picked = module.read_winner(_key())
        assert picked is not None and len(picked["note"]) == limit, picked


# ── 4. 同键只留一行 ─────────────────────────────────────────────────────────

def test_repeated_writes_keep_a_single_row_per_key():
    """同键连写 10 次 → 磁盘上永远 1 行（最后一次胜出），别的键不受牵动。"""
    with _isolated() as work:
        module = _profile_cache
        directory = os.path.join(work, "cache")
        module.configure(directory)
        key = _key()
        other = _key(equation="ab,bc->ac")
        module.write_winner(other, {"kernel": "backend", "wall_ms": 5.0})

        observed = [float(10 - index) for index in range(10)]      # 10.0 … 1.0
        for ms in observed:
            module.write_winner(key, {"kernel": f"k{int(ms):02d}",
                                      "optimize": "auto", "wall_ms": ms})
            _forget()
            state = module.summary()
            assert state["rows"] == 2, ("同键永远只留 1 行", state)
            assert state["keys"] == 2, state

        table = _read_rows(os.path.join(directory, module.PROFILE_FILENAME))
        assert len(table) == 3, ("表头 + 2 行（每个键 1 行）", len(table))
        kept = [row for row in table[1:]
                if row[module.PROFILE_COLUMNS.index("equation")] == "ij,jk->ik"]
        assert len(kept) == 1, kept
        assert kept[0][module.PROFILE_COLUMNS.index("kernel")] == "k01", (
            "留下的是最后一次写入（不是最快的那次）", kept)
        assert kept[0][module.PROFILE_COLUMNS.index("wall_ms")] == "1.000", kept
        assert module.read_winner(key)["kernel"] == "k01", module.read_winner(key)
        assert module.read_winner(other)["kernel"] == "backend", (
            "别的键不许被牵动", module.read_winner(other))


# ── 5. fork 安全（活性判定） ────────────────────────────────────────────────

def test_fork_child_can_summarize_while_parent_holds_lock():
    """父进程持锁时 fork：子进程 ``summary()`` 必须返回（1 s 内退出）。"""
    if not hasattr(os, "fork") or not hasattr(os, "register_at_fork"):
        raise unittest.SkipTest(
            "本平台没有 os.fork / os.register_at_fork，fork 后锁重建不可验")

    with _isolated() as work:
        module = _profile_cache
        directory = os.path.join(work, "cache")
        module.configure(directory)
        # 先落一行：子进程重建锁后要从文件真的读回来，才能证明它拿到的不是
        # "父进程那一刻的内存快照"。
        module.write_winner(_key(), {"kernel": "forked_row", "wall_ms": 2.0})
        _forget()

        status, payload = _fork_summary_while_lock_held(module, _LIVENESS_S)

        assert status is not None, "父进程没有拿到子进程的退出状态"
        assert os.WIFEXITED(status), ("子进程被信号杀死", status)
        assert os.WEXITSTATUS(status) == 0, (
            "子进程里的 summary() 抛异常了", payload.decode("utf-8", "replace"))
        decoded = json.loads(payload.decode("utf-8"))
        assert decoded["enabled"] is True, decoded
        assert decoded["keys"] == 1, ("子进程必须按文件重建解析结果", decoded)
        assert decoded["rows"] == 1, ("子进程必须真的读回那一行", decoded)
        assert decoded["dir"] == os.path.abspath(directory), decoded

        # 父进程这边：持锁线程已退出，锁完好可用（钩子只重建子进程的锁）。
        assert module.summary()["dir"] == os.path.abspath(directory)
        assert module.read_winner(_key())["kernel"] == "forked_row", (
            "父进程的读路径也必须仍然可用")


# ── 6. 关闭零副作用 ─────────────────────────────────────────────────────────

def test_disabled_profile_leaves_no_trace():
    """``PYQCD_EINSUM_PROFILE=0``：不建目录、写静默、读 ``None``、``summary`` 为假。"""
    with _isolated(disabled=True) as work:
        module = _profile_cache
        default_existed = os.path.exists(module.DEFAULT_DIR)
        assert module.enabled() is False
        assert module.write_winner(_key(), {"kernel": "backend"}) is None
        assert module.write_winner(_key(), equivalence={
            "target_equation": "cadb,ac,bd->", "operand_order": [1, 0]}) is None
        assert module.read_winner(_key()) is None, "关闭时读当无命中"

        state = module.summary()
        assert state["enabled"] is False, state
        assert state["rows"] == 0 and state["keys"] == 0, state
        assert state["last_error"] is None, state

        # 零副作用：整个临时 CWD 里连目录都不许出现，默认目录（家目录下）也不许被建。
        assert os.listdir(work) == [], ("关闭时不得产生任何文件", os.listdir(work))
        # 默认目录现在是**用户家目录**下的绝对路径，跑测试前它可能本来就在（用户真的
        # tune 过）。这里能断言的是"关闭这一次调用没有把它新建出来"：比较进入前后。
        assert os.path.exists(module.DEFAULT_DIR) == default_existed, (
            "关闭时不得新建默认目录", module.DEFAULT_DIR, default_existed)


TESTS = (
    test_ensure_configured_keeps_pinned_dir,
    test_corrupt_rows_are_skipped_not_fatal,
    test_nul_in_quoted_multiline_record_does_not_swallow_later_rows,
    test_overlong_text_fields_are_clipped,
    test_repeated_writes_keep_a_single_row_per_key,
    test_fork_child_can_summarize_while_parent_holds_lock,
    test_disabled_profile_leaves_no_trace,
)


def test_einsum_profile_robustness_contracts():
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
            f"{len(failures)} 条并发/健壮性契约失败：" + "; ".join(failures))
    if passed == 0:
        raise unittest.SkipTest(
            f"profile robustness contracts did not run ({skipped} skipped)")


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


if __name__ == "__main__":
    raise SystemExit(main())
