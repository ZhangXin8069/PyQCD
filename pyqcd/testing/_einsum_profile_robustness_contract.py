"""契约：配置画像缓存的**并发/健壮性**口径（``pyqcd.einsum._profile_cache``）。

``_einsum_profile_cache_contract`` 钉住的是"写了什么、读了什么"（CSV 往返、选优、
稀疏写入、只写不读、关闭零 IO、目录优先级）。本模块钉住的是另一半——**坏环境下
还能不能活着回来**，这七条都不会让数值结果变一位，所以普通数值测试永远碰不到：

1. **``ensure_configured`` 只补齐、不改写**：:func:`configure` 钉住 A 之后，
   ``ensure_configured()`` 必须仍是 A（哪怕 ``PYQCD_EINSUM_PROFILE_DIR`` 指向别处）；
   只有 ``configure(None)`` 才按"env > 默认"重算，清空环境后回到 ``./.cache``
   ——默认那档只在临时 CWD 下验证，**不在仓库根留下 ``.cache``**。
2. **坏行只丢那一行**：往 ``profile.csv`` 追加"非法 UTF-8（含 NUL）"与"列数不足"两行，
   ``load(force=True)`` 不抛、``skipped_rows`` 两条都计、好行照常 ``best()`` 命中，
   且 ``enabled()`` 仍为真（一行坏数据不许把整张表判死）。
3. **超长字段写盘前截断**：5000 字符的 ``note`` 落盘 ≤ 200（本契约钉死 ``== 200``），
   ``equivalence_equation`` ≤ 512，截断计数（``truncated_fields``）随之增加；
   内存里也是截断后的值（读写同一口径）。
4. **每键最快 K 条**：同键写 10 个不同 ``(kernel, optimize)``，内存与落盘都只剩
   ``compute_ms`` 最快的 :data:`_MAX_ROWS_PER_KEY` 条，被裁掉的行**不许**借待写队列
   漏进磁盘（``flush()`` 的行数与索引一致）。
5. **``fork`` 安全**：父进程某线程持着 ``_LOCK`` 时 ``fork``，子进程的 ``summary()``
   必须**返回**（父进程 1 s 内等到子进程退出即通过）。这是一条**活性**判定，不是
   性能结论：``threading.Lock`` 不随 ``fork`` 重置，缺 ``os.register_at_fork`` 钩子时
   子进程会永久阻塞，超时即判失败。
6. **关闭零副作用**：``PYQCD_EINSUM_PROFILE=0`` 时开关为假、``record`` 返回 ``False``、
   ``summary()["enabled"]`` 为假，且临时 CWD 里连目录都不出现。

不做任何性能断言：第 5 条只问"是不是 1 s 内**返回**"，不问快慢；其余用例不测时间。
每个用例都用 ``tempfile`` + 临时 CWD，退出时还原环境变量与模块状态（含累计计数器），
所以既不在仓库根留 ``.cache``，也不让上一条用例的目录/计数漏进下一条。

可直接运行::

    python -B -m pyqcd.testing._einsum_profile_robustness_contract
"""
from __future__ import annotations

import contextlib
import csv
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

#: 模块里需要按用例隔离的进程内状态（含三个累计计数器——它们只增不减，
#: 不还原就会让下一条用例的"增量"断言失去意义）。
_STATE_NAMES = (
    "_configured_dir",
    "_ready",
    "_dir",
    "_dir_ok",
    "_last_error",
    "_notice_done",
    "_loaded",
    "_loaded_rows",
    "_skipped_rows",
    "_truncated_fields",
    "_dropped_rows",
    "_disk_rows",
    "_rewrite_rows",
)


# ── 状态隔离 ────────────────────────────────────────────────────────────────

def _snapshot(module: Any) -> dict[str, Any]:
    """把模块的进程内状态抄一份（索引/队列/去重表按值拷贝）。"""
    return {
        "attrs": {name: getattr(module, name) for name in _STATE_NAMES},
        "index": {key: {config: dict(row) for config, row in slot.items()}
                  for key, slot in module._index.items()},
        "pending": [dict(row) for row in module._pending],
        "compare": dict(module._compare_recent),
    }


def _restore(module: Any, saved: dict[str, Any]) -> None:
    """把 :func:`_snapshot` 抄下来的状态放回去（含累计计数器）。"""
    for name, value in saved["attrs"].items():
        setattr(module, name, value)
    module._index.clear()
    module._index.update(saved["index"])
    module._pending[:] = saved["pending"]
    module._compare_recent.clear()
    module._compare_recent.update(saved["compare"])


def _reset(module: Any) -> None:
    """把模块清到"没配置、没加载、没建目录、计数器归零"的初始态。"""
    module._index.clear()
    module._pending.clear()
    module._compare_recent.clear()
    module._configured_dir = ""
    module._ready = False
    module._dir = ""
    module._dir_ok = False
    module._last_error = None
    module._notice_done = False
    module._loaded = False
    module._loaded_rows = 0
    module._skipped_rows = 0
    module._truncated_fields = 0
    module._dropped_rows = 0
    module._disk_rows = -1
    module._rewrite_rows = -1


@contextlib.contextmanager
def _isolated(disabled: bool = False):
    """临时 CWD + 干净的环境变量与进程内状态；退出时全部还原。

    缓存目录可以是相对路径（默认 ``./.cache`` 就是），所以"不在仓库根留垃圾"靠的
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
    """主表键的稀疏写法（``host`` 缺省由模块补本机名，读写两侧一致）。"""
    key = {
        "backend": "numpy",
        "dtype": "complex128",
        "equation": "ij,jk->ik",
        "shape": [(2, 3), (3, 4)],
        "device": "cpu",
        "user_device": "cpu",
    }
    key.update(overrides)
    return key


def _row(**overrides: Any) -> dict[str, Any]:
    """主表行的稀疏写法：键 + 配置 + 观测值。"""
    row = {
        **_key(),
        "kernel": "kupl_dedicated",
        "optimize": "auto",
        "compute_ms": 1.0,
        "calls": 1,
    }
    row.update(overrides)
    return row


def _column(name: str) -> int:
    """``PROFILE_COLUMNS`` 里的列序（避免在测试里写死数字）。"""
    return _profile_cache.PROFILE_COLUMNS.index(name)


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


# ── 1. ensure_configured 只补齐、不改写 ─────────────────────────────────────

def test_ensure_configured_keeps_pinned_dir():
    """``configure(A)`` 后 ``ensure_configured()`` 仍是 A；只有 ``configure`` 会重算。"""
    with _isolated() as work:
        module = _profile_cache
        pinned = os.path.join(work, "pinned")
        env_dir = os.path.join(work, "env_cache")
        default = os.path.join(work, module.DEFAULT_DIR)

        assert module.configure(pinned) == os.path.abspath(pinned)
        assert os.path.isdir(pinned), "configure 必须就位（建出目录）"

        # 1) 已钉住的目录不被改写——哪怕环境变量指向别处。
        os.environ[module.DIR_ENV] = env_dir
        assert module.ensure_configured() == os.path.abspath(pinned), (
            "ensure_configured 不得用环境变量改写已钉住的目录",
            module.ensure_configured())
        state = module.summary()
        assert state["dir"] == os.path.abspath(pinned), state
        assert state["file"] == os.path.join(os.path.abspath(pinned),
                                            module.PROFILE_NAME), state
        assert not os.path.exists(env_dir), (
            "ensure_configured 只管补齐，不得按环境变量另建目录")

        # 2) 清空环境后 configure(None) 才按"env > 默认"重算 → 临时 CWD 下的 ./.cache。
        os.environ.pop(module.DIR_ENV, None)
        assert module.configure(None) == default, (
            "configure(None) 必须回到默认 ./.cache", module.summary())
        state = module.summary()
        assert state["dir"] == default, state
        assert state["file"] == os.path.join(default, module.PROFILE_NAME), state
        assert state["dir"].startswith(os.path.abspath(work)), (
            "默认目录必须落在临时 CWD 下，仓库根不许出现 .cache", state)
        assert os.path.isdir(default), "configure(None) 必须真就位"

        # 3) 再调 ensure：沿用当前目录，不重算。
        assert module.ensure_configured() == default, module.summary()


# ── 2. 坏行只丢那一行 ───────────────────────────────────────────────────────

def test_corrupt_rows_are_skipped_not_fatal():
    """追加非法 UTF-8（含 NUL）与列数不足两行：不抛、计数、好行照常命中。"""
    with _isolated() as work:
        module = _profile_cache
        directory = os.path.join(work, "cache")
        module.configure(directory)
        module.load()
        assert module.record(_row(equation="ij,jk->ik", kernel="good_fast",
                                  compute_ms=1.0)) is True
        assert module.record(_row(equation="ab,bc->ac", shape=[(4, 5), (5, 6)],
                                  kernel="good_slow", compute_ms=9.0)) is True
        assert module.flush() == 2

        path = os.path.join(directory, module.PROFILE_NAME)
        assert len(_read_rows(path)) == 3, ("表头 + 2 行", _read_rows(path))

        # 两行坏数据：一行非法 UTF-8（首字节是 NUL，还带 0xFF/0xFE），一行列数不足。
        with open(path, "ab") as handle:
            handle.write(b"\x00\xff\xfe garbage row after NUL\n")
            handle.write(b"too,few,fields\n")

        assert module.load(force=True) == 2, (
            "坏行之外的两行必须照常读入", module.summary())
        state = module.summary()
        assert state["skipped_rows"] >= 2, (
            "两条坏行都必须计入 skipped_rows（编码 1 + 列数 1）", state)
        assert state["rows"] == 2, ("好行不能被坏行带走", state)
        # 关键：坏行只降级"那一行"，不许把整张表判死。
        assert state["enabled"] is True, ("坏行不得关掉缓存", state)
        assert module.enabled() is True, state
        assert state["last_error"] is None, ("读侧的坏行不是目录不可用", state)

        picked = module.best(_key(equation="ij,jk->ik"))
        assert picked is not None, "坏行之后仍必须 best() 到好行"
        assert picked["kernel"] == "good_fast", picked
        assert float(picked["compute_ms"]) == 1.0, picked


# ── 3. 超长字段写盘前截断 ───────────────────────────────────────────────────

def test_overlong_text_fields_are_clipped():
    """5000 字符 note → 落盘 200；equivalence_equation → 512；计数随之增加。"""
    with _isolated() as work:
        module = _profile_cache
        directory = os.path.join(work, "cache")
        module.configure(directory)
        module.load()
        assert module._CLIP_LIMITS["note"] == 200, module._CLIP_LIMITS
        assert module._CLIP_LIMITS["equivalence_equation"] == 512, (
            module._CLIP_LIMITS)

        before = module.summary()["truncated_fields"]
        long_note = "N" * 5000
        assert module.record(_row(note=long_note)) is True
        state = module.summary()
        assert state["truncated_fields"] >= before + 1, (
            "截断必须计数", before, state)

        assert module.flush() == 1
        table = _read_rows(os.path.join(directory, module.PROFILE_NAME))
        assert len(table) == 2, ("表头 + 1 行", table)
        note_cell = table[1][_column("note")]
        assert len(note_cell) == 200, ("落盘 note 必须是 200 字符", len(note_cell))
        assert note_cell == "N" * 200, note_cell[-16:]
        # 内存与落盘同一口径（读写两侧共用 _clip_row）。
        picked = module.best(_key())
        assert picked is not None and len(picked["note"]) == 200, picked

        # compare 表的超长方程用同一个截断器（512）——旁证表也不许被撑坏。
        mid = module.summary()["truncated_fields"]
        assert module.record_compare({
            **_key(equation="ab,bc->ac"),
            "kernel": "kupl_dedicated",
            "optimize": "auto",
            "compare": "ij,jk->ik",
            "equivalence": {"kind": "permute", "equation": "E" * 5000},
        }) is True
        after = module.summary()["truncated_fields"]
        assert after >= mid + 1, ("compare 的截断同样要计数", mid, after)

        compare_table = _read_rows(os.path.join(directory, module.COMPARE_NAME))
        assert len(compare_table) == 2, ("表头 + 1 行", compare_table)
        equation_cell = compare_table[1][
            module.COMPARE_COLUMNS.index("equivalence_equation")]
        assert len(equation_cell) == 512, (
            "落盘 equivalence_equation 必须是 512 字符", len(equation_cell))


# ── 4. 每键最快 K 条 ────────────────────────────────────────────────────────

def test_per_key_cap_keeps_fastest_k_in_memory_and_on_disk():
    """同键 10 个配置 → 内存与落盘都只剩最快的 4 条，落选行不许漏进磁盘。"""
    with _isolated() as work:
        module = _profile_cache
        cap = module._MAX_ROWS_PER_KEY
        assert cap == 4, ("每键上限就是 4", cap)
        directory = os.path.join(work, "cache")
        module.configure(directory)
        module.load()

        # 由慢到快写 10 个不同 (kernel, optimize)：每个都比当前最优快 >5%，
        # 因此每个都**够格排队**——落选行只能靠"每键最快 K 条"剔除。
        observed = [float(10 - index) for index in range(10)]      # 10.0 … 1.0
        for ms in observed:
            assert module.record(
                _row(kernel=f"k{int(ms):02d}", optimize="auto",
                     compute_ms=ms)) is True, ms

        state = module.summary()
        assert state["rows"] == cap, ("内存只留最快 K 条", state)
        assert state["keys"] == 1, state
        assert state["max_rows_per_key"] == cap, state
        assert state["dropped_rows"] >= len(observed) - cap, (
            "被裁掉的条数必须可见", state)
        assert module.best(_key())["compute_ms"] == "1.000", module.best(_key())

        # 落盘与内存同一口径：待写队列里的落选行必须被剔除。
        assert module.flush() == cap, (
            "flush 的行数必须与索引一致（被裁掉的行不得漏进磁盘）",
            module.summary())
        table = _read_rows(os.path.join(directory, module.PROFILE_NAME))
        assert len(table) == cap + 1, ("表头 + K 行", len(table))
        kept = sorted(float(row[_column("compute_ms")]) for row in table[1:])
        assert kept == [1.0, 2.0, 3.0, 4.0], ("留下的必须是最快的 4 条", kept)

        # 重读一遍：文件里也只有 K 行，索引不反弹。
        assert module.load(force=True) == cap, module.summary()
        assert module.summary()["rows"] == cap, module.summary()
        assert module.best(_key())["compute_ms"] == "1.000"


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
        module.load()
        # 先落一行：子进程复位 _loaded/_index 后要从文件真的读回来，才能证明
        # 它拿到的不是"父进程那一刻的内存快照"。
        assert module.record(_row(kernel="forked_row", compute_ms=2.0)) is True
        assert module.flush() == 1

        status, payload = _fork_summary_while_lock_held(module, _LIVENESS_S)

        assert status is not None, "父进程没有拿到子进程的退出状态"
        assert os.WIFEXITED(status), ("子进程被信号杀死", status)
        assert os.WEXITSTATUS(status) == 0, (
            "子进程里的 summary() 抛异常了", payload.decode("utf-8", "replace"))
        decoded = json.loads(payload.decode("utf-8"))
        assert decoded["enabled"] is True, decoded
        assert decoded["keys"] == 1, ("子进程必须按文件重建索引", decoded)
        assert decoded["rows"] >= 1, ("子进程必须真的读回那一行", decoded)
        assert decoded["dir"] == os.path.abspath(directory), decoded

        # 父进程这边：持锁线程已退出，锁完好可用（钩子只重建子进程的锁）。
        assert module.summary()["dir"] == os.path.abspath(directory)


# ── 6. 关闭零副作用 ─────────────────────────────────────────────────────────

def test_disabled_profile_leaves_no_trace():
    """``PYQCD_EINSUM_PROFILE=0``：不建目录、``record()`` False、``summary`` 为假。"""
    with _isolated(disabled=True) as work:
        module = _profile_cache
        assert module.enabled() is False
        assert module.record(_row()) is False
        assert module.record_compare({**_key(), "compare": "ij->ji"}) is False
        assert module.load() == 0
        assert module.flush() == 0

        state = module.summary()
        assert state["enabled"] is False, state
        assert state["rows"] == 0 and state["keys"] == 0, state
        assert state["loaded"] is False, state
        assert module.notice() is None, "关闭时不该有提示"

        # 零副作用：整个临时 CWD 里连目录都不许出现。
        assert os.listdir(work) == [], ("关闭时不得产生任何文件", os.listdir(work))
        assert not os.path.exists(os.path.join(work, module.DEFAULT_DIR)), (
            "关闭时连 ./.cache 都不许建")


TESTS = (
    test_ensure_configured_keeps_pinned_dir,
    test_corrupt_rows_are_skipped_not_fatal,
    test_overlong_text_fields_are_clipped,
    test_per_key_cap_keeps_fastest_k_in_memory_and_on_disk,
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
