"""契约：``--tune`` 的**暂存落点 / 空间预检 / 失败自清 / 操作数生成**（2026-11-20）。

钉的是用户那条真实故障
----------------------
生产形状（10 操作数、2.56 亿元素、**1.95 GiB**）上跑一轮 tune，29 组**全部**
``error``，``record.csv`` 里是 ``error=OSError`` 与 ``error=KeyError``。真正的原因只有
一条：``/tmp`` 是 tmpfs、只剩 1.3 GiB，写**第 3 个**操作数时 ``ENOSPC``；而失败之后
**没有清理**已经写下的半份文件，2.0 GiB 残留在 ``/tmp`` 里毒害下一轮，于是每一轮都
以"全是 error"收场，而每一轮都以为自己在排查 tune 逻辑。

这四件事**都不会让任何数值测试变红**——落点选错只是慢一点、空间不够只是报错、残留
只是让下一轮也失败、多一个 ``complex128`` 中间体只是慢 10 秒且多占 4 倍内存。所以
必须各有一条断言盯着，否则下次有人"顺手简化"就再也没人发现。

八条断言
--------
1. **落点按顺序挑 tmpfs**：``/dev/shm`` 存在且可写时 ``_pick_stage_root(0)`` 返回
   它；把可写的候选改成只有 ``/run/shm`` 时返回 ``/run/shm``——证明是**按
   ``_TMPFS_ROOTS`` 顺序**挑的，而不是写死某一个路径；
2. **显式配置压过一切**：``PYQCD_EINSUM_TUNE_STAGE_DIR`` 给出时必须原样返回它，
   **即使 tmpfs 可用**（用户说了算，空间不够留到第 6 条去说清）；
3. **空间不足时跳过该落点**：``_available_bytes`` 对 tmpfs 返回 100 字节、对系统临时
   目录返回 1 TiB 时，``needed=1 GiB`` 必须跳过 tmpfs——这正是 2026-11-20 那条
   ``/tmp`` 只剩 1.3 GiB 的形状；
4. **全都不够时仍返回一个落点**：所有候选都只剩 1 字节也不许抛、不许返回 ``None``——
   "挑不到够大的"不该在**选择**阶段失败，失败要在 :func:`_ensure_room` 里说清；
5. **``required_bytes`` 含余量**：三个已知 ``nbytes`` 的数组 → 返回值落在
   ``[sum×1.05, sum×1.10)``；空序列只剩余量本身（4096 字节）；
6. **``_ensure_room`` 的报错正文**：必须同时说清"需要多少 GiB""只剩多少 GiB""把
   ``PYQCD_EINSUM_TUNE_STAGE_DIR`` 指到更大的盘"——用户看不出这三个数就会去怀疑
   tune 逻辑（实测原话）；
7. **写盘中途失败会自清**：第 2 个 ``np.save`` 抛 ``OSError("disk full")`` 时，异常
   必须向上冒、暂存目录必须已经不存在、``operand_*.npy`` 一个都不许留下——残留
   2.0 GiB 是"每轮全是 OSError"的第二个原因；
8. **操作数生成不产生 ``complex128`` 中间体**：``complex64`` 必须在目标精度上直接生成
   （源码里 ``np.empty(shape, dtype=complex_dtype)`` 在位、``complex128`` 不出现在
   ``astype`` 同一行），并用实测耗时（``complex64`` < ``complex128`` × 1.2）与
   "同 seed 两次结果逐位相同"两条旁证钉住——旧实现是 ``rng`` 5.09 s + ``build``
   9.70 s = 14.99 s 且峰值是最终体积的 4 倍。顺带反向钉住"提速不许把双精度也降下去"：
   ``complex128`` 的值不许**全部**等于 ``float32`` 再升精度的结果。

隔离口径：临时目录一律在 ``data/tmp/`` 下建、跑完删干净；**绝不**写
``~/.cache/pyqcd.einsum``、**绝不**在 ``/dev/shm`` 里建真实目录（第 1 条只调
``_pick_stage_root``，它自己不建目录）；不碰配置表、不跑 ``contract()``、不假设本机
一定有 ``/dev/shm``（都用 monkeypatch 造出候选）。第 8 条的 (400,400,400) 在内存不足
时自动降一档形状，判定口径不变。

可直接运行::

    python -B -m pyqcd.testing._einsum_tune_stage_contract
"""
from __future__ import annotations

import ast
import contextlib
import gc
import inspect
import os
import shutil
import tempfile
import textwrap
import time
import unittest
from typing import Any, Iterator, Sequence
from unittest.mock import patch

import numpy as np

from ..einsum import _tune_runner

__all__ = ["test_einsum_tune_stage_contracts"]

#: 临时区环境变量（与 ``_einsum_peak_memory_contract`` 同一套约定）。
_TMP_ENV = "PYQCD_TEST_TMP"

#: 1 GiB 的字节数（``statvfs`` 与报错正文都按 1024² 折算）。
GIB = 2 ** 30

#: 第 6 条那两个数：生产形状实测的"需要 1.95 GiB、只剩 1.30 GiB"。
PROD_NEEDED = int(1.95 * GIB)
PROD_LEFT = int(1.30 * GIB)

#: 第 3/4 条的需求量：1 GiB（远大于伪造的 100 字节/1 字节）。
NEED_ONE_GIB = 1 * GIB

#: 第 8 条的计时形状（生产形状的 2.56 亿元素是同一个数量级）。
GEN_SHAPE = (400, 400, 400)

#: 第 8 条的低内存降档形状（(400,400,400) 的 complex128 就要 512 MB，RNG 还要再来一份）。
GEN_SHAPE_SMALL = (256, 256, 256)

#: 第 8 条的判定条件：``complex64`` 的耗时 < ``complex128`` × 1.2。
#:
#: 本机实测比值 ≈ 0.80（complex64 1.11–1.16 s vs complex128 1.36–1.39 s）；放宽到 1.2
#: 是为了不卡机器抖动，同时仍能抓住"先造 complex128 再 astype"那条旧路（它必然让
#: complex64 比 complex128 **更慢**，因为多了一次 512 MB 的拷贝）。
GEN_MAX_RATIO = 1.2

#: 第 8 条的重复次数（取最小值，压掉抖动）。
GEN_REPEATS = 3

#: 第 8 条启动计时前需要的空闲内存门槛：低于它就降一档形状。
GEN_MEM_FLOOR = 6 * GIB


# ---------------------------------------------------------------------------
# 隔离：临时区在 ``data/tmp/`` 下，退出即删
# ---------------------------------------------------------------------------
def _repo_root() -> str:
    """仓库根目录（本文件在 ``<root>/pyqcd/testing/`` 下）。"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _tmp_root() -> str:
    """临时根：``data/tmp/``；不可写时退回系统临时目录。"""
    override = os.environ.get(_TMP_ENV)
    candidates = [override] if override else []
    candidates.append(os.path.join(_repo_root(), "data", "tmp"))
    for candidate in candidates:
        if not candidate:
            continue
        try:
            os.makedirs(candidate, exist_ok=True)
            probe = tempfile.mkdtemp(prefix="stageprobe-", dir=candidate)
            os.rmdir(probe)
            return candidate
        except Exception:                             # noqa: BLE001 - 换下一个
            continue
    return tempfile.gettempdir()


class _TempDir:
    """``with _TempDir() as path:`` —— 在 ``data/tmp/`` 下建一个用完即删的目录。"""

    def __init__(self, prefix: str = "tune-stage-") -> None:
        self._prefix = prefix
        self.path = ""

    def __enter__(self) -> str:
        self.path = tempfile.mkdtemp(prefix=self._prefix, dir=_tmp_root())
        return self.path

    def __exit__(self, *exc_info: Any) -> None:
        shutil.rmtree(self.path, ignore_errors=True)
        return None


@contextlib.contextmanager
def _env(name: str, value: str | None) -> Iterator[None]:
    """临时设置/清除一个环境变量；退出时**原样**还回去（含"本来就没设"）。"""
    saved = os.environ.get(name)
    try:
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
        yield
    finally:
        if saved is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = saved


@contextlib.contextmanager
def _fake_tmpfs(visible: Sequence[str]) -> Iterator[None]:
    """让 ``os.path.isdir`` / ``os.access`` 只把 ``visible`` 里的 tmpfs 根当成可用。

    为什么不直接依赖本机：``/dev/shm`` 在容器里可能不存在、``/run/shm`` 在某些发行版
    是软链；断言"本机一定有"会让契约在别的机器上假红。这里只伪造**候选根的可见性**，
    其余路径照旧走真实实现（``tempfile`` / ``os.makedirs`` 因此不受影响）。

    注意本函数**不动** ``_available_bytes``：第 1 条的 ``needed=0``，无论真实空间多少
    都会命中第一个可见候选（``room is None or room >= 0`` 恒真）。
    """
    real_isdir = os.path.isdir
    real_access = os.access
    wanted = set(visible)
    roots = set(_tune_runner._TMPFS_ROOTS)

    def _only_root(path: Any) -> str | None:
        try:
            text = os.fspath(path)
        except TypeError:
            return None
        if isinstance(text, bytes):
            text = text.decode("utf-8", "replace")
        return text if text in roots else None

    def fake_isdir(path: Any) -> bool:
        root = _only_root(path)
        return real_isdir(path) if root is None else root in wanted

    def fake_access(path: Any, mode: int, **kwargs: Any) -> bool:
        root = _only_root(path)
        return real_access(path, mode, **kwargs) if root is None else root in wanted

    with patch("os.path.isdir", fake_isdir), patch("os.access", fake_access):
        yield


@contextlib.contextmanager
def _stage_gettempdir() -> Iterator[str]:
    """产出"系统临时目录"：**优先**用真实的 ``tempfile.gettempdir()``。

    只有在真实的那个落在 ``_TMPFS_ROOTS`` 里时才伪造一个（``TMPDIR=/dev/shm`` 的容器
    就是这样）：那时"系统临时目录"与"tmpfs 候选"是同一个路径，第 1/3/4 条的两档根本
    分不开。伪造之后这三条在任何机器上都成立，不必 SkipTest，也不留"看似通过的空
    断言"。
    """
    roots = set(_tune_runner._TMPFS_ROOTS)
    real = tempfile.gettempdir()
    if os.path.abspath(real) not in roots:
        yield real
        return
    with _TempDir(prefix="tune-stage-gettemp-") as directory:
        assert os.path.abspath(directory) not in roots, (
            f"伪造的系统临时目录落在 tmpfs 候选里：{directory!r}")
        with patch("tempfile.gettempdir", lambda: directory):
            yield directory


def _cli() -> Any:
    """CLI 模块（懒取，避免本模块 import 期就拖进 argparse）。"""
    from ..einsum import test as cli

    return cli


def _mem_available_bytes() -> int | None:
    """``/proc/meminfo`` 的 ``MemAvailable``；读不到返回 ``None``（不猜）。"""
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):          # pragma: no cover - 非 Linux
        return None
    return None


# ---------------------------------------------------------------------------
# 1. 落点优先 tmpfs，且按 _TMPFS_ROOTS 的顺序挑
# ---------------------------------------------------------------------------
def test_stage_root_prefers_tmpfs_in_declared_order() -> None:
    """契约 1/8：``needed=0`` 时按 ``_TMPFS_ROOTS`` 的顺序挑第一个可用的 tmpfs。

    回归的故障：落点写死成 ``tempfile.gettempdir()``（本机那是个只剩 1.3 GiB 的
    tmpfs），1.95 GiB 的操作数在第 3 个上 ``ENOSPC``。这里不假设本机有 ``/dev/shm``：
    全部用 monkeypatch 造出"哪些根可见"，因此两种情形在任何机器上都跑得出来。

    * 两个根都可用 → 必须是 ``_TMPFS_ROOTS[0]``（``/dev/shm``）；
    * 只有 ``/run/shm`` 可用 → 必须是 ``/run/shm``（证明不是写死 ``/dev/shm``）。

    顺带：本机第一个根真在的时候再断一次不伪造的结果（不存在就跳过这一小段，不算失败）。
    """
    roots = tuple(_tune_runner._TMPFS_ROOTS)
    assert roots == ("/dev/shm", "/run/shm"), (
        f"_TMPFS_ROOTS 的顺序变了：{roots}——顺序即优先级，改它等于改契约")
    with _stage_gettempdir() as gettemp, _env(_tune_runner.STAGE_ENV, None):
        with _fake_tmpfs(roots):
            first = _tune_runner._pick_stage_root(0)
        assert first == roots[0], (
            f"两个 tmpfs 都可用时必须选第一个 {roots[0]!r}，实际 {first!r}"
            f"（顺序 = {roots}）")
        assert first != gettemp, (
            f"必须落在 tmpfs 而不是系统临时目录 {gettemp!r}：{first!r}")
        with _fake_tmpfs((roots[1],)):
            second = _tune_runner._pick_stage_root(0)
        assert second == roots[1], (
            f"只有 {roots[1]!r} 可用时必须选它，实际 {second!r}"
            "——写死 /dev/shm 的实现会在这里返回不可用的路径")
        assert second != gettemp, (
            f"仍须避开系统临时目录：{second!r} == gettempdir()")
        # 两个 tmpfs 都不可见时必须退回系统临时目录（不抛、不返回不可用路径）。
        with _fake_tmpfs(()):
            fallback = _tune_runner._pick_stage_root(0)
        assert fallback == gettemp, (
            f"tmpfs 全都不可用时必须退回 {gettemp!r}，实际 {fallback!r}")
    # 真实机器（不伪造）：根真在的时候顺带核对一次——不存在就跳过这一小段。
    if os.path.isdir(roots[0]) and os.access(roots[0], os.W_OK):
        with _env(_tune_runner.STAGE_ENV, None):
            real = _tune_runner._pick_stage_root(0)
        assert real == roots[0], (
            f"本机 {roots[0]!r} 存在且可写，_pick_stage_root(0) 却返回 {real!r}")


# ---------------------------------------------------------------------------
# 2. 显式配置压过一切
# ---------------------------------------------------------------------------
def test_explicit_stage_dir_beats_every_candidate() -> None:
    """契约 2/8：``PYQCD_EINSUM_TUNE_STAGE_DIR`` 给出时必须原样返回它。

    回归的故障：落点选择里没有"用户覆盖"这一档——盘满了只能改代码，改不了环境。
    这里**即使 tmpfs 可用**也要求返回显式目录，并且连"1 GiB 放不下"都不该拦它：
    空间判定属于 :func:`_ensure_room`，不属于选择阶段（第 4 条同一口径）。
    """
    roots = tuple(_tune_runner._TMPFS_ROOTS)
    with _TempDir(prefix="tune-stage-explicit-") as scratch:
        explicit = os.path.join(scratch, "user-chosen-stage")
        with _env(_tune_runner.STAGE_ENV, explicit):
            with _fake_tmpfs(roots):
                chosen = _tune_runner._pick_stage_root(0)
                assert chosen == explicit, (
                    f"显式配置 {explicit!r} 必须压过 tmpfs {roots}，实际 {chosen!r}")
                assert os.path.isdir(explicit), (
                    f"显式配置的目录必须被建出来（否则 stage_dir 会在 mkdtemp 处炸）："
                    f"{explicit!r} 不存在")
                # 需要 1 GiB 放不下也照旧：选择的职责只是"用户说了算"。
                huge = _tune_runner._pick_stage_root(NEED_ONE_GIB)
                assert huge == explicit, (
                    f"显式配置在 needed={NEED_ONE_GIB} 时也必须原样返回，实际 {huge!r}")
        # 恢复环境变量之后不许再返回它。
        with _fake_tmpfs(roots):
            restored = _tune_runner._pick_stage_root(0)
        assert restored != explicit, (
            f"恢复环境变量后仍返回旧的显式目录：{restored!r}")


# ---------------------------------------------------------------------------
# 3./4. 空间不足：跳过该落点；全都不够仍返回一个
# ---------------------------------------------------------------------------
def test_insufficient_room_skips_that_root() -> None:
    """契约 3/8：tmpfs 只剩 100 字节、系统临时目录有 1 TiB → 必须跳过 tmpfs。

    形状照抄 2026-11-20：``/tmp``（就是 ``gettempdir()``）只剩一点点，而 ``/dev/shm``
    空着。旧实现在这里会按"顺序取第一个存在的"选落点，1.95 GiB 到第 3 个操作数才炸
    （而那时已经写了 1 GiB 垃圾）。

    ``gettempdir()`` 走 :func:`_stage_gettempdir`：容器里 ``TMPDIR=/dev/shm`` 时，
    "系统临时目录"与"tmpfs 候选"会是同一个路径，两档分不开（那时它就地伪造一个
    独立目录，断言照旧生效）。
    """
    roots = tuple(_tune_runner._TMPFS_ROOTS)
    with _stage_gettempdir() as gettemp:
        visited: list[str] = []

        def fake_available(path: str) -> int | None:
            visited.append(path)
            if path in roots:
                return 100                              # 只剩 100 字节
            if path == gettemp:
                return 10 ** 12                         # 1 TiB
            return 0

        with _env(_tune_runner.STAGE_ENV, None), _fake_tmpfs(roots), patch.object(
                _tune_runner, "_available_bytes", fake_available):
            chosen = _tune_runner._pick_stage_root(NEED_ONE_GIB)
    assert chosen == gettemp, (
        f"tmpfs 只剩 100 字节、{gettemp!r} 有 1 TiB 时，needed={NEED_ONE_GIB} "
        f"必须选 {gettemp!r}，实际 {chosen!r}")
    assert roots[0] in visited, (
        f"必须先检查过 {roots[0]!r} 才可能跳过它；实际访问顺序 = {visited}")


def test_selection_never_fails_when_nothing_fits() -> None:
    """契约 4/8：所有候选都放不下时仍返回 ``gettempdir()``，不抛、不返回 ``None``。

    "挑不到够大的落点"不是错误——机器上可能只有这么一块盘。真正该报错的地方是
    :func:`_ensure_room`（第 6 条），那里才说得清"需要多少、剩多少、怎么办"。
    如果在**选择**阶段抛，用户拿到的会是 ``OSError: no root fits`` 之类毫无信息量的
    东西，而真正的数字（1.95 GiB / 1.30 GiB）一个也不会出现。
    """
    roots = tuple(_tune_runner._TMPFS_ROOTS)
    with _stage_gettempdir() as gettemp:

        def one_byte(path: str) -> int | None:
            return 1

        with _env(_tune_runner.STAGE_ENV, None), _fake_tmpfs(roots), patch.object(
                _tune_runner, "_available_bytes", one_byte):
            try:
                chosen = _tune_runner._pick_stage_root(NEED_ONE_GIB)
            except Exception as exc:                  # noqa: BLE001 - 就是断言它不抛
                raise AssertionError(
                    f"所有候选都只剩 1 字节时 _pick_stage_root 抛了 "
                    f"{type(exc).__name__}: {exc}——选择阶段不许失败，"
                    "报错该由 _ensure_room 说清") from exc
    assert chosen is not None, "兜底落点不许是 None（stage_dir 会在 mkdtemp 处炸）"
    assert chosen == gettemp, (
        f"全都放不下时的兜底必须是 {gettemp!r}，实际 {chosen!r}")


# ---------------------------------------------------------------------------
# 5. required_bytes 的余量
# ---------------------------------------------------------------------------
def test_required_bytes_carries_a_margin() -> None:
    """契约 5/8：``required_bytes`` = ``sum(nbytes) × 1.05 + 4096``。

    余量不是保守主义：``np.save`` 要写头 + 对齐，文件系统还会留 metadata，差几十 KB
    同样让**第 N 个**操作数写到一半失败（那正是实测的现场）。上界 1.10 防的是
    "余量被改成 1.5 倍"这种反向错误——预检如果过于保守，1.95 GiB 会在明明放得下的
    磁盘上被判失败。

    4096 是**余量本身**：空序列也必须返回它（``.npy`` 头 + 对齐）。
    """
    operands = [np.zeros((500, 500), dtype=np.float32) for _ in range(3)]
    total = sum(int(item.nbytes) for item in operands)
    assert total == 3_000_000, f"前置：三个 500×500 float32 应当共 3 MB，实际 {total}"
    needed = _tune_runner.required_bytes(operands)
    assert needed >= total * 1.05, (
        f"required_bytes={needed} 小于 sum×1.05={total * 1.05:.0f}"
        f"（sum={total}）——余量没了，1.95 GiB 会写到一半 ENOSPC")
    assert needed < total * 1.10, (
        f"required_bytes={needed} 已经 >= sum×1.10={total * 1.10:.0f}（sum={total}）"
        "——余量过大同样有害：明明放得下的盘会被预检判死")
    empty = _tune_runner.required_bytes([])
    assert empty == 4096, (
        f"空序列只能剩余量本身 4096（.npy 头 + 对齐），实际 {empty}")
    # 单调性：多一个操作数就多一份（含那 5%），否则预检会漏掉最后几个。
    one = _tune_runner.required_bytes(operands[:1])
    assert 0 < one < needed, (one, needed)


# ---------------------------------------------------------------------------
# 6. _ensure_room 的报错正文
# ---------------------------------------------------------------------------
def test_ensure_room_message_states_need_left_and_fix() -> None:
    """契约 6/8：报错正文必须同时含"需要 X GiB""只剩 Y GiB""换个落点怎么办"。

    回归的故障：旧实现什么都不预检，用户看到的是
    ``OSError: [Errno 28] No space left on device``——**29 组各报一次同一句话**，
    看不出要多少、剩多少、也看不出换个目录就能跑。这条断言把那句话钉成结构：
    四个关键词 + 两个数字缺一个就算回归。
    """
    directory = "/tmp/pyqcd-einsum-tune-probe"       # 只是字符串，不会真写
    with patch.object(_tune_runner, "_available_bytes", lambda path: PROD_LEFT):
        try:
            _tune_runner._ensure_room(directory, PROD_NEEDED)
        except OSError as exc:
            message = str(exc)
        except Exception as exc:                      # noqa: BLE001 - 抓错类型也算失败
            raise AssertionError(
                f"空间不足时必须抛 OSError，实际 {type(exc).__name__}: {exc}") from exc
        else:
            raise AssertionError(
                f"需要 {PROD_NEEDED / GIB:.2f} GiB、只剩 {PROD_LEFT / GIB:.2f} GiB 时"
                "竟然没抛 OSError（预检被绕过了？）")
    for keyword in ("需要", "只剩", "GiB", _tune_runner.STAGE_ENV):
        assert keyword in message, (
            f"报错正文缺 {keyword!r}：{message!r}")
    assert "1.95" in message, (
        f"正文里没有需求数字 1.95（{PROD_NEEDED} 字节）：{message!r}")
    assert "1.30" in message, (
        f"正文里没有剩余数字 1.30（{PROD_LEFT} 字节）：{message!r}")

    # 反面：空间够、或者测不出来，都**不许**抛——否则 tune 在 NFS/FUSE 上直接全 error。
    for label, room in (("空间充足", PROD_NEEDED + 1), ("测不出空间", None)):
        with patch.object(_tune_runner, "_available_bytes", lambda path, r=room: r):
            try:
                _tune_runner._ensure_room(directory, PROD_NEEDED)
            except Exception as exc:                  # noqa: BLE001
                raise AssertionError(
                    f"{label}（room={room}）时 _ensure_room 不许抛，实际 "
                    f"{type(exc).__name__}: {exc}") from exc


# ---------------------------------------------------------------------------
# 7. 写盘中途失败自清
# ---------------------------------------------------------------------------
def test_failed_write_cleans_its_own_residue() -> None:
    """契约 7/8：第 2 个 ``np.save`` 抛 ``OSError`` → 异常上冒 + 暂存目录已不存在。

    回归的故障（2026-11-20 的第二个原因）：写第 3 个操作数时 ``ENOSPC``，**已写的
    2.0 GiB 留在 ``/tmp`` 里**，下一轮更没空间，于是每一轮都以"全是 OSError"收场。
    修复是 ``try/except BaseException`` 里先 ``cleanup(directory)`` 再 ``raise``。

    为了证明测的是"真失败"而不是"一开始就没写"，这里：
    * 第 1 个操作数**真的**写下去（记录文件路径与字节数）；
    * 第 2 个先写半份再抛（模拟 ENOSPC 现场）；
    * 事后整棵 ``scratch`` 递归找 ``operand_*.npy``，必须一个都不剩。
    """
    real_save = np.save
    with _TempDir(prefix="tune-stage-fail-") as scratch:
        stage = os.path.join(scratch, "stage")
        os.makedirs(stage, exist_ok=True)
        operands = [np.zeros((4, 4), dtype=np.complex128),
                    np.ones((4, 4), dtype=np.complex128)]
        saved: list[tuple[str, int]] = []
        attempts: list[str] = []
        cleanup_calls: list[str] = []

        def fake_save(path: str, array: Any, **kwargs: Any) -> Any:
            attempts.append(str(path))
            if len(attempts) == 1:
                real_save(path, array, **kwargs)
                saved.append((str(path), os.path.getsize(path)))
                return None
            # 半份文件：ENOSPC 时 .npy 已经写了一半（这正是残留的来源）。
            with open(path, "wb") as handle:
                handle.write(b"\x93NUMPY-half-written")
            raise OSError("disk full")

        real_cleanup = _tune_runner.cleanup

        def spy_cleanup(directory: str) -> None:
            cleanup_calls.append(str(directory))
            real_cleanup(directory)

        with patch("numpy.save", fake_save), patch.object(
                _tune_runner, "cleanup", spy_cleanup):
            try:
                _tune_runner.stage_operands(operands, stage)
            except OSError as exc:
                raised = exc
            except Exception as exc:                  # noqa: BLE001 - 类型不对也算失败
                raise AssertionError(
                    f"写盘失败必须原样抛出 OSError，实际 {type(exc).__name__}: {exc}"
                ) from exc
            else:
                raise AssertionError(
                    "np.save 在第 2 个操作数上抛了 OSError，stage_operands 却正常返回"
                    "——失败被吞掉了")

        assert "disk full" in str(raised), (
            f"抛出的异常必须是原来那条（不许包装成别的）：{raised!r}")
        assert len(attempts) == 2, (
            f"应当只尝试了 2 次 np.save（第 1 次成功、第 2 次失败），实际 {attempts}")
        assert saved, "前置失败：第 1 个操作数没有真的写下去，这条断言失去意义"
        first_path, first_bytes = saved[0]
        assert os.path.basename(first_path) == "operand_00.npy", (
            f"第 1 个操作数应写成 operand_00.npy，实际 {first_path!r}")
        assert first_bytes > 0, f"第 1 个操作数写出来是空文件：{first_path!r}"
        assert cleanup_calls == [stage], (
            f"失败路径必须调用 cleanup({stage!r}) 一次，实际 {cleanup_calls}")
        assert not os.path.exists(stage), (
            f"失败后暂存目录必须已被删除，实际仍存在：{stage!r}"
            f"（残留 {_residue(stage)}）")
        leftovers = _residue(scratch)
        assert leftovers == [], (
            f"整棵临时区里不许留下 operand_*.npy，实际 {leftovers}"
            "——这正是毒害下一轮那 2.0 GiB 的来源")


def _residue(root: str) -> list[str]:
    """``root`` 下所有 ``operand_*.npy``（递归；目录已删时返回空表）。"""
    found: list[str] = []
    for base, _dirs, files in os.walk(root):
        for name in files:
            if name.startswith("operand_") and name.endswith(".npy"):
                found.append(os.path.join(base, name))
    return sorted(found)


# ---------------------------------------------------------------------------
# 8. 操作数生成不产生 complex128 中间体
# ---------------------------------------------------------------------------
def test_operand_generation_has_no_complex128_detour() -> None:
    """契约 8/8：``complex64`` 在目标精度上直接生成，``complex128`` 只出现在另一支。

    回归的故障：旧实现先造两个 ``float64``、再 ``real + 1j*imag`` 拼出一个
    ``complex128`` 临时体、最后 ``astype(complex64)``——生产形状（2.56 亿元素）上实测
    ``rng`` 5.09 s + ``build`` 9.70 s = **14.99 s**，峰值内存是最终体积的 **4 倍**。

    三条证据（缺一条就不足以排除回归）：

    1. **形状**：``(8,8) complex64`` 的返回 dtype 必须是 ``complex64``，且虚部非零
       （预分配 ``np.empty`` 后忘了填 ``.imag`` 也能让 dtype 对）；
    2. **源码**：必须含 ``np.empty(shape, dtype=complex_dtype)``；``complex128`` 不许
       与 ``astype`` 出现在同一行，也不许写成
       ``np.complex128 if real_dtype is np.float32``（极性反了：单精度实数会配出双精度
       复数，等于白改）；
    3. **实测**：``(400,400,400)`` 上 ``complex64`` 的耗时 < ``complex128`` × 1.2，
       并且同 seed 两次调用逐位相同（提速不许以"每次结果不同"为代价）。
    """
    cli = _cli()
    dtype64 = np.dtype("complex64")
    host, mirror = cli._make_operands(np, [(8, 8)], dtype64, 7)
    assert isinstance(host, list) and len(host) == 1, host
    assert host[0].shape == (8, 8), host[0].shape
    assert host[0].dtype == dtype64, (
        f"complex64 请求必须返回 complex64，实际 {host[0].dtype}")
    assert np.any(host[0].imag != 0.0), (
        "生成结果虚部全零——预分配 np.empty 之后没填 .imag 视图")
    assert np.any(host[0].real != 0.0), "生成结果实部全零——没填 .real 视图"
    assert mirror is host, (
        "xp 为 numpy 时两路结果必须是同一个列表（否则多了一份拷贝）")

    # 只取**代码**：docstring 与注释里本来就要讲"旧的 complex128 + astype 路线"，
    # 直接 grep ``inspect.getsource`` 会被自己的说明文字绊倒（实测过）。
    source = _code_only(cli._make_operands)
    assert "np.empty(shape, dtype=complex_dtype)" in source, (
        "源码里没有 `np.empty(shape, dtype=complex_dtype)`——目标精度预分配被改掉了；"
        f"当前代码：{source}")
    assert "np.complex128 if real_dtype is np.float32" not in source, (
        "三元表达式的极性反了：单精度实数配出了双精度复数（complex64 请求会先造出 "
        f"complex128）；当前代码：{source}")
    for line in source.splitlines():
        if "complex128" in line:
            assert "astype" not in line, (
                f"complex128 与 astype 出现在同一行，怀疑又走回降精度那条路：{line!r}")
    # ``complex_dtype`` 必须由 ``real_dtype`` 决定（单精度实数 → 单精度复数）。
    assert "np.complex64 if real_dtype is np.float32 else np.complex128" in source, (
        f"complex_dtype 的选择不是「单精度实数配单精度复数」：{source}")

    # 双精度路径必须原样可用（改单精度不许把 complex128 一起降下去）。
    host128, _ = cli._make_operands(np, [(64, 64)], np.dtype("complex128"), 7)
    assert host128[0].dtype == np.dtype("complex128"), host128[0].dtype
    rounded = host128[0].real.astype(np.float32).astype(np.float64)
    assert not np.array_equal(host128[0].real, rounded), (
        "complex128 的实部**全部**等于 float32 再升精度的结果——双精度路径被降精度了"
        "（tune 出来的 rel 会变成单精度的口径）")

    # 可复现：同 seed 两次调用逐位相同。
    again, _ = cli._make_operands(np, [(8, 8)], dtype64, 7)
    assert np.array_equal(again[0], host[0]), (
        "同 seed 两次调用结果不同——操作数不再是可复现的")
    other, _ = cli._make_operands(np, [(8, 8)], dtype64, 8)
    assert not np.array_equal(other[0], host[0]), "换了 seed 结果却一样（seed 没接上）"

    # 实测：complex64 不许比 complex128 慢。
    shape = GEN_SHAPE
    need = int(np.prod(shape)) * 16 * 2               # 数组 + RNG 的临时体
    budget = _mem_available_bytes()
    note = ""
    if budget is not None and budget < max(need, GEN_MEM_FLOOR):
        shape = GEN_SHAPE_SMALL
        note = f"（MemAvailable={budget / GIB:.1f} GiB 偏小，降档到 {shape}）"
    fast = _fastest_generation(cli, shape, "complex64")
    slow = _fastest_generation(cli, shape, "complex128")
    assert slow > 0.0, f"complex128 的计时是 0，无法比较：{slow!r}"
    ratio = fast / slow
    assert ratio < GEN_MAX_RATIO, (
        f"生成 {shape} 的 complex64 用了 {fast:.3f}s，complex128 用了 {slow:.3f}s，"
        f"比值 {ratio:.2f} >= {GEN_MAX_RATIO}{note}——说明仍在造 complex128 中间体"
        "（旧的 rng+build 路线实测 14.99 s、峰值 4 倍）")


def _fastest_generation(cli: Any, shape: tuple[int, ...], dtype: str) -> float:
    """``shape`` 上生成一次操作数的最短耗时（秒），每次生成完立刻释放。"""
    best = float("inf")
    for _ in range(GEN_REPEATS):
        started = time.perf_counter()
        host, _mirror = cli._make_operands(np, [shape], np.dtype(dtype), 11)
        elapsed = time.perf_counter() - started
        del host, _mirror
        gc.collect()
        best = min(best, elapsed)
    return best


def _code_only(function: Any) -> str:
    """``function`` 的**纯代码**文本（docstring 去掉、注释自然消失、格式由 AST 归一）。

    为什么不用 ``inspect.getsource`` 直接 grep：``_make_operands`` 的历史就写在它的
    docstring 里（"再造一个 ``complex128`` 临时体、最后 ``astype`` 降精度"），grep 全文
    等于让说明文字把自己判红（第一版实现正是这样假红的）。这里取的是能直接执行的代码。
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    node = tree.body[0]
    assert isinstance(node, ast.FunctionDef), (
        f"inspect.getsource 取到的不是函数定义：{type(node).__name__}")
    body = node.body
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]                                # 去掉 docstring
    return "\n".join(ast.unparse(item) for item in body)


TESTS = (
    test_stage_root_prefers_tmpfs_in_declared_order,
    test_explicit_stage_dir_beats_every_candidate,
    test_insufficient_room_skips_that_root,
    test_selection_never_fails_when_nothing_fits,
    test_required_bytes_carries_a_margin,
    test_ensure_room_message_states_need_left_and_fix,
    test_failed_write_cleans_its_own_residue,
    test_operand_generation_has_no_complex128_detour,
)


# ---------------------------------------------------------------------------
# 汇总入口
# ---------------------------------------------------------------------------
def _run() -> tuple[int, int, list[str]]:
    """跑全部断言；返回 ``(passed, skipped, failures)``。"""
    passed = skipped = 0
    failures: list[str] = []
    for test in TESTS:
        try:
            test()
        except unittest.SkipTest as exc:
            skipped += 1
            print(f"SKIP {test.__name__}: {exc}", flush=True)
        except Exception as exc:                      # noqa: BLE001 - 报告用
            failures.append(f"{test.__name__}: {type(exc).__name__}: {exc}")
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}", flush=True)
        else:
            passed += 1
            print(f"PASS {test.__name__}", flush=True)
    return passed, skipped, failures


def test_einsum_tune_stage_contracts() -> None:
    """``--tune`` 暂存落点/空间预检/失败自清/操作数生成的八条契约。

    失败时抛 :class:`AssertionError` 并列出每条失败；全部跳过时抛
    :class:`unittest.SkipTest`（不留一个"看起来通过"的空套件）。

    收尾把 ``PYQCD_EINSUM_TUNE_STAGE_DIR`` 还原成进套件之前的样子——第 2 条会改它，
    漏还的话后面所有读环境的用例（以及用户自己的 tune）都会指向一个已删的临时目录。
    """
    saved_stage = os.environ.get(_tune_runner.STAGE_ENV)
    started = time.perf_counter()
    passed, skipped, failures = _run()
    if os.environ.get(_tune_runner.STAGE_ENV) != saved_stage:
        failures.append(
            f"teardown: {_tune_runner.STAGE_ENV} 没有还原"
            f"（现为 {os.environ.get(_tune_runner.STAGE_ENV)!r}，"
            f"应为 {saved_stage!r}）")
        if saved_stage is None:
            os.environ.pop(_tune_runner.STAGE_ENV, None)
        else:
            os.environ[_tune_runner.STAGE_ENV] = saved_stage
    elapsed = time.perf_counter() - started
    if failures:
        raise AssertionError(
            f"{len(failures)} 条 tune 暂存契约失败（{passed} passed, {skipped} skipped, "
            f"{elapsed:.2f}s）：" + "; ".join(failures))
    if passed == 0:
        raise unittest.SkipTest(
            f"tune 暂存契约全部跳过（{skipped} skipped）：没有可验证的来源")


def main() -> int:
    """独立运行入口：``python -B -m pyqcd.testing._einsum_tune_stage_contract``。"""
    passed, skipped, failures = _run()
    print(f"{passed} passed, {skipped} skipped, {len(failures)} failed", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
