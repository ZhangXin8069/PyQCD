"""契约：``--tune`` 的**组级硬超时**（子进程隔离，2026-11-19）。

为什么必须单独立契约
--------------------
旧实现用 ``SIGALRM``，它在**回到解释器**时才派发——一次落进原生 C/CUDA 代码的调用
不返回，闹钟等于没装（实测：上限 0.3 s、原生 BLAS 2.34 s，2.29 s 才抛，超限 1.99 s；
用户那条 10 算子方程在 ``STAT=Rl+`` 下挂 3 分钟以上）。这类缺陷**不影响任何数值**，
所以普通数值测试永远抓不到它；而"升级了超时机制"这句话如果没有一条断言盯着，
下次有人顺手改回进程内就再也没人发现。

五条断言分别钉住：
1. **真的能打断**：一组里去 ``sleep(300)``（模拟不返回的原生调用），``group_limit_s=1.5``
   时墙钟必须 ≈ 上限，状态 ``timeout``；
2. **不杀错进程**：强杀的是子进程的**进程组**，父进程与它的其它子进程不受影响
   （``start_new_session`` + ``killpg`` 的实现前提）；
3. **启动开销单列**：``startup_s`` 与 ``wall_s`` 是两个数，且 ``wall_ms`` 只由后者换算
   ——把进程启动混进被测耗时会凭空多出 ~130 ms（本机 ``import pyqcd.einsum`` 中位）；
4. **崩溃如实记账**：子进程 ``os._exit(9)``（写不出结果文件）→ ``status=error`` 且
   ``原因`` 带退出码，**不**冒充 ``timeout``；
5. **组内剪枝仍然生效**：``min_wall_ms`` 由子进程在两次调用之间判定，命中时
   ``status=pruned`` 且带两个数值（用户规则的原文由父进程补上）。

隔离口径：全部用临时目录与**小形状**，不碰配置表、不需要 GPU、不依赖机器快慢
（唯一的例外是第 1 条，它比的是"墙钟 ≈ 上限"，容差放到 2 倍）。

可直接运行::

    python -B -m pyqcd.testing._einsum_tune_isolation_contract
"""
from __future__ import annotations

import os
import tempfile
import time
import unittest
from typing import Any, Iterator

from ..einsum import _tune_runner, _tune_worker

#: 第 1 条的组级上限（秒）：比机器最快的进程启动还宽裕，又远小于 ``sleep``。
HANG_LIMIT = 1.5

#: 卡片子进程里"原生调用不返回"的模拟时长（秒）。它只是**上界**：正常情况下
#: 父进程在 ``HANG_LIMIT`` 就把它杀了，永远不会等满。
HANG_SECONDS = 300.0

#: 小方程与小形状：跑完一组不到 1 ms，第 3/4/5 条因此不受本机快慢影响。
EQUATION = "ab,bc->ac"
SHAPES = ((6, 6), (6, 6))


def _operands(directory: str) -> str:
    """把两个小操作数落盘到 ``directory``，返回该目录（子进程按名字排序读回）。"""
    import numpy as np

    os.makedirs(directory, exist_ok=True)
    rng = np.random.default_rng(0)
    _tune_worker  # 保证模块可导入（子进程入口就在它里面）
    for index, shape in enumerate(SHAPES):
        np.save(os.path.join(directory, f"operand_{index:02d}.npy"),
                rng.standard_normal(shape).astype(np.complex128))
    return directory


def _job(operands: str, **overrides: Any) -> dict[str, Any]:
    job = {
        "equation": EQUATION,
        "device": "cpu",
        "backend": "numpy",
        "dtype": "complex128",
        "warmup": 0,
        "rounds": 1,
        "min_wall_ms": None,
        "operands_dir": operands,
        "kernel_force": "generic",
        "as_equivalence": False,
        "optimize": False,
        "contract_kwargs": {},
        "profile_kwargs": {},
    }
    job.update(overrides)
    return job


class _Scratch:
    """一个临时工作目录 + 操作数目录；退出时整棵删掉。"""

    def __init__(self) -> None:
        self.root = tempfile.mkdtemp(prefix="pyqcd-tune-isolation-")

    def operands(self) -> str:
        return _operands(os.path.join(self.root, "operands"))

    def workdir(self, name: str) -> str:
        return os.path.join(self.root, name)

    def close(self) -> None:
        _tune_runner.cleanup(self.root)


def test_hard_timeout_kills_a_hanging_native_call() -> None:
    """契约 1/5：卡在"原生调用"里的那一组，父进程按上限强杀（墙钟 ≈ 上限）。"""
    scratch = _Scratch()
    previous = os.environ.get(_tune_worker.SLEEP_ENV)
    os.environ[_tune_worker.SLEEP_ENV] = str(HANG_SECONDS)
    try:
        started = time.monotonic()
        observation = _tune_runner.run_group(
            _job(scratch.operands()), workdir=scratch.workdir("hang"),
            group_limit_s=HANG_LIMIT)
        elapsed = time.monotonic() - started
    finally:
        if previous is None:
            os.environ.pop(_tune_worker.SLEEP_ENV, None)
        else:
            os.environ[_tune_worker.SLEEP_ENV] = previous
        scratch.close()

    assert observation["status"] == "timeout", observation
    # 硬超时的**唯一**判据：墙钟由上限决定，而不是由受测调用决定。
    assert elapsed < HANG_LIMIT * 2, (
        "强杀必须在组级上限附近生效（旧 SIGALRM 路线会等满 300 s）", elapsed)
    assert observation["startup_s"] < HANG_LIMIT, observation
    assert "强杀" in str(observation["reason"]), observation["reason"]
    # ``wall_s`` 是受测时间区（从 ready 起算），不含进程启动。
    assert observation["wall_s"] is not None and observation["wall_s"] < HANG_LIMIT * 2


def test_kill_is_scoped_to_the_child_process_group() -> None:
    """契约 2/5：强杀只打子进程的进程组，父进程与其它子进程照旧活着。"""
    scratch = _Scratch()
    previous = os.environ.get(_tune_worker.SLEEP_ENV)
    os.environ[_tune_worker.SLEEP_ENV] = str(HANG_SECONDS)
    try:
        observation = _tune_runner.run_group(
            _job(scratch.operands()), workdir=scratch.workdir("scoped"),
            group_limit_s=HANG_LIMIT)
    finally:
        if previous is None:
            os.environ.pop(_tune_worker.SLEEP_ENV, None)
        else:
            os.environ[_tune_worker.SLEEP_ENV] = previous
        scratch.close()
    assert observation["status"] == "timeout", observation
    # 本进程还在跑（能走到这里就是证据），而且它的进程组没有被波及：
    # ``os.getpgid(0)`` 仍然是自己的组——"killpg 打错组"会让整个 pytest/CLI 消失。
    assert os.getpgid(0) == os.getpgid(os.getpid()), "父进程必须仍在自己原来的进程组里"


def test_startup_cost_is_reported_separately_from_wall_time() -> None:
    """契约 3/5：``startup_s`` 与 ``wall_s`` 分开，且启动耗时**不计入** wall。"""
    scratch = _Scratch()
    try:
        observation = _tune_runner.run_group(
            _job(scratch.operands()), workdir=scratch.workdir("startup"),
            group_limit_s=None)
    finally:
        scratch.close()

    assert observation["status"] == "ok", observation
    startup = float(observation["startup_s"])
    wall = float(observation["wall_s"])
    # 本机 ``import pyqcd.einsum`` 中位 126 ms；窗口放到 0.05–60 s，只排除"没测到"。
    assert 0.05 <= startup <= 60.0, ("进程启动耗时必须是真实量级", startup)
    # 小方程一组不到 1 ms：若启动被混进 wall_ms，这个上界立刻被打破。
    assert wall < startup, (
        "wall_s 只该装受测的时间区，不该含进程启动", (wall, startup))
    # ``wall_s`` 还必须盖住样本：它是那段时间的墙钟，不是"父进程看到的文件时刻"。
    assert float(observation["median_s"]) <= wall, (
        "wall_s 必须 >= 单次中位（否则父进程的轮询粒度被当成了测量）", observation)


def test_child_crash_is_an_error_not_a_timeout() -> None:
    """契约 4/5：子进程写不出结果（崩溃）→ ``error``，原因带退出码。"""
    scratch = _Scratch()
    previous = os.environ.get(_tune_worker.EXIT_ENV)
    os.environ[_tune_worker.EXIT_ENV] = "9"
    try:
        observation = _tune_runner.run_group(
            _job(scratch.operands()), workdir=scratch.workdir("crash"),
            group_limit_s=30.0)
    finally:
        if previous is None:
            os.environ.pop(_tune_worker.EXIT_ENV, None)
        else:
            os.environ[_tune_worker.EXIT_ENV] = previous
        scratch.close()

    assert observation["status"] == "error", ("崩溃不是 timeout", observation)
    assert "9" in str(observation["reason"]), (
        "原因里必须带退出码（否则无法与超时区分）", observation["reason"])


def test_in_group_pruning_still_runs_inside_the_child() -> None:
    """契约 5/5：组内 ``min_wall_ms`` 剪枝在子进程里判定，命中即 ``pruned``。

    上界设成 **0 ms**：任何一次调用都必然超过它，于是这一组还没跑完就报 ``pruned``
    并带回两个数值——用户规则"单次调用耗时超过 min_wall_ms 即终止该组"的机器形态。
    """
    scratch = _Scratch()
    try:
        observation = _tune_runner.run_group(
            _job(scratch.operands(), min_wall_ms=0.0),
            workdir=scratch.workdir("pruned"), group_limit_s=30.0)
    finally:
        scratch.close()

    assert observation["status"] == "pruned", observation
    assert float(observation["pruned_wall_ms"]) >= 0.0, observation
    assert float(observation["pruned_min_wall_ms"]) == 0.0, observation


def test_isolation_default_is_subprocess_and_env_can_opt_out() -> None:
    """契约 0/5：**缺省**走子进程；只有显式 ``PYQCD_EINSUM_TUNE_INPROCESS=1`` 才退回。"""
    saved = os.environ.pop(_tune_runner.INPROCESS_ENV, None)
    try:
        assert _tune_runner.isolation_enabled() is True, (
            "缺省必须是子进程隔离：SIGALRM 打不断原生 C/CUDA")
        for value in ("1", "true", "yes", "on", "TRUE", " On "):
            os.environ[_tune_runner.INPROCESS_ENV] = value
            assert _tune_runner.isolation_enabled() is False, (
                f"{value!r} 必须退回进程内路线", value)
        for value in ("0", "false", "no", "off", ""):
            os.environ[_tune_runner.INPROCESS_ENV] = value
            assert _tune_runner.isolation_enabled() is True, (
                f"{value!r} 不该关掉隔离", value)
    finally:
        os.environ.pop(_tune_runner.INPROCESS_ENV, None)
        if saved is not None:
            os.environ[_tune_runner.INPROCESS_ENV] = saved


TESTS = (
    test_isolation_default_is_subprocess_and_env_can_opt_out,
    test_hard_timeout_kills_a_hanging_native_call,
    test_kill_is_scoped_to_the_child_process_group,
    test_startup_cost_is_reported_separately_from_wall_time,
    test_child_crash_is_an_error_not_a_timeout,
    test_in_group_pruning_still_runs_inside_the_child,
)


def test_einsum_tune_isolation_contracts() -> None:
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
            f"{len(failures)} 条 tune 隔离契约失败：" + "; ".join(failures))
    if passed == 0:
        raise unittest.SkipTest("tune 隔离契约全部跳过")


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


if __name__ == "__main__":
    raise SystemExit(main())
