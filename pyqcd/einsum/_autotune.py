"""首次调用自动 tune：配置表**没有**这次调用的行时，先测一遍再真跑。

为什么要有这一层
----------------
``_profile_cache`` 的设计是"表只有 tune 能写、普通调用只读"。于是第一次在某台
机器 / 某个形状上调用 ``contract()`` 时表里必然没有对应行，用户看到一行警告
"需要先 tune" 然后按通用路径跑——**这一次**的耗时既不是最优、也没有被记录，
下次再调用还是同一句话。用户裁定（2026-11-19）：**默认第一次调用就是"先 tune
再真实运行"**，不要让人手动补这一步。

边界（按顺序判定，任一不满足就退回"警告 + 按未命中执行"）
----------------------------------------------------------
* ``PYQCD_EINSUM_AUTOTUNE=0/false/off/no`` 关闭（解释器启动时读一次）；
* 配置表总开关关闭（``profile=False`` / ``PYQCD_EINSUM_PROFILE=0``）→ 不 tune：
  用户明确说过"别读也别写表"，替它写一张表正好相反；
* **同一个键在一个进程里只自动 tune 一次**：命中失败（跑不出可用组合）也记账，
  否则一次失败的 tune 会在每次调用上重演；
* tune 自己调用 ``contract()`` 时不会回来（``_profile_begin`` 在
  ``_tune_cache_session()`` 期间读当无命中，且子进程路线根本不读表）——这里再加
  一道保护，防止将来某条路径漏掉那两层。保护分两半：**线程本地**标记挡住
  同一线程的重入（递归），**按键集合**挡住同一键的并发重复；两者都不用全局
  布尔——那会让"另一个线程正在为**别的**键 tune"误伤成"我被跳过"。

代价与提示
----------
自动 tune 的墙钟是"整轮遍历"（CPU 上几十组、每组 ``1+warmup+rounds`` 次调用），
生产形状上可能是分钟量级。这**是用户明确要的默认行为**，所以只在 stderr 打一行
说明（含关闭方法），不弹确认、不偷偷跳过——偷偷跳过会让"默认先 tune"变成空话。
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Any, Mapping, Sequence

__all__ = ["AUTOTUNE_ENV", "enabled", "run_for", "reset"]

#: 自动 tune 的开关（缺省**开启**；``0/false/off/no`` 关闭）。
AUTOTUNE_ENV = "PYQCD_EINSUM_AUTOTUNE"

_OFF = frozenset({"0", "false", "off", "no"})

#: ``--dtype`` 认识的四个名字（与 ``test.DTYPES`` 同集合）。键里的 dtype 来自后端
#: 原样的 ``str(dtype)``（``torch.float64`` / ``cupy.complex64``），必须先归一；
#: 归一不出来就**不**自动 tune（传下去必然在 ``np.dtype`` 上抛）。
_CLI_DTYPES = frozenset({"complex128", "complex64", "float64", "float32"})

#: 本进程里已经尝试过自动 tune 的键（成功与失败都记：失败不该每次重演）。
_ATTEMPTED: set[tuple[Any, ...]] = set()
#: 正在自动 tune 的键。**按键记而不是一个全局布尔**：两个线程同时首次调用
#: **不同**的键时，全局布尔会让第二个静默跳过（它并没有任何理由被跳过）。
_RUNNING: set[tuple[Any, ...]] = set()
#: 线程本地的递归保护：``run_tune`` 内部会再调 ``contract()``，那条路必须立刻退出。
#: 用 thread-local 而不是全局标志，理由同上——递归是"同一线程重入"，不是"全局锁"。
_LOCAL = threading.local()
_LOCK = threading.Lock()


def enabled(value: Any = None) -> bool:
    """自动 tune 是否生效（显式 ``value`` 压过环境变量；缺省读环境变量）。

    **字符串按同一个词表解析**，不做布尔转换：``bool("0")`` 是 ``True``，直接把
    ``"0"`` / ``"false"`` 交给布尔判定会把"关掉自动 tune"变成"强制打开"——本仓库在
    ``as_equivalence`` 上已经踩过一次同一个坑（那里的注释写着"反转开关陷阱"）。
    显式 ``value`` 与环境变量因此走**同一条** ``_OFF`` 判定。
    """
    if value is None:
        value = os.environ.get(AUTOTUNE_ENV, "1")
    if isinstance(value, str):
        return value.strip().lower() not in _OFF
    return bool(value)


def reset() -> None:
    """清空"已尝试"记录（只给契约用；生产路径不需要）。"""
    with _LOCK:
        _ATTEMPTED.clear()
        _RUNNING.clear()
    _LOCAL.active = False


def _key_token(key: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(str(key.get(field, "")) for field in
                 ("shape", "dtype", "device", "equation", "backend"))


def run_for(*, key: Mapping[str, Any], subscripts: str,
            operands: Sequence[Any], backend: str | None = None,
            device: str | None = None, dtype: str | None = None,
            language: str | None = None) -> bool:
    """为这个键跑一整轮 ``--tune``；返回"是否可能已经写出可用行"。

    返回 ``True`` 表示 tune 跑完了、调用方应**重读一次表**（tune 自己负责写胜出行；
    本函数不写表、不改调用方的任何状态）。任何异常都收成 ``False``——自动 tune 失败
    只该退回"按未命中执行"，绝不能让一次 ``contract()`` 因为优化步骤而失败。
    """
    if not enabled():
        return False
    token = _key_token(key)
    # 递归保护先看 thread-local：``run_tune`` 内部会再调 ``contract()``，那条路必须
    # 立刻退出。放在锁外是必须的——同一线程重入时不能再抢一次同一把锁。
    if getattr(_LOCAL, "active", False):
        return False
    with _LOCK:
        if token in _RUNNING or token in _ATTEMPTED:
            return False
        _RUNNING.add(token)
        _ATTEMPTED.add(token)
    _LOCAL.active = True
    try:
        return _run_locked(key=key, subscripts=subscripts, operands=operands,
                           backend=backend, device=device, dtype=dtype,
                           language=language)
    finally:
        _LOCAL.active = False
        with _LOCK:
            _RUNNING.discard(token)


def _cli_dtype(dtype: str | None) -> str | None:
    """把键里的 dtype 文本转成 ``--dtype`` 接受的四个名字之一；转不了返回 ``None``。

    键里的 dtype 来自 ``str(operand.dtype)``（见 ``_api._profile_dtype``），它按后端
    不同而不同：numpy 给 ``"float64"``、**torch 给 ``"torch.float64"``**。直接把后者
    交给 ``run_tune`` 会在 ``np.dtype(...)`` 上炸出
    ``TypeError: data type 'torch.float64' not understood``——实测就是这么暴露的
    （``release_check`` 的 docs examples 里有 torch 例子，自动 tune 因此静默失败）。

    于是这里显式去掉后端前缀，并且**只放行 CLI 认识的四个名字**：转不出来就放弃
    自动 tune（返回 ``None``），而不是把一个必然失败的 dtype 传下去。
    """
    text = str(dtype or "").strip().lower()
    if not text:
        return None
    if "." in text:
        text = text.rsplit(".", 1)[-1]           # ``torch.float64`` → ``float64``
    if "+" in text or text not in _CLI_DTYPES:
        return None
    return text


def _cli_device(device: str | None, backend: str | None) -> str:
    """把键里的 device 文本转成 ``--device`` 接受的三档之一（``cpu`` / ``gpu`` / ``cuda``）。

    键里的 device 是 ``_api._profile_devices`` 给的（``cpu`` / ``cupy:0`` /
    ``cuda:0`` / ``torch:0`` 之类）。``run_tune`` 只认 ``cpu``/``gpu``/``cuda``，
    传 ``cupy:0`` 会在参数校验上失败。判据：含 ``:`` 或后端是 cupy/torch 且不是
    ``cpu`` 的一律算 ``gpu``。
    """
    text = str(device or "").strip().lower()
    if text in ("cpu", "gpu", "cuda"):
        return text
    if ":" in text or text in ("cupy", "torch"):
        return "gpu"
    backend_text = str(backend or "").strip().lower()
    return "gpu" if backend_text in ("cupy", "torch") else "cpu"


def _run_locked(*, key: Mapping[str, Any], subscripts: str,
                operands: Sequence[Any], backend: str | None,
                device: str | None, dtype: str | None,
                language: str | None) -> bool:
    try:
        from . import test as cli
    except Exception:                            # noqa: BLE001 - 没有 CLI 就不自动 tune
        return False
    runner = getattr(cli, "run_tune", None)
    if not callable(runner):
        return False
    shapes = []
    for operand in operands:
        shape = getattr(operand, "shape", None)
        if shape is None:
            return False
        shapes.append(tuple(int(extent) for extent in shape))
    backend_text = str(backend or key.get("backend") or "auto")
    # dtype / device 必须转成 CLI 认得的拼写：键里存的是**后端原样**的文本
    # （``torch.float64`` / ``cupy:0``），直接传会在 ``np.dtype`` 或参数校验上失败。
    dtype_text = _cli_dtype(dtype or str(key.get("dtype") or ""))
    if dtype_text is None:
        print(f"[pyqcd.einsum] 配置表没有这次调用的行，但 dtype="
              f"{dtype or key.get('dtype')!r} 不在 --tune 支持的四个 dtype 里"
              f"（{'/'.join(sorted(_CLI_DTYPES))}），跳过自动 tune，本次按未命中执行。",
              file=sys.stderr, flush=True)
        return False
    device_text = _cli_device(device or str(key.get("device") or ""), backend_text)
    print(f"[pyqcd.einsum] 配置表没有这次调用的行 → 先跑一轮 --tune 再执行"
          f"（equation={str(key.get('equation') or '')} shape={key.get('shape', '')} "
          f"dtype={dtype_text} device={device_text}）；"
          f"设 {AUTOTUNE_ENV}=0 可关闭自动 tune。",
          file=sys.stderr, flush=True)
    try:
        runner(equation=subscripts, shapes=shapes, device=device_text,
               dtype=dtype_text, backend=backend_text,
               language=language)
    except Exception as exc:                     # noqa: BLE001 - 优化失败不该让调用失败
        print(f"[pyqcd.einsum] 自动 tune 未完成（{type(exc).__name__}: {exc}），"
              f"本次按未命中执行。", file=sys.stderr, flush=True)
        return False
    return True
