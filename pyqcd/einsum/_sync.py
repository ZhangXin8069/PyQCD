"""设备同步：让"计时"量到的是**算完**的时间，而不是"下发完"的时间。

为什么必须有这一层
------------------
``cupy`` / ``torch(cuda)`` 上的收缩是**异步下发**：``contract()`` 返回时，GEMM 可能
还在 GPU 上排队，``time.perf_counter()`` 却已经停表。于是同一段代码会出现两种截然不同
的数字（2026-11-19 用户实测）：

* 脚本里自己计时：**~900 ms**（后面用到结果时隐式同步，含真实计算）；
* ``pyqcd.einsum.test`` 的 ``--tune`` 写进配置表：**0.32 ms**（只量到入队）。

``0.32 ms`` 那一行还被 tune 当成 ``best`` 写进了配置表——数字小得离谱，却没有任何
地方会因此报错，这正是"静默失真"最坏的一类。

口径
----
:func:`synchronize` 只做一件事：**等到当前设备上的工作全部完成**。CPU 上一律空操作
（没有设备队列可等，也不该为一个空操作引入依赖 import）。调用方式固定成

.. code-block:: python

    synchronize(*operands)     # 进入计时区前
    start = perf_counter()
    result = contract(...)
    synchronize(*operands)     # 离开计时区后
    seconds = perf_counter() - start

**只包在测量代码里**（``test._timed`` / ``_tune_worker``），不塞进 ``contract()``
本身：那会改变调用方的异步语义——用户拿到数组后可能还要接着下发别的 kernel，
替他们同步等于让库替调用方决定什么时候等。
"""
from __future__ import annotations

from typing import Any, Iterable

__all__ = ["synchronize", "tensor_backend"]


def tensor_backend(value: Any) -> str | None:
    """``cupy`` / ``torch`` / ``numpy`` / ``None``（认不出就不猜）。"""
    module = type(value).__module__.split(".", 1)[0]
    if module in ("cupy", "torch", "numpy"):
        return module
    return None


def synchronize(*operands: Any) -> None:
    """等到这些操作数所在设备上的工作全部完成；CPU / 认不出的一律空操作。

    只按**操作数自己的类型**判断，不看 ``--device``：``contract()`` 可以在
    ``device="cpu"`` 的调用里收到 cupy 数组（后端由操作数决定），拿命令行口径去
    同步会漏掉真正在跑的设备。认不出类型时**不做任何假设**——宁可不对齐，也不去
    import 一个可能昂贵的后端。
    """
    backends = {backend for backend in map(tensor_backend, _flatten(operands))
                if backend in ("cupy", "torch")}
    if not backends:
        return
    if "cupy" in backends:
        try:
            import cupy

            cupy.cuda.get_current_stream().synchronize()
        except Exception:                       # noqa: BLE001 - 同步失败不该改结果
            pass
    if "torch" in backends:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:                       # noqa: BLE001 - 同上
            pass


def _flatten(operands: Iterable[Any]) -> Iterable[Any]:
    """展开一层嵌套（``[a, b]`` 与 ``a, b`` 都接受；更深的不猜）。"""
    for item in operands:
        if isinstance(item, (list, tuple)):
            yield from item
        else:
            yield item
