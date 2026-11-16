"""qcd-einsum 的缓存原语：有界 LRU + 命中统计 + 总开关。

为什么单独抽一层
----------------
匹配器（:func:`._specialize.match`、:func:`._equivalence.normalize_equation`）与
规划器（``_find_path`` / ``_measure_steps``）问的都是"同一结构反复问同一个问题"，
答案只随结构变化。此前这些位置有的用 ``functools.lru_cache``、有的用裸 dict、有的
什么都不缓存，**既没有统一开关，也没有命中统计**——于是"缓存到底有没有生效"只能靠
读代码猜。

本模块只提供原语，不替调用方决定缓存键：

* :class:`LRUCache` —— 线程安全、有界、带 ``hits/misses/stores/evictions`` 的 LRU；
* :func:`enabled` —— 三态解析：显式参数 > ``PYQCD_EINSUM_CACHE`` > 缺省开启；
* :func:`stats` / :func:`clear` —— 汇总/清空全部已注册缓存。

关闭时的语义
------------
``enabled()`` 为假时调用方**既不读也不写**缓存（``get`` 返回缺省、``put`` 直接丢弃），
因此关闭不会留下"半热"状态，也不会让统计里出现假命中。

统计口径
--------
``hits`` = 命中次数；``misses`` = 未命中次数；``stores`` = 写入次数；
``evictions`` = 因超界被淘汰的次数；``size`` / ``maxsize`` = 当前/上限条数。
``hit_rate`` 只在有查询时给出（``hits / (hits + misses)``），否则为 ``None``。
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Any, Iterable

__all__ = [
    "CACHE_ENV",
    "LRUCache",
    "clear",
    "enabled",
    "registered",
    "stats",
]

#: 缓存总开关：``0/false/off/no`` 关闭；缺省 ``"1"``（开启）。
CACHE_ENV = "PYQCD_EINSUM_CACHE"

_MISS = object()


def enabled(value: bool | None = None) -> bool:
    """缓存是否启用：显式 ``True``/``False`` 优先，``None`` 读环境变量（缺省开启）。

    在**调用时**求值，与 ``PYQCD_EINSUM_EQUIVALENCE`` / ``_AS_EQUIVALENCE`` 一致，
    因此同进程内改环境变量即时生效。
    """
    if value is not None:
        return bool(value)
    return os.environ.get(CACHE_ENV, "1").strip().lower() not in (
        "0", "false", "off", "no")


class LRUCache:
    """线程安全的有界 LRU，带命中统计。

    ``maxsize`` 按条数计（不是字节）；``_REGISTRY`` 里注册后即可被
    :func:`stats` / :func:`clear` 统一访问。
    """

    __slots__ = ("name", "maxsize", "_data", "_lock", "_hits", "_misses",
                 "_stores", "_evictions")

    def __init__(self, name: str, maxsize: int = 4096) -> None:
        if maxsize < 1:
            raise ValueError(f"maxsize 必须为正：{maxsize!r}")
        self.name = str(name)
        self.maxsize = int(maxsize)
        self._data: OrderedDict[Any, Any] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = self._misses = self._stores = self._evictions = 0
        _REGISTRY[self.name] = self

    # -- 读写 ---------------------------------------------------------------
    def get(self, key: Any, default: Any = None,
            *, use: bool | None = None) -> Any:
        """取值；未命中返回 ``default``（关闭时一律返回 ``default``）。"""
        if not enabled(use):
            return default
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)          # 真 LRU：命中即刷新
                self._hits += 1
                return self._data[key]
            self._misses += 1
            return default

    def put(self, key: Any, value: Any, *, use: bool | None = None) -> None:
        """写入；关闭时直接丢弃（不写、不计数）。"""
        if not enabled(use):
            return
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            self._stores += 1
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)
                self._evictions += 1

    def pop(self, key: Any, default: Any = None, *,
            use: bool | None = None) -> Any:
        if not enabled(use):
            return default
        with self._lock:
            return self._data.pop(key, default)

    # -- 观测 ---------------------------------------------------------------
    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._hits = self._misses = self._stores = self._evictions = 0

    def info(self) -> dict[str, Any]:
        with self._lock:
            queries = self._hits + self._misses
            return {
                "size": len(self._data),
                "maxsize": self.maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "stores": self._stores,
                "evictions": self._evictions,
                "hit_rate": (self._hits / queries) if queries else None,
            }

    def __repr__(self) -> str:                       # pragma: no cover - 调试用
        return f"LRUCache({self.name!r}, maxsize={self.maxsize})"


#: 已注册缓存：``name -> LRUCache``。用于 :func:`stats` / :func:`clear`。
_REGISTRY: dict[str, LRUCache] = {}


def registered() -> Iterable[str]:
    """已注册缓存的名字（排序后返回，便于稳定输出）。"""
    return tuple(sorted(_REGISTRY))


def stats() -> dict[str, dict[str, Any]]:
    """所有已注册缓存的统计快照；``enabled`` 字段反映当前总开关。"""
    return {
        "enabled": enabled(),
        "caches": {name: cache.info()
                   for name, cache in sorted(_REGISTRY.items())},
    }


def clear() -> None:
    """清空所有已注册缓存并归零统计。"""
    for cache in list(_REGISTRY.values()):
        cache.clear()
