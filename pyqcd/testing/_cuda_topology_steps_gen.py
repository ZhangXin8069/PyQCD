"""DEPRECATED: 改用 ``python -m pyqcd.einsum.topology_steps_gen --target cuda``。

本模块现在只是**转发层**：语料枚举与路径求解在
:mod:`pyqcd.einsum._topology_gen_common`，输出格式层在
:mod:`pyqcd.einsum.topology_steps_gen`。

生成物::

    cpp/cuda/einsum/src/topology_steps.h       （活表位置对 + (key, fnv) 二分表）
    cpp/cuda/einsum/src/topology_steps.h.gz

签名与绑定（必须与 ``cpp/cuda/einsum/src/apply_contract.cu`` 逐位相同）：

* ``key`` = 规范签名 ``|`` 连接的 payload 的 blake2b-**128** 摘要**前 8 字节**
  （大端）；payload 与 ``_specialize._signature_digest`` 完全一致，只是摘要位数是
  16 而不是索引侧的 12；
* ``fnv`` = 逐操作数写「rank 字节 + 各标签字节」，再写 ``0xFF``，最后写输出标签，
  64 位 FNV-1a。运行期 ``topology_structure_fingerprint`` 用描述符里的整数标签重算，
  失配即 fail closed。

口径：收缩序取生成器的形状源 ``roles``（``ROLE_SHAPES`` 上的 ``auto`` 规划）。
历史表的"均匀 dim=4 探针"口径已随用户裁定删除，本模块不再转发 ``UNIFORM_DIM``。
运行期由 ``topology_order_acceptable`` 用真实尺寸兜底。

``main(argv)`` 仍可用，等价于 ``topology_steps_gen --target cuda``＋原参数。
"""

from __future__ import annotations

import pathlib
import sys
from typing import Sequence

from ..einsum import topology_steps_gen
from ..einsum._topology_gen_common import (CUDA_OPTIMIZE, MAX_KERNELS,  # noqa: F401
                                           parse)
from ..einsum.topology_steps_gen import (BASE_ENTRIES,  # noqa: F401
                                         collect_cuda as collect,
                                         fnv_fingerprint, layout, parse_header,
                                         render_cuda as render,
                                         step_pairs, table_key,
                                         verify_cuda as verify)

ROOT = pathlib.Path(__file__).resolve().parents[2]
TARGET = ROOT / "cpp/cuda/einsum/src/topology_steps.h"

#: 历史名字：CUDA 表口径的优化器（``greedy``）。
OPTIMIZE = CUDA_OPTIMIZE


def main(argv: Sequence[str] | None = None) -> int:
    """等价于 ``topology_steps_gen --target cuda``＋原参数。"""
    rest = list(sys.argv[1:] if argv is None else argv)
    return topology_steps_gen.main(["--target", "cuda", *rest])


if __name__ == "__main__":
    sys.exit(main())
