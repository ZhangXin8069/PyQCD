"""DEPRECATED: 改用 ``python -m pyqcd.einsum.topology_steps_gen --target kupl``。

本模块现在只是**转发层**：语料枚举与路径求解在
:mod:`pyqcd.einsum._topology_gen_common`，输出格式层在
:mod:`pyqcd.einsum.topology_steps_gen`。保留这个名字只为了不打断既有 import
（``pyqcd/testing/_topology_v3_prototype.py``、``_label_sizes_contract.py``、
``_topology_kernel_contract.py``、``testing/einsum/*`` 都从这里取 ``parse`` 等）。

生成物::

    cpp/kupl/einsum/src/topology_kernels.inc       (13.8 MB，本体不入库)
    cpp/kupl/einsum/src/topology_kernels.inc.gz    (1.8 MB，建库前由 make.sh 解压)

``main()`` 仍可用，等价于 ``topology_steps_gen --target kupl``。形状源只有
``roles``（``ROLE_SHAPES``）：``production`` / ``uniform`` 两条历史口径已随用户裁定
删除，本模块也不再转发它们的辅助函数（曾有的 ``PROBE_DIM`` /
``production_sizes`` 别名一并删除）。
"""

from __future__ import annotations

import sys

from ..einsum import topology_steps_gen
from ..einsum._topology_gen_common import (KUPL_OPTIMIZE, MAX_KERNELS,  # noqa: F401
                                           parse)
from ..einsum.topology_steps_gen import (BASE_ENTRIES, MAX_RANK,  # noqa: F401
                                         _octal_literal, _step_count,
                                         canonical_signature, decompose,
                                         encode_steps, fnv1a, plan_steps,
                                         render_kupl as render,
                                         KUPL_TARGET as TARGET)

#: 历史名字：KUPL 侧只有一个优化器口径（``auto``）。
_OPTIMIZE = KUPL_OPTIMIZE


def main() -> int:
    """等价于 ``topology_steps_gen --target kupl``。"""
    return topology_steps_gen.main(["--target", "kupl"])


if __name__ == "__main__":
    sys.exit(main())
