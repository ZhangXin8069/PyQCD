"""Einsums（C++20 张量代数库）字符串 ``einsum`` 的 ctypes 接口与契约测试。

本目录只放**测试侧/接口侧**代码，不改动 ``pyqcd/einsum`` 的任何生产文件：

- :mod:`._contract`：契约数据（文档 URL、语法说明、``DOC_CASES``、
  ``LIMITATIONS``）与 spec 层（Einsums ``"out <- in1 ; in2"`` ↔ numpy
  ``"in1,in2->out"`` 互转、标签/形状一致性校验）；不加载任何原生库。
- :mod:`._bridge`：``cpp/cpu/cpu-einsums/libpyqceinsums.so`` 的冻结 C ABI 的
  ctypes 绑定（``probe`` / ``native_status`` / ``einsum_into``）。导入不触碰
  文件系统，首次调用才 ``CDLL``；库缺失时如实报不可用，绝不伪造原生成功。
- :mod:`verify`：可执行验证入口，分组断言并在末尾打印 ``PASS x/y``。

**目录命名**：用户口语写作 ``cpu-einsums``（与 C++ 目录 ``cpp/cpu/cpu-einsums``
一致），但连字符不是合法的 Python 标识符，含连字符的目录无法作为包 import，
因此 Python 包名固定为 ``cpu_einsums``（下划线），C++ 侧目录名保持连字符。
两者指向同一件事：主机 CPU 上的 Einsums 后端。

验证入口::

    python -m pyqcd.testing.cpu_einsums.verify
    python -m pyqcd.testing.cpu_einsums.verify --verbose --seed 20260831
    python -m pyqcd.testing.cpu_einsums.verify --json

失败退出码 1；依赖原生库的断言在库缺失时整组 SKIP，退出码仍是 0。
"""

__all__ = ['_contract', '_bridge', 'verify']
