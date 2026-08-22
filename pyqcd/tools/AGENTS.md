# AGENTS.md — pyqcd/tools

后端切换（numpy/cupy/torch）、缓存 einsum、切片工具、IO 读取。
torch 适配层含 numpy-like 数学函数全集（cos/sin/arccos/isnan/clip/maximum
（接受标量）/argwhere/identity/append/random 等）。禁止直接 import cupy 计算
（仅 try/except 探测）。
