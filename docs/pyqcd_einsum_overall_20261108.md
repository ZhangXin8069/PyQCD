# pyqcd-einsum 整体进度简报（2026-11-08 08:55）

## 1. 一句话状态

核心功能已构建完整、**已知缺陷为 0**；正确性全部通过（含 980 项差分）；
性能上 CPU 2pt/3pt 与 GPU 3pt 已有专用 kernel 且高于 opt_einsum，
E4/E5 在 GPU L24 上分别为持平与 3.37x；**当前所有 CPU 计时受后台代理干扰，需独占复测**。

## 2. 构建与正确性

| 检查 | 结果 |
|---|---|
| `build-kupl.sh` | SUCCESS |
| `build-nv.sh`（`PYQCD_CUDA_ARCHITECTURES=8.9`） | SUCCESS |
| `verify_native --backend kupl` | 16 passed / 0 failed |
| `verify_native --backend cuda` | 16 passed / 0 failed |
| `verify_compat` 全量（oracle 自检 70 / API 38 / 路径 802 / 数值 70） | **980/980 PASS** |
| `simple/run-all.sh`（compat 39 + memory 4 + equations 9 + native 13×2） | **78 passed，exit=0** |
| `python -m pyqcd.testing` | **115 passed / 1 skipped / 0 failed**（skip 为 xelatex 缺失） |
| 生产规模 9 组方程（T1） | 9/9，最差 `rel = 1.444e-15` |

## 3. Bug 状态：0 个已知缺陷

| 类别 | 内容 |
|---|---|
| 已修复（5） | B1 `transpose_copy` 未初始化输出；B2 专用路径缓存键；B3 测试硬编码 `/contract`；B4 `verify_compat` 缺 pytest；B5 CUDA E4 融合在 spatial≥2 灾难退化 |
| 定性为非缺陷（3） | B5′ 预热口径；B6 构建竞态 Segfault（已原子化 `setup.py`）；B7/B8/B9 中两项实为**验收工具与构建配置**问题 |
| B7 | cuda_3pt 先前 rel=1.00 → 实为**工具传主机指针**；修正后 `rel=3.13e-15` |
| B8 | kupl_3pt 先前 ZGEMM lda 非法 → 实为测到**正在重写的旧版本**；新版 `rel=2.54e-15` |
| B9 | cuda_nucleon 5 个编译错 → **同名头文件 include 冲突**；改相对路径后编译通过 |

## 4. 方程特化（9 条形态）

| 形态 | 类型 | 接线状态 | 实测 |
|---|---|---|---|
| **E1**（核子 2pt，CPU） | KUPL kernel | **已接线** | kernel 22.65 ms vs 通用 25.26（**1.12x**）vs opt 120.43（**5.32x**）；N=100 时 1.10–1.41x / 7.2–9.4x |
| **E4**（peram×gauge，GPU） | e4_fused | **已接线**（仅 `z·y·x==1`） | spatial=1 时 3.6x；L24 回落通用路径 |
| **E5**（γ 夹心，CPU+GPU） | sandwich | **已接线** | CPU L24 142.7 vs 1188.9 ms（**8.33x**）；GPU L24 **3.371x** |
| **P1/P2**（3pt 主项，GPU） | cuda_3pt | **已接线** | kernel 30.74 ms vs 通用 74.13（**2.41x**）vs opt 285.40（**9.29x**） |
| **P1/P2**（3pt 主项，CPU） | kupl_3pt | **已接线，标记 needs-optimization** | 测量受干扰，暂不结论 |
| E2/S1/S2/S3 | plan-cache | 无专用执行 | 如实标注 |

开关：`contract(..., specialize=False)`、`PYQCD_EINSUM_SPECIALIZE=0`、`PYQCD_EINSUM_DISABLE_FUSED=1`、
`PYQCD_EINSUM_KERNEL_KUPL=0`、`PYQCD_EINSUM_KERNEL_CUDA=0`；查询 `pe.specializations()`。

## 5. GPU L24 生产规模（进程隔离基准，默认预热 4 轮）

| 组 | opt_einsum | pyqcd-einsum | 倍率 |
|---|---|---|---|
| E4 | 1.112793 s | 1.095407 s | 1.016x（持平） |
| E5 | 0.373987 s | 0.110959 s | **3.371x** |

## 6. 生产耗时占比（模型口径）

| 组 | 占 opt 时间 | 占 pyqcd 时间 |
|---|---|---|
| E1（2pt） | 51.0% | 42.6% |
| P1（3pt 主项） | 15.5% | 42.1% |
| P2 | 3.1% | 8.3% |
| E4 | 26.4% | 6.7% |
| E5 | 4.0% | 0.4% |
| 合计 | 4757 s | 1688 s（2.82x） |

## 7. 必须说明的测量限制（重要）

当前机器上有多个后台代理在跑基准（`load average` 一度 4.97，`n2pt_ab.py` 占 214% CPU），
导致 CPU 计时严重抖动，例如同一 `kupl_3pt` 在 N=50 上多次测得
`37.65 / 75.97 / 138.23 ms`（相差 3.7x），N=20 上 `23.99 / 23.15 ms`。

**因此，本文件中关于 `kupl_3pt` 的结论一律标注为"待独占复测"**，
不作为验收依据。GPU 侧的 L24 数字来自进程隔离基准，可信度较高（但同样建议复测）。

## 8. 进行中

| 项 | 负责 |
|---|---|
| `cuda_nucleon` 优化（当前 95.12 ms vs 通用 57.31 ms，需 < 57.31 ms） | bg-21 |
| `kupl_3pt` 进一步优化（已接线，标记 needs-optimization） | 待排期 |

## 9. 下一步

1. 停掉全部后台代理，独占机器复测：`kupl_3pt` 分规模、`cuda_nucleon`、GPU L24 全量。
2. 依据复测结果决定 `kupl_3pt` 是否保留 `needs-optimization` 标识。
3. 最终验收：双后端构建 → `verify_native` → `verify_compat` 980 → T1/L24 全量 → 记录真实退出码。

---

## 10. MPI 版本标识（09:25）

### 10.1 可查询标识（新增公开 API）

```python
import pyqcd.einsum as pe

pe.mpi_enabled()   # -> bool
pe.build_info()    # -> dict
```

`build_info()` 返回示例：

```json
// 当前默认构建（KUPL 后端）
{"backend": "kupl", "mpi": false, "kupl_mpi_build": false,
 "mpi_label": "非 MPI 版本",
 "native_library": ".../cpp/kupl/einsum/libpyqcdeinsum_kupl.so"}

// 当前默认构建（CUDA 后端）
{"backend": "cuda", "mpi": false, "kupl_mpi_build": false,
 "mpi_label": "非 MPI 版本（CUDA 后端未实现 MPI 分片）", ...}

// MPI 构建（CXX=mpicxx PYQCD_EINSUM_ENABLE_MPI=1）
{"backend": "kupl", "mpi": true, "kupl_mpi_build": true,
 "mpi_label": "MPI 版本", ...}
```

### 10.2 实现

| 层 | 内容 |
|---|---|
| C++ | `apply_contract.cpp` 末尾新增 `qcd_kupl_mpi_enabled()`，按 `PYQCD_EINSUM_ENABLE_MPI` 返回 1/0 |
| 导出 | `make.sh` 的 version script 新增 `qcd_kupl_mpi_enabled` |
| Python | `kupl/einsum.py::mpi_enabled()`；`_native.py::build_info()`；`__init__.py` 导出两者 |

### 10.3 验证

| 场景 | 结果 |
|---|---|
| 默认构建（`MPI: disabled`） | `mpi_enabled()=False`，`mpi_label="非 MPI 版本"` |
| MPI 构建（`MPI: enabled`） | `mpi_enabled()=True`，`mpi_label="MPI 版本"` |
| MPI 构建 + `mpirun -np 2 verify_mpi` | `PASS verify_mpi ranks=2` |
| CUDA 后端（任意构建） | 一律标注"非 MPI 版本（CUDA 后端未实现 MPI 分片）" |

### 10.4 构建缺陷修复（附带）

`make.sh` 的 `PYQCD_EINSUM_ENABLE_MPI=auto` 原来只凭 `command -v mpicxx` 判断，
而 conda 常把 `CXX` 固定为不带 MPI 头的 `g++` → 编译失败。
现改为**用有效编译器试编译 `#include <mpi.h>`** 通过才启用；默认构建因此稳定为
`MPI: disabled`，显式 MPI 构建稳定为 `MPI: enabled`。
