# 开发功能 → PyPI 包的连线状态（pyqcd 0.1.17，2026-11-08）

> 判定口径：**从 PyPI 远程安装**的 `pyqcd==0.1.17`（全新 venv），看代码与运行期证据；
> 源码树里的行为不算数。

## 1. 包体构成

| 项 | 实测 |
|---|---|
| 版本 | `pyqcd 0.1.17`（仅 sdist 发布，安装时现场编译） |
| 顶层子包 | 12 个：`analysis` `contraction` `einsum` `gauge` `lattice` `operator` `parallel` `pipeline` `renorm` `smear` `tools` `vertex`（另有 `__main__`） |
| 随包 `.py` | 103 个 |
| 随包原生库 | `libpyqcdeinsum_kupl.so` 0.36 MB、`libpyqcdeinsum_cuda.so` 3.4 MB、`einsum.cpython-312-*.so` 0.19 MB（Cython 桥） |
| 不随包 | `pyqcd.testing/`（测试、基准、回归基线只留在源码仓库） |

`python -m pyqcd` 在纯安装环境的输出：

```
pyqcd 0.1.17  (Python 3.12.4)
  native backend : cuda
  mpi            : False (non-MPI build (CUDA backend has no MPI sharding))
  library        : .../site-packages/pyqcd/einsum/cuda/libpyqcdeinsum_cuda.so
  subpackages    : analysis, contraction, einsum, gauge, lattice, operator, parallel,
                   pipeline, renorm, smear, tools, vertex
  threads        : omp=4 blas=2 [auto kernel (4 physical cores, 1 rank);
                   runtime: blas:scipy_openblas_set_num_threads64_]
  optional not installed: torch (pyqcd[torch]), matplotlib (pyqcd[plot]),
                   cupy (pyqcd[cuda]), mpi4py (pyqcd[mpi])
```

## 2. 原生后端连线

| 层 | 状态 | 证据 |
|---|---|---|
| 后端选择 | 安装时 `auto`：有 `nvcc`+Cython+源码 → CUDA，否则 KUPL，再否则纯 Python | 本次安装 `backend=cuda` |
| KUPL 库 | 已连线 | 导出 `qcd_kupl_nucleon_2pt`、`qcd_kupl_pattern_3pt`、`qcd_kupl_pattern_3pt_supported`、`qcd_kupl_mpi_enabled` |
| CUDA 库 | 已连线 | 导出 `qcd_cuda_nucleon_2pt`、`qcd_cuda_pattern_3pt`、`qcd_cuda_pattern_3pt_free` |
| Python 包装 | 已连线 | `_kernel_kupl.try_nucleon_2pt/try_three_pt`、`_kernel_cuda.try_nucleon_2pt/try_three_pt`，`enabled()` 均为 `True` |
| 线程档位 | 已连线（0.1.17 新增） | 运行时 `runtime: blas:scipy_openblas_set_num_threads64_`，说明对 wheel 里的 OpenBLAS 生效 |

## 3. 方程形态特化：9 条注册，3 条有真 kernel

| 形态 | 类型 | 发行版实际执行 |
|---|---|---|
| `nucleon-2pt(E1)` | kernel | **专用 kernel**（KUPL/CUDA） |
| `sush-3pt-main(P1)` | kernel | **专用 kernel**（KUPL/CUDA） |
| `sush-3pt-main(P2)` | kernel | **专用 kernel**（KUPL/CUDA） |
| `nucleon-3pt(E2)` | plan-cache | 仅计划缓存 |
| `gauge-propagator(E4)` | plan-cache | 仅计划缓存（**开发版曾有融合 kernel，未进发行版**） |
| `gamma-sandwich(E5)` | plan-cache | 仅计划缓存（同上） |
| `sush-2pt(S1)` / `sush-3pt-munu(S2)` / `sush-2pt-mu(S3)` | plan-cache | 仅计划缓存 |

运行期提示据实反映执行路径（`PYQCD_EINSUM_SPECIALIZE_VERBOSE=1` 每次打印）：

```
E1 nucleon 2pt   -> [pyqcd.einsum] 专用路径: nucleon-2pt(E1) [kernel: kupl_nucleon_2pt] ...
E5 sandwich      -> [pyqcd.einsum] 专用路径: gamma-sandwich(E5) [plan-cache] ...
generic matmul   -> (无提示，走通用路径)
```

判据：`[kernel: <名字>]` 才是真跑了专用内核；`[plan-cache]` 只复用了规划结果。

## 4. 线程档位连线（0.1.17）

两档自动切换，且**作用到已加载的运行时**而不只是导出变量：

| 档位 | OpenMP | BLAS | 触发 |
|---|---|---|---|
| `kernel` | 物理核（≤8） | 核数/2（≤4） | 命中上表三条 kernel 形态 |
| `generic` | 核数/2（≤8） | 核数（2–4） | 其余收缩 |

实测（4 物理核笔记本，KUPL）：

| 配置 | 通用链 800³ | 核子 2pt nvec=32 |
|---|---|---|
| `PYQCD_EINSUM_THREADS=off` | 103.8 ms | 25.7 ms |
| **默认（按调用切档）** | **15.0 ms** | **21.1 ms** |
| 钉死 `OMP=1/BLAS=4` | 15.9 ms | 57.5 ms |
| 钉死 `OMP=4/BLAS=8` | 55.7 ms | 21.6 ms |

## 5. 各子包的公开 API 规模

| 子包 | 公开名数量 | 子包 | 公开名数量 |
|---|---|---|---|
| `lattice` | 14 | `renorm` | 91 |
| `gauge` | 13 | `tools` | 26 |
| `smear` | 3 | `analysis` | 113 |
| `operator` | 26 | `pipeline` | 42 |
| `vertex` | 18 | `parallel` | 9 |
| `contraction` | 15 | `einsum` | 31 |

## 6. 未连线 / 未随包（明确清单）

| 项 | 状态 | 说明 |
|---|---|---|
| `pyqcd.testing`（测试、基准、回归基线） | **不随包** | 文档里相关命令均标注"需源码仓库" |
| MPI 分片 | **默认关闭** | `mpi_label: non-MPI build`；需 `CXX=mpicxx PYQCD_EINSUM_ENABLE_MPI=1` 重建，且只分片单输出轴；CUDA 后端无 MPI |
| E4 / E5 融合 kernel | **未进发行版** | 开发期实现过，发行包只有计划缓存；文档已据实标注 |
| `torch` / `matplotlib` / `cupy-cuda12x` / `mpi4py` | **可选 extras** | `pyqcd[torch]`、`pyqcd[plot]`、`pyqcd[cuda]`、`pyqcd[mpi]`；未装时给出可行动的 ImportError |
| CUDA 通用执行器（cupy 输入） | **部分可用** | 特化形态走 kernel；通用执行器可能 `QCD_STATUS_INTERNAL` 回落 Python（单次警告，结果正确） |

## 7. 复现命令

```bash
pip install --index-url https://pypi.org/simple "pyqcd==0.1.17" opt_einsum
python -m pyqcd                              # 后端 / MPI / 线程 / 子包
python -c "import pyqcd.einsum as pe; print(pe.build_info()); print(pe.specializations())"
PYQCD_EINSUM_SPECIALIZE_VERBOSE=1 python -c "
import numpy as np, pyqcd.einsum as pe
r = np.random.default_rng(0)
c = lambda *s: (r.standard_normal(s) + 1j*r.standard_normal(s)).astype(np.complex128)
n = 16
pe.contract('gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M',
            c(4,4,n,n), c(4,4,n,n), c(4,4,n,n), c(4,4), c(4,4),
            c(4,n,n,n), c(4,n,n,n), c(4,4))
"      # 期望看到 [kernel: kupl_nucleon_2pt]（有 CUDA/cupy 时为 cuda_nucleon_2pt）
```
