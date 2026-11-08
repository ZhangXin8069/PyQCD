# KUPL 版 qcd-einsum 的加速底层剖析（2026-11-08）

> 问题：`pyqcd.einsum` 的 **KUPL（CPU）后端**到底靠什么加速——多进程？MPI？还是别的？
> 结论先给：**OpenMP 线程 + 动态加载的 BLAS（主力）+ 可选 MPI（默认关闭）**，
> **没有 multiprocessing**。以下每条都有代码位置与运行期实测证据。

## 1. 结论总览

| 层 | 机制 | 是否默认启用 | 证据 |
|---|---|---|---|
| BLAS（主力） | 运行期 `dlopen` 加载 `libcblas` / `libopenblas`，`dlsym` 取 `zgemm`/`cgemm`/`dgemm`（含 ILP64 变体 `*_64`） | ✅ 有库即用 | `cpp/kupl/einsum/src/apply_contract.cpp:298-303, 319, 347, 349, 365-366` |
| OpenMP | `#pragma omp parallel for` + `#pragma omp simd`，编译期 `-fopenmp` | ✅ | 同目录 `make.sh:42` 的 `DEFAULT_FLAGS`；pragma 计数见 §2 |
| MPI | `applyContractMpiQcd`，**只分片单个输出轴**，`MPI_Allreduce` 归约 | ❌ 需 `CXX=mpicxx PYQCD_EINSUM_ENABLE_MPI=1` 重建 | `include/einsum_mpi.h`、`src/mpi_runtime.cpp:216/223/265/271/301` |
| 多进程 | 无 | — | 并行全在库内，Python 侧只做一次 ctypes 调用 |

CUDA 后端不同：走 cuBLAS/自写 CUDA kernel，**MPI 完全没有实现**
（`build_info()['mpi_label']` 恒为 "non-MPI build (CUDA backend has no MPI sharding)"）。

## 2. OpenMP 的分布

| 文件 | `#pragma omp` 数量 | 用途 |
|---|---|---|
| `src/apply_contract.cpp` | 7 | 打包/拷贝/置换循环并行 + `omp simd` 向量化 |
| `src/nucleon2pt.cpp` | 13 | 核子 2pt 专用 kernel 的循环并行 |
| `src/pattern_3pt.cpp` | 18 | 3pt 主项专用 kernel 的循环并行 |

编译标志（`cpp/kupl/einsum/make.sh`）：

```
DEFAULT_FLAGS="-O3 -fopenmp -shared -fPIC -std=c++17 -fvisibility=hidden -Wall -Wextra"
# 另加（非跨平台场景）：-march=native -funroll-loops
```

线程数不是无脑开满，而是按工作量收敛（`apply_contract.cpp:663-672`）：

```cpp
const int max_threads = omp_get_max_threads();
int threads = 1;
if (max_threads > 1 && destination_count >= 32768) {
    threads = min(max_threads, max(1, destination_count / 65536));
}
```

即：目标元素数 < 32768 时**单线程**；之后按每 65536 元素一个线程增长，封顶 `omp_get_max_threads()`。

## 3. BLAS 的加载方式

- 优先读环境变量 **`PYQCD_EINSUM_CBLAS_LIB`**（`apply_contract.cpp:347`）；
- 否则按序 `dlopen`：`libcblas.so.3` → `libcblas.so` → `libopenblas.so.0` →
  `libopenblas.so` → `libopenblas.dylib` → `libcblas.dylib`（`:365-366`）；
- 取符号用 `dlsym`（`:319`），因此**不产生编译期依赖**，换 BLAS 不需要重编。

运行期实测（`/proc/self/maps`，本机 conda 环境）：

```
/home/zhangxin/miniforge3/lib/libgomp.so.1.0.0
/home/zhangxin/miniforge3/lib/libopenblasp-r0.3.34.so
```

即 OpenMP 运行时是 **libgomp**，BLAS 实际是 **OpenBLAS（pthread 版）**。

## 4. 线程实测：主力是 BLAS，不是 OpenMP

700×700 `float64` 矩阵乘（`ij,jk->ik`），KUPL 后端，5 次取中位：

| `OMP_NUM_THREADS` | `OPENBLAS_NUM_THREADS` | 中位耗时 | 解读 |
|---|---|---|---|
| 1 | 1 | 11.04 ms | 基线 |
| 4 | 1 | 10.80 ms | OpenMP 单独几乎无收益 |
| 1 | **4** | **7.25 ms** | **BLAS 多线程才是主要加速源** |
| 4 | 4 | 8.21 ms | 过度并行，比 1/4 慢 13% |

同一脚本在 CUDA 库路径下（`backend=cuda`，numpy 输入仍走 CPU 库）扩展测试：

| OMP | OPENBLAS | 中位耗时 |
|---|---|---|
| 1 | 1 | 12.29 ms |
| 4 | 1 | 11.26 ms |
| 1 | 4 | 4.83 ms |
| 4 | 4 | 6.41 ms |
| 8 | 8 | 23.77 ms（严重过度并行） |

结论：**稠密 GEMM 的加速来自 OpenBLAS 多线程**；OpenMP 主要服务打包/置换/逐元素循环，
以及不走 BLAS 的专用 kernel 内部循环。

## 5. MPI 的边界（默认不启用）

- 只有在 MPI 构建里才编译（`#if defined(PYQCD_EINSUM_ENABLE_MPI)`）；
- 分片范围：**单个输出轴**（`QCD_PARAM_MPI_SHARD_AXIS`，`mpi_runtime.cpp:271`），
  不是任意张量并行；
- 归约：`MPI_Allreduce` 用于错误码（`MPI_MAX`）与配置区间（`MPI_MIN`/`MPI_MAX`）；
- 当前发行版 `mpi_enabled() == False`，`build_info()['mpi_label'] == "non-MPI build"`。

## 6. 加速来源对照（两条后端）

| 路径 | 触发条件 | 加速机制 |
|---|---|---|
| KUPL 通用收缩 | 任意方程 | `dlopen` 的 BLAS GEMM + OpenMP 打包/拷贝 |
| KUPL 专用 kernel | 操作数满足生产契约（3×(4,4,nvec,nvec) peram + 3×(4,4) γ + 2×(nmom,nvec,nvec,nvec) VVV） | OpenMP 循环并行 + 内嵌 GEMM |
| CUDA 通用收缩 | cupy 数组 | cuBLAS + 自写 kernel；`QCD_STATUS_INTERNAL` 时回落 Python |
| CUDA 专用 kernel | 同上契约 | 自写 CUDA kernel（核子 2pt / 3pt 主项） |

## 7. 实践建议

1. **不要同时开 OMP 与 BLAS 线程**：4/4 比 1/4 慢约 13%，8/8 慢 3 倍以上。
   推荐显式钉住其中一层：`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=<n>`。
2. **指定 BLAS 库**：`PYQCD_EINSUM_CBLAS_LIB=/path/to/libopenblas.so`（注意环境变量名是
   `PYQCD_EINSUM_CBLAS_LIB`，不是 `PYQCD_EINSUM_BLAS`）。
3. **想要 MPI**：`CXX=mpicxx PYQCD_EINSUM_ENABLE_MPI=1 bash install.sh kupl`，
   并记住它只分片一个输出轴，收益取决于该轴大小。
4. **测性能前先看线程环境**：基准脚本应同时打印 `OMP_NUM_THREADS` 与
   `OPENBLAS_NUM_THREADS`，否则数字不可比。

## 8. 复现命令

```bash
# 1) 符号层面：是否有 OpenMP / dlopen
nm -D pyqcd/einsum/kupl/libpyqcdeinsum_kupl.so | grep -oE "GOMP_[a-z_]+|omp_[a-z_]+|dlopen|dlsym"

# 2) 运行期实际加载了哪个 BLAS / OpenMP
PYQCD_EINSUM_NATIVE=kupl python -c "
import numpy as np, pyqcd.einsum as pe
a = np.random.rand(300,300) + 1j*np.random.rand(300,300)
pe.contract('ij,jk->ik', a, a)
print('\n'.join(sorted({l.split()[-1] for l in open('/proc/self/maps')
                        if '.so' in l and ('blas' in l.lower() or 'gomp' in l.lower())})))
"

# 3) 线程扩展（在独立进程里逐个配置跑，避免互相污染）
for cfg in "1 1" "4 1" "1 4" "4 4"; do set -- $cfg; \
  PYQCD_EINSUM_NATIVE=kupl OMP_NUM_THREADS=$1 OPENBLAS_NUM_THREADS=$2 python bench.py; done
```

## 9. 测量条件与限制

- 机器：RTX 4060 Laptop + conda 环境（`libopenblasp-r0.3.34`、`libgomp.so.1.0.0`）；
- 版本：`pyqcd==0.1.14`（KUPL 库由本仓库构建脚本编译，`-O3 -fopenmp`）；
- 矩阵规模 700×700 属"中等"，更大规模下 BLAS 多线程收益会更明显，但 8/8 这类
  过度并行仍会退化；
- 未测：MPI 构建下的分片收益（本机默认非 MPI）、`-march=native` 的边际影响、
  不同 BLAS 实现（MKL/BLIS）的对照。
