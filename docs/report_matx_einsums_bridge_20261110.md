# MatX / Einsums 参考桥：构建、正确性与四引擎性能报告

- 日期：2026-11-10
- 对象：`refer/git-rep/MatX`（`0ac86644`，v1.1.0-48）与
  `refer/git-rep/Einsums`（`628971cd`，v2.0.0a1-22）
- 代码：`cpp/cuda/matx`、`cpp/cpu/cpu-einsums`、`pyqcd/testing/matx`、
  `pyqcd/testing/cpu_einsums`、`pyqcd/testing/einsum/bench_four_engines.py`
- 原始数据：`logs/einsum-four-engine-20261110.txt`
- 本报告只记录可复现的实测与源码核对结果；未运行的项目明确标为“未验证”。

## 1. 摘要

1. 两个上游库都已引入为 Git submodule，并在独立 mamba 环境中**真实编译、链接、
   执行**；不是仅在文档层面核对 API。
2. 两套 C ABI 桥（四符号、版本脚本收敛、fail-closed）与 ctypes 绑定可用；
   桩库在依赖缺失时返回 `UNAVAILABLE(4)`，不伪造结果。
3. 正确性：`verify_matx` 152/152（桩）/ 150/150（真机，SKIP 1），
   `verify_cpu_einsums` 169/169（SKIP 7）；`python -m pyqcd.testing`
   123 passed / 1 skipped / 0 failed。
4. 性能：15-case 四引擎基准（CPU: qcd/opt/einsums；GPU: qcd/opt/matx）完成。
   - qcd 在批量/ellipsis/两个物理 kernel 上显著领先（CPU 8.7–23×，GPU 1.4–2.7×）。
   - **cpu-einsums 的收缩内核并不慢**：GEMM 4096 纯计算 579 ms（81.2% 计算占比），
     element-wise 纯计算 75 ms；端到端慢主要来自桥每次调用的张量分配（element-wise
     仅 6.4% 是计算）。
   - **matx-einsum** 在 GEMM / batched / ellipsis / many-small 上与 cuBLAS/qcd 持平
     或更快（batched 163.4 ms vs qcd 169.7 / opt 176.8）；element-wise、outer、
     full-reduction 慢 12–14×，是 cuTensorNet 网络开销，不是测量假象。
5. MatX 的 `einsum` 固定 32F compute type（`transforms/einsum.h:157`），
   float64 输入也只有 ~1e-7 精度；但 MatX 的 `matmul` 路径是真正的 FP64
   （实测 `rel=1.9e-16`）。

## 2. 环境与构建（base 未改动）

全部依赖装在独立 conda 环境，未修改 base 或系统配置。

### 2.1 环境清单

| 环境 | 用途 | 关键包 |
|---|---|---|
| `einsum-bench` | 基准驱动 | Python 3.12、numpy 2.5.4、opt_einsum 3.4.0、scipy、cupy 14.2（`cupy-cuda13x`） |
| `matx-cu` | MatX 桥构建 | CUDA 13.5 nvcc、cuTENSOR 2.8.1、cuTensorNet 2.14、NVTX dev、cuBLAS/cuSolver/cuRAND/cuFFT/cuSPARSE/nvjitlink dev、CUDA driver stub、GCC 13/14 |
| `einsums-build` | Einsums 库与桥构建 | CMake、Ninja、GCC 13/14、OpenBLAS、HDF5、fmt、spdlog、mimalloc、cpptrace、Catch2、Apiary |

MatX 需要 CCCL 3.x 的新目录布局（`cuda/std/__algorithm`）；conda 的 `cccl`
只带内部头，因此从 codeload 取上游 `v3.4.2` 源码（隔离在 `/tmp/cccl-3.4.2`）。

```bash
mamba create -y -n einsum-bench -c conda-forge \
    python=3.12 numpy opt_einsum=3.4.0 scipy
mamba run -n einsum-bench python -m pip install cupy-cuda13x

mamba create -y -n matx-cu -c conda-forge -c nvidia \
    "cuda-version=12.4" cuda-nvcc cuda-nvtx cuda-cudart-dev cutensor cuquantum
mamba install -y -n matx-cu -c conda-forge \
    "cuda-nvtx-dev=12.4.127" libcublas-dev libcusolver-dev libcurand-dev \
    libcufft-dev libcusparse-dev libnvjitlink-dev cuda-driver-dev gxx=13

mamba create -y -n einsums-build -c conda-forge \
    cmake ninja gxx=13 gcc=13 openblas hdf5 fmt spdlog mimalloc cpptrace \
    catch2 apiary pkg-config
```

### 2.2 桥的构建

```bash
# MatX（GPU，cuTENSOR）
mamba run -n matx-cu bash -c '
  export PYQCD_CUDA_ARCHITECTURES=8.9
  export PYQCD_MATX_ROOT="$PWD/refer/git-rep/MatX"
  export PYQCD_MATX_INCLUDE_DIRS="-I/tmp/cccl-3.4.2/libcudacxx/include -I/tmp/cccl-3.4.2/cub -I/tmp/cccl-3.4.2/thrust"
  export PYQCD_MATX_NVCC_FLAGS="-ccbin $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
  export PYQCD_MATX_LINK_FLAGS="-L$CONDA_PREFIX/lib/stubs -lcuda"
  bash cpp/cuda/matx/make-nv.sh --cutensor'

# Einsums（CPU）
mamba run -n einsums-build bash -c '
  cmake -S refer/git-rep/Einsums -B /tmp/einsums-build -GNinja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DCMAKE_C_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" \
    -DCMAKE_CXX_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++" \
    -DEINSUMS_WITH_TESTS=OFF -DEINSUMS_BUILD_PYTHON=OFF \
    -DEINSUMS_WITH_DOCUMENTATION=OFF
  cmake --build /tmp/einsums-build -j 8
  export PYQCD_EINSUMS_ROOT="$PWD/refer/git-rep/Einsums"
  export PYQCD_EINSUMS_BUILD_TREE=/tmp/einsums-build
  export PYQCD_EINSUMS_LIBS="-L/tmp/einsums-build/lib -lEinsums-abi1 -Wl,-rpath,/tmp/einsums-build/lib"
  bash cpp/cpu/cpu-einsums/make.sh'
```

Einsums 构建在**示例二进制**（`GPU_GPUBasics`、`basic`）处失败，但
`lib/libEinsums-abi1.so`（44 MB，导出 13623 个 einsum 符号）已完整产出，足够
链接桥；该失败不影响本报告的结论。

### 2.3 构建中暴露并修复的真实问题

MatX：

1. MatX 使用 C++20 concept，`make-nv.sh` 原为 `-std=c++17`，必须改 `-std=c++20`。
2. `make_tensor(T*, const index_t (&shape)[RANK], bool owning=false)` 接收 C 数组
   引用；`std::array` 不能绑定。且两参数调用会被转发引用重载抢走，需显式传
   `false`。
3. rank-0 输出不能形成零长 C 数组；改为分配 rank-0 张量 + `cudaMemcpy` D2D。
4. `--cutensor` 必须同时定义 MatX 自己的 `MATX_EN_CUTENSOR`，否则库能编译但
   每次调用抛 `matxNotSupported`。
5. 链接需要 `-lcuda`（driver stub）与 conda 的 `targets/x86_64-linux/include`
   头路径；否则 dlopen 报 `undefined symbol: cuGetProcAddress`。

Einsums：

1. 真实头文件是 `<Einsums/Tensor/Tensor.hpp>` 与
   `<Einsums/ComputeGraph/Operations.hpp>`。
2. `create_tensor<T>(name, Args...)` 约束 `Args` 可转 `std::size_t`。
3. 不存在顶层 `einsums::initialize`；只有 `einsums::comm::initialize`（MPI），
   模块用启动钩子，桥内已删除该调用。
4. `cg::einsum` 没有一元重载；只有二元与 `sizeof...(Operands) >= 3` 的 n 元，
   因此桥只接受恰好 2 个操作数。
5. rank-0 张量不能作为 einsum 输入或输出（上游 `EagerParityGaps.cpp` 明确记录），
   标量结果走 `cg::dot`；桥在共享校验层 fail-closed。
6. 运行时字符串需显式 `EinsumFormatString(std::string_view(spec))`；`const char*`
   需要两次用户定义转换，不能隐式完成。
7. `create_tensor` 默认**列主序**；主机缓冲区是 C 行主序，必须
   `create_tensor<T>(true, ...)`，否则 GEMM 结果是转置、外积全错。
8. `GeneralTensor` 的移动构造会把源对象清成 rank-0；返回张量必须用纯右值
   （C++17 强制省略），不能 `auto t = ...; return t;`。
9. 二元操作数共享 rank 但 extent 可不同（如 60 vs 24）；拷贝必须各用
   `element_count`，复用 A 的计数会越界写坏 B 的邻居（实测表现为 A 的 dim 变 0、
   `free(): invalid size`）。

## 3. 接口与能力边界

### 3.1 C ABI v1（两套桥同构）

```c
int  pyqcmatx_abi_version(void);        int  pyqceinsums_abi_version(void);
int  pyqcmatx_cutensor_enabled(void);   int  pyqceinsums_backend_enabled(void);
int  pyqcmatx_einsum(const char*, long long, long long);
int  pyqceinsums_einsum(const char*, long long, long long);
const char* pyqcmatx_last_error(void);  const char* pyqceinsums_last_error(void);
```

- 状态码 `OK=0, INVALID=1, MEMORY=2, INTERNAL=3, UNAVAILABLE=4`；
  dtype `complex128=0, complex64=1, float64=2, float32=3`。
- `params` = `int32[8]`：`NUM_OPS, INPUT_RANK, DTYPE_ID, OUT_NDIM`。
- `set_ptrs` = `int64[8]`：`OUTPUT, INPUTS, INPUT_SHAPES, OUTPUT_SHAPE`。
- MatX 用设备指针；Einsums 用主机指针。
- 越界/非法参数 `INVALID`，不进入上游；桩库对合法调用返回 `UNAVAILABLE`。

### 3.2 支持矩阵（15-case 语料）

| 引擎 | 设备 | 支持范围 | 本语料可用 |
|---|---|---|---|
| qcd-einsum | CPU/GPU | 全部 | 15/15 |
| opt_einsum | CPU/GPU | 全部 | 15/15 |
| cpu-einsums | CPU | 二元、同 rank、rank 1–4、输出 rank 1–4，无 `...` | 6/15 |
| matx-einsum | GPU | 1–2 操作数、rank ≤4；二元同 rank | 9/15 |

MatX 不支持的 6 例：3/4 算子链、hyper-edge、batch×matrix（异 rank）、
nucleon 2pt（8 算子）、gamma sandwich（3 算子）。Einsums 不支持的 9 例：
full/partial reduction（rank-0/一元）、链、hyper-edge、ellipsis、batch×matrix、
两个物理 kernel。

### 3.3 精度边界

- MatX `einsum` 把 `CUTENSORNET_COMPUTE_32F` 写死（`transforms/einsum.h:157`，
  2022 年沿用），float64/complex128 也按 32 位计算；桥按 dtype 正确传参，
  限制来自上游。实测 float64 7×7 GEMM `max|Δ|=1.9e-7`。
- **这不等于 MatX 不支持 double**：张量与描述符支持 FP64
  （`double → CUDA_R_64F`），cuBLAS 路径使用 `CUBLAS_COMPUTE_64F`；
  本机 `matmul` float64 256² 实测 `rel=1.9e-16`。
- Einsums 走 CPU FP64/FP32，实测 GEMM `4.4e-16`、complex rank-3 `1.4e-17`、
  外积 `0`。

## 4. 正确性验证

| 入口 | 结果 |
|---|---|
| `python -m pyqcd.testing.matx.verify`（桩） | PASS 152/152 |
| 同上（真机 cuTENSOR） | PASS 150/150，SKIP 1 |
| `python -m pyqcd.testing.cpu_einsums.verify`（桩/真机） | PASS 169/169，SKIP 7 |
| `python -m pyqcd.testing` | 123 passed / 1 skipped / 0 failed |
| 15-case 基准内相对误差 | 64F ≤ 5e-15；MatX 32F ≤ 9.6e-8 |

`matx` 语料含 8 条文档用例与 1 条“文档矛盾”（`...`/`->` 的 API 段过时注释，
已由实现源码裁决为 stale）；`cpu-einsums` 语料含 13 条（9 documented /
4 inferred）与 17 条能力限制。

## 5. 四引擎性能（15-case）

### 5.1 口径

- 用例、尺寸、方程与 `docs/zh/performance.md` 的“基准脚本”15 CASES 完全一致。
- 每例先与 `opt_einsum(dp)` 对齐（相对最大范数），再计时。
- 3 轮，每轮轮换引擎顺序；单次不足 0.3 s 的用例在轮内重复取最好值，取中位数。
- **自适应预热**：一次性探测后按 <0.1 s / <1 s / 其他 预热 5 / 2 / 1 次。
- 桥的输出缓冲**按用例分配、用例结束释放**（跨用例保留曾把 GPU 显存推到临界）。
- CPU 固定线程：`PYQCD_EINSUM_THREADS=off OMP_NUM_THREADS=8
  OPENBLAS_NUM_THREADS=8`。`pyqcd.einsum` 导入会写 OMP/BLAS 环境，同进程后加载
  的 Einsums 会继承；不固定时 GEMM 2048 由 144 ms 变 183 ms。
- 机器：i7-13620H（8 线程）+ RTX 4060 Laptop 8 GB。

### 5.2 CPU（numpy；qcd / opt / einsums）

| 用例 | qcd | opt | einsums |
|---|---:|---:|---:|
| matmul 2D (real) | 718.7 ms | 713.0 ms | 944.8 ms |
| matmul 2D (complex) | 386.1 ms | 339.4 ms | 453.5 ms |
| element-wise | 309.0 ms | 351.7 ms | 1547.6 ms |
| outer product | 63.8 ms | 102.4 ms | 294.8 ms |
| full reduction | 58.0 ms | 63.3 ms | N/A |
| partial reduction | 114.7 ms | 121.2 ms | N/A |
| chain 3 ops | 597.5 ms | 589.7 ms | N/A |
| chain 4 ops | 282.1 ms | 275.5 ms | N/A |
| hyper-edge | 91.2 ms | 99.2 ms | N/A |
| batched GEMM | 374.3 ms | 3802.7 ms | 401.0 ms |
| ellipsis | 658.1 ms | 6510.7 ms | N/A |
| batch × matrix | 49.1 ms | 49.5 ms | N/A |
| many small batches | 105.9 ms | 221.1 ms | 179.1 ms |
| nucleon 2pt (kernel) | 2487.4 ms | 43142.1 ms | N/A |
| gamma sandwich (kernel) | 245.8 ms | 5681.2 ms | N/A |

相对 opt 的倍率（<1 为更快）：qcd 在 matmul 1.01/1.14、element-wise 0.88、
outer 0.62、batched 0.10、ellipsis 0.10、many-small 0.48、nucleon 0.06、
gamma 0.04；einsums 在 matmul 1.33/1.34、element-wise 4.40、outer 2.88、
batched 0.11、many-small 0.81。

### 5.3 GPU（cupy；qcd / opt / matx）

| 用例 | qcd | opt | matx (32F) |
|---|---:|---:|---:|
| matmul 2D (real) | 654.3 ms | 649.8 ms | 649.9 ms |
| matmul 2D (complex) | 326.8 ms | 325.0 ms | 327.0 ms |
| element-wise | 15.4 ms | 15.5 ms | 195.5 ms |
| outer product | 2.6 ms | 2.7 ms | 31.8 ms |
| full reduction | 9.5 ms | 20.1 ms | 282.2 ms |
| partial reduction | 8.8 ms | 8.9 ms | 17.9 ms |
| chain 3 ops | 575.0 ms | 584.7 ms | N/A |
| chain 4 ops | 252.4 ms | 251.4 ms | N/A |
| hyper-edge | 98.3 ms | 102.0 ms | N/A |
| batched GEMM | 169.7 ms | 176.8 ms | 163.4 ms |
| ellipsis | 334.0 ms | 324.3 ms | 324.2 ms |
| batch × matrix | 88.6 ms | 93.7 ms | N/A |
| many small batches | 25.0 ms | 25.1 ms | 23.7 ms |
| nucleon 2pt (kernel) | 875.7 ms | 2340.8 ms | N/A |
| gamma sandwich (kernel) | 796.4 ms | 1096.1 ms | N/A |

相对 opt：qcd 在 matmul 1.01、batched 0.96、ellipsis 1.03、nucleon 0.37、
gamma 0.73；matx 在 matmul 1.00/1.01、batched 0.92、ellipsis 1.00、
many-small 0.94，但 element-wise 12.62、outer 11.97、full reduction 14.01、
partial reduction 2.02。

MatX 隔离复测（10 次预热、10 次采样）与上表一致性：element-wise 239.5 ms、
outer 19.3 ms、full reduction 308.3 ms、matmul 648.5 ms、batched 162.8 ms、
partial 17.9 ms。outer/element-wise 在轮询环境中会受其他引擎的显存/时钟影响，
表中与隔离值的差异已在结论中保留。

### 5.4 Einsums 端到端 vs 真正计算

`cpp/cpu/cpu-einsums/tests/bench_components.cpp` 直接测 `cg::einsum`（张量已建好）
与张量创建：

| 用例 | 纯计算 | 张量分配 | 计算占比 | 端到端（基准同轮） |
|---|---:|---:|---:|---:|
| GEMM 4096 | 578.9 ms | 134.2 ms | **81.2%** | 944.8 ms |
| element-wise 12000 | **75.4 ms** | 1103.7 ms | **6.4%** | 1547.6 ms |
| outer 8000 | 70.0 ms | 173.3 ms | 28.8% | 294.8 ms |
| batched 16×1024 | 173.5 ms | 136.4 ms | 56.0% | 401.0 ms |

与 qcd/opt 的纯计算对比：

| 用例 | Einsums 纯计算 | qcd | opt | 倍率（对 qcd / opt） |
|---|---:|---:|---:|---:|
| GEMM 4096 | 578.9 ms | 718.7 ms | 713.0 ms | 1.24× / 1.23× 更快 |
| element-wise 12000 | 75.4 ms | 309.0 ms | 351.7 ms | 4.1× / 4.7× 更快 |
| outer 8000 | 70.0 ms | 63.8 ms | 102.4 ms | 0.91× / 1.46× |
| batched 16×1024 | 173.5 ms | 374.3 ms | 3802.7 ms | 2.16× / 21.9× 更快 |

结论：Einsums 的收缩内核本身很有竞争力；端到端 element-wise/outer 的慢来自
桥每次调用的 `create_tensor` 分配，不是内核。把端到端数字当作 Einsums 算力会
严重低估（element-wise 仅 6.4% 是计算）。

### 5.5 方法论修正（相对首轮数据的低估/高估）

| 问题 | 影响 | 修正 |
|---|---|---|
| Einsums 继承 pyqcd 的 4/2 线程档 | GEMM 2048 183.5 → 144.1 ms | 固定 `PYQCD_EINSUM_THREADS=off OMP=8 BLAS=8` |
| MatX 预热不足 | batched 259.6 → 163.4 ms；outer 34.3 → 19.3 ms（隔离） | 自适应 5/2/1 次预热 |
| 跨用例保留输出缓冲 | 显存临界，核子 2pt 0.88 s → 4.8 s | 每用例分配、用完释放 |
| 基准脚本计时轮询把 einsums 异常冒泡进 qcd/opt | ellipsis 三引擎误报 ERROR | 先逐引擎正确性，再统一轮询计时 |
| Einsums 端到端当算力 | element-wise 被低估约 20×（75 ms vs 1.55 s） | 组件级拆分，报告纯计算与分配占比 |

## 6. 结论

1. **qcd-einsum**：CPU 上通用形态与 opt 持平（0.96–1.05×），批量轴（10.2×）、
   ellipsis（9.9×）、核子 2pt（17×）、γ 夹心（23×）是结构性优势；GPU 上
   通用形态持平，全约简 2.1×、核子 2.7×、γ 夹心 1.4×。
2. **cpu-einsums**：收缩内核在 GEMM/批量上强（GEMM 4096 236 GFLOP/s，
   batched 比 qcd 快 2.2×、比 opt 快 21.9×，element-wise 比两者快 4×+）；
   当前 C ABI 的逐次张量分配掩盖了这一点。若要做生产级 Einsums 后端，
   应引入张量复用（按 dtype/rank/shape 缓存）或直接使用 Einsums 的 Tensor 句柄，
   而不是每次重建。
3. **matx-einsum**：桥无拷贝、开销可忽略；GEMM/batched/ellipsis/many-small 与
   cuBLAS 持平或更好，element-wise/outer/full-reduction 受 cuTensorNet 网络
   开销限制（12–14×）。MatX 的 einsum 固定 32F，与 64F 引擎不是同精度对比。
4. **两套桥**：均 fail-closed；能力边界（MatX 1–2 算子/rank≤4；Einsums 二元同
   rank、无 rank-0/一元/省略号）由契约测试锁定，不支持项明确 N/A。

## 7. 复现

```bash
# 1) 环境（见 §2.1）
# 2) 构建两套桥（见 §2.2）
# 3) 正确性
python -m pyqcd.testing.matx.verify
python -m pyqcd.testing.cpu_einsums.verify
python -m pyqcd.testing

# 4) 15-case 基准
PYQCD_EINSUM_THREADS=off OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  PYTHONPATH=$PWD LD_LIBRARY_PATH="/tmp/einsums-build/lib:$CONDA_PREFIX/lib" \
  python -m pyqcd.testing.einsum.bench_four_engines --device cpu --json cpu.json

CUDA_PATH=/tmp/cuda13hdr \
  LD_LIBRARY_PATH="<matx-cu>/lib:<matx-cu>/lib/stubs" \
  PYTHONPATH=$PWD \
  python -m pyqcd.testing.einsum.bench_four_engines --device gpu --json gpu.json

# 5) Einsums 计算/分配拆分
#    编译并运行 cpp/cpu/cpu-einsums/tests/bench_components.cpp
```

原始输出：`logs/einsum-four-engine-20261110.txt`。基准驱动：
`pyqcd/testing/einsum/bench_four_engines.py`。

## 8. 评估的局限（威胁到效度的因素）

先声明结论强度：本报告能支撑“数量级优势 / 硬限制”的判断，不能支撑 5–10% 的
精细排名，也不能外推到其它机器、精度或生产规模。

### 8.1 构念效度：比较对象不是同一种东西

- **opt_einsum 不是执行引擎**：CPU 上是 NumPy/BLAS，GPU 上是 CuPy/cuBLAS；
  把它放进四引擎表，实际测的是“规划器 + 后端”。
- **qcd 的路径依赖**：无求和标签交后端、批量轴走 KUPL/CUDA 原生执行器、
  物理形态走专用 kernel、通用 cupy 形态又委派回 `cupy.einsum`。同一个
  “qcd-einsum”标签混合了四类执行体，胜负不能简单归因于“引擎更快”。
- **端到端边界不一致**：qcd/opt 每次调用自行分配输出；MatX/Einsums 使用
  预分配 `out`；Einsums 桥还额外分配输入张量并做主机↔张量拷贝。跨引擎比值
  混合了不同成本边界。
- **精度混淆**：MatX einsum 固定 32F，其余 64F。GPU 上 32F 更省带宽/可能更快，
  “MatX 在 batched 上略快”不能当作同精度优势。
- **“matx-einsum / cpu-einsums”实为“本库桥 + 上游引擎”**：Einsums 端到端主要
  反映我们 ABI 的逐次分配；MatX 桥虽薄，仍是我们写的调用层。

### 8.2 统计效度与稳健性阈值

- 每例 **3 轮**，每轮取 inner 内最快值再取中位数：这是“接近最优”的估计，
  不是典型延迟。
- 本轮修正后的同轮 spread：CPU 36 个样本中位 **1.09**、最大 **7.07**
  （γ 夹心 qcd）；GPU 中位 **1.03**、最大 **1.31**（many-small opt）。
  修正前（预热/缓冲未修）GPU batched 曾达 1.67。
- 跨轮复测（同一 case、同设置）：qcd CPU element-wise 244.5 / 301.6 / 309.0 ms；
  Einsums GEMM 4096 836.7 / 893.0 / 944.8 ms；MatX element-wise
  161.1 / 195.5 ms；qcd GPU γ 夹心 877.4 / 796.4 ms——差异 **10–26%**。
- 因此：只有 **>~15%（CPU）/ >~25%（GPU）** 的差异可作方向性结论；
  5–10% 的差异一律不排名。

| 结论 | 强度 | 依据 |
|---|---|---|
| qcd CPU batched 10.2×、ellipsis 9.9×、nucleon 17.3×、γ 23.1× | **强** | 数量级差，远超跨轮波动 |
| MatX GPU element-wise 12.6×、full reduction 14.0× | **强（定性）** | 数量级差；但含 32F 口径 |
| Einsums engine element-wise 4.7×、batched 21.9× | **强（engine-only）** | 组件探针；需标注非端到端 |
| qcd GPU full reduction 2.1×、nucleon 2.7× | **中** | 跨轮 10% 级，方向可信、幅度不稳 |
| Einsums 端到端 GEMM 1.33× | **中** | 跨轮约 13%，方向可信 |
| matx batched 0.92× vs qcd 0.96×、matmul 1.00–1.02×、chain ±5% | **不可判定** | 落在噪声内 |

### 8.3 内部效度：同进程/同卡的状态耦合

- 四个引擎跑在同一进程、同一 GPU：CuPy 内存池、GPU 时钟、CPU turbo、页缓存
  在 case 之间传递状态，轮换顺序只能部分抵消。
- MatX 的隔离稳态 `outer=19.3 ms`、`element-wise=239.5 ms`，与交替基准中的
  `31.8 ms`、`195.5 ms` 相差 25–65%，说明部分数值依赖“和谁一起跑”。
- 自适应预热（5/2/1 次）不能证明长 case 已到稳态；长 case 只预热 1 次。
- Einsums 的纯计算/分配拆分来自另一个 C++ 进程，分配器状态与线程默认值
  与 Python 基准不完全一致；6.4% 是可靠量级，不是精确映射。且“纯计算”列
  **不含主机↔张量拷贝**（element-wise 8000² 实测拷贝 68.4 ms），
  端到端数字才是当前 ABI 的真实调用成本。
- 计划缓存处于热态：qcd/opt 的规划成本被正确性调用与预热排除，
  **冷启动（首次规划 + 首次加载）未测**。
- 桥使用预分配输出、qcd/opt 内部自行分配，二者成本边界不同。

### 8.4 用例覆盖：结构覆盖而非规模覆盖

- 每个家族**只有一个尺寸**，没有 small/medium/large 曲线，因此不知道
  MatX cuTensorNet 开销被摊薄的 crossover，也不知道 Einsums 分配占比随尺寸的变化。
- 未覆盖：多卡/MPI、多 stream、CUDA graph、非连续输入、size-1 广播、
  混合精度、complex64/float32、稀疏/tiled、生产尺度（L24、nvec=100）。
- MatX/Einsums 只在**支持的子集**上参与，聚合“最好/最差”时样本有偏；
  “综合最优引擎”在本语料上不可定义。
- 未测 NaN/denormal、32F 累加稳定性、峰值显存/RSS、能耗、单位成本。

### 8.5 平台与外部效度：一台笔记本 + 两套软件栈

- i7-13620H（8 线程）+ RTX 4060 Laptop 8 GB，受功耗墙、散热与显存容量限制；
  服务器 CPU（NUMA/更多核）、数据中心 GPU（HBM）、AMD/Apple 平台的排序可能变化。
- 线程固定 8/8，但各引擎最优线程档不同（Einsums GEMM 2048 在 4/2 与 8/8 下
  差 27%）；固定线程保证同条件，不保证各自最佳。
- 版本敏感：conda OpenBLAS、CUDA 13.5、cuTENSOR 2.8.1、cuTensorNet 2.14、
  CuPy 14.2、NumPy 2.5、opt_einsum 3.4.0、GCC 13/14；换 BLAS/CUDA 小版本
  可能翻转小倍率结论。
- GPU 未锁频、未做温控，笔记本 clock 波动足以覆盖 10–20% 差异。

### 8.6 指标单一：只有延迟

- 缺峰值内存（opt 的 1.44 GB 中间体、qcd 批量约 3× opt 工作区、Einsums 每次
  3 份张量、MatX workspace），而这直接决定生产规模能否跑起来。
- 缺尾延迟/吞吐/并发、能耗/成本、冷启动与规划成本；32F 的维护/精度成本
  也未计入。

### 8.7 结论表述的风险

- 报告中的 A/B/C/D 评级是作者综合，不是基准输出的统计结论；不应当作排行榜。
- “CPU/GPU 最优”由单次中位数选出；噪声范围内的名次（matmul、chain、
  many-small）不应写死。
- “Einsums kernel 很强”是**潜力**（张量复用的 engine-only），不是当前 ABI 的
  可交付性能；“Einsums 端到端慢”是**本库桥**的属性，不应归因于上游。
- “MatX 在 batched 上更快”同时受 32F 与 cuTensorNet 缓存状态影响，必须带精度限定。

### 8.8 补强实验优先级

1. 每引擎独立进程 + 锁频 + ≥7–10 轮，报告中位数、IQR、p95 与 bootstrap CI；
   统一把 <15%（CPU）/ <30%（GPU）标为不显著。
2. 每个家族做尺寸扫描（≥3 点），给出 scaling 曲线与 MatX/Einsums 的 crossover。
3. 统一精度：MatX 用 64F（直接 cuTENSOR 或改 compute type）与 64F 对手比；
   或所有引擎单独跑 32F 一组。
4. 统一成本边界：同时报告 engine-only、bridge end-to-end、cold-start 三列。
5. 加入峰值内存、尾延迟；生产规模在显存守卫下单独测。
6. 线程扫描：每引擎给“最佳线程档”和“固定 8/8”两组。
7. 第二台机器/GPU 复核，至少验证大倍率结论的方向性。

### 8.9 数据核对说明

第 5 节的表格数字已由脚本与 `/tmp/bench_cpu5.json`、`bench_gpu3.json`、
`components.log` 逐项比对，0 处数值不一致（过程中修正了两处倍率笔误）。
但**转录正确 ≠ 测量有效**：本节的威胁分析不会因数字核对而消失。

## 9. 新对比测试对象的接入工作流

以 MatX/Einsums 的落地过程为模板，新增一个对比引擎按以下顺序进行（每步都有
可检查产物）：

1. **固定上游**：`refer/git-rep/<name>` 以 Git submodule 固定 commit，并在
   `refer/git-rep/README.md` 登记来源、用途、固定 commit；工作树只读。
2. **C++ 桥**：`cpp/<device>/<name>/` 下放 `include/<name>.h`（状态码、dtype、
   平坦 `int64`/`int32` 槽位）、`src/*.cu|*.cpp`、`python/*.h`、构建脚本与
   `AGENTS.md`；导出面用 version script 收敛，构建失败关闭（缺依赖产出
   `UNAVAILABLE(4)` 桩库）。
3. **Python 绑定**：`pyqcd/testing/<name>/` 下 ctypes 编组、`probe()`、
   `einsum_into()`、`verify.py`；契约语料只取官方文档，未核实项标
   `untested`。
4. **入口与总门**：在 `pyqcd/testing/__init__.py` 与 `conftest.py` 注册
   `test_<name>_contracts`，保持 SKIP/失败语义。
5. **基准接入**：在 `pyqcd/testing/einsum/bench_four_engines.py` 增加 runner、
   支持矩阵、精度标签与 `N/A` 原因；不修改既有 15 个 case。
6. **分发隔离**：`find_packages` 排除测试包，`MANIFEST.in` prune 其源码；
   核心命名空间不引入第三方测试代码。
7. **验收清单**：桩构建；依赖具备时真机构建；对独立 oracle 的正确性；同线程、
   同预热、交替顺序的性能；`form-audit`；`git status` 干净；文档更新。

关键判据：任何一步缺失都会在后续暴露——缺 submodule 固定点无法复现；缺
fail-closed 会把“没跑”伪装成“跑对”；缺支持矩阵会把 `N/A` 误读为慢；
缺分发隔离会污染 wheel/sdist。

## 10. 在更多计算环境对比的展望

当前结论受单机、单卡、单一尺寸与 3 轮采样限制。下一步按优先级：

1. **机器维度**：服务器 CPU（NUMA、更多核）、数据中心 GPU（HBM）、
   AMD/Apple；至少三台机器复核大倍率方向性。
2. **并行维度**：MPI/多卡、多 stream、CUDA graph、分布式 batch 分片。
3. **精度维度**：MatX 64F 路径（或全部引擎统一 32F）、complex64/float32、
   混合精度与 32F 累加稳定性。
4. **规模维度**：每个家族 ≥3 个尺寸做 scaling 曲线，定位 MatX cuTensorNet
   开销被摊薄的 crossover；生产 L24/nvec=100 在显存守卫下测。
5. **指标维度**：峰值内存、尾延迟（p95/p99）、吞吐、能耗、冷启动（规划 +
   首次加载）、单位成本。
6. **配置维度**：每引擎线程扫描（固定 8/8 vs 最佳档）、BLAS/CUDA 小版本矩阵。
7. **方法维度**：独立进程、锁频、≥7–10 轮、bootstrap CI，并把不显著阈值
   固化为报告判据。

优先做 1（第二台机器）与 3（精度对齐），因为它们最可能改变现有排序；
规模与并行维度用于确认大倍率结论是否在生产条件下仍成立。

## 11. 相关提交

| 提交 | 内容 |
|---|---|
| `8f7b7a7` | 引入 Einsums/MatX submodule 与受控 C ABI 桥 |
| `e0ee76f` | 按实际源码校正桥接签名与能力边界 |
| `b8c2966` | 独立 mamba 环境完成真机构建与数值验证 |
| `a68420a` | 四引擎 15-case 基准驱动与原始记录 |
| `dc6333f` | 修正预热/缓冲/线程口径，记录计算占比 |
| `388b746` | 区分“MatX 支持 double”与“einsum 固定 32F 计算” |
| `9a00f28` | 将第三方桥测试代码隔离在 wheel/sdist 之外 |
| `fbc05f7` | 将 cpu_einsums 测试子包迁出生产命名空间 |
