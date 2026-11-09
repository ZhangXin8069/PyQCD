# cpp/cuda/matx — MatX `einsum` 的 C ABI 桥

本目录产出 `libpyqcmatx_cuda.so`（soname 同名），把 MatX 的
`matx::cutensor::einsum` 收敛成四个 C 入口，供 Python 侧
`pyqcd/testing/matx/_bridge.py` 用 ctypes 调用。ABI 形状镜像
`cpp/cuda/einsum`（平坦整数槽位 + 版本脚本），cuda/KUPL 的冻结四入口是最初的模板。

## 1. 文件与归属

| 文件 | 职责 |
|---|---|
| `include/matx_einsum.h` | 冻结 C ABI v1：状态码、dtype、槽位枚举、四入口声明 |
| `src/matx_einsum.cu` | 唯一实现：10 条 fail-closed 校验 + 桩/真机双路径 |
| `python/pyqcmatx.h` | 与 `cpp/cpu/cpu-einsums/python/pyqceinsums.h` 同形的平坦声明 |
| `make-nv.sh` | nvcc 直编 + 版本脚本 + 临时文件 `mv -Tf` 原子发布 |
| `build/`、`libpyqcmatx_cuda.so` | 产物与稳定软链（不提交，`*.so` 已被根 `.gitignore` 覆盖） |

## 2. ABI（v1）

```c
int  pyqcmatx_abi_version(void);        // 1
int  pyqcmatx_cutensor_enabled(void);   // 1 = 编入 MATX_EN_CUTENSOR
int  pyqcmatx_einsum(const char* subscripts, long long set_ptrs, long long params);
const char* pyqcmatx_last_error(void);
```

- 状态码：`OK=0, INVALID=1, MEMORY=2, INTERNAL=3, UNAVAILABLE=4`。
- dtype：`complex128=0, complex64=1, float64=2, float32=3`。
- `params` = `int32_t[8]`：`[0]=NUM_OPS, [1]=INPUT_RANK, [2]=DTYPE_ID, [3]=OUT_NDIM`。
- `set_ptrs` = `int64_t[8]`：`[0]=OUTPUT`（设备指针）、`[1]=INPUTS`（指向
  `int64_t[nops]` 设备指针表）、`[2]=INPUT_SHAPES`（`int64_t[nops*rank]`）、
  `[3]=OUTPUT_SHAPE`（`int64_t[out_ndim]`）。
- 上界：`nops ∈ {1,2}`、`rank <= 4`、`out_ndim <= 4`；二元要求两输入同 rank。
  越界一律 `INVALID`，不进入 MatX。
- `pyqcmatx_last_error()` 返回线程局部诊断；每次调用先清空。

## 3. 由实际源码确认（`refer/git-rep/MatX` @ `0ac86644a82d63e578f695e335fb2d01e1cb06ae`）

1. `matx::cutensor::einsum(subscripts, ops...)` 返回算子表达式，需赋值给输出张量并
   `.run(exec)`（`include/matx/operators/einsum.h:161`；上游
   `test/00_tensor/EinsumTests.cu` 的写法 `(c = cutensor::einsum(...)).run(exec)`）。
   `EinsumOp::Exec` 直接调用 `include/matx/transforms/einsum.h` 的 `einsum_impl`。
2. 非拥有张量包装：`matx::make_tensor(T *data, const index_t (&shape)[RANK],
   bool owning=false)`（`include/matx/core/make_tensor.h:321`）——**形状是 C 数组引用**，
   不能用 `cuda::std::array` 直接绑定；本实现用 `Shape<R>::values` 持有 C 数组。
3. 0 秩输出：`make_tensor<T>(cuda::std::array<index_t,0>{})` 分配 rank-0 张量
   （`make_tensor.h:172` 的 `ShapeType&&` 重载），执行后 `cudaMemcpy` D2D 拷回；
   零长 C 数组不是合法 C++，所以 ROUT==0 走这条独立分支。
4. `matx::CUDAExecutor` 是 `cudaExecutor` 的别名（`include/matx/executors/cuda.h:233`）。
5. 复数类型使用 `cuda::std::complex<T>`（上游 `MatXComplexTypesCUDAExec` 测试组）。
6. 上游测试覆盖：`Contraction3D`、`Dot`、`GEMM`、`GEMMTranspose`、`Permute`、`Sum`、
   `Trace`、`Contraction3DImplicitOutput`、`GEMMImplicitOutput`、
   `TraceImplicitOutput`、`BroadcastBatchMatmul`、`BroadcastBatchMatmulImplicitOutput`、
   `BroadcastBatchSizeMismatchThrows`、`ComposedInputLifecycle`。
   `transforms/einsum.h` 的 `InferImplicitOutput` 与 ellipsis 解析确认隐式输出与
   `...` 由**实现**支持；`operators/einsum.h` 顶部的 "Ellipses are not supported yet"
   注释是过时注释。

## 4. 构建

```bash
bash cpp/cuda/matx/make-nv.sh                 # 默认：无 MatX 头 → UNAVAILABLE 桩库
PYQCD_MATX_ROOT=refer/git-rep/MatX bash cpp/cuda/matx/make-nv.sh          # 编入 MatX 头
PYQCD_MATX_ROOT=refer/git-rep/MatX bash cpp/cuda/matx/make-nv.sh --cutensor  # 需 cuTENSOR
bash cpp/cuda/matx/make-nv.sh --clean
bash cpp/cuda/matx/make-nv.sh --require-matx   # 缺 MatX 时硬失败（CI 用）
```

- 架构优先级：`PYQCD_CUDA_ARCHITECTURES` > `CMAKE_CUDA_ARCHITECTURES` >
  `PYQCD_SM_ARCH` > `nvidia-smi compute_cap` > 默认 `6.0 7.0`；每个架构同时嵌入
  cubin 与 PTX。
- `--cutensor` 会先检查 `ldconfig -p` 里的 `libcutensor.so` / `libcutensornet.so`，
  缺失即非零退出；链接 `-lcublas -lcusolver -lcutensor -lcutensornet`。
- 版本脚本只导出四个 `pyqcmatx_*` 符号；`nm -D --defined-only` 校验
  `pyqcmatx_einsum` 存在后才 `mv -Tf` 落位。

## 5. 桩行为（无 MatX/cuTENSOR 时）

- `pyqcmatx_abi_version()` → `1`；`pyqcmatx_cutensor_enabled()` → `0`。
- `pyqcmatx_einsum()`：先跑完整校验，合法入参返回 `UNAVAILABLE(4)`，原因串指出
  `PYQCD_MATX_ROOT` 与 `-DMATX_EN_CUTENSOR=ON`（`kNoMatX` / `kNoCutensor`）。
- 桩绝不写输出缓冲区，也绝不返回 `OK`。

## 6. 已实现 / 未验证

已实现（本机可验证）：

- 四入口、soname、版本脚本收敛、原子发布、`nm` 校验。
- 10 条 fail-closed 校验与 `INVALID(1)` 诊断；桩 `UNAVAILABLE(4)` + 可操作原因串。
- 真机路径的**代码结构**：dtype(4) × rank(0..4) × out_ndim(0..4) × nops(1,2) 展开，
  `make_tensor` 非拥有包装、`(c = einsum(...)).run(exec)`、rank-0 输出 D2D 拷回，
  异常归一为 `INTERNAL(3)`。

未验证（不得声称可用）：

- **真机路径从未编译过**：本机缺少 `nvtx3/nvToolsExt.h`（conda CUDA 12.4 未带 NVTX），
  MatX 全部头文件都无条件 `#include` 它，因此连只读头文件都无法通过 nvcc；更没有
  cuTENSOR/cuTensorNet 头与库。
- `matx::index_t` 与本 ABI 的 `int64_t` 宽度一致性、rank-0 输出 D2D 拷贝路径、
  `cudaMemcpy` 在 managed 内存上的行为、以及真实 cuTENSOR 数值结果，均未实测。
- 元素数溢出、非连续输入、混合精度、多 GPU/stream 不在本桥范围。

## 7. 与 `cpp/cpu/cpu-einsums` 的对称关系

| 维度 | 本目录 | `cpp/cpu/cpu-einsums` |
|---|---|---|
| 产物 | `libpyqcmatx_cuda.so` | `libpyqceinsums.so` |
| 指针 | 设备指针 | **主机指针** |
| 操作数 | 1..2（MatX 支持一元） | **恰好 2**（Einsums 无一元） |
| spec | MatX/cuTENSOR `"ik,kj->ij"` | Einsums `"ij <- ik ; kj"` |
| 后端开关 | `PYQCMATX_HAVE_MATX` + `MATX_EN_CUTENSOR` | `PYQCEINSUMS_HAVE_EINSUMS=1` |
| 构建 | `make-nv.sh`（nvcc） | `make.sh`（g++/OpenMP） |
| 桩 | `UNAVAILABLE(4)` + 原因串 | `UNAVAILABLE(4)` + 原因串 |

## 8. 验收（实测 2026-11-10）

```bash
bash cpp/cuda/matx/make-nv.sh
nm -D --defined-only cpp/cuda/matx/libpyqcmatx_cuda.so | awk '{print $NF}' | sort
readelf -d cpp/cuda/matx/libpyqcmatx_cuda.so | grep SONAME
python -m pyqcd.testing.matx.verify
```

期望：四个 `pyqcmatx_*` 符号、`SONAME libpyqcmatx_cuda.so`、
`PASS n/n`（`native` 组在桩库下断言 `UNAVAILABLE(4)` 与 fail-closed）。
