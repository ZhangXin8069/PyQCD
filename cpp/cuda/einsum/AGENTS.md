# cpp/cuda/einsum — CUDA 后端（与 KUPL 同一套 C ABI）

本目录产出 `libpyqcdeinsum_cuda.so`（soname 同名），导出面与 `cpp/kupl/einsum`
（产物 `libpyqcdeinsum_kupl.so`）相同的冻结四入口与
compiled-context 三入口。冻结契约见 `pyqcd/einsum/AGENTS.md` §5 与
`../../kupl/einsum/include/einsum_define.h`；本文件只登记本目录的落地细节，
冲突时以那两处为准。

## 1. 文件与归属

| 文件 | 职责 | 负责人 |
|---|---|---|
| `include/einsum_cuda.h` | pack/reduce/GEMM/stream 内部接口 | 原生组 |
| `src/apply_plan.cu` | `applyPlanQcd` 薄适配与会话钩子 | 原生组 |
| `src/apply_contract.cu` | version-2 计划、compiled context、stream pool、workspace | 原生组 |
| `src/pack.cu` | pack/transpose/reduce/hadamard CUDA kernel | 原生组 |
| `src/batched_gemm.cu` | cuBLAS K-last batched GEMM | 原生组 |
| `make-nv.sh` | nvcc 直编 + 原子软链发布 | 装配组（本组） |
| `include/einsum_define.h` | 冻结 C ABI 的**逐字节镜像**（不属本组，勿改） | 原生组（规划） |
| `include/einsum_permute.h`、`src/permute.cu` | `qcd_cuda_permute` / `qcd_cuda_hadamard` | A 组 |
| `include/einsum_batched_gemm.h`、`src/batched_gemm.cu` | `qcd_cuda_batched_gemm` | B 组 |
| `python/pyqcdeinsum.h` | Cython 侧的平坦 C 声明 | API 组 |
| `build/`、`libpyqcdeinsum_cuda.so` | 产物与稳定软链（不提交） | — |

本组只写第一行三件 + 本文件；`src/permute.cu`、`src/batched_gemm.cu`、
两个 kernel 头文件、`include/einsum_define.h`、`cpp/kupl/**` 一律不碰。

## 2. 不变量

1. **导出面**：`-Xcompiler -fvisibility=hidden` + `QCD_EINSUM_API` +
   `-Xlinker --version-script`（`build/einsum_cuda_exports.map`）三重收敛，
   `nm -D --defined-only` 只保留冻结四入口与 `applyCompileQcd` /
   `applyCompiledQcd` / `applyDestroyQcd`；`qcd_cuda_*` 是内部符号。
2. **规划核心唯一真源**：`../../kupl/einsum/src/plan_core.cpp` 与
   `../../kupl/einsum/include/einsum_plan.h` 被**直接编译**进本库，不复制、
   不 fork。`applyPlanQcd` 只是 thin adapter（校验地址 → `PlanSpec` →
   `pyqcd_einsum::plan_contract`），因此 CUDA 与 KUPL 的路径/代价由构造决定
   逐位相同。
3. **ABI 镜像守卫**：`include/einsum_define.h` 必须与 KUPL 版本逐字节相同，
   `make-nv.sh:44-49` 用 `cmp -s` 在编译前检查，漂移即报错退出。两库 soname
   已经分开（`libpyqcdeinsum_cuda.so` / `libpyqcdeinsum_kupl.so`），动态链接器
   不会替调用方发现 ABI 漂移，只能靠这道关卡。
4. **弱符号**：`qcd_cuda_permute` / `qcd_cuda_hadamard` /
   `qcd_cuda_batched_gemm` 在 `einsum_cuda.h` 中以 `__attribute__((weak))`
   声明。缺实现时链接成功、运行期符号为 NULL，派发侧返回
   `QCD_STATUS_INTERNAL`(3)。**不用桩函数伪造成功**。
5. **不分配**：`apply_plan.cu` 不申请设备内存、不建流、不拷贝张量数据，
   所有缓冲区由调用方持有；规划路径连 GPU 都不需要。
6. **状态码**：`0/1/2/3` 与 `einsum_define.h` 一致；kernel 返回值是合法枚举
   时原样透传，否则归一为 `3`。异常绝不跨过 C 边界。
7. **fail-closed**：`applyContractQcd` 先校验**全部**步骤，任何一条不合法都
   返回 `1` 且一个 kernel 都不启动。

## 3. 构建

```bash
bash cpp/cuda/einsum/make-nv.sh                  # 产物 cpp/cuda/einsum/libpyqcdeinsum_cuda.so
bash cpp/cuda/einsum/make-nv.sh --clean          # 删除 build/ 与公开软链
bash cpp/cuda/einsum/make-nv.sh --require-kernels # 缺 kernel 实现时硬失败（CI 用）
```

实际命令（`nvcc` 12.4 / sm_89，架构按
`PYQCD_CUDA_ARCHITECTURES > CMAKE_CUDA_ARCHITECTURES > PYQCD_SM_ARCH >` 脚本内
`DEFAULT_ARCHITECTURES="6.0 7.0"` 解析，`make-nv.sh:126-134`，**不探测 `nvidia-smi`**）：

```bash
nvcc -O3 -std=c++17 --shared -Xcompiler -fPIC -Xcompiler -fvisibility=hidden \
     -gencode=arch=compute_89,code=sm_89 \
     -I cpp/kupl/einsum/include -I cpp/cuda/einsum/include \
     cpp/cuda/einsum/src/*.cu cpp/kupl/einsum/src/plan_core.cpp \
     -lcublas -lcudart \
     -Xlinker --version-script=build/einsum_cuda_exports.map \
     -Xlinker -soname -Xlinker libpyqcdeinsum_cuda.so \
     -o build/libpyqcdeinsum_cuda.so
```

- `-I` 顺序把 KUPL 的 include 放在前面：`einsum_plan.h` 与 `einsum_define.h`
  都取自共享目录，本目录的 `include/einsum_define.h` 只是镜像。
- 先链接到 `build/libpyqcdeinsum_cuda.so.tmp.$$`，`mv -Tf` 落位；公开路径用临时
  软链 + `mv -Tf` 原子替换（与 `cpp/kupl/einsum/make.sh`、PyQCU 一致）。
- 失败时打印 nvcc 原始输出（同时留档 `build/build.log`）并非零退出。
- `build/` 可用 `PYQCD_EINSUM_BUILD_DIR` 重定向。
- **缺 kernel 实现时**：默认仍产出"只有规划 ABI"的库，并在 stderr 打印缺哪
  三个符号、`applyContractQcd` 将返回 `3`，退出码 **0**（上层
  `build-nv.sh` / `_native.py` 需要这个库来规划）；`--require-kernels`（或
  `PYQCD_EINSUM_REQUIRE_KERNELS=1`）把这同一种状态变成**明确错误 + 退出码 1**。
- 本机实测的架构行为、失败模式与复现命令见 §7（2026-11-08 实测）。

## 4. 执行与并行

- `applyContractQcd` 解析 version-2 执行计划；left `[B,M,K]`、right `[B,N,K]`，
  K 为两侧最低连续轴，cuBLAS 使用 transpose-B。
- compiled context 首次执行分配并复用中间、packed 与 GEMM workspace；重复调用
  不重新解析计划或 cudaMalloc。
- stream pool 由 `PYQCD_EINSUM_CUDA_STREAMS` 控制，最多 32 条且受 batch 数限制。
  pack 后记录 event，工作流分片并行，所有流完成后在默认流汇合。
- 支持 complex128、complex64、float64、float32，混合精度 fail closed。

## 5. 已实现 / 未实现边界（如实登记）

已实现：

- `applyPlanQcd`：与 KUPL **完全相同**的校验与语义，含 0 算子、1 算子、
  `mem_hi < 0` 无限内存、容量不足、下标越界、尺寸为负等全部 fail-closed 分支。
- `applyContractQcd`、compiled-context 三入口、pack/reduce/hadamard、K-last
  batched GEMM 已实现并通过 GPU 数值对照。
- `applyInitQcd` / `applyEndQcd`：强定义兼容钩子。

**未实现（返回 `1`，不假装支持）**：

- 非连续 view 由 Python 绑定显式转连续或回退；不允许静默错误读取 stride。
- 尚未支持：混合精度、单 operand 重复轴、MPI rank 分片和跨设备 collective。
- 当前 stream pool 只在单一当前 CUDA device 内工作；跨设备生产拓扑尚需 MPI。
- `applyContractMpiQcd` 已导出但与 KUPL 不同，当前为返回 `1` 的显式 stub；CUDA MPI
  rank 分片尚未实现，禁止声称支持。

## 6. 验收（实测 2026-11-06）

```bash
bash cpp/cuda/einsum/make-nv.sh                     # 退出码 0
nm -D --defined-only cpp/cuda/einsum/libpyqcdeinsum_cuda.so | grep Qcd
```

```
applyPlanQcd / applyContractQcd / applyCompileQcd / applyCompiledQcd /
applyDestroyQcd / applyInitQcd / applyEndQcd
```

- `readelf -d`：`NEEDED libcublas.so.12`、`libcudart.so.12`；`SONAME
  libpyqcdeinsum_cuda.so`。`qcd_cuda_*` 在 `.symtab` 中是 `t`（内部）。
- ctypes 对照 KUPL：**8 算子** 310 条表达式（10 条定长 + 300 条随机），
  `path` / `opt_cost` / `largest_intermediate` 逐位相同，差异 **0**；
  同时与 `refer/git-rep/opt_einsum` 的 `contract_path(..., optimize='greedy')`
  对照 **310/310** 一致。
- V100/P100 四种 dtype 与 NumPy 对照通过；compiled context 重复执行
  `max|Δ| ≈ 1.9e-15`；streams=4 时 batch=4/6 使用四流，batch=2 自动限制两流。
- T0 distillation GPU 开发窗口：E1 `3.14x`、E2 `4.12x`、E4 `3.10x`；E5 由
  专用 sandwich fast path 继续优化。

## 7. 本机实测构建矩阵（实测日期 2026-11-08）

只登记**本机实跑**结果。主机 Intel i7-13620H / WSL2 x86_64、nvcc CUDA 12.4.131、
RTX 4060 Laptop（compute capability **8.9**）。跨后端总表与逐条复现命令见
`docs/pyqcd_einsum_build_matrix_20261108.md`。

### 7.1 四条实测事实

1. **默认架构产物在本机不可运行。** `PYQCD_CUDA_ARCHITECTURES=8.9 bash build-nv.sh`
   成功；默认走 `DEFAULT_ARCHITECTURES="6.0 7.0"`（`make-nv.sh:20`），且
   `make-nv.sh:146` 只写 `-gencode=arch=compute_XX,code=sm_XX`、**不生成 PTX**，
   于是默认 fatbin 里没有任何 8.x 目标：构建退出码 0，但在 sm_89 上无法执行。
   本机复核命令（`cuobjdump` 不在本机 PATH，故本次未用它复核）：

   ```bash
   cuobjdump --list-elf cpp/cuda/einsum/libpyqcdeinsum_cuda.so
   cuobjdump --list-ptx cpp/cuda/einsum/libpyqcdeinsum_cuda.so   # 预期为空：无 PTX
   ```

2. **MPI 与 `CXX` 冲突（KUPL 侧事实，本目录不受影响）。** KUPL 的 `make.sh` 默认
   `PYQCD_EINSUM_ENABLE_MPI=auto`，本机 `mpicxx` 存在 → 自动开 MPI；但 conda 导出的
   `CXX=x86_64-conda-linux-gnu-c++` 覆盖了脚本默认的 `mpicxx`，构建在
   `mpi_runtime.cpp:7:10: fatal error: mpi.h: No such file or directory` 处失败。
   `make-nv.sh` 没有 MPI 分支，因此
   `PYQCD_CUDA_ARCHITECTURES=8.9 bash build-nv.sh` 不受该冲突影响；`install.sh` /
   `setup.py` 的 `auto` 分支在有 nvcc 且 `cpp/cuda/einsum/src/*.cu` 存在时选 CUDA
   （`install.sh:29-35`、`setup.py:29-31`），本机默认安装路径恰好绕开该失败。

3. **OpenBLAS 探测打印 `not found`，实际由运行期 `dlopen` 解析。** 构建日志是
   `OpenBLAS: not found`（无 `pkg-config`、无系统 `cblas.h`），但本库不链接 BLAS：
   先试 `RTLD_DEFAULT`（进程内已有符号），再按 `libcblas.so.3` → `libcblas.so` →
   `libopenblas.so.0` → `libopenblas.so` 的顺序 `dlopen`
   （`cpp/kupl/einsum/src/apply_contract.cpp:287-290`）。本机
   `ctypes.CDLL("libcblas.so.3")` 与 `ctypes.CDLL("libopenblas.so.0")` 均成功。

4. **`-march=native` 默认与 PyQCU 相反（对照登记，本目录不涉及）。** `make-nv.sh` 用
   `-O3 -std=c++17`（`make-nv.sh:179`）编译，不含任何 `-march`；`-march=native`
   只出现在 KUPL 的 `make.sh:37-41`，`PYQCD_EINSUM_NATIVE_ARCH` 默认 `1`，而 PyQCU
   对应开关默认 OFF。CUDA 产物的跨机风险来自第 1 条的架构目标，不是主机 ISA。

### 7.2 本目录的可用组合

| 组合 | 命令 | 状态（2026-11-08） |
|---|---|---|
| 脚本默认（`6.0 7.0`，无 PTX） | `bash build-nv.sh` | 构建退出码 0；sm_89 上不可运行（7.1-1） |
| 显式 `8.9` | `PYQCD_CUDA_ARCHITECTURES=8.9 bash build-nv.sh` | 实测通过 |
| 显式 `8.0` | `PYQCD_CUDA_ARCHITECTURES=8.0 bash build-nv.sh` | 未实测 |
| 多架构 | `PYQCD_CUDA_ARCHITECTURES="8.0 8.9" bash build-nv.sh` | 未实测（逗号/分号/空格均可分隔，`make-nv.sh:135-136`） |
| CUDA + MPI | — | 不支持（`applyContractMpiQcd` 是返回 1 的 stub） |

产物核对：

```bash
readelf -d cpp/cuda/einsum/libpyqcdeinsum_cuda.so | grep -E 'SONAME|NEEDED'
# SONAME libpyqcdeinsum_cuda.so；NEEDED libcublas.so.12 / libcudart.so.12 / libstdc++ / libm / libgcc_s / libc
nm -D --defined-only cpp/cuda/einsum/libpyqcdeinsum_cuda.so | awk '{print $NF}' | grep Qcd | sort
# 8 个入口（含 fail-closed 的 applyContractMpiQcd stub）
```

`setup.py` 会把这个文件复制到 `pyqcd/einsum/cuda/libpyqcdeinsum_cuda.so`
（`setup.py:42-53`），Cython 扩展以 `-Wl,-rpath,$ORIGIN` 解析它（`setup.py:119-122`）；
磁盘上的 `cpp/cuda/einsum/libpyqcdeinsum_cuda.so` 是指向 `build/` 的软链。
