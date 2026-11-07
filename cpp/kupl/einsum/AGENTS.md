# cpp/einsum — 原生 einsum 库（C ABI + 规划内核）

本目录是 `pyqcd/einsum` 的原生侧实现。冻结契约见
`pyqcd/einsum/AGENTS.md`（尤其第 5 节 ABI），本文件只登记**本目录**的
落地细节与扩展约定；两者冲突时以 `pyqcd/einsum/AGENTS.md` 为准。

## 1. 文件与归属

| 文件 | 职责 | 负责人 |
|---|---|---|
| `include/einsum_define.h` | 冻结 C ABI、状态码与 compiled-context 扩展声明 | 原生组（规划） |
| `include/einsum_plan.h` | 规划内核的 C++ 接口（`PlanSpec` / `PlanOutcome` / `plan_contract`） | 原生组（规划） |
| `include/einsum_runtime.h` | version-2 执行计划、pair/unary 分析与 K-lowest 布局 | 原生组（执行） |
| `src/apply_plan.cpp` | `applyPlanQcd` 入口：参数校验、地址解包、状态码映射 | 原生组（规划） |
| `src/plan_core.cpp` | greedy 路径搜索 + 代价回放 + 饱和整数运算 | 原生组（规划） |
| `src/runtime.cpp` | 计划解析、label 分类和布局契约 | 原生组（执行） |
| `src/apply_contract.cpp` | KUPL CPU 数值执行、sandwich/batch-dot fast path、compiled context | 原生组（执行） |
| `src/mpi_runtime.cpp` | 单输出轴 MPI shard、对称 preflight、Allgatherv | 原生组（并行） |
| `make.sh` | g++/OpenMP/OpenBLAS 直接构建 | 原生组（构建） |
| `libpyqcdeinsum_kupl.so` | 产物（指向 `build/libpyqcdeinsum_kupl.so` 的稳定软链，不提交） | — |

KUPL 独立编译 `runtime.cpp`、`plan_core.cpp`、`apply_contract.cpp` 与
`apply_plan.cpp`，不包含或链接 CUDA 源码。

## 2. 构建

```bash
bash build-kupl.sh                  # 仓库根入口
bash cpp/kupl/einsum/make.sh        # 直接构建 KUPL 库
```

- 编译器 `g++`（可用 `CXX` 覆盖），默认
  `-O3 -fopenmp -shared -fPIC -std=c++17 -fvisibility=hidden -Wall -Wextra`
  （可用 `PYQCD_EINSUM_FLAGS` 覆盖）。pack 与 sandwich lane 使用 OpenMP，
  GEMM 使用 CBLAS/OpenBLAS。
- 先链接到 `build/libpyqcdeinsum_kupl.so.tmp.$$`，再 `mv -Tf` 覆盖
  `build/libpyqcdeinsum_kupl.so`；公开路径用临时软链 + `mv -Tf` 原子替换，
  与 PyQCU `cpp/kupl/qcu/make.sh` 的稳定软链做法一致。
- 链接时写入 `-Wl,-soname,libpyqcdeinsum_kupl.so`（`make.sh:195`）；CUDA 后端的
  soname 是 `libpyqcdeinsum_cuda.so`，两者可同时 `dlopen`，互不覆盖。
- 失败时打印编译器原始输出（同时留档 `build/build.log`）并以非零退出。
- 导出面由 `-fvisibility=hidden` + `QCD_EINSUM_API` + `-Wl,--version-script`
  三重收敛，动态表只保留冻结四入口和 compiled-context 三入口。
- 构建目录可用 `PYQCD_EINSUM_BUILD_DIR` 重定向（ASAN 旁路构建用）。
- 本机实测的可用开关、失败模式与复现命令见 §9（2026-11-08 实测）。

## 3. ABI 落地约定

`pyqcd/einsum/AGENTS.md` 第 5 节冻结了 `params[0:10]` 与 `set_ptrs[0:8]`。
规划内核额外使用 `set_ptrs` 的保留槽位传递**规划器无法从数据推断的长度**：

| 槽位 | 含义 |
|---|---|
| `set_ptrs[8]` | 符号总数 `num_symbols`；所有下标必须落在 `[0, num_symbols)` |
| `set_ptrs[9]` | 输出下标个数 `output_ndim` |
| `set_ptrs[10]` | `set_ptrs[4]` 的容量（int32 元素数，须 ≥ `2*(num_ops-1)`） |

其余约定：

- **下标是稠密非负整数**，`set_ptrs[3]` 的尺寸表按符号号索引；不传名字、
  不做字符串哈希。
- `params[3]`（`mem_hi`）为负表示**无内存上限**；非负时上限为
  `((uint64)mem_hi << 32) | (uint32)mem_lo`（此编码与 `_native.py` 一致）。
- `params[1] max_ndim`、`params[5:10]`（seed/verbose/use_blas/dtype/layout）
  对规划阶段无影响，只做 `>= 0` 合法性检查。
- `set_ptrs[6]/[7]`（scratch）当前不使用：内核只用自己的局部容器，
  若声明了 `scratch_size > 0` 则要求指针非零，否则视为非法参数。
- **单算子**：greedy 对 `num_ops == 1` 的行为是"一次一元收缩"，线路上无法
  用 `(i, j)` 对表达，因此 `set_ptrs[4]` 写出 **0 个 pair**，但
  `set_ptrs[5]` 仍给出该一元步的 `opt_cost` / `largest_intermediate`；
  调用方（`_native.plan`）据此补出 `((0,),)`。
- **零算子**：返回 `0`，路径为空，`{opt_cost, largest_intermediate} = {0, 1}`
  （与 `PathInfo` 的 `max(size_list, default=1)` 口径一致；opt_einsum 本身
  会在解析阶段拒绝零算子，故此处无参考实现）。

## 4. 规划语义（planner_id = 0）

与 `refer/git-rep/opt_einsum/opt_einsum/paths.py::ssa_greedy_optimize`
逐步对齐：公共下标并入输出 → Hadamard 合并重复操作数 → 贪心内积
（代价 `size12 - size1 - size2`，平局按 `(cost, id2, id1)`）→ 贪心外积。
`ssa_to_linear` 与 `contract_path` 的 `find_contraction` 回放同样照抄，
因此 `opt_cost` / `largest_intermediate` 与
`contract_path(..., optimize='greedy')[1]` 逐位相同。

实现要点：

- 下标集合用**定长 bitset**（`(num_symbols + 63) / 64` 个 `uint64`），
  并集/交集/差集/掩码/子集判定全是整数位运算，键用 interned id 表示。
- 尺寸乘积用 `__int128` 累加，超过 `INT64_MAX` 立即饱和；代价差
  `size12 - size1 - size2` 也在 `__int128` 内计算后饱和，绝不回绕。
- 候选堆元素保存**建堆时**的 `(cost, id1, id2)` 与 `k12`，弹出时重新读取
  `remaining` 中的当前 ssa id，复刻 opt_einsum 的"过期候选跳过"语义。
- 全部状态都在栈/局部容器里，无全局量、无静态缓存，天然可重入且不泄漏；
  工作集超过预算（键表 > 2^24 words）抛 `PlanBudgetExceeded` → 状态码 2。

## 5. 数值执行与布局

- version-2 计划由 `runtime.cpp` 解析；初始 shape、逐步 positions/labels 与最终
  输出全部校验后才进入内核。
- 二元规范布局为 left `[B,M,K]`、right `[B,N,K]`、out `[B,M,N]`，K 是两侧
  最低连续轴；CBLAS 对 right 使用 transpose-B。
- 输入/输出和中间 layout 变换使用增量 odometer、固定宽度 copy 和按元素数限制的
  OpenMP；小矩阵 pack 不过度并行。
- `ab,...bce,cd->...dae` 走结构化 sandwich kernel；`M=N=1` 最终闭合走
  batch-dot，不分配 pack 或调用 1x1 GEMM。
- compiled context 缓存解析后的 `ExecutionPlan`；输入内容、输出和外部指针不缓存。
- MPI 构建只在 `PYQCD_EINSUM_ENABLE_MPI=1` 时启用；默认 g++ 库不包含 MPI 符号或
  动态依赖。第一阶段只分片一个最终保留的 batch/free 输出轴，rank-local 执行后
  Allgatherv；分片轴若在任一步被消去、广播或多轴则 fail closed。
- Python 多进程 fallback 位于 `pyqcd/einsum/kupl/multiprocess.py`，使用 spawn、
  shared memory 和 worker 单 BLAS 线程；小任务不保证加速，必须与 MPI 分开报告。

## 6. fail-closed 与状态码

`applyPlanQcd` 先做**全量校验再写输出**，任何一个不合法参数都返回非零且
不触碰 `set_ptrs[4]/[5]`：

| 状态 | 触发条件 |
|---|---|
| `0` | 成功 |
| `1` | 指针为空；`num_ops/num_symbols/output_ndim/path_capacity` 越界或为负；下标越界；尺寸为负；`planner_id` 非法或非 greedy；设了 `memory_limit`（greedy 在 opt_einsum 中会转 `branch`，本阶段未实现）；容量不足 |
| `2` | 内核工作集超预算 / 分配失败 |
| `3` | 其它内部错误（含异常兜底） |

上限：`num_ops ≤ 65536`、`num_symbols ≤ 262144`、扁平下标 ≤ 2^26。
所有异常都在 C 边界捕获，不会传播到 ctypes 调用方。

## 7. 验收

```bash
bash build-kupl.sh
nm -D --defined-only cpp/kupl/einsum/libpyqcdeinsum_kupl.so | grep Qcd
```

最近一次实测（2026-11-06）：

- 构建退出码 `0`，导出 `applyPlanQcd`、`applyContractQcd`、`applyCompileQcd`、
  `applyCompiledQcd`、`applyDestroyQcd`、`applyInitQcd`、`applyEndQcd`。
- 随机表达式 **3400 条**（600 + 1200 + 800 + 800，三套生成器）与
  `contract_path(..., optimize='greedy')[1]` 的 `opt_cost` / `path` /
  `largest_intermediate` 全部逐位一致，0 不一致。
- 边界：0 算子、1 算子（`ij->i`、`a->`）、300 符号（3×100 与 1×300）、
  超大尺寸饱和（`opt_cost` 精确值 > `INT64_MAX` 时饱和到 `INT64_MAX`）、
  零尺寸维度，均与参考一致或按约定饱和。
- 越界：ASAN 构建（`-fsanitize=address`）下 600 条随机 + 退化用例零报告；
  输出缓冲区前后哨兵 `0x5A5A5A5A` 未被改写。
- 泄漏：连续 10000 次 `applyPlanQcd`，RSS 增长 **0 字节**（阈值 2 MB）。
- 与 `pyqcd/einsum/_native.py::plan` 的 2800 条对照全部一致。

复现脚本不入库（避免占用他人目录），验证逻辑等价于：直接 `ctypes` 调
`applyPlanQcd`，用 `sys.path.insert(0, 'refer/git-rep/opt_einsum')` 的
`contract_path(..., optimize='greedy')` 作 oracle。

## 8. 未实现（如实登记）

- `planner_id = 1/2/3`（optimal / dp / branch）尚未实现，返回状态码 1，
  `_native.plan` 随之返回 `None` 由上层回退 `_plan_py.py`。
- `memory_limit` 非空时 greedy 应转 `branch(nbranch=1)`，同样未实现，
  返回状态码 1。
- MPI 只支持单输出轴且不做跨 rank 浮点 allreduce；多轴、消去轴分片、NUMA 自动绑定
  尚未落地。多进程 fallback 只支持 NumPy host backend。

## 9. 本机实测构建矩阵（实测日期 2026-11-08）

只登记**本机实跑**结果。主机 Intel i7-13620H / WSL2 x86_64、conda-forge gcc 12.4.0、
CUDA 12.4.131、RTX 4060 Laptop（`sm_89`）。跨后端总表与逐条复现命令见
`docs/pyqcd_einsum_build_matrix_20261108.md`。

### 9.1 四条实测事实

1. **MPI 与 `CXX` 冲突：不显式关 MPI 就会失败。**
   `PYQCD_EINSUM_ENABLE_MPI=0 bash build-kupl.sh` 成功。依赖自动探测时（`make.sh:11`
   默认 `auto`），本机 `command -v mpicxx` 命中 → `MPI_ENABLED=1`（`make.sh:20-24`），
   `DEFAULT_CXX` 变成 `mpicxx`，但 `CXX=${CXX:-$DEFAULT_CXX}`（`make.sh:36`）被 conda
   已导出的 `x86_64-conda-linux-gnu-c++` 覆盖；该编译器不带 MPI include，构建在
   `src/mpi_runtime.cpp:7` 的 `#include <mpi.h>` 处失败：

   ```
   mpi_runtime.cpp:7:10: fatal error: mpi.h: No such file or directory
   ```

   修复方向：`CXX=mpicxx` 或显式给出 MPI include（**本次未复跑**，只登记方向）。

2. **OpenBLAS 探测与运行期不一致。** `make.sh` 的 `_discover_openblas` 在本机打印
   `OpenBLAS: not found`（无 `pkg-config`、`/usr/include/cblas.h` 等候选头全缺、
   `ldconfig -p` 无记录），但 BLAS 不在链接期解析：`src/apply_contract.cpp:287-290`
   先试 `RTLD_DEFAULT`（进程内已有符号），再按 `libcblas.so.3` → `libcblas.so` →
   `libopenblas.so.0` → `libopenblas.so` 的顺序 `dlopen`。本机
   `ctypes.CDLL("libcblas.so.3")` 与 `ctypes.CDLL("libopenblas.so.0")` 均成功
   （`ldconfig -p` 看不到，只在 conda 前缀 `/home/zhangxin/miniforge3/lib/` 找到），
   数值路径可用。判定口径：**日志里的 `OpenBLAS: not found` 不能当作 BLAS 不可用**。
   运行期可用 `PYQCD_EINSUM_CBLAS_LIB`（`src/apply_contract.cpp:269`）指定具体库。

3. **`-march=native` 默认 ON，与 PyQCU 相反。** `NATIVE_ARCH=${PYQCD_EINSUM_NATIVE_ARCH:-1}`
   （`make.sh:37-41`）默认追加 `-march=native -funroll-loops`；PyQCU 对应开关默认
   OFF。后果是默认产物绑定构建机 ISA，跨机分发（含 Kunpeng/aarch64）有不可加载 /
   非法指令风险；该风险本次**未实机复跑**，但两边默认值的差异是确定的。
   需要可分发产物时应显式 `PYQCD_EINSUM_NATIVE_ARCH=0`（退回 `make.sh:38`
   不含 `-march` 的基线 flags）。

4. **发布方式与 PyQCU 一致（对照登记，非本次实测）。** 稳定软链 + `mv -Tf` 原子替换
   与 `cpp/kupl/qcu/make.sh` 同款（据 §2 记载）；与 PyQCU 的已知差异集中在第 3 条的
   默认 ISA 开关，其余行为本次未逐项核对。

### 9.2 本目录的可用组合

| 组合 | 命令 | 状态（2026-11-08） |
|---|---|---|
| MPI 关闭 | `PYQCD_EINSUM_ENABLE_MPI=0 bash build-kupl.sh` | 实测通过 |
| MPI 自动探测（默认） | `bash build-kupl.sh` | 实测失败（9.1-1） |
| MPI 开启 + 指定编译器/include | `PYQCD_EINSUM_ENABLE_MPI=1 CXX=mpicxx bash build-kupl.sh` | 未实测 |

产物核对（非 MPI 构建）：

```bash
readelf -d cpp/kupl/einsum/libpyqcdeinsum_kupl.so | grep -E 'SONAME|NEEDED'
# SONAME libpyqcdeinsum_kupl.so；NEEDED 仅 libstdc++/libm/libgomp/libgcc_s/libc/ld-linux
nm -D --defined-only cpp/kupl/einsum/libpyqcdeinsum_kupl.so \
  | awk '{print $NF}' | grep Qcd | sort
# 8 个入口，含 applyContractMpiQcd——非 MPI 构建里它是
# src/apply_contract.cpp:1678 的 fail-closed stub（返回 QCD_STATUS_INVALID=1）。
```
