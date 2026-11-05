# cpp/einsum — 原生 einsum 库（C ABI + 规划内核）

本目录是 `pyqcd/einsum` 的原生侧实现。冻结契约见
`pyqcd/einsum/AGENTS.md`（尤其第 5 节 ABI），本文件只登记**本目录**的
落地细节与扩展约定；两者冲突时以 `pyqcd/einsum/AGENTS.md` 为准。

## 1. 文件与归属

| 文件 | 职责 | 负责人 |
|---|---|---|
| `include/einsum_define.h` | 冻结 C ABI：槽位常量、状态码、四个 `apply*Qcd` 声明 | 原生组（规划） |
| `include/einsum_plan.h` | 规划内核的 C++ 接口（`PlanSpec` / `PlanOutcome` / `plan_contract`） | 原生组（规划） |
| `src/apply_plan.cpp` | `applyPlanQcd` 入口：参数校验、地址解包、状态码映射 | 原生组（规划） |
| `src/plan_core.cpp` | greedy 路径搜索 + 代价回放 + 饱和整数运算 | 原生组（规划） |
| `src/apply_contract.cpp` | `applyContractQcd` CPU 数值域 | 原生组（执行，未落地） |
| `src/cuda/**` | CUDA kernel | 原生组（GPU，未落地） |
| `build.sh` | g++ 直编，无 CMake | 原生组（规划） |
| `libpyqcdeinsum.so` | 产物（指向 `build/` 的稳定软链，不提交） | — |

`build.sh` 只编译 `src/*.cpp`（不递归），因此 `src/cuda/**` 不会混进 CPU 构建。
`applyInitQcd` / `applyEndQcd` 目前是 `apply_plan.cpp` 内的 **weak 空实现**，
只用于让规划侧单独构建时导出面完整；数值域负责人提供强定义后自动覆盖，
不会产生重复符号。

## 2. 构建

```bash
bash cpp/einsum/build.sh            # 产物 cpp/einsum/libpyqcdeinsum.so
bash cpp/einsum/build.sh --clean    # 删除 build/ 与公开软链
```

- 编译器 `g++`（可用 `CXX` 覆盖），默认
  `-O3 -fopenmp -shared -fPIC -std=c++17 -fvisibility=hidden -Wall -Wextra`
  （可用 `PYQCD_EINSUM_FLAGS` 覆盖）。当前源码没有 OpenMP 并行区，链接器
  `--as-needed` 因而未写入 `libgomp` 依赖；数值域加入 `#pragma omp` 后，
  同一条 `-fopenmp` 命令会自动带上。
- 先链接到 `build/libpyqcdeinsum.so.tmp.$$`，再 `mv -Tf` 覆盖
  `build/libpyqcdeinsum.so`；公开路径用临时软链 + `mv -Tf` 原子替换，
  与 PyQCU `cpp/kupl/qcu/make.sh` 的稳定软链做法一致。
- 失败时打印编译器原始输出（同时留档 `build/build.log`）并以非零退出。
- 导出面由 `-fvisibility=hidden` + `QCD_EINSUM_API` + `-Wl,--version-script`
  三重收敛，`nm -D --defined-only` 只剩 `apply*Qcd`。
- 构建目录可用 `PYQCD_EINSUM_BUILD_DIR` 重定向（ASAN 旁路构建用）。

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

## 5. fail-closed 与状态码

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

## 6. 验收

```bash
bash cpp/einsum/build.sh
nm -D --defined-only cpp/einsum/libpyqcdeinsum.so | grep Qcd
```

最近一次实测（2026-11-05）：

- 构建退出码 `0`，导出 `applyPlanQcd`(T)、`applyInitQcd`(W)、`applyEndQcd`(W)；
  `applyContractQcd` 待数值域落地后由 `build.sh` 自动纳入。
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

## 7. 未实现（如实登记）

- `planner_id = 1/2/3`（optimal / dp / branch）尚未实现，返回状态码 1，
  `_native.plan` 随之返回 `None` 由上层回退 `_plan_py.py`。
- `memory_limit` 非空时 greedy 应转 `branch(nbranch=1)`，同样未实现，
  返回状态码 1。
- `applyContractQcd`（CPU 数值域）与 `src/cuda/**` 不在本组范围。
