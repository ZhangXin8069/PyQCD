# cpp/cuda/einsum — CUDA 后端（与 KUPL 同一套 C ABI）

本目录产出 `libpyqcdeinsum.so`，导出面与 `cpp/kupl/einsum` **完全相同**的四个
`apply*Qcd`。冻结契约见 `pyqcd/einsum/AGENTS.md` §5 与
`../../kupl/einsum/include/einsum_define.h`；本文件只登记本目录的落地细节，
冲突时以那两处为准。

## 1. 文件与归属

| 文件 | 职责 | 负责人 |
|---|---|---|
| `include/einsum_cuda.h` | 三个数值 kernel 的 weak 声明 + `applyContractQcd` 的 int32 步骤表编码 | 装配组（本组） |
| `src/apply_plan.cu` | `applyPlanQcd` / `applyContractQcd` / `applyInitQcd` / `applyEndQcd` | 装配组（本组） |
| `make-nv.sh` | nvcc 直编 + 原子软链发布 | 装配组（本组） |
| `include/einsum_define.h` | 冻结 C ABI 的**逐字节镜像**（不属本组，勿改） | 原生组（规划） |
| `include/einsum_permute.h`、`src/permute.cu` | `qcd_cuda_permute` / `qcd_cuda_hadamard` | A 组 |
| `include/einsum_batched_gemm.h`、`src/batched_gemm.cu` | `qcd_cuda_batched_gemm` | B 组 |
| `python/pyqcdeinsum.h` | Cython 侧的平坦 C 声明 | API 组 |
| `build/`、`libpyqcdeinsum.so` | 产物与稳定软链（不提交） | — |

本组只写第一行三件 + 本文件；`src/permute.cu`、`src/batched_gemm.cu`、
两个 kernel 头文件、`include/einsum_define.h`、`cpp/kupl/**` 一律不碰。

## 2. 不变量

1. **导出面**：`-Xcompiler -fvisibility=hidden` + `QCD_EINSUM_API` +
   `-Xlinker --version-script`（`build/einsum_cuda_exports.map`）三重收敛，
   `nm -D --defined-only` 只剩 `applyPlanQcd` / `applyContractQcd` /
   `applyInitQcd` / `applyEndQcd`；三个 `qcd_cuda_*` 是 .so 内部符号。
2. **规划核心唯一真源**：`../../kupl/einsum/src/plan_core.cpp` 与
   `../../kupl/einsum/include/einsum_plan.h` 被**直接编译**进本库，不复制、
   不 fork。`applyPlanQcd` 只是 thin adapter（校验地址 → `PlanSpec` →
   `pyqcd_einsum::plan_contract`），因此 CUDA 与 KUPL 的路径/代价由构造决定
   逐位相同。
3. **ABI 镜像守卫**：`include/einsum_define.h` 必须与 KUPL 版本逐字节相同，
   `make-nv.sh` 用 `cmp -s` 在编译前检查，漂移即报错退出（同一 soname 下
   两套 ABI 不可恢复）。
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
bash cpp/cuda/einsum/make-nv.sh                  # 产物 cpp/cuda/einsum/libpyqcdeinsum.so
bash cpp/cuda/einsum/make-nv.sh --clean          # 删除 build/ 与公开软链
bash cpp/cuda/einsum/make-nv.sh --require-kernels # 缺 kernel 实现时硬失败（CI 用）
```

实际命令（`nvcc` 12.4 / sm_89，架构按
`PYQCD_CUDA_ARCHITECTURES > CMAKE_CUDA_ARCHITECTURES > PYQCD_SM_ARCH > nvidia-smi`
解析）：

```bash
nvcc -O3 -std=c++17 --shared -Xcompiler -fPIC -Xcompiler -fvisibility=hidden \
     -gencode=arch=compute_89,code=sm_89 \
     -I cpp/kupl/einsum/include -I cpp/cuda/einsum/include \
     cpp/cuda/einsum/src/*.cu cpp/kupl/einsum/src/plan_core.cpp \
     -lcublas -lcudart \
     -Xlinker --version-script=build/einsum_cuda_exports.map \
     -Xlinker -soname -Xlinker libpyqcdeinsum.so \
     -o build/libpyqcdeinsum.so
```

- `-I` 顺序把 KUPL 的 include 放在前面：`einsum_plan.h` 与 `einsum_define.h`
  都取自共享目录，本目录的 `include/einsum_define.h` 只是镜像。
- 先链接到 `build/libpyqcdeinsum.so.tmp.$$`，`mv -Tf` 落位；公开路径用临时
  软链 + `mv -Tf` 原子替换（与 `cpp/kupl/einsum/make.sh`、PyQCU 一致）。
- 失败时打印 nvcc 原始输出（同时留档 `build/build.log`）并非零退出。
- `build/` 可用 `PYQCD_EINSUM_BUILD_DIR` 重定向。
- **缺 kernel 实现时**：默认仍产出"只有规划 ABI"的库，并在 stderr 打印缺哪
  三个符号、`applyContractQcd` 将返回 `3`，退出码 **0**（上层
  `build-nv.sh` / `_native.py` 需要这个库来规划）；`--require-kernels`（或
  `PYQCD_EINSUM_REQUIRE_KERNELS=1`）把这同一种状态变成**明确错误 + 退出码 1**。

## 4. `applyContractQcd` 的 plan 编码（version 1）

`plan` 是 int32 流：4 个 header word + `num_steps` 个固定 16 word（8 个 pair）
的步骤记录 + 步骤表之后的 int32 payload 区（存放 permute 的 shape/perm）。
int64 字段用小端两 word 表示：`value = (uint64)(uint32)hi << 32 | (uint32)lo`。

```
plan[0] version(=1)   plan[1] num_steps   plan[2] plan[3] 保留，必须 0
base = 4 + 16 * step
  base+0..3    kind, dst_slot, a_slot, b_slot
  base+4..5    ndim, dtype_id              (dtype_id == -1 → params[8])
  base+6..7    hadamard 元素个数 (lo, hi)
  base+8..9    gemm M (lo, hi)
  base+10..11  gemm N (lo, hi)
  base+12..13  gemm K (lo, hi)
  base+14..15  gemm batch (lo, hi) | permute (shape_offset, perm_off)
kind: 1 = PERMUTE, 2 = HADAMARD, 3 = GEMM
```

- **槽位**：`tensors` 是 int64 数据指针数组，长度
  `num_slots = params[QCD_PARAM_NUM_OPS]`；`slot < num_slots` 取
  `tensors[slot]`，`slot == num_slots` 取 `out[0]`（结果缓冲区）。
  **只有最后一步**可以写结果槽，`out[0]` 必须非空。
- **payload**：`shape_offset` / `perm_off` 是 `plan` 的**绝对 word 下标**，必须
  ≥ payload 起点；`src_shape[ndim]` 与 `dst_perm[ndim]` 均为 int32，adapter 在
  调用 kernel 前把 shape 展宽成 int64。
- **别名**：HADAMARD 允许 `dst == a` 或 `dst == b`（逐元素，可原地）；PERMUTE
  拒绝 `dst == a`，GEMM 拒绝 `dst == a || dst == b`（kernel 不承诺原地安全）。
  PERMUTE 不使用 `b_slot`，必须写 0。
- **校验清单**（任一不满足 → `1`）：空指针；version/保留字；`num_steps`、
  `num_slots`、`dtype_id` 越界；`dst_slot` 越界或非末步写结果槽；`a/b_slot`
  越界或指针为 0；PERMUTE 的 `ndim ∉ [1,8]`、offset 落在步骤表内、shape 为负、
  perm 非 `[0,ndim)` 的置换（含重复轴）；GEMM 的 m/n/k/batch 为负或
  `m*n*k*batch` 溢出 int64；kind 未知。
- **dtype**：`dtype_id` 逐 step 生效，`-1` 表示继承 `params[QCD_PARAM_DTYPE_ID]`；
  取值由数值域定义（当前 `0 complex128 / 1 complex64 / 2 float64 / 3 float32`，
  见 `einsum_permute.h`），本层只做 `[0,63]` 的 sanity 上界。
- **stream**：编码里没有流槽位，adapter 一律传 `nullptr`（默认流），
  排序由调用方用 `cudaDeviceSynchronize`/事件保证；需要显式流的版本属于
  后续 ABI 扩展（`params[10:32]` 保留）。

## 5. 已实现 / 未实现边界（如实登记）

已实现：

- `applyPlanQcd`：与 KUPL **完全相同**的校验与语义，含 0 算子、1 算子、
  `mem_hi < 0` 无限内存、容量不足、下标越界、尺寸为负等全部 fail-closed 分支。
- `applyContractQcd`：`PERMUTE` / `HADAMARD` / `GEMM` 三类步骤的校验、顺序派发、
  kernel 缺失检测、返回值归一；真实 GPU 上 hadamard → permute → gemm 三步链
  与 numpy 结果逐元素相同（见 §6）。
- `applyInitQcd` / `applyEndQcd`：强定义空实现（本后端无 session 状态）。

**未实现（返回 `1`，不假装支持）**：

- **把 `applyPlanQcd` 写出的 `(i, j)` 路径对直接喂给 `applyContractQcd`
  执行一般 einsum**：该编码只有"第几步收缩哪两个操作数"，没有索引集合、
  输出索引和形状，无法推导 permute 的 `src_shape/dst_perm`，也无法推导
  GEMM 的 `m/n/k/batch`。因此本层要求调用方显式给出 kernel 级步骤；
  一般 einsum 的执行编码属于后续版本（需要额外的 index/shape 通道）。
- 任意 stride 的视图（只支持连续缓冲区）、多输出、batched 之外的广播、
  混合精度（`dtype_id` 单值）。
- PERMUTE 的 `ndim > 8`（与 `einsum_permute.h` 的 kernel 上限一致）。
- GEMM / PERMUTE 原地（别名）执行。
- 显式 CUDA 流选择（见 §4，固定默认流）。
- 多线程/多设备调度、错误状态的下钻（kernel 失败统一映射为 `3`）。

## 6. 验收（实测 2026-11-05）

```bash
bash cpp/cuda/einsum/make-nv.sh                     # 退出码 0
nm -D --defined-only cpp/cuda/einsum/libpyqcdeinsum.so | grep Qcd
```

```
0000000000003fa0 T applyContractQcd
0000000000004140 T applyEndQcd
0000000000004130 T applyInitQcd
0000000000003f70 T applyPlanQcd
```

- `readelf -d`：`NEEDED libcublas.so.12`、`libcudart.so.12`；`SONAME
  libpyqcdeinsum.so`。`qcd_cuda_*` 在 `.symtab` 中是 `t`（内部）。
- ctypes 对照 KUPL：**8 算子** 310 条表达式（10 条定长 + 300 条随机），
  `path` / `opt_cost` / `largest_intermediate` 逐位相同，差异 **0**；
  同时与 `refer/git-rep/opt_einsum` 的 `contract_path(..., optimize='greedy')`
  对照 **310/310** 一致。
- `applyContractQcd` 校验用例 17 条全部命中预期状态码；真实 GPU 上
  HADAMARD→PERMUTE→GEMM 三步执行 `status 0`，与 numpy `max|diff| = 0.000e+00`。
- 缺少 kernel 实现时（把本目录源码复制到临时树上、只留 `apply_plan.cu`）：
  默认构建打印缺哪三个符号并以 **0** 退出（规划 ABI 完整，`applyContractQcd`
  实测返回 **3**）；`--require-kernels` 以 **1** 退出并给出明确错误。
