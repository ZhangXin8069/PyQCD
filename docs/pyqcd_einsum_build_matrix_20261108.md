# pyqcd-einsum 构建矩阵（实测日期 2026-11-08）

> 口径：本文只写**本机实际跑过**的命令、可直接复核的产物事实，以及脚本中可对照的
> 行号。未跑过的组合一律标「未实测」，不写结论；全文**不含任何性能数字**。
> 工作树：`/home/zhangxin/PyQCD`；同日后端实现事实见
> `cpp/kupl/einsum/AGENTS.md` §9 与 `cpp/cuda/einsum/AGENTS.md` §7。

## 0. 本机可用的两条命令（TL;DR）

```bash
PYQCD_EINSUM_ENABLE_MPI=0 bash build-kupl.sh      # KUPL：必须显式关 MPI
PYQCD_CUDA_ARCHITECTURES=8.9 bash build-nv.sh     # CUDA：必须显式给 sm_89
```

不带这两个变量时，本机默认行为分别是：KUPL **编译失败**（`mpi.h` 找不到）、
CUDA **构建成功但产物在 sm_89 上无法执行**（默认只出 sm_60/sm_70 的 SASS）。

## 1. 实测环境

| 项 | 值 | 复核命令 |
|---|---|---|
| 主机 | WSL2 x86_64，Linux `6.18.54.1-microsoft-standard-WSL2` | `uname -srm` |
| CPU | 13th Gen Intel(R) Core(TM) i7-13620H，`nproc` = 8 | `grep -m1 'model name' /proc/cpuinfo` |
| 主机编译器 | conda-forge gcc 12.4.0（`g++` 与 `x86_64-conda-linux-gnu-c++` 同版本） | `g++ --version` |
| `CXX` | `/home/zhangxin/miniforge3/bin/x86_64-conda-linux-gnu-c++`（conda 预置，会覆盖 `make.sh` 的默认编译器） | `echo "${CXX:-<unset>}"` |
| MPI wrapper | `/home/zhangxin/miniforge3/bin/mpicxx`（conda 包装，`--version` 报 conda-forge gcc 12.4.0） | `command -v mpicxx` |
| nvcc | CUDA 12.4.131（`cuda_12.4.r12.4/compiler.34097967_0`） | `nvcc --version` |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，compute capability **8.9** | `nvidia-smi --query-gpu=name,compute_cap --format=csv` |
| `cuobjdump` | **不在 PATH**，本次未做 fatbin 逐架构复核 | `command -v cuobjdump` |
| Python | 3.12.4（miniforge3） | `python3 -V` |
| OpenBLAS | 只在 conda 前缀找到：`libopenblas.so.0`、`libcblas.so.3 -> libopenblasp-r0.3.34.so`；`ldconfig -p` 无记录、无 `pkg-config`、系统无 `cblas.h` | `ls /home/zhangxin/miniforge3/lib/lib*blas*` |

## 2. 实际产物与 soname（2026-11-08 逐项核对）

| 后端 | 公开路径 | 指向 | SONAME | 关键 `NEEDED` |
|---|---|---|---|---|
| KUPL | `cpp/kupl/einsum/libpyqcdeinsum_kupl.so` | `build/libpyqcdeinsum_kupl.so` | `libpyqcdeinsum_kupl.so` | `libstdc++`、`libm`、`libgomp`、`libgcc_s`、`libc`、`ld-linux`（**无 BLAS、无 MPI**） |
| CUDA | `cpp/cuda/einsum/libpyqcdeinsum_cuda.so` | `build/libpyqcdeinsum_cuda.so` | `libpyqcdeinsum_cuda.so` | `libcublas.so.12`、`libcudart.so.12`、`libstdc++`、`libm`、`libgcc_s`、`libc` |
| Cython 扩展 | `pyqcd/einsum/cuda/einsum.cpython-312-x86_64-linux-gnu.so` | 实体文件 | — | `libpyqcdeinsum_cuda.so`、`libc` |
| 打包副本 | `pyqcd/einsum/cuda/libpyqcdeinsum_cuda.so` | `setup.py` 从 `cpp/cuda/einsum/` 复制 | `libpyqcdeinsum_cuda.so` | 同上 |

两个原生库都导出同一组 8 个 `Qcd` 入口（`applyPlanQcd`、`applyContractQcd`、
`applyContractMpiQcd`、`applyCompileQcd`、`applyCompiledQcd`、`applyDestroyQcd`、
`applyInitQcd`、`applyEndQcd`）。非 MPI 构建里的 `applyContractMpiQcd` 是
`cpp/kupl/einsum/src/apply_contract.cpp:1678` 的 fail-closed stub（返回
`QCD_STATUS_INVALID` = 1），**不是**真实 MPI 实现。

```bash
readelf -d cpp/kupl/einsum/libpyqcdeinsum_kupl.so    | grep -E 'SONAME|NEEDED'
readelf -d cpp/cuda/einsum/libpyqcdeinsum_cuda.so    | grep -E 'SONAME|NEEDED'
nm -D --defined-only cpp/kupl/einsum/libpyqcdeinsum_kupl.so \
  | awk '{print $NF}' | grep Qcd | sort
```

## 3. 构建组合矩阵

### 3.1 总表

| ID | 后端 | 开关 / 命令 | 状态（2026-11-08） | 产物 |
|---|---|---|---|---|
| C1 | KUPL | `PYQCD_EINSUM_ENABLE_MPI=0 bash build-kupl.sh` | **实测通过** | `libpyqcdeinsum_kupl.so` |
| C2 | KUPL | `bash build-kupl.sh`（`PYQCD_EINSUM_ENABLE_MPI` 未设 → `auto`） | **实测失败**（`mpi.h: No such file or directory`） | 无 |
| C3 | KUPL | `PYQCD_EINSUM_ENABLE_MPI=1 CXX=mpicxx bash build-kupl.sh` | 未实测（修复方向） | — |
| C4 | KUPL | `PYQCD_EINSUM_ENABLE_MPI=1 PYQCD_EINSUM_FLAGS='... -I<mpi include>' bash build-kupl.sh` | 未实测（修复方向） | — |
| C5 | CUDA | `PYQCD_CUDA_ARCHITECTURES=8.9 bash build-nv.sh` | **实测通过**（本机唯一可运行组合） | `libpyqcdeinsum_cuda.so` |
| C6 | CUDA | `bash build-nv.sh`（默认 `6.0 7.0`，无 PTX） | **构建退出码 0，sm_89 上不可运行** | `libpyqcdeinsum_cuda.so` |
| C7 | CUDA | `PYQCD_CUDA_ARCHITECTURES=8.0 bash build-nv.sh` | 未实测 | — |
| C8 | CUDA | `PYQCD_CUDA_ARCHITECTURES="8.0 8.9" bash build-nv.sh` | 未实测 | — |
| C9 | CUDA + MPI | 无对应开关 | **未支持**（stub 返回 1） | — |
| C10 | DCU | 仓库内无 `cpp/dcu` | **未支持**（无源码） | — |

补充：C1 与 C5 的产物 soname 不同（见 §2），因此同一进程内可同时 `dlopen`
两个库；本机 `ctypes.CDLL` 加载两者均成功。

### 3.2 与 PyQCU 的对照差异

| 维度 | pyqcd-einsum（本仓库，实测） | PyQCU（外部仓库） | 结论 |
|---|---|---|---|
| 主机 ISA 优化 | `make.sh:37-41`：`PYQCD_EINSUM_NATIVE_ARCH` 默认 **1**，追加 `-march=native -funroll-loops` | 对应开关默认 **OFF**（任务给定事实，本次未在 PyQCU 侧复跑） | **默认值相反**；本仓库默认产物因此绑定构建机 ISA |
| 稳定软链发布 | `build/` 内实体 + 相对软链 + `mv -Tf` 原子替换（`make.sh:219-222`、`make-nv.sh:220-223`） | `cpp/kupl/qcu/make.sh` 同款做法（据 `cpp/kupl/einsum/AGENTS.md` §2 记载） | 做法一致（未复跑） |
| 产物命名 | `libpyqcdeinsum_kupl.so` / `libpyqcdeinsum_cuda.so` | 未核对 | 仅登记本仓库事实 |
| 其余行为 | — | — | 本次未逐项核对，不作结论 |

## 4. 逐条组合：可运行条件与已知失败模式

### C1 — KUPL，MPI 关闭（实测通过）

命令：

```bash
cd /home/zhangxin/PyQCD
PYQCD_EINSUM_ENABLE_MPI=0 bash build-kupl.sh
```

- 可运行条件：`CXX`（本机为 conda gcc 12.4.0）可用；`libgomp` 存在（`-fopenmp`）；
  BLAS 不必在链接期存在（见本节失败模式第三条）。
- 成功判据：脚本打印 `MPI: disabled` 且以 `make.sh: SUCCESS` 结束；产物
  `build/libpyqcdeinsum_kupl.so` 存在，公开软链指向它；`readelf -d` 无 BLAS/MPI
  动态依赖。
- 已知失败模式：
  - 设成 `auto` 时会走 C2（`make.sh:19-25` 用 `command -v mpicxx` 判定）。
  - `-march=native`（默认，`make.sh:37-41`）使产物绑定构建机 ISA：跨机分发有
    不可加载 / 非法指令风险（**本次未实机复跑**；避免方法：`PYQCD_EINSUM_NATIVE_ARCH=0`）。
  - 日志里 `OpenBLAS: not found` 属正常现象（`_discover_openblas`，`make.sh:68-125`），
    不影响编译；BLAS 在运行期 `dlopen`（`src/apply_contract.cpp:287-290`）。

### C2 — KUPL，MPI 自动探测（实测失败）

不设 `PYQCD_EINSUM_ENABLE_MPI` 时 `make.sh:11` 取 `auto`；本机 `mpicxx` 存在
→ `MPI_ENABLED=1`（`make.sh:20-24`）→ `DEFAULT_CXX=mpicxx`。但 `make.sh:36` 是
`CXX=${CXX:-$DEFAULT_CXX}`，conda 已导出的 `x86_64-conda-linux-gnu-c++` 优先，
该编译器不带 MPI include，于是 `src/mpi_runtime.cpp:7`（`#include <mpi.h>`）失败：

```
mpi_runtime.cpp:7:10: fatal error: mpi.h: No such file or directory
```

- 命中条件（可自检）：`make.sh` 头部打印 `MPI: enabled`，且 `compiler:` 一行是
  `x86_64-conda-linux-gnu-c++` 而不是 `mpicxx`。
- 影响面（脚本调用链推得，本工作树未单独复跑）：`build-kupl.sh`、
  `install.sh kupl`、`setup.py`（`PYQCD_EINSUM_BACKEND=kupl`）都经
  `cpp/kupl/einsum/make.sh`，因此同环境同参数下会遇到同一失败。`install.sh`
  与 `setup.py` 的 `auto` 分支在本机选 CUDA（见 C5），恰好绕开。

### C3 / C4 — KUPL，MPI 开启且手动指定编译器或 include（未实测）

修复方向（**仅登记，未复跑**）：把 `CXX` 指向 MPI wrapper，使编译命令带上 MPI
include/lib：

```bash
PYQCD_EINSUM_ENABLE_MPI=1 CXX=mpicxx bash build-kupl.sh
# 或显式 include（路径按 MPI 安装位置替换）
PYQCD_EINSUM_ENABLE_MPI=1 PYQCD_EINSUM_FLAGS="-O3 -fopenmp -shared -fPIC -std=c++17 \
  -fvisibility=hidden -Wall -Wextra -I<mpi include>" bash build-kupl.sh
```

未实测项包括：`mpicxx` 是否能同时提供运行期 `libmpi` 依赖、`applyContractMpiQcd`
是否按单输出轴分片约束工作、以及与 Python 多进程 fallback 的对照（后者见
`cpp/kupl/einsum/AGENTS.md` §5、§8）。

### C5 — CUDA，显式 sm_89（实测通过）

```bash
cd /home/zhangxin/PyQCD
PYQCD_CUDA_ARCHITECTURES=8.9 bash build-nv.sh
```

- 可运行条件：nvcc 12.4；产物中带 sm_89 的 cubin；运行时能加载
  `libcublas.so.12`、`libcudart.so.12`（本机由产物 RPATH 解析到 conda 前缀，见 §7）。
- 成功判据：`make-nv.sh: SUCCESS`；`build/libpyqcdeinsum_cuda.so` 存在；
  `SONAME libpyqcdeinsum_cuda.so`。
- 变体：`--require-kernels`（或 `PYQCD_EINSUM_REQUIRE_KERNELS=1`）会把「缺
  `qcd_cuda_permute`/`qcd_cuda_hadamard`/`qcd_cuda_batched_gemm`」从警告变成
  退出码 1（`make-nv.sh:206-218`），CI 用它防「只有规划 ABI」的库蒙混过关。

### C6 — CUDA，脚本默认架构（构建通过，本机不可运行）

- 根因：`DEFAULT_ARCHITECTURES="6.0 7.0"`（`make-nv.sh:20`），
  `make-nv.sh:146` 只生成 `-gencode=arch=compute_XX,code=sm_XX`，**不生成 PTX**
  （没有 `code=compute_XX`），因此 fatbin 里没有 8.x 目标，sm_89 设备上无可用
  kernel image。构建本身退出码 0。
- 判定方式（本机缺 `cuobjdump`，未执行）：

  ```bash
  cuobjdump --list-elf cpp/cuda/einsum/libpyqcdeinsum_cuda.so   # 期望看到 sm_60/sm_70
  cuobjdump --list-ptx cpp/cuda/einsum/libpyqcdeinsum_cuda.so   # 默认构建期望为空
  ```

- 结论：默认 `build-nv.sh` 适用于构建机侧的规划 ABI 与老架构设备，**不能**直接
  在本机 RTX 4060（sm_89）上跑数值。

### C7 — CUDA，显式 sm_80（未实测）

```bash
PYQCD_CUDA_ARCHITECTURES=8.0 bash build-nv.sh
```

未实测：本机无 sm_80 设备，也未在 sm_89 上验证该 cubin 是否能运行
（同大版本向前兼容属厂商规则，本次未复跑，不作为结论）。

### C8 — CUDA，多架构（未实测）

```bash
PYQCD_CUDA_ARCHITECTURES="8.0 8.9" bash build-nv.sh
```

`make-nv.sh:135-136` 把逗号、分号、空格都归一为分隔符，逐个生成 `-gencode`；
该命令本次未执行，产物体积与运行表现未记录。

### C9 — CUDA + MPI（未支持）

`make-nv.sh` 没有任何 MPI 探测或编译分支；`cpp/cuda/einsum/src/apply_contract.cu:1749`
的 `applyContractMpiQcd` 是返回 1 的显式 stub。与 KUPL 不同，**禁止声称 CUDA 支持
MPI rank 分片**（另见 `cpp/cuda/einsum/AGENTS.md` §5）。

### C10 — DCU 后端（未支持）

本工作树 `cpp/` 下只有 `cuda/` 与 `kupl/`（`ls cpp`），没有 DCU 源码、构建脚本或
产物。任何 DCU 相关声明都缺实现基础。

## 5. 与仓库脚本的对应关系

### 5.1 入口与内部脚本

| 文件 | 作用 | 关键行号 | 落到的产物 |
|---|---|---|---|
| `build-kupl.sh` | KUPL 仓库根入口：设 `PYQCD_EINSUM_BACKEND=kupl` → `source env.sh` → 选 build dir → 调 `make.sh` | 7-15、18 | `cpp/kupl/einsum/libpyqcdeinsum_kupl.so` |
| `cpp/kupl/einsum/make.sh` | g++ 直编：MPI 判定 11-30、`CXX` 36、`NATIVE_ARCH` 37-41、flags 38-42、OpenBLAS 探测 68-125、源集合 128-158、version-script 161-175、编译链接 190-196（soname 在 195）、原子发布 219-222 | 同上 | 同上 |
| `build-nv.sh` | CUDA 仓库根入口：设 `PYQCD_EINSUM_BACKEND=cuda` → `source env.sh` → 调 `make-nv.sh` | 7-15、18 | `cpp/cuda/einsum/libpyqcdeinsum_cuda.so` |
| `cpp/cuda/einsum/make-nv.sh` | nvcc 直编：`DEFAULT_ARCHITECTURES` 20、ABI `cmp -s` 守卫 44-49、架构解析 126-136、链接 179-189（soname 在 188）、缺 kernel 判定 206-218、发布 220-223 | 同上 | 同上 |
| `install.sh` | `kupl` / `cuda` / `auto` 选择 → 先构建原生库 → `setup.py build_ext --inplace` → 检查产物 → 打印 `PYQCD_EINSUM_LIB` | 模式 26-38、auto 判定 29-35、原生构建 40-46、setup 调用 52-55、产物检查 57-65、打印 67-70 | 原生库 + `pyqcd/einsum/cuda/einsum*.so` |
| `setup.py` | backend 解析 23-34、原生库复制 42-53、`build_ext` 钩子 56-65、Cython 扩展定义 77-129、`package_data` 143-146 | 同上 | `pyqcd/einsum/<backend>/libpyqcdeinsum_<backend>.so`、`pyqcd.einsum.cuda.einsum` 扩展 |
| `env.sh` | 统一 `PYQCD_EINSUM_{KUPL,CUDA}_LIB` 与 build dir（13-17）、backend 选择（19-51）、`PYTHONPATH`/`LD_LIBRARY_PATH`（54-65） | 同上 | — |

要点：

- `install.sh` 与 `setup.py` 的 `auto` 判定相同：`nvcc` 可用且
  `cpp/cuda/einsum/src/*.cu` 存在就选 CUDA（`install.sh:29-35`、`setup.py:29-31`）。
  本机两条件都成立，所以**不带参数的 `bash install.sh` 走 CUDA 路径**。
- `setup.py` 只给 CUDA 后端编 Cython 扩展（`setup.py:78-79` 对非 cuda 直接返回空）；
  缺 Cython 时抛 `RuntimeError` 并提示改用 `bash install.sh kupl`（`setup.py:83-88`）。
- 扩展链接方式是 `libraries=["pyqcdeinsum_cuda"]` +
  `runtime_library_dirs=["$ORIGIN", <native dir>]`（`setup.py:117-122`），
  所以 `pyqcd/einsum/cuda/` 下那份 `libpyqcdeinsum_cuda.so` 副本是运行时依赖。

### 5.2 环境变量

| 变量 | 生效位置 | 默认 | 本机实测效果 |
|---|---|---|---|
| `PYQCD_EINSUM_ENABLE_MPI` | `make.sh:11-30` | `auto` | `0` 成功；`auto` 选中 MPI 后失败（C2） |
| `CXX` | `make.sh:36` | MPI 关闭时 `g++`，开启时 `mpicxx` | conda 预置值覆盖默认 → C2 的根因 |
| `PYQCD_EINSUM_NATIVE_ARCH` | `make.sh:37-41` | `1`（追加 `-march=native -funroll-loops`） | 默认 ON；与 PyQCU 默认 OFF 相反 |
| `PYQCD_EINSUM_FLAGS` | `make.sh:42` | `make.sh:38-41` 的 `DEFAULT_FLAGS` | 未使用（可用作 include/flag 覆盖） |
| `PYQCD_CUDA_ARCHITECTURES` / `CMAKE_CUDA_ARCHITECTURES` / `PYQCD_SM_ARCH` | `make-nv.sh:126-136` | 三级均空时用 `6.0 7.0` | 只有显式 `8.9` 才在本机可运行 |
| `PYQCD_EINSUM_BUILD_DIR` | `make.sh:6`、`make-nv.sh:7`、两个根入口 6-15 | `cpp/<backend>/einsum/build` | 未改动 |
| `PYQCD_OPENBLAS_FLAGS` / `_LIBRARY` / `_INCLUDE_DIR` | `make.sh:72-85`、`make-nv.sh:55-68` | 空 | 未使用（本机探测结果为 `not found`） |
| `PYQCD_EINSUM_CBLAS_LIB` | 运行期 `cpp/kupl/einsum/src/apply_contract.cpp:269` | 空 | 未使用（候选表自动命中） |
| `PYQCD_EINSUM_REQUIRE_KERNELS` | `make-nv.sh:18`、214-217 | `0` | 未使用 |
| `PYQCD_EINSUM_BACKEND` | `env.sh:19-51`、`setup.py:15-34` | `auto` | 未显式设置 |
| `PYQCD_SKIP_NATIVE_BUILD` / `PYQCD_EINSUM_SKIP_CYTHON` | `install.sh:21-22`、40、`setup.py:60`、80 | 未设 | 未使用 |
| `NVCC` | `make-nv.sh:12` | `nvcc` | 使用 PATH 中的 conda nvcc 12.4 |
| `PYTHONPATH` / `LD_LIBRARY_PATH` | `env.sh:54-65` | 追加仓库根与两个原生目录 | 生效 |

## 6. 与 PyQCU 的对照差异（结论汇总）

1. **`-march=native` 默认相反**：本仓库 `make.sh:37-41` 默认 ON，PyQCU 对应开关
   默认 OFF。后果是本仓库默认产物绑定构建机 ISA（PyQCU 默认不绑定）；对 Kunpeng
   （aarch64）与跨平台分发，本仓库默认产物不可用（**风险判断，未在 Kunpeng 上复跑**）。
2. **稳定软链 + `mv -Tf` 的发布方式一致**（见 `cpp/kupl/einsum/AGENTS.md` §2），
   这部分没有漂移。
3. **产物命名**：本仓库为 `libpyqcdeinsum_kupl.so` / `libpyqcdeinsum_cuda.so`；
   PyQCU 侧命名本次未核对，不作结论。
4. 其余行为（线程绑定、BLAS 选择、构建缓存）本次未对照。

## 7. 产物可移植性问题（实测，未修复）

三个本机产物都带**指向 conda 前缀的绝对 RPATH**（conda 工具链默认注入）：

```bash
for f in cpp/kupl/einsum/libpyqcdeinsum_kupl.so \
         cpp/cuda/einsum/libpyqcdeinsum_cuda.so \
         pyqcd/einsum/cuda/einsum.cpython-312-x86_64-linux-gnu.so; do
  echo "--- $f"; readelf -d "$f" | grep -E 'RPATH|RUNPATH' || echo "    (none)"
done
```

实测结果：

| 产物 | RPATH |
|---|---|
| `libpyqcdeinsum_kupl.so` | `/home/zhangxin/miniforge3/lib` |
| `libpyqcdeinsum_cuda.so` | `/home/zhangxin/miniforge3/lib` |
| `einsum.cpython-312-x86_64-linux-gnu.so` | `/home/zhangxin/miniforge3/lib:$ORIGIN:/home/zhangxin/PyQCD/cpp/cuda/einsum` |

后果：产物把构建机的路径写死，换机/换前缀需 `patchelf --set-rpath` 重新处理
（**本次未执行**）。这也解释了为什么 C1/C5 的运行期 `dlopen` 能命中 conda 的
`libcblas.so.3` / `libopenblas.so.0`：调用方 DSO 自身的 RPATH 参与搜索。

## 8. 尚未支持（如实登记）

| 项 | 现状 | 依据 |
|---|---|---|
| **MPI CUDA** | 无 MPI 探测/编译分支，`applyContractMpiQcd` 为返回 1 的 stub | `cpp/cuda/einsum/make-nv.sh`、`cpp/cuda/einsum/src/apply_contract.cu:1749` |
| **DCU 后端** | 仓库无 `cpp/dcu`（`ls cpp` 只有 `cuda`、`kupl`），无源码/脚本/产物 | `ls cpp` |
| **跨架构分发** | KUPL 默认 `-march=native`；CUDA 默认只出 `6.0 7.0` SASS 且无 PTX；两库都带绝对 RPATH；没有 aarch64/Kunpeng 构建记录；没有 manylinux/wheel 发布流程（`setup.py` 只做本地 inplace 构建） | §1、§3、§7 |
| 多 GPU / 跨设备 collective | stream pool 只在单一当前设备内工作；跨设备需 MPI | `cpp/cuda/einsum/AGENTS.md` §5 |
| KUPL MPI 多轴分片 | 只支持单输出轴、不做跨 rank 浮点 allreduce | `cpp/kupl/einsum/AGENTS.md` §8 |

## 9. 未核实项（本次没做的事）

- C3/C4（MPI 开启 + `CXX=mpicxx` 或显式 include）、C7（sm_80）、C8（多架构）**未跑**。
- `cuobjdump` 不在本机 PATH，未逐架构复核两个 CUDA 产物的 fatbin；C6 的机制结论来自
  脚本行号（`make-nv.sh:20`、`146`）与实测现象，不是 fatbin 反查。
- `install.sh` / `setup.py` 本次**只读脚本，没有重跑**（`pyqcd/**` 同时有他人改动）。
  §5 中凡涉及它们的行为均来自脚本行号，不是执行记录。
- PyQCU 为工作树外仓库，本次未复跑；对照只覆盖 §6 的两项。
- 未在 Kunpeng/aarch64 上构建；未做 `patchelf` 重写 RPATH 的验证。
- 未记录任何性能数字——本文不涉及。
