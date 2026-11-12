# pyqcd.einsum 实测经验与陷阱（2026-11-08 会话沉淀）

> 全部结论来自当轮实跑，附复现方式。凡"必须/不要"都是踩过坑后的硬约束。

## 1. 构建

| 事实 | 说明 |
|---|---|
| **CUDA 必须显式指定架构** | `make-nv.sh` 默认 `6.0 7.0` 且**不生成 PTX**，在 sm_89（RTX 4060）上无法运行。必须 `PYQCD_CUDA_ARCHITECTURES=8.9 bash build-nv.sh`。 |
| **MPI 构建已整体删除（2026-11-12）** | `PYQCD_EINSUM_ENABLE_MPI`、`PYQCD_EINSUM_MPI_LINK_FLAGS`、`mpi_runtime.cpp`、`einsum_mpi.h`、`applyContractMpiQcd` 与 `kupl_mpi_enabled()` 全部移除；`make.sh` 不再探测 `mpicxx`。历史上 conda 固定 `CXX=g++` 导致 `mpi.h: No such file` 的坑因此不再存在，不要再按旧文档设置这些变量。 |
| **同名头文件冲突（重要）** | `cpp/kupl/einsum/include/` 与 `cpp/cuda/einsum/include/` 下曾有同名 `nucleon2pt.h`、`pattern_3pt.h`（两者已于 2026-11-12 删除），而 `make-nv.sh` 把 **KUPL 的 include 放前面** → CUDA 源文件会包含错头，报"标识符未定义"。新增同名头文件时仍要用相对路径 `#include "../include/xxx.h"`。 |
| **Cython 生成物不必入库** | `pyqcd/einsum/cuda/einsum.cpp`（约 1 万行）是 `einsum.pyx` 的生成物；删除后 `install.sh cuda` 仍可重建扩展（实测 192 KB `.so`）。 |
| **OpenBLAS 探测提示会误导** | `make.sh` 打印 `OpenBLAS: not found` 属探测失败，但运行期是 dlopen `libopenblas.so.0`，数值路径正常。 |
| 导出面 | `-Wl,--version-script` 收敛；新增 kernel 符号必须**显式**加入 `make.sh` / `make-nv.sh` 的 map，否则 `nm -D` 看不到、Python 侧取不到。 |

## 2. 测量口径（最容易出错的地方）

| 陷阱 | 后果 | 正确做法 |
|---|---|---|
| **预热不足** | CUDA 前 3–4 次调用因显存池增长慢 3–4 倍（实测 4.16/2.95/3.57 s → 稳态 1.095 s），`warmup=2` 会把中位数拖到非稳态 | 默认 `MIN_WARMUP = 4`；报告里写明预热轮数 |
| **对照的是自己** | 验收工具若用 `contract(..., optimize=...)` 当 "generic"，一旦接线就变成 kernel 比 kernel | "generic" 必须显式 `specialize=False` |
| **测的不是生产库** | 用 nvcc 现编独立 `.so`（缺 `-fvisibility=hidden` / `-lcublas`）比生产库差 **1.8x** | 验收工具优先复用 `cpp/cuda/einsum/build/libpyqcdeinsum_cuda.so` |
| **并发干扰** | 后台代理跑基准时同配置多次测量差 **3.7x**（37.65/75.97/138.23 ms） | 正式数字必须在无并发构建/基准时采集，并记录 `load average` |
| **逐元素相对误差** | 抵消元素上 `|ref|≈0` 会放大到 5.7e-12，误判"超契约" | 契约用**范数口径** `max\|Δ\|/max\|ref\|`；逐元素值只作诊断 |
| **opt_einsum 不走 BLAS** | E2 在 T1 单次 2.18 s（我们 0.56 s）→ 38x 倍率是**它的问题**，不能算我们的优势 | 必须同时给绝对耗时；倍率单独看没有意义 |
| **规划器无红利** | E1 的 `greedy/dp/branch-2/auto` 代价完全相同（19,204,160,000，差 <0.002%） | 别把希望放在换 planner 上 |
| **短任务不算对照** | 双方都 <100 ms 的单元格受调度噪声支配 | `MIN_MEANINGFUL_SECONDS = 0.1`，低于即标 `too_short`，不参与倍率判定 |
| **只测 1–2 算子漏掉灾难性退化** | direct 模式在两算子上"持平/慢 10×"，三算子链慢 **3900×**、四算子链 10 分钟未返回（0.2.5/0.2.6 实测，已回滚） | 任何执行路径改动的基准必须覆盖 2/3/4 算子递增形态 |
| **"跳过规划"当加速项** | `backend.einsum(optimize=False)` 是朴素 n 元循环：不调 BLAS、不切中间体；省下的规划开销（31–84%）远小于丢掉 BLAS 的代价 | 要省规划用计划缓存（0.2.4 已做），不要绕过规划器 |
| **公平性红线** | 多卡/多进程 vs 单进程 opt_einsum 是欺诈 | 单 GPU 对单 GPU、同线程、同 dtype、同 planner |

## 3. 正确性陷阱

| 缺陷 | 症状 | 根因与修法 |
|---|---|---|
| **输出全 1 维时未写结果** | `abcde,cedb->ac` 返回 5e-310、`rel=1.00`；命中 E2 的 `greedy/auto` | `transpose_copy()` 里 `extent == 1` 的轴被全部跳过后 `axes.empty()` 直接 `return true`。修：校验 `source/destination_count == 1` 后 `memcpy` 单元素 |
| **设备 kernel 缺连续性校验** | 非连续 cupy 输入仍"命中"，`rel=1.69` 静默算错 | 设备端不做 stride 解释；每个 kernel 绑定必须校验 `flags['C_CONTIGUOUS']`（或对大张量回退、小张量显式拷贝） |
| **kernel 异常抛给调用方** | 契约要求"原生失败 → warning 一次 → 回退"，实测直接抛异常 | 分派必须包在 `try/except` 里，`warn` 一次后走通用路径 |
| **开关未进缓存键** | 先 `specialize=False` 再切 `True` 不生效（CUDA E4 1.376→1.159 ms，而非 ~0.68 ms） | 原生程序缓存键必须包含"是否禁用融合"；`_contract_signature` 与 `contract_expression` 的 `native_cache` 键同样处理 |
| **环境变量被自己抹掉** | `PYQCD_EINSUM_SPECIALIZE=0` 不生效 | `contract` 的 `specialize` 默认值必须是 `None`（让 env 决定），且只回收"本模块自己设置"的 `FUSED_DISABLE` |
| **专用路径在大规模退化** | E4 融合 kernel 在 `z·y·x=13824` 时 36.4 s vs 通用 1.1 s（慢 33 倍）并可 SIGSEGV | 融合 kernel 必须带**收益判据**（E4 实测仅 `z·y·x==1` 占优，阈值 `kE4FusedMaxSpatialVolume`） |

## 4. 专用 kernel 接线流程（按此顺序，缺一步就会埋雷）

1. **导出符号**：加入对应 `make.sh` / `make-nv.sh` 的 version script。
2. **绑定层**（`pyqcd/einsum/_kernel_*.py`）：
   - 严格形态校验（下标规范化后逐字匹配 + 每个操作数形状/dtype/连续性）；
   - 契约细节必须逐个对齐头文件：KUPL 与 CUDA 的 `dims` 布局不同
     （例：KUPL 3pt `[nvec,mom_src,mom_sink,link,dirac,proj]`；CUDA 3pt `[nev,mom,link,4,2]`；
     CUDA 2pt 是 **10 项**并含 perm 码，恒等=123）、算子顺序也可能不同；
   - **CUDA 要设备指针**（`cupy` → `data.ptr`），KUPL 是主机指针。
3. **分派**（`_api.contract`）：仅在 `out is None`、`specialize` 生效时尝试；异常 → warning 一次 → 回退。
4. **验收**：`kernel_acceptance` / `check_kernels` 两个脚本随核子 2pt/3pt kernel
   一并删除（2026-11-12）；新入口改用 `python -m pyqcd.testing._topology_kernel_contract`
   与 `verify_native` 双后端 + `verify_compat` + `python -m pyqcd.testing`。

## 5. 已知限制（不要声称已支持）

| 项 | 现状 |
|---|---|
| `contract_expression` | 数值正确，但**不走专用 kernel**（实测调用计数 0 vs `contract` 的 1） |
| MPI | 只有 KUPL 有真实实现（单输出轴分片 + `Allgatherv`，无跨 rank 归约）；**CUDA MPI 是 fail-closed stub**；默认构建是非 MPI（用 `pe.build_info()['mpi_label']` 查询） |
| E4 融合 | 仅 `z·y·x==1` 启用，L24 回落通用（1.016x 持平） |
| 2pt/3pt CUDA kernel | 已接线但标 `needs-optimization`（收益随通用路径波动在 0.86–1.88x） |

## 6. 测试资产位置

| 入口 | 用途 |
|---|---|
| `pyqcd/testing/einsum/verify_native.py` | 原生四 dtype/chain/batch/unary/dot/sandwich/缓存 |
| `pyqcd/testing/einsum/verify_compat.py` | 与 opt_einsum 的 980 项差分（**需要 pytest**；子模块不可用时设 `PYQCD_EINSUM_ORACLE=installed`） |
| `pyqcd/testing/einsum/bench_native.py` | 严格基准（线程钉扎 + 进程隔离；必须经它启动，否则线程口径无效） |
| `pyqcd/testing/einsum/simple/` | 一键套件（compat/memory/equations/native/kernels）+ `workflow.py`（生产时间占比模型） |
| `pyqcd/testing/simple/` | 逐函数 API 对照（`contract`/`contract_path`/`contract_expression`/helpers/...），**参考实现为 `torch.einsum`** |

## 7. 环境相关

- `torch` 缺失会让 `pyqcd.lattice.gamma` 直接 ImportError（`pyqcd/tools/_torch_backend.py` 顶层 import）。
- `verify_compat` 依赖 `opt_einsum.testing`，后者 import pytest；缺 pytest 会整套不可用。
- 退出期 SIGSEGV 的根因是**映射期间就地截断 `.so`**（原子替换 0/100 崩、就地截断 50/50 崩）；
  `setup.py::_stage_native_library()` 已改为"临时文件 + `os.replace`"。**不要在进程运行时用 `cp`/`shutil.copy2` 覆盖已加载的 `.so`**。

## 8. 拓扑特化内核陷阱表（2026-11-11，全部本机实跑）

机器：本机 Linux / RTX 4060 Laptop `sm_89` / OpenBLAS（`libopenblas.so.0` dlopen）/
CuPy 14.2.0。复跑命令均以 `python -B` 起，避免 `.pyc` 混入证据。

| # | 现象 | 根因（证据） | 判据 | 正确做法 |
|---|---|---|---|---|
| T1 | 声称"2582 条拓扑已有特化内核"，实测每个方程耗时与通用路径同量级（1.0–1.5x） | 只完成了**匹配 + 转发**：`_specialize.match()` 命中后由 `_api` 分派到 `qcd_kupl_topology_apply`，该入口在 `84e68bb` 里只做 `parse_execution_plan` 再 `execute_parsed_contract`（`cpp/kupl/einsum/src/topology_apply.cpp:107-127`，提交版同逻辑）；CUDA 同构转发 `execute_prepared`（`cpp/cuda/einsum/src/apply_contract.cu:1992`）。匹配成功 ≠ 有专用数值内核 | 只认**执行**证据：`[kernel: kupl_topology]` 标签 + `_kernel_kupl.LAST_INFO['path']=='kupl_topology'`；若要确认走了**特化数值**路径，必须另有 trace（如 `PYQCD_EINSUM_TOPOLOGY_TRACE=1` 的 `topology-specialized:` 行）。仅看见 `2582/2582 命中` 不能证明实现 | 把"匹配层"与"执行层"分成两条验收线：匹配层用命中率/误命中率；执行层用原生 trace 或符号级证据。任何"已实现特化"的结论必须附执行层证据 |
| T2 | 生成 2582 个 per-拓扑"静态展开"内核，dim=4 时不可用 | 生成器曾把每条方程展开成**每符号一层**的嵌套循环（`for i_x … acc += p0[…] * p1[…] * …`）。每式循环深度 = 全式不同符号数，实测：元数 6→8 层、7/9→12 层、10/12→**16 层**、13→**20 层**、14/16→**24 层**；dim=4 时 4^20=1.1e12、4^24=2.8e14 次乘加。该形态已在本轮被删除，现行 `pyqcd/testing/_topology_kernel_gen.py` 只生成步表（`kTopoCode` 字节流 + `topo_NNNN` 单行入口，`_topology_kernel_gen.py:226-295`），文件从 13 MB 降到 2.5 MB | 生成内核规模按 `dim^符号数` 估：任一方程"全式不同符号数 > 16"即朴素循环不可用。可复跑：统计每式符号数（见 8.1），或 `grep -c 'for (int i_' topology_kernels.inc`（旧形态每条 8–24 个循环头） | 走**步表 + 参数化 GEMM 执行器**：`contract_path(optimize='greedy')` 把每条拓扑固化成若干 arity≤2 步，每步 `pack → cblas gemm → unpack`（`cpp/kupl/einsum/src/topology_gemm.h`）。绝不为一条 16 算子方程写 24 层循环 |
| T3 | "CUDA 拓扑入口 dim≥8 结果不正确（dim=4 对）" | **当前树无法复现**。实测 `arity∈{6,7,9,10,12,13,14,16}`、`dim∈{4,8}` 共 32 组全部 `rel ≤ 4.1e-7`、`correct=True`；`dim=12` 亦正确（arity13 `rel=5.8e-7`、arity16 `rel=2.3e-7`），交替 6 条拓扑 × 3 轮也 0 失败。构建产物 `pyqcd/einsum/cuda/libpyqcdeinsum_cuda.so` 已导出 `qcd_cuda_topology_apply` | 复跑：若 `correct` 判据是"与 numpy 逐元素比"，高元数 `dim≥8` 时**参考侧本身**会先超时（`np.einsum` 在 16 算子 × 8^24 上不可能算完），容易把"参考算不动"误报成"kernel 错" | 正确性门必须用**同 dtype 的通用路径**当参考（`contract(..., specialize=False)`），不要在高动量大形状上直接对 numpy。若要在 dim≥8 断言 CUDA 错误，需给出具体方程、形状与两端数值 |
| T4 | 特化分支"看起来在跑"但结果始终是通用执行器给的 | `build_canonical_signature` 用 `step.positions` 当**输入下标**索引 `labels[position]`，而 `positions` 语义是**当前活动槽**（`cpp/kupl/einsum/src/runtime.cpp:235-243` 只校验 ≥0；执行器按 `active[...]` 取用，`apply_contract.cpp:1793-1801`）。第 5 个步之后活动槽里已有中间体，于是同一槽被读入两种 label 向量 → 签名构建失败 | 在临时副本里插桩后实测输出 `[dbg] sig: slot conflict at 4: had 4 [20,21,22,23,] now 2 [8,12,]`，随即 `miss: signature build failed`，`rc=0`（回落到通用路径仍给出正确数值，所以纯正确性测试**抓不到**） | 签名必须由 `execution.input_shapes` 顺序 + 各 input 的原始 label 构建，或用"计划里首次出现"重编号后与生成期口径对齐；不要把 `positions` 当输入序号。验证时必须同时看 trace 命中率，不能只看 `rc==0` |
| T5 | 步表执行器"未接线"却以为已生效 | `topology_kernels.inc` 是 `.gitignore:8` 的 `*` 通配忽略项（生成物不进库），且 `make.sh` 只 glob `src/*.cpp`（`cpp/kupl/einsum/make.sh:400-413`），**不编译 `.inc`**；只有 `topology_apply.cpp` 显式 `#include "topology_kernels.inc"`（`topology_apply.cpp:17`）才会进构建 | `nm -D` 只有 `qcd_kupl_topology_apply`；`strings libpyqcdeinsum_kupl.so \| grep -c topology-specialized` → 生产库 **0**、而临时重建副本 **1** | 改生成器/执行器后必须重建并检查符号内字符串；`git status` 看不到 `.inc` 变化不代表没改。生成物不入库，全靠 `make.sh` 时的 `#include` 与本地生成步骤 |
| T6 | 语料全扫描时匹配开销远高于单条基准 | `_signature_of_normalized` 的 `lru_cache(maxsize=512)`（`pyqcd/einsum/_specialize.py:151-153`）挡不住 2582 条轮转。实测：单条重复 3.45 µs、100 条轮转 5.02 µs、**2582 条轮转 16.0 µs**，`cache_info()` 显示 `currsize=512` 持续抖动 | 报 `match` 开销必须写清**工作集**；只报"单条 µs"会低估真实扫描成本 3–5 倍 | 基准分三档（单条 / 100 条 / 全语料）；若要做语料级吞吐，先加大缓存或改用 `frozenset` 摘要直查，不要沿用 512 项 LRU 的乐观数字 |
| T7 | dtype 能力"文档说四个都支持"与 Python 侧实际下发的集合不一致 | `TOPOLOGY_DTYPES` 只声明 `(complex64, float32)`（`_kernel_kupl.py:275`、`_kernel_cuda.py:269`）；双精度要经 `_probe_double` 实测才加入（`_kernel_kupl.py:291`）。KUPL 四 dtype 实测全通（`float64 rel=4.5e-16`、`complex128 rel=6.3e-16`），CUDA 探测后也全通 | `topology_dtypes()` 在探测前后不同：探前只有 2 项，探后 4 项 | 写"支持 double"前先跑 `_kernel_kupl.topology_dtypes()`；契约要求 double **绝不降级**，未实现时应返回 `None` 回落而不是转 float32 |

### 8.1 复现命令（逐条对应上表）

```bash
# T1/T5：确认原生入口存在且看是否真的走特化数值路径
nm -D --defined-only pyqcd/einsum/kupl/libpyqcdeinsum_kupl.so | grep topology
strings pyqcd/einsum/kupl/libpyqcdeinsum_kupl.so | grep -c topology-specialized   # 未接步表时 0

# T2：每式循环深度 / 是否已改成步表
head -20 cpp/kupl/einsum/src/topology_kernels.inc
grep -c '^static int topo_' cpp/kupl/einsum/src/topology_kernels.inc
python -B -c "
import collections
from pyqcd.testing._topology_equations_contract import _parse_equations as P
d=collections.defaultdict(list)
for e in P():
    lhs,out=e.split('->'); syms=[]
    for t in lhs.split(','):
        for c in t:
            if c not in syms: syms.append(c)
    for c in out:
        if c not in syms: syms.append(c)
    d[lhs.count(',')+1].append(len(syms))
for ar in sorted(d):
    print(f'arity={ar:3d} n={len(d[ar]):5d} depth={max(d[ar])} 4^depth={4**max(d[ar]):.2e}')
"

# T3/T7：CUDA 与 dtype 能力（参考必须用 specialize=False）
python -B -c "from pyqcd.einsum import _kernel_cuda; print(_kernel_cuda.topology_dtypes())"

# T6：三档匹配开销
python -B -c "
from pyqcd.einsum import _specialize as S
from pyqcd.testing._topology_equations_contract import _parse_equations as P
import time
eqs=P()
for n,reps in ((1,20000),(100,100),(2582,4)):
    seq=eqs[:n]; S.match(seq[0]); t=time.perf_counter()
    for _ in range(reps):
        for e in seq: S.match(e)
    print(n, 'x', reps, '-> %.2f us/call' % ((time.perf_counter()-t)/(n*reps)*1e6))
"
```

### 8.2 本轮明确**未**验证、因此不得写入结论的项

| 项 | 状态 |
|---|---|
| 步表执行器的数值正确性 | 只验证到"能编译 + 会进入 `run_specialized`"，未跑通 2582 条对照；`build_canonical_signature` 当前必失败（T4），因此步表分支实际从未执行 |
| 拓扑 kernel 相对通用路径的加速比 | 本机实测 1.02–1.50x（float32，dim=4/8，采样 4 个元数）；**不是**任务书里的 ">90 s 断崖"，也不是"已达标"。断崖说法对应 `logs/v202611112059.txt`/`v202611112110.txt` 的旧构建，本轮未复现 |
| CUDA 步表/GEMM 重写 | `cpp/cuda/**` 本轮无改动；只有既有 `qcd_cuda_topology_apply` 转发路径 |
| `dim≥8 CUDA 错数` | 32 组 + dim=12 抽查全部正确；无可复现的错误用例 |

## 9. [K2] 全量固定步表（8638 条式子）：生成、执行与对照口径

> 2026-11-11 实跑。入口 `python -m pyqcd.testing._dedicated_kernel_batch`；生成物
> `pyqcd/einsum/_dedicated_step_tables.py`；任务表 `pyqcd/einsum/_wick_tasklist.py`。
> 采集时机器 load average 11–15（多代理并发），耗时类数字标了重复测量的范围。

### 9.1 规模、编码与生成口径

| 事实 | 实测值 |
|---|---|
| 式子条数 | 8638（基础表 2582 + 附录 6056），**全部唯一**，行号严格递增 |
| 元数 / 输出秩 | arity 6–16；输出 rank 0–6（32 条标量输出）；最多 27 个不同指标字母 |
| 固化步数 | 121724 步，最多 **15 步/条**，池位置槽号 ≤ 14 |
| 步表编码 | 每行 `<左槽>-<右槽>:<保留指标>;…`，槽号沿用 `contract_path` 线性路径语义（取出两个槽、先删大号再删小号、结果追加池尾） |
| 对齐指纹 | 首行 `!8638:4fb8bc94` 是 8638 条式子串联的 crc32，`verify_alignment()` 每次批量验证前比对，不一致直接退出 2 |
| 文件体量 | 1.15 MB（单字符串块） |
| 运行期开销 | 追加 import **1.1 ms**，`splitlines` 1.1 ms，首次解码全部 8638 条 **71.9 ms** |
| 生成期口径 | **实测 8638/8638 与 `contract_path(optimize="greedy")`（探针 dim=3、external=2）逐步一致** |

**关键事实：greedy 路径依赖探针形状。** 同一批 8638 条式子逐步比对：

| 探针形状 | 与步表一致条数 |
|---|---|
| dim=3, external=2 | **8638 / 8638** |
| dim=2, external=2 | 6262 / 8638（基础表 2482/2582，差 104 条） |
| dim=4, external=4 | 6002 / 8638 |

所以"greedy 结果与维度无关、随便用什么探针尺寸都一样"是**错的**：pyqcd 的 greedy 带
memory/尺寸判据，dim=2 与 dim=3 在 6 夸克类方程上会给出不同的配对顺序。K1 的
`_topology_kernel_gen` 用 `PROBE_DIM=2`，其 2582 条基础式里 104 条的配对顺序与本表不同；
两者都是合法收缩，数值验证都通过。**固化步表时必须把探针形状写进文档**，否则无法复现。

### 9.2 最大的坑：`np.einsum(optimize="greedy")` 不能当全量参考

numpy 自己的 greedy 规划会把**末尾若干步合并成一个多操作数 `c_einsum` 调用**，
合并步按朴素多操作数 einsum 执行，工作集是全部指标之积：

| 观测 | 数值 |
|---|---|
| numpy 路径全是二元步的式子 | 仅 **844 / 8638** |
| 合并步最宽操作数个数 | 4–12 |
| 合并步朴素工作集 中位数 / p90 / 最大 | 8.6e7 / 1.4e10 / **2.79e10** |
| 典型卡死例 | `aibj,cqdr,ewfx,munv,okpl,gsht,ce,og,ik,su,bdf,nph,jlr,tvx->aqwm`：numpy 把 10 个操作数并成 `gklp,mnsv,deqr,abjk,tvx,jlr,nph,bdf,gsht,ewfx->aqwm`，**单条 >90 s 不返回**（3 次 90 s 超时实测） |

修法：对照必须走**显式路径**，把步表反解成 numpy 的路径表：

```python
np.einsum(eq, *ops, optimize=["einsum_path", *pairs])   # 首元素必须是字符串 'einsum_path'
```

`pairs` 是"当前操作数池"里的 `(i, j)`（与本表槽位语义一致）。若仍要跑 numpy 独立 greedy，
先 `np.einsum_path(..., einsum_call=True)` 估算合并步工作集，超门限就跳过并计数。
要**全覆盖**独立规划，用"greedy 路径分步"口径：沿用 numpy 的 greedy 路径，把合并步按
"中间张量最小"拆成两两 `np.einsum`，实测约 1 ms/条，8638 条全跑得动。

### 9.3 三条对照口径（`_dedicated_kernel_batch`）

| 口径 | 含义 | 覆盖 |
|---|---|---|
| A | `np.einsum(eq, *ops, optimize=["einsum_path", *pairs])`：numpy 内核执行**本步表**的路径 | 100% |
| B | `np.einsum(eq, *ops, optimize="greedy")`：numpy 自己的 greedy 路径整体执行 | 3422/8638（余下 5216 条合并步工作集 >1e7） |
| C | numpy greedy **路径**逐步执行、合并步两两拆分 | 100% |

### 9.4 全量验证结果与成本（dim=3，external=2，float64）

| 范围 | 通过 | 墙钟 | 每条 | 峰值 RSS |
|---|---|---|---|---|
| `--limit 50` | 50/50 | 0.08–0.10 s | 1.6–2.1 ms | 68.8 MiB |
| `--limit 500` | 500/500 | 1.83–2.54 s | 3.7–5.1 ms | 69.5–69.9 MiB |
| 全量 8638 | **8638/8638** | 71.8 s（A+B）/ 85.9 s（A+B+C，load≈14） | 8.3 / 9.9 ms | 78.3 MiB（入口 67.8） |

- 偏差：A 最大 **1.65e-12**、B **2.57e-13**、C **2.56e-14**；另抽 50 条对
  `opt_einsum.contract(optimize='greedy')` 最大 **1.45e-15**；`--dtype float32` 抽 20 条 **1.69e-6**。
- **执行器本身只占 2.2 s（0.26 ms/条，p99 4.2 ms）**，其余 ~97% 是 numpy 对照。
- 内存解析上界：任一步保留指标串在 dim=3/external=2 下最大 **8748 元素**，不可能 OOM。

| 情形 | 条数 | 内核均值 | 参考均值 |
|---|---:|---:|---:|
| 基础/1、2、5、6、9（介子·重子 2pt/3pt/4pt） | 62 | 0.04–0.09 ms | 0.29–0.91 ms |
| 基础/3、10、13（五夸克 2pt / 重子 4pt / 五→三 3pt） | 360 | 0.13–0.16 ms | 1.2–1.4 ms |
| 基础/4、7、14（六夸克 2pt / 五夸克 3pt / 五→三 4pt） | 2160 | 0.13–0.24 ms | 1.4–3.7 ms |
| 动量/*（附录 19 个子表） | 6056 | 0.28 ms | 9.8 ms |

### 9.5 外推的口径陷阱

用前 500 条（3.66–5.07 ms/条）外推剩余 8138 条得 ≈30–41 s，**实测全量 71.8 / 85.9 s，
低估 2.4–2.9×**。
原因见上表：附录 6056 条的参考对比均值 9.8 ms，是前 500 条的 ~2 倍。
结论：前缀外推只可用于同段；跨情形必须按情形加权，或直接全量跑（全量也只要 ~80 s）。

### 9.6 交叉核对

| 对照 | 结果 |
|---|---|
| `contract_path(greedy, dim=3/ext=2)` 逐步反解 | **8638/8638 一致**（证明步表就是 greedy 的固化） |
| K1 `_topology_kernel_gen.plan_steps`（2582 条基础式，探针 dim=2） | 配对+保留指标**集合** 2478/2582 一致，104 条因探针维度不同而路径不同（都合法） |
| `opt_einsum.contract(optimize='greedy')`（50 条跨全部情形） | 最大 1.45e-15 |

### 9.7 环境陷阱（本轮踩到）

| 陷阱 | 现象 | 修法 |
|---|---|---|
| 从 `/tmp` 跑脚本 | `ModuleNotFoundError`（`sys.path[0]` 是脚本目录，import 到 site-packages 里的旧 pyqcd） | `export PYTHONPATH=<repo>` |
| `pkill -f probe.py` | 把执行该命令的 `sh -c` 自己杀掉（命令行里含同样字符串），返回 143、无输出 | 用 `pkill -f "[p]robe.py"` |
| 管道里取退出码 | `python ... \| head` 后 `$?` 是 `head` 的，失败被吞 | 先重定向到文件再取 `$?` |
| numpy 的 C 内核不响应信号 | `signal.setitimer` 超时只能打断 Python 侧循环，打不断 `c_einsum` | 用工作集预算**事前**拦，而不是超时事后杀 |
| 多代理共享机器 | load 12–15 时同代码全量 71.8 s → 85.9 s，import 0.12 s → 2.1 s | 耗时数字标注负载；性能结论不要在这种机器上下 |

## 10. [K2] 批量生成 + 全量验证流水线（8638 条，当前磁盘状态实测）

> 2026-11-11/12 实跑。入口 `python -m pyqcd.testing._dedicated_kernel_batch`；
> 数据 `pyqcd/einsum/_wick_tasklist.py` + `pyqcd/einsum/_dedicated_step_tables.py`。
> 本节记录的是**当前磁盘上**的十进制槽位编码（`<左>-<右>:<保留指标>`，`;` 连接），
> 与第 9 节的 base-36 单字符编码不是同一版产物，两节数字不可混用。

### 10.1 生成物与 import 成本

| 项 | 实测 |
|---|---|
| 任务表 | 8638 条（基础 2582 + 附录 6056），块 711646 B，模块 717 KB，角色表仅 11 条 |
| 步骤表 | 121724 步（= Σ(arity-1)），最多 15 步/条，块 1150340 B，模块 1.15 MB |
| 编码 | 任务行 `行号|T号|式子|角色号`，表头 `@情形|子表`；步表 `左-右:保留` |
| import 增益 | `pyqcd.einsum` 本身 146–159 ms；任务表 **+1.8–2.7 ms**，步表 **+1.1–1.6 ms** |
| 解析 | `tasks()` 9.5 ms；步表 `splitlines` 0.8 ms；8638 条步表全解码 42 ms |

- 数据写成**单块字符串**即可，import 增益是毫秒级；写成 8638 个字面量才会拖慢。
- 真正的 import 大头是 `pyqcd.einsum` 自己的 native/kupl 探测（≈150 ms），与数据量无关
  ——引用 import 成本时必须把这两部分分开说，否则会把 150 ms 记到数据模块头上。

### 10.2 三个坑（本轮都踩过）

| 坑 | 现象 | 修法 |
|---|---|---|
| `np.einsum(optimize='greedy')` 不能直接当全量参考 | numpy 把末尾若干步并成一个多元 `c_einsum`，朴素工作集实测最大 **2.79e10**；单条 50 s 不返回 | 改用 `np.einsum_path(..., optimize='greedy', einsum_call=True)` 的 `contraction_list` 分步执行；多元步内部再按"中间张量最小"两两拆 |
| 误读 `einsum_call=True` 的返回值 | 它**不是** `(path, contraction_list)`：第一项是解析后的操作数（ndarray 列表）。当路径用会抛 `TypeError: only integer scalar arrays can be converted to a scalar index`，而且报错信息里带一大堆数组，容易误判成内存问题 | 路径在 `contraction_list[i][0]`（每步消耗的操作数下标），自己取 |
| 用前 500 条外推全量 | 500 条 4.68 ms/条 → 外推 8138 条 ≈38 s，**实测全量 86.5 s**，低估 2.3× | 按子表加权；本机全量也只要 ~90 s，直接跑全量 |

### 10.3 实测结果（dim=3，external=2，float64，seed=0）

| 范围 | 通过 | 墙钟 | ms/条 | 峰值 RSS（入口） |
|---|---|---|---|---|
| `--limit 50` | 50/50 | 0.119 s | 2.38 | 69.1 MiB（68.0） |
| `--limit 500` | 500/500 | 2.340 s | 4.68 | 69.7 MiB（67.8） |
| 全量 8638 | **8638/8638** | **86.466 s** | 10.01 | **72.4 MiB**（67.5）；`/usr/bin/time` 74124 KB |

- **执行器不是瓶颈**：内核总计 2.56 s（均值 0.296 ms/条，p50 0.154 ms，p99 4.30 ms，
  最大 49.5 ms），其余 97% 是 numpy 对照。
- 偏差上限：同路径 A **1.65e-12**；numpy 独立 greedy B（3422 条工作集允许）**2.57e-13**；
  greedy 路径分步 C（**100% 覆盖**）**2.56e-14**；`opt_einsum.contract(greedy)`
  抽 74 条跨全部 19 个子表 **1.33e-15**。
- `--dtype float32` 抽 20 条：最大 6.4e-6，仍过 1e-4。
- **内存有解析上界**：任何一步的保留指标串在 dim=3/external=2 下最大 **8748 元素**
  （float64 = 68 KiB），OOM 风险为零；分段跑 5×500 条峰值 RSS 稳定在 69 MiB 左右，
  与条数无关；全量只比入口高 4.9 MiB。

每千条耗时随子表强烈波动（实测 2.8–21.1 s/千条，这是外推必须分段的直接证据）：

| 索引区段 | 条数 | 内核 s | 参考 s | 合计 s（每千条） |
|---|---:|---:|---:|---:|
| 0–999 | 1000 | 0.27 | 4.48 | 4.75 |
| 1000–1999 | 1000 | 0.24 | 2.58 | 2.82 |
| 2000–2999 | 1000 | 0.42 | 9.75 | 10.17 |
| 3000–3999 | 1000 | 0.49 | 20.56 | 21.05 |
| 4000–4999 | 1000 | 0.24 | 8.41 | 8.64 |
| 5000–5999 | 1000 | 0.30 | 17.04 | 17.34 |
| 6000–6999 | 1000 | 0.27 | 4.43 | 4.70 |
| 7000–7999 | 1000 | 0.18 | 12.36 | 12.54 |
| 8000–8637 | 638 | 0.15 | 1.91 | 3.24（2.07 s） |

- 参考成本高的区段（3000–3999、5000–5999）是附录里 16 项、带 `M/N/G/H` 外轴的式子：
  numpy 独立 greedy 在允许工作集内仍要 50–260 ms/条。要压总时间只能减参考，不能优化内核。
- `--greedy-max-elements 1e7` 下 B 只跑 3422/8638 条（其余 5216 条工作集最大 2.79e10），
  但 C 覆盖 100%，所以"每条都有独立 greedy 对照"这个口径成立。

### 10.4 复现与自检

```bash
cd ~/PyQCD
python -m pyqcd.einsum._wick_tasklist            # 8638 条、字段缺失 0、式子唯一 8638
python -m pyqcd.einsum._dedicated_step_tables    # 对齐 True + 前 20 条数值自检
python -m pyqcd.testing._dedicated_kernel_batch --limit 50
python -m pyqcd.testing._dedicated_kernel_batch --limit 500
python -m pyqcd.testing._dedicated_kernel_batch                      # 全量 8638
python -m pyqcd.testing._dedicated_kernel_batch --start 4000 --limit 500 --json /tmp/a.json
python -m pyqcd.testing._dedicated_kernel_batch --no-greedy-decomposed  # 关掉 C 对照
```

- 步表与任务表的对齐靠块首 `!8638:<crc32>`：式子串联的 crc32 不一致即 `verify_alignment()`
  返回 False，入口会直接退出 2，不会带病跑。
- 仓库全量测试 `python -B -m pyqcd.testing`：**127 passed, 1 skipped, 0 failed**（含 K1 的
  拓扑契约测试），本流水线新增的两个数据模块不参与 import，不拖慢该入口。

## 11. [P3] dedicated 步表生成端：dim=8「超时」其实是**参考实现**爆炸，不是内核

2026-11-11 复核 `qcd_kupl_dedicated_apply`（`pyqcd/einsum/_kernel_kupl.py`）。
症状转述为「dim=4 正确 5.77e-06、dim=8 超时 >60 s」，实测结论相反：**原生入口
在所有维度都是毫秒级，超时全部来自 Python 侧的对照实现**。先量再改。

### 11.1 实测：原生入口的 dim 无关性

方程 `cwdx,eufv,msnt,okpl,igjh,qarb,ac,mo,Ggi,Hkq,su,bd,fnp,Mhj,Mlr,tvx->MGHew`
（16 算子 / 15 步 / 14 中间槽，小写=dim、大写=external=2）：

| dim | 步数 | 槽位 | plan 字节 | max batch | max 中间体（元素） | 原生耗时 | 偏差 |
|---|---|---|---|---|---|---|---|
| 2 | 15 | 14 | 2764 | 2 | 32 | 1.7 ms | 3.61e-08 |
| 3 | 15 | 14 | 2764 | 2 | 243 | 0.88 ms | 6.92e-08 |
| 4 | 15 | 14 | 2796 | 2 | 1024 | 1.1 ms | 5.61e-07 |
| 5 | 15 | 14 | 2796 | 2 | 3125 | 1.0 ms | 3.00e-07 |
| 6 | 15 | 14 | 2796 | 2 | 7776 | 1.0 ms | 3.20e-07 |
| 8 | 15 | 14 | 2796 | 2 | 32768 | 1.4 ms | 4.63e-07 |

- 步表**结构**（槽位号、各 rank、batch_rank/m_rank/k_rank、三张轴序表）对
  dim≥4 逐字节相同，只有 `batch/M/N/K` 这 4 个 int64 元素数随轴长变；dim=2/3
  因为 greedy 选路不同才换了路径。**没有任何 `dim**k` 上界、没有按最坏 rank
  写死的轴序表、没有 dim=4 硬编码的槽大小。**
- 全语料 2582 条：`max_intermediate` 只有 16384（dim=4）/ 2097152（dim=8），
  比值 128 = 8^7/4^7 的正常量级；结构自检（A/B 乘积 = 槽元素数、轴序表长度 =
  rank、轴序是排列、C 乘积 = 目的槽）**0 处违规**。

### 11.2 真正的 60 秒在哪：两条对照各自爆炸

| dim | 步表 max 中间体 | numpy 独立 greedy 工作集 | `_dedicated_kernel_001.kernel` 峰值 |
|---|---|---|---|
| 2 | 32 | 128 | 1.7e7（64 MiB） |
| 3 | 243 | 3.8e7 | 2.8e11（>1 TiB，实测 OOM） |
| 4 | 1024 | 2.1e9 | 2.8e14 |
| 8 | 32768 | 3.5e13 | 4.7e21 |

- `np.einsum(eq, optimize='greedy')` 实测**dim=4 已 20.3 s**、dim=5 超 90 s；
  `_dedicated_kernel_001.kernel` 在 dim=3 就 `Unable to allocate 1.03 TiB`。
  这两条都不是内核路径，用它们当基准必然得出"dim=8 超时"的假象。
- 可信对照只有两种：① `optimize=['einsum_path', *plan_pairs]`（用步表自己的
  配对顺序喂 numpy，工作集与内核一致）；② 更小的 probe dim。
  `_dedicated_kernel_batch` 的 A/C 对照正是这个口径，B（独立 greedy）必须用
  `--greedy-max-elements` 限流——这也是它跳过大量条目的原因。

### 11.3 判据速查：怀疑「内核随 dim 爆炸」时先跑这三条

```bash
# 1. 原生耗时是否随 dim 变（应当基本恒定；变化只来自总 FLOPs）
#    直接调 try_dedicated，dim=2/3/4/8 各一次，看 LAST_FAILURE 与耗时
# 2. plan 字节数是否随 dim 变结构（只有 batch/M/N/K 变 = 正常）
# 3. 把对照换成 path-matched：np.einsum(eq, *ops, optimize=['einsum_path', *pairs])
```
- 若 1 恒定而 2 结构不变，问题一定在对照或调用方，改内核只会白改。
- `_dedicated_plan` 每次 0.8–2.5 ms（dim 无关），是纯 Python 规划成本；
  热路径必须命中计划缓存，否则单次调用比内核本身贵 4–9 倍。

### 11.4 本轮实际修的两处生成端缺陷（都在 `_kernel_kupl.py`）

1. **负结果不缓存**：`_dedicated_plan` 返回 `None`（一元步/重复下标/单向自由
   指标等）时旧代码直接 `return None` 且不写缓存，于是**每次调用都重跑一遍
   greedy 规划**。实测「大方程 2.459 ms/次 → 命中后 0.277 ms/次」。现在
   `None` 也入缓存（与正结果同一 LRU/字节预算）。
2. **原生上限没在 Python 侧校验**：`dedicated_apply.h` 声明
   `MAX_SLOTS=256 / MAX_OPERANDS=4096 / MAX_STEPS=4096`，越界时原生返回 4；
   旧 Python 只校验 rank≤32，会白构造一份注定被拒的 plan。现在提前拒绝
   （如 300 算子的式子：旧代码构造成功、原生 status 4；现在编码期即 `None`）。

### 11.5 不要误信的两条「顺手优化」

- **轴序组内排序方向**：代码 `batch.sort(key=left.index)`（升序）与注释
  "让最后一个轴尽量取到 unit stride" 看似矛盾，但实测**升序优于降序**——
  C 连续下 unit-stride 轴是**最后**一个物理轴，升序把最大轴号放在组尾；
  降序反而把 unit-stride 挤出内层。全语料 unit-stride 命中率升序 51.3%、
  降序 22.3%；dim=8 原生总耗时 48 ms vs 135 ms。**保持升序**，别按注释改。
- **`_family_plan`/`try_family_apply` 与 dedicated 共用编码器，但 family 原生
  侧在 dim=6/7/8 会段错误（dim=4/5/9 正常）**，与维度不是单调关系、家族表头
  与 `family_apply.h` 逐字段一致（arity 16 / out_rank 5 / rank≤4 / steps 15 /
  slots 14 全在 caps 内），因此是 C++ `family_apply.cpp` 的越界，不是生成端。
  该入口当前**无任何调用方**（`grep try_family_apply pyqcd/` 只在定义处），
  接线前必须先修原生侧并补 dim=6/8 用例。

### 11.6 复现命令

```bash
cd ~/PyQCD
python -m pyqcd.testing._dedicated_kernel_001        # 小 dim 自检（注意它的 kernel() 对照 dim>=3 会 OOM）
python -B -m pyqcd.testing                           # 127 passed, 1 skipped, 0 failed
python -B -m pyqcd.testing.einsum.verify_topology    # 2582/2582 PASS, max dev 2.795e-06
```

## 11. 匹配器索引工程：编码选择 ≫ 压缩率（2026-11-12 实测）

目标：把"按拓扑签名命中语料"的首次匹配压到 1 ms 以内。四版迭代实测：

| 版本 | 结构 | 首次匹配 | 源码体积 |
|---|---|---:|---:|
| v1 | 8638 条 hex 摘要 + 8638 行代表写法，单字面量 | **284 ms** | 1.11 MB |
| v2 | 按元数分桶 + `zlib` + `base85` | 9.1–21 ms | 171 KB |
| v3 | (元数, **槽位总数**) 二级键 + 二进制对齐 `find` | 2.6–9 ms | 272 KB |
| v4 | v3 + **hex 编码替代 base85** | **1.04 ms** | 272 KB |
| v5 | v4 + 摘要 **128→96 位** | **0.55 ms** | 204 KB |

### 单点最大的坑：`base85decode` 比 `fromhex` 慢 180 倍

| 操作 | 输入 | 耗时 |
|---|---|---:|
| `base64.b85decode` | 43 K 字符 | **5.43 ms** |
| `zlib.decompress` | 34 KB | 0.01 ms |
| **`bytes.fromhex`** | 69 K 字符 | **0.03 ms** |
| `bytes.find`（对齐查找） | 34 KB | 0.007 ms |

结论：**压缩本身几乎免费，解码器选择才是瓶颈**。`base85` 看起来省 33% 体积，
却贵 180 倍——用 hex 换速度是压倒性正确的选择。

### 反直觉点：不要对桶内记录做字符串化

对 2160 条摘要做 `raw[i:i+16].hex()` 建 set，比直接在原始字节串上
`bytes.find()` + 16 字节对齐校验慢两个数量级。**保持 bytes，别转 str**。

### 摘要长度是可以安全缩短的

8638 条在 2^96 内碰撞概率 ≈ 1e-21，96 位足够；源码从 272 KB 降到 204 KB，
首次匹配 1.04 → 0.55 ms。**再降到 64 位（1e-12）属边缘效应，不值得。**

### 分支条件的正确选择：用"实现相关特征"，不是"语义特征"

严格拓扑签名下 8638 条**两两互异**（忽略操作数顺序仍 6958 条），按签名分桶等于没分。
真正能分桶的是**从方程本身可算出的结构量**：

- 第 1 层：操作数元数（8 个值）；
- 第 2 层：指标槽位总数 = 各操作数 rank 之和 + 输出 rank（桶数 8 → 24，
  最大桶 5760 → 2160 条）。

两者都由调用方方程直接算出，不需要预解析语料。

### 又一个真 bug：只收录一半语料

索引生成脚本原用 `text.split("## 动量指标与流分量指标")[0]`，**把附录整段切掉**，
导致 6056 条（70%）永远命不中。用户实测某条附录方程时暴露。

修法：生成必须遍历 `_wick_tasklist.TASKS` 全量，并在生成脚本里断言
`覆盖数 == N_TASKS`。

更深的教训：**"全量验证通过"必须区分"调用成功"与"内核真跑"**。
那次事故里 6056 条走的是通用路径回落，通用 vs 通用当然"正确"——
测的是回落的正确性，不是内核的正确性。验证脚本必须显式统计
`match() != None`、`try_dedicated() != None`、`偏差 <= 阈值` 三个独立计数。

### 回归：改摘要长度会打断下游

把摘要从 128 位改到 96 位时，CUDA 描述符按"4×int32 = 128 位"打包，
校验 `len(digest) != 16` 直接抛错，`test_topology_kernel_contracts` 失败。

修法：Python 侧 `.ljust(16, b"\0")` 零填充（单射、不改 C++ 格式）。
**改任何摘要/常量长度前，先 grep 全仓的 `16`/`128 bits` 校验点。**

## 12. [R2] 8638 条全量 allowlist 普查：谁真的更快（2026-11-12 实测）

问题：默认 `contract(eq, *ops)` 会命中拓扑入口，但它**不是全局更快**。
逐条测 8638 条 × dim ∈ {4,6,8}，产出 `pyqcd/einsum/_dedicated_allowlist.py`。

口径：`complex128`（双精度探测通过，实际标签恒为 `kupl_topology`）；
每条路径预热 1 次，再**轮内 A/B 顺序奇偶交替**、各 5 轮取中位；
`opt_cost > 1e8` 的单元格不测（TBD）。生成物键是连续 `seq`。

| dim | 实测 | 特化更快 | 更慢 | ratio 中位 | 最坏 | A/A \|ratio-1\| p95 |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 8638 | 8055 | 583 | 0.797 | **77.79** | 0.956 |
| 6 | 8040 | 7806 | 234 | 0.909 | 2.080 | 0.132 |
| 8 | 5364 | 4932 | 432 | 0.915 | 2.169 | 0.105 |

"一次语料扫描"的合计耗时（各单元格中位之和）：

| dim | 全通用 | 全特化 | 查 allowlist |
|---:|---:|---:|---:|
| 4 | 9.96 s | 9.33 s | **6.54 s** |
| 6 | 7.47 s | 6.90 s | **6.88 s** |
| 8 | 10.13 s | 9.20 s | **9.15 s** |

### 坑 1：`announce` 每进程每形态名只播报一次 → 第二个 dim 白名单全为 0

测完 dim=4（8638 条，全部播报过）再测 dim=6 时，同一批方程的 `topology:<digest>`
已经进过 `_announced`，**没有提示可读** → 命中判定全空 → 白名单 0 条。
第一次全量跑就踩了：dim=6/8 的 13404 条全部 `kernel=""`。

修法两层：① 换 dim 时主动重启 worker（新进程，播报重新生效）；
② 判定"内核真跑"改用执行侧证据 `_kernel_kupl/_kernel_cuda.LAST_INFO` 非空，
不再依赖打印。**任何靠 announce 做判定/统计的脚本都必须按进程隔离或改判据。**

### 坑 2：亚毫秒单元格的 ratio 噪声 ≈ 15%，且"后测更快"

dim=4 大量单元格在 0.1 ms 量级。A/A 对照（同一路径前后两块）显示第二块中位快
10%，`|ratio-1|` p95 达 0.956；dim=6/8 降到 0.132/0.105。
若不处理，`ratio<1` 的判据会把一半的持平单元格算成"更快"。

修法：每轮交替 `[spec, gen]` / `[gen, spec]`，把"槽位/漂移"一阶项抵消；
报告里同时给出 A/A 噪声与 `ratio<0.90/0.75/0.50` 的边际计数，
**不要引用落在噪声带内的单条比值**。

### 坑 3：分布是重尾，最坏 78 倍，且集中在同一类结构上

dim=4 的 583 条反向单元格里，最坏 10 条全是 16 元、**输出含大写外置指标**
（附录的动量/流分量展开式），专用 50 ms vs 通用 0.65 ms。
原因不是 FLOPs，而是原生拓扑入口在这类计划上的固定开销。

所以"平均更快"没有意义：`all_spec` 在 dim=4 只比 `all_gen` 快 6%，
但避开 583 条反向格后合计快 34%。**必须逐条判，不能按均值判。**

### 坑 4：dim=6/8 不需要查表，dim=4 必须查表

`all_spec` 与 allowlist 的差距：dim=6 只差 0.3%，dim=8 差 0.5% —— 直接全特化即可。
dim=4 差 43%（9.33 s vs 6.54 s），**必须查 allowlist**。

想用结构判据替代查表也不行：按"输出是否含大写外置指标"分流，
在 dim=4 误判 64%（大写组里 95% 其实特化更快，反向的是少数但极贵）。
判据只能落在 `seq`/摘要这一级。

### 坑 5：`_TOPOLOGY_PLAN_CACHE` 仅 32 项，语料级扫描必然重规划

`_kernel_kupl._TOPOLOGY_PLAN_CACHE` 是 32 项 FIFO。逐条扫描 8638 条时每条都是
cache miss，每次调用重跑 `contract_path`（实测约 2.7 ms/条），
这在 dim=4 的小单元格上与执行时间同量级，直接把 ratio 推向 1.0。
要测"内核本身"，必须按单元格连续调用（让同一 key 命中），或在扫描前扩大缓存。

### 复现

```bash
# 全量（断点续跑；--raw 默认在系统临时目录）
python -B -m pyqcd.testing._dedicated_census --dims 4,6,8 --max-cost 1e8 \
    --raw /tmp/census.jsonl --out pyqcd/einsum/_dedicated_allowlist.py
# 只重算报告/生成物
python -B -m pyqcd.testing._dedicated_census --report-only --dims 4,6,8 \
    --max-cost 1e8 --raw /tmp/census.jsonl --out pyqcd/einsum/_dedicated_allowlist.py
```

生成物 103 KB；`ALLOW` 惰性解析（import 不加载，首次访问 1.5 ms）。
`ALLOW[dim]` 成员是十进制字符串，按 int 查用 `allow(dim, seq)`。

## 12. 手写归约的累加精度：float 累加会劣于 BLAS（2026-11-12 实测）

### 现象

全量 8638 条与 `opt_einsum` 对比（float32，rtol=1e-4），**只有 1 条失败**：

```
seq=1973  T00111  偏差 2.58e-04
方程: cadb,ewfx,msnt,okpl,igjh,qurv,ac,mo,gi,kq,su,bd,fnp,hj,lr,tvx->ew
```

### 定位过程（关键：先把"算法错"与"精度差"分开）

| 对比 | 偏差 | 结论 |
|---|---:|---|
| pyqcd **float64** vs opt_einsum **float64** | **0.00e+00** | 算法完全正确 |
| 特化 float32 vs 真值 float64 | 2.97e-04 | 特化路径有问题 |
| 通用 float32 vs 真值 float64 | 3.95e-05 | 通用路径正常 |
| opt_einsum f32 vs f64 | 3.95e-05 | float32 自身误差量级 |

再用**步表逐行模拟**（Python，复刻 C++ 的每步 `einsum`）：得到 3.95e-05，
与通用路径一致 → **说明偏差不来自收缩顺序，而来自 C++ 实现本身**。

### 根因

`cpp/kupl/einsum/src/dedicated_apply.cpp` 的 `gemm_block()` 里，两个手写归约分支用
**float32 累加**：

```cpp
if (n == 1) { T total = 0; total += left[row*k+inner] * right[inner]; out[row] = total; }
if (m == 1) { out[column] += scale * row[column]; }   // 同样是 float
```

而 `cblas_sgemm` 内部按块做 float 乘 + **更高精度求和**。归约轴较长时，float 长链
求和的有效位数损失明显，差值可达 BLAS 的 7.5 倍。

### 修法

两个分支改用**双精度累加器**，累加后舍入回 `T`：

```cpp
using Accum = std::conditional_t<std::is_same<T, float>::value, double, T>;
Accum total = Accum(0);
for (...) total += Accum(left[...]) * Accum(right[...]);
out[row] = static_cast<T>(total);
```

**存储精度不变，只改累加口径**，与 BLAS 对齐。

### 效果

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| seq=1973 特化 f32 偏差 | 2.97e-04 | **5.35e-06** |
| 全量总耗时（8638 条） | 99.4 s | **21.4 s** |
| 单条中位 | 10.42 ms | **1.53 ms** |
| 单条最大 | 202.17 ms | **34.69 ms** |

性能提升是副产品：双精度累加让编译器得以向量化这两个分支。

### 判据（写测试时必须遵守）

1. **先分 float64 / float32 两路**：float64 一致 = 算法对；float32 才有偏差 = 精度问题。
2. **不要用单一容差判"对错"**：float32 在 14–16 算子链上自身就可差 5e-5。
   本库口径：float32 用 `rtol=1e-4`，float64 用 `rtol=1e-12`。
3. **手写归约必须与 BLAS 同口径**：任何 `T total = 0` 形式的累加，在 `T=float`
   且归约轴长时会劣于 `cblas_sgemm`。

## 13. 三方性能对照（特化 vs 通用 vs opt_einsum，2026-11-12 实测）

### 口径（必须遵守，否则数字不可比）

1. **交错计时**：每轮内三个臂相邻各跑一次，取该轮最快值，最后报中位数；
2. **opt_einsum 只计执行**：`contract_path` 事先求出并复用 path，否则 16 算子方程上
   greedy 搜索（实测 1.2 ms/条）会主导，把"执行快"变成"规划慢"；
3. **预热一次**：特化要建步表（冷启动 0.5–3 ms），opt_einsum 要建 path 缓存；
4. **同 dtype / 同操作数 / 同机**，并记录维度与工作集（见 `local-sizing.md`）。

### dim=3（200 条，float32，工作集 ≪ L2）

| 臂 | 中位耗时 |
|---|---:|
| **pyqcd 特化** | **74.2 µs** |
| pyqcd 通用 | 217.1 µs |
| opt_einsum（path 复用） | 285.2 µs |

| 对照 | 中位 | min | max | 快于对手 |
|---|---:|---:|---:|---:|
| 特化 / 通用 | **0.34×** | 0.29× | 0.49× | **200/200** |
| 特化 / opt_einsum | **0.26×** | 0.12× | 0.41× | **200/200** |

### dim=6（400 条）

| 对照 | 中位 | min | max | 快于对手 |
|---|---:|---:|---:|---:|
| 特化 / 通用 | **0.46×** | 0.05× | 0.99× | **400/400** |
| 特化 / opt_einsum | **0.33×** | 0.01× | 0.75× | **400/400** |

### 按元数（dim=3）：元数越大优势越大

| 元数 | 条数 | vs 通用 | vs opt_einsum | 特化中位 |
|---:|---:|---:|---:|---:|
| 9 | 1 | 0.42× | 0.41× | 54.7 µs |
| 10 | 2 | 0.41× | 0.29× | 62.8 µs |
| 13 | 22 | 0.36× | 0.28× | 64.5 µs |
| 14 | 39 | 0.36× | 0.27× | 71.0 µs |
| **16** | **136** | **0.34×** | **0.26×** | 75.9 µs |

### 复现

```bash
python -m pyqcd.testing._topology_perf_compare --sample 200 --dim 3 --rounds 3
python -m pyqcd.testing._topology_perf_compare --sample 400 --dim 6 --rounds 3
python -m pyqcd.testing._topology_perf_compare --sample 0 --json /tmp/perf.json
```

### 反面教训

* 用 `np.einsum(optimize='greedy')` 当基线是**错的**：16 算子病态（>25 s）；
* 顺序计时（先跑完 A 再跑 B）在忙碌机器上漂移 20–40%，足以伪造 2 倍差距；
* 大尺寸（dim≥12）吞吐由内存带宽主导，不能作为"内核快慢"的证据。

## 14. 性能 harness 自身的三个坑（2026-11-12 实测，都伪造过假结论）

### 坑 1：顺序测量伪造 2–3 倍差距

三个臂"先跑完 A、再跑完 B"时，忙碌机器上的漂移足以让同一对式子从 0.46× 跳到 3.35×。
**必须交错**：每轮内三臂相邻各测一次，取**每轮比值**的中位数。
独立复核（单条、7 轮、交错）得 0.70–0.84×，与 harness 报的 3.35× 完全相反。

### 坑 2：重复预算过大反而制造假慢

为了让计时分辨率可靠，很容易给"每个样本至少 1–2 ms"的重复预算。
但预算越大，**单个臂占用的连续时间越长**，越容易跨过 CPU 频率/调度变化窗口——
结果就是"某个臂看起来特别慢"，其实是被测顺序碰上了降频。
实测：2 ms 预算 → 16 算子出现 1740 µs vs 571 µs（3.0×）；
改 1 ms 预算 + 上限 50 次后，同一条变成 0.81×。

### 坑 3：不能用"最快一次"当样本

小任务上最快一次几乎等于计时器空转开销（~40–80 ns 量级），会把所有臂都测成常数。
正确做法：`probe 一次 → 按预算算重复次数 → 重复后取平均值`。

### 可靠的对照臂口径（缺一不可）

| 项 | 要求 |
|---|---|
| 交错 | 每轮三臂相邻，报每轮比值中位 |
| 预热 | 特化要建步表（冷 0.5–3 ms）；opt_einsum 要建 path 缓存 |
| opt_einsum 只计执行 | `contract_path` 事先求出并复用；否则 16 算子上 greedy 搜索（1.2 ms/条）主导 |
| 同 dtype/同操作数/同机 | 并记录工作集（见 `local-sizing.md`） |
| 单条复核 | 任何"比对手慢"的结论，必须用独立脚本单条交错复测后才可采信 |

### dim=10（L2 内，300→150 条）最终可信结果

| 元数 | 条数 | vs 通用 | vs opt_einsum | 特化中位 |
|---:|---:|---:|---:|---:|
| 10 | 5 | 0.15× | 0.45× | 796.9 µs |
| 12 | 2 | 0.37× | 0.43× | 123.8 µs |
| 13 | 16 | 0.55× | 0.42× | 328.9 µs |
| 14 | 30 | 0.47× | 0.25× | 3172.8 µs |
| 16 | 97 | 0.46× | 0.43× | 692.8 µs |

最差 1.29×（seq=2719，14 算子，两边都是 100 ms 量级），其余 ≤1.02×。

## 15. GPU（CUDA）三方性能对照（2026-11-12，RTX 4060 sm_89）

### 口径（GPU 特有，缺一不可）

| 项 | 要求 | 违反后果 |
|---|---|---|
| **同步计时** | 每次调用前后 `cp.cuda.Device(0).synchronize()` | 只测到 kernel launch（µs 级），结果全错 |
| **预热两次** | cuBLAS handle 创建 + kernel 首次编译可达**秒级** | 实测 `cupy.einsum` 首调 1.63 s 混进中位 |
| **对照臂用 opt_einsum** | 加 `backend="cupy"` + path 复用 | `cupy.einsum` 对多算子是逐对朴素收缩：13 算子 **80+ ms/次**，当基线会得出"特化快 1000 倍"的荒谬结论 |
| 交错计时 | 每轮三臂相邻 | 同 CPU 版 §14 |

### dim=4（50 条，float32，rounds=2）

| 臂 | 中位 |
|---|---:|
| **pyqcd CUDA 特化** | **444.5 µs** |
| pyqcd 通用 | 1423.5 µs |

| 对照 | 中位 | min | max |
|---|---:|---:|---:|
| 特化 / 通用 | **0.29×** | 0.19× | 0.68× |
| 特化 / opt_einsum | **0.41×** | 0.26× | 0.84× |

最大偏差（vs 通用）：7.09e-07

### 按元数（dim=4）

| 元数 | 条数 | vs 通用 | vs opt_einsum | 特化中位 |
|---:|---:|---:|---:|---:|
| 7 | 1 | 0.50× | 0.72× | 204.6 µs |
| 13 | 7 | **0.30×** | 0.38× | 346.2 µs |
| 14 | 9 | **0.27×** | 0.40× | 356.7 µs |
| **16** | **33** | **0.31×** | 0.43× | 487.5 µs |

### CPU vs GPU 对照

| 平台 | 特化 / 通用 | 特化 / opt_einsum |
|---|---:|---:|
| CPU dim=3（200 条） | 0.34× | 0.26× |
| CPU dim=6（400 条） | 0.46× | 0.33× |
| CPU dim=10（150 条） | 0.46–0.55× | 0.25–0.43× |
| **GPU dim=4（50 条）** | **0.29×** | **0.41×** |
| **GPU dim=3（20 条）** | **0.29×** | **0.37×** |

GPU 相对通用路径的优势（3.4×）略大于 CPU（2.2–2.9×），符合"Python 调度开销占比更高"的预期。

### 复现

```bash
python -m pyqcd.testing._topology_perf_compare       --sample 200 --dim 3
python -m pyqcd.testing._topology_perf_compare_gpu   --sample 50  --dim 4 --rounds 2
python -m pyqcd.testing._topology_perf_compare_gpu   --sample 20  --dim 3 --dtype complex128
```

## 16. 复数特化验收：参考必须用 complex128 真值（2026-11-12）

### 核心教训：不能用低精度结果当参考

初版验收拿"通用路径的 complex64 结果"当基准，然后把特化路径判为"7 条超差
（最差 1.48e-04）"。**这个判断是错的**，因为通用路径自身的 complex64 误差就有 1.5e-04
（与 opt_einsum 的 complex64 同量级）。

改用 **complex128 真值**（操作数升精度 + opt_einsum 固定 path）作参考后，结论完全翻转：

| 指标 | 特化 c64 | 通用 c64 |
|---|---:|---:|
| 最大偏差（vs c128 真值） | **3.42e-05** | 1.50e-04 |
| 更准的条数（全量 8638） | **1844** | 1045 |

即**特化路径比通用路径准 4.4 倍**，之前判定的"超差"是参考选错造成的假象。

### 正确判据

| dtype | 判据 | 依据 |
|---|---:|---|
| complex128 | 1e-12 | 自身误差 ~1e-15，留 3 个数量级 |
| **complex64** | **1e-4** | 该 dtype 在 13–16 算子链上的固有极限就是 ~3.4e-05 |
| float64 | 1e-12 | 同上 |
| float32 | 1e-4 | 同 §12 |

**不要**对 complex64 用 1e-5——那是 float32 口径，对 14–16 算子链不成立。

### 全量验收结果

| dtype | 检查 | 特化真跑 | 通过 | 特化最大偏差 | 通用最大偏差 | 耗时 |
|---|---:|---:|---:|---:|---:|---:|
| complex128 | 8638 | 8638 | **8638** | 3.04e-14 | 1.58e-14 | 33 s |
| complex64 | 8638 | 8638 | **8638** | **3.42e-05** | 1.50e-04 | 33 s |

### 性能（10 条路径 × 3 档尺寸，opt_einsum 参考，交错 + 自洽性检查）

**complex128**

| dim | 工作集 | 层级 | 特化 | vs 通用 | vs opt_einsum | 最大偏差 |
|---:|---:|---|---:|---:|---:|---:|
| 4 | 0.04 MB | L1 | 0.09 ms | 0.36× | **0.27×** | 1.0e-15 |
| 6 | 0.25 MB | L2 | 0.28 ms | 0.57× | **0.41×** | 1.8e-15 |
| 8 | 0.91 MB | L3 | 0.70 ms | 0.65× | **0.50×** | 2.0e-15 |

**complex64**

| dim | 工作集 | 层级 | vs 通用 | vs opt_einsum | 最大偏差 |
|---:|---:|---|---:|---:|---:|
| 6 | 0.13 MB | L2 | 0.55× | **0.37×** | 6.6e-07 |
| 8 | 0.46 MB | L3 | 0.60× | **0.46×** | 1.1e-06 |

复数与实数的加速比同量级（2–3.7 倍），全部尺寸方向一致。

### 复现

```bash
python -m pyqcd.testing._complex_specialized_verify --dtype complex128   # 全量正确性
python -m pyqcd.testing._complex_specialized_verify --dtype complex64
python -m pyqcd.testing._topology_perf_multidim --dtype complex128 --dims 4,6,8
```

## 17. 提示文案的语言：跟随 locale，别硬编码中文（2026-11-12，发布前拦截）

### 事故形态

发布终检 `release_check` 报 `FAIL en comments in English: CJK found: ['usage']`。
追下去发现：`docs/en/usage.md` 里那段"日志示例"是**库真实打印的中文**，
不是文档作者写错。也就是说——

> 英文环境的用户在控制台看到的是中文提示。

这是真实可用性缺陷，不是文档瑕疵。删掉示例只能掩盖问题（下一个人重新加回来还会犯），
所以修的是库。

### 修法（`pyqcd/einsum/_specialize.py`）

```python
LANGUAGE_ENV = "PYQCD_EINSUM_LANG"

def notice_language() -> str:
    explicit = os.environ.get(LANGUAGE_ENV, "").strip().lower()
    if explicit.startswith("zh"): return "zh"
    if explicit.startswith("en"): return "en"
    for name in ("LC_ALL", "LC_MESSAGES", "LANG"):
        value = os.environ.get(name, "").strip().lower()
        if value:
            return "zh" if value.startswith("zh") else "en"
    return "en"
```

优先级：`PYQCD_EINSUM_LANG` → `LC_ALL`/`LC_MESSAGES`/`LANG` → 默认 **en**。
默认取 en 而不是 zh：`C`/`C.UTF-8`（HPC 与容器最常见）必须给英文，否则又回到原问题。

`announce()` 与 `announce_generic()` 各分中英两支，措辞对称、都带上关闭方法。

### 连带修了检查器的一处误判

`release_check` 的 "en/zh code identical" 用 AST 比对，把**日志示例块**也算了进去——
而那两块本来就该按语言不同。判据加了豁免：

```python
def _is_output_sample(block): return "[pyqcd.einsum]" in block
```

规则：**可执行代码块**要求跨语言 AST 等价；**输出示例块**要求各自本地化、不比对。

### 教训

1. 「文档里的中文」不一定是文档问题——先问"这是库的输出吗"。
2. 面向用户的提示一律走语言选择函数，不要 `print("中文")` 硬编码。
3. 新加任何带语言的检查前，先区分"代码块"与"输出块"，否则会得到互相矛盾的两条门。
