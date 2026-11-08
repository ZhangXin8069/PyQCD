# pyqcd.einsum 实测经验与陷阱（2026-11-08 会话沉淀）

> 全部结论来自当轮实跑，附复现方式。凡"必须/不要"都是踩过坑后的硬约束。

## 1. 构建

| 事实 | 说明 |
|---|---|
| **CUDA 必须显式指定架构** | `make-nv.sh` 默认 `6.0 7.0` 且**不生成 PTX**，在 sm_89（RTX 4060）上无法运行。必须 `PYQCD_CUDA_ARCHITECTURES=8.9 bash build-nv.sh`。 |
| **MPI 自动探测会误判** | conda 常把 `CXX` 固定为不带 MPI 头的 `g++`，而 `PYQCD_EINSUM_ENABLE_MPI=auto` 曾只看 `command -v mpicxx` → 编译失败（`mpi.h: No such file`）。已改为**试编译 `#include <mpi.h>`** 才启用。显式 MPI 构建：`CXX=mpicxx PYQCD_EINSUM_ENABLE_MPI=1 bash build-kupl.sh`。 |
| **同名头文件冲突（重要）** | `cpp/kupl/einsum/include/` 与 `cpp/cuda/einsum/include/` 下有同名 `nucleon2pt.h`、`pattern_3pt.h`，而 `make-nv.sh` 把 **KUPL 的 include 放前面** → CUDA 源文件会包含错头，报"标识符未定义"。修法：CUDA 源文件用相对路径 `#include "../include/xxx.h"`。 |
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
4. **验收**（`python -m pyqcd.testing.einsum.simple.kernel_acceptance --nvec N --only <name>`）：
   - 判据：**慢于 opt_einsum → BUG**；快于 opt 但慢于通用路径 → `OK-WITH-CAVEAT`（不接线）；快于两者 → `PASS`。
   - 同时看 nvec=50 与 100；两者结论可能相反。
5. **接线后回归**：`simple/check_kernels.py`（命中/开关/complex64/非连续/异常回退），
   再跑 `verify_native` 双后端 + `verify_compat` + `python -m pyqcd.testing`。

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
