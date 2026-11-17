# PyQCD（中文）

[English](https://gitlab.itp.ac.cn/ZhangXin8069/PyQCD/-/blob/main/README.md) | **中文**

PyQCD 是一个用 Python 编写的格点 QCD 工具库，可选原生加速。它提供传播子与算符收缩、
梯度流、重整化、胶子 TMD-PDF 提取所需的基础组件，并自带一个
**与 `opt_einsum` 兼容**的收缩引擎。

## 安装

需要 **Python 3.9+**。当前 PEP 639 构建元数据要求 `setuptools>=77`，因此 Python 3.8
已不再支持。

```bash
pip install pyqcd
```

**KUPL 原生内核默认是必需的**：`pip install` 会现场编译，缺 C++ 工具链时直接
以可操作的错误信息中止安装，而不是悄悄给你一个只会走纯 Python 规划的包。装上
`g++`/`c++` 与 `bash`、`binutils` 即可；确实只想要纯 Python 规划时显式：
`PYQCD_SKIP_NATIVE_BUILD=1 pip install pyqcd`（`PYQCD_EINSUM_REQUIRE_NATIVE=0` 等价）。
CUDA 仍是机会性构建：缺 `nvcc` 只是拿不到 GPU 内核，不影响 CPU 原生路径。

确认实际装上了什么（不需要写 import）：

```bash
python -m pyqcd            # version, native backend, optional deps, subpackages
python -m pyqcd --json     # machine-readable
```

> **CUDA 构建。** 构建时会自动探测显卡（`nvidia-smi compute_cap`）并始终嵌入 PTX，
> 当前显卡开箱即用；若要原生 SASS，或在无 GPU 的登录节点上为已知目标构建，
> 设 `PYQCD_CUDA_ARCHITECTURES=<算力>`（例如 `8.9`）。

> **线程按调用自动调档。** 两档自动切换（专用形态用 `kernel` 档，BLAS 链用 `generic` 档），
> 直接作用到已加载的运行时；用 `OMP_NUM_THREADS` /
> `OPENBLAS_NUM_THREADS`（或 `PYQCD_EINSUM_OMP_THREADS` / `PYQCD_EINSUM_BLAS_THREADS`）覆盖，
> `PYQCD_EINSUM_THREADS=off` 关闭，`pyqcd.einsum.thread_info()` 查看。

## 0.5.1 发布声明 — qcd-einsum：画像缓存不再推荐走不通的路

`0.5.1` 修掉 0.5.0 画像缓存的三个缺陷（均由一次真实生产运行 `ambn,cgdh,…->ka`
（10 操作数）暴露）：

* 经**等价改写**跑成的调用，过去被记成初始值 `generic`——缓存说"走的通用路径"，
  而内层其实跑的是 `kupl_topology`。现在如实标为 `equivalence`，且这类行**不落盘**
  （在原方程上强制 `topology` 必然"未尝试"，它不是可重放的建议）。
* `generic` 与 `backend` 行**保留**在缓存里并参与 `best()` 选优：同一条 10 操作数方程上
  generic 实测 **385 ms**、`kupl_topology` **2735 ms**，把它排除等于丢掉最大的一笔收益。
  只有 `equivalence` 这个诊断标签既不落盘也不参与选优。
* CLI 的命中行在建议**未被采纳**（`generic` 行）时不再说"命中"，改为
  `命中但未采纳 → kernel=generic（该取值不可重放，本次仍走原分派）`；
  真正换了路的建议仍打 `命中 →`。

无数值变化，无接口变化。

## 0.5.0 发布声明 — qcd-einsum：profile 缓存、`kernel_force` 收窄、详解模式

`0.5.0` 是 **minor** 版本，因为有一处公开接口不兼容（`kernel_force`）。其余均为新增。

**破坏性变更 —— `kernel_force` 取值**：逻辑值收敛为
`auto | dedicated | topology | generic | backend`，具体标签已删除且会报错：

| 旧 | 新 |
|---|---|
| `kupl_dedicated`、`cuda_dedicated` | `dedicated` |
| `kupl_topology`、`cuda_topology` | `topology` |

`dedicated`/`topology` 按操作数后端解析（numpy → kupl，cupy → cuda）。`backend` 是新增值，
含义是"完全不用专用内核"：直接调 `backend.einsum`、继承调用方的 `optimize`，并跳过专用分派
与等价改写。`LAST_INFO['path']` 里的运行时路径名（`kupl_dedicated` 等）**不变**——那是观测值，
不是输入。

**profile 画像缓存**（`pyqcd/einsum/_profile_cache.py`，默认开启）：为
`(host, user_device, backend, dtype, equation, shape, device)` 记录一行 CSV，含实际使用的
kernel/optimize 与实测 `compute_ms`；下次对同一键直接选最快的那条记录。`profile.csv` 读写、
`profile_compare.csv` 只写不读（compare 与 equivalence 字段）。目录优先级：
`profile_dir=` > `PYQCD_EINSUM_PROFILE_DIR` > `./.cache`；关闭时**不读、不写、不建目录**。
写入是**稀疏**的：只有"新键"或"比该键最优快 >5%（`_IMPROVE_RATIO=0.95`）"才落盘，
性能接近的配置不会把文件撑大。每键最多保留 4 条、文件超过 5 万行会重写一次、
坏行跳过；模块用 `os.register_at_fork` 处理 fork，MPI rank 不会因父进程持锁而死锁。

**详解模式**：`--explain` 在**首次**调用时打印 `plan :`（怎么匹配到/没匹配到）、
`kernel :`（选了谁）、`why :`（为什么放弃专用内核：不可用 / 连续门 / dtype 策略 /
profile 认为 backend 更快）。之后的调用只打一行：命中、未命中、或"profile 已关闭"。
既有输出行不动，新行只在显式开启时追加。

**热路径削减**（按结构计数，不按墙钟）：计划缓存覆盖到 **2 算子非语料**方程（原 ≥3）、
语料已命中时跳过等价门链、线程档不变时走**零环境变量读、零 `/proc` 读**的快路径
（原每调用 5 次环境变量读）。

**修复**：同进程内切换 `profile_dir` 会被静默忽略（单布尔门闸）；`cache=False` 仍可能经
等价层登记一次匹配缓存命中；省略 `profile_dir` 参数时会把已钉住的目录重置回 `./.cache`。

## 0.4.5 发布声明 — qcd-einsum：静默回落开始留痕

`0.4.5` 是诊断修复：过去有四处**回落到通用路径却不留任何痕迹**，或只报
"进程内第一次"故障——长跑作业里第二个**不同原因**的失败会被吞掉，而唯一的提示仍指向
第一个原因，于是照着它修永远修不好。

* `_shape_paths.try_execute` 现在写 `LAST_FAILURE`（`{"name", "reason"}`），并对
  已注册形状的首次失败按 `(形态, 异常类型)` 提示一行，不再吞异常。"在表里但回落了"
  与"根本没注册"从此可区分。
* `_compile_native_expression` 不再无声返回 `None`：原生表达式编译失败会带上异常
  类型与消息告警，于是"表达式建好了、但每次调用都慢一个量级"这种情形有了线索。
* `_warn_native_fallback_once` 与 `_warn_specialized_kernel_once` 改为按**根因**去重，
  不再用一个进程级布尔：每个不同根因各报一次（`NativeUnavailable: …` 签名 / 异常
  类型+消息），同一根因重复不报。同因噪声保留（库缺失仍只报一条），而**不同**故障
  不会再被吞。
* 新增契约 `_einsum_fallback_visibility_contract`（4 项）钉住以上全部，含"两个不同的
  原生根因各报一次"与"编译层与执行层的同一根因只报一次"。

不改任何数值行为，不改分派次序。

## 0.4.4 发布声明 — qcd-einsum：强制内核、自动等价改写、缓存

`0.4.4` 是**下一个关键 qcd-einsum 稳定版本**，`stab31` 声明的冻结由本次发布解除。
下列三处行为改动正是它作为关键版本（而非补丁）的原因。

* **强制指定内核**：`contract(..., kernel_force="cuda_topology")`、
  `PYQCD_EINSUM_KERNEL_FORCE`、两个 CLI 的 `--kernel-force`。与自动分派不同，
  强制而跑不成时抛 `KernelUnavailableError`（CLI 退出码 2），**不再** warning 后
  静默回落到通用路径；强制会旁路负缓存，保证真试一次；`last_kernel()` 给出
  `requested` / `ran` / `attempted` 与逐标签 `reasons`。
* **自动施加等价写法**：`contract(..., as_equivalence=)`、`PYQCD_EINSUM_AS_EQUIVALENCE`、
  `--as-equivalence`（**缺省开启**）。命中映射时改写方程、重排操作数、把结果转回
  原输出轴序，使未命中任何特化形态的收缩仍能走到特化路径。**返回的数组始终是原
  方程的答案**；另有独立的输出形状守卫，形状不符即报错而不是交出错结果。
  `out=`、额外 kwargs、逐元素方程、显式优化器路径与 `specialize=False` 时不施加。
* **带命中统计的缓存**：`_cache` 为匹配器（`specialize.match`）、等价查询
  （`equivalence.normalize`）与规划器（`api.find_path`）提供有界、线程安全的 LRU，
  并统计 `hits/misses/stores/evictions`；`PYQCD_EINSUM_CACHE`、`contract(..., cache=)`、
  `--cache`、`cache_stats()`、`cache_clear()` 控制与查看；`cache=False` 既不读也不写。
* **修复**：同桶存在多条同构写法时，优先选**无需操作数/输出转置**的那条
  （8638/8638 条 order 变体为零代价，修前为 772/8638），不再凭空制造被原生
  dedicated 连续门拒掉的非连续视图。

验证：einsum 各契约模块全绿（等价映射 44、强制内核 9、as-equivalence/缓存 7、
分派 8、缓存 8、拓扑 25、路由 6、查询子集 8、CLI 49 且含其既有 1 项失败）；
全套为 `120 passed, 2 skipped, 10 failed`——这 10 项是既有 GPU 环境问题
（nvrtc 不接受 `compute_70`），与本版无关。

## 0.4.2 发布声明 — 冻结稳定版的文档修正

`0.4.2` 只纠正本地化文档中的一处语义反转，其余完全不变。

* 代码、数值、分派链与原生 ABI 与 `0.4.1` 完全一致，无行为变化。
* 中文页面把英文与数据都指向的 **"not slower than"（不慢于）** 写成了
  「不快于」（不更快），与同表数据自相矛盾（CPU 中位 `0.861x`、26/43 至少快
  10%、唯一反例仅慢 0.7%）。共修正 13 处；`_topology_perf_multidim.py` 失败
  分支里的「不快于」语义本就正确，未改动。
* `stab31` 起的关键稳定版本冻结继续有效：在下一个 qcd-einsum 关键稳定版本之前，
  任何影响 qcd-einsum 行为的改动仍默认禁止。

## 0.4.1 发布声明 — qcd-einsum 冻结稳定版

`0.4.1` 是 `pyqcd.einsum` 的**关键稳定版本**（冻结基线标签 `stab31`）。

* 新增 `python -m pyqcd.einsum.verify_specialized "<方程>"`：查询一条方程走哪条
  特化路径——语料命中、拓扑摘要、声明内核与出处（`seq` / T 号 / 角色），
  **不跑收缩**（不生成操作数、不改全局状态）。退出码：命中 `0`、语料外 `1`、
  方程非法 `2`。库内接口：`pyqcd.einsum.query_specialization(eq)`。
* 冻结约定：在下一个 qcd-einsum 关键稳定版本之前，任何影响 qcd-einsum 行为的
  改动**默认禁止**，除非用户明确强制。三条例外见 `pyqcd/einsum/AGENTS.md` §0。
* `0.4.0` 的资源门控拓扑特化（8638 条方程）全部保持不变；数值、分派链与原生
  ABI 均未改动。

## 0.4.0 发布声明

`0.4.0` 是 `pyqcd.einsum` 的生产特化版本。

* 拓扑特化表覆盖 **8638 条 6–16 算子方程**，KUPL 与 CUDA 共用同一拓扑语义；
  每个计划在原生分配前都经过 fail-open 资源门。
* 43 条 full-scene 扩展审计中，CPU 对 PyQCD 内部通用路径 **42/43** 不慢，
  中位 selected/generic 为 `0.861x`；CUDA **42/43** 不慢，中位 `0.276x`。
  保留的两个反例是 CPU `seq=856`（慢 0.7%）与 GPU `seq=6911`（慢 36.2%）。
* CPU 长链 `seq=6585` 从 **22.445 s** 降到 **10.310 s**（`0.459x`）。
* CUDA 对 `opt_einsum 3.4.0`（`dp`）在 **42/43** 条审计路径上不慢；
  CPU 为 **36/43**。本版本**不声明 CPU 全面胜过 `opt_einsum`**，仍有 7 条 CPU
  路径较慢。

特化是生产优化，不是“所有形态必快”的承诺。原生资源超限、pack 重而 FLOPs 低、
内核缺失或已注册内核运行失败时会主动回退通用路径，数值结果保持不变。

## 文档

* English — https://py-qcd.readthedocs.io/en/latest/usage.html
* 中文 — https://py-qcd.readthedocs.io/zh-cn/latest/usage.html

## 代码托管

| 平台 | 用途 | 地址 |
|---|---|---|
| GitLab（gitlab.itp.ac.cn） | 国内开发——主开发仓库 | https://gitlab.itp.ac.cn/ZhangXin8069/PyQCD |
| Gitee | 国内发布——国内镜像 | https://gitee.com/zhangxin8069/PyQCD |
| GitHub | 国外发布——海外镜像 | https://github.com/zhangxin8069/PyQCD |

## 快速开始

```python
import numpy as np
import pyqcd.einsum as pe

a = np.arange(6.0).reshape(2, 3)
b = np.arange(12.0).reshape(3, 4)

pe.contract("ij,jk->ik", a, b)                       # like a @ b
path, info = pe.contract_path("ij,jk->ik", a, b)     # (path, PathInfo)
int(info.opt_cost)

expr = pe.contract_expression("ij,jk->ik", (2, 3), (3, 4))
expr(a, b)                                           # reusable plan
```

公开接口与 `opt_einsum` 对齐：`contract`、`contract_path`、`contract_expression`、
`shared_intermediates`、`PathInfo`、`paths`、`backends`、`helpers`、`parser`、`blas`。

## 包内有什么

| 子包 | 内容 |
|---|---|
| `pyqcd.lattice` | 格点常数、DeGrand–Rossi `gamma`、SU(2) Clebsch–Gordan |
| `pyqcd.gauge` | Wilson/Polyakov 圈、Clover 场强、拓扑荷 |
| `pyqcd.smear` | HYP 与 Stout 涂抹 |
| `pyqcd.operator` | Clover / 对偶场强、梯度流胶子 OPE 算符 |
| `pyqcd.vertex` | VdV/VVV 顶点、动量涂抹、本征模压缩 |
| `pyqcd.contraction` | Wick 收缩、重子算符、蒸馏收缩 |
| `pyqcd.renorm` | 梯度流、Z_R、混合方案、NLO 匹配、TMD 提取 |
| `pyqcd.analysis` | 重采样、拟合、ratio / 有效质量 / 能量分析 |
| `pyqcd.pipeline` | 集中配置与多步管线 |
| `pyqcd.parallel` | CPU、内存与 GPU 的只读资源探测 |
| `pyqcd.einsum` | 收缩引擎（本包的核心特性） |

## 收缩引擎

* 纯 Python 规划（`greedy` / `optimal` / `branch` / `dp`）永远可用。
* 两个可选原生后端：**KUPL**（CPU）与 **CUDA**（GPU），导出同一套 C ABI，
  共用同一份规划核心。
* 回退是硬性不变量：缺库、原生调用失败或形状被拒时静默降级到 Python 路径 ——
  结果永不改变。
* 完整的 8638 条格点 QCD 拓扑语料会进入资源门控的专用分派；默认开启，可按调用
  （`contract(..., specialize=False)`）或按进程（`PYQCD_EINSUM_SPECIALIZE=0`）关闭。

可查询标识：

```python
import pyqcd.einsum as pe

pe.build_info()        # backend, native_available, array_backend, native_library
pe.specializations()   # registered equation shapes（本版本 8638 条）
```

## 性能

以下均为 A100 计算节点上的 full-scene 基准，而不是 smoke 或小维合成数据。
输入为 `Nx=Ny=Nz=32`、`Lt=64`、`Nev=100`、`Ns=4`、`Mom=8`、`complex128`；
计时前先逐路径对照通用路径。表中为 `selected/generic`，小于 `1` 表示更快。

| 证据集 | CPU | GPU |
|---|---|---|
| 43 条分层生产路径：中位比 | **0.861×** | **0.276×** |
| 43 条：不慢于 PyQCD generic | **42/43** | **42/43** |
| 43 条：比 PyQCD generic 至少快 10% | **26/43** | **40/43** |
| 43 条对 `opt_einsum(dp)`：不慢 | **36/43** | **42/43** |

六条核心路径（比值小于 `1` 表示更快）：

| `seq` | CPU / generic | GPU / generic | CPU / `opt_einsum(dp)` | GPU / `opt_einsum(dp)` |
|---:|---:|---:|---:|---:|
| 2584 | 0.856× | 0.265× | 0.598× | 0.152× |
| 866 | 0.936× | 0.248× | 0.891× | 0.213× |
| 855 | 0.982× | 0.177× | 0.842× | 0.130× |
| 129 | 0.791× | fallback，1.017× | 1.297× | 1.179× |
| 1863 | 0.974× | 0.157× | 0.837× | 0.025× |
| 6585 | **0.459×** | fallback | 未完成 | 0.606×，fallback |

全部 **8638** 条计划另做了不执行大张量的结构审计：CPU 资源门放行 3833 条、
拦截 4805 条；CUDA 门放行 2864 条、拦截 5774 条。超大静态 arena、超大单步
GEMM 与“pack 重而 FLOPs 低”的计划不会进入特化路径。43 条 full-scene 路径的
最大相对误差为 CPU `6.95e-15`、GPU `6.68e-15`。

```python
import numpy as np
import pyqcd.einsum as pe

expr = "gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M"   # nucleon 2pt
fast = pe.contract(expr, *operands)                    # specialised path (default)
base = pe.contract(expr, *operands, specialize=False)  # general path
assert np.allclose(fast, base)                         # same numbers, different speed
```

按调用 `specialize=False` 或按进程 `PYQCD_EINSUM_SPECIALIZE=0` 都能关掉这套优化，
两条路径的数值必须一致。

- **内存。** Python 执行路径的峰值 RSS 约为 `opt_einsum` 的 1.4–2 倍
  （实测 1200² 时 18→41 MB，2000² 时 73→104 MB），原因是每一步会多保留一份与输出同尺寸的
  中间量。长时间任务请据此预留内存。

优势来自拓扑识别、资源感知的计划选择、收益明确时的融合/专用执行、按生命周期复用
workspace，以及资源边界不安全时主动回退。它**不是普适的**：`opt_einsum(dp)` 仍在
7 条审计过的 CPU 路径上更快，内部也有两个明确的 full-scene 反例；资源门会有意把
不安全计划送回通用路径。完整口径、证据与复现方式：
https://py-qcd.readthedocs.io/zh-cn/latest/performance.html

### PyTorch

通过 `pip install "pyqcd[torch]"` 支持 PyTorch 张量：`contract()` 按操作数类型分派，
`torch.Tensor` 进 → 同设备 `torch.Tensor` 出（由 `torch.einsum` 执行，不走原生库）。
想要原生 kernel，请在边界处转成 numpy 或 cupy。完整对照表与示例：
https://py-qcd.readthedocs.io/zh-cn/latest/usage.html

## 测试

这些套件位于 `pyqcd/testing/`，**不在 PyPI 发行包里**；需在源码仓库中运行：

```bash
python -m pyqcd.testing                       # integration suite
bash pyqcd/testing/simple/run-all.sh          # per-function API parity vs opt_einsum
bash pyqcd/testing/einsum/simple/run-all.sh   # einsum sub-system suite
```

## 许可证

MIT
