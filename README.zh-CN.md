# PyQCD（中文）

[English](https://gitlab.itp.ac.cn/ZhangXin8069/PyQCD/-/blob/main/README.md) | **中文**

PyQCD 是一个用 Python 编写的格点 QCD 工具库，可选原生加速。它提供传播子与算符收缩、
梯度流、重整化、胶子 TMD-PDF 提取所需的基础组件，并自带一个
**与 `opt_einsum` 兼容**的收缩引擎。

## 安装

```bash
pip install -U pyqcd
```

**请逐行阅读安装过程的输出与提醒。** 原生内核（CPU 的 KUPL、GPU 的 CUDA）在安装时
现场构建，输出会说明是否建成、缺哪个编译工具链或驱动、跳过了哪些可选依赖。
忽略这些提示，很容易拿着一个只能走纯 Python 规划的包，却以为它在用原生加速。
需要 Python 3.9+。

核对实际装上了什么（不需要写 import）：

```bash
python -m pyqcd           # 版本、原生后端、可选依赖、子包
python -m pyqcd --json    # 机器可读
```

源码编译步骤、CUDA 架构、线程与环境变量见 `pyqcd/einsum/AGENTS.md`，本文不重复。

## 快速开始

```python
import numpy as np, pyqcd.einsum as pe
a, b = np.ones((2, 3)), np.ones((3, 4))
c = pe.contract("ij,jk->ik", a, b)      # 等价于 a @ b
```

公开接口与 `opt_einsum` 对齐：`contract`、`contract_path`、`contract_expression`、
`shared_intermediates`、`PathInfo`、`paths`、`backends`、`helpers`、`parser`、`blas`。

## 性能：先实测，再 tune

> **正式使用前强烈建议先跑一次 tune：**
> `python -m pyqcd.einsum.test --tune --equation "<你的方程>" --shape "<你的操作数形状>"`
>
> **任何性能都必须以实测为准。本 README 不引用任何加速比、倍率或基准表——
> 从别处抄来的数字只反映别人的机器、形状与负载。**

`--tune` 是配置表**唯一的写者**。在固定的方程 / 形状 / dtype / 设备 / 后端上，
它按**有序清单**遍历 `kernel_force × as_equivalence × optimize`（两个可特化分支各 22 组、
通用分支 21 组），超过当前 `min_wall_ms` 上界的组直接剪枝，最后只写胜出的一行。
普通调用只读这张表；没有表时只警告一次、打印该配置的 tune 命令，然后照常计算。

单条方程一律用 `python -m pyqcd.einsum.test` 复核——一次性脚本的结论不予采信：

```bash
python -m pyqcd.einsum.test --equation "ij,jk->ik" --shape "[4,4],[4,4]->[4,4]"
# PASS ij,jk->ik  <本次实测>ms  rss+<N>MiB

python -m pyqcd.einsum.test --equation "cadb,menf,ac,em,bd,fn->" --scene distill --verbose true
```

它真的算一遍、按容差判对错并报告实际走的路线。退出码：`0` 通过 / `1` 数值或形状不符 /
`2` 参数或环境错误。`--json <path>` 写完整记录，`--verbose true` 打逐项明细（含
`route` / `kernel` 证据行）。省略 `--shape` 时按 `--scene` 从语料 roles 反推尺寸——
那些形状只作路径与正确性证据，永远不是生产规模。

## 特化路径：先查再假设

```bash
python -m pyqcd.einsum.verify_specialized "cadb,menf,ac,em,bd,fn->"
```

强烈推荐在追特化路线之前先跑它。这是**纯查询**：不生成操作数、不调用 `contract()`、
不改全局状态。它依次匹配形状特化 → 8638 条语料索引 → 等价映射，给出 `HIT`
（含摘要、语料出处与声明内核）或 `MISS`（若存在等价写法，同时给出操作数与输出的置换）。
退出码：`0` 命中 / `1` 未命中 / `2` 方程非法。`--json` 给机器可读结果，
其中 `reason_key` / `reason_args` 与语言无关。

`HIT` 只说明"语料里有这条拓扑"，**不说明原生内核会跑**：资源门、dtype 策略、连续性与
库可用性都可能把它送回通用路径。要看"这次真的走了哪条"，读
`python -m pyqcd.einsum.test` 的 `route` / `kernel` 行。

## 版本变更摘要

* **未发布** — 配置表**只有** `--tune` 会写，`contract()` 只读；同键 1 行，
  只有一列计时（`wall_ms` = 整次 `contract()` 的墙钟）。`--kernel-force`
  只接受单个逻辑值（`generic` / `dedicated` / `topology` / `backend`）。
  tune 按分支的有序清单遍历，`min_wall_ms` 剪枝，只写胜出的一组。
* **0.5.13** — `kernel_force="original"` 在缺 `opt_einsum` 时抛 `KernelUnavailableError` 并点明装法（`pip install "pyqcd[einsum]"`），不再是一句裸 `ModuleNotFoundError`；同时把 `opt_einsum>=3.3` 声明为新的 `einsum` 可选依赖。
* **0.5.12** — 新增 `original` 内核：把收缩交给 `opt_einsum.contract`（同样吃 `--optimize`，`branch_1`/`branch_2`/`branch_all`/`none` 会被转成 `branch-1`/`branch-2`/`branch-all`/`False`——下划线拼写在那里是 `KeyError`）；参与 `--tune`，排在 `backend` 之前。`--tune` 逐组记录**峰值内存/显存**（按组测量，并扣掉操作数本身的常驻量——那是输入不是开销），全量快照写进 `pyqcd.einsum.record.csv`（只作参考、绝不参与命中；命中时若存在更省内存的配置会打一行提示）。剪枝截止改为**最好耗时的 2 倍**，缺省改成 **warmup 0 / rounds 1**，配置表移到 `~/.cache/pyqcd.einsum`，缺行时首次调用自动跑一轮 tune（`PYQCD_EINSUM_AUTOTUNE=0` 可关）。`optimize_topology_v2` 彻底删除；操作数暂存改到 tmpfs 并做空间预检（`/tmp` 满曾让 29 组全部变成 `error`）。
* **0.5.11** — GPU 侧补齐 `bypass_plan_gate()`：`--device gpu --tune` 的 `dedicated` 不再被 `plan arena exceeds the measured CUDA dedicated budget` 判死（`_api` 用 `getattr(module, ..., None)` 取旁路，缺名字就静默不旁路）。`--help` 的 `--tune-timeout` 写明窗口是**按组**的`单次上限 × (1+warmup+rounds)`（缺省 30 s × 7 = 210 s），看到 ~80 s 返回不代表上限失效；`docs/` 里每条历史 GPU 耗时行都带行内 `[作废]` 标注。
* **0.5.10** — `--tune` 不再跳过任何一格：只看规划器的 "predicted" 预判剪枝已全部删除，每一组都真的被测，结局只有 `ok` / `timeout`（硬上限：父进程 kill 子进程）/ `error` / 按**实测** `min_wall_ms` 终止的 `pruned`。`--atol` 不写 = **自动档**：64 位 dtype 用 1e-12，单精度用该 dtype 机器精度的 64 倍（7.6e-6）——一条 10 算子 complex64 收缩会积累约 16 eps，拿 1e-12 判会让每次运行都失败；给了数值则原样生效。0.5.9 之前记录的 GPU 耗时作废，见 `docs/ERRATA_gpu_timing_20261119.md`。
* **0.5.9** — `--compare opt_einsum` 现在真的跑在操作数自己的后端上（`backend=` 传给 `oe.contract` 并写进参照行）；每次测量在样本前后都做设备同步，异步的 cupy/torch 因此报的是计算时间而不是入队时间（同一条方程脚本里 ~900 ms、tune 表里 0.32 ms）；加速设备把 warmup 抬到 ≥2 并如实标注；`optimize_topology_v2` 从 `contract` 签名删除（唯一接收方 topology v2 执行器 0.5.4 已删），再传它抛 `TypeError`；`pyqcd.einsum.last_kernel` / `last_profile` 正式导出。
* **0.5.8** — `--tune` 与 `--kernel-force <内核>` 不再被内核的**预算门**挡住：arena / 单步 GEMM / pack 流量是**自动分派**的选路判据，不是能力上限，现在由 `_kernel_kupl.bypass_plan_gate()` 在这两条路上穿过，超预算的事实仍写进 `LAST_INFO['budget_gate']`。准入只保留环境性检查（原生库、dtype、torch 后端、无法编码的收缩），于是每一组 tune 都真的跑出 `wall_ms` ——`dedicated` 此前因 `pack/flops = 3.950` 被整档删掉 7 组。自动分派不变。
* **0.5.7** — `--tune` 的子进程组一律 `profile=False`：遍历期间配置表既不读也不写，命中行不会再和父进程逐格显式下发的三轴冲突（0.5.6 引入的 9 组 `ProfileConflict` 回归，已修）。
* **0.5.6** — `--tune` 的超时升级为**硬上限**：每组在全新解释器里跑、超时由父进程 kill 子进程，原生 C/CUDA 代码真的能被打断（SIGALRM 做不到）；进程启动开销单列，**不计入** `wall_ms`。两条**只看规划器**的预判剪枝把病态组挡在运行之前（`optimize=none` 在生产形状上要算 ~7.6e17 次浮点；`optimal` 仅规划就要 38 s）。设 `PYQCD_EINSUM_TUNE_INPROCESS=1` 退回旧的尽力而为路线。
* **0.5.5** — `--tune pass`（快速通过：只走 `[if_equivalence=false, if_specialized=false]` 这条线，禁用 `--compare` 与全部无关功能）、初始化阶段的 `tune init :` 起止进度行，以及**如实**的 `--tune-timeout` 契约（尽力而为：能打断 Python 循环、打不断原生 C/CUDA）；性能页删掉了过时的加速比，只保留"先跑 `--tune`、按自己的形状实测"。
* **0.5.4** — 拓扑步表统一到 `ROLE_SHAPES`：`production`（生产异形轴）与 `uniform`（等长 `dim=4` 探针）**已删除**，`sizes` 改为必填；`optimize` 恰为七个下划线叶子；两张表统一用 `auto` 生成；删除 `kupl_topology_v2`、`--backend kupl|cuda`、`--compare numpy`；`--tune` 不再因参照收缩卡死，且**先改写再强制内核**。
* **0.5.3** — 等价路由表、提示整句单语、`--tune` 加固。
* **0.5.2** — 引入 `--tune` 与等价备忘。
* **0.5.1** — 画像缓存不再推荐走不通的路。
* **0.5.0** — 画像缓存、`kernel_force` 收窄为逻辑名、详解模式。
* **0.4.x** — 8638 条语料的原生特化分派；回退是硬性不变量：数值不变，只有路线变。

## 文档与代码托管

* 文档 — https://py-qcd.readthedocs.io/zh-cn/latest/usage.html
* GitLab（主开发） — https://gitlab.itp.ac.cn/ZhangXin8069/PyQCD
* Gitee — https://gitee.com/zhangxin8069/PyQCD
* GitHub — https://github.com/zhangxin8069/PyQCD

## 许可证

MIT
