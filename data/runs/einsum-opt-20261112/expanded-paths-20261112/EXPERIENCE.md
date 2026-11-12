# qcd-einsum 扩展生产路径测试经验

日期：2026-11-12

## 覆盖方法

从剩余 8632 条路径中，按 12 个 case 和元数 6/7/9/10/12/13/14/16 分层选取 19 条
full-scene 路径：

```text
1,3,9,849,856,130,1599,1623,1743,1864,879,
2583,2590,3431,6070,6911,7798,7918,5920
```

CPU 与 GPU 同时提交，均使用 `Nx=32/Lt=64/Nev=100/Ns=4/Mom=8`、double、
参考路径 `contract(..., specialize=False)`。

## 首轮发现

首轮 19 条全部数值正确，但确认两类亏损：

- CPU `seq=7918`：generic 0.0665 s，dedicated 0.1283 s，约慢 1.93x。
- GPU `seq=7918`：dedicated/generic=1.087x。
- GPU `seq=6911`：dedicated/generic=1.220x。
- CPU `seq=856`：约慢 1.0%，属于噪声级反例。

`seq=7918` 的计划指标：

```text
FLOPs=249,872,384
pack bytes=1,201,867,264
arena bytes=602,914,944
max step MACs=64,000,000
```

判定为 pack-bound。新增 CPU/GPU 通用门：

```text
pack_bytes >= 1 GiB and plan_flops <= 1G -> fallback
```

尝试过的 CUDA small-plan 门会误伤 `seq=2583/3431`，因此撤回；GPU `seq=6911`
作为极少数小计划反例保留。

## 第二轮结果

冻结 Python 源码快照和 KUPL `.so` SHA-256
`78af9202cfc7eb131d46d9c870165c8a5c482d6182d89ac02fb4b70bb8070a82`，
CUDA `.so` SHA-256
`e3e41b33848a20cf3b793b1daaa816a4d8d3a04b3de0adf2d821d77470172413`。

CPU job `1648351`：

| seq | path | ratio |
|---:|---|---:|
| 1 | kupl_dedicated | 0.7677 |
| 3 | kupl_dedicated | 0.9451 |
| 9 | kupl_dedicated | 0.6862 |
| 849 | kupl_dedicated | 0.6357 |
| 856 | kupl_dedicated | 1.0067 |
| 130 | kupl_dedicated | 0.8094 |
| 1599 | kupl_dedicated | 0.5963 |
| 1623 | kupl_dedicated | 0.9680 |
| 1743 | kupl_dedicated | 0.9489 |
| 1864 | kupl_dedicated | 0.8257 |
| 879 | kupl_dedicated | 0.9503 |
| 2583 | kupl_dedicated | 0.7246 |
| 2590 | kupl_dedicated | 0.9926 |
| 3431 | kupl_dedicated | 0.6298 |
| 6070 | kupl_dedicated | 0.3471 |
| 6911 | kupl_dedicated | 0.6333 |
| 7798 | kupl_dedicated | 0.3738 |
| 7918 | kupl_topology fallback | 0.9964 |
| 5920 | kupl_dedicated | 0.8667 |

CPU：中位 0.8094x，12/19 至少快 10%，18/19 不快于 generic，最大相对误差
6.95e-15。

GPU job `1648352`：

| seq | path | ratio |
|---:|---|---:|
| 1 | cuda_dedicated | 0.7485 |
| 3 | cuda_dedicated | 0.4803 |
| 9 | cuda_dedicated | 0.4486 |
| 849 | cuda_dedicated | 0.5379 |
| 856 | cuda_dedicated | 0.2732 |
| 130 | cuda_dedicated | 0.7528 |
| 1599 | cuda_dedicated | 0.5273 |
| 1623 | cuda_dedicated | 0.2875 |
| 1743 | cuda_dedicated | 0.2212 |
| 1864 | cuda_dedicated | 0.1218 |
| 879 | cuda_dedicated | 0.2359 |
| 2583 | cuda_dedicated | 0.6570 |
| 2590 | cuda_dedicated | 0.5764 |
| 3431 | cuda_dedicated | 0.5273 |
| 6070 | cuda_dedicated | 0.4380 |
| 6911 | cuda_dedicated | 1.3615 |
| 7798 | cuda_dedicated | 0.1571 |
| 7918 | plan-cache fallback | 0.9577 |
| 5920 | cuda_dedicated | 0.5642 |

GPU：中位 0.5273x，17/19 至少快 10%，18/19 不快于 generic，最大相对误差
5.61e-15。唯一反例为 `seq=6911`。

## 判据

1. 元数和 dtype 都不是收益的充分条件。
2. 大 pack + 低 FLOPs 是明确负收益信号。
3. 极小计划的固定开销可能形成个位数/十几条中的反例，但不能用过宽的小计划门
   牺牲正常 tiny dedicated 路径。
4. 资源门必须优先于性能门：静态 arena 或单步 MAC 超限应在原生分配前 fail-open。
5. 性能结论只认 full-scene；路径选择和结构覆盖率可另做全 8638 条计划审计，但不得
   将其冒充端到端性能。

## 第二批扩展路径

第二批从其余路径继续按 case/元数分层抽取 24 条：

```text
2139,2161,1872,5172,6044,5175,853,854,850,3457,
1838,1844,1849,6924,6918,6933,1621,1608,1613,
1030,1562,1551,1668,1662
```

CPU job `1648377`：24/24 全部快于 generic，中位 0.8788x，范围
0.4866–0.9934x，14/24 至少快 10%，最大相对误差 3.47e-15。

GPU job `1648376`：24/24 全部快于 generic，中位 0.2168x，范围
0.1285–0.9371x，23/24 至少快 10%，最大相对误差 6.68e-15。

## 两批合并统计

| 设备 | 路径数 | 中位比 | 范围 | ≥10% 加速 | 不快于 generic | 反例 |
|---|---:|---:|---:|---:|---:|---|
| CPU | 43 | 0.8612x | 0.3471–1.0067x | 26 | 42/43 | `seq=856` 1.0067x |
| GPU | 43 | 0.2760x | 0.1218–1.3615x | 40 | 42/43 | `seq=6911` 1.3615x |

该结果满足“普遍加速、允许极少数反例”的当前验收口径。

## 全计划门控审计

Job `1648360` 对完整 8638 条生产计划执行结构审计，未实际计算大张量：

```text
plan unavailable = 0
CPU allowed = 3833
CPU blocked = 4805
GPU allowed = 2864
GPU blocked = 5774
```

CPU 拦截原因：

```text
arena >= 8 GiB: 3651
single GEMM >= 100G MAC: 953
pack >= 1 GiB and FLOPs <= 1G: 201
```

GPU 拦截原因：

```text
arena >= 3 GiB: 5191
single GEMM >= 40G MAC: 382
pack >= 1 GiB and FLOPs <= 1G: 201
```

计数按门控判定顺序归入首个原因。最大审计计划出现 `5.04 PB` 静态 arena 与
`65.536P MAC` 单步，证明若只检查拓扑命中会造成不可接受的资源风险。

## 当前源码验收

- CPU job `1648387`：KUPL native 16/16、plan-gate 11/11（CUDA 用例按环境跳过）、
  热路径 4/4，16 个当前源码文件 SHA-256 全部 OK。
- GPU job `1648386`：统一套件 `117 passed / 2 skipped / 2 failed`；两项失败均为
  `gvar` 缺失，plan-gate 12/12、einsum native 与热路径均通过。

## opt_einsum 对比

使用已安装的 `opt_einsum 3.4.0`，同输入、同 dtype、同设备，比较：

```text
qcd_specialized / opt_einsum(dp)
```

核心 6 条：

| seq | CPU | GPU |
|---:|---:|---:|
| 2584 | 0.598x | 0.152x |
| 866 | 0.891x | 0.213x |
| 855 | 0.842x | 0.130x |
| 129 | **1.297x** | **1.179x** |
| 1863 | 0.837x | **0.025x** |
| 6585 | CPU 四路计时超 10 min，已 scancel | 0.606x（qcd fallback） |

43 条扩展路径：

| 设备 | 中位比 | 范围 | ≥10% 加速 | 不快于 dp |
|---|---:|---:|---:|---:|
| CPU | 0.528x | 0.0283–2.502x | 31/43 | 36/43 |
| GPU | 0.125x | 0.0172–1.032x | 40/43 | 42/43 |

CPU 慢于 opt_einsum dp 的 7 条为
`9,130,6070,7918,2161,1030,1562`，其中 `1030/1562` 分别约慢 2.5x/2.44x。
GPU 只有 `seq=6070` 慢约 3.2%。

结论：GPU 对 opt_einsum 已接近普遍领先；CPU 对内部 generic 普遍领先，但对
`opt_einsum(dp)` 还不是普遍领先。发布说明不得把 CPU 结果表述为全面胜过
opt_einsum。

## 发布门禁快照

`python -m pyqcd.testing.release_check --offline` job `1648472` 当前为
`5/9 PASS`：

- 版本三处一致：通过，`0.3.9`。
- 文档示例与中英代码结构：通过。
- 文档版本：通过。
- sdist 构建：失败，当前发布环境缺 `build` 模块。
- sdist 内容：未执行，因无产物。
- Sphinx 中英 `-W`：失败。
- `pyqcd/py.typed` 当前不存在，但元数据仍声明 `Typing :: Typed`。
- PyPI 最新版本已为 `0.3.9`，与本地版本相同，不能原版本重发。
- 当前 Git 为 `dev56-dirty`，存在大量未提交/未跟踪改动。
- 全量套件 `117 passed / 2 skipped / 2 failed`，两项均因缺 `gvar`。

因此当前源码不具备正式发布条件。
