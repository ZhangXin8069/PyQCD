# 性能

> **正式使用前强烈建议先跑 `python -m pyqcd.einsum.test --tune`。**
> **任何性能都需以实测为准，本文档不胡言。** 具体加速比与基准表已整体删除：早先那些
> 数字出自已废弃的拓扑步表口径，与现行的 `ROLE_SHAPES` 口径**不可比**。

## 为什么用 tune 而不是信文档

同一个方程在同一个后端上，最快配置取决于**形状、dtype、设备与内核可用性**四件事。
本库不猜：`--tune` 在**你的**形状上真跑一遍，把胜出配置写进
`pyqcd.einsum.profile.csv`；之后普通调用只读这张表，命中即按记录执行。

```bash
python -m pyqcd.einsum.test --equation "ab,bc,cd->ad" \
  --shape "[100,100],[100,100],[100,100]->[100,100]" --tune
```

`--tune` 带取值，不是开关：

| 形态 | 行为 |
|---|---|
| `--tune`（留空）/ `--tune true` | **完整模式**：先探测方程能否特化、等价改写后能否特化，选定分支后按**有序清单**逐组实跑 |
| `--tune pass` | **快速通过**：直接走 `[if_equivalence=false, if_specialized=false]` 这条线，禁用 `--compare` 与全部与 tune 无关的功能，专注快速、无阻塞 |
| 不给 `--tune` | 不遍历，按配置表或内置缺省执行 |

遍历**不是**笛卡尔积：它按分支走一份**有序**清单（两个可特化分支各 22 组，普通分支
21 组），任何一组一旦超过当前最快耗时就被当场终止。表里的 `wall_ms` 是**整个**
`contract()` 调用的墙钟，因此读到的数就是你程序要付的数。

### 决定"测什么"的几个选项

| 选项 | 合法取值 | 说明 |
|---|---|---|
| `--backend` | `auto` / `numpy` / `cupy` / `torch` | 只选数组库；强制原生内核用 `--kernel-force`，不是后端 |
| `--compare` | `opt_einsum` / `backend` / `none` | 缺省 **`none`**；"与 numpy 对照"的写法是 `--backend numpy --compare backend` |
| `--optimize` | `greedy` / `optimal` / `dp` / `branch_1` / `branch_2` / `branch_all` / `none` | `auto` 是内置的缺省选择器，**不是**可写取值 |
| `--kernel-force` | `generic` / `dedicated` / `topology` / `backend` | 只接受单值，缺省 `generic`；逗号链式与 `auto` 一律拒绝 |
| `--shape` | `[..],[..]->[..]` | 省略时按该方程在语料里的 roles 反推尺寸 |
| `--explain` | `auto` / `true` / `false` | 省略 `--shape` 且经等价写法命中时，打印 `shape source: corpus(equivalence) via <方程>` |

`--shape` 不限于字面命中：方程可以经等价改写（重排操作数、置换输出轴）到达某条语料
条目。这时 `--explain true` 会点名提供尺寸的那条语料方程，轴序因此不会显得凭空而来。

### 那个计时到底量的是什么（0.5.9 起）

两条规则让报告里的毫秒与你自己脚本里的数字可比：

- **每个样本前后都做设备同步。** `cupy` 与 CUDA 版 `torch` 是异步下发，只把
  `perf_counter()` 套在 `contract()` 外面，表会停在**入队**那一刻。现在每个被测样本
  进出各同步一次（CPU 上是空操作），所以 GPU 上报的也是计算时间。0.5.9 之前不是这样，
  GPU 耗时因此小了若干个量级——那些历史数字**全部作废**，见
  `docs/ERRATA_gpu_timing_20261119.md`（仓库内，不在本站）。
- **加速设备上 `--warmup` 至少抬到 2。** 那里的首调还要建 CUDA 上下文、cuBLAS 句柄与
  workspace，只丢一次会让中位低于脚本里的稳态。报告 `time :` 给的是**实际使用**的值，
  被抬过时另有一行 `time note:`；`--json` 里是 `warmup_effective` / `warmup_raised`。

`--compare opt_einsum` 的参照跑在**与操作数同一个后端**上（`backend=` 显式传入并写进
参照行，例如 `opt_einsum.contract(backend='cupy', optimize='greedy')`）。若 `opt_einsum`
没有注册该后端，它会如实报错而不会悄悄改用 numpy 量一遍——跨设备的"参照"不叫参照。

## 怎么读输出

初始化阶段会在任何计量之前先自述过程。它的字段是稳定的机器可读 token，所以这一段
在任何语言下都逐字相同（逐步的 `<n>ms` 耗时在此省略，它每次运行都不同）：

```text
tune init : mode=all  equation=ab,bc,cd->ad  operands=3  dtype=complex128  device=cpu  backend=auto
tune init : shapes ok  total_elements=192  est_bytes=0MiB
tune init : branch probe done  raw_ok=false  equiv_ok=false  branch=B3
tune init : kernel admission done  unavailable=topology
tune axes : kernel_force=dedicated,generic,topology,backend  as_equivalence=false,true  optimize=greedy,branch_1,branch_2,dp,branch_all,optimal,none
tune branch: branch=B3  raw_ok=false  equiv_ok=false  target_equation=-
tune plan : planned=21  kept=21  trimmed=0  skipped-kernels=-
```

* `tune axes` —— 被遍历的三个轴。`optimize` 只会出现七个叶子
  `greedy` / `optimal` / `dp` / `branch_1` / `branch_2` / `branch_all` / `none`；
  `kernel_force` 只会出现 `dedicated` / `generic` / `topology` / `backend`。
* `tune branch` —— 方程本身或它的等价写法能否特化，这决定清单里有没有 `topology` 组。
* `tune plan` —— 计划多少组、经内核准入后还剩多少组；对本形状不可用的内核在这里被
  剔除并如实报告一次，因此"少了一组"不会被误读成静默跳过。

之后每个实测组各有一行 `tune comb`（三个轴取值、`wall_ms` 与状态 token），末尾是
`tune best`（胜出组）与 `tune prune`（把慢组截断的动态上界）。状态词表机器可读、
**从不翻译**：`ok`、`pruned`（比当前上界慢而被剪枝）、`timeout`、`error`、
`mismatch`（超出容差）、`skip`（该内核对本形状本质上没有路）。

## `--tune-timeout` 的能力边界（如实说明）

单次上限用 `SIGALRM` 实现，是**尽力而为（best-effort）**上限：

- 能打断 **Python 侧长循环**（例如 `optimal` 在 9 个以上算子上的穷举规划）；
- **打不断原生 C/CUDA 代码**：信号只在 Python 字节码之间派发，原生代码不返回解释器时
  handler 永不执行。曾尝试 fork 子进程隔离并**已回退**——CUDA 上下文不跨 fork 复用，
  子进程必然 `cudaErrorInitializationError`。

因此 GPU 上请把 `--tune-timeout` 取足够大，或先用 `--tune pass` 快速通过。

## 正确性优先于速度

速度只有在数字正确时才有意义。本库的每一段文档示例都以 `assert` 结尾，构建会执行全部
示例，任何一条不再自证就失败。

### 自己验证

三个独立实现、同一批输入，必须逐元素一致：

```python
import numpy as np
import torch
import pyqcd.einsum as pe

rng = np.random.default_rng(0)


def complex_noise(*shape):
    """按给定形状生成复数随机数组（实部与虚部分别采样）。"""
    real_part = rng.standard_normal(shape)
    imag_part = rng.standard_normal(shape)
    return (real_part + 1j * imag_part).astype(np.complex128)


def check(equation, *operands):
    """把一次结果对照两个独立参照。

    - numpy 用 optimize="optimal" 自行规划路径；
    - torch 用原生实现跑同一批操作数；
    三者逐元素一致（rtol 1e-10）才算通过。
    """
    ours = np.asarray(pe.contract(equation, *operands))
    numpy_ref = np.einsum(equation, *operands, optimize="optimal")
    torch_ref = torch.einsum(equation, *[torch.as_tensor(o) for o in operands]).numpy()
    assert np.allclose(ours, numpy_ref, rtol=1e-10, atol=1e-12), "differs from numpy"
    assert np.allclose(ours, torch_ref, rtol=1e-10, atol=1e-12), "differs from torch"
    print(f"{equation:32s} ok  max|d| = {np.max(np.abs(ours - numpy_ref)):.2e}")
```

### 每次改动都会跑的检查

- `python -m pyqcd.testing`：全套契约（数值、分派不变量、配置表语义、CLI 行为）；
  仅源码检出可用；
- `python -m pyqcd.einsum.verify_specialized "<equation>"`：纯查询，告诉你这条方程
  命中哪个形态族、语料来源与被声明的内核——**不构造张量、不调用 `contract()`**；
- `python -m pyqcd.einsum.test --tune ...`：在真实形状上标定一次。

## 关于"快多少"

本文档**不给**具体加速比。原因不是没有数据，而是**数据的口径必须与你的形状一致**：

- 收缩序对形状敏感，同一方程在 `{4,5,40,50}` 这类混杂轴与在等长探针上的最优序不同；
- 内核可用性取决于设备与库（CUDA 步表、KUPL 步表、BLAS 复数支持各不相同）；
- 因此任何"在某某加速卡上快多少"的说法，换到你的机器上通常不成立。

要数字，就自己跑一次 tune——那条命令打印的就是你这台机器、这个形状的真实中位耗时。
