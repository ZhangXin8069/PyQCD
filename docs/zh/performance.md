# 性能

`pyqcd.einsum` 与 `opt_einsum` 在同一台机器、同一 dtype、同一线程设置下对比，两种实现
**每轮交替谁先运行**，以抵消缓存与时钟漂移。下面每个数字都是实测，每张表都由同一份脚本产生。

## 生产特化（首发于 0.4 系列）

发布证据使用完整生产场景，而不是 smoke 小形状：

| 项 | 设置 |
|---|---|
| 硬件 | A100-40GB 节点，20 个 CPU worker，`complex128` |
| 问题 | `Nx=Ny=Nz=32`、`Lt=64`、`Nev=100`、`Ns=4`、`Mom=8` |
| 语料 | 43 条分层路径，覆盖 case 与 6/7/9/10/12/13/14/16 元数 |
| 内部对照 | 相同输入、相同计划上的 `contract(..., specialize=False)` |
| 外部对照 | `opt_einsum` 3.4.0，`dp` 规划器 |
| 线程 | `OMP_NUM_THREADS=8`、`OPENBLAS_NUM_THREADS=4` |
| 正确性 | 每条计时路径先做数值对照 |

表中为 selected/generic，小于 `1` 表示更快。

| 证据 | CPU | GPU |
|---|---|---|
| 43 条：selected/generic 中位比 | **0.861×** | **0.276×** |
| 43 条：不慢于 PyQCD generic | **42/43** | **42/43** |
| 43 条：比 PyQCD generic 至少快 10% | **26/43** | **40/43** |
| 43 条：selected/`opt_einsum(dp)` 中位比 | **0.528×** | **0.125×** |
| 43 条：不慢于 `opt_einsum(dp)` | **36/43** | **42/43** |

内部反例是 CPU `seq=856`（`1.0067×`，慢 0.7%）与 GPU `seq=6911`
（`1.3615×`，由固定 launch/pack 开销主导）。CPU 慢于 `opt_einsum(dp)` 的路径为
`seq=9,130,6070,7918,2161,1030,1562`，其中最后两条约慢 `2.50×` 与 `2.44×`；
GPU `seq=6070` 约慢 `3.2%`。因此本版本声明的是“相对 PyQCD 通用路径的广泛加速”，
不是“CPU 全面胜过 `opt_einsum`”。

### 六条核心生产路径

| `seq` | CPU / generic | GPU / generic | CPU / `opt_einsum(dp)` | GPU / `opt_einsum(dp)` |
|---:|---:|---:|---:|---:|
| 2584 | 0.856× | 0.265× | 0.598× | 0.152× |
| 866 | 0.936× | 0.248× | 0.891× | 0.213× |
| 855 | 0.982× | 0.177× | 0.842× | 0.130× |
| 129 | 0.791× | fallback，1.017× | 1.297× | 1.179× |
| 1863 | 0.974× | 0.157× | 0.837× | 0.025× |
| 6585 | **0.459×** | fallback | 未完成 | 0.606×，fallback |

`seq=6585` 是 CPU 上最清楚的生产收益：generic `22.445 s`，selected
`10.310 s`。CPU 的 `opt_einsum` 对照超过 10 分钟资源上限后被取消，不作为证据。

### 全部 8638 条计划的资源门

完整语料只做计划审计，不分配大张量：

| 门控结果 | CPU | GPU |
|---|---:|---:|
| 放行原生计划 | **3833** | **2864** |
| fail-open 回退 | **4805** | **5774** |
| 计划不可用 | **0** | **0** |

CPU 拦截 arena 至少 8 GiB、单步 GEMM 至少 100G MAC，或
`pack >= 1 GiB && FLOPs <= 1G` 的计划；GPU 阈值为 3 GiB 与 40G MAC，
pack-heavy 规则相同。审计中最大静态 arena 约 `5.04 PB`，最大单步约
`65.536P MAC`，因此这些是保证正确性与资源安全的门，不是装饰性启发式。

43 条 full-scene 路径的最大相对误差为 CPU `6.95e-15`、GPU `6.68e-15`。
已提交的测量台账与命令记录见
`docs/einsum_production_optimization_20261112.md`。

```bash
python -m pyqcd.testing.einsum.bench_opt_einsum_production --help
python -m pyqcd.testing.einsum.verify_plan_gate
```

## 历史通用路由口径

| 项 | 设置 |
|---|---|
| 对照 | `opt_einsum` 3.4.0，`dp` 策略 |
| dtype | 注明处为 `complex128`，其余 `float64`（双方一致） |
| 计时 | 预热 1 次；每点 3 轮，**单次不足 0.3 s 的用例在轮内重复取最好值** |
| 顺序 | 每轮交替"谁先跑"，否则后跑的一方白拿 10–20% 的假优势 |
| 正确性 | 每条用例先按相对最大范数与 `opt_einsum` 对齐，再计时 |
| 机器 | 4 物理核（8 线程）+ RTX 4060 Laptop，8 GB |

## 历史 case 覆盖

十五个用例，每个结构性类别各一个——不是挑出来的"高光表"：

| 类别 | 用例 |
|---|---|
| 两操作数收缩 | 二维矩阵乘（实数）、二维矩阵乘（复数） |
| 无求和指标 | 逐元素、外积 |
| 纯约简 | 全约简（`ij,ij->`）、部分约简（`ij->i`） |
| 多操作数链 | 三操作数链、四操作数链 |
| 闭环三角 | hyper-edge（`ij,jk,ik->i`） |
| batch 结构 | 批量 GEMM、ellipsis 批量、批×矩阵、大量小批量 |
| 生产 kernel | 核子 2pt、γ 夹心 |

两张表用**同样的十五个用例、同样的尺寸**，因此同一行可直接跨设备比较。单次耗时
0.003–2.4 s，每点 3 轮，每轮交替先跑顺序。

### 历史 CPU 结果（`numpy` 数组，KUPL 后端）

| 用例 | `pyqcd.einsum` | `opt_einsum`（`dp`） | 倍率 |
|---|---|---|---|
| 核子 2pt，nvec=112（kernel） | 2.38 s | 44.24 s | **18.6×** |
| 批量 GEMM 16×1024² | 0.25 s | 2.91 s | **11.9×** |
| ellipsis 256×512² | 0.65 s | 5.92 s | **9.18×** |
| γ 夹心 8192×1024（kernel） | 0.25 s | 1.93 s | **7.83×** |
| 大量小批量 4096×64² | 0.087 s | 0.201 s | **2.32×** |
| 外积 8000 | 0.051 s | 0.083 s | **1.62×** |
| 逐元素 12000² | 0.20 s | 0.25 s | **1.25×** |
| 三操作数链 3×3072 | 0.54 s | 0.59 s | **1.07×** |
| 二维矩阵乘（复数）2048² | 0.31 s | 0.32 s | 1.04× |
| 全约简 12000² | 0.039 s | 0.040 s | 1.02× |
| 部分约简 16000² | 0.090 s | 0.092 s | 1.02× |
| 批×矩阵 8×1024×512 | 0.044 s | 0.045 s | 1.01× |
| 四操作数链 4×2048 | 0.26 s | 0.26 s | 1.00× |
| 二维矩阵乘（实数）4096² | 0.63 s | 0.63 s | 0.99× |
| hyper-edge 2048² | 0.088 s | 0.082 s | 0.93× |

### 历史 GPU 结果（`cupy` 数组，CUDA 后端）

| 用例 | `pyqcd.einsum` | `opt_einsum`（`dp`） | 倍率 |
|---|---|---|---|
| γ 夹心 8192×1024（融合 kernel） | 0.51 s | 2.05 s | **4.04×** |
| 核子 2pt，nvec=112（kernel） | 0.91 s | 2.43 s | **2.68×** |
| 全约简 12000² | 0.017 s | 0.036 s | **2.13×** |
| 二维矩阵乘（复数）2048² | 0.48 s | 0.55 s | **1.15×** |
| 批×矩阵 8×1024×512 | 0.071 s | 0.080 s | **1.13×** |
| ellipsis 256×512² | 0.52 s | 0.58 s | **1.11×** |
| 三操作数链 3×3072 | 0.57 s | 0.63 s | **1.10×** |
| 四操作数链 4×2048 | 0.37 s | 0.41 s | **1.10×** |
| 批量 GEMM 16×1024² | 0.35 s | 0.38 s | **1.09×** |
| 二维矩阵乘（实数）4096² | 0.67 s | 0.71 s | **1.05×** |
| 逐元素 12000² | 0.015 s | 0.015 s | 1.02× |
| hyper-edge 2048² | 0.157 s | 0.160 s | 1.02× |
| 外积 8000 | 0.003 s | 0.003 s | 1.00× |
| 部分约简 16000² | 0.009 s | 0.009 s | 1.00× |
| 大量小批量 4096×64² | 0.034 s | 0.033 s | 0.96× |

GPU 行是**同卡对同卡**。0.05 s 以下的行在 GPU 上受带宽限制，倍率基本是噪声；CPU 上
0.09 s 那几行同理。

## 历史通用行为什么有高倍率

那些大倍数**不是**普遍加速，而是三个可测的具体效应：

1. **参考实现要落地一个巨大的中间体。** 核子 2pt 网络在 `opt_einsum` 的 `dp` 路径下需要
   7 步，峰值中间体 89.9 M 元素（`complex128` 下 1.44 GB），而输入总量只有 11.8 M 元素。
   我们的 kernel 按 `n³` 分块结构一趟算完。44 s → 2.4 s 的 18.6× 就来自这里——算术量
   相同，内存流量不同。
2. **参考实现不会把 batch 轴变成批量 BLAS 调用。** `bij,bjk->bik`（16×1024²）在
   `opt_einsum` 里那一步要 2.99 s，而同一批数组上 `numpy.matmul` 只要 0.28 s、
   `numpy.einsum(optimize=…)` 0.27 s；我们的 0.25 s 就是地板值。所以 11.9× 的含义是
   "参考实现高出硬件地板 11 倍"，不是"我们比 BLAS 快 11 倍"。
3. **参考实现在夹心形态上按 batch 打包拷贝。** `ab,...bce,cd->...dae` 在 `opt_einsum`
   里 2 步、中间体 33.6 M 元素（537 MB），融合 CUDA kernel 只写一次结果，这就是 GPU 的
   4× 与 CPU 的 7.8×。

诚实的小结：通用形态两边都在同一档（0.93–1.15×），两位数倍数只出现在"省掉中间体"或
"batch 轴变成一次批量 GEMM"的地方。

## 什么情况走哪条路径

| 输入 / 形态 | 执行位置 | 说明 |
|---|---|---|
| 逐元素、转置、对角、迹、外积 | backend 自己的 `einsum` | 没有求和标签，规划器无事可做 |
| `numpy` + batch 标签（被两个以上操作数共享**且**出现在输出里） | KUPL 原生计划执行器 | 一次批量 GEMM 取代 numpy 的逐 batch 循环 |
| `numpy` + 无 batch 标签 | numpy 分步（tensordot） | 原生执行器在这里慢 6–10% |
| `cupy` + γ 夹心 | CUDA 原生执行器 | 目前唯一的融合 CUDA kernel |
| `cupy` + 其它通用形态 | `cupy.einsum`（cuBLAS） | 原生批量 GEMM 在 16×1024² 上慢约 5 倍 |
| 核子 2pt / 3pt 主项形态 | KUPL/CUDA 专用 kernel | 命中提示里显示 `[kernel: …]` |

规划器在 `auto` 下会对 ≤8 算子的图取 `opt_einsum` 的 `dp` 路径：等价代价的路径实际
速度可以差 1.75×（`ij,jk,ik->i` 两种顺序估算都是 1.72e10，实测一个慢 75%），而用户
对照的正是参考顺序。`opt_einsum` 不可用时自动回退到内置规划。

命中提示会告诉你实际走了哪条路：`[kernel: kupl_nucleon_2pt]` 表示跑了专用 kernel，
`[plan-cache]` 表示只复用了计划。如果注册了 kernel 的形态拿不到 kernel，
`pyqcd.einsum` 会**立即告警并说明原因**，而不是静默降级。

## 拓扑特化内核

`refer/sush/wick_12cases.md` 的十一个强子拓扑情形定义 **2582 条唯一收缩拓扑**
（元数 6–16）。索引按**拓扑签名**而不是拼写：指标按全式首次出现顺序重编号，因此
`ij,jk->ik`、`ij,jb->ib`、`ab,bc->ac` 是同一条目，而 `ij,kj->ik` 与 `ij,jk->ki`
是不同条目。空白与省略输出不改变签名；**操作数顺序**和**操作数内指标顺序**会改变。

命中时整条收缩计划交给一次原生调用（`qcd_kupl_topology_apply` 或
`qcd_cuda_topology_apply`），不再在 Python 中逐步执行。命中提示区分两种情况：
`[kernel: kupl_topology]` 表示原生拓扑入口真的执行了计划，`[plan-cache]` 表示只
识别了形态、复用了计划——例如传了 `out=`、操作数混合 dtype，或原生入口不可用。
回落只改变耗时，数值完全一致。

早期尚未解决的尺寸断崖已由上面的资源门取代。当前实现会在原生分配前
计算完整计划指标，并在 arena、单步 GEMM 或 pack-heavy 超限时回退。43 条
full-scene 证据是本版本的权威性能声明；`docs/topology_kernels_20261111.md`
中的历史单路径观察不是当前发布数据。

```bash
python -m pyqcd.testing._topology_equations_contract
python -m pyqcd.testing._topology_kernel_contract
```

第一条命令把 2582 条方程逐条与 `numpy` 对照，并覆盖改名同签名与结构负例；第二条
检查原生拓扑入口本身——dtype 策略、plan/指针/`out` 参数，以及 120 条 `dim=4` 语料
抽检。

## 增益从哪来

1. **后端定向的拓扑分派**：命中后只尝试与实际数组后端对应的原生入口。
2. **资源感知规划**：原生分配前先计算静态 arena、最大单步 GEMM 与
   pack-heavy 低 FLOP 风险。
3. **按生命周期复用 workspace**：CPU 与 CUDA arena 复用物理 buffer，而不是让
   每个逻辑中间槽同时存活。
4. **热计划缓存**：完整 8638 拓扑匹配缓存不再淘汰，专用计划缓存从 16384 项起步。

## 重复调用

完整语料冷启动首轮约 `0.54–0.57 s`，之后热 `match()` 扫描约 `0.53 µs/次`。
8638 个专用计划的热扫描约 `0.195 s`，即 `22.7 µs/计划`，缓存占用约
`46.1 MiB`。这些是路由缓存数字，不是算子内核加速。

## 内存

单次收缩的峰值 RSS，用 `resource.getrusage` 在独立进程中测量（`float64`，`d = 1200`）：

| 形态 | `opt_einsum` | `pyqcd.einsum` |
|---|---|---|
| 2 操作数，1200² | 18 MB | 16 MB |
| 3 操作数，1200² | 29 MB | 27 MB |
| 4 操作数，1200² | 29 MB | 27 MB |
| 批量 8×512² | 16 MB | **52 MB** |

GEMM 链已与 `opt_einsum` 持平或更省。批量形态是例外：它走原生执行器，打包工作区大约
是参考实现的 3 倍。长时间批量作业请据此预留内存。

## 正确性

只有在数字正确时，速度才有意义。引擎在三个层次上验证，而且**本文档每个示例结尾都有
`assert`**——构建会执行全部示例，任何一块不再自验证就会失败。

### 自己验证一次

下面用两个独立参考实现（NumPy 与 PyTorch）对同一批输入做对照，三者必须一致：

```python
import numpy as np
import torch
import pyqcd.einsum as pe

rng = np.random.default_rng(0)


def complex_noise(*shape):
    """按给定形状生成复数随机数组：实部与虚部各自独立采样。"""
    real_part = rng.standard_normal(shape)
    imag_part = rng.standard_normal(shape)
    return (real_part + 1j * imag_part).astype(np.complex128)


def check(equation, *operands):
    """用两套独立参考实现校验同一个结果。

    - numpy 用 optimize="optimal" 自行规划路径；
    - torch 用它对同一批输入的原生实现；
    三者逐元素一致（rtol 1e-10）才算通过。
    """
    ours = np.asarray(pe.contract(equation, *operands))
    numpy_ref = np.einsum(equation, *operands, optimize="optimal")
    torch_ref = torch.einsum(equation, *[torch.as_tensor(o) for o in operands]).numpy()
    assert np.allclose(ours, numpy_ref, rtol=1e-10, atol=1e-12), "differs from numpy"
    assert np.allclose(ours, torch_ref, rtol=1e-10, atol=1e-12), "differs from torch"
    print(f"{equation:32s} ok  max|d| = {np.max(np.abs(ours - numpy_ref)):.2e}")


# 逐类覆盖：从最简单的矩阵乘一路到生产形状的核子 2pt
check("ij,jk->ik", complex_noise(8, 8), complex_noise(8, 8))
# 三算子链：规划器要决定先乘哪一对
check("ij,jk,kl->il", complex_noise(8, 8), complex_noise(8, 8), complex_noise(8, 8))
# 省略号批轴：同一方程作用在一批矩阵上
check("...ij,...jk->...ik", complex_noise(2, 4, 5), complex_noise(2, 5, 6))
# 重复标签：取对角元（输入输出维度不同）
check("ii->i", complex_noise(6, 6))
# 全缩并成标量：没有输出标签
check("ij,ij->", complex_noise(5, 7), complex_noise(5, 7))

# 生产形状：8 算子的核子 2pt。Nvec=8 只是为了让文档门禁秒级跑完，
# 形状关系与真实计算完全一致（perambulator、gamma、VVV、投影算符）。
n = 8
check(
    "gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M",
    complex_noise(4, 4, n, n),        # perambulator（sink）
    complex_noise(4, 4, n, n),        # perambulator
    complex_noise(4, 4, n, n),        # perambulator
    complex_noise(4, 4),              # gamma_0
    complex_noise(4, 4),              # gamma_4
    complex_noise(4, n, n, n),        # sink VVV
    complex_noise(4, n, n, n),        # source VVV
    complex_noise(4, 4),              # 投影算符
)
```

### 每次改动都会跑的检查

| 检查 | 范围 |
|---|---|
| `python -m pyqcd.testing` | 120 项契约通过，1 项因环境跳过（依赖 CUDA/TeX） |
| 差分套件（源码仓库） | 对冻结版 `opt_einsum` 的 980 个用例 |
| `python -m pyqcd.testing.docs_examples` | 全部文档代码块，每块自带断言 |
| `verify_native --backend kupl\|cuda` | 原生库 vs Python 规划器 |

### 不可破坏的不变量

* **换执行路径不改变结果**：缺库、形状被拒或 kernel 报错都回落（最多一次警告），数值一致；
* **`contract_path` 与 `opt_einsum` 一致**：显式优化器名下 path 与 `opt_cost` 相同；
* **精度保持**：`complex128` 进 `complex128` 出；
* **错误要报不要猜**，措辞与 `opt_einsum` 完全一致。

## 复现

该系列的权威生产驱动只存在于源码仓库，因为它依赖 full-scene 输入构造器与计划审计。
请在计算节点运行：

```bash
python -m pyqcd.testing.einsum.bench_opt_einsum_production --device cpu --precision double --seqs 2584,866,855,129,1863,6585 --strategies dp --rounds 1 --warmup 0 --oracle installed --out /tmp/pyqcd-cpu.json
python -m pyqcd.testing.einsum.verify_plan_gate
```

下面两个随包发布的脚本仍可用于通用路由与固定/抽样拓扑检查。参数为先
`cpu|gpu`、再 `single|double`，使用交错轮次，并在计时前校验正确性。

### 历史通用内核基准（15 种通用路径）

通用路径指**不命中** `wick_12cases` 语料、走通用分步规划器的形状：

```bash
python -m pyqcd.einsum.bench_generic cpu double
python -m pyqcd.einsum.bench_generic cpu single
python -m pyqcd.einsum.bench_generic gpu double
python -m pyqcd.einsum.bench_generic --list
```

覆盖 15 种结构类别：matmul（实/复）、逐元素、外积、全归约、部分归约、
3/4 算子链、hyper-edge、批量 GEMM、ellipsis、batch×matrix、many small
batches、trace、diagonal。

### 特化内核 smoke/抽样基准（15 条固定路径）

特化路径指**命中**语料、走拓扑特化内核的形状：

```bash
python -m pyqcd.einsum.bench_specialized cpu double
python -m pyqcd.einsum.bench_specialized cpu single
python -m pyqcd.einsum.bench_specialized gpu double
python -m pyqcd.einsum.bench_specialized --list
python -m pyqcd.einsum.bench_specialized --sample 16 --seed 0
```

覆盖 15 条取自语料的真实方程，跨 6 到 16 个算子（2pt/3pt/4pt/5pt/6pt 几何，
含与不含动量投影）。每行的 `path` 列据实报告实际执行了什么
（`kernel: …` 或 `plan-cache`），出现 `WRONG` 说明该行数字无效。支持的全部
路径见 {doc}`appendix_specialized`。

`--sample N` 换成按元数分层从 8638 条语料里随机抽 `N` 条（排除固定 15 条），
`--seed` 固定同一批抽样结果。固定 15 条只能说明"挑出来的能跑"，抽样才说明
语料整体都落在同一执行路径上。

若某行的 `path` 显示 `REF-FAIL`，是 `opt_einsum` 自己在该路径上要的中间体超出内存
（高元数路径的 dp 只按 flops 选路，不管内存；实测 seq 3186 要 95.4 GiB），该行
不给倍率、脚本继续跑后面的用例——**这不是内核错误**。

两个脚本参数一致（先 `cpu|gpu` 再 `single|double`）、计时协议一致
（交错轮次、比值取中位数、每条样本约 0.3 s、计时前先用 `opt_einsum` 校验
数值），因此数字可直接横向比较。

两个脚本随包发布（位于 `pyqcd/einsum/` 下），`pip install pyqcd` 即可使用，
无需源码仓库。

### 基准实现

```python
import statistics
import sys
import time
from functools import partial

import numpy as np
import opt_einsum as oe
import pyqcd
import pyqcd.einsum as pe

# docs-skip: 15 个用例的全尺寸基准（单次 0.4–2.7 s），文档门禁不跑
DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cpu"     # "cpu" 或 "gpu"
PRECISION = sys.argv[2] if len(sys.argv) > 2 else "double"  # "single" 或 "double"
ROUNDS = 3
REAL = np.float32 if PRECISION == "single" else np.float64
COMPLEX = np.complex64 if PRECISION == "single" else np.complex128
RTOL = 1e-5 if PRECISION == "single" else 1e-10                                               # 交错计时轮数

if DEVICE == "gpu":
    import cupy as cp

    def to_device(array):
        """把 numpy 数组搬到显存。"""
        return cp.asarray(array)

    def synchronise():
        """GPU 计时前必须同步，否则只量到 kernel 下发的时间。"""
        cp.cuda.Device(0).synchronize()

    def release_memory():
        """每个用例后清空显存池，避免上一次的残留把下一个用例挤爆。"""
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    device_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
else:
    def to_device(array):
        return array

    def synchronise():
        return None

    def release_memory():
        return None

    device_name = "cpu"

# 先打印版本与路径：site-packages 里的旧版会把每个数字都变成"PyQCD 很慢"
print("pyqcd", pyqcd.__version__, pyqcd.__file__, "|", device_name,
      "|", PRECISION)

rng = np.random.default_rng(0)


def random_array(*shape):
    """实数随机数组（float64）。"""
    return to_device(rng.standard_normal(shape).astype(REAL))


def random_complex_array(*shape):
    """复数随机数组，精度可切换：实部与虚部独立采样。"""
    real_part = rng.standard_normal(shape)
    imag_part = rng.standard_normal(shape)
    return to_device((real_part + 1j * imag_part).astype(COMPLEX))


def one_call(function, repeat=1):
    """调用 repeat 次并返回最快的一次；repeat>1 用来摊掉固定开销。"""
    synchronise()
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        synchronise()
        best = min(best, time.perf_counter() - start)
    return best


# 15 个结构类别，形状与上面两张表完全一致。
# 每行 = (标签, 方程, 造数据的函数, 各操作数的形状)
CASES = [
    ("matmul 2D (real)", "ij,jk->ik", random_array,
     [(4096, 4096), (4096, 4096)]),
    ("matmul 2D (complex)", "ij,jk->ik", random_complex_array,
     [(2048, 2048), (2048, 2048)]),
    ("element-wise", "ij,ij->ij", random_array,
     [(12000, 12000), (12000, 12000)]),
    ("outer product", "i,j->ij", random_array,
     [(8000,), (8000,)]),
    ("full reduction", "ij,ij->", random_array,
     [(12000, 12000), (12000, 12000)]),
    ("partial reduction", "ij->i", random_array,
     [(16000, 16000)]),
    ("chain 3 ops", "ij,jk,kl->il", random_array,
     [(3072, 3072), (3072, 3072), (3072, 3072)]),
    ("chain 4 ops", "ij,jk,kl,lm->im", random_array,
     [(2048, 2048), (2048, 2048), (2048, 2048), (2048, 2048)]),
    ("hyper-edge", "ij,jk,ik->i", random_array,
     [(2048, 2048), (2048, 2048), (2048, 2048)]),
    ("batched GEMM", "bij,bjk->bik", random_array,
     [(16, 1024, 1024), (16, 1024, 1024)]),
    ("ellipsis", "...ij,...jk->...ik", random_array,
     [(256, 512, 512), (256, 512, 512)]),
    ("batch x matrix", "abc,cd->abd", random_array,
     [(8, 1024, 512), (512, 1024)]),
    ("many small batches", "bij,bjk->bik", random_array,
     [(4096, 64, 64), (4096, 64, 64)]),
]

print(f"{'case':26s} {'pyqcd(s)':>9s} {'opt(s)':>9s} {'ratio':>6s}")
for label, equation, builder, shapes in CASES:
    operands = [builder(*shape) for shape in shapes]

    # 1) 先验正确性：与 opt_einsum 逐元素对照，快而不对的数字没有意义
    reference = oe.contract(equation, *operands, optimize="dp")
    got = pe.contract(equation, *operands)
    ours_value = cp.asnumpy(got) if DEVICE == "gpu" else np.asarray(got)
    their_value = cp.asnumpy(reference) if DEVICE == "gpu" else np.asarray(reference)
    scale = max(float(np.max(np.abs(their_value))), 1e-30)
    assert np.max(np.abs(ours_value - their_value), initial=0.0) <= RTOL * scale, label
    del reference, got, ours_value, their_value
    release_memory()

    # 2) 先探一次耗时，据此决定内层重复次数，让每个样本至少约 0.3 s
    ours_call = partial(pe.contract, equation, *operands)
    theirs_call = partial(oe.contract, equation, *operands, optimize="dp")
    probe = one_call(ours_call)
    inner = max(1, min(9, int(0.3 / max(probe, 1e-6))))

    # 3) 交错计时：每轮交换先后顺序，机器漂移在比值里互相抵消
    ours_times, theirs_times = [], []
    for round_index in range(ROUNDS):
        if round_index % 2 == 0:
            ours_times.append(one_call(ours_call, inner))
            theirs_times.append(one_call(theirs_call, inner))
        else:
            theirs_times.append(one_call(theirs_call, inner))
            ours_times.append(one_call(ours_call, inner))
        release_memory()

    # 4) 取中位数再相除，避免被单次抖动带偏
    median_ours = statistics.median(ours_times)
    median_theirs = statistics.median(theirs_times)
    print(f"{label:26s} {median_ours:9.3f} {median_theirs:9.3f} "
          f"{median_theirs / median_ours:6.2f}")
    del operands
    release_memory()
```

### 动手之前

```bash
python -m pyqcd --gpu-probe
```

`site-packages` 里残留的旧 `pyqcd` 会把所有数字变成"PyQCD 很慢"，脚本因此打印
`pyqcd.__version__` 与 `pyqcd.__file__`。
