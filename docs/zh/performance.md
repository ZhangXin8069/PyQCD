# 性能

`pyqcd.einsum` 与 `opt_einsum` 在同一台机器、同一 dtype、同一线程设置下对比，两种实现
**每轮交替谁先运行**，以抵消缓存与时钟漂移。下面每个数字都是实测，每张表都由同一份脚本产生。

## 测量口径

| 项 | 设置 |
|---|---|
| 对照 | `opt_einsum` 3.4.0，`dp` 策略 |
| dtype | 注明处为 `complex128`，其余 `float64`（双方一致） |
| 计时 | 预热 1 次；每点 3 轮，**单次不足 0.3 s 的用例在轮内重复取最好值** |
| 顺序 | 每轮交替"谁先跑"，否则后跑的一方白拿 10–20% 的假优势 |
| 正确性 | 每条用例先按相对最大范数与 `opt_einsum` 对齐，再计时 |
| 机器 | 4 物理核（8 线程）+ RTX 4060 Laptop，8 GB |

## case 覆盖

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

### CPU（`numpy` 数组，KUPL 后端）

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

### GPU（`cupy` 数组，CUDA 后端）

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

## 为什么有的行是 10–19×，有的只有 1.0×

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

## 增益从哪来

1. **形态特化**——核子 2pt/3pt 主项有专用 kernel；CUDA 执行器有融合的 γ 夹心 kernel
   （体积门限制：`z·y·x ≤ 1`）。
2. **不做无用规划**——逐元素/置换类方程、以及通用 `cupy` 方程直接交回 backend。
3. **批量执行**——存在 batch 标签时，一次批量 GEMM 取代 numpy 的逐 batch 循环。
4. **计划缓存 + 绑定操作数**——三个以上操作数的方程会缓存收缩列表，重复调用完全跳过
   规划。小/中型图上规划占总调用时间的 31–84%（8 算子链 0.86 ms 里占 0.76 ms），
   解析只占 0.002 ms，所以重复调用的收益主要来自这里。

## 重复调用

同方程、同形状、同 dtype、同后端的第二次及以后调用会跳过规划（三个以上操作数；
两算子只有一步，建键比规划更贵）。每次调用中位耗时，8×8 算子：

| 用例 | `pyqcd.einsum` | `opt_einsum`（`dp`） | 倍率 |
|---|---|---|---|
| 8 算子链 `ab,bc,cd,de,ef,fg,gh,hi->ai` | 95 µs | 291 µs | **3.07×** |
| 3 算子链 `ij,jk,kl->il` | 51 µs | 72 µs | **1.41×** |
| 混合秩 `ef,b,e->be` | 52 µs | 60 µs | **1.14×** |

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
| `python -m pyqcd.testing` | 124 项契约：数值边界、并发、元数据 |
| 差分套件（源码仓库） | 对冻结版 `opt_einsum` 的 980 个用例 |
| `python -m pyqcd.testing.docs_examples` | 全部文档代码块，每块自带断言 |
| `verify_native --backend kupl\|cuda` | 原生库 vs Python 规划器 |

### 不可破坏的不变量

* **换执行路径不改变结果**：缺库、形状被拒或 kernel 报错都回落（最多一次警告），数值一致；
* **`contract_path` 与 `opt_einsum` 一致**：显式优化器名下 path 与 `opt_cost` 相同；
* **精度保持**：`complex128` 进 `complex128` 出；
* **错误要报不要猜**，措辞与 `opt_einsum` 完全一致。

## 批量独立收缩（`einsums`）

`pyqcd.parallel.einsum.einsums` 把一批**相互独立**的收缩分给多个 MPI rank。与上面几张表
不同，这里的对照对象是**同一个库的串行执行**——要回答的只是"把一队任务拆到多进程上是否
划算"。

口径：同机、`complex64`/`float32`、`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1`
（单线程基准，使并行与串行拿到相同的单核预算）、3 轮内串行与并行背靠背计时、取每轮比值的
中位数；每批结果在给倍率前都先与串行 `contract` 对照。

**两种口径都报。** `einsums(..., return_info=True)` 会带回 worker 自报的纯计算时间
（`info["timings"]`），所以自建进程池可以按两种方式计费：

* **计入初始化**（`×wall`）：调用方实际等待的时间，含 `mpirun` 启动、任务/结果落盘与计算。
  报端到端成本时用这个数。
* **不计入初始化**（`×compute`）：只算 worker 实测的收缩时间。这是任务拆分本身能达到的
  上限（进程池已存在时成立，例如你本来就在 `mpirun` 里，或同一个池要跑大得多的批次）。

### 已在 `mpirun -np 4` 内（collective，无数据搬运）

| 场景 | 任务数 | 串行 | 并行 | `×wall` | `×compute` |
|---|---|---|---|---|---|
| 通用：大量小任务（128² 矩阵乘） | 400 | 42.5 ms | 25.7 ms | 1.65× | 3.81× |
| 通用：一对一（1200² 矩阵乘，任务数 = rank 数） | 3 | 101.0 ms | 41.8 ms | 2.42× | 2.97× |
| 通用：偏斜（3 个重 + 9 个轻） | 12 | 103.8 ms | 37.3 ms | 2.78× | 3.21× |
| 专用核：核子 2pt（E1） | 8 | 184.4 ms | 46.1 ms | **4.00×** | **4.07×** |
| sush：NN 2pt（S1） | 8 | 197.4 ms | 50.6 ms | **3.90×** | 3.96× |
| sush：3pt 主项（P1） | 8 | 134.7 ms | 39.7 ms | **3.39×** | 3.51× |
| sush：NN 2pt，Nvec=100（大任务） | 8 | 6066.8 ms | 1536.7 ms | **3.95×** | 3.95× |
| 专用核：γ 夹心（E5，约 1 ms/任务） | 8 | 0.5 ms | 0.7 ms | 0.73× | 1.59× |

这里两种口径几乎重合（搬运为零，`×compute` 只少了计划广播那次通信），这正是推荐
collective 的原因。

> **内存警告：collective 模式会让每个 rank 各持一份整批输入。**
> 规划发生在 `einsums` 内部，所以每个 rank 都必须先构造**整批**任务：输入占用
> 变成「单份 × rank 数」。rank 0 另外持有**全部结果**，`broadcast=True` 时每个 rank
> 还会各持一份完整结果。用 `resource.ru_maxrss` 在 float32 矩阵乘上实测
> （整批输入 134 MB，64 个 512² 任务）：串行峰值 **234 MB**，collective 4 rank
> 聚合约 **1.15 GB**（`4 × 输入 + 输出`）。在 GPU 上这意味着每张可见卡各放一整批——
> 让 rank 数与可见设备数相等，否则显存成倍放大。批次放不下时改用自建进程池：
> 任务经磁盘分发，worker 只读自己那一箱，代价是每次调用多 0.15–0.3 s 固定开销。
> `einsums` 每次进入 collective 都会打印这条警告，与 `reminder=` 无关。

### 自建进程池（单进程、外部没有 `mpirun`）

| 场景 | 任务数 | 串行 | 池 wall | 池 compute | `×wall` | `×compute` |
|---|---|---|---|---|---|---|
| 通用：大量小任务 | 400 | 38.7 ms | 283.3 ms | 14.1 ms | 0.14× | 2.74× |
| 通用：一对一 | 3 | 77.8 ms | 206.9 ms | 28.3 ms | 0.38× | 2.74× |
| 通用：偏斜 | 12 | 80.0 ms | 222.3 ms | 30.4 ms | 0.36× | 2.64× |
| 专用核：核子 2pt（E1） | 8 | 142.4 ms | 337.2 ms | 162.2 ms | 0.42× | 0.88× |
| sush：NN 2pt（S1） | 8 | 140.0 ms | 218.7 ms | 57.2 ms | 0.64× | **2.45×** |
| sush：3pt 主项（P1） | 8 | 101.6 ms | 327.6 ms | 146.9 ms | 0.31× | 0.69× |
| sush：NN 2pt，Nvec=100（大任务） | 8 | 4997.0 ms | 1879.7 ms | 1586.8 ms | **2.66×** | **3.15×** |
| 专用核：γ 夹心（E5，约 1 ms/任务） | 8 | 0.6 ms | 156.7 ms | 6.4 ms | 0.00× | 0.10× |

读这张表的三个要点：

* **进程池每次调用固定花约 0.15–0.3 s**（`wall − compute`）：`mpirun` 启动（约 150 ms）
  加上把所有操作数写出再读回（本机约 1.5 GB/s）。只有批次本身到秒级，端到端才赚
  （Nvec=100 那行：**2.66×**）。
* **`×compute` 反映拆分本身的能力。** 剔掉初始化后，小批次能到 2.6–2.7×——说明装箱器
  在工作，损失纯粹来自固定成本。
* **专用核方程在 `×compute` 上低于 1×**，因为每个新 worker 的首次调用要付一次性
  kernel/计划预热（实测 E1 首调 140 ms vs 热调 17 ms）。跑完一批就退出的池里这是真实成本，
  长期驻留的池里会被摊薄。

两个口径都能用包内 harness 复现：

```bash
python -m pyqcd.testing.parallel_einsum bench --equation S1 --nvec 100 --procs 4
mpirun -np 4 python -m pyqcd.testing.parallel_einsum bench-collective --equation S1
```

自建池 harness 会同时打印 `speedup … (init included)` 与
`speedup … (worker compute only, init excluded)`，并给出实测池开销。

### 批量基准脚本

```python
# docs-skip: 会拉起 mpirun worker 进程
import statistics
import time
from functools import partial

import numpy as np
import pyqcd.einsum as pe
from pyqcd.parallel import einsums

ROUNDS = 3        # 串行与并行各测几轮，最后取每轮比值的中位数
PROCS = 4         # 自建进程池的进程数；已在 mpirun 内运行时由 MPI rank 数决定

rng = np.random.default_rng(0)


def random_array(*shape):
    """单精度实数随机数组：受限机器上用 float32 就够暴露并行收益。"""
    return rng.standard_normal(shape).astype(np.float32)


def random_complex_array(*shape):
    """单精度复数随机数组。"""
    real_part = rng.standard_normal(shape)
    imag_part = rng.standard_normal(shape)
    return (real_part + 1j * imag_part).astype(np.complex64)


# 真实格点 QCD 方程：sush 蒸馏的核子两关联（NN 2pt），8 个操作数
SUSH_NN_2PT = "eofp,ambn,cgdh,ce,mo,Nbdf,Mnph,ga->MN"


def sush_nn_2pt(nvec, count=8):
    """造 count 个 NN 2pt 任务：同一方程、同一形状、不同数据。

    形状照 refer/sush/_distillation/contraction.NN.2pt.py：
    三个 perambulator + 两个 gamma 矩阵 + 两个 VVV 顶点 + 一个投影算符。
    每个任务用独立随机数，模拟"同一方程、不同组态"的真实批量。
    """
    tasks = {}
    for index in range(count):
        operands = [
            random_complex_array(4, 4, nvec, nvec),        # perambulator
            random_complex_array(4, 4, nvec, nvec),        # perambulator
            random_complex_array(4, 4, nvec, nvec),        # perambulator
            random_complex_array(4, 4),                    # gamma_7
            random_complex_array(4, 4),                    # gamma_7
            random_complex_array(1, nvec, nvec, nvec),     # sink VVV
            random_complex_array(1, nvec, nvec, nvec),     # source VVV
            random_complex_array(4, 2),                    # 投影算符（前两列）
        ]
        tasks[f"S1-{index}"] = (SUSH_NN_2PT, operands)
    return tasks


def general_matmul(size, count, heavy=3):
    """通用矩阵乘批次。

    ``heavy`` 控制前几个任务使用大尺寸，用来制造负载偏斜：
    中间那组 12 个任务里只有 3 个是重的，看装箱器会不会把重的堆在同一个进程上。
    """
    tasks = {}
    for index in range(count):
        side = size if index < heavy else max(64, size // 6)
        tasks[f"m{index}"] = (
            "ij,jk->ik",
            [random_array(side, side), random_array(side, side)],
        )
    return tasks


CASES = {
    "general: many small": general_matmul(128, 400),    # 大批量、小单量（本功能的目标场景）
    "general: one-to-one": general_matmul(1200, 3),     # 任务数 = 进程数，一对一
    "general: skewed": general_matmul(1200, 12),        # 3 个重任务 + 9 个轻任务
    "sush: NN 2pt": sush_nn_2pt(40),                    # 真实蒸馏方程
    "sush: NN 2pt large": sush_nn_2pt(100),             # 同上，单任务更大
}


def timed(function, rounds=ROUNDS):
    """跑 rounds 轮，返回耗时中位数与最后一次的返回值。"""
    durations = []
    result = None
    for _ in range(rounds):
        start = time.perf_counter()
        result = function()
        durations.append(time.perf_counter() - start)
    return statistics.median(durations), result


def run_serial(tasks):
    """逐个串行调用 contract —— 这就是并行要打败的基准。"""
    return [pe.contract(equation, *operands)
            for equation, operands in tasks.values()]


def run_parallel(tasks):
    """一次调用交给 einsums；返回与输入同键的字典。"""
    return einsums(tasks, max_procs=PROCS, reminder=False, verbose=False)


print(f"{'case':24s} {'tasks':>5s} {'serial(s)':>9s} "
      f"{'parallel(s)':>11s} {'speed-up':>8s}")
for label, tasks in CASES.items():
    serial_seconds, _ = timed(partial(run_serial, tasks))
    parallel_seconds, results = timed(partial(run_parallel, tasks))

    # 先验正确性：并行结果必须等于串行结果，否则倍率没有意义
    for key, (equation, operands) in tasks.items():
        reference = pe.contract(equation, *operands)
        scale = max(float(np.max(np.abs(reference))), 1e-30)
        deviation = float(np.max(np.abs(results[key] - reference))) / scale
        assert deviation <= 1e-5, f"{label} {key}: rel={deviation:.1e}"

    print(f"{label:24s} {len(tasks):5d} {serial_seconds:9.3f} "
          f"{parallel_seconds:11.3f} {serial_seconds / parallel_seconds:7.2f}x")
```

## 复现

一份脚本出两张表：`python bench.py cpu`，再 `python bench.py gpu`。

### 基准脚本

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
ROUNDS = 3                                               # 交错计时轮数

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
print("pyqcd", pyqcd.__version__, pyqcd.__file__, "|", device_name)

rng = np.random.default_rng(0)


def random_array(*shape):
    """实数随机数组（float64）。"""
    return to_device(rng.standard_normal(shape))


def random_complex_array(*shape):
    """复数随机数组（complex128）：实部与虚部独立采样。"""
    real_part = rng.standard_normal(shape)
    imag_part = rng.standard_normal(shape)
    return to_device((real_part + 1j * imag_part).astype(np.complex128))


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
    ("nucleon 2pt (kernel)", "gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M",
     random_complex_array,
     [(4, 4, 112, 112), (4, 4, 112, 112), (4, 4, 112, 112), (4, 4), (4, 4),
      (4, 112, 112, 112), (4, 112, 112, 112), (4, 4)]),
    ("gamma sandwich (kernel)", "ab,...bce,cd->...dae", random_complex_array,
     [(4, 4), (8192, 4, 4, 1024), (4, 4)]),
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
    assert np.max(np.abs(ours_value - their_value), initial=0.0) <= 1e-10 * scale, label
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
