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

## 批量独立收缩（`init` + `run`）

`pyqcd.parallel.einsum` 通过 `init` + `run` 把一批**相互独立**的收缩分给多个 MPI rank。与上面几张表
不同，这里的对照对象是**同一个库的串行执行**——要回答的只是"把一队任务拆到多进程上是否
划算"。

口径：同机、`complex64`/`float32`、`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1`
（单线程基准，使并行与串行拿到相同的单核预算）、3 轮内串行与并行背靠背计时、取每轮比值的
中位数；每批结果在给倍率前都先与串行 `contract` 对照。

**规划与执行分开计时。** `init` 负责校验与 LPT 装箱——`contract_path` 就在这里跑；
`run` 负责拉起进程池、分发任务、回收结果。把两者放进同一个计时桶会把规划混进计算，
所以下表分三列报：

* **plan**（`init`）——校验 + 代价模型 + 装箱；
* **exec**（`run`）——真正的并行工作，含 `mpirun` 启动与任务/结果搬运；
* **total**——plan + exec，即每次调用都重建进程池时调用方实际等待的时间。

倍率一律以「纯串行 `contract` 循环」为分母：`×exec` 是拆分本身买到的收益，
`×total` 是端到端拿到的收益。

### 自建进程池（单进程、外部没有 `mpirun`）

| 场景 | 任务数 | 串行 | plan | exec | total | `×exec` | `×total` |
|---|---|---|---|---|---|---|---|
| 通用：大量小任务 | 400 | 25 ms | 1 ms | 282 ms | 283 ms | 0.09× | 0.09× |
| 通用：一对一 | 3 | 85 ms | <1 ms | 248 ms | 248 ms | 0.34× | 0.34× |
| 通用：偏斜 | 12 | 85 ms | <1 ms | 252 ms | 252 ms | 0.34× | 0.34× |
| sush：NN 2pt（S1） | 8 | 165 ms | 1 ms | 255 ms | 256 ms | 0.65× | 0.65× |
| sush：NN 2pt，Nvec=100（大任务） | 8 | 5310 ms | 1 ms | 2241 ms | 2242 ms | **2.37×** | **2.37×** |

读这张表的要点：

* **这里规划可以忽略**（8–400 个任务、同一条方程时约 1 ms）：代价模型按「方程 + 形状」
  只算一次，重复同一条方程不会成倍增长。混有多种形状的批次这一列会变大——实测，不要假设。
* **进程池每次调用固定花约 0.15–0.3 s**（已含在 `exec` 里）：`mpirun` 启动（约 150 ms）
  加上把所有操作数写出再读回（本机约 1.5 GB/s）。只有批次本身到秒级才赚
  （Nvec=100 那行：**2.37×**）。
* **小批次在 `×exec` 上亏**，因为固定成本占主导；装箱器没有问题，只是批次太短，
  摊不掉进程启动。
* worker 自报的纯计算时间仍可从 `state["timings"]` 取得（仅自建池），用来把搬运从计算里剥出来。

`collective=True`（每个 rank 各持一份整批输入）**已弃用且不再纳入测试**：它只在批次
本来就分布在每个 rank 上时才有意义，代价是 `rank 数 × 输入内存`（实测 134 MB 批次在
4 rank 上聚合约 1.15 GB）。

用包内 harness 复现这张表：

```bash
python -m pyqcd.testing.parallel_einsum bench --equation S1 --nvec 100 --procs 4
```

harness 会分别打印 `plan`、`exec`、`total` 三个中位数，并给出它们各自相对串行基准的倍率
以及 worker 自报的计算时间。

### 批量基准脚本

用法与上面的 15-case 基准完全一致，同样两个开关：`python bench.py cpu single`（默认）
或 `python bench.py gpu double`。`gpu` 下进程数被固定为 1——自建池会把每个操作数落盘成
numpy；因此上表是 CPU 实测，GPU 那一轮测的是批量接口在设备上的进程内开销。

```python
# docs-skip: launches MPI worker processes
import statistics
import sys
import time
from functools import partial

import numpy as np
import pyqcd
import pyqcd.einsum as pe
from pyqcd.parallel import init, run

# Same two switches as the 15-case benchmark: device first, precision second.
DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cpu"          # "cpu" or "gpu"
PRECISION = sys.argv[2] if len(sys.argv) > 2 else "single"    # "single" or "double"
ROUNDS = 3        # serial and parallel are each timed this many times; take the median ratio
# The self-launched pool stages every operand as numpy, so on the GPU only the
# in-process path (one process) measures device work; extra ranks would time the CPU.
PROCS = 4 if DEVICE == "cpu" else 1

REAL = np.float32 if PRECISION == "single" else np.float64
COMPLEX = np.complex64 if PRECISION == "single" else np.complex128
RTOL = 1e-5 if PRECISION == "single" else 1e-12

if DEVICE == "gpu":
    import cupy as cp

    def to_device(array):
        """Move a numpy array to the GPU."""
        return cp.asarray(array)

    def to_host(array):
        """Bring a device result back as numpy for the comparison."""
        return cp.asnumpy(array)

    def synchronise():
        """Synchronise before timing: otherwise you only measure kernel launch."""
        cp.cuda.Device(0).synchronize()

    device_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
else:
    def to_device(array):
        return array

    def to_host(array):
        return array

    def synchronise():
        return None

    device_name = "cpu"

# Print version and path first: a stale copy in site-packages blames the wrong build.
print("pyqcd", pyqcd.__version__, pyqcd.__file__, "|", device_name,
      "|", PRECISION, "| procs", PROCS)

rng = np.random.default_rng(0)


def random_array(*shape):
    """Real random array in the requested precision, on the requested device."""
    return to_device(rng.standard_normal(shape).astype(REAL))


def random_complex_array(*shape):
    """Complex random array: real and imaginary parts sampled separately."""
    real_part = rng.standard_normal(shape)
    imag_part = rng.standard_normal(shape)
    return to_device((real_part + 1j * imag_part).astype(COMPLEX))


# A real lattice-QCD equation: the sush distillation nucleon two-point (NN 2pt), 8 operands.
SUSH_NN_2PT = "eofp,ambn,cgdh,ce,mo,Nbdf,Mnph,ga->MN"


def sush_nn_2pt(nvec, count=8):
    """Build ``count`` NN 2pt tasks: same equation, same shapes, different data.

    Shapes follow refer/sush/_distillation/contraction.NN.2pt.py: three perambulators,
    two gamma matrices, two VVV vertices and one projector. Each task uses its own
    random numbers, mimicking the real case of "one equation, many configurations".
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
            random_complex_array(4, 4),                    # projector (g, a)
        ]
        tasks[f"S1-{index}"] = (SUSH_NN_2PT, operands)
    return tasks


def general_matmul(size, count, heavy=3):
    """Batch of plain matmuls.

    ``heavy`` sets how many of the first tasks use the large size, which creates the
    load skew: in the 12-task case only 3 tasks are heavy, so we can see whether the
    bin packer puts them on the same process or spreads them out.
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
    "general: many small": general_matmul(128, 400),    # many tasks, tiny each (the target case)
    "general: one-to-one": general_matmul(1200, 3),     # tasks == processes, one each
    "general: skewed": general_matmul(1200, 12),        # 3 heavy tasks + 9 light tasks
    "sush: NN 2pt": sush_nn_2pt(40),                    # real distillation equation
    "sush: NN 2pt large": sush_nn_2pt(100),             # same, with a larger single task
}


def timed(function, rounds=ROUNDS):
    """Run ``rounds`` times and return the median duration plus the last value."""
    durations = []
    result = None
    for _ in range(rounds):
        synchronise()
        start = time.perf_counter()
        result = function()
        synchronise()
        durations.append(time.perf_counter() - start)
    return statistics.median(durations), result


def run_serial(tasks):
    """Call contract one task at a time — this is the baseline parallel has to beat."""
    return [pe.contract(equation, *operands)
            for equation, operands in tasks.values()]


def output_shape(equation, operands):
    """Shape of the contraction output, used to lay out the result buffers."""
    shapes = tuple(tuple(int(n) for n in operand.shape) for operand in operands)
    _, info = pe.contract_path(equation, *shapes, shapes=True, optimize="greedy")
    return tuple(info.size_dict[label] for label in info.output_subscript)


def run_parallel(tasks):
    """One ``init`` + ``run`` pair, timed in two phases.

    The buffers stay host-side on purpose: the self-launched pool stages numpy, and
    `init`/`run` write results in place, so a host buffer keeps both devices honest.
    """
    keys = list(tasks)
    results = [np.zeros(output_shape(*tasks[key]),
                        dtype=np.result_type(*[o.dtype for o in tasks[key][1]]))
               for key in keys]
    config = {"task_id": keys,
              "equation": [tasks[key][0] for key in keys],
              "operands": [list(tasks[key][1]) for key in keys],
              "results": results}

    synchronise()
    start = time.perf_counter()
    state = init(config, max_proc=PROCS, reminder=False)    # planning only
    plan_seconds = time.perf_counter() - start
    synchronise()

    start = time.perf_counter()
    run(state)                                             # execution only
    synchronise()
    exec_seconds = time.perf_counter() - start

    return ({key: results[index] for index, key in enumerate(keys)},
            plan_seconds, exec_seconds)


def timed_parallel(tasks, rounds=ROUNDS):
    """Median of the separately measured plan and exec times, plus the last result."""
    plans, execs = [], []
    results = None
    for _ in range(rounds):
        results, plan_seconds, exec_seconds = run_parallel(tasks)
        plans.append(plan_seconds)
        execs.append(exec_seconds)
    return statistics.median(plans), statistics.median(execs), results


print(f"{'case':24s} {'tasks':>5s} {'serial(s)':>9s} {'plan(s)':>8s} "
      f"{'exec(s)':>8s} {'total(s)':>8s} {'x exec':>7s} {'x total':>8s}")
for label, tasks in CASES.items():
    serial_seconds, _ = timed(partial(run_serial, tasks))
    plan_seconds, exec_seconds, results = timed_parallel(tasks)
    total_seconds = plan_seconds + exec_seconds

    # Correctness first: the parallel result must equal the serial one, otherwise the
    # speed-up means nothing.
    for key, (equation, operands) in tasks.items():
        reference = to_host(pe.contract(equation, *operands))
        scale = max(float(np.max(np.abs(reference))), 1e-30)
        deviation = float(np.max(np.abs(results[key] - reference))) / scale
        assert deviation <= RTOL, f"{label} {key}: rel={deviation:.1e}"

    print(f"{label:24s} {len(tasks):5d} {serial_seconds:9.3f} {plan_seconds:8.3f} "
          f"{exec_seconds:8.3f} {total_seconds:8.3f} "
          f"{serial_seconds / exec_seconds:6.2f}x "
          f"{serial_seconds / total_seconds:7.2f}x")

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
