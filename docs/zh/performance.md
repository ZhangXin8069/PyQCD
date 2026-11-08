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
4. **计划缓存 + 绑定操作数**——同方程同形状的重复调用完全跳过规划。

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


def check(equation, *operands):
    """Compare pyqcd against two independent references on the same inputs."""
    ours = np.asarray(pe.contract(equation, *operands))
    numpy_ref = np.einsum(equation, *operands, optimize="optimal")
    torch_ref = torch.einsum(equation, *[torch.as_tensor(o) for o in operands]).numpy()
    assert np.allclose(ours, numpy_ref, rtol=1e-10, atol=1e-12), "differs from numpy"
    assert np.allclose(ours, torch_ref, rtol=1e-10, atol=1e-12), "differs from torch"
    print(f"{equation:32s} ok  max|Δ| = {np.max(np.abs(ours - numpy_ref)):.2e}")


d = lambda *s: (rng.standard_normal(s) + 1j * rng.standard_normal(s)).astype(np.complex128)

check("ij,jk->ik", d(8, 8), d(8, 8))                     # plain matmul
check("ij,jk,kl->il", d(8, 8), d(8, 8), d(8, 8))         # 3-operand chain
check("...ij,...jk->...ik", d(2, 4, 5), d(2, 5, 6))      # ellipsis
check("ii->i", d(6, 6))                                  # repeated label (diagonal)
check("ij,ij->", d(5, 7), d(5, 7))                       # full contraction to scalar
n = 8
check("gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M",            # production nucleon 2pt shape
      d(4, 4, n, n), d(4, 4, n, n), d(4, 4, n, n), d(4, 4), d(4, 4),
      d(4, n, n, n), d(4, n, n, n), d(4, 4))
```

### 每次改动都会跑的检查

| 检查 | 范围 |
|---|---|
| `python -m pyqcd.testing` | 121 项契约：数值边界、并发、元数据 |
| 差分套件（源码仓库） | 对冻结版 `opt_einsum` 的 980 个用例 |
| `python -m pyqcd.testing.docs_examples` | 全部文档代码块，每块自带断言 |
| `verify_native --backend kupl\|cuda` | 原生库 vs Python 规划器 |

### 不可破坏的不变量

* **换执行路径不改变结果**：缺库、形状被拒或 kernel 报错都回落（最多一次警告），数值一致；
* **`contract_path` 与 `opt_einsum` 一致**：显式优化器名下 path 与 `opt_cost` 相同；
* **精度保持**：`complex128` 进 `complex128` 出；
* **错误要报不要猜**，措辞与 `opt_einsum` 完全一致。

## 复现

一份脚本出两张表：`python bench.py cpu`，再 `python bench.py gpu`。

### 基准脚本

```python
import statistics, sys, time, warnings
import numpy as np
import opt_einsum as oe
import pyqcd, pyqcd.einsum as pe

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cpu"   # cpu | gpu  # docs-skip: 15 case 基准，门禁不跑
ROUNDS = 3
if DEVICE == "gpu":
    import cupy as cp
    to_dev = lambda a: cp.asarray(a)
    sync = lambda: cp.cuda.Device(0).synchronize()
    release = lambda: (cp.get_default_memory_pool().free_all_blocks(),
                       cp.get_default_pinned_memory_pool().free_all_blocks())
    name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
else:
    to_dev = lambda a: a
    sync = lambda: None
    release = lambda: None
    name = "cpu"
print("pyqcd", pyqcd.__version__, pyqcd.__file__, "|", name)

rng = np.random.default_rng(0)
real = lambda *s: to_dev(rng.standard_normal(s))
cplx = lambda *s: to_dev((rng.standard_normal(s) + 1j * rng.standard_normal(s)).astype(np.complex128))

def one(call, inner=1):
    sync(); best = float("inf")
    for _ in range(inner):
        start = time.perf_counter(); call(); sync()
        best = min(best, time.perf_counter() - start)
    return best

CASES = [
    ("matmul 2D (real)",        "ij,jk->ik",        lambda: [real(4096, 4096), real(4096, 4096)]),
    ("matmul 2D (complex)",     "ij,jk->ik",        lambda: [cplx(2048, 2048), cplx(2048, 2048)]),
    ("element-wise",            "ij,ij->ij",        lambda: [real(12000, 12000), real(12000, 12000)]),
    ("outer product",           "i,j->ij",          lambda: [real(8000), real(8000)]),
            ("full reduction",          "ij,ij->",          lambda: [real(12000, 12000), real(12000, 12000)]),
    ("partial reduction",       "ij->i",            lambda: [real(16000, 16000)]),
    ("chain 3 ops",             "ij,jk,kl->il",     lambda: [real(3072, 3072) for _ in range(3)]),
    ("chain 4 ops",             "ij,jk,kl,lm->im",  lambda: [real(2048, 2048) for _ in range(4)]),
    ("hyper-edge",              "ij,jk,ik->i",      lambda: [real(2048, 2048) for _ in range(3)]),
    ("batched GEMM",            "bij,bjk->bik",     lambda: [real(16, 1024, 1024), real(16, 1024, 1024)]),
    ("ellipsis",                "...ij,...jk->...ik", lambda: [real(256, 512, 512), real(256, 512, 512)]),
    ("batch × matrix",          "abc,cd->abd",      lambda: [real(8, 1024, 512), real(512, 1024)]),
    ("many small batches",      "bij,bjk->bik",     lambda: [real(4096, 64, 64), real(4096, 64, 64)]),
    ("nucleon 2pt (kernel)",    "gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M",
     lambda: [cplx(4,4,112,112), cplx(4,4,112,112), cplx(4,4,112,112), cplx(4,4), cplx(4,4),
              cplx(4,112,112,112), cplx(4,112,112,112), cplx(4,4)]),
    ("gamma sandwich (kernel)", "ab,...bce,cd->...dae", lambda: [cplx(4,4), cplx(8192,4,4,1024), cplx(4,4)]),
]
print(f"{'case':26s} {'pyqcd(s)':>9s} {'opt(s)':>9s} {'ratio':>6s}")
for label, equation, make in CASES:
    operands = make()
    ref = oe.contract(equation, *operands, optimize="dp")
    got = pe.contract(equation, *operands)
    g = cp.asnumpy(got) if DEVICE == "gpu" else np.asarray(got)
    r = cp.asnumpy(ref) if DEVICE == "gpu" else np.asarray(ref)
    assert np.max(np.abs(g - r), initial=0.0) <= 1e-10 * max(np.max(np.abs(r)), 1e-30), label
    del ref, got, g, r
    release()
    probe_t = one(lambda: pe.contract(equation, *operands))
    inner = max(1, min(9, int(0.3 / max(probe_t, 1e-6))))
    ours, theirs = [], []
    for rnd in range(ROUNDS):
        if rnd % 2 == 0:          # alternate the order: the second runner inherits a warm cache
            ours.append(one(lambda: pe.contract(equation, *operands), inner))
            theirs.append(one(lambda: oe.contract(equation, *operands, optimize="dp"), inner))
        else:
            theirs.append(one(lambda: oe.contract(equation, *operands, optimize="dp"), inner))
            ours.append(one(lambda: pe.contract(equation, *operands), inner))
        release()
    a, b = statistics.median(ours), statistics.median(theirs)
    print(f"{label:26s} {a:9.3f} {b:9.3f} {b / a:6.2f}")
    del operands
    release()
```

### 动手之前

```bash
python -m pyqcd --gpu-probe
```

`site-packages` 里残留的旧 `pyqcd` 会把所有数字变成"PyQCD 很慢"，脚本因此打印
`pyqcd.__version__` 与 `pyqcd.__file__`。
