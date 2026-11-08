# 性能

`pyqcd.einsum` 与 `opt_einsum` 在同一台机器、同一 dtype、同一线程设置下对比，并且两种实现
**在同一循环里交错执行**，以抵消机器漂移。下面每个数字都是实测，每张表都附带产生它的脚本。

## 测量口径

| 项 | 设置 |
|---|---|
| 对照 | `opt_einsum` 3.4.0，`dp` 策略 |
| dtype | 注明处为 `complex128`，其余 `float64`（双方一致） |
| 计时 | 预热 1 次，然后 **3 次交错计时**；取每轮比值的中位数 |
| 尺寸 | 生产表：单次 **0.4–2.7 s**；小尺寸表：1–300 ms |
| 正确性 | 每条用例先与 `numpy.einsum` 对齐，再计时 |
| 机器 | 4 物理核（8 线程）+ RTX 4060 Laptop，8 GB |

## 生产尺度结果

尺寸控制在"单次 0.4–2.7 s"——更短的运行测到的是缓存预热与计时分辨率，而不是收缩本身，
也绕开了生产形状所处的 BLAS 分块区间。

### CPU（KUPL 后端，`numpy` 数组）

| 用例 | `pyqcd.einsum` | `opt_einsum`（`dp`） | 倍率 |
|---|---|---|---|
| 批量矩阵乘 32×1024² | 0.61 s | 6.98 s | **11.6×** |
| ellipsis 256×512² | 0.91 s | 6.48 s | **7.1×** |
| Hadamard 16000² | 0.44 s | 0.55 s | **1.26×** |
| 矩阵乘 4600² | 1.05 s | 1.06 s | 1.01× |
| 三操作数链 3×3570 | 1.00 s | 0.98 s | 0.98× |
| 复数矩阵乘 3100² | 1.36 s | 1.34 s | 0.96× |

### GPU（CUDA 后端，`cupy` 数组）

GPU 是**同卡对同卡**：两边都在同一块设备上执行，倍率反映实现差异，不是 CPU 与 GPU 的算力差。

| 用例 | `pyqcd.einsum` | `opt_einsum`（`dp`） | 倍率 |
|---|---|---|---|
| γ 夹心，8192×1024（融合 kernel） | 0.76 s | 1.54 s | **2.0×** |
| 批量 GEMM 16×1024² | 0.75 s | 0.74 s | 0.98× |
| 核子 2pt，nvec=112（kernel） | 2.67 s | 2.65 s | 0.99× |

两点如实说明：γ 夹心这一行不再是小尺寸上的 7–16×，因为在 8192×1024 下融合 kernel 受
2.1 GB 工作区带宽限制，而不是算力；核子 2pt 在 nvec=112 上已降到持平——该形态此时由
cuBLAS 主导，kernel 已无余量。

## 小尺寸结果

只用于定位问题，不要外推：这个尺度上数字由每次调用的固定开销主导。

### CPU

| 用例 | `pyqcd.einsum` | `opt_einsum`（`dp`） | 倍率 |
|---|---|---|---|
| 批量矩阵乘 32×256² | 9.3 ms | 97.1 ms | **10.6×** |
| ellipsis 16×128×128 | 1.1 ms | 6.3 ms | **5.8×** |
| 三操作数链 3×500 | 3.1 ms | 3.2 ms | 1.03× |
| 复数矩阵乘 800² | 20.4 ms | 21.0 ms | 1.03× |
| Hadamard 4000² | 30.1 ms | 30.2 ms | 1.00× |
| 外积 2000×2000 | 4.1 ms | 4.1 ms | 1.00× |
| 四操作数链 300 | 1.15 ms | 1.11 ms | 1.00× |
| 矩阵乘 1000² | 10.7 ms | 10.2 ms | 0.97× |

### GPU

| 用例 | `pyqcd.einsum` | `opt_einsum`（`dp`） | 倍率 |
|---|---|---|---|
| γ 夹心 512×512 | 3.3 ms | 53.1 ms | **16.2×** |
| 核子 2pt，nvec=64（kernel） | 146 ms | 263 ms | **1.81×** |
| 批量 GEMM 32×512² | 178 ms | 167 ms | 0.94× |

## 什么情况走哪条路径

| 输入 / 形态 | 执行位置 | 说明 |
|---|---|---|
| 逐元素、转置、对角、迹、外积 | backend 自己的 `einsum` | 没有求和标签，规划器无事可做 |
| `numpy` + batch 标签（被两个以上操作数共享**且**出现在输出里） | KUPL 原生计划执行器 | 一次批量 GEMM 取代 numpy 的逐 batch 循环 |
| `numpy` + 无 batch 标签 | numpy 分步（tensordot） | 原生执行器在这里慢 6–10% |
| `cupy` + γ 夹心 | CUDA 原生执行器 | 目前唯一的融合 CUDA kernel |
| `cupy` + 其它通用形态 | `cupy.einsum`（cuBLAS） | 原生批量 GEMM 在 16×1024² 上慢约 5 倍 |
| 核子 2pt / 3pt 主项形态 | KUPL/CUDA 专用 kernel | 命中提示里显示 `[kernel: …]` |

命中提示会告诉你实际走了哪条路：`[kernel: kupl_nucleon_2pt]` 表示跑了专用 kernel，
`[plan-cache]` 表示只复用了计划。如果注册了 kernel 的形态拿不到 kernel，
`pyqcd.einsum` 会**立即告警并说明原因**，而不是静默降级。

## 增益从哪来

1. **形态特化**——核子 2pt/3pt 主项有专用 kernel；CUDA 执行器有融合的 γ 夹心 kernel
   （体积门限制：`z·y·x ≤ 1`）。
2. **不做无用规划**——逐元素/置换类方程、以及通用 `cupy` 方程直接交回 backend。
3. **批量执行**——存在 batch 标签时，一次批量 GEMM 取代 numpy 的逐 batch 循环
   （就是上面 7–12× 那一列）。
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
* **`contract_path` 与 `opt_einsum` 一致**：path 与 `opt_cost` 相同；
* **精度保持**：`complex128` 进 `complex128` 出；
* **错误要报不要猜**，措辞与 `opt_einsum` 完全一致。

## 复现

### CPU，生产尺度

```python
import statistics
import time

import numpy as np
import opt_einsum as oe
import pyqcd

import pyqcd.einsum as pe

print("measuring pyqcd", pyqcd.__version__, pyqcd.__file__)   # docs-skip: 生产尺度基准，单次 0.4–2.7 s，门禁不跑


def one(call):
    start = time.perf_counter()
    call()
    return time.perf_counter() - start


def compare(ours, theirs, rounds=3):
    """One warm-up call, then `rounds` interleaved timed calls (seconds)."""
    ours()
    theirs()
    left, right, ratios = [], [], []
    for _ in range(rounds):
        a = one(ours)
        b = one(theirs)
        left.append(a)
        right.append(b)
        ratios.append(b / a)
    return statistics.median(left), statistics.median(right), statistics.median(ratios)


rng = np.random.default_rng(0)
real = lambda *s: rng.standard_normal(s)
cplx = lambda *s: (rng.standard_normal(s) + 1j * rng.standard_normal(s)).astype(np.complex128)

# Sized so that one call takes roughly 0.5-1.5 s: shorter runs measure timer and
# cache noise, not the contraction, and they hide the BLAS register-blocking regime.
CASES = [
    ("matmul 4600^2", "ij,jk->ik", [real(4600, 4600), real(4600, 4600)]),
    ("chain 3x3570", "ij,jk,kl->il", [real(3570, 3570)] * 3),
    ("batched 32x1024^2", "bij,bjk->bik", [real(32, 1024, 1024), real(32, 1024, 1024)]),
    ("ellipsis 256x512^2", "...ij,...jk->...ik",
     [real(256, 512, 512), real(256, 512, 512)]),
    ("complex mm 3100^2", "ij,jk->ik", [cplx(3100, 3100), cplx(3100, 3100)]),
    ("hadamard 16000^2", "ij,ij->ij", [real(16000, 16000), real(16000, 16000)]),
]

print(f"{'case':22s} {'pyqcd(s)':>9s} {'opt(s)':>8s} {'ratio':>6s}")
for label, equation, operands in CASES:
    reference = np.einsum(equation, *operands, optimize="optimal")
    assert np.allclose(np.asarray(pe.contract(equation, *operands)), reference), label
    ours, theirs, ratio = compare(lambda: pe.contract(equation, *operands),
                                  lambda: oe.contract(equation, *operands, optimize="dp"))
    print(f"{label:22s} {ours:9.2f} {theirs:8.2f} {ratio:6.2f}")
```

### GPU，生产尺度

```python
import statistics
import time
import warnings

import numpy as np
import pyqcd

import cupy as cp
import opt_einsum as oe
import pyqcd.einsum as pe

print("measuring pyqcd", pyqcd.__version__, pyqcd.__file__, "|",  # docs-skip: GPU 基准
      cp.cuda.runtime.getDeviceProperties(0)["name"].decode())


def one(call):
    call()
    cp.cuda.Device(0).synchronize()
    start = time.perf_counter()
    call()
    cp.cuda.Device(0).synchronize()
    return time.perf_counter() - start


def compare(ours, theirs, rounds=3):
    left, right, ratios = [], [], []
    for _ in range(rounds):
        a = one(ours)
        b = one(theirs)
        left.append(a)
        right.append(b)
        ratios.append(b / a)
    return statistics.median(left), statistics.median(right), statistics.median(ratios)


rng = np.random.default_rng(0)


def g(*shape):
    return cp.asarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


n = 112                                                  # production-shaped nucleon 2pt
nucleon = "gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M"
nucleon_ops = [g(4, 4, n, n), g(4, 4, n, n), g(4, 4, n, n), g(4, 4), g(4, 4),
               g(4, n, n, n), g(4, n, n, n), g(4, 4)]
sandwich_ops = [g(4, 4), g(8192, 4, 4, 1024), g(4, 4)]
batched_ops = [g(16, 1024, 1024), g(16, 1024, 1024)]

CASES = [("nucleon 2pt nvec=112", nucleon, nucleon_ops),
         ("gamma sandwich 8192x1024", "ab,...bce,cd->...dae", sandwich_ops),
         ("batched GEMM 16x1024^2", "bij,bjk->bik", batched_ops)]

print(f"{'case':26s} {'pyqcd(s)':>9s} {'opt(s)':>8s} {'ratio':>6s}")
for label, equation, operands in CASES:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = cp.asnumpy(oe.contract(equation, *operands, optimize="dp"))
        assert np.allclose(cp.asnumpy(pe.contract(equation, *operands)), reference,
                           rtol=1e-10, atol=1e-12), label
        ours, theirs, ratio = compare(
            lambda: pe.contract(equation, *operands),
            lambda: oe.contract(equation, *operands, optimize="dp"))
    print(f"{label:26s} {ours:9.2f} {theirs:8.2f} {ratio:6.2f}")
```

动手之前先确认你测的是哪一份构建——`site-packages` 里残留的旧 `pyqcd` 会把所有数字
变成"PyQCD 很慢"：

```bash
python -m pyqcd --gpu-probe
```
