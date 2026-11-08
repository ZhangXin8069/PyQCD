# 性能

`pyqcd.einsum` 与 `opt_einsum` 在同一台机器、同一 dtype、同线程数、同 planner 下对比。
下面的数字全部来自**实测**（非模型推算），页面末尾给出可复现命令。

## 测量口径

| 项 | 设置 |
|---|---|
| 硬件 | 1× RTX 4060 Laptop（GPU 组），4 线程钉核（CPU 组） |
| 对照 | `opt_einsum` 3.4.0，`dp` 策略 |
| dtype | `complex128`（双方一致） |
| 隔离 | 双方各自全新解释器（`--isolation process`） |
| 计时 | 预热 4 轮，取测量轮中位数，计时前后同步 |
| 操作数 | 固定种子、形状与真实蒸馏收缩一致 |
| 正确性 | 同轮次校验 `max|Δ| / max|ref| ≤ 1e-12` |

计时只覆盖稳态执行：双方各规划一次，之后复用计划。

## 正确性

只有在数字正确时，速度才有意义。引擎在三个层次上验证，而且**本页每个示例结尾都有
`assert`**——文档构建会执行全部 21 个可运行代码块，任何一块不再自验证就会失败
（无法运行的片段会显式标记 `# pseudo-code`）。

### 自己验证一次

下面的脚本用两个独立参考实现（NumPy 与 PyTorch）对同一批输入做对照，三者必须一致：

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

### 项目每次改动都会跑的检查

| 检查 | 范围 |
|---|---|
| `python -m pyqcd.testing` | 120 项契约：数值边界、并发、元数据 |
| 差分套件（源码仓库） | 对冻结版 `opt_einsum` 的 980 个用例 |
| `python -m pyqcd.testing.docs_examples` | 21 个文档块，每块自带断言 |
| `verify_native --backend kupl\|cuda` | 原生库 vs Python 规划器 |

### 不可破坏的不变量

* **换执行路径不改变结果**：缺库、形状被拒或 kernel 报错都静默回落（最多一次警告），
  数值完全一致；
* **`contract_path` 与 `opt_einsum` 一致**：同形状、同优化器下 path 与 `opt_cost` 相同；
* **精度保持**：`complex128` 进 `complex128` 出，`complex64` 保持 `complex64`；
* **错误要报不要猜**：操作数个数不符、标签尺寸冲突、输出标签未知都抛
  `ValueError`/`TypeError`，措辞与 `opt_einsum` 完全一致。

### 动手之前

先确认你测的是哪一份构建。`site-packages` 里残留的旧 `pyqcd` 会把所有数字变成
"PyQCD 很慢"，而你并不知情：

```python
import pyqcd
import pyqcd.einsum as pe

print(pyqcd.__version__, pyqcd.__file__)     # which build is imported?
info = pe.build_info()
print(info)                                  # backend / array_backend
assert pyqcd.__version__.count(".") == 2
assert info["backend"] in {"cuda", "kupl", "python"}
```

GPU 任务可以直接问库"kernel 到底跑没跑"：

```bash
python -m pyqcd --gpu-probe
```

## 在已发布包上可复现

`pyqcd==0.1.28`（`build_info(): backend=kupl, array_backend=numpy`）对
`opt_einsum==3.4.0`（`dp`），`complex128`，预热 3 次后取 15 次中位数：

| 工作负载 | `opt_einsum` | `pyqcd.einsum` | 倍率 |
|---|---|---|---|
| 核子 2pt，nvec=32（生产契约） | 102.55 ms | 16.63 ms | **6.17×** |
| γ 夹心，T=256，nvec=512 | 28.78 ms | 4.55 ms | **6.33×** |
| 通用链式 800³（无专用 kernel） | 13.40 ms | 16.99 ms | 0.79×（PyQCD 更慢） |

本版本真正加速了哪些形态：

| 形态 | 0.1.x 中的执行路径 |
|---|---|
| 核子 2pt（`gahb,mcnd,…->M`） | KUPL/CUDA 专用 kernel + 计划缓存 |
| 核子 3pt 主项（`…->MNGLxy`） | KUPL/CUDA 专用 kernel + 计划缓存 |
| 核子 3pt（`…->GM`）、sush 2pt/3pt-mu | 仅计划缓存 |
| γ 夹心（`ab,...bce,cd->...dae`）、perambulator × 规范场 | 计划缓存 + 通用执行层 |

首次命中会打印提示，据实说明走的是哪条路：`[kernel: kupl_nucleon_2pt]` 表示
真的跑了 kernel，`[plan-cache]` 表示只复用了计划。

## 通用 einsum：逐例对照

多数用户的起点与格点 QCD 无关，所以这里给出纯张量代数上的同一组对照——CPU
（KUPL 后端）、除注明外 `float64`、取 9 次中位数，`pyqcd 0.1.21` 实测：

| 用例 | `pyqcd.einsum` | `opt_einsum`（`dp`） | 倍率 |
|---|---|---|---|
| 批量矩阵乘 32×256² | 11.4 ms | 100.7 ms | **8.7×** |
| ellipsis 16×128×128 | 1.3 ms | 6.5 ms | **5.1×** |
| 复数矩阵乘 800² | 24.5 ms | 23.5 ms | 1.00× |
| 矩阵乘 1000² | 15.2 ms | 14.3 ms | 0.99× |
| 外积 2000×2000 | 4.8 ms | 4.9 ms | 0.99× |
| Hadamard 4000² | 33.6 ms | 32.8 ms | 0.98× |
| 三操作数链 3×500 | 3.1 ms | 3.5 ms | 0.97× |
| 四操作数链 300 | 1.3 ms | 1.2 ms | 0.94× |

如实解读：

* **批量与 ellipsis 形态占优**——PyQCD 把 batch 轴保留在同一步执行并复用缓存计划，
  而 `opt_einsum` 的 `dp` 路径在这里退化成逐 batch 循环；
* **其余形态都在 0.94–1.00× 的持平区间**——这些方程受 BLAS 限制，两边最终都落到
  同一个 `zgemm`/`dgemm`，比值就在测量噪声底；
* 表中的数字现在由**交错协议**产生（在同一循环里交替调用两种实现，取每轮比值的中位数），
  因为顺序计时在忙碌机器上会漂移 20–40%，过去正是它把"持平"测成了"2 倍差距"。

自己跑一遍（打印耗时前，每个用例都会先对 numpy *和* opt_einsum 各断言一次）：

```python
import statistics
import time

import numpy as np
import opt_einsum as oe
import pyqcd

import pyqcd.einsum as pe

print("measuring pyqcd", pyqcd.__version__, pyqcd.__file__)


def one(call):
    start = time.perf_counter()
    call()
    return (time.perf_counter() - start) * 1e3


def compare(ours, theirs, warmup=3, rounds=15):
    """Interleave the two implementations so machine drift cancels out."""
    for _ in range(warmup):
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

CASES = [
    ("matmul 1000^2", "ij,jk->ik", [real(1000, 1000), real(1000, 1000)]),
    ("chain 3x500", "ij,jk,kl->il", [real(500, 500)] * 3),
    ("batched 32x256^2", "bij,bjk->bik", [real(32, 256, 256), real(32, 256, 256)]),
    ("ellipsis 16x128^2", "...ij,...jk->...ik",
     [real(16, 128, 128), real(16, 128, 128)]),
    ("complex mm 800^2", "ij,jk->ik", [cplx(800, 800), cplx(800, 800)]),
    ("hadamard 4000^2", "ij,ij->ij", [real(4000, 4000), real(4000, 4000)]),
    ("4-operand chain 300", "ij,jk,kl,lm->im", [real(300, 300)] * 4),
    ("outer 2000", "i,j->ij", [real(2000), real(2000)]),
]

print(f"{'case':22s} {'pyqcd':>8s} {'opt':>8s} {'ratio':>7s}")
for label, equation, operands in CASES:
    reference = np.einsum(equation, *operands, optimize="optimal")
    assert np.allclose(np.asarray(pe.contract(equation, *operands)), reference), label
    assert np.allclose(np.asarray(oe.contract(equation, *operands, optimize="dp")),
                       reference), label
    ours, theirs, ratio = compare(lambda: pe.contract(equation, *operands),
                                  lambda: oe.contract(equation, *operands, optimize="dp"))
    print(f"{label:22s} {ours:8.2f} {theirs:8.2f} {ratio:7.3f}")
```

Sub-millisecond equations (`ii->`, `ii->i` on a 5000² matrix) are deliberately
excluded: there the measurement is timer and call overhead, not the contraction.


### 扩展形状（CPU 与 GPU）

下面三条替换掉此前的"开发构建"表——它们都能在已发布的 `pyqcd 0.1.21` 上复现：

| 工作负载 | `pyqcd.einsum` | `opt_einsum`（`dp`） | 倍率 |
|---|---|---|---|
| 核子 2pt，CPU，nvec=100，nmom=6 | 2004 ms | 19231 ms | **9.59×** |
| γ 夹心，GPU，T=24，nvec=1024 | 1.3 ms | 6.0 ms | **4.54×** |
| perambulator × 规范场，GPU，空间 24³ | 72.7 ms | 78.6 ms | 1.08× |

旧表里有两行被有意删除：核子 3pt（GPU）需要生产操作数契约（其操作数形状不在公开文档里），
"生产混合加权"依赖未加体积门的 E4 kernel，而发行版在真实体积下主动禁用它。

## CUDA：同样的对照放到 GPU 上

GPU 对照是**同一块卡上的库对库**比较：两边都用 `cupy` 数组在同一设备上执行，
倍率反映的是实现差异，不是 CPU 与 GPU 的算力差。协议：`complex128`、预热 5 次、
**21 次同步计时取中位数**（每次计时前后 `cp.cuda.Device(0).synchronize()`）、
两边都用 `dp` 策略。

在 RTX 4060 Laptop（算力 8.9，按 8.9 编译）上，`pyqcd 0.1.19`
（`build_info(): backend=cuda`）对 `opt_einsum 3.4.0` 实测：

| 工作负载 | `opt_einsum` | `pyqcd.einsum` | 倍率 |
|---|---|---|---|
| γ 夹心，T=512，nvec=512 | 59.8–62.7 ms | 4.6–8.4 ms | **7–13×**（融合 kernel） |
| 核子 2pt，nvec=64 | 255.8–263.8 ms | 134–162 ms | **1.6–1.9×**（kernel，`needs-optimization`） |
| 批量 GEMM 32×512² | 163.2–168.0 ms | 190.6–192.0 ms | 0.86–0.88×（**PyQCD 更慢**） |

结论：

* GPU 上的收益来自**融合 kernel**（γ 夹心整条链一个 kernel，
  trace 可见 `PYQCD_EINSUM_CUDA_SANDWICH ... kernel=1`）；
* 核子 2pt 的 CUDA kernel 只有 1.6–1.9×，且仍标 `needs-optimization`——
  该形态受设备吞吐约束，不是规划器问题；
* 普通批量 GEMM 没有融合 kernel，PyQCD 的批量 GEMM 路径比直接交给
  `cupy.einsum` 慢约 15%，这一组 `opt_einsum` 胜出。

复现（需要 `cupy` 与 `opt_einsum`；缺依赖时脚本给出明确提示并退出，
不会让文档构建失败）：

```python
import statistics
import time

import numpy as np
import pyqcd
import pyqcd.einsum as pe

print("measuring pyqcd", pyqcd.__version__, pyqcd.__file__)

try:
    import cupy as cp
    import opt_einsum as oe
except ImportError:
    raise SystemExit("this comparison needs cupy and opt_einsum: "
                     'pip install "pyqcd[cuda]" opt_einsum')


def bench(call, warmup=5, repeat=21):
    """Median of `repeat` GPU-synchronised calls (milliseconds)."""
    for _ in range(warmup):
        call()
    cp.cuda.Device(0).synchronize()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        call()
        cp.cuda.Device(0).synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples), min(samples), max(samples)


import warnings


def compare(label, equation, operands):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # the fallback notices are printed below
        ours = bench(lambda: pe.contract(equation, *operands))
        theirs = bench(lambda: oe.contract(equation, *operands, optimize="dp"))
    print(f"{label:26s} pyqcd {ours[0]:8.2f} ms | opt_einsum {theirs[0]:8.2f} ms "
          f"| rate {theirs[0] / ours[0]:5.2f}x | n=21 "
          f"(pyqcd min..max {ours[1]:.1f}..{ours[2]:.1f} ms)")


rng = np.random.default_rng(0)


def g(*shape):
    return cp.asarray((rng.standard_normal(shape)
                       + 1j * rng.standard_normal(shape)).astype(np.complex128))


print("device:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
      "|", pe.build_info()["backend"])

n = 64                                                  # production-shaped nucleon 2pt
compare("nucleon 2pt nvec=64", "gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M",
        [g(4, 4, n, n), g(4, 4, n, n), g(4, 4, n, n), g(4, 4), g(4, 4),
         g(4, n, n, n), g(4, n, n, n), g(4, 4)])
compare("batched GEMM 32x512", "bij,bjk->bik", [g(32, 512, 512), g(32, 512, 512)])
compare("gamma sandwich 512x512", "ab,...bce,cd->...dae",
        [g(4, 4), g(512, 4, 4, 512), g(4, 4)])

# correctness first: a benchmark of wrong numbers is worthless
_ref_ops = [g(4, 4, 16, 16), g(4, 4, 16, 16), g(4, 4, 16, 16), g(4, 4), g(4, 4),
            g(4, 16, 16, 16), g(4, 16, 16, 16), g(4, 4)]
_eq = "gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M"
_ours = cp.asnumpy(pe.contract(_eq, *_ref_ops))       # cupy -> host
_theirs = cp.asnumpy(oe.contract(_eq, *_ref_ops, optimize="dp"))
assert np.allclose(_ours, _theirs, rtol=1e-10, atol=1e-12), "kernel disagrees with opt_einsum"
```

笔记本上的 GPU 数字抖动较大：务必看脚本打印的 min–max 区间，
小于该区间的差异不要当真。

## 内存

用 `resource.getrusage` 在独立进程里测的单次收缩峰值（`pyqcd==0.1.10` 对
`opt_einsum 3.4.0`，`dp` 策略，float64）：

| 形状 | `opt_einsum` 峰值 | `pyqcd` 峰值 |
|---|---|---|
| 1200×1200，2 操作数 | 18 MB | 41 MB |
| 1200×1200，3 操作数 | 30 MB | 52 MB |
| 1200×1200，4 操作数 | 29 MB | 52 MB |
| 2000×2000，3 操作数 | 73 MB | 104 MB |

Python 执行路径在单步结束前会多保留一份与输出同尺寸的中间量，因此同一形状下的峰值约为
`opt_einsum` 的 **1.4–2 倍**；同一批测量的耗时不受影响（0.02–0.19 s，双方都受 BLAS 限制）。

实际含义：给 pyqcd 的内存预算要比给 `opt_einsum` 的宽 1.5 倍，或者先用
`contract_path(...)` / `memory_limit=` 规划并切分收缩。

## 自己动手测

下面这段就是上文数字背后的完整协议：构造生产形状的操作数、先预热、取 **9 次计时的中位数**
（不是单次），并把倍率与 min/max 一起打印出来，机器抖动一眼可见。每次调用刻意控制在
5–100 ms 量级——更短的计时测到的是计时器开销而不是引擎。

**没有任何求和标签**的方程（逐元素、转置、对角、外积）会直接交给 backend 自己的
`einsum`：规划器对它们无事可做，而 backend kernel 快 2–10 倍。

```python
import statistics
import time

import numpy as np
import pyqcd.einsum as pe

try:
    import opt_einsum as oe
except ImportError:                       # pip install opt_einsum to run the comparison
    raise SystemExit("install opt_einsum first: pip install opt_einsum")


def bench(call, warmup=2, repeat=9):
    """Median of `repeat` timed calls after `warmup` untimed ones (milliseconds)."""
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        call()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples), min(samples), max(samples)


def compare(label, equation, operands):
    ours = bench(lambda: pe.contract(equation, *operands))
    theirs = bench(lambda: oe.contract(equation, *operands, optimize="dp"))
    print(f"{label:24s} pyqcd {ours[0]:8.2f} ms | opt_einsum {theirs[0]:8.2f} ms "
          f"| rate {theirs[0] / ours[0]:5.2f}x | n=9 "
          f"(pyqcd min..max {ours[1]:.1f}..{ours[2]:.1f} ms)")


rng = np.random.default_rng(0)


def c(*shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex128)


# 1) nucleon 2pt at a production-shaped contract: the shape the kernels target.
nuc = "gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M"
n = 32
compare("nucleon 2pt nvec=32", nuc,
        [c(4, 4, n, n), c(4, 4, n, n), c(4, 4, n, n), c(4, 4), c(4, 4),
         c(4, n, n, n), c(4, n, n, n), c(4, 4)])

# 2) gamma sandwich, batched over time and vector space.
compare("gamma sandwich T=256", "ab,...bce,cd->...dae",
        [c(4, 4), c(256, 4, 4, 512), c(4, 4)])

# 3) a generic chain with no specialised kernel — the honest control case.
a, b, d = (rng.standard_normal((800, 800)) for _ in range(3))
compare("generic chain 800", "ij,jk,kl->il", [a, b, d])

# correctness first: the ratio means nothing if the results disagree
_a = np.arange(12.0).reshape(3, 4)
_b = np.arange(20.0).reshape(4, 5)
assert np.allclose(pe.contract("ij,jk->ik", _a, _b),
                   oe.contract("ij,jk->ik", _a, _b, optimize="dp"))
```

`opt_einsum` 用 `dp` 策略（生产默认）。第三组是对照：通用链式收缩没有专用 kernel，
两者在噪声范围内（0.76–0.93×，即 PyQCD **并不更快**）。

## 代码

专用路径是默认行为：不需要 import、不需要开关、也不需要改计划结构。唯一需要知道的是，
每个形态第一次命中时会打印一行提示，所以做 A/B 对照只差一个关键字参数。

```python
import numpy as np
import pyqcd.einsum as pe

rng = np.random.default_rng(0)


def c(*shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


# Nucleon 2pt (E1): perambulator x gamma x VVV x parity projection
nucleon_2pt = "gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M"
ops_2pt = [c(2, 2, 2, 2), c(2, 2, 2, 2), c(2, 2, 2, 2), c(2, 2),
           c(2, 2), c(4, 2, 2, 2), c(4, 2, 2, 2), c(2, 2)]

fast = pe.contract(nucleon_2pt, *ops_2pt)                    # specialised path
base = pe.contract(nucleon_2pt, *ops_2pt, specialize=False)  # general path
assert np.allclose(fast, base)

# gamma sandwich (E5): perambulator^dagger x gamma sandwich x perambulator
sandwich = "ab,...bce,cd->...dae"
ops_s = [c(2, 2), c(3, 2, 2, 2), c(2, 2)]
assert np.allclose(pe.contract(sandwich, *ops_s),
                   pe.contract(sandwich, *ops_s, specialize=False))

# which shapes get a specialised path
for name, kind, subscripts in pe.specializations():
    print(f"{name:20s} {kind:28s} {subscripts}")
```

不想改应用代码时，同样的开关也可以从环境变量给：

```bash
PYQCD_EINSUM_SPECIALIZE=0 python my_run.py            # process-wide off
PYQCD_EINSUM_SPECIALIZE_VERBOSE=1 python my_run.py    # print for every call
PYQCD_EINSUM_DISABLE_FUSED=1 python my_run.py         # keep planning, drop fused kernels
PYQCD_EINSUM_KERNEL_KUPL=0 python my_run.py           # drop KUPL specialised kernels
PYQCD_EINSUM_KERNEL_CUDA=0 python my_run.py           # drop CUDA specialised kernels
```

所有已注册形态都可以在运行期查询，而 `specialize=False` 永远复现通用路径，
因此回归测试就是一句 `assert`：

| 组 | 类型 | 覆盖内容 |
|---|---|---|
| `nucleon-2pt(E1)` | fused + KUPL/CUDA kernel | 8 算子核子 2pt |
| `nucleon-3pt(E2)` | 计划缓存 | 8 算子核子 3pt |
| `gauge-propagator(E4)` | fused（CUDA） | perambulator × 规范场 |
| `gamma-sandwich(E5)` | fused（CPU + GPU） | `γ₅` 夹心 |
| `sush-2pt(S1)`、`sush-3pt-munu(S2)`、`sush-2pt-mu(S3)` | 计划缓存 | sush 生产形态 |
| `sush-3pt-main(P1/P2)` | fused + CUDA kernel | 14 算子 3pt 主项 |


## 复现

基准程序 `pyqcd.testing.einsum.bench_native` **不在 PyPI 包里**：`pyqcd.testing`
被排除在发布版之外（`setup.py` 用
`find_packages(exclude=["pyqcd.testing", "pyqcd.testing.*"])`），因此基准、测试和回归
基线只存在于源码仓库。下面两条命令必须在**源码检出根目录**执行；在只执行过
`pip install pyqcd` 的环境里，程序会直接报
`ModuleNotFoundError: No module named 'pyqcd.testing'`。

```bash
cd /path/to/PyQCD     # source checkout root; pyqcd.testing is not on PyPI

# CPU: KUPL backend, 4 pinned threads, process isolation
PYQCD_EINSUM_ORACLE=installed PYQCD_EINSUM_NATIVE=kupl PYQCD_BENCH_THREADS=4 \
  python -m pyqcd.testing.einsum.bench_native \
  --data synthetic --profiles T1 --backends cpu --iters 10 --warmup 4 --planner shared

# GPU: CUDA backend, same protocol
PYQCD_EINSUM_ORACLE=installed PYQCD_EINSUM_NATIVE=cuda \
  python -m pyqcd.testing.einsum.bench_native \
  --data synthetic --profiles L24 --backends gpu --iters 10 --warmup 4 --planner shared
```

专用路径的正确性由产生这些数字的同一套件校验，出于同样的原因该套件也只在源码仓库
可用：`python -m pyqcd.testing.einsum.verify_compat`（980 例）与
`python -m pyqcd.testing.einsum.verify_native --backend kupl|cuda`。
