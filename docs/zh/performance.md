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

## 在已发布包上可复现

`pyqcd==0.1.13`（`build_info(): backend=kupl, array_backend=numpy`）对
`opt_einsum==3.4.0`（`dp`），`complex128`，预热 3 次后取 15 次中位数：

| 工作负载 | `opt_einsum` | `pyqcd.einsum` | 倍率 |
|---|---|---|---|
| 核子 2pt，nvec=16（生产契约） | 12.95 ms | 2.15 ms | **6.02×** |
| 核子 2pt，nvec=24 | 41.45 ms | 7.11 ms | **5.83×** |
| γ 夹心，T=8，nvec=64 | 0.169 ms | 0.058 ms | **2.91×** |

本版本真正加速了哪些形态：

| 形态 | 0.1.x 中的执行路径 |
|---|---|
| 核子 2pt（`gahb,mcnd,…->M`） | KUPL/CUDA 专用 kernel + 计划缓存 |
| 核子 3pt 主项（`…->MNGLxy`） | KUPL/CUDA 专用 kernel + 计划缓存 |
| 核子 3pt（`…->GM`）、sush 2pt/3pt-mu | 仅计划缓存 |
| γ 夹心（`ab,...bce,cd->...dae`）、perambulator × 规范场 | 计划缓存 + 通用执行层 |

首次命中会打印提示，据实说明走的是哪条路：`[kernel: kupl_nucleon_2pt]` 表示
真的跑了 kernel，`[plan-cache]` 表示只复用了计划。

## 历史（开发构建）测量

下表来自**带额外融合 kernel 的开发构建**（γ 夹心、perambulator × 规范场）。
这些 kernel **不在 0.1.x 发行版里**，因此数字**无法从 PyPI 复现**，仅作为
"这些形态本身能有多贵"的记录：

| 工作负载 | opt_einsum | 开发构建 | 倍率 |
|---|---|---|---|
| 核子 2pt（CPU，N=100） | 120.43 ms | 22.65 ms | 5.32× |
| 核子 3pt（GPU） | 285.40 ms | 30.74 ms | 9.29× |
| γ 夹心 L24（GPU） | 0.373987 s | 0.110959 s | 3.371× |
| γ 夹心 L24（CPU） | 1188.9 ms | 142.7 ms | 8.33× |
| perambulator × 规范场 L24（GPU） | 1.112793 s | 1.095407 s | 1.016×（持平） |

基于这些数字的"生产混合加权 ≈2.82×（4757 s → 1688 s）"同样继承该前提：
它假设那些额外 kernel 存在。


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
