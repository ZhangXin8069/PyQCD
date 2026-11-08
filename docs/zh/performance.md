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

## 实际更快的场景

| 工作负载 | 方程形态 | opt_einsum | pyqcd.einsum | 倍率 |
|---|---|---|---|---|
| 核子 2pt（CPU，N=100） | 8 操作数：perambulator × γ × VVV × 宇称投影 | 120.43 ms | 22.65 ms（特化）/ 25.26 ms（通用） | 对 opt_einsum **5.32×**，对自身通用路径 1.12× |
| 核子 3pt（GPU） | 8 操作数：源端 perambulator × current × γ × VdV × 投影 | 285.40 ms | 30.74 ms（特化）/ 74.13 ms（通用） | 对 opt_einsum **9.29×**，对自身通用路径 2.41× |
| γ 夹心，L24（GPU） | `γ₅ · peramb† · γ₅`，整个空间体积 | 0.373987 s | 0.110959 s | **3.371×** |
| γ 夹心，L24（CPU） | 同形态，4 线程 | 1188.9 ms | 142.7 ms | **8.33×** |
| perambulator × 规范场，L24（GPU） | 4 操作数，含格点 `z,y,x` 体积 | 1.112793 s | 1.095407 s | 1.016×（持平） |
| perambulator × 规范场，`z·y·x = 1`（GPU） | 同形态但无空间体积 | — | — | **3.6×** |

两种倍率要分开看：

* **对自身通用路径**的倍率反映格点 QCD 专用 kernel 带来了多少（2pt 1.12×、3pt 2.41×、
  CPU 夹心 8.33×）；
* **对 `opt_einsum`** 的倍率还叠加了通用执行层在这些形态上本来就有的优势。

## 生产混合口径的汇总

按真实生产中五类收缩的耗时占比加权（2pt 51%、3pt 15.5% + 3.1%、
perambulator × 规范场 26.4%、夹心 4%）：

| 实现 | 总耗时 | 相对 |
|---|---|---|
| `opt_einsum` | 4757 s | 1.00× |
| `pyqcd.einsum` | 1688 s | **2.82×** |

这是由上面逐形态测量加权得到的**模型估计**，不是一次端到端实跑。

## 优势来源

1. **方程形态特化**：识别一小批格点 QCD 形态（核子 2pt/3pt、γ 夹心、
   perambulator × 规范场），直接走专用 kernel，而不是通用 BLAS 链。
2. **融合**：中间收缩结果留在寄存器/共享内存，不再物化成临时张量。
3. **缓冲区复用**：操作数打包与输出缓冲跨调用池化，中间步骤不重复付固定开销。
4. **原生规划**：规划核心编译进原生库，路径与代价和 Python 规划器完全一致
   （由 980 例差分套件验证），而派发开销更低。

## 需要如实说明的限制

| 情形 | 实测表现 |
|---|---|
| GPU 核子 2pt/3pt 对 `opt_einsum` | ≈1× —— 四种独立实现都落在 373–444 ms，该形态受硬件吞吐约束，不是规划器问题 |
| perambulator × 规范场 全 L24 体积（GPU） | 持平（1.016×）；融合 kernel 目前只在 `z·y·x = 1` 时生效 |
| 小张量（Nvec=20、`spatial=1`） | 0.51×–1.32×，单次调用固定开销占主导 |
| 本机 GPU 的复数稠密 `zgemm` 吞吐 | 约 71 GFLOP/s，而 4 线程 CPU 约 228 GFLOP/s；只有工作量足够大时 GPU 才占优 |
| 无专用 kernel 的形态 | 只走通用路径：计划仍与 `opt_einsum` 兼容，但没有额外加速 |
| MPI | 仅 KUPL 后端分片，且只分片单个输出轴；这些对比中从未使用 |

**硬件口径提醒**：以上 GPU 数字来自笔记本显卡，绝对耗时不能代表数据中心卡；
但双方在同一块设备上测量，因此**倍率**仍然有意义。

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
