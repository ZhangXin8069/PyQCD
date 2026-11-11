# 使用方式

下面的示例在纯 CPU、只装 `numpy` 的环境里即可运行；原生后端只影响速度，不影响结果。

## 收缩

```python
import numpy as np
import pyqcd.einsum as pe

rng = np.random.default_rng(0)
a = (rng.standard_normal((3, 4)) + 1j * rng.standard_normal((3, 4))).astype(np.complex128)
b = (rng.standard_normal((4, 5)) + 1j * rng.standard_normal((4, 5))).astype(np.complex128)

result = pe.contract("ij,jk->ik", a, b)
path, info = pe.contract_path("ij,jk->ik", a, b)
print(int(info.opt_cost), path)

expr = pe.contract_expression("ij,jk->ik", (3, 4), (4, 5))
for _ in range(3):
    expr(a, b)           # reuses the compiled plan

assert np.allclose(result, a @ b)
assert int(info.opt_cost) > 0
assert np.allclose(expr(a, b), a @ b)
```

共享中间结果（相同子表达式只算一次）：

```python
with pe.shared_intermediates() as cache:
    pe.contract("ab,bc->ac", a, b, backend="numpy")
    pe.contract("ab,bc->ac", a, b, backend="numpy")
print(pe.count_cached_ops(cache))

ops = pe.count_cached_ops(cache)          # Counter, not a plain int
assert sum(ops.values()) >= 1, ops
```

### 后端与精度

```python
from pyqcd.tools import set_backend, get_backend, get_backend_name, get_precision

set_backend("numpy")          # also: "cupy", "torch"
print(get_backend_name(), get_precision())

assert get_backend_name() == "numpy"
assert get_precision() in {"complex64", "complex128"}
```

### 关闭方程特化

```python
pe.contract("ij,jk->ik", a, b, specialize=False)   # per call

assert np.allclose(pe.contract("ij,jk->ik", a, b, specialize=False), a @ b)
```

环境变量开关：`PYQCD_EINSUM_SPECIALIZE=0`、`PYQCD_EINSUM_DISABLE_FUSED=1`、
`PYQCD_EINSUM_KERNEL_KUPL=0`、`PYQCD_EINSUM_KERNEL_CUDA=0`。

## 格点常数与 Dirac 代数

```python
import pyqcd.lattice as L

print(L.Nc, L.Nd, L.Ns, L.fm2GeV)
print(L.gamma(5).shape)          # (4, 4) DeGrand–Rossi basis

assert (L.Nc, L.Nd, L.Ns) == (3, 4, 4)
assert L.gamma(5).shape == (4, 4)
assert L.fm2GeV > 0
```

## 规范场观测量与涂抹

```python
import numpy as np
import pyqcd.gauge as G
import pyqcd.smear as S

# a trivial 2^4 lattice of unit links: U[t, z, y, x, mu, colour, colour]
U = np.zeros((2, 2, 2, 2, 4, 3, 3), dtype=np.complex128)
for mu in range(4):
    U[..., mu, :, :] = np.eye(3)

print(G.wilson_loop(U, 1, 1, 0, 1))
print(G.polyakov_loop(U).shape)
print(S.stout_smear(U, nstep=1).shape)     # (2, 2, 2, 2, 4, 3, 3)
print(S.hyp_smear(U).shape)

assert abs(complex(G.wilson_loop(U, 1, 1, 0, 1)) - 1.0) < 1e-12     # unit links
assert G.polyakov_loop(U).shape == (2, 2, 2) or G.polyakov_loop(U).size > 0
assert S.stout_smear(U, nstep=1).shape == U.shape
assert S.hyp_smear(U).shape == U.shape
```

## 梯度流与胶子算符

```python
import numpy as np
import pyqcd.gauge as G
import pyqcd.operator as O
import pyqcd.renorm as R

rng = np.random.default_rng(0)
U = np.zeros((2, 2, 2, 2, 4, 3, 3), dtype=complex)      # random SU(3) links
for mu in range(4):
    q, _ = np.linalg.qr(rng.standard_normal((2, 2, 2, 3, 3))
                        + 1j * rng.standard_normal((2, 2, 2, 3, 3)))
    U[..., mu, :, :] = q

print(R.flow_time_gev_m2(1.0, a_fm=0.1))               # flow time in GeV^-2
F = {(mu, nu): G.clover_field_strength(U, mu, nu)
     for mu in range(4) for nu in range(4) if mu != nu}
print(O.compute_dual_field_strength(F, 0, 1).shape)    # (2, 2, 2, 2, 3, 3)

assert R.flow_time_gev_m2(1.0, a_fm=0.1) > 0
assert set(F) and all(np.shape(v) == (2, 2, 2, 2, 3, 3) for v in F.values())
assert O.compute_dual_field_strength(F, 0, 1).shape == (2, 2, 2, 2, 3, 3)
```

基于梯度流的重正化 TMD：

```python
result = R.gradient_flow_renormalized_tmd(
    U, tau=1.0, z_list=[0, 2, 4], b_list=[0], z_dir=2, b_dir=0)

assert np.shape(result) == (3, 1), np.shape(result)
assert np.all(np.isfinite(np.asarray(result))), "non-finite TMD result"
```

## 蒸馏收缩

```python
import pyqcd.contraction as C

# pseudo-code: the registries must be filled with real perambulators,
# vertices and gamma matrices before this block runs.
result = C.dynamic_contraction(
    operator_groups, peram_registry=..., v_registry=..., gamma_registry=...,
    Cpt="2pt", verbose=True)
```

`parity_and_boundary` 施加宇称投影与反周期边界的符号；`classify_structure` 与
`identify_equivalent_diagrams` 用于在收缩前归并等价算符结构。

## 分析工具

```python
import numpy as np
import pyqcd.analysis as A

data = np.random.default_rng(0).standard_normal((20, 5))   # 20 configs x 5 slices
samples = A.Bootstrap(data, Nconf_axes=0)                  # dict of resampled arrays
print(sorted(samples))

assert {"data_mean", "data_err", "data_sample"} <= set(samples)
assert np.shape(samples["data_mean"])[0] == 5
```

## 并行规划

```python
import pyqcd.parallel as P

resources = P.detect_resources()
plan = P.plan_parallel(m=12, a_mem_mb=500, resources=resources)
print(P.format_plan(plan))

assert isinstance(P.format_plan(plan), str)
assert {"N", "Y", "X", "m"} <= set(plan)      # plan_parallel returns a dict
assert plan["N"] >= 1 and plan["m"] == 12
```

规划**从** `N*a = n*b` 出发（a 为单元任务显存、b 为单卡可用显存、n 为 GPU 数），
给出进程数 `N`、批次 `X = ceil(m/N)` 与每卡进程数 `Y = N/n`。该等式是**未封顶**的目标：
随后 `Y` 会被压到"每个进程至少一个任务"（`Y <= m/n`）、不超过 CPU 线程数、且受内存限制，
因此任务数少时会出现 `N*a < n*b`——压缩过程写在 `notes` 里（如 `Y capped from 13 to 3`），
而 `gpu_ok` 仍告诉你单个 worker 是否放得进单卡预算。

## 使用 PyTorch 张量

PyTorch 支持是可选依赖（`pip install "pyqcd[torch]"`），并且是**按操作数类型**分派的：
`pyqcd.einsum.contract` 看的是你传进去的数组，不是某个全局开关。

| 操作数 | 实际执行 | 返回类型 | 是否用原生 KUPL/CUDA 库 |
|---|---|---|---|
| 全是 `torch.Tensor` | 张量所在设备上的 `torch.einsum` | `torch.Tensor`（同设备） | **不用** |
| 全是 `numpy.ndarray` | KUPL（CPU）或 BLAS 规划器 | `numpy.ndarray` | 是 |
| 全是 `cupy.ndarray` | CUDA kernel / cuBLAS | `cupy.ndarray` | 是 |
| numpy 与 torch 混合 | 回落到 numpy 路径 | `numpy.ndarray` | 是 |

需要知道的几点：

* torch 张量不会进入 KUPL/CUDA 库，所以专用 kernel 与按调用切档都不适用——
  你拿到的是 `torch.einsum` 的水平。首次尝试原生路径会打印一次
  `RuntimeWarning: native plan unavailable … Cannot interpret 'torch.complex128'
  as a data type`，这是预期行为。
* 想用原生加速，请在边界处转换：KUPL 用
  `t.detach().cpu().contiguous().numpy()`，CUDA 用 `cupy.asarray(t)`。
* `set_backend("torch")` 改变的是 `pyqcd.tools.get_backend()` 给物理模块的返回值，
  **不会**让 numpy 输入的 `contract()` 改走 torch。
* autograd 可用（复数输出要取实部标量，例如 `y.real.sum()`）。

```python
import numpy as np
import torch
import pyqcd.einsum as pe
import pyqcd.tools as T

# 1) torch tensors are dispatched to torch.einsum — no extra flag needed
a = torch.randn(4, 5, dtype=torch.complex128)
b = torch.randn(5, 6, dtype=torch.complex128)
out = pe.contract("ij,jk->ik", a, b)          # Tensor, same device as the inputs
print(type(out).__name__, out.dtype, bool(torch.allclose(out, a @ b)))

# 2) a CUDA tensor keeps running on the GPU
if torch.cuda.is_available():
    ga, gb = a.to("cuda"), b.to("cuda")
    print("device:", pe.contract("ij,jk->ik", ga, gb).device)

# 3) autograd flows through the contraction (use a real dtype for the scalar)
x = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
y = pe.contract("ij,jk->ik", x, torch.randn(5, 6, dtype=torch.float64))
y.sum().backward()
print("grad ok:", tuple(x.grad.shape))

# 4) make torch the default backend for pyqcd.tools / the physics modules
T.set_backend("torch", device="cuda" if torch.cuda.is_available() else "cpu")
print(T.get_backend_name(), T.get_precision(), T.get_torch_device())
T.set_precision("complex64")                  # switch the global complex precision
T.set_precision("complex128")
T.set_backend("numpy")                        # back to the default

assert torch.allclose(out, a @ b)

# 5) and the correctness contract: same numbers as numpy
assert np.allclose(np.asarray(out), np.asarray(a) @ np.asarray(b))
```

`set_precision("complex64")` 同样会把 torch 路径的访存量减半；
`pyqcd.lattice.gamma` 等辅助函数跟随同一个全局精度。

## 参见

相对 `opt_einsum` 的加速、测量口径，以及**没有加速**的情形见 :doc:`performance`。

### 与 `opt_einsum` 3.4.0 的差异

只有两处**增量**差异：

* `optimize=` 还接受**可调用对象** `(input_subsets, output_subset, size_dict) -> path`；
  `opt_einsum` 3.4.0 对同样的参数会抛 `TypeError: 'function' object is not iterable`。
* `pyqcd.einsum.PathInfo` 在包层级直接导出（上游放在 `opt_einsum.paths` 下）。

其余行为一致——包括 `contract_expression` 的 `constants=` 处理（两边抛出同样的
`ValueError`）。
