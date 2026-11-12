# 使用方式

下面的示例在纯 CPU、只装 `numpy` 的环境里即可运行；原生后端只影响速度，不影响结果。

## 收缩

```python
import numpy as np
import pyqcd.einsum as pe

rng = np.random.default_rng(0)

# 两个复数矩阵：contract 的输入与 numpy 完全一样，只是多了一个更聪明的规划器
a = (rng.standard_normal((3, 4)) + 1j * rng.standard_normal((3, 4))).astype(np.complex128)
b = (rng.standard_normal((4, 5)) + 1j * rng.standard_normal((4, 5))).astype(np.complex128)

# 1) 直接算：契约解析 -> 规划 -> 执行，一步到位
result = pe.contract("ij,jk->ik", a, b)

# 2) 只想看它打算怎么算：contract_path 返回收缩顺序与代价，不做实际运算
path, info = pe.contract_path("ij,jk->ik", a, b)
print(int(info.opt_cost), path)

# 3) 同一方程要反复调用时，把规划结果编译成一个表达式对象复用
expr = pe.contract_expression("ij,jk->ik", (3, 4), (4, 5))
for _ in range(3):
    expr(a, b)           # 复用已编译的计划，不再重新规划

# 三种用法必须给出同一个矩阵乘结果
assert np.allclose(result, a @ b)
assert int(info.opt_cost) > 0
assert np.allclose(expr(a, b), a @ b)
```

共享中间结果（相同子表达式只算一次）：

```python
# 上下文管理器内的重复子收缩会被记住：第二次调用直接取缓存
with pe.shared_intermediates() as cache:
    pe.contract("ab,bc->ac", a, b, backend="numpy")
    pe.contract("ab,bc->ac", a, b, backend="numpy")
print(pe.count_cached_ops(cache))

# 计数值是一个 Counter（按算子签名分组），不是单个整数
ops = pe.count_cached_ops(cache)
assert sum(ops.values()) >= 1, ops
```

### 后端与精度

```python
from pyqcd.tools import set_backend, get_backend, get_backend_name, get_precision

# 全局后端：numpy（默认，CPU）/ cupy（NVIDIA GPU）/ torch（CPU 或 CUDA）
set_backend("numpy")          # 也可以写 "cupy"、"torch"
print(get_backend_name(), get_precision())

assert get_backend_name() == "numpy"
assert get_precision() in {"complex64", "complex128"}
```

### 关闭方程特化

```python
# specialize=False 只影响这一次调用：不走方程特化的专用路径，改用通用规划
pe.contract("ij,jk->ik", a, b, specialize=False)

# 结果必须与走专用路径时完全一致——特化只改速度，不改数值
assert np.allclose(pe.contract("ij,jk->ik", a, b, specialize=False), a @ b)
```

环境变量开关：`PYQCD_EINSUM_SPECIALIZE=0`、`PYQCD_EINSUM_DISABLE_FUSED=1`、
`PYQCD_EINSUM_KERNEL_KUPL=0`、`PYQCD_EINSUM_KERNEL_CUDA=0`。

## 格点常数与 Dirac 代数

```python
import pyqcd.lattice as L

# 基本常数：颜色数、时空维数、旋量维数、格距换算因子
print(L.Nc, L.Nd, L.Ns, L.fm2GeV)

# gamma 矩阵用 DeGrand–Rossi 基，gamma(5) 是 4x4
print(L.gamma(5).shape)

assert (L.Nc, L.Nd, L.Ns) == (3, 4, 4)
assert L.gamma(5).shape == (4, 4)
assert L.fm2GeV > 0
```

## 规范场观测量与涂抹

```python
import numpy as np
import pyqcd.gauge as G
import pyqcd.smear as S

# 造一个 2^4 的平凡规范场：所有链接都是单位矩阵
# 布局是 U[t, z, y, x, mu, colour, colour]
U = np.zeros((2, 2, 2, 2, 4, 3, 3), dtype=np.complex128)
for mu in range(4):
    U[..., mu, :, :] = np.eye(3)

# Wilson 圈、Polyakov 圈、Stout 涂抹、HYP 涂抹
print(G.wilson_loop(U, 1, 1, 0, 1))
print(G.polyakov_loop(U).shape)
print(S.stout_smear(U, nstep=1).shape)     # (2, 2, 2, 2, 4, 3, 3)
print(S.hyp_smear(U).shape)

# 单位链接下 1x1 Wilson 圈必须精确等于 1；涂抹不改变张量形状
assert abs(complex(G.wilson_loop(U, 1, 1, 0, 1)) - 1.0) < 1e-12
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

# 每个格点每个方向做一个随机 SU(3) 链接（QR 分解后取正交因子）
rng = np.random.default_rng(0)
U = np.zeros((2, 2, 2, 2, 4, 3, 3), dtype=complex)
for mu in range(4):
    q, _ = np.linalg.qr(rng.standard_normal((2, 2, 2, 3, 3))
                        + 1j * rng.standard_normal((2, 2, 2, 3, 3)))
    U[..., mu, :, :] = q

# 流时换算：a=0.1 fm 时 tau=1.0 对应多少 GeV^-2
print(R.flow_time_gev_m2(1.0, a_fm=0.1))

# Clover 场强张量 F_{mu nu}，每对 (mu, nu) 一个张量；再取对偶 F~_{01}
F = {(mu, nu): G.clover_field_strength(U, mu, nu)
     for mu in range(4) for nu in range(4) if mu != nu}
print(O.compute_dual_field_strength(F, 0, 1).shape)    # (2, 2, 2, 2, 3, 3)

assert R.flow_time_gev_m2(1.0, a_fm=0.1) > 0
assert set(F) and all(np.shape(v) == (2, 2, 2, 2, 3, 3) for v in F.values())
assert O.compute_dual_field_strength(F, 0, 1).shape == (2, 2, 2, 2, 3, 3)
```

基于梯度流的重正化 TMD：

```python
# 梯度流重整化后的胶子 TMD：给定流时 tau、两套横距 (z, b)，返回矩阵元
result = R.gradient_flow_renormalized_tmd(
    U, tau=1.0, z_list=[0, 2, 4], b_list=[0], z_dir=2, b_dir=0)

# 形状是 (len(z_list), len(b_list))，且必须全部有限
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

# 20 个组态 x 5 个时间片的假数据；第 0 维是组态轴
data = np.random.default_rng(0).standard_normal((20, 5))
samples = A.Bootstrap(data, Nconf_axes=0)                  # 返回重采样后的字典
print(sorted(samples))

# 字典里同时给出均值、误差与逐样本，切片的维度与输入一致
assert {"data_mean", "data_err", "data_sample"} <= set(samples)
assert np.shape(samples["data_mean"])[0] == 5
```

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
