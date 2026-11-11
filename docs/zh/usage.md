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

## 并行规划

```python
import pyqcd.parallel as P

# 先探测资源，再按 N*a = n*b 规划进程数
resources = P.detect_resources()
plan = P.plan_parallel(m=12, a_mem_mb=500, resources=resources)
print(P.format_plan(plan))

assert isinstance(P.format_plan(plan), str)
assert {"N", "Y", "X", "m"} <= set(plan)      # plan_parallel 返回字典
assert plan["N"] >= 1 and plan["m"] == 12
```

规划**从** `N*a = n*b` 出发（a 为单元任务显存、b 为单卡可用显存、n 为 GPU 数），
给出进程数 `N`、批次 `X = ceil(m/N)` 与每卡进程数 `Y = N/n`。该等式是**未封顶**的目标：
随后 `Y` 会被压到"每个进程至少一个任务"（`Y <= m/n`）、不超过 CPU 线程数、且受内存限制，
因此任务数少时会出现 `N*a < n*b`——压缩过程写在 `notes` 里（如 `Y capped from 13 to 3`），
而 `gpu_ok` 仍告诉你单个 worker 是否放得进单卡预算。

## 批量独立收缩（`init` + `run`）

`pyqcd.parallel.einsum` 把**相互独立**的一批收缩按工作量分箱后交给多个 MPI 进程并行
执行，每个子任务仍是一次普通的 `contract(...)`，参数与语义都不变。接口只有两步：
`init` 规划，`run` 执行。

```python
import numpy as np
import pyqcd.einsum as pe
from pyqcd.parallel import init, plan_tasks, run

# 一批相互独立的收缩：同一个方程、同一形状、不同数据
rng = np.random.default_rng(0)
left = [rng.standard_normal((64, 64)) for _ in range(4)]
right = [rng.standard_normal((64, 64)) for _ in range(4)]
out = [np.zeros((64, 64)) for _ in range(4)]          # 调用方自己的输出缓冲区

# 列式字典；task_id 可省略，其余三个必需
config = {
    "task_id": [f"t{i}" for i in range(4)],
    "equation": ["ij,jk->ik"] * 4,
    "operands": [[left[i], right[i]] for i in range(4)],   # 只读，不会被复制
    "results": out,                                        # 可写，结果原地写入
}

plan = plan_tasks(config["equation"], config["operands"], max_procs=4)
state = init(config, max_proc=1, reminder=False)      # max_proc=1 让本块走串行
run(state)                                           # 启动器：返回 None

# 结果与输入一一对应：同键、同序；装箱覆盖全部任务且没有短板
assert plan["workers"] == min(4, plan["limits"]["limit"])
assert sorted(i for bin_ in plan["bins"] for i in bin_) == [0, 1, 2, 3]
assert plan["skew"] >= 1.0 and len(plan["bins"]) == plan["workers"]

# 每个缓冲区现在正好等于 contract 的结果，且是同一批对象
for i in range(4):
    assert np.array_equal(out[i], pe.contract("ij,jk->ik", left[i], right[i]))
    assert out[i] is config["results"][i]
```

```python
# docs-skip: 会拉起 mpirun worker 进程
from pyqcd.parallel import init, run

state = init(config, max_proc=4, reminder=False)      # 自建 MPI 进程池
run(state)
```

* **`init` 规划，`run` 执行。** `init` 校验字典并算分箱；`run` 是启动器——返回
  `None`、结果**原地**写入，因此规划与执行可以分别计时，也不会重新分配输出数组。
* **0.3.0 起不再提供一体化包装**：`einsums` 与 `plan_einsums` 已移除，执行改用
  `init` + `run`，只规划改用 `plan_tasks`。
* **输入字典**为 `{"task_id": [...], "equation": [...], "operands": [...],
  "results": [...]}`；`task_id` 可省略（用下标）。长度必须一致、每个结果缓冲区必须
  等于该收缩的输出形状、且必须可写，否则 `init` 会在开工前失败。`operands` 从不被
  复制或修改。
* **工作量分配**：用 `contract_path` 的优化 FLOP 数估计工时（对无求和的逐元素方程以
  输入/输出元素数兜底），再用 LPT（longest-processing-time-first）装箱；最慢进程
  不超过最优解的 `4/3 - 1/(3K)`，所以各箱任务数可以不同但不会出现短板。
* **进程数**默认取 `min(子任务数, CPU 线程数, 可用内存允许值)`，可用 `max_proc=`
  收紧。不在 MPI 下时它自己拉起 `mpirun -np K` 进程池，经临时目录交换任务与结果。
* **collective 模式默认关闭。** 已在 `mpirun -np N` 内时默认 `collective=False`：
  只有 rank 0 执行整批任务，其余 rank 立即返回，不做任何复制。需要摊到全部 `N` 个
  rank 时显式传 `collective=True`——不落盘交换，但每个 rank 都会持有一份整批输入
  （见下条内存说明）。
* **内存：collective 会把输入复制到每个 rank。** `collective=True` 时每个 rank 都必须
  先构造**整批**任务，输入占用变成「单份 × rank 数」；rank 0 另外持有**全部结果**，
  `broadcast=True` 时每个 rank 还会各持一份完整结果。用 `resource.ru_maxrss` 实测
  （float32，64 个 512² 任务，整批输入 134 MB）：串行峰值 234 MB，collective 4 rank
  聚合约 1.15 GB。在 GPU 上这意味着**每张可见卡各放一整批**，所以 rank 数应与可见
  设备数对齐。这正是它默认关闭、且每次使用都会打印警告的原因；批次大到放不下时改用
  自建进程池——任务经磁盘分发，worker 只读自己那一箱，代价是每次调用多 0.15–0.3 s
  固定开销。
* **精度跟随操作数**：`complex64`/`float32` 直接把内存与算术量减半；
  `init`/`run` 自己不做任何类型转换。

### 失败处理与调试

```python
# 结果缓冲区形状不符的任务会在 init 阶段、开工之前就失败。
try:
    init({"equation": ["ij,jk->ik"],
          "operands": [[np.eye(3), np.eye(3)]],
          "results": [np.zeros((4, 4))]}, max_proc=1, reminder=False)
except ValueError as exc:
    assert "does not match" in str(exc)

# verbose=True 会按步骤打印带 task_id 的调试行。
state = init({"task_id": ["probe"],
              "equation": ["ij,jk->ik"],
              "operands": [[left[0], right[0]]],
              "results": [np.zeros((64, 64))]},
             max_proc=1, verbose=True, reminder=False)
run(state)
assert state["mode"] == "serial"
```

`run` 是启动器，没有返回值；它观测到的执行统计会写回你传入的 state 字典
（`state["mode"]`、`state["timings"]`），这就是你用来卡时间的东西。

在本机（4 物理核 / 8 线程、16 GB；真实 sush 蒸馏方程 S1 = NN 2pt，8 算子；
`complex64`；`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1` 作为单线程公平基准；
并行结果均与串行 `contract` 对照通过）实测：

| 模式 | 工作负载 | 串行 | `init` + `run` | 加速比 | 峰值 RSS（本 rank / 全部 rank） |
|---|---|---|---|---|---|
| `mpirun -np 2`（collective） | S1，Nvec=40，8 任务 | 168 ms | 88 ms | **1.86×** | rank 0 415 MB / 合计约 702 MB |
| `mpirun -np 4`（collective） | S1，Nvec=40，8 任务 | 183 ms | 59 ms | **3.09×** | rank 0 383 MB / 合计约 1.15 GB |
| `mpirun -np 4`（collective，默认线程） | S1，Nvec=40，8 任务 | 138 ms | 53 ms | 2.28× | — |
| 自建进程池（4 进程） | S1，Nvec=60，8 任务 | 736 ms | 622 ms | 1.18× | 磁盘上一份任务 |
| 自建进程池（4 进程） | S1，Nvec=100，8 任务 | 5326 ms | 2160 ms | **2.47×** | 磁盘上一份任务 |
| 自建进程池（4 进程） | S1，Nvec=40，8 任务 | 217 ms | 391 ms | 0.56× | 磁盘上一份任务 |

两条可复现结论（入口分别是 `python -m pyqcd.testing.parallel_einsum bench`
与 `mpirun -np 4 python -m pyqcd.testing.parallel_einsum bench-collective`）：

* **已在 `mpirun` 内部（collective）是高效模式**：4 rank 上 3.09×，且完全不落盘。
* **自建进程池有固定成本**：MPICH `mpirun` 启动约 150 ms，每个操作数还要落盘再回读
  （本机约 1.5 GB/s）。串行耗时约 1 秒以上才开始划算（0.74 s 时 1.18×，5.3 s 时 2.47×），
  小批次反而更慢。**不要假设加速**——先在同机同 dtype 下用 `bench` 复核。

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
