# 纯规范观测量 reference

## 当前 PyQCD 公开 API

`pyqcd.gauge` 接受统一布局
`(Nt,Nz,Ny,Nx,4,Nc,Nc)`，链接方向为 `0=x,1=y,2=z,3=t`，方向到坐标轴
映射为 `axis=3-direction`。输入 dtype 只允许
`float32/float64/complex64/complex128`；bool、整数、`float16/bfloat16/complex32`
及扩展精度在公开入口早期 `ValueError`，不能等 Clover 的底层复数运算偶然失败。

NumPy/CuPy 输入按当前 backend 转换。当前 backend 已是 Torch 且输入本身是 Tensor 时，
该 Tensor 的 device 与 dtype 是更具体的所有权契约：即使全局 `set_backend("torch",
device="cpu")`，CUDA 输入仍须在原 CUDA device 完成所有临时量、矩阵乘和归约，不能先
搬回 CPU。`set_precision` 也不覆盖这份现有 Tensor 的精度。输出 dtype 由观测量决定：
Wilson/拓扑量为输入位宽对应的 `float32/64`；Polyakov 对实输入保持实 dtype、对复输入
保持 `complex64/128` 及中心相位；Clover 对实或复输入均返回对应位宽的
`complex64/128`。

所有归约入口（Wilson `average=True`、Polyakov `average=True`/`polyakov_loop_average`、
拓扑总荷和密度体积平均）都直接返回当前 backend 的 0-D 归约对象，不转为 Python
`float`/`complex`：Torch 是继承输入 device/dtype 的 `shape=[]` 0-D Tensor；CuPy 是
`shape=()` 的 0-D `cupy.ndarray`；NumPy 是保留对应 dtype、`shape=()` 的 NumPy 标量。
因此文档中的“标量平均/总量”不是 Python scalar；归约不会因 `.item()`、`float()` 或
`complex()` 被隐式搬到 host 或改变 dtype。

| 入口 | 返回语义 |
|---|---|
| `wilson_rectangle(gauge,R,T,mu,nu,*,average=True)` | 默认返回全起点 `Re Tr(U_C)/Nc` 平均；`False` 返回 `(Nt,Nz,Ny,Nx)` 逐点圈 |
| `polyakov_loop(gauge,time_dir=3,*,average=False,direction=None)` | `False` 返回移去 `axis=3-time_dir` 后、其余轴原顺序的逐点复数 `Tr(prod U)/Nc`；`True` 返回其复标量平均 |
| `clover_field_strength(gauge,mu,nu,*,traceless=True)` | 返回 `(Nt,Nz,Ny,Nx,Nc,Nc)`；默认为 Hermitian 且无迹的 `su(Nc)` Clover 场，`False` 只复现历史未去迹场 |
| `clover_topological_charge_density(gauge,*,traceless=True)` | 返回 `(Nt,Nz,Ny,Nx)` 的 bare `q(x)` |
| `clover_topological_charge(gauge,*,traceless=True)` | 返回 `sum_x q(x)`，不是体积平均 |
| `clover_topological_charge_density_average(gauge,*,traceless=True)` | 返回 `mean_x q(x)`，不是 ensemble 平均 |

`polyakov_loop` 中的 `direction` 是 `time_dir` 的兼容别名；非 `None` 时它覆盖
`time_dir`。新代码只传 `time_dir`，不同时设置两者。

`R=0` 或 `T=0` 按单位圈返回 1。`average`、`traceless`只接受布尔值，避免
整数真值静默改变物理定义。当前可执行边界契约为：

```bash
python -m pyqcd.testing._gauge_observables_contract
```

该契约覆盖周期路径、真逆链接、Polyakov 中心相位、局域规范不变性、Clover
反对称/默认去迹、拓扑系数、精确 dtype 门、Torch 输入设备继承和后端一致性；它不替代流化真实系综或已知
instanton 的连续极限验证。

`clover_field_strength` 返回的是格点单位中的无量纲离散场：令
`C_mu_nu(x)` 为四个同向 plaquette 之和，代码实现
`-i(C_mu_nu-C_mu_nu†)/8`，未显式包含 `1/(a^2 g_0)`。它不会自动流化输入；需要流化拓扑量时，
先用已记录 `tau/eps` 的场调用该 API。`traceless=False` 仅复现历史未去迹
Clover **基元**，不等于复现 Wilson 线、对偶场缩并和空间求和组成的完整旧 OPE；
完整 OPE 任务转交 `pyqcd-tmd-algorithm` 并使用其已验证组合入口。

## Wilson 圈与静态势

矩形 Wilson 圈应明确四段路径：沿 $\mu$ 前进 $R$，沿 $\nu$ 前进 $T$，沿 $\mu$
退回 $R$，沿 $\nu$ 退回 $T$。以 $N_c$ 维基本表示为例，常用标量是

\[
W_{\mu\nu}(R,T)=\frac1{N_c}\,\mathrm{Re\,Tr}\,\mathcal P
\left[U_\mu^R U_\nu^T U_{-\mu}^R U_{-\nu}^T\right].
\]

静态势的长时间极限为

\[
V(R)=-\lim_{T\to\infty}\frac1T\log\langle W(R,T)\rangle,
\qquad V(R)=V_0+\frac{\alpha}{R}+\sigma R.
\]

实际拟合必须保留有限 $T$、自相关、协方差和 $R/T$ 窗口；`σ>0` 是禁闭模型中的
物理预期，不是可以替数据强加的约束。

## PyQUDA 路径调用

```python
from pyquda_utils.core import X, Y, Z, T

path_xt = [X] * R + [T] * t_len + [-X] * R + [-T] * t_len
path_yt = [Y] * R + [T] * t_len + [-Y] * R + [-T] * t_len
path_zt = [Z] * R + [T] * t_len + [-Z] * R + [-T] * t_len
result = gauge.loop([[path_xt], [path_yt], [path_zt], [path_xt]], [1, 1, 1, 0])
```

`gauge.loop` 的外层组数量和权重是版本相关的硬约束；使用前查当前签名。GPU 返回
对象不能直接 `trace`：转 host 后 reshape 为 `(...,Nc,Nc)`，再取 `Re Tr`。

```python
import numpy as np
from pyquda_utils import core

i = 0       # XT；1/2 分别对应 YT/ZT
Nc = 3      # SU(3)
matrix = result[i].getHost().reshape(-1, Nc, Nc)
re_tr = np.trace(matrix, axis1=-2, axis2=-1).real

# 这里只需要全局总和：先在 rank 内归约，避免依赖 flattened 数组被当作
# gatherLattice suffix 保留的版本细节，也避免传输整个局域场。
local_total = np.asarray(re_tr).sum(dtype=np.float64)
payload = np.asarray(local_total, dtype=np.float64).reshape(1, 1, 1, 1)
gathered = core.gatherLattice(payload, [-1, -1, -1, -1])

if core.getMPIRank() == 0:
    global_sum = float(np.asarray(gathered).sum())
```

当前参考实现的 `gatherLattice` 在 root 返回数组、非 root 返回 `None`，因此 root 上的
最终 `.sum()` 不能省略。若需要保留逐点场而非总和，应恢复并传入当前版本要求的局部
parity/tzyx 布局；除非当前 API 明确支持其他归约，否则不要猜测 `mpi4py.Allreduce` 或
`core.allreduce` 的语义。

## Polyakov 圈与拓扑荷

Polyakov 圈是固定空间点沿完整时间方向的有序乘积及颜色迹；其复中心相位是
物理信息，不得默认取实部。要区分单点、空间平均和 ensemble 平均。拓扑荷可由
Clover 场强构造：

\[
Q=\frac1{32\pi^2}\sum_x\epsilon_{\mu\nu\rho\sigma}
\mathrm{Tr}[F_{\mu\nu}(x)F_{\rho\sigma}(x)].
\]

有限格距上 `-i(Q-Q†)/8` 可含单位阵分量，标准 `SU(Nc)` 观测量先做

\[
F^{\mathrm{su}(N_c)}_{\mu\nu}
=F_{\mu\nu}-\frac{\operatorname{Tr}F_{\mu\nu}}{N_c}I.
\]

PyQCD 的默认拓扑密度使用该无迹场及等价三项式

\[
q(x)=\frac{\operatorname{Tr}(F_{01}F_{23})
-\operatorname{Tr}(F_{02}F_{13})
+\operatorname{Tr}(F_{03}F_{12})}{4\pi^2}.
\]

必须在文档中注明 $F$ 的格点归一化、是否流化、边界和整数化方式；未流化的离散
`Q` 受 UV 噪声影响，不能只凭接近整数判断实现正确。

## 最小验收矩阵

| 检查 | 证据 |
|---|---|
| 退化圈 | $R=0$ 或 $T=0$ 的实现结果与单位元约定一致 |
| 方向反转 | 反向路径与原路径共轭关系成立 |
| 旋转 | XT/YT/ZT 在各向同性设置下可比较 |
| 归一化 | 逐点、体积平均和 ensemble 平均的分母明确 |
| 颜色代数 | 默认 Clover 满足 `F(nu,mu)=-F(mu,nu)`、Hermitian 且 `Tr F=0` |
| 幺正性 | raw/流化链接的 $U^\dagger U-I$ 范数有记录 |
