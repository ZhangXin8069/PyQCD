# TMD geometry reference — flow, field strength, and staple

## 适用范围

任务涉及梯度流场、Clover/对偶场强、有限 staple、横向位移、表示选择、正负方向或
直线 OPE 对照时读取。本文件只定义几何和算符层，不负责核子统计、重整化或外推。

## 坐标与量纲契约

先固定纵向 `z`、横向向量 `b_perp`、staple 长度 `ell`、无量纲流时间
`tau=t/a²`、动量 `Pz`、表示和 Lorentz 通道，并把它们写进每个产物元数据。
当前低层几何接口的 `z/b` 使用格点整数，而 `quasi_tmd_pdf` 的 `z_grid` 用 fm、
`pz_gev` 用 GeV；内部转换后 `nu=z*Pz` 必须无量纲。

中心几何可取

\[
x_1=+z\hat z/2+\boldsymbol b_\perp,\qquad x_2=-z\hat z/2,
\]

而当前部分矩阵代码使用从 `x` 到 `x+z\hat z+\boldsymbol b_\perp` 的锚定几何。
两者不能在同一数据集混用；周期绕回、正负方向和 Fourier 符号必须显式记录。

## 流化场与算符

Wilson flow 的格点右端可写为

\[
\dot V_\mu(x)=Z_\mu\lbrack V\rbrack(x)V_\mu(x),\qquad
Z_\mu=P_{\rm ah}[\Omega_\mu V_\mu^\dagger],
\]

当前 `wilson_flow` 使用三阶 RK：

\[
\begin{aligned}
W_1&=e^{\epsilon Z_0/4}W_0,\\
W_2&=e^{(8\epsilon Z_1/9-17\epsilon Z_0/36)}W_1,\\
V_{t+\epsilon}&=e^{(3\epsilon Z_2/4-8\epsilon Z_1/9+17\epsilon Z_0/36)}W_2.
\end{aligned}
\]

在同一个 `V_tau` 上构造 Clover `F`、对偶 `F_tilde` 和场强双点：

\[
\mathcal O^{g,\tau}_{\mu\nu;\rho\sigma}=
G^a_{\mu\nu}(x_1;\tau)
[W_{\rm st}^{\mathcal R}(x_1,x_2;\ell)]^{ab}
G^b_{\rho\sigma}(x_2;\tau).
\]

实现次序固定为：

1. 每个 `tau` 都从同一原始 `U` 独立流化并独立构造算符；核心
   `tmd_matrix_elements(_time)` 只做调用内复用，不隐式跨调用或跨 `tau` 缓存。test9
   的显式 flowed-gauge/OPE 持久缓存另按 validation reference 的完整身份校验执行。
   不得串接前一流时间的 `V_tau`，或混接不同流时间的场强与线。
2. 选定表示：伴随线要验证
   `W_adj[a,b]=2*Tr(T[a] W_fund T[b] W_fund†)`；基本表示要保留带色迹的矩阵结构。
3. 两端 Clover 场先逐点投影
   $F^{\mathrm{TL}}=F-\operatorname{Tr}(F)I/N_c$，再以显式路径构造纵向段、横向段、
   回程段和转角；加入任意逐点单位阵分量不得改变闭合 bilocal。
4. 令 `i,j` 为 `z_dir` 之外的两个空间方向，使用
   `O=M^{ti;ti}+M^{tj;tj}-2M^{ij;ij}`；只有 `z_dir=2` 时才写成历史
   `tx/ty/xy` 组合，且该组合不能代替完整张量混合矩阵。

## 当前接口边界

`staple_wilson_line` / `gluon_tmd_operator` 当前使用整数 `b_perp` 和 `L`；
`z_dir/b_dir` 必须是互不相同的非布尔整数 `0,1,2`。`L=None` 时取 `L=abs(z)`。
路径从起点依次走 `-L*z_dir`、`b_perp*b_dir`、
`(L+z)*z_dir`，终点为 `x+z*z_dir+b_perp*b_dir`。颜色闭合固定为
`Tr[F(x) W(x,y) F(y) W†(x,y)]`，两个端点间隔着 `F(y)`，不能把 Wilson 线相邻抵消。

批量 `tmd_matrix_elements` 与 `tmd_matrix_elements_time` 在一次调用中只预计算三个
Clover 场 `F_ti/F_tj/F_ij`，其中 `i,j` 随 `z_dir` 动态选择；并对每个 `(z,b)` 只构造
一条 staple，复用于 `M^{ti;ti}`、`M^{tj;tj}`、`M^{ij;ij}`。这是调用内复用，不是
跨调用缓存；扫描多个流时间时，每个 `tau` 必须从原始 `U` 重新流化并独立构造。若由
test9 显式持久化，则只能复用通过完整 JSON/SHA/metadata 校验的同一物理契约。

`pipeline.step_tmd` 对一个 `tau` 只执行一次 `wilson_flow`，随后让同一流场同时供
TMD 与 `t²E` 使用。公开 `operator.staple_operator` 复用上述唯一几何，返回同 Lorentz
对的规范不变、两端逐点去迹 bilocal `M^{mu nu;mu nu}`；完整非极化动态 transverse
组合仍由 `renorm.gluon_tmd_operator` 负责。

`gluon_ope_operator_z0` 是直线 OPE（支持 `±z` 和交叉 Lorentz 对），
`gluon_ff_operator_z0` 是固定规范、无 Wilson 线的对照；二者都不能替代非零
`b_perp` 的规范不变空间 staple TMD。

### 直线 OPE 的显式通道身份

公开入口 `from pyqcd.operator import OPEChannelSpec, gluon_ope_channel` 要求把影响物理
含义的字段全部写入不可变 spec：`mode`、两组 Lorentz 对、`z_dir`、第二插入、`±z`
方向、空间归约、归一化、输出投影和场强投影。当前通道族为：

| `mode` | 第二插入 | 物理边界 |
|---|---|---|
| `legacy_dual` | `Ftilde` | docker-v20260805 的兼容通道；`+z`、取实部、全空间裸求和 |
| `unpolarized` | `F` | donghx unpolarized 的 `F·F` 单个 Lorentz 通道；完整非极化组合仍需显式系数 |
| `helicity` | `Ftilde` | `F·Ftilde` 螺旋度通道；不自动继承 legacy 的实部投影或方向 |
| `custom` | `F` 或 `Ftilde` | 只用于调用者已明确写出全部语义的受控扩展 |

完整公开调用形状为：

```python
import numpy as np
from pyqcd.operator import (
    FieldStrengthCache,
    OPEChannelSpec,
    gluon_ope_channel,
)

# A fast, exact smoke fixture: identity SU(3) links on a 2^4 lattice.
# The layout is (Nt,Nz,Ny,Nx,4,Nc,Nc); tau=0 is the unflowed boundary.
Nt, Nz, Ny, Nx, Nc = 2, 2, 2, 2, 3
gauge = np.zeros((Nt, Nz, Ny, Nx, 4, Nc, Nc), dtype=np.complex128)
gauge[...] = np.eye(Nc, dtype=gauge.dtype)
tau = 0.0

spec = OPEChannelSpec(
    mode="unpolarized",
    mu=3,
    nu=0,
    mu2=3,
    nu2=0,
    z_dir=2,
    second_insert="F",
    direction=+1,
    sum_kind="full",
    normalization="bare_spatial_sum",
    output_projection="complex",
    field_projection="legacy_untraced",
)
cache = FieldStrengthCache(gauge, flow_time=tau, max_entries=2)
ope = gluon_ope_channel(
    gauge,
    spec,
    delta_z=2,
    Nt=Nt,
    Nx=Nx,
    compute_dtype=np.complex128,
    field_strength_cache=cache,
    flow_time=tau,
)
# ope.shape == (2, 2)，第 0 轴对应 |z|/a = 0,1；第 1 轴是时间。
assert ope.shape == (2, Nt)
assert ope.dtype == np.dtype(np.complex128)
assert np.allclose(ope, 0.0)  # identity links have vanishing Clover field
```

`delta_z` 是非负位移的计数上界，`direction` 决定每个 `zi` 使用 `+zi` 或 `-zi`；
不能同时把 signed `z` 编进 `delta_z`。`compute_dtype` 选 `complex64/complex128`；即使
`output_projection="real"`，兼容路径也按该复 dtype 返回、虚部为零，避免落盘接口因
投影改变 dtype。`sum_kind="full"`、`normalization="bare_spatial_sum"` 的确切含义是
对每个时间片求 `sum_(z,y,x) Tr[...]`，不除以空间体积、`Nc`，也不另乘耦合、格距或
连续场强归一化。返回 shape 固定为 `(delta_z,Nt)`。spec 的闭集值、退化 Lorentz 对、
布尔方向或互相矛盾的 mode/insert 触发 `ValueError`；wrapper 收到非 spec 对象触发
`TypeError`。

`mode` 与 `second_insert` 冲突时立即拒绝；`direction` 只接受非布尔整数 `+1/-1`。
`output_projection="real"` 是显式投影，`"complex"` 保留复数；一般配置上不得假设
$O(-z)=O(+z)^*$，只能检验二者在 `z=0` 的共同局部极限。当前直线 OPE 仅实现
`field_projection="legacy_untraced"`，不得把它冒充 `pyqcd.gauge` 默认无迹 Clover，
也不得把公共纯规范拓扑量的 `traceless=True` 反向套入 docker 数值基线。

`gluon_ope_operator_z0` 的省略参数路径继续复现历史 `F·Ftilde` 行为；新代码应优先构造
`OPEChannelSpec` 后调用 `gluon_ope_channel`。兼容组合固定为
$-O_{30}-O_{31}+2O_{01}$，每个分量都是 `legacy_dual`，不能因 “unpolarized” 文件名
而改成 `F·F`。

`get_ope_lorentz_pairs(zdir,"unpol"|"helicity")` 给出三个单通道的
`(mu,nu,mu2,nu2)` 指派；`zdir=2` 时依次为 `(3,0,3,0)`、`(3,1,3,1)`、
`(0,1,0,1)`。它只给 Lorentz 指派，不替调用者决定目标算符的线性组合：legacy 的系数
由上述 docker 基线固定，新的 unpolarized/helicity 组合必须来自声明的物理定义并写入
元数据，不能套用文件名或另一通道的系数。当前所有直线通道（包括新 unpolarized）都只
接受 `legacy_untraced`；如需无迹直线 OPE，应先新增独立 field-projection 实现与基线，
当前调用会明确拒绝而不是静默切换。

### 场强缓存所有权与内存上界

`FieldStrengthCache(gauge, flow_time=..., max_entries=...)` 只存 `mu < nu` 的 canonical
Clover 场，反向请求严格返回负号；它绑定同一 gauge 对象、backend、device、dtype、
shape 和 flow-time token。公共默认 `max_entries=6`，管线逐通道执行时显式用 `2`；命中
必须提升 LRU 次序，超限淘汰最久未用项，不能同时常驻 `(mu,nu)` 与 `(nu,mu)`。

NumPy/CuPy 没有 O(1) 变更计数，缓存存活期间由调用者保证 gauge 不变；受控原地修改后
必须调用 `refresh()`。Torch 对可追踪原地写使用 `_version` 自动失效，但外部别名写入仍
需 `refresh()`。消费结束调用 `clear()` 并释放 cache 所有者；清理异常不得覆盖原计算异常。
cache 返回的是借用的场强对象，调用者不得原地修改；`max_entries` 约束 cache 自身持有的
引用，不约束调用者另行保留的场强或算符临时张量。

对 `72×24^3`、`Nc=3`、`complex128`，单个全场 payload 为
`72*24^3*3*3*16 / 2^20 = 136.6875 MiB`；六项与两项上界分别为
`820.125 MiB` 和 `273.375 MiB`。这里只计算缓存数组 payload，不是 allocator、RSS 或
显存峰值的严格上界。

上述结论来自当前源码与受控契约测试；尚无真实 ILDG 组态上的端到端物理验证，也未
证明 CuPy/Torch 大体积执行的数值与性能边界。后端可调用、形状测试或 CPU 对照不能
替代真实数据、多后端大体积和完整物理链验证。

## 几何验证门

| 检查 | 必须看到的证据 |
|---|---|
| 流 | `V†V≈I`、减小步长后稳定、`flow_action_density` 按声明的定义满足预期 |
| 场强 | Clover 反对称、对偶定义和 Lorentz 指派一致；两端逐点去迹，`F+c(x)I` 不改变 bilocal |
| 路径 | 每段链长/端点正确；反向路径与共轭关系成立；`W W†≈I` 只作幺正性检查 |
| 极限 | `b_perp→0` 回到直线基准，`z→0` 为局部极限，独立 `ell` 可区分 |
| 规范性 | 规范变换前后最终标量不变，颜色指标闭合 |

未通过几何门时，结果只能标为“接口/测试骨架”，不能进入软因子、CS 或 PDF 解释。

直线 OPE 与缓存的当前可执行门为：

```bash
python -B -m pyqcd.testing._ope_channel_contract
python -B -m pyqcd.testing._field_strength_cache_contract
```

complex128 小格点的独立 `z=0` 与非零 `+z` oracle 使用 `atol=2e-13`；完整 docker
中间张量仍按 `examples/docker-v20260805/output/output_20260802_120104` 的一致性门
检查。函数级 OPE 与 docker 实现使用最大绝对差 `<1e-10`；磁盘中间量使用

\[
d(a,b)=\begin{cases}
\lVert a-b\rVert_2/\lVert b\rVert_2,&\lVert b\rVert_2\ne0,\\
\lVert a\rVert_2,&\lVert b\rVert_2=0,
\end{cases}
\qquad d<10^{-6}.
\]

shape 和 NaN 位置必须完全相同；不同时令 `d=inf`，两边全 NaN 时令 `d=0`。完整验收
固定执行：

```bash
python examples/pyqcd/verify_consistency.py \
  --run-dir examples/docker-v20260805/output/output_20260802_120104
python examples/test0/main.py verify --run-dir <candidate-run> \
  --conf-ids 6250,6450,6650,6850,7050,7250,7450,7650,7850,8050
```

第一条的 `_reference_requirements()` 是逐文件清单权威：十个组态都必须含三份
`ops_mu{0_1,3_0,3_1}_dz24`、`ope_combined` 及 B--E 所需关联函数/分析产物；缺任一项
明确退出 `2`，不是 FAIL=数值超差，也不是 PASS。第二条写出 `test0_verify.json`，只有
`n_fail=0` 且 `n_missing=0` 才签署磁盘基线兼容；退出 `1` 表示超差或缺候选产物。
受控 oracle 通过不等于参考产物缺失时已完成逐文件比较，也不等于真实 ILDG 大体积 GPU
验证。当前基线内容没有单独发布 hash，因此不得编造 hash；用精确目录、文件清单、命令
和代码版本共同标识本门。
