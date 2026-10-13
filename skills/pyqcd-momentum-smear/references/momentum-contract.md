# Momentum-smearing 公式、数据与验证契约

本文件是 `pyqcd-momentum-smear` 的按需参考。读取时机：实现 link kernel、
审计 source/sink/sequential phase chain，或定义生产与验收元数据。

## 1. 物理对象

固定时间片上，phase-smearing mode 与 physical momentum 分离：

$$
k_{\rm smear}=(k_x,k_y,k_z),\qquad
P_i,\ P_f,\qquad q=P_f-P_i.
$$

`k_smear` 是无量纲实数 mode；hadron momentum $P_i,P_f$ 由最终 Fourier
projection 定义；$q$ 由 insertion 定义。三者需要分别保存。

## 2. Phased-link Wuppertal kernel

复制 gauge 后，仅修改空间 links：

$$
U_j^{(k)}(x)=e^{+i\kappa_j}U_j(x),\qquad
\kappa_j=\frac{2\pi k_j}{L_j^{\rm global}},\quad j=x,y,z;
$$

$$
U_t^{(k)}(x)=U_t(x).
$$

单步 Wuppertal operator（忽略整体归一化）为

$$
\begin{aligned}
(\Phi^{(k)}q)(x)=q(x)+\alpha\sum_{j=1}^{3}\big[&
U_j(x)e^{+i\kappa_j}q(x+\hat j)\\
&+U_j^\dagger(x-\hat j)e^{-i\kappa_j}q(x-\hat j)\big].
\end{aligned}
$$

重复 $N$ 次得到 $(\Phi^{(k)})^Nq$。每一步都必须使用 phased links。把已完成
ordinary smearing 的 field 乘一个 plane wave 不是同一算子。

PyQUDA `gaussianSmear` wrapper 的常用映射为

$$
\alpha=\frac{1}{4N/\rho^2-6},\qquad
\rho>0,\quad N\ge1,\quad \rho^2<\frac{2N}{3}.
$$

对该参数域之外的值 fail closed。参数合法性不证明给定 $\rho,N$ 最优。

## 3. Mode、坐标与轴序

| 对象 | 约定 | 必须记录 |
|---|---|---|
| smearing mode | `(kx,ky,kz)` | fractional 值、符号、每个 flavor/line 的 ownership |
| PyQCD basis momentum | `(pz,py,px)` | 与 `(kx,ky,kz)` 的显式映射 |
| source coordinate | 接口若用 `(x,y,z,t)` | 与数组 `(t,z,y,x)` 分离 |
| PyQCD gauge | `(Nt,Nz,Ny,Nx,4,3,3)` | Lorentz 标签 `0=x,1=y,2=z,3=t` |
| PyQUDA fermion | `(parity,Lt,Lz,Ly,Lx//2,...)` | parity、spin/color 与 sink/source orientation |
| spatial phase | global `(x,y,z)` | global extent、rank offset、`grid_coord` |

在 rank 上构造 Fourier phase 时必须使用

$$
\phi_{\mathbf p}(x,y,z)=
\exp\left[-2\pi i\left(
\frac{p_xx_{\rm global}}{L_x}+\frac{p_yy_{\rm global}}{L_y}
+\frac{p_zz_{\rm global}}{L_z}\right)\right],
$$

其中 $(x_{\rm global},y_{\rm global},z_{\rm global})=(x_0+x,y_0+y,z_0+z)$。

## 4. Source、sink 与 free-field 起点

meson 常用相反 mode 的两条 line：spectator `+k`，active `-k`；S-to-S 时 sink
分别使用 `+k_f` 与 `-k_f`。在来源实现的共轭路径下，free-field heuristic 为
$P_{\rm meson}\sim2k$。这些只是 tuning 起点。

degenerate baryon 可复用相同的 `+k` line；free-field heuristic 为
$P_{\rm baryon}\sim-(k_1+k_2+k_3)$。不等 mass/flavor 时，每条 line 独立记录
$k_i$，不能沿用 degenerate helper 的假设。

最终精确动量仍由

$$
C_2(t,\mathbf P)=\sum_{\mathbf x}
e^{-i\mathbf P\cdot(\mathbf x-\mathbf x_0)}C_2(t,\mathbf x)
$$

定义。修改 $k$ 不会自动修改物理 momentum label。

## 5. S-to-P、S-to-S 与 one-smear

| 模式 | source kernel | sink endpoint | sequential inversion |
|---|---|---|---|
| S-to-P | 恰好一次 | 禁止 sink mode | 不适用 |
| S-to-S | 恰好一次 | 恰好一次并记录 | 若 builder 已涂抹，传 `already_smeared=True` 且 mode 为 `None` |
| 未涂抹 sequential source | 已由上游完成 | 尚未完成 | 传 `already_smeared=False` 与一个明确 sink mode，inversion 前应用一次 |

任何“可能已涂抹”的状态不可用默认值补齐。必须区分 0 次与 2 次涂抹并拒绝。

## 6. Fourier 与 dagger sign chain

完整链条：

```text
source point
  -> source link phase and Wuppertal kernel
  -> forward propagator
  -> optional sink endpoint
  -> physical +P_f sink projection exp[-i P_f.(x-x0)]
  -> sequential candidate exp[+i P_f.(x-x0)]
  -> caller-proved downstream dagger/conjugation
  -> insertion phase carrying q=P_f-P_i
```

`sequential12` 一类仅选 sink time-slice 的接口不提供 dagger。除非 contraction
显式证明该共轭，正相位 candidate 必须拒绝。不能同时在 sequential source 与最终
sink 都使用 $e^{-iP_fx}$ 来猜测抵消。

## 7. Artifact 元数据

每条输出至少记录：

| 类别 | 字段 |
|---|---|
| ensemble | configuration ID、gauge path/hash/revision、action、mass、boundary |
| lattice | global/local extents、grid、`grid_coord`、precision |
| line | flavor、source/sink mode、source coordinate、S-to-P/S-to-S、phase owner |
| physics | $P_i,P_f,q$、Fourier 符号、sequential dagger owner |
| solver | action、tolerance、iterations、reported/true residual、`mrhs`/`restart` |
| cost | setup、solve、total time；若测试则记录 device/rank |
| artifact | shape、轴序、dtype/backend、checksum、artifact ID、command、version |

不得把 momentum-smeared source/propagator 标成 blending `eigvecs`。需要组合时必须
建立显式 callback/solve/assembly 边界并保留两侧独立 ID。

## 8. 验收 recipe

1. Unit-gauge/constant-gauge 小格点检查 link phase 与每一步 backward conjugate。
2. 翻转 mode 或 Fourier sign，确认 observable/phase oracle 随预期变化。
3. 检查 time links 数值不变、raw gauge 未被覆盖、global offset 改变时 phase 正确。
4. 对 S-to-P 非空 sink mode、缺少 one-smear flag、双涂抹状态分别断言 fail closed。
5. 用 $\gamma_5$ Hermiticity 和独立 complex contraction 检查 source/sink orientation。
6. 对每次真实 solve 保存 true residual；将 contraction 与独立 reference 的复数
    tolerance 比较，不能只比较绝对值或峰位置。

以上 recipe 是计划与最小门，不表示当前 PyQCD 已在 GPU/MPI/QUDA 上运行。
`no-runtime-claim` 始终有效。
