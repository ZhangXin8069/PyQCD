# Gluon bare-observable 物理契约

## 1. 证据上限

来源提交：`90bcbb8739c80b41a3965f41dab50bdfc9841b16`。

局部函数构造 bare lattice fields 或 gauge-fixed candidates。当前没有闭合 operator
mixing、continuum matching、renormalization matrix 或 `MS-bar` scheme。最高证据只到
选定小数组的代数和 backend contract；不能说 production renormalized observable。

## 2. 方向、量纲与链接

来源固定 Lorentz order：

```text
0=x, 1=y, 2=z, 3=t
epsilon[0,1,2,3] = +1
```

PyQCD 生产布局为 `(Nt,Nz,Ny,Nx,4,3,3)`；迁移来源的 PyQUDA
`(direction,e,t,z,y,xh,Nc,Nc)` 只用于解释算法，不能直接当作本库 layout。

Links 无量纲：

```text
U_mu(x) ~ exp[i a g0 A_mu(x + a*mu_hat/2)]
A_code   ~ a A_continuum
F_code   ~ a^2 F_continuum
```

因此 `FF`、`FW_FW`、`FW_tildeFW` 在求和/平均前带有隐式 `a^4`；物理量换算还需
reviewed volume normalization 和 operator normalization。

## 3. `g0` normalization

```text
beta = 6 / g0^2                       # wilson
beta = 10 / (g0^2 * u0^4)             # legacy_tadpole_10
```

Wilson 分支必须省略 `u_0`；legacy 分支必须同时显式给出 `u_0` 和 normalization。
ambiguous positional call 必须 fail closed。

## 4. `A_mu`、Clover 与 dual

half-link：

```text
A_code(x + mu_hat/2)
  = (1/g0) [ (U_mu - U_mu^dagger)/(2i) ]_traceless
```

site-centered branch 对前后 anti-Hermitian link parts 取平均。`A_mu` 是 gauge
dependent，必须记录 gauge-fixing functional、stopping residual、residual-gauge
convention 和 Gribov-copy policy。

Clover：

```text
F_mu_nu = -F_nu_mu
F_tilde_alpha_beta = 1/2 epsilon_alpha_beta_mu_nu F_mu_nu
(F_xy,F_xz,F_yz) = (B_z,-B_y,B_x)
```

magnetic tuple 不是 `(B_x,B_y,B_z)`。须按 pair map 读取，不能按位置猜符号。

## 5. Wilson-line products

代码定义的 closed-color-trace 结构：

```text
-1/2 Tr[F1(x) W_d(x,x+n*d_hat)
         F2(x+n*d_hat) W_d^dagger]
```

来源的零长 oracle：

```text
FW_FW(link_length=0) = -1/2 FF
```

仅在 fields、orientation、communicator 和 reduction convention 相同的情况下成立。
它不提供 continuum power divergence、line self-energy、mixing 或 matching。

## 6. Fourier 和 half-link

single-mode phase 使用 global centered coordinates：

```text
exp[-2*pi*i * sum_mu p_mu x_mu/(G_mu L_mu)]
```

整数标签先 canonicalize 到 FFT fundamental bins；even extent 的 Nyquist
representative 固定为 `-L_mu/2`。global sum 后除以 global volume。

link-centered `A_mu(x+mu_hat/2)` 的方向 `mu` 单独乘：

```text
exp(-i p_mu/2)
```

single-momentum 与 all-momentum paths 都必须保持这一 direction-specific correction。
历史 `_MPI` 名称不证明 distributed FFT；若实现要求 one rank，应显式拒绝 nontrivial grid。

## 7. Time decomposition 与 MPI

time-resolved observable 只有在一个 rank 拥有完整 global-time decomposition 时才能安全
做 world reduction。否则 `Gt>1`/跨时间归约会混合不同 global time slices。

需要保存：

```text
local/global extent
process grid and rank coordinates
ownership: local-time | spatial-reduced | globally-assembled
root output and non-root return convention
```

fake communicator 或单 rank test 最多证明接口分支。

## 8. EMT 与 current blockers

legacy EMT candidate 的结构可写为

```text
T_mu_nu_code = 2 sum_x,rho Tr[F_mu_rho F_nu_rho]
               - delta_mu_nu * 1/2 sum_x,alpha,beta
                 Tr[F_alpha_beta F_alpha_beta]
```

`gaugeEMT_mumu*` 是另外的离散形式，不能假定可互换。

gauge-dependent spin/current candidate：

```text
(E x A)_i_code = sum_x epsilon_ijk Tr[E_j A_k]
K_mu_code ~ epsilon_mu_nu_rho_sigma
             Tr[A_nu F_rho_sigma
                - (2 i g0/3) A_nu A_rho A_sigma]
```

以下 blocker 未关闭：

1. exact `a/g0/trace/volume` normalization；
2. Euclidean-to-Minkowski map；
3. gauge-fixing functional/residual/Gribov copies；
4. EMT trace anomaly、vacuum subtraction 和 hypercubic mixing；
5. quark/gluon mixing basis 与 renormalization matrix；
6. scheme、scale、matching 和 covariance；
7. independent interacting gauge-fixed/renormalized reference。

因此这些输出只能称 `bare_emt_candidate`、`diagnostic_current` 或
`gauge_fixed_candidate_*`。

## 9. 独立验收记录

任何 bare contraction 的独立 oracle 必须：

- 不由同一 contraction/control flow 产生；
- 保存 source、SHA-256 和 component-resolved complex values；
- 比较 real/imaginary parts，报告绝对/相对容差与逐分量差值；
- 明确声明它只验证 tested bare contraction，不提供 normalization/renormalization。

`bare`、`gauge_fixed`、`mixed`、`renormalized` 必须作为独立 artifacts 保存。
