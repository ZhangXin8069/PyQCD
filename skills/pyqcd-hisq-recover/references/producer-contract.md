# HISQ recover producer contract

本参考文件定义 Kawamoto-Smit reconstruction 的公式、producer metadata、MPI ownership
和 fail-closed 规则。入口工作流见 [`../SKILL.md`](../SKILL.md)。

## 1. Transformation

使用 global coordinates：

```text
x = (t,z,y,x)

Omega(x) =
  gamma_x^(x mod 2)
  gamma_y^(y mod 2)
  gamma_z^(z mod 2)
  gamma_t^(t mod 2)

G_naive(x,x0) = Omega(x) G_chi(x,x0) Omega(x0)^dagger
```

Gamma matrices 不可交换，因此 order 固定为 `x,y,z,t`。不能交换乘法顺序，也不能将
`Omega(x0)^dagger` 写成 `Omega(x0)`。

`G_chi` 必须是同一 source、gauge/background 和 boundary condition 下的
one-component staggered/HISQ color propagator。taste-projected block、naive-spin
block、checkerboard field 或带额外 phase 的 source 都需要单独 adapter，不能直接套用
本公式。

## 2. Gamma basis

`gamma_ops` 必须严格包含：

```text
keys = {t, z, y, x, unit}
```

每个 value 必须是同一 documented Euclidean basis 下的 numerical、finite `(4,4)`
matrix。调用者还要验证：

- Clifford algebra；
- `unit` 与 basis 一致的 identity；
- producer 使用的 gamma matrix mapping；
- `Omega(x)` 的 order；
- staggered eta phase、origin 和 boundary wrap 是否与该 basis 匹配。

小矩阵必须在 reconstruction 前 fail closed。matrices 应转为 propagator dtype，使
complex64 输入不会静默提升为 complex128。

可用的最小 spin oracle 包括：

- 一个 odd coordinate 的 `Omega`；
- `x,y,z,t` 多个 odd coordinates，用于暴露非交换 order；
- `Omega(x0)^dagger` 的 reversed order；
- color identity 输入，使输出只保留 spin factor。

Kawamoto-Smit 方法中的 spin-diagonalization consistency 通常还要检查：

```text
Omega(x)^dagger gamma_mu Omega(x + mu_hat)
    = eta_mu(x) I
```

其中 eta convention、coordinate origin 与 boundary wrap 必须由 producer 固定。该恒等式
是 producer check 的一部分，不替代 metadata。

## 3. Array layouts

Global resident API：

```text
input:  (GLt,GLz,GLy,GLx,3,3)
output: (GLt,GLz,GLy,GLx,4,4,3,3)
```

MPI-local API：

```text
input:  (Lt,Lz,Ly,Lx,3,3)
output: (Lt,Lz,Ly,Lx,4,4,3,3)
```

输出 axes 的语义为：

```text
(t,z,y,x,sink_spin,source_spin,sink_color,source_color)
```

输入 dtype 只允许 complex64 或 complex128；输出保持相同 dtype 和 NumPy/CuPy backend。
输出是 Wilson-like storage 中的 naive four-spin lift，不是 Wilson-action propagator。

## 4. Source and boundary conventions

推荐的 source API 是 full global coordinate：

```text
x0 = (t0,z0,y0,x0)
```

历史 scalar `tsource` 的含义必须严格解释为：

```text
x0 = (t0,0,0,0)
```

对 displaced spatial source 传 scalar 会得到错误的 `Omega(x0)^dagger`。

本 transformation 不插入额外的：

- staggered `eta_mu(x)` phase；
- taste phase；
- anti-periodic temporal wrap sign；
- source normalization；
- coordinate-origin shift。

这些 factor 必须声明为 producer 已包含、下游显式处理，或由 versioned adapter 转换。
跨 temporal boundary 的 sign 可能改变，即使 `Omega(x)` 只看 coordinate parity；两个
效应不能混为一谈。

## 5. MPI coordinates and parity

PyQUDA metadata order 是：

```text
size, grid_size, grid_coord = (x,y,z,t)
```

arrays 和 source 是：

```text
(t,z,y,x)
```

若 local shape 为 `(Lt,Lz,Ly,Lx)`，rank coordinate 为 `(gx,gy,gz,gt)`，local
coordinates 必须使用：

```text
t = gt * Lt + local_t
z = gz * Lz + local_z
y = gy * Ly + local_y
x = gx * Lx + local_x
```

先形成 global coordinates，再取 parity。仅使用 `arange(Lt/Lz/Ly/Lx)` 会把每个 rank
都错误地当成 origin parity。

## 6. Gather ownership

`gather=False` 的输出契约：

- 每个 rank 返回其 local reconstructed field；
- NumPy 输入保持 NumPy；
- CuPy 输入保持 CuPy；
- 下游优先在 local backend 做 contraction，再 reduce compact observables。

`gather=True` 的输出契约：

1. CuPy field 必须显式转到 host；
2. 由 PyQUDA communicator collective gather 完整 volume field；
3. active communicator rank 0 返回 global NumPy field；
4. 其他 ranks 返回 `None`；
5. rank ownership 不以可能陈旧的 geometry snapshot 为准。

`gather` 只能接受 Python/NumPy Boolean。字符串 `"false"`、整数 `0/1` 或任意 truthy
object 必须在 metadata access 与 reconstruction 前拒绝。

full gather 的 memory size 至少是 global volume、spin factor 和 color factor 的乘积。
它可能远大于 local contraction 后的小 observable，必须显式规划而非作为默认路径。
本契约不证明 CUDA-aware transport 或 multi-rank gather 已验证。

## 7. Versioned producer metadata

在调用 generic kernel 前，production adapter 必须提供：

```text
producer_name
producer_version
source_file_format
source_dataset
global_shape_tzyx
local_shape_tzyx
grid_size_xyzt
grid_coord_xyzt
input_layout
input_checkerboarding
rank_ownership
source_global_tzyx
source_normalization
gamma_basis_id
omega_gamma_order
staggered_eta_convention
taste_or_naive_component
lattice_origin_tzyx
coordinate_origin_base
boundary_signs_xyzt
temporal_wrap_already_applied
complex_dtype
configuration_or_gauge_id
checksum
```

建议同时保存 adapter version、producer command、software version 和 source file checksum。

以下情况都不能由 shape 推断：

- `G_chi` 是否为公式期望的 one-component object；
- source normalization；
- checkerboard half-volume 是否已经 applied；
- temporal boundary sign 是否已经 included；
- taste/operator basis；
- origin 与 staggered eta convention。

## 8. Fail-closed decision

若 producer metadata 缺失、版本未知、file checksum 不匹配，或 input layout、
checkerboarding、taste、origin、boundary 中任一项未固定，则：

```text
status = BLOCKED_BY_PRODUCER_METADATA
```

不得：

- 用 shape match 宣告 producer compatibility；
- 猜默认 DeGrand-Rossi basis；
- 把 scalar source 当 full coordinate；
- 补乘或撤销未声明的 eta/taste/boundary phase；
- 把 `evengrid` 解释成 even-site subset；
- 把 local parity 当 global parity；
- 在非 root rank 使用 `gather=True` 的 return field。

## 9. Verification matrix

Producer-specific promotion 至少覆盖：

| Case | 必要断言 |
|---|---|
| `1^4` 或 `2^4` color identity | 每个 coordinate 的 `Omega(x) Omega(x0)^dagger` 与 explicit product 一致 |
| multiple odd coordinates | 暴露 `x,y,z,t` noncommuting order |
| displaced source | 四个 source parity 正确，尤其空间坐标非零 |
| global `2^4`/small lattice | 每个 site 与独立 oracle 一致 |
| complex64 / complex128 | dtype 和数值精度按契约保持 |
| single rank | global API 与 MPI `gather=False` 一致 |
| multi-rank | rank global offsets、global parity 与重组结果一致 |
| temporal boundary | wrap sign 与 producer metadata 一致 |
| optional gather | rank 0 ownership、非 root `None`、local backend 行为一致 |
| producer comparison | 与 producer 自己的 spin lift 或独立 reference 在冻结 checksum 上对照 |

在这些 checks 完成前，只能报告 generic reconstruction contract 或 candidate，
不能报告 named-producer compatibility、GPU、MPI 或 interacting observable validation。

## 10. Sources and evidence ceiling

来源快照：

```text
skills/LQCD_Skills/skills/pyquda_hisq_recover/
commit 90bcbb8739c80b41a3965f41dab50bdfc9841b16
```

方法参考：

- N. Kawamoto and J. Smit, Nucl. Phys. B192 (1981) 100,
  DOI `10.1016/0550-3213(81)90196-6`: spin diagonalization.
- E. Follana et al., arXiv:hep-lat/0610092:
  HISQ action and naive/staggered taste context.

这些来源只支持方法和背景，不支持任意 producer 的 gamma、taste、origin、source 或
boundary convention。来源脚本未复制到本技能。

本任务没有运行实现，也没有验证 CuPy、PyQUDA gather、multi-rank transport、producer
files 或 interacting correlators。最高证据保持为 PyQCD producer/metadata contract。
