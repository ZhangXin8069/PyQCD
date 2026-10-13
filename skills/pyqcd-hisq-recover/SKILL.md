---
name: pyqcd-hisq-recover
description: |
  Use when a PyQCD task needs to reconstruct a staggered or HISQ one-component
  propagator into a naive four-spin Wilson-like layout, validate Omega(x) and
  its gamma order, handle global source coordinates, interpret the historical
  evengrid name, or review MPI-local parity and gather ownership.
metadata:
  provenance:
    source_commit: 90bcbb8739c80b41a3965f41dab50bdfc9841b16
---

# pyqcd-hisq-recover - HISQ 四旋量恢复

## 目的与边界

本技能用于实现审查 Kawamoto-Smit spin lift：

```text
G_naive(x,x0) = Omega(x) G_chi(x,x0) Omega(x0)^dagger
```

输入是一个把 color indices 作为唯一 fermion indices 的 staggered/HISQ
one-component propagator；输出是 Wilson-like spin-color storage。这里的
"Wilson-like" 只描述数组布局，不是 Wilson action conversion。

本技能不执行 Dirac inversion、taste projection、correlator contraction、fit 或
continuum extrapolation，也不补齐或撤销 producer 已施加的 staggered phase、taste
phase、origin、source normalization、temporal wrap sign。

在读取 producer 数据、实现 MPI helper 或接受任何兼容性结论前，必须阅读
[`references/producer-contract.md`](references/producer-contract.md)。

## 物理契约

### 1. Omega(x) 的固定次序

坐标定义为 `x = (t,z,y,x)`，恢复因子为：

```text
Omega(x) =
  gamma_x^(x mod 2)
  gamma_y^(y mod 2)
  gamma_z^(z mod 2)
  gamma_t^(t mod 2)
```

Gamma matrices 不可交换，因此 `x,y,z,t` 的乘法次序是接口的一部分。`gamma_ops` keys
必须恰为 `t,z,y,x,unit`；每个 matrix 必须是同一约定下的 numerical、finite `(4,4)`。
不能仅凭 shape 或变量名推断 producer 使用相同 gamma basis。

### 2. Source coordinate

source 必须解释为 global `(t,z,y,x)` coordinate。历史的 scalar `tsource` 只等价于
`(t_source,0,0,0)`；对空间位移源继续传 scalar 会使用错误的 `Omega(x0)^dagger`。

MPI-local reconstruction 也必须先在 sink coordinates 上加入 rank 的 global offset，再取
coordinate parity。不能把 local parity 当成 global parity。

### 3. `evengrid` 和完整网格

`evengrid` 是历史函数名。它指完整 lexicographic grid，不是 even-site subset、
checkerboard half-volume 或 parity-projected field。输入与输出 axes 固定为：

```text
input:  (t,z,y,x, sink_color, source_color) = (...,3,3)
output: (t,z,y,x, sink_spin, source_spin,
                sink_color, source_color) = (...,4,4,3,3)
```

输出保持 NumPy/CuPy backend 和 complex64/complex128 dtype。它是 naive four-spin lift，
不是 Wilson-action propagator。

### 4. MPI global parity 与 gather ownership

PyQUDA `size`、`grid_size`、`grid_coord` 的轴序是 `(x,y,z,t)`，而 propagator 和 source
使用 `(t,z,y,x)`。每次 local reconstruction 都要显式重排并核对 global offsets。

`gather=False` 返回每个 rank 的 local field，并保持原 backend；生产收缩应优先采用此
路径。`gather=True` 表示完整 volume field 先转到 host，再由 PyQUDA gather：

- rank 0 返回 global NumPy field；
- 非 root ranks 返回 `None`；
- root ownership 依据 active communicator rank，而不是可能陈旧的 geometry snapshot；
- 不允许用字符串或整数 truthiness 控制 gather。

### 5. Producer metadata fail-closed

目标 producer 必须提供参考契约中的 versioned metadata record。尤其要固定：

- producer name/version、file format/dataset、configuration ID 和 checksum；
- global/local `tzyx` shape、input layout、checkerboarding、rank ownership；
- full source coordinate 和 source normalization；
- gamma basis、`Omega` gamma order、staggered eta convention、taste component；
- lattice origin、coordinate origin base、boundary signs、temporal wrap state；
- complex dtype。

缺少 taste、origin、boundary、layout、source 或 gamma metadata 时，输出
`BLOCKED_BY_PRODUCER_METADATA` 并停止；不得根据 shape 猜测兼容性。

## 工作流

1. 冻结 producer metadata 与 checksum，将 metadata 保存到 reconstruction 产物旁。
2. 固定 `Omega(x)` 的 `x,y,z,t` order、gamma basis、`(t,z,y,x)` 与
   `(x,y,z,t)` 之间的全部 reorder。
3. 规范化 source 为完整 global `(t,z,y,x)`；只在明确原点源兼容场景接受 scalar。
4. 对 global resident field 或 MPI-local field 构造 global-coordinate parity，执行 spin
   lift，并保持 dtype/backend。
5. 用 color-identity 小格点、displaced source 和跨 boundary coordinate 做数值 oracle；
   对 MPI 路径核对 non-root ownership 和 optional gather。
6. 普通 contraction 优先保持 local `gather=False`，只向下游传递 compact observables；
   确需 full field 时才允许 root gather。

## 验收门

- [ ] `Omega` 严格按 `gamma_x, gamma_y, gamma_z, gamma_t` 顺序相乘，source dagger 使用
  `Omega(x0)^dagger`；非交换顺序 mutation 能被测试区分。
- [ ] `gamma_ops` 恰好包含 `t,z,y,x,unit`，所有矩阵 finite、numeric、精确 `(4,4)`，
  并来自 producer 已核对的 basis。
- [ ] full source `(t,z,y,x)` 的四个 parity 都被覆盖；scalar source 只在
  `(t,0,0,0)` 场景使用。
- [ ] `evengrid` 输入被证明是完整 lexicographic grid，不是 even-site/checkerboard
  subset。
- [ ] 输出 axes 为 `(t,z,y,x,4,4,3,3)`，complex64/complex128 与 backend 均保持不变。
- [ ] MPI coordinates 使用 global offsets 计算 parity；`(x,y,z,t)` 与 `(t,z,y,x)`
  reorder 在代码和 metadata 中同时可见。
- [ ] `gather=False` 保持 local field；`gather=True` 只由 rank 0 返回 host global field，
  非 root 返回 `None`，且未把完整 volume gather 当作默认生产路径。
- [ ] producer metadata 不完整时 fail closed，不生成兼容性结论。

## 技能交接

| 对象 | 交接内容 |
|---|---|
| `pyqcd-conventions` | gamma basis、`(t,z,y,x)` 轴序、单位、时间边界、结果状态和 metadata 最低集合；先在此固定共享契约 |
| `pyqcd-propagator` | producer/source 信息、inversion output layout、source normalization、boundary 处理、residual、backend 和 dtype；本技能只做 spin lift，不重新求逆 |
| `pyqcd-physics-correlator` | 明确下游需要 one-component、naive four-spin 或 taste-projected object，以及 flavor、operator、source/sink 和 Wick structure；未固定 taste/operator 前不进入 contraction |

交接时必须附：producer metadata、source coordinate、完整 coordinate reorder 表、
`Omega` gamma order、输入输出 shape/dtype、MPI rank ownership、local 或 gathered 状态、
oracle 命令及证据上限。

## 来源与证据上限

来源是只读目录
`skills/LQCD_Skills/skills/pyquda_hisq_recover/`，提交
`90bcbb8739c80b41a3965f41dab50bdfc9841b16`。方法背景来自 Kawamoto-Smit spin
diagonalization；HISQ action/taste 背景来自 arXiv:hep-lat/0610092。来源脚本未复制到
本技能。

本技能当前只建立 PyQCD 自有的 producer, coordinate, storage 和 ownership contract。
没有命名 producer adapter，也没有在本任务中运行 implementation。GPU/CuPy、
PyQUDA `gatherLattice`、multi-rank parity、gather memory、producer compatibility 和
interacting observable 均未验证；不得把 shape 一致或单 rank oracle 写成这些结论。

只有 versioned producer adapter、small-lattice independent oracle、跨 rank/global
parity 对照、temporal boundary crossing 和 complex64/complex128 检查都通过后，才可提升
producer-specific evidence；升级前只允许报告 naive reconstruction candidate。
