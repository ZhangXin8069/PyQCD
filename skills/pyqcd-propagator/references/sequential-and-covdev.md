# 顺序源与协变位移 reference

## 适用范围

任务需要重子三点函数、顺序源、非局部 Wilson 线或 `covDev` 时读取。本文件只说明
传播子生产与上下文边界；算符、费米子符号和最终 einsum 必须先由
`pyqcd-physics-correlator` 给出。

## 三点顺序源

固定汇时刻、汇动量、sink smearing 和投影后，按以下顺序执行：

1. 用两条 spectator 线构造汇块 `B`；
2. 按约定做第一次 `gamma5 B† gamma5`；
3. 以 `source.sequential12(B, t_seq)` 构造源并对相应 flavor 求逆；
4. 对顺序传播子做第二次 dagger；
5. 与 current-side 传播子和 `Gamma_current` 收缩，再在上下文外归约。

示意代码只表达轴和资源生命周期：

```python
B = core.LatticePropagator(latt_info)
B.data = ...  # einsum 必须来自 correlator 交接表
B.data = contract(
    "AB,wtzyxCBji,CD->wtzyxADij", G5, B.data.conj(), G5
)
src_seq = source.sequential12(B, t_seq)
with dirac_l.useGauge(gauge_stout):
    prop_seq = core.invertPropagator(dirac_l, src_seq)
prop_seq.data = contract(
    "AB,wtzyxCBji,CD->wtzyxADij", G5, prop_seq.data.conj(), G5
)
three_pt_site = contract(
    "wtzyxijba,jk,wtzyxkiab->wtzyx",
    prop_seq.data, gamma_current, prop_current.data,
)
```

实际 `B` 的 color/spin 转置必须和物理推导逐指标对照；不能把这段示意字符串当成
所有重子或 flavor 的通用公式。`t_seq`、源汇动量、投影、current 和求逆残差随产物保存。

## 非局部协变位移

非局部双线性
`bar(q)(x) Gamma W(x,x+z) q(x+z)` 使用 raw 规范场构造 Wilson 线。求逆可使用
涂抹场，但两者必须保留为独立对象：

```python
gauge_raw = io.readChromaQIOGauge(cfg_path)
gauge_raw.toDevice()
gauge_stout = gauge_raw.copy()
gauge_stout.stoutSmear(n_step, rho, n_dim)
```

对每个 `z` 从原始传播子 copy 出发，逐一遍历 4 个 spin 和 3 个 color 分量；一次
`covDev` 只移动一步。局部相关函数先缩到二维 `(z, Lt_local)`，再离开 raw 场上下文
逐 `z` 调用 `gatherLattice`：

```python
C_loc = cp.zeros((zmax + 1, latt_info.Lt), dtype=cp.complex128)
with gauge_raw.use() as dirac_shift:
    for zsep in range(zmax + 1):
        prop_shift = prop_l.copy()
        for _ in range(zsep):
            for spin in range(4):
                for color in range(3):
                    field = prop_shift.getFermion(spin, color)
                    field = dirac_shift.covDev(field, Z)
                    prop_shift.setFermion(field, spin, color)
        C_loc[zsep] = contract(
            "wtzyxjiba,wtzyxjiba->t", prop_l.data.conj(), prop_shift.data
        )
for zsep in range(zmax + 1):
    C_full[zsep] = core.gatherLattice(C_loc[zsep].get(), [0, -1, -1, -1])
```

实际代码需按本地数组长度分配 `C_full`，并只由 root 写出。`C_loc` 不能保留已收缩的
parity 维；不要对整个传播子一次 `covDev`，不要复用前一个 `z` 的累积结果。

## 必检边界

| 错误 | 检查 |
|---|---|
| raw 场被 smear 原地覆盖 | smear 前 copy，并记录 raw/stout 用途 |
| 求逆和 covDev 上下文重叠 | 先退出 `useGauge`，再进入 `raw.use()` |
| 在 QUDA 上下文内 gather | 把归约移到上下文外，避免 MPI/资源死锁 |
| 只移动一个 spin/color | 对每个分量逐步检查输出 shape 和颜色闭合 |
| 端点/长度错误 | 对 `z=0,1` 手算路径，并核对反向路径共轭关系 |
