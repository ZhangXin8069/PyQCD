# 纯规范观测量 reference

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
matrix = result[i].getHost().reshape(-1, Nc, Nc)
re_tr = np.trace(matrix, axis1=-2, axis2=-1).real
global_sum = core.gatherLattice(re_tr, [-1, -1, -1, -1])
```

仅 root 写出 `global_sum`。除非当前 API 明确支持其他归约，否则不要猜测
`mpi4py.Allreduce` 或 `core.allreduce` 的语义。

## Polyakov 圈与拓扑荷

Polyakov 圈是固定空间点沿完整时间方向的有序乘积及颜色迹；要区分单点、空间平均
和 ensemble 平均。拓扑荷可由 Clover 场强构造：

\[
Q=\frac1{32\pi^2}\sum_x\epsilon_{\mu\nu\rho\sigma}
\mathrm{Tr}[F_{\mu\nu}(x)F_{\rho\sigma}(x)].
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
| 幺正性 | raw/流化链接的 $U^\dagger U-I$ 范数有记录 |
