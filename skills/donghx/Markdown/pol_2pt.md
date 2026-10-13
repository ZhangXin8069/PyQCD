# 2pt 相关函数的宇称降维与 Pauli 极化投影

本文记录从保留完整旋量指标的四维 2pt 结果，先进行宇称投影的 `4 -> 2` 降维，再进行 x、y、z 三个方向极化投影的计算流程。

## 1. 输入数据与指标约定

2pt 计算保存完整的旋量结构

$$
M_{yxil},
$$

其中 `y` 是 sink 时间，`x` 是 source 时间，`i` 和 `l` 是两个旋量指标。典型文件为

```text
twopt_slice_pp_..._contract_confXXXX.npy
```

数组形状为

```text
(Nt, Nt, 4, 4)
```

当前生产代码的投影是逐元素旋量收缩：

```python
np.einsum("il,yxil->yx", projection, M)
```

因此它表示

$$
C_{yx}=\sum_{i,l}P_{il}M_{yxil},
$$

而不是直接把 `M[y,x]` 当作普通矩阵进行 `P @ M[y,x]`。

## 2. 四维宇称投影

在当前 Euclidean DR gamma 约定下，正、负宇称投影子为

$$
P_+ = \frac{1+\gamma_4}{2},
\qquad
P_- = \frac{1-\gamma_4}{2}.
$$

它们满足

$$
P_\pm^2=P_\pm,
\qquad
P_\pm^\dagger=P_\pm,
\qquad
\operatorname{rank}(P_\pm)=2.
$$

因此正宇称只包含两个独立的旋量自由度。

## 3. 构造正宇称的 4×2 基

不能在 DR 表示中简单取 `P_plus[:, 0:2]` 作为 4×2 基，因为 `P_plus` 的像空间通常不是前两个标准基方向。

应对 `P_plus` 做厄米特本征分解：

```python
evals, evecs = np.linalg.eigh(P_plus)
U = evecs[:, evals > 0.5]
```

得到

$$
U\in\mathbb C^{4\times2},
\qquad
U^\dagger U=I_2,
\qquad
UU^\dagger=P_+.
$$

`U` 的列向量构成正宇称子空间的正交归一基。二维坐标对应的旋量指标记为 `a,b=0,1`。

## 4. 四维旋量张量降为二维旋量张量

由于生产结果的旋量收缩是逐元素形式，降维时应保持相同的指标结构：

$$
M^{(2)}_{yxab}
=
U_{ia}\,M_{yxil}\,U^*_{lb}.
$$

对应代码为：

```python
M2 = np.einsum("ia,yxij,jb->yxab", U, M, U.conj())
```

这里使用的是 `U` 和 `U.conj()`，而不是普通矩阵乘法中的 `U.conj().T`。原因是原始投影为 `P_il M_il` 的逐元素收缩。

如果采用普通矩阵乘法约定，即把某个旋量指标定义为矩阵行、另一个定义为矩阵列，则应使用相应的 `U.conj().T @ M @ U`。两种写法的选择取决于原始数据的指标定义，不能混用。

## 5. 三个方向的四维自旋算符

三个空间方向的自旋算符定义为

$$
S_k=i\gamma_k\gamma_5,
\qquad k=1,2,3.
$$

在当前 gamma 约定下，`pol15`、`pol25`、`pol35` 分别对应 `k=1,2,3`。

这些算符与正宇称投影子对易：

$$
[P_+,S_k]=0.
$$

因此它们可以限制在二维正宇称子空间内。

## 6. 二维 Pauli 矩阵

在正宇称子空间中定义

$$
\sigma_k=U^\dagger S_k U
=U^\dagger(i\gamma_k\gamma_5)U.
$$

代码为：

```python
S4 = 1j * gamma(k) @ gamma(5)
sigma_k = U.conj().T @ S4 @ U
```

数值上应检查

$$
\sigma_k^\dagger=\sigma_k,
\qquad
\sigma_k^2=I_2.
$$

由于二维基底的排序和相位不是唯一的，计算结果可能得到 `+sigma_i` 或 `-sigma_i` 的等价表示；只要正反向投影和最终差值使用同一基底，物理结果不变。

## 7. 正向和反向极化投影

二维正向、反向极化投影子为

$$
\Pi_k^{+}=\frac{I_2+\sigma_k}{2},
\qquad
\Pi_k^{-}=\frac{I_2-\sigma_k}{2}.
$$

它们满足

$$
\Pi_k^++\Pi_k^-=I_2,
\qquad
\Pi_k^+-\Pi_k^-=\sigma_k.
$$

分别计算

$$
C_{yx}^{k,+}
=\operatorname{Tr}\left[\Pi_k^+ M^{(2)}_{yx}\right],
$$

$$
C_{yx}^{k,-}
=\operatorname{Tr}\left[\Pi_k^- M^{(2)}_{yx}\right].
$$

极化相关函数定义为正向减反向：

$$
C_{yx}^{k,\mathrm{pol}}
=C_{yx}^{k,+}-C_{yx}^{k,-}
=\operatorname{Tr}\left[\sigma_k M^{(2)}_{yx}\right].
$$

在保持原始逐元素收缩约定时，代码写成：

```python
forward = np.einsum(
    "ab,yxba->yx", 0.5 * (np.eye(2) + sigma_k.T), M2
)
backward = np.einsum(
    "ab,yxba->yx", 0.5 * (np.eye(2) - sigma_k.T), M2
)
polarized = forward - backward
```

这里出现 `sigma_k.T` 是因为原始四维投影使用的是系数矩阵的逐元素收缩。若数据已经明确转换成普通矩阵乘法约定，则应采用相应的无转置形式。

## 8. 与原四维投影的等价性

四维直接投影为

$$
C_{yx}^{k,(4)}
=\sum_{i,l}(P_+S_k)_{il}M_{yxil}.
$$

利用

$$
P_+=UU^\dagger,
\qquad
S_k=U\sigma_kU^\dagger
\quad\text{(在正宇称子空间内)},
$$

并保持逐元素收缩的转置约定，可得

$$
C_{yx}^{k,(4)}
=\operatorname{Tr}\left[\sigma_k^T M^{(2)}_{yx}\right].
$$

因此二维正反向极化差与四维直接投影完全等价。

验证时应同时检查：

```python
direct = np.einsum("il,yxil->yx", P_plus @ S4, M)
reduced = forward - backward
error = np.max(np.abs(direct - reduced))
```

在 `L24x72` 的 `conf=4050` 实际数据上，x、y、z 三个方向以及多个动量的最大误差约为

```text
1.1e-18
```

达到 double 精度下的数值一致。

## 9. 时间边界符号

`L24x72_diffpol.py` 和当前 2pt 结果的后处理还包含反向时间的符号处理。对 sink 时间小于 source 时间的条目使用

```python
pol[t_sink < t_source] *= -1
```

因此比较四维与二维结果时，必须在两条路径中使用完全相同的时间符号规则，否则会得到人为的差异。

## 10. 推荐的完整流程

```python
P_plus = 0.5 * (gamma(0) + gamma(4))

# 正宇称二维基
evals, evecs = np.linalg.eigh(P_plus)
U = evecs[:, evals > 0.5]

# 四维旋量算符
S4 = 1j * gamma(k) @ gamma(5)

# 二维 Pauli 算符
sigma = U.conj().T @ S4 @ U

# 保持原逐元素收缩约定的二维张量
M2 = np.einsum("ia,yxij,jb->yxab", U, M4, U.conj())

# 正向和反向极化
P_forward = 0.5 * (np.eye(2) + sigma.T)
P_backward = 0.5 * (np.eye(2) - sigma.T)

forward = np.einsum("ab,yxba->yx", P_forward, M2)
backward = np.einsum("ab,yxba->yx", P_backward, M2)
polarized = forward - backward

# 与四维直接结果比较
direct = np.einsum("il,yxil->yx", P_plus @ S4, M4)
assert np.max(np.abs(direct - polarized)) < 1e-12
```

该流程可分别令 `k=1,2,3`，得到 x、y、z 三个方向的极化结果，同时保留正向和反向极化分量用于独立检查。
