# `conjugate_operator` 物理与实现修复报告

## 1. 范围

- 日期：2026-08-28
- 修改前 Python 源码聚合 SHA-256：`007a8d6f8ea2b14154ff959e85955fe23447eab5aeb9ff75362b7b1f7535aad4`
- 修改后 Python 源码聚合 SHA-256：`6355781464145eb0cf3c98bc9f34626184b0378e41cb28c868a115c28d4aec90`
- 主要实现：`src/lqcddb/contraction/baroperator.py`

目标仅是正确实现既有 source/bar-operator 约定，不改变 gamma 基、强子插值场、Wick 配对、Fourier 相位或关联函数定义。

## 2. 物理定义

对多强子项

$$
\mathcal O=c\,H_1H_2\cdots H_n,
$$

共轭乘积必须反序：

$$
\mathcal O^\dagger
=c^*H_n^\dagger\cdots H_2^\dagger H_1^\dagger.
$$

实现保留这一精确顺序，不把共轭后的强子重新排回原顺序。因此不会额外加入未经定义的粒子间交换符号；所有费米场次序由返回 token 本身和后续 Wick 展开处理。

介子 gamma 链采用

$$
\Gamma^H=\gamma_4\Gamma^\dagger\gamma_4.
$$

对规范重子 token

```text
| q3 q1 D... q2 |
```

source/bar-operator 的 diquark 矩阵是

$$
(D^H)^T.
$$

代码现在显式执行 gamma 链的 $H$ 再执行 $T$，而不是根据字符串名字猜测 diquark transpose sign。

## 3. 已修复问题

### 3.1 多粒子因子顺序

旧实现逐个共轭但保持 $H_1,H_2,\ldots$ 原顺序。新实现反向遍历已解析的 `|...|` factors，严格返回 $H_n^\dagger,\ldots,H_1^\dagger$。

例如输入

```python
[2+3j, *proton, *pion]
```

返回系数 `2-3j`，token 顺序为 `pion†` 后接 `proton†`。

### 3.2 数值系数

所有 `int/float/complex` token 被视为一个全局乘法系数：

- 先相乘；
- 再取复共轭；
- 最后乘各强子内部的 gamma/费米符号。

系数不再被 `split_hadrons` 丢弃。

### 3.3 重子 diquark transpose 与 gamma alias

新变换直接计算 $(D^H)^T$。因此数值完全相同的

$$
\gamma_7=C\gamma_5
$$

自动得到相同 source coefficient `-1.0`，对应 proton 两点 Wick signs 均为

```text
[-1,+1].
```

不再依赖 `C * ...` 字符串是否恰好出现在特殊列表中。

### 3.4 baryon/antibaryon 双向规范 grammar

支持并区分：

```text
baryon:     | q3   q1   D... q2   |
antibaryon: | q1^d D... q2^d q3^d |
```

因此单重子、多重子和混合 meson-baryon 项均满足

$$
(\mathcal O^\dagger)^\dagger=\mathcal O
$$

（返回列表含显式全局系数）。

### 3.5 解析与 fail-closed

现在会拒绝：

- 未闭合、空或 token 位于 `|...|` 外的算符；
- 未知 quark/gamma token；
- quark/antiquark 数量不符合 meson/baryon grammar 的因子；
- 语义不明确的单个 `|...|` 四夸克及更高多夸克结构；
- `gamma_mu`、`sigma_mu_nu` 及对应 $C$ 结构等分量相关 gamma。

最后一类不能用一个全局 $\pm1$ 系数表达所有 Lorentz 分量，必须先展开成 `gamma_1` 至 `gamma_4` 等显式分量。报错比继续给出错误相对相位更符合物理安全要求。

多粒子算符仍受支持，但必须表示成多个各自配对的 `|...|` hadron factors。

## 4. 不变项与使用影响

- pion source 仍为 `[-1.0,'|','d^d','gamma_5','u','|']`；
- `C * gamma_5` proton source 仍使用系数 `-1.0`；
- 基础 pion/proton Wick signs 未改变；
- `wick_contraction`、dynamic contraction 和 tensor einsum 未修改；
- gamma 矩阵数值、Euclidean 约定与 perambulator 方向未修改。

行为上有意改变的是多强子 source 因子顺序。依赖旧错误顺序的 source VDV/VVV 注册或手写 contraction plan 需要按新返回 token 顺序重新核对；这不是新的物理约定，而是恢复乘积共轭的标准反序规则。

## 5. 测试

新增：`src/lqcddb/test/test_conjugate_operator.py`

9 项通过：

1. 单 pion/proton 约定；
2. `gamma_7` 与 `C * gamma_5` alias 相位及 Wick signs；
3. 单项/多项复系数共轭；
4. proton-pion 因子反序；
5. 两个不同重子的反序且不做额外 canonical reordering；
6. 单粒子和三粒子乘积的 involution；
7. 12 种显式 baryon gamma 结构的 involution；
8. 参数化 gamma fail-closed；
9. malformed/generic grammar fail-closed。

完整回归：

- `test_conjugate_operator.py`：9 项通过；
- `test_requested_fixes.py`：7 项通过；
- `test.cache.py`：15 项通过；
- `test_equivalent_diagrams_strict.py`：13 项通过；
- `test_contraction_njnp_3pt_gevp.py`：沙箱外 MPI 环境 2 项通过；
- 40 个 `.py` 与 7 个 `.pyi`，共 47 个文件 AST 解析：0 错误；
- 顶层 `from lqcddb import conjugate_operator` 与多粒子 involution 检查通过。

## 6. 修改文件

- `src/lqcddb/contraction/baroperator.py`
- `src/lqcddb/test/test_conjugate_operator.py`
- `src/lqcddb/contraction/__init__.pyi`
- `src/lqcddb/__init__.pyi`
- `README.md`
- `changelog.md`
- `review_report_2026-08-27.md`（追加后续修复状态）
- `fix_report_2026-08-28.md`（追加后续修复状态）
- `conjugate_operator_fix_report_2026-08-28.md`
