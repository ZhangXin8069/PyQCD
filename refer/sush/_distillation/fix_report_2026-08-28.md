# LQCDDB 指定问题修复报告

> **后续更新**：本报告第 3.1 节所列 `conjugate_operator` 保留问题已在后续任务中修复。当前实现和测试结果见 [conjugate_operator_fix_report_2026-08-28.md](/public/home/sush/distillation/lqcddb_update/conjugate_operator_fix_report_2026-08-28.md)。本报告其余内容仍对应其记录的源码哈希。

## 1. 范围与原则

- 日期：2026-08-28
- 工作区：`/public/home/sush/distillation/lqcddb_update`
- 修改前 Python 源码聚合 SHA-256：`087a77e6da01868ec0c44614684d6031c1a7c61e7da714458e7e2faac7676f0c`
- 修改后 Python 源码聚合 SHA-256：`007a8d6f8ea2b14154ff959e85955fe23447eab5aeb9ff75362b7b1f7535aad4`

本轮只实现用户明确指定的接口、符号、统计默认值和文档同步，不改变关联函数、流算符、Fourier 相位、gamma 基、perambulator 方向或其它物理目标。

## 2. 已完成修改

### 2.1 Stout 输入形状采用 fail-fast 校验

文件：`src/lqcddb/base/smear_gauge.py`

`stout_smear_ndarray` 现在只接受完整规范场布局

$$
(N_d,N_t,N_z,N_y,N_x,N_c,N_c),
$$

并检查：

- 总维数必须为 7；
- 方向轴大小必须为 $N_d=4$；
- 颜色矩阵必须为 $N_c\times N_c=3\times3$；
- $N_t,N_z,N_y,N_x$ 均为正数；
- 非数组对象抛出 `TypeError`，形状错误抛出 `ValueError`。

没有修改正确的 `3-nu` 方向映射，也没有修改任何 $Q$ 分支或 $Q=0$ 行为。

### 2.2 Lambda→proton 使用左手 $V-A$ 流

文件：`src/lqcddb/test/contraction.lambda.proton.weak.mpi.py`

目标流固定为

$$
\Gamma_\mu^{V-A}=\gamma_\mu(1-\gamma_5)
=\gamma_\mu-\gamma_\mu\gamma_5.
$$

本轮开始时数值表达式已经由外部更新为减号；本轮保留该实现，修正仍写成 $V+A$ 的注释、README 和 `physics.md`，并增加 AST 与矩阵恒等式回归测试，防止符号再次回退。

### 2.3 Bootstrap 默认使用 $M=N_{\rm conf}$

文件：`src/lqcddb/analyse/analyse.py`

保留 `M=0` 作为默认 sentinel，但内部现在设置

$$
M=N_{\rm conf},
$$

不再使用 $N_{\rm conf}-5$。显式非零 `M` 的用户选择保持不变。

### 2.4 `ratio_3pt` 禁止输入形状广播

文件：`src/lqcddb/analyse/analyse.py`

三点函数、初态两点函数和末态两点函数必须具有完全相同的 shape。任一不一致立即抛出 `ValueError`；时间轴长度不一致也单独报错。轴索引现在同时检查正向和过小的负向越界。

此修改只限制输入契约，不改变原三点比值公式：

$$
R=\frac{C_3}{C_2^F(t_{\rm sep})}
\sqrt{\frac{C_2^I(t_{\rm sep}-\tau)C_2^F(\tau)C_2^F(t_{\rm sep})}
{C_2^F(t_{\rm sep}-\tau)C_2^I(\tau)C_2^I(t_{\rm sep})}}.
$$

### 2.5 修正 `Mom_cross_sigma` 手性

文件：`src/lqcddb/constant/sigma_matrix.py`

物理叉积采用右手 $(X,Y,Z)$ 坐标系。包内分量存储顺序 $(Z,Y,X)$ 是奇置换，因此存储基中的 Levi-Civita 张量应带额外负号。实现现在使用

```python
-levi_civita_tensor(3)
```

并保持输入/输出轴顺序 `[Z,Y,X]` 不变。

对 $\mathbf p=\hat x=[0,0,1]_{ZYX}$，当前返回

$$
(\mathbf p\times\boldsymbol\sigma)_{ZYX}
=(\sigma_y,-\sigma_z,0),
$$

与标准右手叉积一致。

### 2.6 momentum shell 返回前去重

文件：`src/lqcddb/base/base_functions.py`

`creat_mom_list` 现在把所有候选动量转换为 tuple set，去重后按字典序排序，再恢复为 `list[list]`。`fix_Q2` 和 `only_g0` 的物理筛选定义不变。

验证结果：

- `[-1,0,1]`, `fix_Q2=True`：由旧的 48 项恢复为 12 个唯一动量；
- `[1,1,0]`, `fix_Q2=False`：返回 27 个唯一立方动量；
- `only_g0=True` 的结果也不含重复项。

### 2.7 文档以当前代码为准同步

已同步：

- `README.md`；
- `physics.md`；
- 源码 docstring；
- 顶层及各子包 `.pyi` 类型存根；
- `changelog.md`。

主要同步项包括：Stout 七维布局、$V-A$ 顺序、Bootstrap 默认 $M$、`ratio_3pt` 同 shape 契约、右手叉积、动量去重、Jackknife 正确公式、GEVP 保留复厄米信息、`meff` 的 float64 仅为实现限制、P/V/G 缺省前缀为空字符串、MPI 查询函数实际导出范围、`conjugate_operator` 的 token-list 类型，以及 0.0.3 尚未发布的状态。

README 中的 wildcard imports 也改为具名导入，避免无意解析 optional MPI API。

## 3. 未修改事项

### 3.1 因存在物理影响而保留

以下路径没有纳入本轮修复，以免在缺少单独物理推导时改变目标量：

- 多强子 `ignore_dis` 的图论连通性问题；
- `conjugate_operator` 的系数、多重子次序、gamma alias 与参数化分量问题；
- 复 Jackknife/Bootstrap covariance 的 Hermitian convention；
- `vector_creator.check()` 的复 Gram 验证；
- 等价图中 $u,d\to\text{light}$ 的精确简并前提；
- conserved temporal vertex 的非零动量相位和完整 current normalization；
- Omega stochastic 权重与有效质量 mask/time-coordinate 问题。

这些问题可能改变关联函数、相位、拓扑、统计量或矩阵元，故本轮仅保留，不做未经授权的物理重定义。

### 3.2 本轮忽略的工程问题

MPI optional import、异常消息风格、旧 I/O helper、性能 adviser、生成的 `build/` 与 `egg-info` 工件等没有改变本轮目标物理量，未做扩展修改。Stout 的 $Q$ 问题按用户要求不处理。

GEVP complex dtype 已在本轮开始前由外部更新，本轮仅让文档与当前实现一致，没有把它计作本轮代码修复。

## 4. 测试与独立验证

### 4.1 新增回归测试

文件：`src/lqcddb/test/test_requested_fixes.py`

7 项全部通过，覆盖：

1. Stout 合法七维输入及四类错误输入；
2. Lambda→proton AST 减号与 $\gamma_\mu(1-\gamma_5)$ 矩阵恒等式；
3. 默认 Bootstrap 与显式 `M=Nconf` 在同一 RNG seed 下逐元素一致；
4. `ratio_3pt` 对不一致 shape/时间轴报错；
5. 相同 shape 的复三点信号不发生广播且保持不变；
6. `Mom_cross_sigma` 的 2x2 与 4x4 右手取向；
7. momentum shell 唯一性、排序和非负筛选。

### 4.2 现有测试

- `test.cache.py`：15 项通过；
- `test_equivalent_diagrams_strict.py`：13 项通过；
- `test_contraction_njnp_3pt_gevp.py`：沙箱外 MPI 环境 2 项通过；
- 39 个 `.py` 与 7 个 `.pyi`，共 46 个文件 AST 解析：0 错误；
- 修改后的公开 API 具名导入通过。

### 4.3 独立数值检查

- Stout `3-nu` staple 与逐格点显式周期坐标循环：残差 0；
- Stout 一步结果与显式 staple + `scipy.linalg.expm`：$3.38\times10^{-16}$；
- Stout 随机 SU(3) 场幺正性/行列式残差：$\lesssim1.8\times10^{-15}$；
- 全尺寸默认 Bootstrap error / 参考宽度：$1.0049$（12000 次重采样）；
- 已知不等初末态三点比值平台：最大残差 $4.44\times10^{-16}$；
- $\hat x\times\boldsymbol\sigma$：最大残差 0；
- momentum shell 大小：12 与 27，均无重复。

## 5. 修改文件

- `README.md`
- `physics.md`
- `changelog.md`
- `src/lqcddb/base/smear_gauge.py`
- `src/lqcddb/base/base_functions.py`
- `src/lqcddb/analyse/analyse.py`
- `src/lqcddb/constant/sigma_matrix.py`
- `src/lqcddb/contraction/autowick.py`
- `src/lqcddb/contraction/dynamic.py`
- `src/lqcddb/test/contraction.lambda.proton.weak.mpi.py`
- `src/lqcddb/test/test_requested_fixes.py`
- `src/lqcddb/__init__.pyi`
- `src/lqcddb/base/__init__.pyi`
- `src/lqcddb/analyse/__init__.pyi`
- `src/lqcddb/constant/__init__.pyi`
- `src/lqcddb/contraction/__init__.pyi`
- `fix_report_2026-08-28.md`
