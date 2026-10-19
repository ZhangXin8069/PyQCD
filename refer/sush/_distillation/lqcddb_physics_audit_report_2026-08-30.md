# LQCDDB 全函数物理与数值一致性审查报告

> **后续修复（2026-08-30）**：本报告中 `analyse.py` 的统计/谱学问题以及 lazy API/.pyi 不一致已经修复。完整定义、兼容性变化与 60 项回归见 [analyse_api_fix_report_2026-08-30.md](analyse_api_fix_report_2026-08-30.md)。Stout、Wick/dynamic、顶点、eigenvector、性能 adviser 和两个 MPI 布局问题仍未修复。
>
> **科学澄清**：QCD 不禁止 $q q \bar q \bar q$ 色 singlet 插值场；双介子/多强子算符也是有限体积谱学的合法且必要输入。原 H16 应理解为“当前单 factor 多夸克 grammar 未定义时必须 fail closed”的接口限制，不是“四夸克物理态被标准模型禁止”。

## 1. 审查信息

- 日期：2026-08-30
- 工作区：`/public/home/sush/distillation/lqcddb_update`
- 审查范围：活动代码 `src/lqcddb/**/*.py`、公开类型存根 `src/lqcddb/**/*.pyi`
- 排除：`build/`、根目录文档与历史报告；`src/lqcddb/test` 仅作为验证资产和生产用例，不作为函数包 API 的物理定义来源
- 活动实现规模：24 个非测试 Python 文件，239 个函数定义（含方法和嵌套辅助函数），16 个类定义
- 修改前源码/存根聚合 SHA-256：`b8dc2d931d2838ca720881295fbf9d3b9b1000bc3912f019eebf4044ec1a0873`
- 授权重命名后聚合 SHA-256：`791067b67d2c2c3a4de72bb77a0bc7fcc262237b8b37dd775b5ed206621731a5`

审查遵循

$$
\text{物理定义}\longrightarrow
\text{约定与归一化}\longrightarrow
\text{张量公式}\longrightarrow
\text{数值算法}\longrightarrow
\text{实现}.
$$

## 2. 总体结论

结论不是“所有函数都符合对应物理”。当前代码中有一批核心路径在明确输入域内通过独立验证，但也存在会静默改变关联函数、拓扑、相位、统计协方差或规范场的确认缺陷。因此，当前源码不应在没有下述安全门的情况下直接作为通用生产分析链。

通过独立验证的主要核心包括：

- DR 基 Euclidean gamma 代数、$\gamma_5$、电荷共轭恒等式与 Pauli 代数；
- `Mom_cross_sigma` 在 `(Z,Y,X)` 存储中的右手取向；
- `conjugate_operator` 对介子、规范重子、多强子反序和复系数的处理；
- pion/proton 基本 Wick 符号以及严格 VVV/gamma 相位等价测试；
- `seq_peram` 的带轴序 $\gamma_5\tau^\dagger\gamma_5$ 恒等式；
- VDV 的零位移与单位规范链接空间位移、VVV 的反对称性、源汇 Hermitian 共轭；
- 标准 delete-one Jackknife 样本、默认 $M=N_{\rm conf}$ Bootstrap、log/cosh 有效质量的有效时间点；
- 复 Hermitian 谱表示上的 GEVP 本征值；
- 2-rank MPI 的 Gather、Allgather、TGather、Scatter、TScatter、Bcast 和 Send；
- 授权后的 `phase_exp_VdV`、`phase_exp_VVV` Fourier 相位一致性。

最需要立即规避的路径是：Stout smearing、默认 `ignore_dis=True`、未经精确同位旋条件证明的等价图、复杂 Wick token/gamma 链、大费米子数时的输出指标、复协方差、异位置时间轴的三点比值、守恒时间顶点的非零动量，以及性能 adviser 的定量输出。

## 3. 本次使用的约定

- Euclidean gamma：$\{\gamma_\mu,\gamma_\nu\}=2\delta_{\mu\nu}$，$\gamma_5=\gamma_1\gamma_2\gamma_3\gamma_4$；当前矩阵满足残差 0。
- 电荷共轭：$C=\gamma_4\gamma_2$，满足 $C\gamma_\mu C^{-1}=-\gamma_\mu^T$，残差 0。
- 空间存储：`(z,y,x,color)`；规范场完整布局为 `(Nd,Nt,Nz,Ny,Nx,Nc,Nc)`，方向为 `(X,Y,Z,T)`。
- 相位：当前顶点生成器使用 $e^{-i\mathbf p\cdot\mathbf x}$。
- perambulator 基本轴序：`(spin_sink,spin_source,ev_sink,ev_source)`；反向对象按带轴序的

$$
\tau(t_0,t)=\left(\gamma_5\otimes I_{N_v}\right)
\tau(t,t_0)^\dagger
\left(\gamma_5\otimes I_{N_v}\right)
$$

处理。
- source 算符必须是完整 $\mathcal O^\dagger$，包括 `conjugate_operator` 返回的首项系数。
- 等价图默认关闭；disconnected 默认按收缩图连通分量判定，而不是按时间标签猜测。

## 4. 验证证据

### 4.1 现有回归

重命名前和重命名后均通过：

- `test_conjugate_operator.py`：9/9；
- `test_requested_fixes.py`：7/7；
- `test_equivalent_diagrams_strict.py`：13/13；
- `test.cache.py`：15/15（测试内硬编码旧路径，审查时只在管道输入中替换为当前路径，没有修改该测试）；
- `test_contraction_njnp_3pt_gevp.py`：2/2，在允许 MPI/UCX 本地通信的环境运行；
- 合计：46/46。

### 4.2 独立不变量与反例

- 主独立探针：79 项，其中 42 项满足不变量、36 项复现缺陷、1 项诊断信息；
- 补充物理/I/O/压缩探针：16 项，其中 12 项满足、4 项复现缺陷；
- 单 rank MPI/dynamic：9 项满足、6 项复现缺陷；
- 2-rank MPI：8 项满足、2 项复现缺陷；
- 47 个 `.py/.pyi` 全部通过 `ast.parse`；
- GPU：CuPy 13.0.0 可导入，但 CUDA 驱动不足，GPU 数值路径未执行。

这些“失败”是期望物理/数值不变量的失败，不是简单把已有测试重新跑一遍。

## 5. 确认的关键问题

### C1. Stout 的合法固定点产生 NaN（严重）

位置：`src/lqcddb/base/smear_gauge.py:72-121`。

单位 SU(3) 规范场和 `rho=0` 都给出 $Q=0$。代码继续计算 $c_0/c_{0,\max}$ 和 $(9u^2-w^2)^{-1}$，没有实现 $Q\to0$ 极限。`nstep=1` 时两条合法路径均产生非有限结果。

必要不变量

$$
U_\mu(x)=I\Longrightarrow U'_\mu(x)=I,
\qquad
\rho=0\Longrightarrow U'=U
$$

均不成立。

### C2. Stout 空间方向映射错误（严重）

位置：`src/lqcddb/base/smear_gauge.py:62-70`。

在 `U.shape=(mu,t,z,y,x,c,c)` 中，$X,Y,Z$ 应对应 roll 轴 $4,3,2$，即 `4-nu`；当前使用 `3-nu`，对应 $3,2,1$，把 $Z$ staple 混入时间方向。随机 SU(3) 场与独立正确坐标 staple 加 `scipy.linalg.expm` 的一步结果最大残差为

$$
2.015\times10^{-1}.
$$

错误轴下的结果仍可保持 $UU^\dagger=I$ 和 $\det U=1$，所以仅检查 SU(3) 性质不能发现此问题。

### C3. Wick 对未知 token 和 `sigma_mu_nu` 静默丢弃（严重）

位置：`src/lqcddb/contraction/autowick.py:330-340`。

gamma 识别依赖字符串包含 `gamma`；`sigma_mu_nu`、`C`、未知 token 会只打印 warning 后消失。这样生成的 einsum 与原算符不是同一个物理量。必须 fail closed。

### C4. 多 gamma 链被错误拆成一端口张量（严重）

位置：`src/lqcddb/contraction/autowick.py:395-405`。

对 `qbar gamma_5 gamma_1 q`，每个 gamma 只找到一侧相邻夸克，实际得到四个一字符 gamma operand；示例输出包含 `...,a,c,e,m,...`，而不是两个完整两端口矩阵链。`baroperator` 支持 gamma 链，不代表 Wick 层也正确支持。

### C5. 大费米子数时大写收缩指标被误当自由指标（严重）

位置：`src/lqcddb/contraction/autowick.py:101-134,232-233`。

`keep_unique_letters` 对小写字母只保留出现一次者，却无条件保留所有大写字母；同时 `contraction_index` 在第 14 个场后开始分配大写字母。四个 flavor-diagonal 介子两点函数的 16 个场产生 RHS `ABECFD`，其中每个字母在 LHS 出现两次，本应被求和。这会把标量关联函数变成带伪自由轴的张量。

### C6. 默认 disconnected 启发式会删除图论连通图（严重）

位置：`src/lqcddb/contraction/dynamic.py:925-935`，默认参数见 `:672-677`。

当前判据是“任一 peram 满足 $t_q=t_{\bar q}$”。对两个 flavor-diagonal 介子的 24 个 Wick 图，该判据标记 20 个。明确反例图 9 的强子因子边为

$$
(0,1),(1,2),(2,3),(3,0),
$$

形成单一连通环，但含 sink-sink 和 source-source peram，因此被默认删除。生产应先用 `ignore_dis=False`，再按图连通分量分类。

### C7. 等价图无条件使用 $u,d\to\text{light}$（严重）

位置：`src/lqcddb/contraction/autowick.py:481-503`。

分别构造满足各自 $\gamma_5$ 厄米关系、但逐组态数值不同的 $u,d$ perambulator。带电 pion 正负通道被归为同一等价类；分组和未分组结果残差为

$$
2.107\times10^{1}.
$$

等质量作用量或总体 isospin limit 不自动推出注册数组逐组态相同。除非使用同一个 `light` 数组并明确采用所需精确假设，否则必须 `use_equivalence=False`。

## 6. 其它高优先级问题

| ID | 位置 | 结论与复现 |
| --- | --- | --- |
| H1 | `analyse/analyse.py:710-734` | `ratio_3pt` 在 `tau_axes` 与 `t_sink_axes` 位置不同时没有移动两点轴。即使三数组 shape 完全相同，使用不等初末态仍得到 $1.279\times10^{-1}$ 残差。 |
| H2 | `analyse/analyse.py:393,514` | 复 Jackknife/Bootstrap covariance 使用 $r_ir_j$ 而非 $r_ir_j^*$；厄米性残差分别为 $1.44\times10^{-1}$、$1.54\times10^{-1}$。 |
| H3 | `analyse/analyse.py:375-382` | `Nconf_axes!=0` 时 covariance 直接用降维后的均值广播；合法 shape `(4,7)`、组态轴 1 报广播错误。 |
| H4 | `analyse/analyse.py:402-519` | 显式 m-out-of-n Bootstrap 未把 replicate 宽度缩放回 $N_{\rm conf}$ 估计量；$N=30,M=10$ 时误差/目标为 1.741，接近 $\sqrt{3}$。 |
| H5 | `analyse/analyse.py:915-939` | `dis_connect` 没有完整轴契约；同 shape `(Nconf,Nt,Nt)` 的 C2/bubble 输入试图把 80 元素 reshape 成 `(5,1,4)`。 |
| H6 | `eigvectors/vertex.py:503-521` | conserved/T 分支完全不使用 `phase_exp`，改变相位后残差 0，输出 shape 只有 `(2,Nev,Nev)`，没有动量轴。 |
| H7 | `contraction/dynamic.py:1275-1279` | 绝对阈值 `abs(total_coeff)<1e-12` 把合法非零小归一化清零；$10^{-13}$ 算符得到范数 0。 |
| H8 | `contraction/dynamic.py:1341-1385` | `validate_plan` 仅查名字，不查 shape/einsum；故意注册不匹配的 spin extent 仍返回空问题列表。 |
| H9 | `eigvectors/vector.py:53` | `vector_creator.check` 在 Gram 检查前 `.real`；$v_1=(1,0),v_2=(i,0)$ 完全平行但返回 `True`。 |
| H10 | `eigvectors/vector.py:376-385` | `Z_5` 候选含模长 2 的值，不是单位根；字面参数 `Z_N` 会执行 `int('N')` 而失败。 |
| H11 | `eigvectors/vertex.py:320-329` | `create_omega_accelerate(normal=True)` 先逐行归一化再对称化，破坏行和；9 行示例得到 `3.375, 5.625, 9` 三组行和。 |
| H12 | `contraction/contractadviser.py:201-298,340-424` | ellipsis batch 轴完全未计入；多输入 `opt_einsum` FLOP 再乘一次 `flops_per_madd`，实数案例正好多算 2 倍。 |
| H13 | `contraction/contractadviser.py:81-151` | V100/A100/A800 的 `compute_efficiency` 为 2–2.5，超过物理可达比例 1；预设峰值与通用 complex einsum kernel 也未区分。定量时间不可采信。 |
| H14 | `analyse/analyse.py:86-161` | `loop_tsrc` 对 Antiperiodic 统一乘一个负号，未依赖绕边界的费米线数；这不是通用 meson/baryon 规则。 |
| H15 | `contraction/dynamic.py:18,29-37` | dynamic 在模块加载时解析 optional MPI，并对普通注册错误调用 `MPI.Abort`；库级输入错误可终止整个作业。 |
| H16（重分类） | `contraction/autowick.py:93-99,422-435` | QCD 允许四夸克/多强子插值场，但当前单 factor grammar 未定义。标准 `conjugate_operator` 流程已拒绝该 grammar；若直接调用 `wick_contraction`，仍应 fail closed，而不能生成缺顶点网络。本项是 unsupported-input 防御问题，不是被物理禁止的输入。 |
| H17 | `contraction/dynamic.py:863-881` | `wick_contraction` 有 wildcard 时返回 list，`run_wick_analysis` 随后以 dict 下标访问，复现 `TypeError`。 |
| H18 | `contraction/contractadviser.py:594-715` | “增大指标以提高 GPU 利用率”的建议会改变 $N_v$、动量或物理问题，而不是纯性能优化；不可按建议直接修改科学计算。 |

## 7. 中低优先级与接口问题

| 模块/函数 | 问题 |
| --- | --- |
| `gamma_index` | 固定分配 4 个非零元，`gamma(16/17)` 各有更多非零元并触发 `IndexError`。 |
| `gamma` / `sigma` | 非法 `gamma` 调用 `sys.exit(-3)`；非法 `sigma` 静默返回 `None`。库接口均未 fail cleanly。 |
| `PFF_Mom_to_gamma_new` | `allow_t=True` 的零动量和混合支撑动量列表均触发 reshape 错误；内部 reshape 依赖最后一个 momentum 的 `k`。 |
| `cached_contract` | ellipsis 按左侧位置对齐；NumPy 合法的 `(2,3,4,5)` 与 `(3,5,6)` 右对齐广播被拒绝。普通及复杂非 ellipsis 回归通过。 |
| `Jackknife` | $N_{\rm conf}=1$ 产生 warning/NaN 而不拒绝。 |
| `Bootstrap` | 实输入固定提升为 `complex128`；第 0 个原均值被混入 replicate 分布，有限 $N$ 下改变中心与宽度。 |
| `meff` | 有效 log/cosh 点正确，但结尾无效点填 0；拒绝 `float32` 和复数据，属于实现限制而非物理要求。 |
| `ratio_3pt` | `maximum(...,0)` 会把负平方根参数静默投影为 0；应由投影/相位约定决定或保留 mask/NaN。 |
| `_fold_link` | 奇数长度结果正确；偶数长度没有显式检查并以越界异常结束。 |
| `solve_gevp` | 已保留复 Hermitian 信息并通过谱模型，但各 $t$ 独立排序；近简并/能级交叉没有重叠跟踪。 |
| `vector_creator.normal/orthnormal/creat_noise` | 零范数或线性相关输入产生非有限量；没有秩和最大可生成向量数检查。 |
| `compress_matrix_V1..V4` | V1/V2/V3/V4 的若干合法小规模路径保持正交；但大量校验使用 `raise print(...)`，实际产生 `TypeError`，V3/V4 还把 bool `check` 结果与字符串比较。 |
| `vertex_creator.src_sink_MPI_tran` | 使用从未初始化的 `self.backend_name`，正常 forward 路径复现 `AttributeError`。 |
| `vertex_creator.normal` | 对 backend 数组调用 `np.sqrt`，CuPy 路径未验证。 |
| `VdV_sink_t_link` | 无链接、单位链接 $\Delta x=-1,0,+1$ 通过；假设立方空间。`link_dir='all'` 实际除以 3，是方向平均而非注释所称求和。 |
| `sink2src` | 数值 VDV/VVV 变换正确，但类内未标 `@staticmethod`，实例调用方式不一致；类型存根漏掉该方法。 |
| `get_mpi_tlist` | 2-rank 时标量 $t=-1$ 返回 `(1,2)`，列表 `[-1]` 返回 `(0,2)`；非法 mode 返回 `None`。 |
| `get_mpi_data` | 2-rank 七种基本传输通过；单 rank Gather/Allgather 不保留 rank 轴；Transport 返回额外 singleton rank 轴。 |
| MPI tags/GPU | TScatter tag 未按 `MPI_TAG_UB` 查询；非 root 的 `was_gpu` 来自本地 `data=None`，可能导致各 rank 返回类型不一致。未在 GPU 上实测。 |
| `_composition` | 对 `d=1` 的索引更新不会推进循环，静态分析显示可无限循环；常用 4D 路径通过。 |
| `readin_eigvecs/readin_peram` | 小型二进制轴序 round trip 通过；使用 native endian、文件大小推断和非 context-manager 文件句柄，缺少完整元数据/尺寸验证。 |
| `write_data_ascii/safe_save/check_files_existence/savetxt` | 审查路径通过；`check_write('basename')` 对空目录调用 `os.mkdir('')` 失败。`safe_save` 的站点专用自动回退可能产生意外外部写入。 |
| lazy API / `.pyi` | 顶层 53 个导出名与顶层 stub 集合一致，但签名存在 5 类不一致：`conjugate_operator` 关键字名、`run_wick_analysis` 缺 Projection/Oindex、顶层 VdV 旧签名、`sink2src` 缺失、`check_dir_path` 关键字名。`lqcddb.io` stub 还宣称子包导出 `safe_save`，运行时不导出；`lqcddb.contraction` stub 多声明 6 个运行时不可解析类。 |
| plotting helpers | 不参与数值物理；静态检查通过。图背景区域构造中 source/sink 列表命名有交换，影响图示而不改变 contraction。 |

## 8. 全函数审查矩阵

状态：`通过` 表示在所列输入域内通过独立不变量；`条件` 表示公式可用但必须固定约定/输入域；`缺陷` 表示已有可复现反例；`工程` 表示不直接改变物理值但影响可靠使用；`未实测` 表示受环境限制。

| 文件 | 函数/方法（含其嵌套辅助函数） | 状态 | 结论 |
| --- | --- | --- | --- |
| `constant/gamma_matrix.py` | `gamma` | 条件 | 0–15 的 DR 基代数正确，16/17 是带投影结构；非法索引会退出进程。 |
| 同上 | `gamma_index` | 缺陷 | 只支持恰有 4 个非零元，16/17 崩溃。 |
| 同上 | `tran_indx_to_gamma` | 通过 | shape 与逐项矩阵一致。 |
| 同上 | `PFF_Mom_to_gamma_new` | 缺陷 | 基本零动量路径可用，`allow_t` 零动量及混合支撑失败。 |
| `constant/sigma_matrix.py` | `sigma` | 条件 | Pauli 矩阵正确；非法索引返回 `None`。 |
| 同上 | `Mom_times_sigma` | 通过 | `(Z,Y,X)` 点积正确；输入长度缺少检查。 |
| 同上 | `Mom_cross_sigma` | 通过 | 2x2/4x4 的右手叉积正确。 |
| `base/backend.py` | `get_backend`, `set_backend` | 工程 | NumPy 正常；非法 backend 用 `assert`，CuPy 设备路径未实测。 |
| `base/base_functions.py` | `levi_civita_tensor` | 通过 | 排列奇偶正确。 |
| 同上 | `creat_mom_list`、内嵌 `add_negative_signs` | 通过/条件 | shell 唯一、排序和 $Q^2$ 正确；输入结构缺少严格校验。 |
| 同上 | `ArraySlicer.__init__/get_slices/slice/assign/get_slice_shape/get_info` | 通过/工程 | 常用 Cartesian slice/assign 通过；空组、过小负轴和 reshape 错误缺少清晰契约。 |
| 同上 | `_validate_einsum_shapes` | 缺陷 | ellipsis 对齐不符合 NumPy。 |
| 同上 | `cached_contract` | 条件 | 15 项现有网络及直接参考通过；合法 ellipsis 反例失败。 |
| 同上 | `clear_cache`, `get_cache_keys` | 通过 | 缓存清理/键检查正确。 |
| `base/cg_coeff.py` | `_to_sympy`, `SU2combine`, `SU2decompose` | 通过/条件 | $1/2\otimes1/2$ 归一与 singlet 分解正确；多体 `intermediate_Js` 长度缺少完整验证。 |
| `base/smear_gauge.py` | `stout_smear_ndarray` | 缺陷 | C1、C2；当前不可用于生产。 |
| `base/mpi_init.py` | `_MPILogger` 全部方法 | 工程 | rank 过滤有效；warning/error 人工附带 traceback，`critical` 抛异常。 |
| 同上 | `getRankFromCoord`, `getCoordFromRank` 及内嵌 `_process` | 通过/工程 | 两种 map 的合法坐标互逆；不查维数/范围。 |
| 同上 | `_composition`, `_factorization`, `_partition`, `getDefaultGrid` | 条件 | 常用 4D 分解通过；`_composition(d=1)` 风险。 |
| 同上 | `initGrid`, `initDevice` 及三个内嵌 CUDA helper, `mpinit` | 通过/未实测 | 单 rank NumPy 初始化通过；真实 CUDA/HIP/MPS 未实测，MPI backend 与全局 array backend 分离。 |
| 同上 | `isGridInitialized` 至 `isNumpyBackend` 的查询/设置函数 | 通过/工程 | 正常状态读取通过；`setGridMap` 不验证输入。 |
| 同上 | `get_mpi_tlist` | 缺陷 | 负时间标量/列表不一致。 |
| 同上 | `_next_send_tags`, `_gpu_to_numpy`, `_numpy_to_gpu` | 工程/未实测 | tag 上界与跨 rank GPU 类型存在风险。 |
| 同上 | `get_mpi_data` | 条件 | 2-rank 七种模式通过；单 rank shape 和 Transport 失败。 |
| `analyse/analyse.py` | `Mom2GeV` | 条件 | continuum $2\pi n/(La)$ 定义正确；scalar `Mom` 被解释为 $|n|^2$，需明确。 |
| 同上 | `loop_tsrc` | 条件/缺陷 | periodic 相对时间求和正确；边界符号不通用，非法类型静默零输出。 |
| 同上 | `mean_over_array_of_list`, `sum_over_array_of_list` | 通过/工程 | 与独立循环一致；空 group/重复 axis 缺少校验。 |
| 同上 | `Jackknife` | 条件/缺陷 | delete-one 样本正确；任意轴 covariance、复 covariance、$N=1$ 失败。 |
| 同上 | `Bootstrap` | 条件/缺陷 | 默认 $M=N$ 与 seed 重现正确；复 covariance、m-out-of-n、dtype 有问题。 |
| 同上 | `meff` | 条件 | log/cosh/GEVP 有效点正确；端点、dtype 和非法 estimator 处理不可靠。 |
| 同上 | `_validate_axes`, `_fold_link` | 缺陷/工程 | exact shape 没解决轴语义；偶数 link 未明确拒绝。 |
| 同上 | `ratio_3pt` | 缺陷 | 复信号保留；异位置轴与负平方根处理错误。 |
| 同上 | `solve_gevp` | 通过/条件 | 复谱模型通过；需 HPD conditioning 与 state tracking。 |
| 同上 | `dis_connect` | 缺陷 | 缺少可执行的通用 tensor contract。 |
| `contraction/baroperator.py` | `_H`, `_T`, `_C`, `_sign`, `_sigma` | 通过/条件 | 显式矩阵恒等式正确；参数化结构不能用单一代表分量外推。 |
| 同上 | `Gamma.H/T/C`, `transpose_gamma`, `charge_conjugation_gamma`, `diquark_symmetry` | 条件 | 显式结构可用；参数化分量必须先展开。 |
| 同上 | token/grammar helpers `is_*`, `dagger_quark`, `split_hadrons`, `classify_structure` | 通过 | fail-closed grammar 正确。 |
| 同上 | gamma-chain helpers | 通过/条件 | bar 层矩阵反序正确；Wick 层不支持多 gamma 链。 |
| 同上 | `conjugate_meson/baryon/generic/single_hadron/operator` | 通过 | 9 项回归与 involution 通过；完整返回系数必须保留。 |
| `contraction/seqperam.py` | `seq_peram` | 通过 | 与 combined spin-distillation 矩阵计算残差 0，且为 involution。 |
| `contraction/autowick.py` | `wick_contraction` 与其 7 个构建/符号 helper | 条件/缺陷 | 基本 pion/proton 符号正确；C3–C5、H16。 |
| 同上 | `_normalise_peram_flavours` 及内嵌 `_normalise` | 缺陷 | 无条件合并 $u,d$。 |
| 同上 | `_per_region_v_keys`, `_equivalence_port_role`, `_build_equivalence_network`, `_permutation_parity`, `_allowed_tensor_port_permutations`, `_equivalence_networks_are_isomorphic` 及搜索 helper | 通过/条件 | 精确 tensor/time/output/VVV/gamma 网络测试通过；依赖 flavor 前提。 |
| 同上 | `identify_equivalent_diagrams` | 条件/缺陷 | 严格网络相位测试通过；C7 阻止通用启用。 |
| 同上 | `plot_figure_wick` 及 6 个绘图 helper | 工程 | 不参与数值；可视布局有 source/sink 命名交换。 |
| `contraction/dynamic.py` | `_log`, `_abort`, `_check_not_none`, `_check_min_ndim` | 工程 | 检查本身简单；`_abort` 对库输入错误过于破坏性。 |
| 同上 | `PeramRegistry` 全部方法 | 条件 | lookup 正常；只查 `ndim>=4`，不证明轴/shape/flavor 正确。 |
| 同上 | `VRegistry` 全部方法 | 条件 | lookup 正常；无 shape/identity 验证。 |
| 同上 | `GammaRegistry` 全部方法 | 通过/条件 | concrete transpose sign fail-closed；任意 shape 注册仍需外部验证。 |
| 同上 | `_per_region_v_names`, `_build_projection_einsum` | 条件 | 编号一致；projection 必须先定义是 trace 还是双侧基变换，并检查标签冲突。 |
| 同上 | cache/hash/gamma-sign helpers | 条件 | gamma signature 纳入缓存；返回可变 plan 引用，第一组 gamma 数用于所有组。 |
| 同上 | `_format_cost`, `_analyze_contraction_path` | 工程 | 诊断用；complex128 字节数硬编码。 |
| 同上 | `run_wick_analysis` 及内嵌 `_get_output_labels` | 缺陷 | disconnected、wildcard、错误仅打印等问题。 |
| 同上 | `calculate_contraction` | 条件/缺陷 | pion 对直接 einsum 残差 0；小系数被清零。 |
| 同上 | `validate_plan` | 缺陷 | 仅名字检查。 |
| 同上 | `dynamic_contraction.__init__/calculate/calculate_all/__len__/__getitem__` | 条件 | N–J–Nπ 生产结构回归通过；默认过滤、MPI Abort、跨图 shape 求和仍需规避。 |
| `contraction/contractadviser.py` | `HardwareSpec` 属性方法 | 缺陷 | 公式形式正确，预设效率/峰值不可靠。 |
| 同上 | `ParsedContraction` 属性、`parse_subscript` | 缺陷 | 显式简单表达式可解析；ellipsis、scalar trace、未知 RHS 标签失败。 |
| 同上 | cost/AI helpers 与 `estimate_cost` | 缺陷 | 多输入 FLOP 双计、I/O/cache penalty 为未校准启发式。 |
| 同上 | suggestion dataclasses 与生成 helper | 条件/缺陷 | 数据结构正常；read reduction 指标和 upsizing 物理含义不可靠。 |
| 同上 | `analyze_bandwidth`, `quick_check`, 格式化/summary helper, `printGPUinfo` | 工程/缺陷 | 可输出报告但不得作为硬件验证结果。 |
| `eigvectors/vector.py` | `__init__`, `inner_product` | 通过/条件 | 复内积正确；实例捕获 backend，之后全局切换不会同步。 |
| 同上 | `check`, `normal`, `orthnormal`, `creat_noise` | 缺陷 | 复 Gram、零范数、秩检查问题。 |
| 同上 | `compress_matrix_V1/V2/V3/V4` 及内嵌随机 helper | 条件/缺陷 | 正常小型正交路径通过；输入校验和 `Z_N` 失败。 |
| `eigvectors/vertex.py` | `__init__`, `check`, `normal`, `src_sink_MPI_tran`, `perm_comb` | 混合 | complex Gram check 正确；其余见接口问题。 |
| 同上 | `create_omega_accelerate` | 缺陷/未证明 | normal 路径失败，完整 stochastic 权重公式未得到独立物理证明。 |
| 同上 | `phase_exp_VdV`, `phase_exp_VVV` | 通过 | $e^{-ipx}$、$p\leftrightarrow-p$ 共轭及彼此相位一致。 |
| 同上 | `VdV_sink_t_link` | 条件/缺陷 | 无链接/空间链接通过；时间守恒流动量失败。 |
| 同上 | `Mom_VdV_sink_t`, `Mom_VVV_sink_t`, `sink2src` | 通过/工程 | 数值恒等式通过；静态方法/存根问题。 |
| `io/write_data.py` | `readin_eigvecs`, `readin_peram` | 条件 | 小型轴序 round trip 通过；元数据/endian/资源管理不足。 |
| 同上 | `write_data_ascii`, `savetxt`, `check_dir_path`, `check_files_existence` 及内嵌 helper | 通过/工程 | 临时目录 round trip 通过；旧 `check_write` basename 失败。 |
| 同上 | `check_write` | 缺陷 | basename-only 路径失败。 |
| 同上 | `safe_save` 及内嵌 `_try_save` | 条件 | NumPy round trip 通过；自动站点回退需显式授权。 |
| 各 `__init__.py` | 14 个 `__getattr__/__dir__` | 工程 | 懒加载主体正常；submodule stub/export 不一致，wildcard 会解析 optional MPI。 |

## 9. 推荐的 contraction 构建流程

### 9.1 从物理算符到 source

对

$$
\mathcal O=cH_1H_2\cdots H_n,
$$

必须使用

$$
\mathcal O^\dagger=c^*H_n^\dagger\cdots H_1^\dagger
$$

生成 source，并保留返回系数。先只使用每个 `|...|` 中一个明确 meson 或 canonical baryon 的 grammar；参数化 gamma 先展开为具体分量。

### 9.2 逐图审查，而不是只收集 einsum

每图记录：系数、费米置换号、每条 peram 的 flavor 和 $(t_q,t_{\bar q})$、gamma 身份/分量/端口、VDV/VVV 时间/动量/位移、输出轴，以及图连通分量。先手算 pion 两点号和 proton 两个置换图。

### 9.3 建 tensor contract table

每个 operand 必须写清：物理意义、完整 shape、轴顺序、时间方向、味、共轭/转置状态、外部指标。然后才把 Wick term 翻译成 einsum。检查每个 dummy index 的 incidence 与 RHS 输出顺序；不要赋予大小写字母 einsum 本身没有的物理语义。

### 9.4 顶点与 perambulator

- `phase_exp_VdV`：VDV 使用的含颜色复制相位；
- `phase_exp_VVV`：VVV 使用的无颜色相位；
- 零位移验证 $V^\dagger V$；
- VVV 验证三腿置换奇偶；
- 反向 peram 用 combined spin-distillation 矩阵显式验证 $\gamma_5\tau^\dagger\gamma_5$。

### 9.5 保守生成 plan

第一版必须：

```python
run_wick_analysis(
    operator_groups,
    use_equivalence=False,
    ignore_dis=False,
)
```

注册表查名通过后仍要检查 shape。每条 plan 用小型确定性 tensor 与直接 `opt_einsum.contract` 比较。跨图求和前检查输出 shape/轴语义完全一致。

### 9.6 等价与 disconnected

- disconnected 用图连通分量；
- 等价必须具有相同 tensor identity、时间、拓扑、输出轴和已证明的 gamma/VVV 相位；
- flavor relabel 只有在实际 tensor 逐组态相等时成立；
- 必须比较 grouped/ungrouped 小型 tensor 和至少一个物理 fixture。

## 10. 推荐的分析流程

1. 固定每个轴及输入层级：raw configuration、source average、Jackknife 还是 Bootstrap sample。
2. source average 的边界符号按绕边界费米线数推导。
3. 明确 resampling estimator、seed、$M,N$、中心值、误差 prefactor 与复 covariance 约定。
4. 三点比值先显式移动时间轴，再计算；用等初末态、已知不等能量 plateau 和复信号三类合成数据验证。
5. 有效质量按谱形式选 log/cosh，屏蔽无效端点。
6. GEVP 保留复 Hermitian 矩阵，检查 $C(t_0)$ 正定/conditioning，并用本征矢重叠追踪态。
7. 报告 fit window、covariance regularization、自由度、实/虚投影与所有 mask。

## 11. 用户授权的重命名

完成审查后只做了以下 API 变更，没有修复其它发现：

- `vertex_creator.phase_exp_2pt` → `vertex_creator.phase_exp_VdV`；
- `vertex_creator.phase_exp_3pt` → `vertex_creator.phase_exp_VVV`。

同步了 `src/lqcddb` 内调用点以及顶层/子包 `.pyi`。旧方法名未保留，因此这是显式的 breaking rename。新方法直接探针：

- `phase_exp_VdV([1,0,-1]).shape=(3,3,3,3)`；
- `phase_exp_VVV([1,0,-1]).shape=(3,3,3)`；
- 去掉 VDV 颜色轴后的相位最大残差 $1.14\times10^{-16}$；
- 旧属性均不存在。

## 12. 环境限制与未覆盖项

- 没有真实 ensemble/perambulator/gauge 数据，不能验证站点二进制文件的物理元数据和真实相关函数归一化；
- CUDA 驱动不足，CuPy 数值 kernel、GPU cache、GPU MPI 和多 GPU 未验证；
- 仅执行 1-rank 和 2-rank MPI，未覆盖不等 local size 的 TGather、大 rank tag 上界及多节点网络；
- 没有对性能 adviser 做真实硬件计时标定；
- 没有对 `create_omega_accelerate` 的完整 stochastic estimator 给出独立论文级推导；
- GEVP 未构造真实 noisy crossing 数据做 state-tracking 研究；
- 审查没有修改上述缺陷。

## 13. 本任务改动文件

授权重命名及同步：

- `src/lqcddb/eigvectors/vertex.py`
- `src/lqcddb/eigvectors/__init__.pyi`
- `src/lqcddb/__init__.pyi`
- `src/lqcddb/test/contraction.pi.py`
- `src/lqcddb/test/contraction.pi.mpi.py`
- `src/lqcddb/test/contraction.proton.mpi.py`
- `src/lqcddb/test/contraction.lambda.proton.weak.mpi.py`
- `src/lqcddb/test/test.dynamic.py`
- `src/lqcddb/test/contraction.proton.mpi.dynamic.py`

交付物：

- 本报告 `lqcddb_physics_audit_report_2026-08-30.md`
- 个人技能 `/public/home/sush/.codex/skills/lqcddb-contraction-analysis/`

除上述精确重命名、调用同步、报告和技能外，没有修改其它 `src/lqcddb` 实现。
