# `analyse.py` 物理修复与 lazy API/.pyi 一致性报告

## 1. 范围与基线

- 日期：2026-08-30
- 工作区：`/public/home/sush/distillation/lqcddb_update`
- 修改前源码/存根聚合 SHA-256：`791067b67d2c2c3a4de72bb77a0bc7fcc262237b8b37dd775b5ed206621731a5`
- 修复完成时源码/存根聚合 SHA-256：`2c0a040dbc1047f644fb80eaa46ee7b82018ef09421f2459969182b4c8664d86`
- 增加语义兼容性测试后的当前聚合 SHA-256：`48d59cb5b68269b9b1ef209ace6597a1e9f8df6ccba6109fa19affcae2d60c0e`
- 实现范围：`src/lqcddb/analyse/analyse.py` 中先前确认的物理/统计问题，以及 lazy export、公开函数签名、调用点和 `.pyi` 一致性
- 未修改：Wick contraction 的物理、Stout、等价图、顶点归一化、性能 adviser 的算法和 MPI 数据搬运算法

## 2. 关于四夸克算符的科学澄清

“标准模型禁止四夸克粒子/四夸克算符”并不成立。QCD 允许颜色 singlet 的

$$
q q \bar q \bar q
$$

复合算符；有限体积谱学也必须使用双介子、多强子以及可能的紧致四夸克插值场。一个插值场是否合法，与某个实验峰最终被解释为紧致 tetraquark、hadronic molecule 或散射态是不同问题。

原审查的准确结论应表述为：当前 lqcddb grammar 只明确定义了单介子和规范重子 factor；对未支持的单 factor 多夸克 grammar 应 fail closed，不能静默生成缺失 VDV/VVV 的网络。该结论是接口输入域问题，不是声称 QCD 禁止四夸克算符。本轮按用户指定范围没有扩展 contraction grammar。

## 3. `analyse.py` 修复

### 3.1 `Mom2GeV`

- 要求 `Nx>0`、`alttc>0`；
- scalar `Mom` 明确表示 $|\mathbf n|^2$，禁止负值；
- 动量分量和质量必须为实数，质量非负；
- 空质量列表不再静默返回 0。

保留原 continuum 动量约定

$$
|\mathbf p|=\frac{2\pi}{N_x a}|\mathbf n|\,\hbar c.
$$

### 3.2 `loop_tsrc`

- `Boundary_Conditions` 现在明确指完整关联函数的边界条件，不是单条夸克线；
- 新增 `boundary_phase`，允许显式提供一次绕时间边界的复相位；
- 2pt 依据 `t_sink<t_src` 判断绕回；3pt 依据固定 `t_sep` 判断物理 sink 是否绕回；
- 3pt 的第二时间轴明确是 current/insertion，而物理 sink 由 `t_sep` 固定；
- 校验轴、时间 extent、`Ctype`、边界类型和 `t_sep`；
- 直接在当前 NumPy/CuPy backend 上运算，不再反向转换 backend 类型。

### 3.3 分组求和/均值

- 支持合法负轴；
- 拒绝重复聚合轴、空 group、重复索引、漏索引、非整数索引。

### 3.4 Jackknife

标准 delete-one 样本保持为

$$
x^{(k)}=\frac{\sum_i x_i-x_k}{N_{\rm conf}-1}.
$$

- 任意位置的 `Nconf_axes` 均支持 covariance；
- 均值在配置轴位置显式恢复 singleton 后再构造 residual；
- 复 covariance 使用

$$
\operatorname{Cov}_{ij}^{\rm JK}
=\frac{N-1}{N}\sum_k r_{k,i}r_{k,j}^*;
$$

- 拒绝 $N_{\rm conf}<2$、重复 covariance 轴和把配置轴当 covariance 轴。

### 3.5 Bootstrap

- 全部 $N$ 个样本都是独立的有放回 replicate，不再把原始均值混入第 0 个样本；
- `data_mean` 是原始 $N_{\rm conf}$ 个组态的均值；
- 实输入保留实 dtype；
- replicate 标准差与 covariance 使用 `ddof=1`；
- 对大小为 $M$ 的 m-out-of-n replicate，方差乘 $M/N_{\rm conf}$，误差乘

$$
\sqrt{M/N_{\rm conf}},
$$

从而估计原始 $N_{\rm conf}$ 样本均值的不确定度；
- 复 covariance 使用共轭 residual，保持 Hermitian；
- 新增可选 `seed`。

在 $N_{\rm conf}=30,M=10$、8000 replicates 的独立测试中，返回误差与原始样本均值目标误差之比在 4% 容差内；更高统计探针得到 1.0004。

### 3.6 `meff`

- 支持 `float32/float64/complex`；
- log/GEVP 仅填 $t=0,\ldots,N_t-2$；
- cosh 使用中心公式

$$
E_{\rm eff}(t)=\operatorname{arccosh}
\frac{C(t-1)+C(t+1)}{2C(t)},\qquad t=1,\ldots,N_t-2;
$$

- 公式域外和非有限点为 NaN，并返回同 shape `valid_mask`；
- 非法 estimator、重采样类型、轴和格距 fail closed。

### 3.7 `ratio_3pt`

- C3 的 insertion 轴和 C2 的 sink 轴分别移动到统一语义位置，再比较完整 canonical layout；
- 支持时间轴位于不同位置而不依赖偶然 broadcasting；
- 一维与带 source 绝对时间的二维模式使用同一物理公式；
- 复数域计算平方根，负实参数保留主值相位，不再 `maximum(...,0)`；
- 只保留 $0\le\tau\le t_{\rm sep}$ 且分母有限非零的点，其余为 NaN；
- 返回 `valid_mask`；
- link folding 要求奇数 extent $2L+1$。

独立验证包括：

- 异位置时间轴、不等初末态直接公式；
- 负平方根参数产生非零虚部而非 0；
- 2D 周期绕回；
- $E_I=0.25,E_F=0.55$ 的合成三点函数恢复平台矩阵元 1.7，最大误差低于 $2\times10^{-14}$。

### 3.8 `solve_gevp`

- 保留复 Hermitian 非对角元；
- 可选、显式的 $(C+C^\dagger)/2$ 投影；
- 要求 $C(t_0)$ Hermitian positive definite，并用 `rcond` 控制最小本征值阈值；
- 默认用

$$
\left|v_i(t)^\dagger C(t_0)v_j(t+1)\right|
$$

和 Hungarian assignment 跟踪态，并相位对齐；
- `track_states=False` 保留逐时间片独立谱排序作为显式选项。

复谱模型与 `scipy.linalg.eigh` 一致；人工 crossing 中 overlap tracking 保持同一 eigenvector branch，而独立排序按预期交换标签。

### 3.9 `dis_connect`

明确输入轴契约：

```text
(configuration, source time, absolute sink/insertion time, ...)
```

- bubble 可具有完整 source extent 或 singleton source 轴；
- 对配置作中心化后，把 C2 与 bubble 分别移到相对时间；
- 保留物理算符乘积 $\delta C_2\,\delta B$，没有错误加入统计 covariance 的复共轭；
- PDF/PFF 两条原有时间区间定义由显式轴运算实现；
- 对任意轴排列与独立循环参考一致。

## 4. lazy API 与 `.pyi` 修复

### 4.1 运行时导出

- `lqcddb.base.get_cache_keys` 现在可用；
- `lqcddb.io.safe_save` 现在可用；
- `lqcddb.contraction` 真实导出 `HardwareSpec`、`ParsedContraction`、`ContractionCost`、`SlicingSuggestion`、`UpsizingSuggestion`、`BandwidthAnalysis`；
- `vertex_creator.sink2src` 明确为 `@staticmethod`，类和实例调用均一致。

### 4.2 optional dependency lazy 行为

- `from lqcddb import *`、`from lqcddb.base import *`、`from lqcddb.contraction import *` 不再初始化 optional MPI/SymPy；
- MPI/SymPy API 仍可显式 named import；
- `dynamic.py` 导入不再加载 MPI。若调用者已经显式初始化 lqcddb MPI，dynamic 日志继续读取真实 rank；否则按 serial rank 0 工作；
- 5 个 `src/lqcddb/test` MPI 示例从 wildcard 改为完整 named imports。

2-rank 顶层 named lazy MPI `Allgather` 实测通过。

### 4.3 签名与存根

已同步：

- `conjugate_operator(tokens=...)`；
- `run_wick_analysis(..., Projection, Oindex)`；
- `dynamic_contraction(..., Oindex)`；
- `vertex_creator.VdV_sink_t_link` 的当前参数顺序、默认值和 `eigvecs_max`；
- `sink2src`；
- `check_dir_path(save_path) -> None`；
- analyse 新增参数与返回 mask；
- contract adviser 数据类构造器、属性方法；
- optional `mpi4py` 不再被 `.pyi` 无条件导入；
- `pathlib.Path`、`Any/Dict/Tuple` 等存根依赖完整导入。

自定义 AST 比较覆盖顶层和 6 个子包的 112 个 callable 映射：签名不一致 0；59 个顶层 mapping 及所有子包 mapping 均有存根声明。

## 5. 兼容性变化

这些变化是为恢复明确物理/统计定义而有意发生：

- Bootstrap 的 `data_sample[0]` 不再是原始均值；全部条目都是 bootstrap replicates；
- Bootstrap `data_mean` 现在严格是原始样本均值；
- `ratio_3pt` 和 `meff` 的无效点由 0 改为 NaN，并新增 `valid_mask`；
- `ratio_3pt` 结果使用复 dtype，以保留平方根相位；
- GEVP 默认启用 state tracking，且非正定 $C(t_0)$ fail closed；
- optional MPI/SymPy 名称不再进入 wildcard `__all__`，但显式 named import 保留；
- unsupported keyword names 以当前实现为准，旧的 stub-only 名称不再宣称可用。

## 6. 验证

更新前后作用/物理语义的差分验证另见 [analyse_semantic_compatibility_report_2026-08-30.md](analyse_semantic_compatibility_report_2026-08-30.md)。

### 6.1 串行回归

- `test_conjugate_operator.py`：9/9；
- `test_equivalent_diagrams_strict.py`：13/13；
- `test_requested_fixes.py`：7/7；
- `test.cache.py`：15/15；
- `test_contraction_njnp_3pt_gevp.py`：2/2；
- 新增 `test_analyse_physics.py`：10/10（每个方法包含多个独立不变量）；
- 新增 `test_lazy_api.py`：4/4；
- 原修复回归：60/60；增加 8 项语义兼容性检查和 1 项 lazy 对象同一性检查后，总计 69/69。

### 6.2 静态/API/MPI

- 当前加入语义兼容性测试后共 50 个 `.py/.pyi`，AST 解析 0 错误；
- callable mapping 签名比较：112 项，0 不一致；
- wildcard optional-import 隔离：通过；
- 2-rank 顶层 named lazy MPI `Allgather`：通过；
- 原 MPI 数据搬运探针中 Gather/Allgather/TGather/Scatter/TScatter/Bcast/Send 均继续通过。

原 MPI 探针仍报告两个本轮未授权修复的既有问题：负时间 scalar/list rank 映射不一致、`Transport` 多一个 singleton rank 轴。这两项不是本轮回归。

## 7. 仍未修复的主要问题

- Stout 的 $Q=0$ 与空间 roll 轴；
- dynamic `ignore_dis` 的时间启发式；
- 等价图的无条件 $u,d\to\text{light}$；
- Wick 未知 token、多 gamma 链与大字段数输出指标；
- conserved temporal VDV 的动量相位；
- eigenvector 验证、`Z_N` 与 omega normal；
- contraction adviser 的 ellipsis、FLOP 与硬件效率；
- 上述两个 MPI 数据布局问题。

## 8. 修改文件

核心与存根：

- `src/lqcddb/analyse/analyse.py`
- `src/lqcddb/analyse/__init__.pyi`
- `src/lqcddb/__init__.py`
- `src/lqcddb/__init__.pyi`
- `src/lqcddb/base/__init__.py`
- `src/lqcddb/base/__init__.pyi`
- `src/lqcddb/contraction/__init__.py`
- `src/lqcddb/contraction/__init__.pyi`
- `src/lqcddb/contraction/dynamic.py`
- `src/lqcddb/eigvectors/vertex.py`
- `src/lqcddb/eigvectors/__init__.pyi`
- `src/lqcddb/io/__init__.py`
- `src/lqcddb/io/__init__.pyi`

调用同步：

- `src/lqcddb/test/test.dynamic.py`
- `src/lqcddb/test/contraction.proton.mpi.dynamic.py`
- `src/lqcddb/test/contraction.lambda.proton.weak.mpi.py`
- `src/lqcddb/test/contraction.proton.mpi.py`
- `src/lqcddb/test/contraction.pi.mpi.py`
- `src/lqcddb/test/test_requested_fixes.py`

新增测试：

- `src/lqcddb/test/test_analyse_physics.py`
- `src/lqcddb/test/test_lazy_api.py`

报告/技能：

- `analyse_api_fix_report_2026-08-30.md`
- `lqcddb_physics_audit_report_2026-08-30.md`（追加后续状态与科学澄清）
- `/public/home/sush/.codex/skills/lqcddb/`
- `/public/home/sush/.codex/skills/lqcddb-contraction-analysis/`
