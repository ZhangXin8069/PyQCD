# TMD validation reference — status, gates, and failure boundaries

## 四种状态

| 状态 | 最低证据 | 可以说什么 |
|---|---|---|
| 实现存在 | 函数/文件可导入或调用 | 接口存在 |
| 测试通过 | 受控输入满足形状/数值断言 | 该断言通过 |
| 方案闭合 | soft、rapidity、流方案、匹配和尺度约定均明确 | 方案内物理量可讨论 |
| 真实数据验证 | 真实逐组态核子三点、系统扫描和误差账本 | 可报告物理结果 |

没有同几何软因子、rapidity subtraction、真实胶子 TMD 三点和完整胶子匹配证据时，
结论必须写成“原型”“接口”或“测试骨架”。

## 门控顺序

1. **输入/群性质**：形状、边界、dtype/单位和 `U†U≈I`。
2. **流积分**：`V†V≈I`、步长收敛、按实际定义检查 `E`；`E` 与 `tau²<E>` 的单调性
   不能混用。
3. **场强/路径**：Clover 反对称、对偶一致、端点/链长、反向共轭和颜色闭合。
4. **几何极限**：`b_perp→0` 直线极限、`z→0` 局部极限、独立 `ell` 和正负方向对照。
5. **外态/统计**：逐组态 `C2*Lg`、真空扣除、多个 `t_sep`、共享索引、协方差和平台。
6. **重整化**：同流时间/表示/几何的软对象，明确短距、Z_R、混合和 `tau` 窗口。
7. **CS/匹配**：至少两个 `Pz`，尺度/阶数稳定，`alpha_s→0` 为单位核。
8. **连续/系统**：多 `a/Pz/tau/ell`，必要时多 `m_pi/L`，协方差加权外推和误差带。

当前调用内 Clover/staple 复用和 `step_tmd` 单次 flow 复用只证明源码契约与受控回归；
在真实 ILDG 组态及 CuPy/Torch 大体积上完成数值、内存和性能验证前，不得把它们升级为
真实数据或生产性能结论。

## 调用内复用与 test9 持久缓存

低层 `tmd_matrix_elements`/`tmd_matrix_elements_time` 的 Clover/staple 复用只存在于
一次调用；它不提供跨调用或跨 `tau` 的隐式缓存。`flow_gauge_for_config`、
`compute_vertices_multi`、`compute_2pt_multi` 和 `compute_tmd_ope_time` 才是显式持久缓存
入口。每个入口的 canonical HDF5 都必须满足同一组发布门：

1. 以排序键、无空白分隔符、`ensure_ascii=True`、`allow_nan=False` 生成 canonical JSON，
   对其 UTF-8 字节计算 SHA-256；文件属性必须同时保存
   `pyqcd_cache_contract_json` 与 `pyqcd_cache_contract_sha256`，加载时重算并比较两者。
2. 顶层键必须严格为唯一的 `data` dataset；读取与写入都检查完整 shape、精确 dtype 和
   等价于 `np.isfinite(data).all()` 的 finite 条件。任何 NaN/Inf、缺 dataset、额外 dataset、
   shape/dtype 不符或属性缺失都只能判 cache miss，不能隐式 cast、降级或复用。
3. contract 必须覆盖真正影响结果的身份：`schema`、`conf_id`、算法版本、精度，以及
   `NT` 和全部空间 lattice 尺寸；还要记录下表的物理参数。输出 shape 不能独立充当 lattice
   identity。

| artifact | contract 至少包含 |
|---|---|
| flow | algorithm、`dtype`/precision、`shape`、lattice (`NT/NX`)、`conf_id`、`tau`、`eps` |
| OPE | algorithm、`dtype`/precision、`shape`、lattice (`NT/NX`)、`tau`、`eps`、`z_dir/b_dir`、有序 `z_list/b_list`、`staple_length`、`color_normalization` |
| full-channel OPE | 上述字段外，另须 channel/orientation/component labels 与 `(2,4,6,nz,nb,Nt)` shape；独立 algorithm/artifact，禁止命中旧 3D `tmd_ope_time.v1` |
| 2D full-channel OPE | 另须有序 `b_x/b_y` 网格、两个横向方向、`transverse_order` 与 `(2,4,6,nz,nbx,nby,Nt)` shape；独立 v3 artifact，禁止与 1D cache 互命中 |
| vertex | algorithm、`dtype`/precision、`shape`、lattice (`NT/NX`)、完整有序动量列表、`VdV/VVV` shape、`NEV/NEV1` |
| multi-2pt v1 | algorithm、`dtype`/precision、`shape`、lattice (`NT/NX`)、完整有序动量列表、通道集合、`v_kind`、vertex 上游 algorithm/artifact/contract SHA |
| source-resolved C2 v3 | 另须 complex128 `(source,dt)`、`polarization=nopol|pol35`、投影定义、`Cg5g4`、反周期边界与 `time_axis_order=source,sink`；旧 v2 禁止复用 |

动量标签必须是带边界的单射编码；`(1,23,0)` 与 `(12,3,0)` 等输入不得共享文件名、字典
键或 accumulator 槽。文件名中的可读参数标签只是定位辅助，不能替代 JSON/SHA。

canonical 写入必须先在同目录临时文件中完成数据和属性，再用 `os.replace` 发布；失败时
旧最终文件保持完整，临时文件清理不参与完成判定。纯 loader 只构造路径并读取，不能因
cache miss 创建 `run_dir/data/conf<id>`。

### 当前证据：canonical persistence gates GREEN

当前代码与受控 contract 已验证以下实现级门：flow/OPE/vertex/multi-2pt 均使用 canonical
JSON 与 SHA-256 身份；HDF5 属性保存 JSON/SHA，顶层只有唯一 `data` dataset，并严格检查
请求的 schema、shape、dtype 和 finite。flow/OPE contract 记录 algorithm、dtype、lattice、
shape；vertex contract 记录算法、完整有序动量和 `VdV/VVV` shape；multi-2pt contract 还
记录实际 vertex 上游的 algorithm、artifact、contract SHA、完整有序动量、通道集合与
`v_kind`。有限性检查按首轴分块，降低大数组临时内存。

canonical 写入门已覆盖 `_save_contract_h5` 的同目录临时文件加 `os.replace` 原子发布；
写入失败不发布 partial 最终文件，旧文件保持不变，临时路径清理不参与完成判定。动量
标签保持 `P000/P200/P400` 兼容，多位/负分量使用边界明确的 `P10_-2_0` 形式，重复动量
拒绝且实际 multi-2pt accumulator 不合并多位动量。纯 `load_tmd_ope_all`/multi-2pt loader
在 miss 时只读探测，不创建缺失的 run/conf 目录。

进入准 TMD/PDF 前还需独立的数据状态门：只接受正纵向动量、`Nsample>=2` 的三维逐样本
`c0_plateau`，并要求配套 `plateau_status=identifiable`；不允许用二维均值或无状态 legacy
数值文件替代重采样输入。该门只证明输入可追溯，仍不证明 soft/rapidity 与连续极限闭合。

本轮实际命令证据为：

```bash
python -B -m pyqcd.testing._tmd9_hybrid_contract
# Ran 44 tests ... OK
```

`44/44` 是当前工作树实际测试集合（包含多位动量 accumulator 用例）；修复前的失败记录
不能继续作为当前状态。上述证据只证明缓存契约与受控输入边界，
不等于真实物理数据闭环：smoke/demo 也不能称为物理完成。仍须通过真实逐组态三点、同几何
soft/rapidity、匹配、多尺度扫描和连续极限等物理门，才可升级到“方案闭合”或“真实数据
验证”。

旧 `.npy` 或缺少完整属性的旧 HDF5 只能只读探测并拒绝复用；不得迁移、删除或改写未证明
的 legacy 文件。只有 JSON、SHA、属性、唯一 dataset、shape、dtype、finite 和请求身份均
精确一致时，旧 HDF5 才可只读复用。验证入口：

```bash
python -B -m pyqcd.testing._tmd9_hybrid_contract
```

任一门失败都保留 raw/intermediate，停止向下游升级状态，不用“文件存在”代替物理验证。

有限横距 ratio-v2 运行前先执行只读输入审计：

```bash
python -B -m pyqcd.testing.tmd.audit_finite_b_inputs \
  --run-dir <run_dir> --conf-ids <id1,id2,...> --out <audit.json>
```

审计要求至少两个组态、每个组态都在同一 channel/momentum 下同时具有
`nopol` 与 `pol35` source-resolved C2 `(source,dt)`，且存在
`(channel,orientation,component,z,b,t)` 七轴 TMD OPE。source-averaged C2 或 3D
非极化组合 OPE 会得到 `status=blocked`，不得通过广播或补零进入 ratio-v2。
同一生产目录中的旧 `tmd_ope_time.v1` 可以与 full-channel v2 共存；审计只对
`artifact=tmd_ope_time_full_channel` 的候选逐项验证，不会因合法旧产物存在而误阻塞，
但任何已识别的 full-channel 候选若 contract/shape/labels 不符仍会使整组失败。
审计 schema v2 将 nopol/pol35 成对覆盖纳入结构 hard gate；空的极化元数据不能由
文件名猜测。通过后仍须验证 `t_sep`、协方差和物理接受门。

二维 OPE 使用同一审计入口的显式模式，网格、方向和 staircase order 必须与读取请求一致：

```bash
python -B -m pyqcd.testing.tmd.audit_finite_b_inputs \
  --run-dir <run_dir> --conf-ids <ids> --dimension 2d \
  --bx-list 0,1,2,3,4 --by-list 0,1 --transverse-order 0,1
```

当前受控集成契约已覆盖两配置小格点的完整材料链：
`full-channel producer → canonical OPE cache → source-resolved C2 cache → audit ready
→ strict loaders → cross-configuration ratio-v2 → Im(R_H)`。该链使用合成 C2 和小规范场，
只证明缓存身份、共享索引、统计接口和最终相位投影能够闭合；不能升级为真实系综结果。

2026-10-25 的真实合同核查使用 `4150/6250`：两组均生成 complex128
`nopol/pol35` `(source,dt)=(72,72)` 缓存并通过 audit v2。`4150 P200` 与 `/public`
`momsmear0_Cg5g4` 参考的映射误差为 nopol `max|d|=1.92e-8`、`rel_L2=3.51e-8`，
pol35 `max|d|=1.38e-8`、`rel_L2=3.71e-6`；两组随机时间对的直接合同 oracle 最大误差
`2.89e-10`。`P200,t_sep={5,6,7}` 的两组态 ratio-v3 接线只标为 diagnostic；Nconf=2
不构成统计结论，也未通过 soft/rapidity、匹配或连续极限门。

后续 11 组态真实诊断 run 已覆盖 `4150..8050` 的 v3 C2、1D full-channel OPE、
`b_x=0..4,b_y=0..1` 2D OPE、audit v2、多 `t_sep` ratio 与 delete-one JK samples。
1D/2D 三个动量的 valid ratio 和 helicity phase 均全 finite；JK 协方差正秩为 10
（`Nconf-1`）。t_sep 漂移超过 JK error，因此当前显著状态必须是
`plateau_unstable/diagnostic_only`，不能输出物理 PDF。`(1,0)` 对 `(0,1)` 的 2D
staircase 比较在 `b_y=1` 上相对 L2 差约 1.8%–2.0%，`b_y=0` 逐元素一致。

GPU 小显存运行必须使用按需 roll 的 `staple_6`。旧预计算路径同时保留 8 个整场副本，
在 8 GiB 卡上会 OOM；两条路径在 NumPy/Torch 受控测试中逐元素一致。修复后的真实
`72×24³` flow 峰值 allocated/reserved 为 `3.97/4.36 GiB`。

## 典型错误

| 现象 | 边界判断 |
|---|---|
| 只有纵向 `z` | 是 quasi-PDF 几何，不是完整 TMD |
| `b_perp` 有参数但路径空段 | 是错误几何；逐段计数前不得解释 |
| 梯度流被当作 rapidity subtraction | UV 平滑不等于快度重整化 |
| 单配置 disconnected | 真空扣除恒等为零，不能作矩阵元 |
| 无依据取实部/偶化 | 可能丢 helicity 或奇对称通道 |
| smoke/demo 图 | 只能证明链路/形状，不是物理结果 |
| 消费者自己拼产物文件名 | 生产者（`ratio_at` 等）定义落盘名；两处各写一遍会得到空表而不是报错。从生产者取名字，或对缺失文件直接失败。 |
| 沿用上一阶段的派生数字 | 窗内点集、交集、计数每次都从产物重算。本轮「共同窗 4 点」实为 16 点，源于旧检查只看了 `z=5,6`。 |

## 比值分析层的典型错误（2026-10-29 实测）

以下每一条都来自本轮 ratio-v3 流时扫描、tsep 扩展与两态拟合的实际误判，
下次做同类分析时先逐条排查。

| 现象 | 边界判断与正确做法 |
|---|---|
| 拿复比值直接与实数数组比较 | 已发布 `unpolarized_ratio` 是复数；比较前取 `Re`。取错会得到 0.43/0.92/16.4 的假最大差，取对后为 2e-14。 |
| 把 Im(R) 当作误差或信号 | 缓存 OPE 虚部严格为零，Im(R) 来自分母 C2 的相位；`complex` 与 `real` 分母约定只差 0.2%/0.25%（P000/P200），P400 10.7% 但已因分母趋零被排除。 |
| 跨流时比较把两个误差相加 | 两个比值共用同一批组态，必须对**差值**做配对 delete-one JK。 |
| 中位 σ≈0.5 就宣布「与流时无关」 | 同时报告中位 `|ΔR|/|R_ref|`。本轮为 0.88/0.78/3.19：差值比误差还大，结论只能是「检测不到，且检验无区分能力」。 |
| 用 τ/a²=3 的 b≥1 结果解释 TMD | 流窗是 `sqrt(8τ) < |z|,|b|`；τ=3 半径 4.90a，所有 b≥1 都在窗外。τ=2 半径 4.00a，窗内没有格点，判为「不可判定」而不是「一致」。 |
| 流窗内格点为空时仍走默认分支 | 判定逻辑必须先写 `if not inside: 不可判定`；只在窗内格点上计算检验功效。本轮曾因此把 τ=2 误标为「存在显著流时依赖」。 |
| 把 tsep=5…10 当作平台 | 扩到 16 后 P000 的 `|R|` 从 0.024 单调升到 0.845，SNR 在 8–12 约 2，之后误差主导。单调上升是激发态污染的典型形状。 |
| 认为加大 tsep 必然提高 B00 精度 | 9 组态 delete-one JK 的协方差秩上限是 8；两态联合拟合的 `|B00|/σ≥2` 通过数为 0/12。 |
| 按字段下标猜测数组轴 | `sigma` 是 `[z][b][tsep]` 而不是按 b 的一行。写分析脚本时先打印 shape 与一个切片，再写循环。 |

### 零成本区间的扩展配方

- full-channel OPE 缓存的插入轴覆盖整条时间序列（72 点），C2 也按整条序列加载，
  因此把 `tsep` 从 5…10 扩到 5…16 只需改分析脚本的 `tseps` 参数：不流化、不重算
  OPE、不写缓存。
- 每次扩展都保留旧区间的逐位复现检查：本轮 `tsep≤10` 子集与已发布产品最大差
  5.6e-17（P000/P200）与 8.9e-16（P400）。
- 先证明 C3 本身对网格大小不敏感：把 `max_tsep` 从 10 改到 16 后，重叠的
  `(tsep, insertion)` 应逐位相同。本轮 3315 个格点的最大差为 **0.0**，
  有限性掩码完全相同；若不为零，「零成本扩展」的说法就不成立。
- 复现检查只能改一个变量。本轮第一次做 `[5,10]` 子窗口复现时，把 C2 拟合区间
  也一并扩到了 16，得到 `|ΔB00|=3.2e-3` 的假差异。修掉上界、只写 `c2_t <= 10`
  后仍有 `5.2e-3`：已发布模块的 C2 区间是 `C2_TMIN=1, C2_TMAX=10`，缺了**下界**
  就把 `t=0` 带进了拟合。按 `(c2_t >= 1) & (c2_t <= 10)` 取掩码后复现差降为
  **0.0**。教训：复现检查里不要按物理直觉重写区间，要直接复用已发布模块的
  `C2_TMIN/C2_TMAX` 常量；上下界各改一次就已经是两个变量了。
- 主拟合固定 C2 区间、另设一个 C2 区间敏感性变体，才能把「新增 C3 数据点」
  与「C2 模型变化」分开。
- 复用既有拟合而不复制代码：用 `importlib.util.spec_from_file_location` 按路径导入
  已发布脚本，只替换模块级 `TSEPS`，再把旧窗口结果与已发布 JSON 逐位比对。
  复制一份拟合代码会悄悄引入第二套符号约定。

### 需要重算的批次轴扩展（b 轴）

`tsep` 是分析层参数，扩它零成本；**几何轴 `b` 不是**。full-channel OPE 的 contract
键含完整有序 `b_list`，扩 `b` 必须重新流化、重算 OPE。可执行配方：

1. **先读缓存键再动手。** 打开已有 HDF5 的 `pyqcd_cache_contract_json`，确认 `b_list`
   在键里、`eps`/`staple_length`/`precision` 与新请求一致；确认新文件名（本轮
   `..._b5_0to4_...` → `..._b7_0to6_...`）不会覆盖旧缓存。
2. **用「旧切片逐位一致」证明非侵入性。** 不要用「独立重算」当参考：同一份代码
   没有独立性，而且 24³×72 单线程 numpy 流化要几十分钟。拿磁盘上已发布的 torch
   生产缓存做参考，逐文件比较新缓存的 `b≤4` 切片：
   `max|Δ| = 0` 且 NaN/有限性掩码完全相同才算通过（本轮 9/9 个 torch 产物为 0.0）。
3. **后端不同则判据不同，且容差要按量级校准。** 若新缓存用 CuPy/P100 算，会差
   `1e-6` 相对量级（本轮 `max|Δ| ≈ 6e-4~1.9e-3`，出现在 `|ref|≈0` 的格点）。
   **不要用绝对容差**：缓存跨 1e4 量级，float32 的 ~1e-7 相对舍入就是 1e-3 的
   绝对差。用相对 Frobenius 范数 `||Δ||/||ref||`，按实测地板校准阈值：本轮
   9 个 CuPy 组态为 `4.8e-7~7.0e-7`，阈值取 `1e-5`（14 倍余量，仍比真实索引/
   几何错误低 5 个数量级）。把阈值与实测范围写进产物。判据标签必须与实测值一致：
   只有 `max|Δ| = 0` 才叫 `bit_exact`；全部组态同一后端但差 `1e-14` 量级
   （BLAS 归约顺序）写 `same_backend_1e-12`；含 CuPy 写 `float32_equivalent`。
   2026-10-29 收尾检查发现比值层 τ=3 的 `2.1e-14` 曾被标成 `bit_exact`——
   数值无碍但标签错误，这类「看起来通过、语义却错」的标签要在产物里用
   `criteria` + `exactly_equal` 两个字段同时表达。
   配对用的通配符必须钉住 schema：同目录还有 `tmd_ope_v1_*` 与 2D staircase 的
   同 `tau` 缓存，`*b5_0to4_tau1_*` 会随机命中 v1 并给出 1e3 量级的假失败。
4. **新格点单独验健全性。** b=5,6 必须 100% 有限，中位量级必须延续 b=3,4 的
   趋势。口径要写清：先在**每个组态**的 channel/component/z/t 全部有限元素上取
   `|OPE|` 中位数，再对 **9 个组态**取中位数——本轮 27/27 全部到位后，τ=3 的
   b=3→6 为 `2.405/1.779/1.566/1.493`（τ=1 为 `3.315/3.107/2.971/2.872`，
   τ=2 为 `2.628/2.104/1.974/1.898`）。换口径（如只取 z≥5、或先合并组态再取
   中位数、或用均值）会差 5%–10%，两边看起来都合理，所以数字必须带定义；
   只跑了一半组态时得到的数不能写进结论（本轮 τ=2 早期只有 4 个组态，
   `2.66/2.14/2.01/1.91` 与最终值差 1%–2%，但仍是错的来源）。量级跳变或出现
   NaN 说明批量分组或几何索引有问题，不能进入比值。
5. **下游同步轴长。** 已发布拟合模块的 `B_LIST` 是模块级常量（0..4），必须同步，
   否则 `_c3_for` 只构造 b≤4 的轴、`_flatten_c3(z,5)` 越界。优先以缓存自带的
   `b_list` 为准，并在缓存里存 `flow_tau` 供下游核对，避免把别的流时缓存
   当成本流时使用。新缓存用带后缀的名字，保留旧缓存以维持已发布数字的可复现性。
6. **检查器要能发现「同名多文件」。** 同一 (tau,conf) 若有多份命中同一 contract
   的文件，`glob(...)[0]` 的结果就依赖目录顺序，检查结论不再可复现。把
   `len(matches) > 1` 直接判为错误，而不是静默取第一个。

### 流窗与流时的取舍（2026-10-29 实测）

- 流窗判据 `sqrt(8τ) < |z|, |b|` 把可用 \((z,b)\) 与流时锁死：
  \(\tau=1\) 半径 2.83a，只有 \(b=3,4\) 在窗内；\(\tau=2\) 半径 4.00a，
  \(b\le4\) 全部出窗；\(\tau=3\) 半径 4.90a 同理。**在 \(b\le4\) 的网格上只有
  \(\tau/a^2=1\) 存在窗内格点**，任何「流时外推」都无法进行；要覆盖 \(\tau=2,3\)
  必须先把横向距离扩到 \(b\ge5\)。这是生产几何问题，不是分析脚本能解决的。
- 2026-10-29 把 \(b\) 扩到 6 后，三个流时终于有了共同窗：\(\tau=1\) 的窗内集合是
  \(b=3..6,z=3..12\)（40 点），\(\tau=2,3\) 是 \(b=5,6,z=5..12\)（各 16 点），
  **交集为 \(z\ge5,b=5,6\) 的 16 点**。跨流时比较只能在这个交集上做：
  把 \(\tau=1\) 的 40 点与 \(\tau=2,3\) 的 16 点混在一起，会把只在 \(\tau=1\)
  有效的 \(b=3,4\) 带进比较。
- 共同窗只有 16 个 \((z,b)\)，仍属小样本。报告必须同时给出 \(\sigma\) 分布与
  「最靠近流半径的 4 点 \((5,5),(5,6),(6,5),(6,6)\)」子集的结果，用于区分
  边缘效应与整体趋势；\(\sigma\) 中位数小于 1 时结论是「无判别力」。
- \(\tau=1\) 的窗内结果必须单独标注。本轮把 `tsep` 扩到 24 后，`[5,24]`
  P000 (4,3) 给出 `B00 = 1.04e-2±2.3e-3`（4.5σ），**但它不是 τ=1 的矩阵元**：
  1. 固定插入时间 \(\tau\) 且除以 `exp(−m0·tsep)` 后，`B01` 项留下常量
     `B01·e^{(m0−m1)τ}`，物理上可提取的是
     \(P(\tau)=B_{00}+B_{01}e^{(m_0-m_1)\tau}\)。
     逐 delete-one 复制品计算 P 得 `corr(B00,B01) = −0.76`，
     `P(1) = +3.2e-4 ± 2.9e-3`，**0.11σ**；`B00` 的 4.5σ 是反相关交换的产物。
  2. 信号在更晚的插入点：`P(3) = 8.3e-3 ± 1.8e-3`（4.6σ）、
     `P(5) = 9.9e-3 ± 2.2e-3`（4.5σ）。但这些 \(\tau\ge3\) 在 `b≤4`
     网格上全部出流窗，不能用作流窗内结果。
  3. 晚期窗口不可用：`[10,24]` 的 `B01` 被噪声推到 `−6e-2`，
     `[16,24]` 的 `B00` 误差已达 `1e-1`。只有 `[5,24]` 稳定，
     `[5,24]` 与 `[5,16]` 的 `B00` 仅差 1%（1.036e-2 / 1.024e-2）。
     此前「`B00` 随窗口单调漂移」的读法被 `[11,16]` 的噪声窗口误导，
     不是系统趋势。
  4. 低自由度：9 组态协方差秩 8、7 个参数；单窗口高显著性可以在相关
     参数之间自由交换，不构成平台证据。

  结论：**流窗内（τ=1）没有可提取信号；τ≥3 的 4.5σ 组合出窗。**
  报单参数 `B00` 的显著性会系统性夸大流窗内的物理结论。
  误差必须从同一批 delete-one 复制品的组合量求，不能把参数误差平方相加。
- 判据：用逐 tsep 有效振幅 `B00_eff(tsep)` 检查平台（见
  `multistate_fit_tsep16/b00_tsep_plateau.py`）。只有 `B00_eff` 在误差内平坦，
  才能谈矩阵元；联合拟合在某个窗口的高显著性不构成平台证据。

### 判读前必须通过的数据审计

- 误差字段不得为负、不得恒为零；`χ²/dof` 越界、设计矩阵条件数超过 `1e10` 都要报警。
- 比值类数据出现 `1e6` 量级的大值，先查分母是否趋零，不能当作信号。
- 窗口误差应能由 JK 样本独立重算并与已存值逐格一致（本轮差 0.0）。
- **JK 样本的轴顺序要单独验。** `samples_one` 是 `(nconf, Nobs, Nmom)`，直接
  `reshape(shape+(nconf,))` 会把组态轴与 observable 轴混在一起；这种错误下
  「误差由样本重算」仍然逐格通过，因为两边用的是同一份错样本。判据是
  `|mean(JK)-R|/err_JK`：比值是非线性统计量，偏差应在噪声之下（本轮中位 0.014、
  p99 0.09、最大 0.28）；混序会让它在所有格点上 O(1)。修复是
  `np.moveaxis(samples_one, 0, -1).reshape(...)`。
- 只报告 min/max/mean/median/std 与 NaN/Inf 计数还不够；还要把极端值与物理上
  已知的放大机制（如 P400 的 C2 趋零）逐一对上。
- ratio-v3 在插入时间超过 `tsep` 的格点上**按约定写 NaN**（1D 975/4290、2D
  1950/8580）。审计必须先验证 NaN 掩码与「`insertion_index > tsep`」逐格一致，
  再统计其余数值；直接按「有 NaN 就是错误」报警会掩盖真问题，也会让审计失去
  可信度。
- NaN 掩码检查器本身也必须参数化。把 `tsep` 扩到 24 后的 C3 缓存形状是
  `(z,b,tsep=20,insertion=25)`，而审计里写死「轴长 6 / tseps=5..10」的版本
  对不上轴，把 12350 个约定内的 NaN 误报成 4 条硬错误；改为从 NPZ 的
  `tseps` 数组推导轴映射后归零。**审计器的轴假设与产物的轴约定必须同源**，
  否则检查器会制造假阳性。
- 同一条越界在不同模型下含义不同：C2-only 单指数拟合在 `t=1…10` 上 `χ²/dof`
  高达 225，这是缺激发态的已知模型不足，应标 `info`；两态联合拟合若出现同样
  的数值才是真张力。审计输出要分级 `error/warn/info`。
- 审计脚本跨目录收集产物时不要用相对父目录推导路径（本轮 `parents[1]` 指错，
  导致已发布 NPZ 完全没被审计）；用仓库绝对根路径，或显式断言目录存在。
- 从拟合 JSON 里挑一个「参考质量」`m0` 时，不能按字典序取最后一个窗口。
  实测 `c2_only.P200` 的 `5..16` 两态拟合退化到 `m0=0.0601`（其余窗口
  0.78--0.80，联合拟合 0.7912--0.7937），而 `sorted(keys)[-1]` 正好选中它；
  用它做 `exp(-m0 tsep)` 归一后 `B00_eff` 掉到 `1e-7` 量级、完全失真。
  规则：**显式指定窗口**（优先与主拟合相同的 C2 区间），并做两项退化检查——
  `m0` 落在 `(0.2, 1.5)` 格点单位内，且与其余可用窗口中位数相差不超过 20%；
  不满足就报错，不要静默退回。凡是「从多个拟合里自动挑一个」的地方都要
  先验证挑选规则本身，而不是只用它算出一个数。
