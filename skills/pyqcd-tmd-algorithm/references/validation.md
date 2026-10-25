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
