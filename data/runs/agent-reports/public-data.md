# `/public` 数据与运行环境审计（PyQCD 胶子 TMD-PDF 链）

审计日期：2026-10-22  
工作目录：`/root/PyQCD`  
范围：只读检查指定 `/public` 路径、当前仓库代码和 `/root/configure`；唯一写入为本报告。

## 结论

1. `conf6250` 可用于当前 `pyqcd.testing.tmd.pdf_workflow`：gauge、72 个本征矢文件和 288 个 perambulator 文件均齐全，命名、大小和读取器契约匹配。
2. `/public` 没有现成的 `conf6250` 梯度流 TMD OPE、`c0(z,b,Pz)` 或 TMD-PDF 成品；这些必须由当前生产代码从原始数据计算，不能从其他组态或参考结果复制。
3. `donghx/Ope_Gluon/TMD_Ope_Gluon` 只有 `conf4150` 的未流化、分 Lorentz 通道参考 OPE，形状和语义均不是当前 workflow 的 `(nz,nb,Nt)` 流化算符缓存。
4. `/public/home/sush/distillation` 本身不是数据目录，只含两个指向 `/root/lattice-pdf/...` 的悬空链接；`/public/group/lqcd/donghx/Gradient_Flow_GluonPDF` 已不存在。
5. 单组态 `--smoke` 只能验证计算链和不可辨识状态；要实际进入当前 PDF
   分支，至少需要 2 个真实组态。后续连续轮次已把 conf4150 纳入，当前
   `/public` 可用的 11 个组态均生成了当前 workflow 的 strict 2pt/OPE。

## 检查证据

- 对 eigensystem/perambulator 按精确文件名集合检查，而非只数文件数；11 个组态均满足 `eigvecs_t000..071` 和 `perams.{cid}.{0..3}.{0..71}`。
- 实测读取 `conf6250` 的一个本征矢：输出 `(100,13824,3) complex128`，有限，首个本征矢范数 `1.0000000000000002`，耗时约 `0.55 s`。
- 实测读取 `conf6250` 的 `t_source=0` perambulator：输出 `(72,4,4,100,100) complex128`，有限，约 `175.8 MiB`，耗时约 `1.92 s`。
- 未全量读取或遍历 136 GiB perambulator 内容；目录总大小只由文件元数据求和得到。
- 检查到现有本地 `pdf_workflow --smoke` 产物含严格 HDF5 contract JSON/SHA-256；单组态被正确标为 `statistically_unidentifiable`，未伪造 PDF 结果。

## 数据清单

### 1. Gauge configurations

路径：

```text
/public/group/lqcd/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72
```

| 项目 | 实测结果 |
|---|---|
| 文件 | 11 个 `.lime`，总计 `6,307,830,640 bytes`（`5.875 GiB`） |
| conf6250 | `beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_6250.lime`，`573,439,144 bytes`（`546.87 MiB`） |
| 覆盖 | `4150, 6250, 6450, 6650, 6850, 7050, 7250, 7450, 7650, 7850, 8050` |
| 格式 | SciDAC/ILDG LIME；内层 big-endian `float64` 的交错实虚部 |
| 数据轴 | 磁盘 payload `(Nt,Nz,Ny,Nx,dir,Nc,Nc,2)=(72,24,24,24,4,3,3,2)`；读取后 `(72,24,24,24,4,3,3) complex128` |
| 分类 | raw gauge ensemble input |
| 元数据 | LIME 内嵌 SciDAC XML：dims `24 24 24 72`、`QDP_D3_ColorMatrix`、precision `D`、colors `3`；6250 内嵌 checksum `eac26005/5ec97fc5` |
| 外部 manifest/hash | 未发现批次 manifest、SHA256 清单或 sidecar metadata；只有文件内 LIME checksum |
| conf6250 可用 | 是，路径与 `get_gauge_path(6250)` 完全一致 |
| 资源 | 单文件读取约 `547 MiB` host memory；后续 Wilson flow 建议 GPU |

### 2. Distillation eigensystem

路径：

```text
/public/group/lqcd/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72
```

| 项目 | 实测结果 |
|---|---|
| 文件 | 792 个，总计 `52,553,318,400 bytes`（`48.944 GiB`） |
| 每 conf | 72 个文件；conf6250 总计 `4,777,574,400 bytes`（`4.449 GiB`） |
| 单文件 | `eigvecs_tTTT_6250`，`66,355,200 bytes`（`63.281 MiB`） |
| 覆盖 | 11 个组态，同上；每个组态 `t=000..071` 完整 |
| 格式 | `float64` 交错复数；磁盘 `(100,24^3,3,2)`，读取为 `(100,13824,3) complex128` |
| 分类 | intermediate，由 gauge 求得的 Laplacian distillation basis |
| 元数据 | 文件名仅编码 conf/time；未发现 manifest、hash、JSON、日志或 XML |
| conf6250 可用 | 是；已实测读取 `t=000` 且有限 |
| 资源 | 逐时间片读取，单文件约 `63.3 MiB`；不应一次载入整组态 72 文件 |

### 3. Light perambulators

路径：

```text
/public/group/lqcd/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light
```

| 项目 | 实测结果 |
|---|---|
| 文件 | 3168 个，总计 `145,981,440,000 bytes`（`135.956 GiB`） |
| 每 conf | 288 个文件；conf6250 总计 `13,271,040,000 bytes`（`12.363 GiB`） |
| 单文件 | `perams.6250.d.t`，`46,080,000 bytes`（`43.945 MiB`），`d=0..3,t=0..71` |
| 覆盖 | 11 个组态，同上；每个组态 `4 x 72` 文件精确齐全 |
| 格式 | 每文件 `float64` 交错复数；每时间源合并 4 个 `d_source` 文件后读取为 `(72,4,4,100,100) complex128` |
| 分类 | intermediate，distillation-space light quark propagator |
| 元数据 | 文件名仅编码 conf/Dirac source/time；未发现 manifest、hash、JSON、日志或 XML |
| conf6250 可用 | 是；已实测读取 `t_source=0` 且有限 |
| 资源 | 当前读取器逐时间源载入，每片约 `175.8 MiB` host memory；不可把整个 12.36 GiB 组态当作一个小数组载入 |

### 4. Donghx gluon OPE reference results

路径：

```text
/public/group/lqcd/donghx/Ope_Gluon/TMD_Ope_Gluon
```

| 项目 | 实测结果 |
|---|---|
| 内容 | 2 个 `Calc_ope_unpol_new*.py` 脚本、`Operator.py`、`Result/<ensemble>/{xdir,ydir,zdir}/4150` |
| 结果规模 | 58 个文件，`2,697,924 bytes`（`2.573 MiB`）；其中 57 个 `.npy`、1 个 `.log` |
| 覆盖 | 只有 `conf4150`；没有 `conf6250` 或其余默认组态 |
| xdir | 6 个 `(72,9,8) complex128` |
| ydir | 6 个 `(72,9,8) complex128` |
| zdir | 45 个 `complex128`：27 个 `(72,4,5)`、6 个 `(72,12,5)`、6 个 `(72,9,8)`、3 个 `(72,3,5)`、3 个 `(72,6,5)` |
| 数值检查 | 全部 57 个数组可读且未见 NaN/Inf |
| 分类 | intermediate/reference OPE matrix elements，不是 final PDF |
| 元数据 | 仅 `4150_new.log` 记录维度和耗时；无 manifest、SHA256、flow 参数或完整输入 provenance |
| 语义 | 按 `mu,nu,direction,dz/dx/zmax` 分文件，直接输入 gauge/plaquette 后计算；未发现 Wilson-flow/tau 处理 |
| conf6250 可用 | 否。既无 6250，也不满足当前 `O=(tx;tx)+(ty;ty)-2(xy;xy)` 的 `(13,5,72)` 流化缓存契约 |
| 资源 | 读取成品无需 GPU；源计算日志显示 3 个 MPI rank 约 `51 min`，会直接读取约 `547 MiB` gauge 并做全时间/位移循环 |

该目录可用于 `conf4150` 的独立参考/算符比较；仓库入口是
`pyqcd/testing/comparisons/common/manifest_4150.py:19` 和
`pyqcd/testing/comparisons/donghx/run_4150_hyp_ope.py:39`。它不是生产输入。

### 5. Sush distillation links

路径：

```text
/public/home/sush/distillation
```

| 项目 | 实测结果 |
|---|---|
| 根目录 | 存在，目录 `4096 bytes` |
| 内容 | 仅两个符号链接，没有真实数组文件 |
| `lqcddb` | `-> /root/lattice-pdf/examples/sush/lqcddb` |
| `function_contraction` | `-> /root/lattice-pdf/examples/sush/function_contraction` |
| 目标状态 | `find -L` 仍将两者识别为符号链接而非可遍历目录，即当前为悬空链接 |
| 边界 | 目标在 `/public` 外，按本任务路径白名单未读取其内容 |
| 分类 | 无可用 raw/intermediate/final 数据 |
| conf6250 可用 | 否；不能作为 gauge/eigen/peram/OPE 数据源 |
| 资源 | 无数组可加载，无 GPU/大内存需求 |

### 6. Referenced Gradient Flow tree

声明路径：

```text
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF
```

| 项目 | 实测结果 |
|---|---|
| 状态 | `stat` 返回 `ENOENT`，该路径已消失 |
| 分类 | 无可用 raw/intermediate/final 数据 |
| conf6250 可用 | 否 |
| 影响 | 仓库文档和参考代码中指向它的命令/结果路径均为过期引用，不能作为当前真实数据或可运行输入 |

## 当前代码读取入口

| 数据/阶段 | 入口 |
|---|---|
| 路径解析 | `pyqcd/pipeline/_config.py:25` 默认 10 组态；`:32-38` 三类 `/public` 根路径；`:116-127` 生成 gauge/eigen/peram 路径 |
| Gauge | `pyqcd/operator/_gluon_ope.py:912` `read_gauge_lime()`；返回 complex128 `(72,24,24,24,4,3,3)` |
| Eigenvectors | `pyqcd/tools/_io.py:33` `readin_eigvecs()`；`:67` GPU/backend 版本 |
| Perambulators | `pyqcd/tools/_io.py:116` full reader；`:188` `readin_peram_time_slice()`，生产 2pt 使用后者 |
| Workflow | `pyqcd/testing/tmd/pdf_workflow.py:48` 默认组态；`:129-159` vertex/2pt/flow/OPE；`:216-293` 有状态 plateau 门控和 PDF 链 |
| Vertex/2pt | `pyqcd/pipeline/_tmd9.py:559` `compute_vertices_multi()`；`:572` `compute_2pt_multi()` |
| Flow/OPE | `pyqcd/pipeline/_tmd9.py:592` `flow_gauge_for_config()`；`:743` `compute_tmd_ope_time()`；输出严格 HDF5 `(13,5,72) float64` |
| 读取缓存 | `pyqcd/pipeline/_tmd9.py:828` `load_multi_2pt()`；`:879` `load_tmd_ope_all()` |
| 原始数据守卫 | `pyqcd/pipeline/_validate.py:60` `check_raw_data()` |

## 缺失项与不可伪造输入

- `/public` 缺少 `conf6250` 的 flowed gauge、流化 TMD OPE、`c0_plateau_P200.npy`、`c0_plateau_status_P200.npz`、准 TMD-PDF 和匹配后 PDF。
- `/public` 缺少全部默认 10 组态的当前生产 TMD OPE/PDF；已有 `donghx` OPE 只覆盖 4150 且语义不同。
- eigensystem、perambulator 和 donghx OPE 没有外部批次 manifest/SHA256；可检查形状和有限性，但不能证明长期字节级不变。
- 不能把 `conf4150` 的 OPE 改名或插值成 `conf6250`；不能复制另一个组态的本征矢/perambulator/gauge；不能把单组态 NaN fit 或先验值冒充 PDF。
- 当前 `pdf_workflow` 要求 `c0_plateau` 至少有 2 个样本、`plateau_status=identifiable`、有限的 `z=0` 归一化和 `Nsample>=2`；单组态会被正确拒绝（`pdf_workflow.py:251-267`）。
- 不能使用 `sush/distillation` 悬空链接或已消失的 `Gradient_Flow_GluonPDF` 树补齐输入。

## 最小真实数据闭环

### A. 单组态冒烟：已验证代码兼容性

输入：

```text
conf6250 gauge + eigensystem/6250 + perambulators/light/6250
P200, z=0..12, b=0..4, tau=3, eps=0.05
```

- 原始数据总量约 `17.34 GiB`，无需复制；程序按时间片流式读取。
- 现有 CLI：`python -m pyqcd.testing.tmd.pdf_workflow --smoke --backend torch --precision complex64`
- 预期：vertex、2pt、Wilson flow、`(13,5,72)` OPE、ratio 和状态文件均生成；单组态 fit 必须为 `statistically_unidentifiable`，PDF 链应跳过。
- 参考实测：test9 在 RTX 4060 上对 3-momentum 单组态测得 vertex `38 s`、2pt `430 s`、flow `208 s`、OPE `126 s`，合计约 `13 min`；现有单 P200 artifact 的落盘跨度约 `16 min`，可作为单动量冒烟上界。
- 资源量级：host 建议至少约 `2-4 GiB` 可用内存用于单组态流式工作区；GPU 建议至少 `8 GiB`，实测瓶颈是 flow/OPE；单 P200 输出 vertex cache 约 `0.54 GiB`。

### B. 两真实组态进入 PDF 分支：最小代码闭环

建议真实输入：

```text
conf6250, conf6450; P000/P200/P400; z=0..12; b=0..4
```

- 现成 CLI 可运行：`python -m pyqcd.testing.tmd.pdf_workflow --conf-ids 6250,6450 --momenta Z --backend torch --precision complex64`
- 两个真实组态会产生 bootstrap `Nsample=200`，可满足 plateau 状态分支；这是端到端代码验证，不应冒充统计生产结果。
- 预计计算约 `27-35 min` GPU，单组态 3-momentum vertex cache 约 `1.6 GiB`，两 conf 输出约 `3-4 GiB`；host/GPU 内存要求同 A。

### C. 当前设计的生产统计闭环

使用默认 10 组态：

```text
6250,6450,6650,6850,7050,7250,7450,7650,7850,8050
```

- 10 组态原始输入合计约 `173.43 GiB`，均已在 `/public` 且无需搬移。
- 按 test9 的 RTX 4060 实测，单组态计算链约 `13 min`，10 组态约 `2.2 h`，不含后续分析和作图。
- 这是可获得当前代码设计统计量级的最小完整集合；每个时刻仍逐片读取，不需要把 136 GiB perambulator 树整体装入内存。

## 未执行项与阻塞

- 未重跑完整 `--smoke` 或 10 组态链：本任务仅允许写本报告，workflow 会创建 H5/图/日志并需要长时间 GPU 计算。
- 未对 `/public/home/sush/distillation` 的目标目录做内容审计：链接已悬空，且目标在允许路径之外。
- 未验证 eigensystem/perambulator/donghx OPE 的字节级哈希：这些目录没有提供可核对的 manifest/hash。
