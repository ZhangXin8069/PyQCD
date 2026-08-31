# 管线运行 reference

## 基线与输出目录

默认一致性基线是
`examples/docker-v20260805/output/output_20260802_120104`。每次运行使用独立的
`examples/test0/v<YYYYMMDDHHMM>/` 目录，不能覆盖基线或另一轮运行。记录实际组态列表，
不要把预期网格当作存在性证明。

## 冒烟到全量

```bash
python examples/test0/main.py run --conf-ids 6250
bash examples/test0/run-local.sh
python examples/test0/main.py verify --run-dir examples/test0/v<ts>
```

冒烟适合检查导入、单组态路径和文件格式；它不能验证 ensemble 统计。全量回归至少保存
运行命令、backend/precision、组态 ID、开始结束时间和失败清单。

## 输出根与 dry-run

未显式给出 `run_dir` 时，串行入口、`make_run_dir` 和 MPI 默认路径都从
`pyqcd.pipeline._config.OUTPUT_DIR` 取根目录；显式路径逐字保留，不追加时间戳或 tag。
MPI `--dry-run` 在规划和 collective preflight 后立即返回，不创建 run directory、
`run_config.json`、`data`、`analysis` 或 `plots`；直接 `run_pipeline` 是执行入口，不把
“打印计划”当作它的 dry-run。

## 断点续跑

普通（非 OPE）`step_2pt` 的断点门当前是 `_2pt_all_present`：每个请求通道的 `P0/P2` 都须通过
精确数组检查，HDF5 顶层只能有 `data`，shape 必须为 `(NT,)`、dtype 必须为
`float64` 且数据有限；旧 `.npy` 也必须满足同样的 shape/dtype/finite 条件。空文件、损坏
HDF5、缺少 `data`、额外 dataset、非数值 dtype 或错误 shape 按未完成处理。`.h5` 优先，
损坏的首选 HDF5 不能被同名旧 `.npy` 掩盖。该通用门已验证数组完整性，但不替代 test9
canonical 的 JSON/SHA 身份门。

通用 vertex resume 同样严格检查唯一 `data`、期望 shape、精确 `complex64/complex128`
dtype 和 finite；不做隐式 cast 或用旧格式掩盖损坏的首选 HDF5。大型数组的 finite 检查
按首轴逐块执行，避免构造完整布尔临时副本。

test9 的 vertex/multi-2pt 必须使用 canonical 完成门：请求的所有 artifact 均须通过完整
JSON/SHA、顶层 schema、唯一 `data` dataset、精确 shape/dtype 和 finite 检查；任一缺失或
不匹配即 cache miss，并在读取 eigvec/perambulator 前决定是否重算。契约还必须绑定算法版本、
组态、精度、完整有序动量列表、`v_kind`/通道集合和 lattice identity；可读文件名或数组
shape 不能替代这些字段。

当前生产 vertex 管线的动量列表表示顶点 Fourier 动量 `q_sink`，尚未消费
`apply_momentum_smearing` 的 `p_basis`。若未来启用非零相位 basis，contract 必须另行绑定
`p_basis_sink/source`、`lattice_shape=(Lz,Ly,Lx)`、展平顺序、Fourier 符号，并要求
perambulator 的 basis provenance 一致；独立目录名或 `q_sink` 不能替代这些字段。在该身份
闭环和端到端收缩测试完成前，只能声明局部 basis 相位 API 已验证，不能声明蒸馏链已闭合。

直接串行 `run_pipeline(..., recompute_2pt=True)` 强制重算；MPI `--recompute-2pt` 透传到
每个 2pt 元任务；size=1 的 `run_parallel_pipeline` 真实透传该选项并调用串行入口，不得
静默忽略。dry-run 不调用 `run_pipeline`，只返回推荐计划。不要通过删除未知文件强制重跑；
重启前重新执行输入守卫并核对已有产物的 shape/dtype。

## 原子 HDF5 发布

管线 `save_array` 将旧 `.npy/.npz` 请求规范化为 `.h5`，在目标目录创建临时文件，写入
`data` 后以 `os.replace` 发布，并在 `finally` 清理临时路径。写入中断时旧的最终文件
保持不变，临时残留不应成为完成标记；清理阶段的 `OSError` 会被压制，不能覆盖写入或
发布阶段的主异常。低层 `save_tensor_h5` 本身不是断点完成门。
需要缓存身份校验的 test9 TMD HDF5 另按 `pyqcd-tmd-algorithm` 的缓存契约执行。

## canonical 缓存已验证完成门

以下是缓存红队审查涉及、现已由当前代码与受控 contract 覆盖的实现门。它们
只证明缓存身份、完整性和失败边界；不能升级为真实 ILDG 系综或完整 TMD 物理闭环：

- `_tmd9._save_contract_h5` 在同目录临时 HDF5 中一次写入 `data` 与完整属性，完成后用
  `os.replace` 原子发布；写入中断保留旧最终文件（或没有最终文件），临时路径在
  `finally` 清理，不能把 partial HDF5 当作完成标记。
- 动量编码是带边界的单射；例如 `(1,23,0)` 与 `(12,3,0)` 不共享标签、字典键、spec 或
  accumulator 槽。既有 `P000/P200/P400` 兼容调用保持可用，多位/负分量使用
  `P10_-2_0` 形式；重复动量在目录、缓存或计算副作用前拒绝。
- multi-2pt contract 绑定实际 vertex 上游的 algorithm、artifact 和 contract SHA；上游
  身份变化必须 cache miss，不能只凭任意传入的 `vertices` 命中旧缓存。
- OPE contract 记录所有影响数值的格点身份（当前包括 `NT/NX` 及完整输出 shape）；即使
  输出 shape 相同，空间尺寸变化也必须 miss。
- 通用 vertex/2pt 的完成门精确检查 schema、shape、dtype 和 finite；错误精度、额外 HDF5
  dataset、缺少 `data`、NaN/Inf 或错误 shape 都不能命中，且不隐式 cast。
- 纯 `load_tmd_ope_all`/multi-2pt loader 只构造路径并只读探测，cache miss 不创建缺失的
  `run_dir/data/conf<id>` 目录。
- TMD PDF 链只消费带 `c0_plateau_status_<tag>.npz` 且状态为 `identifiable` 的三维逐样本
  plateau；动量必须满足 `P_z>0, P_y=P_x=0`。二维 `c0_mean`、缺状态 legacy、非有限或
  `z=0` 归一化为零的数组必须显式跳过，不能伪装成可用 PDF 输入。

实现级 contract 的可执行复核命令如下（测试数量以当前工作树命令的实际输出为准）：

```bash
python -B -m pyqcd.testing._tmd9_hybrid_contract
python -B -m pyqcd.testing._pipeline_runtime_contract
python -B -m pyqcd.testing._pipeline_persistence_contract
```

以退出码为零、输出无 `FAIL` 且各模块自行报告通过为门；具体测试数量随当前工作树变化，
不得把旧运行的数量抄作固定验收标准。这些是实现级 contract 证据，不是真实物理数据验证。

## OPE 通道身份、缓存与兼容读取

`compute_ope_for_config` 的 docker 兼容路径必须显式构造三个 `legacy_dual` spec：
`(0,1)`、`(3,0)`、`(3,1)` 都使用 `Ftilde`、`direction=+1`、`sum_kind="full"`、
`normalization="bare_spatial_sum"`、`output_projection="real"` 和
`field_projection="legacy_untraced"`，组合保持 `-O30-O31+2*O01`。这里的裸求和是
逐时间片 `sum_(z,y,x) Tr[...]`，没有体积或 `Nc` 除因子；不能因参考文件名含
`unpol` 就把 legacy 数值改为 `F*F`。

同一组态只创建一个 `FieldStrengthCache(..., max_entries=2)`；Lorentz pair canonical 化后，
三个 legacy 通道合计仍需六个互不重叠的 Clover 场计算，缓存的收益是限制 cache-owned
驻留并支持其他重叠通道复用，不得误报为 `9 -> 6` 次计算加速。miss 在构造新全场前先
淘汰真正 LRU，故 cache 自身不会短暂持有 `max_entries+1` 个场；调用者另持引用与 OPE
scratch 不在这个上界内。

fresh compute 把通道身份以 canonical JSON 写入 combined HDF5 的三个 attrs：

```text
pyqcd_ope_metadata_schema = "1"
pyqcd_ope_channel_specs_json = <canonical JSON list>
pyqcd_ope_combined_spec_json = <canonical JSON object>
```

### 完整 artifact set contract

`compute_ope_for_config` 的严格 resume 单位是三个 component（`(3,0)`、`(3,1)`、`(0,1)`）
加一个 combined 的完整集合。每个 HDF5 都必须只有 `data` dataset，并带
`pyqcd_cache_contract_json` 与对应的 `pyqcd_cache_contract_sha256`；combined 另须带上面的
三个 OPE channel metadata attrs。请求 contract 的身份字段如下：

| 字段 | 绑定内容 |
|---|---|
| `schema`、`algorithm_version` | OPE cache schema 与算法版本 |
| `conf_id` | 组态 ID |
| `channel_spec(s)`、`combined_spec` | `OPEChannelSpec` 字段、通道集合/系数与组合语义 |
| `delta_z`、`z_dir` | 非局部长度与 Wilson 线方向 |
| `lattice` | 运行时 `NT`、`NX` |
| `precision`、`compute_dtype`、`output_dtype` | 请求的复数计算/存储精度；`output_projection="real"` 只投影数值，不把 dtype 转为实数 |
| `shape`、`dtype` | 当前 component/combined 均为 `(delta_z, NT)`，dtype 必须精确等于请求的 `complex64` 或 `complex128` |
| `gauge_source` | 解析后的绝对路径与 stat 身份 |
| `artifact`、`component_contracts` | component/combined 类型及 combined 对各 component contract 的 SHA 引用 |
| `pyqcd_ope_payload_sha256` | 每个 component/combined 的 canonical C-order data bytes SHA-256 |

`gauge_source` 的路径由 `abspath`/`realpath` 规范化；`.lime.contents` 只接受其中固定的
`msg02.rec04.ildg-binary-data`，reader 与 identity 共用该 record realpath。可用时 stat 身份包括
`st_dev`、`st_ino`、`st_size`、`st_mtime_ns`、`st_ctime_ns`，并记录
`stat_available`。这只是低成本的替换/变更检测，不是内容哈希，不能证明文件字节内容已被
读取或核验，也不能防御完全恢复同一 realpath/stat 的 ABA。`stat_available=false` 时没有
正向来源证据，严格 loader 必须直接按保守 miss 拒绝 cache hit；不能把不可用来源当作已验证
的 gauge cache hit，恢复时应让源读取或守卫显式暴露问题。

OPE payload SHA 专用于这些小型 component/combined，不扩大到大型 vertex/2pt；它检测数据
变化但 attrs 未同步变化的损坏。digest 与 payload 共址，不提供对同时改写数据和 digest 的
认证保证。

组件或 combined 任一缺失、contract/SHA/字段不匹配、metadata 缺失或非法、payload SHA
不匹配、array shape/dtype 不匹配、含 NaN/Inf，均是 cache miss。即使各文件 metadata 看似
合法，combined 仍须由已加载的三个 component 按 `-O30-O31+2*O01` 重新组合并与缓存值逐元素
一致；不一致也必须 miss，并走完整 OPE 重算，不能只修补或单独复用 combined。命中路径在
加载后和返回前重新 stat source；fresh 路径在 reader/validation 后以及完整 set staging
完成后、发布前重新 stat。staging 后逐文件 `os.replace`，不是跨四文件事务。

读取层只有在 schema、完整字段集合、canonical JSON、`OPEChannelSpec` 守卫、分量集合/系数
长度及共享字段全部校验后，才返回 `metadata_status="validated"` 和两个 spec。旧的无严格
contract attrs HDF5/NPY 产物只允许由 `load_ope` 兼容读取数组，返回 `metadata_status="missing"`
且不附 spec；部分或非法 OPE metadata attrs 返回 `"invalid"`。它们不能成为
`compute_ope_for_config` 的 resume hit。不得从文件名、当前默认配置或 array shape 给旧产物
补造物理身份；OPE metadata schema 只记录通道/组合身份，不替代上述完整 artifact contract。

可执行门：

```bash
python -B -m pyqcd.testing._ope_channel_contract
python -B -m pyqcd.testing._field_strength_cache_contract
```

前者含非零 `+z` 独立 roll oracle、`F*F`/`F*Ftilde` 局部 oracle、方向/投影、规范性和
metadata 失败边界；后者含 canonical/LRU、backend/device/dtype/shape/flow-time 所有权、
refresh 和失败清理。它们验证实现契约，不替代 docker 参考文件齐全时的逐文件比较。

## 串行失败与资源清理

串行入口最终委托 `pyqcd.pipeline._steps.run_pipeline`。进入步骤调度循环后，无论正常
完成、普通 `Exception`，还是 `KeyboardInterrupt`/`SystemExit` 等 `BaseException`，
`finally` 都会依次尝试 `free_gpu_memory()` 与 `gc.collect()`。两项清理各自包在
`except BaseException` 中：前一项失败不阻止后一项，任何清理失败都不会覆盖原始计算
异常或用户中断。这是 best-effort 清理，不是清理成功证明，也不把清理异常升级成新的
管线结果。

用当前真实契约入口验证，不要编造 failpoint 或 CLI 参数：

```bash
python -m pyqcd.testing._pipeline_runtime_contract
python -m pyqcd.testing._pipeline_persistence_contract
```

其中 `test_serial_pipeline_base_exception_cleans_and_propagates` 明确注入
`KeyboardInterrupt`，断言 GPU 与 GC 清理都执行且原异常对象继续传播。MPI 各阶段的
跨 rank 失败同步由 `pyqcd-infra/references/parallel.md` 负责，不在本节重复定义。

步骤级 `_timer` 先捕获当前 GPU synchronizer：CuPy 使用 null stream，Torch 仅当全局
device 为 `cuda*` 时同步对应 device；Torch CPU 不调用 CUDA runtime。计算函数已经抛出
异常时，后同步仅作 best-effort，必须继续传播原计算/用户中断；计算成功而后同步失败时，
后同步异常仍须报告。OPE 的
`FieldStrengthCache.clear()` 与 GPU 释放属于同类 best-effort 清理。验证这些边界时运行
`_pipeline_runtime_contract` 与 `_pipeline_persistence_contract`，不要用“清理函数也抛异常”
的未实现 CLI 参数替代真实 contract。

## 一致性验证

验证报告至少分开列出：

- 中间张量的相对最大差（通常目标 `<1e-6`）；
- 分析结果的相对差（通常目标 `<1e-8`）；
- NaN 位置是否一致、组态数和各阶段 PASS/FAIL 数；
- 缺失基线时的明确退出状态，而不是把空比较当作通过。

若首个阶段已经超差，停止继续比较下游，先核对输入、精度、后端和版本。只在基线和
当前运行使用相同物理/数值约定时解释逐位差异。

## TMD 示例

`test9_gluon_tmd_nucleon.py --smoke` 用于最小链路检查；`--only-plot` 只消费已算产物。
运行结果需再交给 `test9_verify.py` 和 `pyqcd-tmd-algorithm` 的门控，不能以示例程序
退出 0 宣称完整 TMD-PDF。
