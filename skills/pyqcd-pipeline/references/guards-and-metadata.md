# 数据守卫、进度与元数据 reference

## 输入守卫

运行前同时检查：

1. 组态目录和必需模板是否存在；
2. 文件大小是否非零且与模板/预期类型一致；
3. 数组能否读取，shape、dtype、有限性和轴序是否符合入口契约；
4. 组态 ID 是否唯一，输出目录是否属于当前运行。

`check_files_existence` 的模板占位符可组合多种文件名；缺失、空文件和大小/内容异常
要归入可解释的 `corrupted`/`missing` 清单，而不是让后续步骤产生空结果。

## 进度日志

使用 `pyqcd.pipeline._validate.ProgressLog` 或 `progress_log` 记录带时间戳的阶段、
组态、状态、耗时和 ETA。ETA 是运行规划信息，不是物理结果；中断时保留已完成与失败
任务，重启后从守卫重新判断。

## 环境快照

每个运行目录保存 `pyqcd.tools._env.dump_env` 生成的 `env.json`，至少含 git 状态/版本、
Python 与依赖版本、XeLaTeX、GPU/后端、命令行、时间和数据路径。快照只记录必要的
环境信息，不写入凭据、token 或原始机密配置。

## 产物状态

建议为每个阶段保存 `started/running/completed/failed/skipped` 状态和原因。完成标记只有
在文件写完、能读回并通过 shape/元数据检查后才能落盘；写临时文件后原子替换，避免
断点续跑把半文件当成完整产物。

## OPE 严格 artifact 守卫

OPE 的 resume 检查必须把三个 component 与一个 combined 当作不可拆分的完整 artifact set。
每个文件的 strict contract 绑定以下身份：算法/schema 版本、`conf_id`、channel spec 与
combined spec、`delta_z`、`z_dir`、运行时 `NT/NX`、`precision/compute_dtype/output_dtype`、
完整 `shape/dtype`，以及 gauge source 的解析路径和 stat 身份。当前 component/combined 的
数组契约是 shape `(delta_z, NT)`、存储 dtype 必须精确等于请求的 `complex64` 或 `complex128`；
`output_projection="real"` 只表示实部投影，不改变复数 dtype。HDF5 顶层只允许 `data`，并以
canonical JSON + SHA 属性校验 contract。

gauge source 先规范化为 `realpath(abspath(path))`。如果 stat 可用，身份包含设备、inode、
字节大小、mtime 和 ctime 的纳秒值；`.lime.contents` 必须进一步解析到固定的
`msg02.rec04.ildg-binary-data` record，reader 与 contract 使用同一 realpath。source stat 是
廉价的替换/变更探测，不是 gauge 内容哈希；完整 ABA 不在其保证内。若源无法 stat，
`stat_available=false` 明确禁止 cache hit，必须按保守 cache miss 处理，不能把不可用来源
当作已验证命中。

每个 component/combined 的 `data` 另保存 canonical C-order bytes 的 OPE payload SHA-256。
对每个 strict HDF5 artifact，严格 loader 在同一个 HDF5 handle 内读取 `data[...]`，并在同一
份已加载数组上完成 shape/dtype/finite/payload SHA-256 校验；随后直接使用这份数组，不为
digest 二次打开或二次读取 payload。接着才核对 combined 的线性关系；因此同步修改分量和
combined、但保留旧 attrs 也不能命中。digest 与数据共址，不是带密钥认证；能够同时改写
payload 与 digest 的写者不在该完整性门的威胁模型内。

任一 component 或 combined 缺失、strict contract/SHA 不匹配、metadata 缺失/非法、shape 或
dtype 不匹配、数据非有限，或者 combined 与组件按既定系数重组不一致，均必须判为 miss。
miss 后完整重算受影响的 OPE set；不得用剩余组件拼出“完成”状态，也不得仅因文件可读、
文件名或数组 shape 相同而恢复。

cache 命中在 artifact 加载后和返回前重新 stat source。fresh 计算在 reader 返回后、gauge
validation 后以及完整 set staging 完成后、发布前重新 stat；任一变化都拒绝最终发布。四个
artifact 先各自在同目录 staging，再逐一 `os.replace`；四个最终文件发布完成后还会再检查
一次 source identity。该发布后检查只捕捉在检查前仍可观察到的 source identity 变化；它不
能捕捉已经发生又恢复的 ABA，也不约束检查返回后的变化。没有协作锁或跨文件版本协议，
多个 replace 仍不构成跨文件事务或四文件线性化，发布阶段自身失败或检查报错时可能留下
部分新 set。

旧的无 strict attrs OPE 仅由 `load_ope` 提供兼容读取，并保留 `metadata_status="missing"`
与 `source_identity_status="missing"`，且不伪造 spec；它不是 `compute_ope_for_config` 的
resume 命中。部分或错误类型的 OPE
metadata attrs 属于 `invalid`，也不能降级成 legacy 身份。

`load_ope` 是历史 artifact 的 generic/load-only 入口，不是 resume 判定器。对 OPE metadata
本身合法、但当前 `get_gauge_path(conf_id)` 的 source identity 已与 strict contract 不同或已
无法 stat 的 canonical combined，它仍返回原 combined，并保留已验证的 `channel_specs` 与
`combined_spec`；同时把 `metadata_status` 降为 `stale`，用 `source_identity_status` 区分
`validated/stale/unavailable/unverified/invalid/missing`，并记录 stale 警告。这里的 `invalid`
只表示 strict/source 或 OPE metadata 结构不可验证，不用来代替“来源后来变化”。该状态检查
与数组/metadata 读取不是锁定的原子快照；要求当前来源一致的分析入口必须显式检查并拒绝
`stale`，而历史复盘可在保留警告和 provenance 的前提下消费它。

## 原子写入与异常优先级

`save_array` 在同目录临时 HDF5 完成写入后才 `os.replace`；`finally` 的临时文件清理是
best-effort，清理 `OSError` 不得覆盖写入/发布主异常。`_timer` 在主计算已失败时尽力做
CuPy null-stream 或 Torch CUDA-device 后同步，但继续传播原异常；主计算成功而后同步失败时
则报告后同步异常，Torch CPU 不触碰 CUDA runtime。OPE cache、GPU 和 GC 的释放异常同样
不能把主计算异常替换掉。

可执行验证入口（通过条件以退出码和模块自身的 PASS/FAIL 输出为准，数量随当前工作树
变化）为：

```bash
python -B -m pyqcd.testing._pipeline_runtime_contract
python -B -m pyqcd.testing._pipeline_persistence_contract
python -B -m pyqcd.testing._ope_channel_contract
```
