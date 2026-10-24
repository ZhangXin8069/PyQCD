# Data transfer reference — comparison-data inventory and sampled downloads

## 适用范围

任务涉及跨项目对照数据的体积盘点、组态级抽样、存储预算、机器可读下载清单、远程
传输或断点续传时读取。本文件只定义数据选择和传输契约；“2pt、OPE、ratio、matching
或 spin-2 结果在物理上代表什么”仍由 `pyqcd-tmd-chain`、`pyqcd-analysis` 及对应领域
技能定义。

## 先确定权威数据集合

1. 从当前代码、manifest、inventory 或已冻结分析契约中取得组态 ID，不凭目录中“看起来
   最新”的文件重新发明样本。若下游已经声明 `Nconf=20`，优先沿用同一组 ID。
2. 同一物理链的配置、eigenvector、perambulator、2pt 和 OPE 应保持可追踪的组态对应；
   允许按体积分层取不同的 `Nconf`，但必须在清单中明确哪一类用了多少个组态。
3. 对照数据不只包含 OPE/ratio。按实际物理链纳入适用的谱学/2pt、matching、flow
   外推、Wilson/static potential、line-mass/Zchi、spin-2、LaMET 和跨 ensemble 最终
   结果；原始 TB 级中间体只在复算确有必要时纳入。
4. 私人文档、代码缓存和不参与对照的巨型中间树不作为“完整性”缺口，也不能用它们
   抬高关键数据体积。

## 体积盘点

对明确列出的路径做只读元数据遍历：

```bash
ionice -c 3 nice -n 19 du -sB1 -x <path>
```

- `du -sB1 -x` 给实际文件系统分配字节；文件逻辑长度不同，不能混称。
- 默认不跟随符号链接；若业务依赖链接目标，必须单独登记目标、去重口径和跨文件系统行为。
- 巨大目录按顶层子目录或配置目录受控并行；不要对 `/public`、`/home` 或整个共享盘做
  无边界扫描。
- `find -printf` 可统计类型、文件数或路径，但不要对百万级目录逐项格式化输出后再聚合，
  这会增加管道开销；需要总量时优先 `du`。
- 权限错误、被拒绝目录和主动跳过项必须逐项保存。一个路径只有在无未解释缺口时才能标为
  `complete`；否则标为 `partial` 或 `inaccessible`，大小未知时不要估算补洞。
- 报告同时给十进制 `TB` 与 IEC `TiB`，并记录统计日期、路径范围和是否包含目录元数据。

## 预算与分层抽样

先测单组态体积，再解总量约束：

\[
S_{\rm total}=n_{\rm gauge}S_{\rm gauge}+n_{\rm eigen}S_{\rm eigen}
+n_{\rm peram}S_{\rm peram}+S_{\rm light}+S_{\rm result}.
\]

轻量 2pt/OPE 可以覆盖较多组态；eigenvector 和 perambulator 必须按实测单组态成本单独
分层。不要把一个 `Nconf=20` 机械应用到所有类别。

一个已验证的 L32x96 标定（仅作预算起点，不是固定参数）：

| 数据层 | 单组态或汇总体积 | 采样含义 |
|---|---:|---|
| raw CLOVER `.lime` | 约 `1.812 GB` | 可覆盖较多组态 |
| eigenvector 目录 | 约 `84.375 GiB` | TB 级，先测再定额度 |
| light perambulator | 约 `351.562 GiB` | 20 组态远超 2 TB |
| OPE 目录 | 约 `152.278 MiB` | 通常可跟轻量组态层 |
| proton 2pt 目录 | 约 `14.375 MiB` | 通常可跟轻量组态层 |
| 计算对照结果 | 按目录实测 | 最终结果通常比原始中间体便宜且价值高 |

在 `2 TB` 预算下，2026-10-25 的 L32x96 实测组合为：20 个 gauge、4 个 eigen、4 个 perambulator、
20 个 OPE、20 个 2pt 和 33 个 donghx 最终结果目录，总计 `1,958,277,360,128 B`
（`1.958277 TB` / `1.781043 TiB`）。这类分层方案既保留较多轻量输入，又保留少量完整
传播子链和全链对照结果。

## 机器可读清单

清单应逐行保存实际源路径，不展开目录中的每个中间文件：

```text
size_bytes | human_iec | dataset | conf | source_path
```

文件头至少记录：

- 统计日期、ensemble、选择依据和明确包含的组态 ID；
- 每一类的条目数与总字节；
- 总条目数、总字节、十进制与 IEC 单位；
- 未纳入的巨型中间体和主动跳过项；
- 全部源路径存在于生成时的声明。

生成后校验：

1. 所有数值行字段数一致；
2. 数值求和等于头部总量，分类求和等于分类总量；
3. 源路径无重复且逐条存在；
4. `partial`/`inaccessible` 不被描述成完整；
5. 清单不与私有目录或无关共享盘混写。

## 下载脚本

下载器读取清单，不重复硬编码路径。建议接口：

- `--dry-run`：本地解析、计数、求和并打印映射，不建立 SSH 连接。
- `--yes`：非交互运行；默认仍逐项确认。
- `--skip-existing` / `--resume`：仅跳过脚本已记录完成的条目。
- `--dest-root PATH`：保留远端绝对层级；默认落在技能或任务目录下，显式 `--dest-root /`
  才镜像绝对路径。

传输实现：

1. 校验预期条目数和总字节，防止清单截断后“成功下载一部分”。
2. 每个远端路径先在线上确认是文件还是目录；`--dry-run` 可按 `.lime/.npy/.npz` 等后缀
   做只读预判。
3. 用 SSH ControlMaster 复用连接；优先 `rsync -aP --partial`，缺少 rsync 时回退
   `scp -r`。
4. 成功传输后写入本地状态标记。不要只用本地 `du` 与远端分配字节比较来决定重跑，
   不同文件系统块大小会使相等内容得到不同分配量。
5. 单条失败继续汇总并在末尾返回非零；保留 partial 文件供下一轮恢复。
6. 真实传输前先运行：

```bash
bash -n <download-script>
bash <download-script> --dry-run
```

## 常见错误

| 错误 | 正确动作 |
|---|---|
| 用 `du` 扫整个 `/public` 找“相关数据” | 从代码/manifest 提取明确根目录，再逐项盘点 |
| 20 个组态应用到 perambulator 层 | 先测单组态体积，按预算分层定额 |
| 只下载 OPE/ratio 就称对照完整 | 按物理链补齐最终计算对照，明确仍未覆盖项 |
| 把权限拒绝目录忽略后仍标 `complete` | 标 `partial`/`inaccessible`，不估算缺失大小 |
| 下载脚本按自己的硬编码列表运行 | 解析权威清单并校验 count/bytes |
| 只按远端分配字节判断本地已完成 | 使用成功标记，并让 rsync 按文件内容/大小继续核对 |
| dry-run 仍连接远端或写目标目录 | dry-run 只做本地解析和映射预览 |

## 交接

向物理/分析技能交接时提供：清单路径、选中的组态 ID、各数据层 `Nconf`、清单 SHA256、
总字节、未覆盖项和结果目录意义。向 `pyqcd-pipeline` 交接下载后的绝对路径、完整性状态
和断点标记；向 `pyqcd-docs` 交接统计口径与差值，不把“能下载”表述成“物理链已验证”。
