# qcd-einsum 扩展生产路径测试最终报告

状态：`finished`

## 覆盖

- 在原始 6 条主验收路径外，按 12 个 case 与元数 6/7/9/10/12/13/14/16 分层测试
  43 条 full-scene 路径。
- CPU/GPU 同时提交，double，真实 `Nx=32/Lt=64/Nev=100/Ns=4/Mom=8` 输入。
- 全部 8638 条生产方程另做计划资源门审计；该审计不执行大张量，不冒充性能数据。

## 合并性能

比值为 selected/generic，小于 1 更快。

| 设备 | 路径 | 中位比 | 范围 | ≥10% 加速 | 不快于 generic |
|---|---:|---:|---:|---:|---:|
| CPU | 43 | 0.8612x | 0.3471–1.0067x | 26 | 42/43 |
| GPU | 43 | 0.2760x | 0.1218–1.3615x | 40 | 42/43 |

极少数反例：

- CPU `seq=856`：1.0067x，慢 0.7%。
- GPU `seq=6911`：1.3615x，极小计划固定 launch/pack 开销。

第一轮 `seq=7918` 为 pack-bound，CPU 慢 1.93x。新增
`pack>=1 GiB && FLOPs<=1G` 门后，CPU/GPU 均安全回退至约
`0.996x/0.958x`。

## 全计划门控

Job `1648360`：

```text
tasks=8638
plan unavailable=0
CPU allowed=3833  blocked=4805
GPU allowed=2864  blocked=5774
```

CPU 门：arena ≥8 GiB、单 GEMM ≥100G MAC、pack-heavy。
GPU 门：arena ≥3 GiB、单 GEMM ≥40G MAC、pack-heavy。

审计中最大静态 arena 约 `5.04 PB`、最大单步约 `65.536P MAC`。

## 正确性

- 43 条 CPU full-scene 最大相对误差 `6.95e-15`。
- 43 条 GPU full-scene 最大相对误差 `6.68e-15`。
- CPU job `1648387`：KUPL native 16/16、plan-gate 11/11、热路径 4/4。
- GPU job `1648386`：plan-gate 12/12；统一套件
  `117 passed / 2 skipped / 2 failed`。
- 两项失败均为环境缺可选依赖 `gvar`。

## 数据产物

- `results.json`：86 条 CPU/GPU 路径结果。
- `summary.json`：合并统计和全计划门审计。
- `EXPERIENCE.md`：规律、反例和门控原因。
- `jobs/`：全部 Slurm 原始输出。
- Python source manifest SHA-256：
  `606c285728978366f5efab2991d96b4f7a2acbe47d9c65e5c905c3113014036f`
- 语义源码 manifest SHA-256：
  `fb4179bb48d74f75c7f36668d75d8ac879f5d48c7d2bcf1440bf5d5e66471750`
- `results.json` SHA-256：
  `2d9a649e4121df9e3353612d56069294ea50219f8dcdf32a85c524bb4ad5701a`
- `summary.json` SHA-256：
  `c24f760038581cfe495949a996781c32ad6584313c115e011e4cd0f17a4d3b72`

## 结论

当前结果满足“普遍加速、允许极少数反例”：CPU/GPU 各 42/43 条路径不快于
generic，GPU 中位加速约 3.62x，CPU 中位约 1.16x。最新额外资源门已通过当前源码
专项测试和统一套件验证。
