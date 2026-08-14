# AGENTS.md — logs/test0

test0 —— **docker-v20260805 全量蒸馏 GPU 管线一致性测试**工作目录。
目标：复现 `/root/PyQCD/examples/docker-v20260805/output/output_20260802_120104`
的完整管线结果（10 组态蒸馏：vertex → 2pt/3pt/4pt → OPE → analysis → plots →
LaTeX 报告），中间数据与图表与基线一样完整保存；总体形式参照
`/root/PyQCU/logs/test12`（版本化产物目录 + env.json + 汇总 json + run 脚本）。

## 与 test12 的形式对照

| 方面 | test12 (PyQCU) | test0 (本目录) |
|---|---|---|
| 产物位置 | `logs/test12/v<ts>/` | **`logs/test0/v<ts>/`** |
| 版本目录 | `v<YYYYMMDDHHMM>`（同分钟加 `-<SS>`） | 同左 |
| 环境快照 | `env.json`（GPU/驱动/torch/git/命令） | **`env.json`**（GPU/cupy/git/命令） |
| 输出重定向 | `--outdir` / `TEST12_OUTDIR` | `--outdir` / `TEST0_OUTDIR` |
| 汇总 | `test12_results.json` + 表/图 | `test0_results.json` + `test0_verify.json` |

## 目录结构

```
logs/test0/
├── main.py           全部测试代码（子命令入口，--outdir 公共参数）
├── run-local.sh      本地运行脚本（全量 10 组态 → verify → collect → report）
├── AGENTS.md         本文件（复现与比对指南）
├── config.py         照抄 docker-v20260805（AGENT_LOGS_DIR 改为本地 logs/）
├── utils.py          照抄（日志/计时/精度/数组 I/O）
├── lib/              照抄 lqcddb 蒸馏收缩框架快照（自包含，不 import examples/）
├── compute_vertex.py 照抄（VdV/VVV，GPU x-slice 分解）
├── compute_contraction.py 照抄（Wick/动态收缩：2pt/3pt/4pt）
├── compute_ope.py    照抄（donghx 胶子算符：Clover F̃ + Wilson 线）
├── analyze.py        照抄（Jackknife/meff/ratio_3p + code_1.py 不相连拟合）
├── report.py         照抄（LaTeX 物理报告生成与编译）
└── v<YYYYMMDDHHMM>/  每次运行生成的版本目录
    ├── run-local-<ts>.log           完整终端输出（tee 归档）
    ├── env.json                     环境快照（比对基准）
    ├── output_YYYYMMDD_HHMMSS/      管线输出（与基线 output_20260802_120104 同构）
    │   ├── data/conf{id}/           VdV/VVV/corr/3pt/4pt/OPE 中间数据
    │   ├── data/analysis/           meff/corr/ratio 统计数组
    │   ├── analysis/disconnected/   不相连比值与拟合
    │   ├── plots/                   correlators/meff/ratio 图
    │   ├── analysis_summary.json    数值一致性比对基准
    │   ├── physics_report.tex/.pdf  LaTeX 报告
    │   └── run_config.json          运行配置
    ├── test0_verify.json            一致性验证结果（vs 基线）
    └── test0_results.json           collect 汇总
```

## 快速开始

```bash
cd /root/PyQCD
bash logs/test0/run-local.sh                # 实际执行；--dry-run 只打印命令
```

环境前提：GPU + cupy（输入数据在 /public/group/lqcd/，本地可访问）；
xelatex 用于编译中文报告。

## main.py 子命令

```bash
python logs/test0/main.py <subcommand> [options] [--outdir <dir>]

env       --conf-ids ...         # 数据/GPU 自检 + env.json
pipeline  --steps env,vertex,2pt,ope,3pt,4pt,analysis,plots,report
                                 # 完整管线（--conf-id/--conf-ids/--precision/
                                 #   --Nev1/--skip-*/--channels/--run-dir 透传）
verify    --ref <基线目录>        # 数值一致性（rtol=1e-3，atol=1e-8）→ test0_verify.json
collect   --run-dir <子目录>      # 汇总 → test0_results.json
report    --run-dir <子目录>      # 生成并编译 physics_report.tex/.pdf
```

`--outdir` 为公共参数；未指定时读 `TEST0_OUTDIR` 环境变量，再默认 `logs/test0/`。
每次调用自动在输出目录写 `env.json`（含命令与硬件/软件快照）。

## 版本目录约定

- **命名**：`v<YYYYMMDDHHMM>`（如 `v202608141400`）；同分钟重复运行自动追加 `-<SS>`。
- **创建**：run-local.sh 开头 `mkdir -p` + `export TEST0_OUTDIR=$VDIR`；主流程命令
  不带 `--outdir`，经环境变量生效。
- **env.json**：每次子命令调用刷新，作为该版本目录的环境基准。
- **清理**：版本目录不再需要时可整体删除（不影响代码与后续运行）。

## 数值一致性基准（勿改）

基线 `examples/docker-v20260805/output/output_20260802_120104/analysis_summary.json`：
proton_P0 E0=1.1183±0.0075、proton_P2 E0=1.5585、pion_P0 E0=0.2863、
pion_P2 E0=1.1779（GeV，复杂平台窗）；pn 2pt=0（味守恒）；OPE 与 v20260802 相关
系数 1.0。verify 容差：rtol=1e-3 / atol=1e-8（fp32 合理水平）。

## 关键约定（沿用 test12 / docker-v20260805）

- 代码文件位于根目录**不入版本目录**；运行产物全部进版本目录。
- 自包含照抄：不 import `examples/`、不 import `refer/`。
- 日志与产物不入库，由用户决定保留或清理。
- 单步失败仅记录并继续（run 脚本 `timeout` + `[warn]` 机制）。
