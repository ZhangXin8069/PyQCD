# AGENTS.md — examples/test1

test1 —— **调用 pyqcd 包复现 docker-v20260805 全量蒸馏 GPU 管线一致性测试**。
目标：以 `main.py` 只调用 `pyqcd` 包的方式，复现
`/root/PyQCD/examples/docker-v20260805/output/output_20260802_120104`
的完整管线结果（10 组态蒸馏：vertex → 2pt/3pt/4pt → OPE → analysis → plots →
LaTeX 报告），中间数据与图表与基线一样完整保存；总体形式参照
`logs/test0`（即 PyQCU/logs/test12 形式：版本目录 + env.json + 汇总 json + run 脚本）。

## 与 test12 / test0 的形式对照

| 方面 | test12 (PyQCU) | test0（自包含照抄） | **test1（本目录，调用 pyqcd 包）** |
|---|---|---|---|
| 产物位置 | `logs/test12/v<ts>/` | `logs/test0/v<ts>/` | **`examples/test1/v<ts>/`** |
| 版本目录 | `v<YYYYMMDDHHMM>` | 同左 | 同左（本次 `v202608140630`） |
| 环境快照 | `env.json` | `env.json` | **`env.json`**（GPU/cupy/git/命令） |
| 输出重定向 | `--outdir` / `TEST12_OUTDIR` | `TEST0_OUTDIR` | `--outdir` / `TEST1_OUTDIR` |
| 汇总 | `test12_results.json` | `test0_results.json` | **`test1_results.json`** + `test1_verify.json` |
| 计算实现 | torch 自包含 | 照抄 lib/（不 import examples/） | **只 import `pyqcd` 包 + numpy/matplotlib** |

## 目录结构

```
examples/test1/
├── main.py           全部子命令入口（env/pipeline/verify/collect/report），只调用 pyqcd
├── run-local.sh      本地运行脚本（全量 10 组态 → verify → collect → report）
├── AGENTS.md         本文件（复现与比对指南）
└── v202608140630/    版本目录（本次运行，格式 v<YYYYMMDDHHMM>）
    ├── run-local-<ts>.log           完整终端输出（tee 归档）
    ├── env.json                     环境快照
    ├── output_YYYYMMDD_HHMMSS/      管线输出（与基线 output_20260802_120104 同构）
    │   ├── data/conf{id}/           VdV/VVV/corr/3pt/4pt/OPE 中间数据
    │   ├── data/analysis/           meff/corr/ratio 统计数组
    │   ├── analysis/disconnected/   不相连比值与拟合
    │   ├── plots/                   correlators/meff/ratio 图
    │   ├── analysis_summary.json    数值一致性比对基准
    │   ├── physics_report.tex/.pdf  LaTeX 报告
    │   └── run_config.json          运行配置
    ├── test1_verify.json            一致性验证结果（vs 基线）
    └── test1_results.json           collect 汇总
```

## 快速开始

```bash
cd /root/PyQCD
bash examples/test1/run-local.sh                  # 实际执行；--dry-run 只打印命令
python examples/test1/main.py env                 # 环境自检
```

环境前提：GPU + cupy（输入数据在 /public/group/lqcd/，本地可访问）；
xelatex 用于编译中文报告。

## main.py 子命令

```bash
python examples/test1/main.py <subcommand> [options] [--outdir <dir>]

env       --conf-ids ...         # 数据/GPU 自检 + env.json
pipeline  --steps env,vertex,2pt,ope,3pt,4pt,analysis,plots,report
                                 # 完整管线（--conf-id/--conf-ids/--precision/
                                 #   --Nev1/--skip-*/--channels/--run-dir 透传）
verify    --ref <基线目录>        # 数值一致性（rtol=1e-3，atol=1e-8）→ test1_verify.json
collect   --run-dir <子目录>      # 汇总 → test1_results.json
report    --run-dir <子目录>      # 生成并编译 physics_report.tex/.pdf
```

`--outdir` 为公共参数；未指定时读 `TEST1_OUTDIR` 环境变量，再默认 `examples/test1/`。
每次调用自动在输出目录写 `env.json`（含命令与硬件/软件快照）。

## 实现约定（与基线/包的关系）

- **计算全部走 pyqcd 包**：
  - vertex：`pyqcd.vertex`（phase_exp_2pt/3pt、Mom_VdV_sink_t）+ `pyqcd.tools._io.readin_eigvecs_gpu`
  - 2pt/3pt/4pt：`pyqcd.contraction`（dynamic_contraction/PeramRegistry/VRegistry/GammaRegistry/seq_peram）+ `pyqcd.tools.readin_peram_time_slice` + `pyqcd.lattice.gamma`
  - OPE：`pyqcd.operator`（plaquette_clover/compute_dual_field_strength/gluon_ope_operator_z0/read_gauge_lime）
  - analysis：`pyqcd.analysis`（Jackknife/meff/ratio_3pt/run_disconnected_ratio/sem）
- **VVV 顶点例外**：pyqcd.vertex.Mom_VVV_sink_t 为单 einsum，8GB GPU 在
  Nev=100/Nx=24 下超时 → main.py 用基线 x-slicing 因子化（`_compute_vvv_single_t_gpu`
  算法，数学等价：逐 x 切片累加 Levi-Civita 六项），仍只依赖 pyqcd backend/phase。
- 绘图（matplotlib）与 LaTeX 报告在 main.py 内实现（pyqcd 无绘图/报告模块），
  输出与基线 plots/、physics_report.tex 同构。
- 不 import `examples/`、不 import `refer/`（包级依赖仅 pyqcd + numpy/matplotlib/lsqfit）。

## 版本目录约定

- **命名**：`v<YYYYMMDDHHMM>`（本次 `v202608140630`）；同分钟重复运行追加 `-<SS>`。
- **创建**：run-local.sh 开头 `mkdir -p` + `export TEST1_OUTDIR=$VDIR`；主流程命令
  不带 `--outdir`，经环境变量生效。
- **env.json**：每次子命令调用刷新，作为该版本目录的环境基准。
- **清理**：版本目录不再需要时可整体删除（不影响代码与后续运行）。

## 数值一致性基准（勿改）

基线 `examples/docker-v20260805/output/output_20260802_120104/analysis_summary.json`：
proton_P0 E0=1.1183±0.0075、proton_P2 E0=1.5585、pion_P0 E0=0.2863、
pion_P2 E0=1.1779（GeV，复杂平台窗）；pn 2pt=0（味守恒）；OPE 与 v20260802 相关
系数 1.0。verify 容差：rtol=1e-3 / atol=1e-8（fp32 合理水平）。

## 关键约定（沿用 test12 / docker-v20260805）

- 代码文件位于根目录不入版本目录；运行产物全部进版本目录。
- 日志与产物不入库，由用户决定保留或清理。
- 单步失败仅记录并继续（run 脚本 `timeout` + `[warn]` 机制）。
- GPU 显存 8GB 限制：VVV 必须 x-slicing；每步计算后释放 GPU 内存。
