# AGENTS.md — logs/test0

**test0** —— pyqcd「输入数据路径 → 数据分析并作图」功能测试套件
（参考 /root/PyQCU/logs/test12 形式）。

被测功能：`pyqcd.analysis.analyze_3dir`（独立实现，功能对齐
`refer/huangcl/05_ana_3dir_diff_sem/code_ana_3dir_diff_sem.py`，不 import refer）。
数据目录结构约定：`<data_root>/<conf>/Pz<Pz>/{x,y,z,ave}_dir/ratio.npy` 与
`corr2_{x,y,z,ave}.npy`。

## 运行

```bash
python logs/test0/main.py env                        # 环境自检
python logs/test0/main.py makedata                   # 合成数据（含 truth.json）→ input/
python logs/test0/main.py run --data-root logs/test0/input   # 分析+作图 → v<ts>/
python logs/test0/main.py verify --run-dir <v<ts>>   # 断言（存在性+数值自洽+物理自洽）
python logs/test0/main.py check  --run-dir <v<ts>>   # 断言门（exit 0/1）
python logs/test0/main.py collect --run-dir <v<ts>>  # 产物清单
bash logs/test0/run-local.sh                         # 一键：env→makedata→run→verify→check→collect
TEST0_DATA_DIR=/tmp/xx bash logs/test0/run-local.sh  # 自定义数据目录
```

## 约定

- **版本目录**：`logs/test0/v<YYYYMMDDHHMM>/`（test12 约定），一次运行一个版本目录，
  `--outdir` > `$TEST0_OUTDIR` > `v<ts>/` 优先级；产物互不覆盖，跨环境可直接 diff/叠图。
- **main.py 只含测试/编排代码**：分析+作图全部委托 `pyqcd.analysis.analyze_3dir`，
  无核心计算逻辑。
- **合成数据**（`makedata`）：corr2 = A·e^(−m·t)·(1+ε·N)，meff 可解析恢复 m
  （x/y/z 方向取 1.10/1.12/1.15，模拟方向差异）；ratio 三方向 base 不同 + z 方向
  高斯包络。ground-truth 写入 `input/truth.json`。
- **verify 断言**：A 产物存在性；B 统计量（mean/sem）与独立 numpy 重算
  rel < 1e-8（不调用 pyqcd 自身统计函数，避免自验）；C meff 恢复 truth m
  （|dev| < 5·sem + 0.02）；D 相关系数矩阵对称/对角 1/|offdiag| ≤ 1。
- 产物：`v<ts>/env.json`、`v<ts>/<conf>/{ratio,corr2,eff_mass}/*.png`、
  `v<ts>/<conf>/ana_3dir_summary.json`、`test0_verify.json`、`test0_collect.json`。
