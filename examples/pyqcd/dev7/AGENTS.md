# AGENTS.md — examples/pyqcd/dev7

**dev7** —— dev6 的收敛迭代（~auto-all）：基于已有收缩结果补齐 test0/test6
同类型全部图表（10 型齐全）。输入只读；Part C 为本次新增。

## 输入（只读）

`${HOME}/data/beta6.20_mu-0.2770_ms-0.2400_L24x72/conf<cid>/`，**262 组态**
（按实际存在扫描，五文件齐备判据；数据目录自 dev6 运行后由外部删减，
405 → 262，不假设网格）。每 conf 消费：
`corr_pp_P0/P2_<cid>.npy`（质子 2pt）+ `ops_mu{0,1|3,0|3,1}_dz24_<cid>.npz`
（胶子 OPE 单分量，Part C）；VdV/VVV/ope_combined 不消费。

## 运行

```bash
python main.py --debug                 # 前 5 组态 → 0_debug/（冒烟 ~1 min）
python main.py                         # 全量 262 组态 → v<ts>/（实测 4m35s，峰值 8.7 GB）
python verify_dev7.py <run_dir>        # 断言门 38 项（exit 0 = ALL PASS）
```

## 输出（v<ts>/）

- test6 型 7 图 + test0 型 3 图（含 **ratio_3pt_all_channels.png**：disconnected
  胶子 OPE/2pt 真空扣除比值 R(τ)，z∈{0,2,4,6}@t_sep=10 面板；非 perambulator
  连通 3pt，图题明注）
- 02 链产物 `L24x72/ratio_Pz2_Nsam*_dtmax20.npy` + 三窗口
  fit_*/{0_fit_data.npz,1_fit_report.txt,ratio.png,c0.png,chi2.png}
- 数据：corr2_{P0,P2}.npy + 1_fit_data.npz + 2_fit_report.txt +
  analysis_summary.json + run_verify.log
- 暂存 `input_stage/`（02 链布局：切片矩阵 + ops 符号链接，可删）

## 关键约定与结论

- Part A/B 照抄 dev6；Part C 布局整理照抄 logs/test8/main.py makedata 整理段
  （切片 C[sink,src]=C((sink−src) mod Nt)），调用配置照抄其 run_02_ratio。
- 拟合窗口取三代表窗 (6,11,2)/(7,11,3)/(9,11,4)（资源约束；test8 用六窗）。
- 实测结论：E0(P0)=1.143(A)/1.092(13)(B) GeV、E0(P2)=1.551(A)/1.515(42)(B) GeV；
  与 dev6 基线互差 5/3 MeV（A 型）；色散偏差 3.1%；c0(z≤4)=[0.011,0.047,0.020,
  −0.005,−0.001]±~0.02（与零一致——disconnected 通道需更多统计或方差缩减）。
- 小样本警示：--debug 下 c0 会被先验 (10,5) 主导（z=2..4≈10），不可用于物理解读。
- 分析报告：docs/analy_dev7_20260823.pdf（29 页，Overfull=0/Float=0/Missing=0）。
