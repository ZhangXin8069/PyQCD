# TODO — test1 全量运行与一致性验证（v202608140630）

任务：调用 pyqcd 包实现与 docker-v20260805/output/output_20260802_120104 一致的结果
（10 组态全量蒸馏 GPU 管线），中间数据与图表完整保存，输出目录 v202608140630，
总体形式参考 logs/test12（main.py 只是调用 pyqcd）。

## 状态图例
- [x] 完成（含验证证据）
- [~] 进行中
- [ ] 待办

## 阶段 A：环境与实现（已完成）

- [x] A1. 预授权确认（auto Step 1，L2 全默认）— `.auto.2026-08-14-13-32-28.log`
- [x] A2. 调查基线 `docker-v20260805`（run_pipeline.py/compute_vertex/compute_contraction/
      compute_ope/analyze/report/config）与 pyqcd 包接口（vertex/contraction/operator/
      analysis/tools/lattice/pipeline）
- [x] A3. 确认数据路径 `/public/group/lqcd/` 可访问（10 组态 eigvec/peram/gauge 全 OK）、
      cupy 14.0.1 + GPU 8GB、xelatex 可用
- [x] A4. 关键工程约束：pyqcd.Mom_VVV_sink_t 单 einsum 在 Nev=100/Nx=24 下超时（8GB GPU）
      → main.py 采用基线 x-slicing 因子化（`_compute_vvv_single_t_gpu`，数学等价）
- [x] A5. 编写 `examples/test1/AGENTS.md`（复现与比对指南）
- [x] A6. 编写 `examples/test1/main.py`（只调用 pyqcd 包：env/pipeline/verify/collect/report）
- [x] A7. 编写 `examples/test1/run-local.sh`（版本目录 + env → pipeline → verify → collect → report）
- [x] A8. 语法检查：`py_compile` OK、`bash -n` OK

## 阶段 B：冒烟测试（单组态 6250，数值预检）（已完成）

- [x] B1. `env` 自检：10 组态数据 OK，GPU/CuPy OK
- [x] B2. vertex 预检：VdV/VVV 与基线 `conf6250` 逐位一致（max|diff|=0.0）
- [x] B3. 2pt 预检：corr_pp/pn/pion P0/P2 与基线逐位一致（pn=0 味守恒）
- [x] B4. 3pt 预检：proton/pion P0/P2 与基线逐位一致
- [x] B5. 4pt 预检：pjnnjnp 与基线逐位一致
- [x] B6. OPE 预检：ops_mu{0,1}/{3,0}/{3,1} + ope_combined 与基线逐位一致
- [x] B7. analysis/plots 路径验证：单组态统计为 nan 属预期（Nconf=1），
      修复 Nconf<2 时跳过 disconnected（协方差退化 LinAlgError 保护）

## 阶段 C：全量运行（10 组态，GPU）— 进行中

- [~] C1. 全量 pipeline：`output_20260814_145340`，PID 978（timeout 86400）
      - [~] vertex 10 组态（单组态 ~148s → 预计 ~25min）
      - [ ] 2pt 10 组态（单组态 ~626s → 预计 ~105min）
      - [ ] OPE 10 组态（含读 gauge，单组态 ~589s → 预计 ~100min）
      - [ ] 3pt 10 组态（单组态 ~1805s → 预计 ~300min）
      - [ ] 4pt 10 组态（单组态 ~740s → 预计 ~125min）
      - [ ] analysis（Jackknife/meff/ratio/不相连拟合）
      - [ ] plots（correlators/meff/ratio 图）
      - [ ] report（analysis_summary.json）
      - 注：与 test0 管线进程（PID 83061）共享 GPU，实际耗时可能上浮
- [ ] C2. 全量 verify vs 基线（rtol=1e-3/atol=1e-8，含 summary 标量）→ test1_verify.json
- [ ] C3. collect 汇总 → test1_results.json
- [ ] C4. report：physics_report.tex/.pdf（xelatex 两遍）

## 阶段 D：收敛评估与收尾

- [ ] D1. 收敛判据检查：verify 全 PASS + 产物清单与基线同构
- [ ] D2. 若 verify FAIL：debug 定位（重点检查 3pt/4pt 大数组与统计窗差异）
- [ ] D3. diff 复查（git diff --check + 文件清单）
- [ ] D4. auto 汇总报告 + `.auto.2026-08-14-13-32-28.log` 归档

## 验证基准（勿改）
基线 `examples/docker-v20260805/output/output_20260802_120104/analysis_summary.json`：
proton_P0 E0=1.1183±0.0075、proton_P2 E0=1.5585、pion_P0 E0=0.2863、pion_P2 E0=1.1779 GeV。
verify 容差：rtol=1e-3 / atol=1e-8（fp32 合理水平）。
