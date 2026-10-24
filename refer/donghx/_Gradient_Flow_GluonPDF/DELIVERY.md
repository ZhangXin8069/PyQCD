# L32x64 B方案固定 dF 试算交付

日期：2026-10-20。会话标题：固定基础表示线质量减除与quark短距离残余质量入口。

## 已实际完成

CPU作业3861825（合成测试）、3861826（真实数据减除/绘图）、3861827（独立审计）
均在cpueicc/cpu009以COMPLETED/0:0完成，分别耗时14、56、3秒。
以`audit/complete.json`为计算验收入口，不能仅凭PDF存在判断完成。

- a=0.0897fm，B方案，20 conf，原100组bootstrap（seed1115）不变。
- 7个tau=0.2..0.8；unpolarized/helicity/transversity × connected u/d/u-d；n_z=3,4,5,6。
- 63个line-only NPZ，10584点，0缺失；全中心值和100replica经独立物理单位公式核验。
- 非代数零且从未fallback的点555；至少一次fallback的点9777；odd z=0代数零252。
  这些标签全部来自上游，未因减除而改变，不能把“无缺失”说成“质量均通过”。
- 线性减除因子仅施加一次，z=0完全不变。最大因子370.786114（tau=.2，z/a=20），
  绝对误差同倍放大，固定dF条件下的相对误差不因此改变。
- 原13PDF/31页已渲染并核对；短标签修订和额外小z视图位于`presentation_v2/`，
  其完成状态单独看`presentation_v2/complete.json`，不替换或修改冻结生产结果。
- 排版作业3861832也在cpueicc/cpu009以COMPLETED/0:0完成，耗时47秒，
  回执PASS，生成15PDF/33页；已人工查看unpolarized全z/小z、transversity全z、
  helicity前后对比的渲染，图例和y标签未与相邻子图重叠。

## 主要结果地址

推荐查看`presentation_v2/figures/`中的版本（不同tau横坐标偏移，无连接线，
不同动量分行，实/虚分列；短y标签避免相邻行文字重叠）。

| 通道 | 全flow u-d | 减除前/后u-d（每tau一页） | 小z u-d（仅视图，不是拟合） |
|---|---|---|---|
| unpolarized | `presentation_v2/figures/line_subtracted_unpolarized_u_minus_d_allflow.pdf` | `presentation_v2/figures/before_after_unpolarized_u_minus_d.pdf` | `presentation_v2/figures/line_subtracted_unpolarized_u_minus_d_small_z.pdf` |
| helicity | `presentation_v2/figures/line_subtracted_helicity_u_minus_d_allflow.pdf` | `presentation_v2/figures/before_after_helicity_u_minus_d.pdf` | `presentation_v2/figures/line_subtracted_helicity_u_minus_d_small_z.pdf` |
| transversity | `presentation_v2/figures/line_subtracted_transversity_u_minus_d_allflow.pdf` | `presentation_v2/figures/before_after_transversity_u_minus_d.pdf` | `presentation_v2/figures/line_subtracted_transversity_u_minus_d_small_z.pdf` |

u、d各自的全flow图在同目录，文件名将`u_minus_d`改成`u`、`d`。
原始减除因子图：`figures/line_subtraction_factors.pdf`。
表：`analysis/matrix_elements.csv`、`analysis/line_factors.csv`、`analysis/small_z_geometry_audit.csv`。
NPZ：`data/line_subtracted_tau0p20_unpolarized_u.npz`等63个。
入口/模式/统计/符号的完整定义见`README.md`；全部输入与产物SHA见`complete.json`。

## 结果不是 MSbar：尚未完成的部分

$$H_{\rm line}=e^{+d_F|z/a|/\sqrt{8\tau}}H_{\rm bare},\qquad d_F=0.374137.$$

这是用户指定候选值的**条件试算**。dF已含fundamental表示，不再乘CF或2；
原候选flow拟合chi2/dof=62.06299、Q=1.5493e-52，并非正式批准dF。
原`final_d_approved=false`状态未修改；dF误差/模型误差尚未传播。

`analysis/m0_status.json`明确：m0MSbar=null，real_data_fit_performed=false。
原因是只有非零Pz的核子bilocal，缺少同算符、同态的MSbar短距离目标；
尚未声明B方案端点归一化/有限GF转换和零flow处方，且小z与rF/z的窗口重叠不足。
不能用arXiv:2607.13024v1的伴随真空色电关联器PT公式替代quark目标。

`code/fit_m0_cpu.py`已实现有输入门的相关GLS拟合，
`code/m0_core.py`用已知m0=.37GeV的合成数据通过符号、单位、固定/浮动截距、
全部100replica恢复及非法输入拒绝测试；这不是实际物理m0测量。
符号为$\log(H_{\rm line}/H_{MS}^{\rm target})=b-m_0 z/(\hbar c)$，
最终乘$e^{-b}e^{+m_0z/(\hbar c)}$。

## 下一步需要作出的决定

1. 建议先准备同B方案、同20conf、同7flow的Pz=0 unpolarized参考（须用户授权补算）；
   helicity/transversity的Pz=0短距离OPE与归一化须分别验证，不能照搬向量通道。
   或提供独立验证的非零Pz quark OPE目标与矩，就不必把Pz=0视为唯一方法。
2. 固定端点归一化（ringed/Ward/明确替代处方）和有限GF转换，避免重复减线质量。
3. 单格距仅能做带cutoff诊断的零flow试算，不宣称连续极限。
4. 在全局登记的小z窗口拟合m0，保留PT阶次/尺度/窗口及dF模型不确定度；
   当前rf/z<.5且z<=.3fm的窗口至多1处，不能硬拟合b和m0两参数。
5. 所有新数值/绘图任务仍只交CPU；不自动提交未经授权的新Pz=0 GPU计算。

不能随意改动：B权重、20conf/100bootstrap索引、投影和gamma相位、线质量符号和单次计数、
候选dF的未批准状态、quality/fallback分层以及transversity外部符号待确认状态。
