# cmp1 对照单测映射表（PyQCD ↔ refer/sush/lqcddb & refer/donghx）

生成：~auto-all 20260825 · 运行入口 `python examples/pyqcd/cmp1/main.py --group all`

## 用例总览

| 组 | id | 功能 | 参照 | pyqcd | 结果/说明 |
|---|---|---|---|---|---|
| lqcddb | L01–L03 | γ 表 / σ 与 p·σ / Levi-Civita | constant/*, base | lattice/_gamma,_sigma, tools/_base | 逐位 0 差 |
| lqcddb | L04 | 动量壳列表（立方壳+fix_Q2+only_g0） | base_functions.creat_mom_list | tools/_base | **本轮修复对齐**（原缺立方壳/only_g0） |
| L05/L23 | cached_contract(+clear/get_keys)/ArraySlicer | base | tools/_base | 逐位 0 差；get_* 为**本轮补充** |
| L06/L07 | Wick 收缩+等价图识别 | autowick | _autowick | 逐位；pyqcd 快 ~50× |
| L08 | seq_peram（真实 peram） | seqperam | _seqperam | 逐位 |
| L09–L22 | Jackknife/meff/Mom2GeV/GEVP/loop_tsrc/ratio_3pt/dis_connect(PDF) 等 | analyse | analysis/_analyse | 见下方"修复与差异" |
| L24 | 算符共轭/转置/C 对称/diquark | baroperator | _baroperator | 逐位 |
| L25 | Stout 涂抹（真实组态） | smear_gauge | smear/_stout | 幅值一致；逐位 O(1) 差异→backlog |
| L26–L28 | 本征模基元/V1(I,B)/V2-V4 结构 | eigvectors/vector | vertex/_eigcompress | V1 参数映射 ref(N_eigen,N_sum)≡pq(N_sum) 逐位 |
| L29 | 相位/Mom_VdV/Mom_VVV/sink2src | eigvectors/vertex | vertex/_vertex | VdV 逐位；**Mom_VVV 本轮重写为参照算法** |
| L30 | Wick 图出图 | figure | _wickplot | B9 视觉等价（结构性） |
| donghx | D01/D02 | DR γ(cupy) / ASCII IO | gamma_DR, input_output_4_cupy | lattice, tools/_io | 逐位 |
| D03/D04 | Clover F 全叠 / F̃ 全叠(μ<ν) | Operator.py | operator/_gluon_ope | F 逐位；F̃ 存在固定约定差（见下） |
| D05/D07/S09 | ΔG 双场强 ±z×平面/全和、FF 无 Wilson 线、unpol F·F 开关 | Operator.py / Calc_ope_unpol | _helicity, _gluon_ope(second_insert) | 形状/接口契约达成；数值受 F̃ 约定差传导→backlog |
| D06 | Lorentz 指派表四模式 | Calc_ope_* rank 分派 | get_ope_lorentz_pairs | 一致 |
| D08 | Mom_VVV（Nev=24） | Calc_VVV 核 | Mom_VVV_sink_t | 重写后与 ref 同式 |

## 本轮修复的 pyqcd 缺陷（由对照单测发现）

1. `lattice/_sigma.py`：`from .base_functions import …` 错误相对导入 → ModuleNotFound。
2. `contraction/_autowick.py`：`from .baroperator import …` 同类错误。
3. `tools/_base.creat_mom_list`：语义缺失（立方壳枚举/add_negative_signs/only_g0），按参照重写。
4. `analysis/_analyse.loop_tsrc`：ArraySlicer squeeze/broadcast 失配，≥5 维输入崩溃 → 纯 numpy 重写（与 ref 逐位一致，且 ~8× 提速）。
5. `analysis/_analyse.dis_connect`：原"补全"未复刻参照 reshape 平坦重解释语义；按参照逐行镜像（PDF 逐位一致；PFF 装配窗口语义登记差异）。
6. `vertex/_vertex.Mom_VVV_sink_t`：原为简化实现（单点直接收缩，非参照 dir 循环六置换）→ 忠实重写。

## 登记的差异（有意或待查）

| 项 | 差异 | 处置 |
|---|---|---|
| fm2GeV | pyqcd=0.1973269804(ħc)，ref=0.197 | 有意精度提升；meff/Mom2GeV 呈恒定比例 0.998343，容差通过 |
| meff cosh clamp | pyqcd 加 arccosh 定义域保护 | 有意增强；log 支路不受影响 |
| dis_connect PFF | ref 装配依赖 reshape 平坦重解释副作用 | pyqcd 按文档意图实现，实测差异登记 |
| F̃ 约定 | ref plaquette_clover_all_tilde 与 compute_dual_field_strength 轴序/符号存在固定线性关系 | D04 以候选关系判定锁定；下游 D05/D07 数值传导→optim/backlog |
| stout 逐位 | 根因闭合（生产形状喂入）：参照需 (dir,z,y,x,t,c,c) 7D 输入；单例 t 假轴使 nu=0 staple 滚动失效是此前全部 O(1) 差异来源。修正喂入后 S10 曾达逐位一致；全量混跑下仍有非确定 inf（arccos 域外 NaN 传播，疑 einsum 缓存跨用例交互）| S10 结构性登记 backlog；pyqcd 默认路径性质由 conftest 保证 |
| unpol F·F(S09) | rel≈2.33：U† 链 roll 方向疑异 | second_insert 接口已落地；backlog |
| MPI 层 25 项 / contractadviser 完整版 | 范式不同（pyqcd.parallel 元任务）／Roofline 顾问 | 维持既有判定：替代性覆盖，不移植；核心 FLOPs 诊断已内嵌(B9) |
| inner_product | ref 逐点 (Nc,Nc) 外积 vs pyqcd Nc 内积 | 语义分歧登记 |

## 性能摘要（CPU 单机、单次采样，详见 results.json t_ref/t_pq）

显著更快：Wick 引擎(~50×)、loop_tsrc(~8×)、readin_eigvecs(~1.8×)、check_files_existence 等。
~~Mom_VVV 慢~~ 已向量化：35s→1.4s 与 ref 同量级（Nev=32，einsum 切片循环向量化空间大）。
