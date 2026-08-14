# AGENTS.md — pyqcd

**PyQCD 主包**：格点 QCD 蒸馏管线 + 胶子 OPE + 梯度流重整化 TMD-PDF 计算库。
架构参考 /root/PyQCU（子包 + `_` 前缀私有模块 + 显式 re-export），
内容照抄自 examples/docker-v20260805（成功实例）与 refer/（逻辑参考，不 import）。

## 核心目标

**计算使用梯度流重整化方案的核子中的胶子 TMD-PDF**：
裸矩阵元（蒸馏 2pt + 胶子 OPE staple 算符）→ Wilson flow 梯度流涂抹 →
混合方案/自重整化 Z_R → λ 外推 → 傅里叶 → NLO 匹配 → 连续极限外推。

## 子包

| 子包 | 内容 | 来源 |
|---|---|---|
| `lattice/` | 常数（Nc/fm2GeV）、DR 基 γ/σ 矩阵 | lib/constants,gamma,sigma |
| `tools/` | 后端切换（numpy/cupy）、缓存 einsum、切片、数据读取 | lib/backend,base,io_readers |
| `vertex/` | VdV/VVV 顶点、相位因子 | lib/vertex |
| `contraction/` | 自动 Wick、重子算符、seqperam、动态收缩 | lib/autowick,baroperator,seqperam,dynamic |
| `operator/` | Clover 场强 F、对偶 F̃、胶子 OPE 算符、.lime 读取、TMD staple 扩展 | compute_ope.py + 新写 |
| `analysis/` | Jackknife/Bootstrap/meff/ratio_3pt + disconnected(code_1)/meff/3pt 编排 | lib/analyse + analyze.py 逻辑 |
| `renorm/_tmdextract.py` | ★ 准 TMD-PDF/CS 核/SFTX 1 圈匹配 | 理论文档新写 |
| `renorm/` | ★ 自重整化 Z_R、混合方案、NLO 匹配、外推、梯度流、TMD 提取 | refer/zengch 逻辑移植 + 理论文档新写 |
| `pipeline/` | 集中配置 + 9 步管线调度（+tmd 步） | config.py/run_pipeline.py |
| `testing/` | 集成测试函数（examples/pyqcd/conftest.py 入口） | 新写 |

## 关键约定

- 张量布局：gauge `(Nt,Nz,Ny,Nx,4,3,3)`；链接/场强 `(…,3,3)`；维序 t,z,y,x。
- 后端：`from pyqcd.tools import set_backend/get_backend`；numpy/cupy 通用；
  禁止直接 import cupy 计算（仅 try/except 探测）。
- 梯度流：`wilson_flow(U, tau, eps=0.01)`（RK3，Luescher 2010）；
  流时间物理约定 τ=3a²（NieMiera 2025）。
- 重整化：z 单位 fm（内部转 GeV⁻¹ 用 fm_to_GeV=0.197）；μ=2 GeV 默认。
- 日志：`print` + `verbose` 参数；管线产物写 logs/（gitignore 豁免）。
- 测试：`python examples/pyqcd/conftest.py`（7 项）；一致性验证：
  `python examples/pyqcd/verify_consistency.py`（A–E 五组对照，全部 0 差异）。

## 反模式（勿重复）

- 不 import refer/ 或 examples/ 代码——逻辑照抄，自包含实现。
- 不做逐点 for 循环求逆/矩阵运算（批量 einsum）。
- 不改动 refer/（第三方参考）；examples/docker-v20260805 为成功实例基线。
