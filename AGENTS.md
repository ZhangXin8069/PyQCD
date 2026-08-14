# AGENTS.md — PyQCD

**PyQCD By ZhangXin**：格点 QCD 研究仓库（lattice-pdf 迁移），核心目标为
**计算使用梯度流重整化方案的核子中的胶子 TMD-PDF**。

## 概要命令

```bash
source ./env.sh                      # 环境（若存在）
python examples/pyqcd/conftest.py    # 全量测试（7 项：γ/重整化/梯度流/TMD/匹配/混合/TMD提取链）
python examples/pyqcd/verify_consistency.py   # 一致性验证（vs docker-v20260805 输出，A–E 全 0 差异）
python examples/pyqcd/tmd_gradient_flow_demo.py   # 梯度流 TMD 全链示例
cd docs && xelatex <文档>.tex        # 编译中文 LaTeX 文档（xelatex，两遍）
```

## 目录结构

| 目录 | 内容 |
|---|---|
| `pyqcd/` | 主包（lattice/tools/vertex/contraction/operator/analysis/renorm/pipeline/testing） |
| `examples/` | 成功实例（docker-v20260805 基线）+ pyqcd 规范示例/测试 |
| `docs/` | 47 篇中文 LaTeX 笔记（xelatex 编译，文件名统一中文） |
| `refer/` | 参考代码/文献（zengch/donghx/huangcl/sush/zhangxin/papers/books），只读 |
| `logs/` | 按 tag 归档产物（stab0/ 等） |
| `cpp/` | C++ 后端占位 |

## 核心物理链（pyqcd/renorm）

1. **梯度流**（`_gradient_flow.py`）：Wilson flow（Luescher 2010），RK3 积分，
   τ=3a² 方案（Monahan–Orginos 2017 / NieMiera 2025）。
2. **胶子 TMD 算符**（`operator/_gluon_ope.py` + `renorm/_tmd.py`）：
   Clover F_μν、对偶 F̃、staple Wilson 线、组合 O = M^{tx;tx}+M^{ty;ty}−2M^{xy;xy}。
3. **自重整化**（`_zr.py`）：Z_R 参数化与全局拟合（arXiv:2510.17758 Eq.3-8）。
4. **混合方案**（`_hybrid.py`）：短距比值 + 长距 Z_R，λ 外推，傅里叶→准 PDF。
5. **NLO 匹配**（`_matching.py`）：胶子单圈匹配核 g_0..g_3。
6. **连续极限**（`_extrapolate.py`）：a/Pz/mπ/L 联合外推。

## 关键约定

- 张量布局：gauge `(Nt,Nz,Ny,Nx,4,3,3)`；γ 矩阵 DeGrand-Rossi 基。
- 后端：numpy/cupy 切换（`pyqcd.tools.set_backend`）。
- 编译：docs 与 logs 的 tex 一律 xelatex（中文）；`\quad` 后跟中文需空格。
- 测试：无 pytest 框架依赖，examples/pyqcd/conftest.py 直接运行。
- refer/ 只读参考：pyqcd 逻辑照抄但不 import。
- git tag 约定：stab<N>/dev<N>/bug<N>/test<N>（当前 stab0）。

## 反模式

- 不 import refer/、不 import examples/（照抄逻辑，自包含）。
- 不修改 refer/ 与成功实例基线的"已验证物理结论"（pn 2pt=0、meff≈1.12 GeV 等）。
