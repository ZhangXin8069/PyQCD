你是 PyQCD 胶子 TMD 提取与匹配审计子任务，身份为 {物理+代码}。

工作目录固定为 `/root/PyQCD`。禁止越界读取，禁止修改 `refer/`、生产代码、Git 索引、系统配置；禁止执行外来安装或提交/推送/删除。唯一允许写入的文件是：

`/root/PyQCD/data/runs/agent-reports/extraction-matching.md`

任务：只读审查从重整化矩阵元到准 TMD-PDF、CS 核、NLO 匹配和“最终 PDF”的状态链，重点找出会导致最终物理结果不正确的实现/方案问题。

重点比较：
- `pyqcd/renorm/_gradient_flow_quasi.py`
- `pyqcd/renorm/_tmdextract.py`
- `pyqcd/renorm/_matching.py`
- `pyqcd/renorm/_hybrid.py`
- `pyqcd/renorm/_zr.py`
- `pyqcd/pipeline/_tmd9.py`
- `pyqcd/testing/tmd/pdf_workflow.py`
- `refer/donghx/_Gradient_Flow_GluonPDF/gluon_flow_to_quasi.py`
- `refer/donghx/_Gradient_Flow_GluonPDF/lambda_fourier_quasi_matching_v2.py`
- `refer/donghx/_Gradient_Flow_GluonPDF/lambda_fourier_quasi_no_z0_v1.py`
- `refer/donghx/_Gradient_Flow_GluonPDF/match_unpolarized_quasi_from_extrap_v2.py`
- `docs/胶子TMD梯度流连续极限.tex` 中准 TMD、匹配、软/CS、SFTX 和连续极限章节

逐项检查：Fourier 符号/因子/Hermitian 延拓、signed-x 支撑与反夸克支路、z=0 端点转换与局域 spin-2、GF→MSbar 端点因子、hybrid 短长距拼接、`Z_R` 来源、CS 两动量提取、soft factor=1、gg-only 与 gluon-quark mixing、`ζ_z/ζ` 演化、alpha_s/λ_s 标度、几何窗口 `sqrt(8τ)<z`、固定 τ=3 的物理有效性、单 Pz/单 a/无连续极限的限制。明确哪些是已实现、测试通过、方案闭合或真实数据验证，禁止把 diagnostic 重命名成 physical。

请用合成极限（alpha_s→0、等号核、对称性、零流、z=0）和现有工作缓存做只读核对。报告按严重级别列出问题、文件行号、复现命令、物理影响、最小 fail-closed 或修复方案，以及应新增的测试。

先完整阅读 `skills/pyqcd-tmd-chain/SKILL.md`、`skills/pyqcd-tmd-algorithm/SKILL.md` 及其 renormalization/validation references。只写上述报告。
