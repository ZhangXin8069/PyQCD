# donghx 4150 格点 QCD 算法复现与对照设计

## 目标

在 `/root/PyQCD` 中完整梳理并尽可能复现 `/root/PyQCD/refer/donghx` 的格点 QCD
计算链，以组态 `4150` 为核心样本，对 eigvec、VVV/VdV、2pt、OPE、3pt、ratio 和
barematrix 的中间结果、最终结果、约定和性能进行可追溯对照；生成中文研究报告，
并把已验证的代码与证据提交、标记和推送。

## 范围与不可变约束

- 工作目录为 `/root/PyQCD`；参考代码目录 `/root/PyQCD/refer/donghx` 只读。
- 只读取用户明确引入的 `/public/group/lqcd/...` 数据和参考结果；默认写入
  `/root/PyQCD/data` 或版本化的 `examples/pyqcd`/`logs` 产物目录。
- 不以参考结果的“看起来合理”为算法证明；每个数值差异必须经过 shape、轴序、
  dtype、相位/符号、归一化和不变量检查后再分类。
- 复数计算保持复数；只有由明确投影、共轭关系或参考定义支持时才取实部。
- 不删除数据、不改写已推送历史、不修改系统配置；提交和推送仅使用本任务明确授权
  的普通 commit、annotated tag 和远端更新。
- TMD 属于进阶项：核心链未闭合或缺少物理所需的 `z/b_perp/staple/flow` 元数据时，
  只报告已有实现和限制，不把准 TMD 输出冒充完整物理 TMD-PDF。

## 物理与数据契约

### 共同布局

- 规范链接布局固定为 `(Nt,Nz,Ny,Nx,4,3,3)`，空间轴为 `z,y,x`，颜色矩阵最后两轴。
- eigvec 二进制按实部/虚部交错的 float64 读取，再形成复数
  `(Nev, Nsite, Nc)`；时间片批量时显式记录时间轴。
- VdV 是两个蒸馏向量与动量相位的颜色收缩；VVV 是三个蒸馏向量经
  Levi-Civita 颜色收缩和六置换组合的重子顶点。任何相位使用都记录 Fourier 符号。
- perambulator 的源/汇时间、spin、eigenvector 轴必须在进入 einsum 前写入清单；
  输出不因数组相等而丢失 flavor、smear 或方向标签。

### 参考算法主链

1. 从 Laplacian eigensystem 读取或求得低本征模，并按参考约定进行归一化/相位处理。
2. 用动量相位与 eigvec 构造 VdV/VVV；需要规范输运时显式加入链接路径。
3. 读取 light perambulator，按照 Wick 配对、颜色/自旋收缩和宇称投影构造质子 2pt；
   3pt 使用固定 sink 的顺序源或参考代码提供的等价收缩。
4. 从 raw、3D smear 和 4D smear 规范场计算 Clover 场强、对偶场强和 Wilson 线，
   构造 OPE；固定规范 FF 算符作为独立分支保存。
5. 用 3pt/2pt 形成 ratio，按 z、方向、动量和 `t_sep` 拟合/平均得到 bare matrix
   element；统计层记录 resampling、协方差、SVD 和窗口。
6. 只有上面主链的输入、流时间和非局部几何足够时，才继续准 PDF/TMD 提取。

### smear 与动量覆盖

至少区分以下标签，而不是把它们合并为一个“smear”参数：

- 规范场：3D 1、3、5 次，以及 4D 10 次；
- 动量涂抹：x/y/z 方向和实际动量幅度（至少覆盖参考 4150 目录中存在的幅度）；
- 2pt 算符：`Cg5`、`Cg5g4` 及参考产物中出现的 polarisation/contract 标签。

## 子项目与边界

### A. 算法清单与资产清单

扫描参考代码中的函数、脚本入口、stdin 参数、文件命名和输出 shape；扫描 4150 的
eigvec、peram、gauge、hpy、2pt、OPE、Contraction 资产。产物是机器可读 manifest、
源码行号映射和“可真实比较/只有代码/缺少输入”的证据分级。

### B. 低层对象

按以下顺序真实运行：eigvec reader/正交性 → phase → VdV/VVV → Clover/dual/Wilson
line/OPE。每一步保留输入摘要、输出数组或摘要 hash、耗时、峰值资源和不变量；
对参考现成数组使用相位等价、共轭等价或逐元素误差的明确比较器。

### C. 费米子关联函数

以 `perambulator + VVV/VdV` 为输入，先做最小 `Nev`/单方向 smoke，再做 4150 的
参考组合。2pt 和 3pt 的每个输出都带 `conf_id`、smear、momentum、direction、
operator、source/sink/time 轴；若参考目录只有脚本没有同配置成品，只验收结构和受控
数据，不强行制造“现成结果”。

### D. 分析与结果闭环

把 OPE/3pt/2pt 整理为 ratio 和 barematrix 输入，复现参考窗口与归一化；同时保留
未拟合数据、拟合参数、协方差和图表。差异先按整体相位、取实部、归一化、轴序、
统计方法、随机种子和真实算法差异依次排查。

### E. 报告与版本收尾

报告至少包含：物理公式链、代码映射、4150 输入/产物可见性、逐节点对照表、性能表、
差异闭环、已验证/推断/未验证边界、图表和后续工作。XeLaTeX 编译两遍，检查页数、
Overfull、Float too large、Missing character、全页渲染、裁切、遮挡和页脚安全区。

## 接口与证据格式

每个比较案例统一记录：

```json
{
  "case": "VdV|VVV|2pt|...",
  "conf_id": "4150",
  "variant": {"gauge_smear": "3D_3", "momentum": [0, 0, 2], "direction": "z"},
  "input": [{"path": "...", "shape": [], "dtype": "..."}],
  "output": [{"path": "...", "shape": [], "dtype": "..."}],
  "comparison": {"metric": "rel_maxdiff", "value": 0.0, "tolerance": 1e-10},
  "timing": {"reference_s": null, "pyqcd_s": null},
  "status": "pass|diff|unverified|blocked",
  "evidence": "confirmed|inferred|unverified",
  "note": ""
}
```

`pass` 只能表示本次命令真实运行且比较器通过；`diff` 必须有可复现差异和根因/待查
假设；`unverified` 表示输入或参考成品不存在；`blocked` 只用于同一外部阻碍重复三轮
且没有安全替代路径的情形。

## 验收门

- 资产门：4150 的实际输入、参考结果和缺失项有新鲜 stat/count/shape 证据。
- 低层门：eigvec、phase、VdV/VVV、Clover/dual/OPE 各有至少一个真实运行和不变量。
- 费米子门：至少一条 2pt 和一条 3pt 真实链完成；所有存在参考数组的组合给出数值差异。
- 分析门：ratio/barematrix 的输入、拟合/平均和最终结果可以沿路径回放；统计差异不
  被误报为物理算法差异。
- 回归门：`python examples/pyqcd/conftest.py`、针对性对照测试和语法检查通过；
  失败项有明确分类。
- 文档门：报告两遍 XeLaTeX 成功，`Overfull=0`、`Float too large=0`、
  `Missing character=0`，且 expected/actual/rendered/checked 页数一致，视觉闸门全零。
- Git 门：`git diff --check` 通过；提交包含源码、测试、证据和报告；annotated tag
  的 peeled object 等于提交 HEAD；按授权推送 commit 与 tag，并读取远端确认。

## 默认迭代策略

采用 `all` 的最多 5 轮和“一轮一个主导问题”：先解决数据/接口/算法错误，再处理
性能或报告质量；每轮用新产物和命令输出更新 manifest。收益低于 5%、连续两轮无实质
变化或所有通过门达成时收敛。任何外部权限、远端缺失或超时都降低对应证据等级，不能
用合成数据替代真实 4150 结果。
