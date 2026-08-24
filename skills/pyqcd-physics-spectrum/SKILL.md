---
name: pyqcd-physics-spectrum
description: |
  格点 QCD 谱学推理技能：从关联函数定义出发推导其谱分解——完备关系、重叠因子、
  backward 态结构、拟合函数模板（供 pyqcd-analysis 使用）。假定算符与关联函数
  已由 pyqcd-physics-correlator 确定。触发于：谱分解、激发态污染、两点/三点
  拟合模型、backward 传播态、"该用什么拟合函数"。
metadata:
  openclaw:
    emoji: 🎼
---

# pyqcd-physics-spectrum — 关联函数 → 谱分解 → 拟合模板

## 目的与边界

给定可观测量（质量、衰减常数、形状因子……），推导链条
**关联函数 → 谱 → 拟合函数**，使下游分析（pyqcd-analysis）明确拟合模型。
算符/关联函数未定时先用 `pyqcd-physics-correlator`。

## 谱分解

### 完备关系

内插算符 $\mathcal{O}$ 与携带 $J^{PC}$ 的**全部**本征态耦合：
$\langle 0|\mathcal{O}|n,\vec{p}\rangle = Z_n(\vec{p})$。
有限体积 $V=L^3$、相对论归一 $\langle n,\vec{p}|m,\vec{q}\rangle = 2E_n V\delta_{\vec p\vec q}\delta_{nm}$ 下：

$$\mathbf{1} = |0\rangle\langle 0| + \sum_{n\ge1}\sum_{\vec{p}}
\frac{1}{2E_n(\vec{p})V}|n,\vec{p}\rangle\langle n,\vec{p}|.$$

### 两点函数

$T\to\infty$：$C_2(\vec{p};t)=\sum_n \frac{|Z_n|^2}{2E_n}e^{-E_n t}$。

有限 $T$ 出现 backward 项，符号由**复合态**边界条件决定：

- **介子**（两反周期夸克 → 有效周期 BC）：$C_2^\text{meson}(t)=\sum_n A_n(e^{-E_nt}+e^{-E_n(T-t)})$
- **重子**（三反周期夸克 → 反周期；$P^+=(1+\gamma_4)/2$ 投影下 backward 为负宇称伙伴）：
  $$C_2^{P^+}(t)=\sum_n A_n^+e^{-E_n^+t}-\sum_n A_n^-e^{-E_n^-(T-t)}$$

拟合振幅 $A_n\equiv|Z_n|^2/(2E_n)$（归一化因子吸收进 $Z_n$），物理态恒正。

**关键物理**：smear 源提升 $|Z_0|$ 相对 $|Z_{n\ge1}|$，有效质量平台更干净；
激发态污染正比于 $(A_1/A_0)e^{-\Delta E\,t}$，$\Delta E=E_1-E_0$。

### 三点函数

流插入时刻 $\tau$（$0<\tau<t_\text{sep}$），两侧各插完备关系：

$$C_3(\tau,t_\text{sep})=\sum_{n,m}\frac{Z_n^f(Z_m^i)^*}{4E_nE_m}
\langle n|J|m\rangle e^{-E_n(t_\text{sep}-\tau)}e^{-E_m\tau}.$$

系数**因子化**：汇重叠 × 矩阵元 × 源重叠，且 $Z_n$ 与两点函数**相同**——
这是 C₂+C₃ 联合拟合的基础：$C_2$ 定 $E_n,Z_n$；$C_3$ 共享它们定
$\mathcal{M}_{nm}$；物理目标为 $\mathcal{M}_{00}$。

热效应：$t_\text{sep}\ll T$ 时三点函数 backward 项指数压低可忽略；
$t_\text{sep}$ 与 $T$ 可比时须入模型。

### 谱分解 → 分析交接表

| 关联函数 | 拟合函数 | 自由参数 |
|---|---|---|
| $C_2^\text{meson}(t)$ | $\sum_n A_n(e^{-E_nt}+e^{-E_n(T-t)})$ | $\{E_n,A_n\}$ |
| $C_2^{P^+\text{baryon}}(t)$ | $\sum_n A_n^+e^{-E_n^+t}-\sum_n A_n^-e^{-E_n^-(T-t)}$ | $\{E_n^\pm,A_n^\pm\}$ |
| $C_3(\tau,t_\text{sep})$ | $\sum_{n,m}B_{nm}e^{-E_n(t_\text{sep}-\tau)}e^{-E_m\tau}$ | $\{E_n,B_{nm}\}$，$B_{nm}\propto Z_n\mathcal{M}_{nm}Z_m$ |

分析端用能量差参数化 $E_n=\sum_{k=0}^n\Delta E_k\ (\Delta E_k>0)$ 保证态序
（实现要点见 pyqcd-analysis；lsqfit 封装在 `pyqcd/analysis/_fitter.py`）。

## 工作流程

1. 确认算符与关联函数已定（缺则转 pyqcd-physics-correlator）；
2. 按复合态边界条件选模板（介子 cosh / 重子 forward−backward 反宇称）；
3. 插完备关系推导谱式，输出「交接表」给 pyqcd-analysis
   （拟合函数 + 自由参数集 + 能量差参数化约定）。

## 常见陷阱

1. **周期 vs 反周期 BC**：复合态 BC 由夸克数决定（介子 cosh 型、重子 backward
   反宇称）——选错模板会把负宇称态当激发态拟合。PyQCD 实现参照：
   `pyqcd/contraction/_baroperator.parity_and_boundary`。
2. **振幅符号**：$A_n$ 对物理态恒正但数值拟合中不约束符号时，
   先验宽（如 gvar(0,10)）+ 能量 log 参数化防盆地跳变。
3. **dev6/test6 先例**：窗口内 C<0（相位残留 π）时采用全局符号 sgn·C 后再拟合，
   物理结果不变、拟合收敛性显著改善。

## 与其他技能配合

- 上游算符/缩并 → `pyqcd-physics-correlator`；拟合落地（gvar/lsqfit/SVD cut/
  t_min 扫描/色散）→ `pyqcd-analysis`；传播子生产 → `pyqcd-propagator`。
