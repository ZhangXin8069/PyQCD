# 关联函数预处理 reference

## 适用范围

本文件处理“组态级原始 C(t)”进入重采样和拟合前的确定性步骤：源时间平移、APBC
边界符号、介子折叠和有效质量。它不定义算符、不生成传播子，也不替代
`pyqcd-statistics` 的样本、协方差和 SVD 契约。

## 逐组态时间平移

源时刻会随组态改变。必须在任何 ensemble average 或 fold 之前，把每个组态移到共同
参考 `t_src=0`：

\[
t_{\rm phys}=(t_{\rm raw}-t_{\rm src})\bmod T.
\]

若时间轴是末轴：

```python
C_shifted = np.roll(C_raw, -t_src, axis=-1)
```

不同组态不能先平均再平移。对所有动量、方向、`t_sep` 和关联函数使用同一组态顺序，
并保存 `t_src[cid]`。

### APBC 符号

夸克时间方向通常取反周期边界。若传播子/收缩尚未把跨边界相位写出，则 `np.roll`
移到尾部的 `T-t_src` 个切片需要相应的 fermion-loop 边界符号；不能无条件套用固定
正号，也不能对已经吸收相位的 PyQUDA 收缩重复乘号：

```python
def shift_with_apbc_if_needed(C_raw, t_src, T, boundary_sign):
    C = np.roll(C_raw, -t_src, axis=-1)
    if t_src > 0:
        C[..., T - t_src:] *= boundary_sign
    return C
```

调用者必须显式传入推导得到的 `boundary_sign`；它由该通道包含的跨边界夸克线数和
收缩符号决定。应在单组态上检查 `C(T-1)` 平移前后。若上游明确已经施加 APBC，使用
普通 `np.roll`。

## 介子 fold 与重子不 fold

介子有偶数条价夸克线，有效时间边界为周期；在已完成逐组态平移后，可对每个组态折叠
前向与 backward：

\[
C_{\rm fold}(t)=\frac{C(t)+C(T-t)}{2},\qquad
t=0,\ldots,\left\lfloor T/2\right\rfloor .
\]

```python
Thalf = T // 2
C_fwd = C[..., :Thalf + 1]
C_bwd = np.concatenate(
    [C[..., :1], C[..., -1:Thalf - 1:-1]], axis=-1
)
C_fold = (C_fwd + C_bwd) / 2
```

必须先平移、后折叠。重子使用 \(P^+\) 时，backward 传播的是反宇称伙伴，不能把它
当作同一粒子的重复测量折叠；保留完整未折叠时间轴，并用 forward/backward 反宇称
谱式拟合。若只做早期 forward-window 诊断，可暂时不 fold，但必须明确截断边界。

## Log 与 cosh 有效质量

`pyqcd.analysis.meff` 支持 `log`、`cosh` 和 `GEVP`：

\[
m_{\rm log}(t)=\frac{1}{a}\log\frac{C(t)}{C(t+1)},
\]

\[
m_{\rm cosh}(t)=\frac{1}{a}\operatorname{arccosh}
\frac{C(t+1)+C(t-1)}{2C(t)}.
\]

- 对数型适合单指数主导且两端同号、为正的 forward window；噪声尾区必须 mask，
  不能把符号翻转点当成物理质量。
- cosh 型适合已正确折叠、以 \(\cosh[m(t-T/2)]\) 为主的周期通道；在调用前确认
  fold 的边界处理。
- 重子完整时间轴上出现 backward 反宇称态时，log 只是 forward-window 诊断，
  最终拟合应使用 `pyqcd-physics-spectrum` 的 forward/backward 模型。

现有 `run_meff_jackknife` 的 `{'proton': 'cosh', 'pion': 'log'}` 是历史通道配置，
不是“粒子名决定公式”的通用规则。新数据必须根据 fold、backward 态和平台稳定性
重新核对。

## 交付检查

至少随产物记录 `t_src[cid]`、是否施加 APBC 符号及符号来源、是否 fold、每个通道的
`meff_type`、窗口、单位和无效点 mask。合成数据应恢复已知质量；真实数据应给出
相邻窗口稳定性，不能仅凭某个窗口的数值宣称平台。

来源参考：`skills/LQCD_Skills/skills/lqcd-analysis/SKILL.md`，submodule 提交
`90bcbb8739c80b41a3965f41dab50bdfc9841b16`；公式和状态已按当前
`pyqcd.analysis.meff` 与本仓库分析入口复核。
