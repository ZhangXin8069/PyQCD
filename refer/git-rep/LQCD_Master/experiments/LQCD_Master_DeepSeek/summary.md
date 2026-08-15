# LQCDMaster × DeepSeek 实验汇总

**Backbone:** DeepSeek V4 Pro | **Endpoint:** api.deepseek.com | **Ensemble:** C24P29, cfg 10000

---

## 综合结果

| Observable   | Run               | Tasks | Exact | Mismatch | Failure | Acc.      |
| ------------ | ----------------- | ----- | ----- | -------- | ------- | --------- |
| 2pt local    | `20260706_034834` | 20    | 16    | 0        | 4       | **80.0%** |
| 2pt nonlocal | `20260706_093900` | 10    | 8     | 0        | 2       | **80.0%** |
| wilson loop  | `20260706_140501` | 12    | 12    | 0        | 0       | **100%**  |
| 3pt meson    | `20260701_040719` | 13    | 7     | 1¹       | 5       | **53.8%** |
| 3pt baryon   | `20260701_040733` | 15    | 13    | 1²       | 1       | **86.7%** |
| **Total**    |                   | **70** | **56** | **2**    | **12**  | **80.0%** |

> ¹ task 13 Bc→J/ψ Gx: max rel error = 2.000, 数值量级匹配、符号相反 — 纯符号翻转。  
> ² task 9 Λb→Λ b→s: |dRe|=1.75×10⁻¹⁶, |dIm|=1.07×10⁻¹⁶ 均为机器精度，仅 tau 对齐偏移导致被判定为 mismatch，物理结果正确。  
>
> 2pt 用 t=1 Re 比较；3pt 用 max relative error < 1×10⁻³。

---

## Failure 详情

### 2pt local

| Task | Particle | Standard t=1 Re |   DS Result    | 问题               |
| :--: | -------- | :-------------: | :------------: | ------------------ |
|  3   | ηₛ       |   +1.3067e-01   |  +4.1349e-02   | 量级偏差 ~3×       |
|  11  | Ξ⁻       |   +1.6109e-02   |  +1.6135e-02   | ~0.16% 偏差        |
|  13  | Σ⁻       |   +2.0465e-02   | **0.0000e+00** | 代码未产生有效结果 |
|  17  | Ω_c⁰     |   +1.5357e-02   | **0.0000e+00** | 代码未产生有效结果 |

### 2pt nonlocal

| Task | Particle | Standard t=1 Re |   DS Result    | 问题               |
| :--: | -------- | :-------------: | :------------: | ------------------ |
|  8   | J/ψ      |   +2.8368e-02   |  +7.2615e-03   | 量级偏差 ~4×       |
|  10  | Dₛ\*     |   +4.0202e-02   | **0.0000e+00** | 代码未产生有效结果 |

### 3pt meson

| Task | Process   | 问题                                     |
| :--: | --------- | ---------------------------------------- |
|  13  | Bc→J/ψ Gx | max rel err = 2.000, pure sign flip      |
|  3   | B→D Gx    | max rel err = 1.16, inferred_tau_offset  |
|  4   | B→K Gx    | max rel err = 1.15, inferred_tau_offset  |
|  5   | B→π Gx    | max rel err = 2.72                       |
|  10  | Bs→Ds Gx  | max rel err = 0.53, result_has_re_only   |
|  11  | K→π Gx    | max rel err = 0.87, inferred_tau_offset  |

### 3pt baryon

| Task | Process         | 问题                                                                   |
| :--: | --------------- | ---------------------------------------------------------------------- |
|  9   | Λb→Λ b→s vector | max rel err = 4.59, \|dRe\|=1.75×10⁻¹⁶ → 机器精度, tau 对齐伪影       |
|  11  | Ξ→Λ s→u vector  | max rel err = 0.87, \|dRe\|=1.64×10⁻⁸                                 |
