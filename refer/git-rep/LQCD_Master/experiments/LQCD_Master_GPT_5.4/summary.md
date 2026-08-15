# LQCDMaster × GPT-5.4 实验汇总

**Backbone:** GPT-5.4 | **Endpoint:** gpugeek.com | **Ensemble:** C24P29, cfg 10000

---

## 综合结果

| Observable         | Run               | Tasks | Exact | Mismatch | Failure | Acc.      |
| ------------------ | ----------------- | ----- | ----- | -------- | ------- | --------- |
| 2pt local          | `20260607_145349` | 20    | 20    | —        | 0       | **100%**  |
| 2pt nonlocal       | `20260607_183958` | 10    | 8     | 2¹       | 0       | **100%**  |
| wilson loop        | `20260606_193545` | 12    | 12    | 0        | 0       | **100%**  |
| 3pt meson          | `20260608_080509` | 13    | 13    | 0        | 0       | **100%**  |
| 3pt baryon (Run B) | `20260608_123913` | 15    | 10    | 1        | 4       | **73.3%** |
| **Total**          |                   | **70** | **63** | **3**    | **4**   | **90.0%** |

> ¹ π⁺ 和 K 的整体符号翻转，γ₅ Hermiticity 相位约定，物理量不受影响。

### Baryon 失败详情

| Task | Process                     | 问题         |
| :--: | --------------------------- | ------------ |
|  1   | proton→proton u→u γx vector | 代码生成错误 |
|  3   | Λ→Λ s→s γx vector           | 代码生成错误 |
|  4   | Λ→Λ s→s γxγ₅ axial          | 代码生成错误 |
|  8   | Λb→Λc b→c γx vector         | 代码生成错误 |
|  5   | Λ→p s→u γx vector           | ✅ sign flip |

### Timing

| Category         | Tasks | Wall      | GPU      | Avg/task |
| ---------------- | ----- | --------- | -------- | -------- |
| 2pt local        | 20    | 80.8 min  | 14.2 min | 242s     |
| 2pt nonlocal     | 10    | 49.6 min  | 5.1 min  | 298s     |
| wilson loop      | 12    | 42.5 min  | 1.3 min  | 212s     |
| 3pt meson        | 13    | 81.1 min  | 10.7 min | 374s     |
| 3pt baryon Run B | 15    | 137.0 min | 17.6 min | 548s     |
