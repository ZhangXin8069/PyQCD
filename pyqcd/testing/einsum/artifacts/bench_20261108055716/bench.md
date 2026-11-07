# qcd-einsum vs opt_einsum 3x 硬门禁基准

## 结论

- 状态：**NON_ACCEPTANCE**
- acceptance evaluated：True
- provisional：True
- 说明：目标 L24x64/ms0.2310 数据集不存在；本次使用 conf6250/L24x72 真实 perambulator，结论标记 provisional
- 阻断：RuntimeError: 固定 opt_einsum 子模块不可用；如仅需非冻结诊断，显式设置 PYQCD_EINSUM_ORACLE=installed
- 阻断：验收缺少 profiles: ['T1', 'T2']
- 阻断：验收缺少 equations: ['E1', 'E2', 'E4', 'S1', 'S2', 'S3']
- 阻断：验收缺少 backends: ['gpu']
- 阻断：没有可评估的基准行
- 阻断：请求的 backend 未测成功: ['cpu']
- 阻断：没有运行 T1/T2 生产规模 profile，不能作为性能验收
- 阻断：oracle is not the frozen opt_einsum submodule
- 阻断：benchmark uses a proxy dataset rather than the requested target

## 口径

- dtype：`complex128`
- threads：`4`（numpy 导入前强制）
- optimize：`dp`
- isolation：`process`
- acceptance profiles：`['T1', 'T2']`；diagnostic profiles：`['T0', 'T0n']`
- 计时：预热后 PROCESS_ISOLATED，每实现不少于 10 次，取中位数
- planner：`production`（双方各自规划一次，只计时稳态执行）
- 可比性门槛：双方中位耗时均 ≥ `0.1` s，否则记 `too_short` 且不计入 3x 判定
- 正确性：`max |actual-expected| / max(|expected|, tiny)` ≤ `1e-12`
- GPU：每次计时前后调用 default stream synchronize

## 环境

- oracle：error；version=`unknown`；frozen=False
- oracle path：`RuntimeError: 固定 opt_einsum 子模块不可用；如仅需非冻结诊断，显式设置 PYQCD_EINSUM_ORACLE=installed`
- pyqcd.einsum：`unknown`；native=`unknown`
- pyqcd path：``
- Python：`3.12.4`；NumPy：`2.5.4`
- CPU：`8` logical CPUs
- cpu device：`{'type': 'cpu', 'name': 'x86_64', 'threads': 4}`

## 结果

| profile | eq | backend | isolation | worker pids | optimize | planner | opt median/s | qcd median/s | speedup | max rel err | planner path match | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
