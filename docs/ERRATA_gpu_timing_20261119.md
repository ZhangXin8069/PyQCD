# 勘误：0.5.9 之前的 GPU 耗时全部作废（2026-11-19）

本页是**历史文档的勘误索引**。凡是在 0.5.9 之前记录的 GPU 耗时、GPU 倍率与
GPU 相对比较，一律**不再成立**，不得作为证据引用。原因有两条，都是库自身的缺陷。

## 1. 计时没有设备同步 → 量到的是"入队时间"

`cupy` / `torch(cuda)` 的收缩是**异步下发**：`contract()` 返回时 GEMM 可能还在
队列里，而 `time.perf_counter()` 已经停表。0.5.9 之前 `pyqcd.einsum.test._timed`
与 `_tune_worker` 的计时都**没有**在计时区前后同步设备，于是量到的是"把命令排进
队列要多久"。

同机同操作数的实测对照（本机 RTX 4060 Laptop / cupy 14.2，2026-11-19）：

| 场景 | 不同步 | 同步后 |
|---|---|---|
| 单次 `500×500` complex128 `cupy.matmul` | 0.106 ms | 3.017 ms |
| 用户生产方程（10 算子 / 2.56e8 输入元素） | 脚本里 ~900 ms | tune 表里写着 **0.32 ms** |

那个 `0.32 ms` 还被 `--tune` 当成 `best` 写进了配置表——数字小得离谱，却没有任何
地方会因此报错。**这是静默失真最坏的一类：报告看起来完全正常。**

## 2. `--compare opt_einsum` 只走了 numpy

`oe.contract` 的 `backend` 缺省是 `'auto'`，按**数组类型**选后端；旧实现只把
`host`（numpy 副本）交给它。于是 GPU 上跑 `--compare opt_einsum` 时，"参照"其实
在 CPU 上算，与主路径的 GPU 秒数并列展示，两个数**不可比**。

## 修复（0.5.9 起）

| 项 | 现状 |
|---|---|
| 设备同步 | 新增 `pyqcd/einsum/_sync.py`；`test._timed` 与 `_tune_worker` 在**每个样本前后**各同步一次，CPU 上是空操作 |
| 参照后端 | `--compare opt_einsum` 按操作数类型传 `backend=`，参照行打印 `backend='cupy'` 等；不支持就如实报错，绝不悄悄退回 numpy |
| 冷启动 | 加速设备把 `--warmup` 抬到 ≥2（`ACCELERATED_MIN_WARMUP`），报告给 `warmup_effective` / `warmup_raised` 与一行 `time note:` |
| 可观测证据 | 引用 GPU 耗时时，输出里**必须**同时有 `backend=` 与 `warmup=2`；`--json` 里有 `warmup_effective` |

## 受影响的历史文档（本页统一引用，逐文件顶部已加一行指向这里）

| 文档 | 受影响的结论类型 |
|---|---|
| `einsum_production_optimization_20261112.md` | GPU 特化执行器的端到端耗时与加速比 |
| `report_einsum_benchmark_20261109.md` | 与 opt_einsum 的 GPU 对照倍率 |
| `report_matx_einsums_bridge_20261110.md` | 与 MatX/其他桥接实现的 GPU 倍率 |
| `pyqcd_einsum_performance_status_20261108.md` | E1/E2 四方实现的 GPU 耗时区间 |
| `pyqcd_einsum_gpu_l24_20261108.md` | L24 真实组态的 GPU 计时 |
| `pyqcd_einsum_overall_20261108.md` | E4/E5 的 GPU 加速比 |
| `pyqcd_einsum_progress_20261108*.md` | 各轮 GPU 数字与"是否打平"的判定 |
| `pyqcd_einsum_agents_20261108.md` | e4_fused 的 GPU 0.635 ms / 16.7× |
| `pyqcd_einsum_kernels_20261108.md` | 各 kernel 的 GPU 耗时表 |
| `pyqcd_einsum_bugs_20261108.md` | "L24 E4 的 4.99 s 是预热口径问题"（结论方向对，数字作废） |
| `pypi_release_audit_20261108.md` | 引用 GPU 数字的发布判定 |

## 没有被作废的部分

- **CPU 计时**：`perf_counter` 对同步执行本来就是真墙钟，CPU 数字照旧可用。
- **数值结论**：`rel`、形状、dtype、是否命中特化形态——全都与计时无关，照旧成立。
- **相对快慢的定性方向**（例如"cupy 自己的 einsum 比自研执行器快"）多数仍然成立，
  但**不得再引用其倍率**，要用就在 0.5.9 之后重测。

## 教训：怎样避免再留一个无出处的数字

本页的存在是因为 0.5.9 之前的 GPU 耗时全是**入队时间**，而当时的注释里还留着
`8.9 s` / `0.39 s` / `23×` 三个**仓库内无对应记录**的数字（与日志实测 5.1–5.2 s
也对不上）。可执行的检查方法：

```bash
# ① 注释/docstring 里出现的耗时/倍率，逐个回到日志或报告里找出处
grep -rn -E "[0-9]+\.[0-9]+ ?(s|ms)\b|[0-9]+(\.[0-9]+)?×" pyqcd/einsum/*.py \
  | grep -v -E "测|实测|实测为|logs/|docs/" | head -40

# ② 每个"实测"字样都必须能落到仓库里真实存在的文件
grep -rn "实测" pyqcd/einsum/*.py | grep -oE "logs/[A-Za-z0-9_.-]+|docs/[A-Za-z0-9_.-]+" \
  | sort -u | while read -r f; do [ -e "$f" ] || echo "MISSING EVIDENCE: $f"; done
```

判据：一个性能数字要么带**可追溯的路径**（`logs/v202611151120.txt:45` 这种），要么
写 `unverified`。做不到就删掉那个数字——"看起来合理"不是证据。

同一条规则适用于**能力**描述：`_probe.py` 曾把六族不同的失败压成一句写死的
"2x2 matmul disagreed with numpy"，用户据此去重建原生库却修不掉。诊断必须给出
**真实阶段与真实异常**（`stage=native: NativeUnavailable: ...`），不许用一句
听起来合理的固定措辞代替。
