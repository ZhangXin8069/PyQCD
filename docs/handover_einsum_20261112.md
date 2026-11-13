# einsum 任务线接手文档（2026-11-12）

> 读者：接手 `pyqcd.einsum` 的下一个会话 / agent。
> 本文只记**当前状态、已验证事实、坑、下一步**，不重复技能细节——
> 细节与判据以 `skills/pyqcd-einsum/SKILL.md` 为准，本文与它冲突时以实跑证据为准并修正技能卡。

## 1. 一句话状态

| 项 | 值 |
|---|---|
| 分支 / 提交 | `main` @ `1550aaa`（已推 `origin/main`） |
| 包版本 | `pyqcd 0.3.9` |
| 功能状态 | 匹配层（拓扑签名）+ 执行层（`kupl_dedicated` / `cuda_topology`）均已打通；固定 15 条与随机抽样都命中专用路径 |
| 回归 | `python -B -m pyqcd.testing` → **118 passed / 1 skipped / 0 failed**（skip 为 `physics.sty` 缺失，与本线无关） |
| 工作区 | 干净（无未提交、无未跟踪） |
| 本机资源 | 15 GB RAM / 32 GB swap；4 物理核 / 8 线程；GPU RTX 4060 Laptop（compute 8.9） |

`python -m pyqcd --gpu-probe` 实测：`native_available=True`、`cupy 14.2.0`、
`kernel status: ok — topology entry returned the reference result`、
`verdict: CUDA fast path is active`。

## 2. 代码地图

| 文件 | 职责 |
|---|---|
| `pyqcd/einsum/_specialize.py` | `canonical_signature`（全式首次出现重编号）、`match`（单段：签名→语料索引）、`lru_cache(512)` |
| `pyqcd/einsum/_topology_registry.py` | **生成物，勿手改**；`N_ENTRIES=8638`，按 `(元数, 槽位总数)` 分 24 桶，摘要截断到 96 位 |
| `pyqcd/einsum/_wick_tasklist.py` | 8638 条任务；主键是连续 `seq`（1…8638），不是 T 号/行号；含 `equation`/`roles`/`arity` |
| `pyqcd/einsum/_api.py` | `contract` 分派链：elementwise → cupy 委派 → 原生执行器 → Python 分步；专用链里 `kupl_dedicated` **优先于** `kupl_topology` |
| `pyqcd/einsum/_kernel_kupl.py` / `_kernel_cuda.py` | 原生适配层；`LAST_FAILURE` / `LAST_INFO`；Python 计划缓存 32 项 FIFO |
| `pyqcd/einsum/_bench_scene.py` | `full` / `scene` / `smoke` 三档维度，两个基准脚本共用 |
| `pyqcd/einsum/bench_specialized.py` | 特化路径基准；`--sample N --seed S` 抽样；`REF-FAIL` 逐行降级 |
| `pyqcd/einsum/bench_generic.py` | 通用路径基准（不命中语料） |
| `pyqcd/einsum/verify_specialized.py` | 单条方程的特化路径**纯查询**（生产包，随 pip 发布，不跑数值） |
| `cpp/kupl/einsum/src/dedicated_apply.cpp` | KUPL 特化执行器（约 1064 行） |
| `cpp/kupl/einsum/src/topology_apply.cpp` | 拓扑入口（转发通用执行器；步表实验路线） |
| `cpp/kupl/einsum/src/topology_gemm.h` | `BlockPool` + 模板 `run_typed<float|double>` |
| `cpp/{kupl,cuda}/einsum/include/einsum_define.h` | **双侧必须逐字节同步**（`make-nv.sh` 用 `cmp` 比较） |

构建：`bash cpp/kupl/einsum/make.sh`、`bash cpp/cuda/einsum/make-nv.sh`。

## 3. 已验证事实

### 3.1 本轮（2026-11-12）实跑

| 项 | 数字 | 命令 |
|---|---|---|
| 全量测试 | 118 passed / 1 skipped / 0 failed | `python -B -m pyqcd.testing` |
| 拓扑契约 | 6 passed | `python -m pyqcd.testing._topology_equations_contract` |
| 语料规模 | 8638 条方程、**8638 个唯一签名** | `len({canonical_signature(t.equation) for t in TASKS})` |
| 冒烟档抽样 | 8/8 命中 `kupl_dedicated`、0 WRONG；可用 3 条中位 **0.14×** | `--sample 8 --seed 0 --scene smoke` |
| `--scale 16` 抽样 | 7 条出数 + 1 条 `REF-FAIL`；**7/7 命中、0 WRONG**；可用 4 条中位 **0.33×**（0.91/0.45/0.16/0.22×） | `-u -m … --sample 8 --seed 0 --scale 16` |
| 16 算子单点 | seq 6585：0.069 s vs 0.323 s = **0.22×** | 同上 |
| 抽样可复现 | 同 seed 输出 md5 一致、异 seed 不同；n=1/8/16/100 唯一且不泄漏固定 15 条 | 见 §6 |

### 3.2 技能卡记录、本轮**未**复测

- dim=4 特化/通用中位比 **0.19×**（400 条抽检）、dim=8 0.76×、dim=12 0.96×、dim=16 0.91×。
- 匹配开销 3.45 µs（单条重复）/ 16.0 µs（2582 条轮转）——**该数字来自 2582 条语料时代，语料已增至 8638，需重测**。
- CUDA 侧 dim∈{4,8,12} 32+ 组 `correct`、`rel ≤ 4.1e-7`。

## 4. 必须知道的坑（含本轮新增）

1. **scene 档吃内存**：单张 peram `(40,64,4,4,50,50)` complex128 = **1.53 GiB**；
   6 传播子 16 算子路径光操作数 ~9 GiB，实测峰值 RSS **14.4 GiB**。15 GB 机器跑它
   会换页（swap 进出 ~80 MB/s、CPU 13%、`wchan=__folio_lock_or_retry`），数字不可用。
   → 用 `--scale 16`（peram 312 MiB）或 `--scene smoke`。
2. **`opt_einsum(dp)` 只按 flops 选路**：seq 3186（14 算子）在 scale=16 下要
   `(32,32,50,50,50,50)` complex128 = **95.4 GiB** 中间体，直接 `ArrayMemoryError`。
   脚本现按行标 `REF-FAIL` 跳过并继续；**这是参考实现病态，不是内核错**。
3. **表头必须带 `scale`**：`run()` 曾漏传 `scale`，`--scale 16` 时表头仍写 `Nx=64`，
   照日志引用会把数字贴到错误维度。已修；读历史日志先核对表头与 `--scale`。
4. **参考一律用同 dtype 通用路径** `contract(..., specialize=False)`；高元数（dim≥8）
   别用 `numpy.einsum` 当参考，它可能先超时/爆内存，会被误判成 kernel 错。
5. `out=` 会**绕过全部**专用 kernel；验证拓扑入口不要传 `out=`。
6. Python 计划缓存只有 32 项 FIFO，键是**调用方写的方程**（不是拓扑签名）；整语料
   单遍扫描会反复建计划，不能当稳态耗时。
7. `topology_kernels.inc` 是生成物且被 `.gitignore` 忽略；改语料后需重跑
   `python -m pyqcd.testing._topology_kernel_gen` 再重建。
8. **`_topology_registry_regen` 不存在**（技能卡 docstring 指向它但仓库无此模块）；
   改语料只能按口径手工重算 96 位摘要。
9. CUDA `TopologyProgram` 是**单槽**缓存（按计划字节 `memcmp`），换拓扑即覆盖；
   入口要求 `stream == nullptr`。
10. 静态生成内核只展开 `float32`，未接线；double/complex 走 `_probe_double` 实测后才启用。
11. **标签可能滞后于实际执行器**：`verify_specialized` 对 seq 1 显示
    `特化路径: … [kernel(topology)]`、`可用内核: kupl_topology, cuda_topology`，
    但 `实际执行: kupl_dedicated`。判定只认「实际执行」行、
    `PYQCD_EINSUM_TOPOLOGY_TRACE=1` 的 `topology-specialized:` 或基准里的
    `[kernel: kupl_dedicated]`，不要只看标签名。

## 5. 下一步建议（按优先级，附验收）

| 优先级 | 任务 | 验收方式 |
|---|---|---|
| P1 | **修文档漂移**：`SKILL.md` §1 仍写「语料 2582 / 摘要 2582 条」，实际 8638/8638 | 以 §3.1 的数字改表；改完重空跑 `_topology_equations_contract` |
| P2 | **CUDA 侧抽样复核**：本轮只跑 CPU/KUPL，`cuda_topology` 的抽样命中与数值未复测 | `--sample 8 --seed 0 gpu single`，路径列须为 `kernel: cuda_topology` 或 `kupl_dedicated`，0 WRONG |
| P3 | **scene/full 档数字**：需 ≥16 GB 可用内存机器 | 先按「peram 尺寸 × 传播子数」估算，跑前 `free -g` 留足余量；记录 `--scene`/`--scale` 与表头 |
| P4 | **重测匹配开销**（8638 条轮转、`lru_cache(512)` 命中率） | 输出 µs/call 与 `cache_info()`，写回技能卡 §1 |
| P5 | 若要给 `REF-FAIL` 用例补数值线 | 用 `contract(..., specialize=False)` 作参考，但报告里必须标"非外部 oracle" |

## 6. 复现命令

```bash
# 原生是否可用
python -c "import pyqcd.einsum as pe; print(pe.build_info()['native_available'])"

# 列表与抽样（不导入 opt_einsum）
python -m pyqcd.einsum.bench_specialized --list
python -m pyqcd.einsum.bench_specialized --sample 16 --seed 0 --list

# 抽样基准：冒烟 / 可控规模
python -m pyqcd.einsum.bench_specialized --sample 8 --seed 0 --scene smoke
python -u -m pyqcd.einsum.bench_specialized --sample 8 --seed 0 --scale 16

# 单条路径验证（走特化/数值/是否真跑原生）
python -m pyqcd.testing.einsum.verify_specialized "cadb,menf,ac,em,bd,fn->" --dim 6

# 单条路径查询（纯查询，随 pip 发布；命中 0 / 语料外 1 / 非法 2）
python -m pyqcd.einsum.verify_specialized "agbh,cmdn,eqfr,ospt,ikjl,ce,mo,gi,kq,bdf,np,hj,lrt->as"

# 契约与全量
python -m pyqcd.testing._topology_equations_contract
python -B -m pyqcd.testing

# 原生构建与诊断
bash cpp/kupl/einsum/make.sh
bash cpp/cuda/einsum/make-nv.sh
python -m pyqcd --gpu-probe
```

抽样契约自查（本轮用过）：

```python
from pyqcd.einsum import bench_specialized as B
fixed = {seq for _, seq, _ in B.CASES}
got = [seq for _, seq, _ in B.sample_cases(16, 0)]
assert len(got) == len(set(got)) == 16 and not (set(got) & fixed)
assert B.sample_cases(16, 0) == B.sample_cases(16, 0)
assert B.sample_cases(16, 0) != B.sample_cases(16, 1)
```

## 7. 交付与记录规范

- **Git**：只暂存本任务文件，禁 `git add -A`；提交消息写「原因 / 改动 / 验证」；
  普通推送，禁 force push 与改写已推送标签；单文件 >10,000,000 字节不入库。
- **日志**：常驻 `log` 技能，`logs/v<YYYYMMDDHH>.txt`（`final` 模式含完整报告与用户指令），
  索引在 `logs/AGENTS.md`。本轮：`logs/v2026111217.txt`。
- **标签**：`dev<N>` 递增；消息格式
  `follow <前一标签>, 1. …; 2. …; [CommandCodeBot].`；前一标签必须用
  `git describe --tags --abbrev=0 <新标签>^` 取，**不得按序号推断**。
- **目录/命名**：`docs/` 用内容名+日期（本文件 `handover_einsum_20261112.md`）；
  `refer/` 只读；`data/`、`logs/` 为归档区。

## 8. 本轮交接物

- 提交 `1550aaa`：`bench_specialized --sample` + `REF-FAIL` 降级 + 表头 `scale` +
  `_bench_scene` 内存估算修正 + 文档/技能卡同步。
- 日志：`logs/v2026111217.txt`（7550 字节）。
- 技能卡：`skills/pyqcd-einsum/SKILL.md` 新增「抽样复核：固定 15 条之外的语料（2026-11-12）」。
- 遗留：见 §5 P1–P5；无阻塞项，无未提交改动。
