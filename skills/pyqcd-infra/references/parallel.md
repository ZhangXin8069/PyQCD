# Parallel reference — MPI meta-task scheduling

## 适用范围

任务涉及 MPI 元任务、GPU 绑定、显存规划、并行管线或残缺产物补跑时读取。本机制
不是 lattice domain decomposition；传播子内部的 PyQUDA grid 仍由 `pyqcd-propagator`
处理。

## 规划

按用户给定公式

\[
N\,a=n\,b,
\]

其中 `a` 是单元任务显存，`b` 是单卡可用显存（通常取 80%），`n` 是 GPU 数，求进程数
`N`；批次 `X=m/N`，每卡进程 `Y=N/n`。实际调用前以 `plan_parallel(...)` 的返回值
为准，不手算后覆盖。

资源探测必须保留三态语义：`None` 是 unknown，`0` 是成功探测到的 known-zero，正数是
known-positive。GPU 数量、单卡可用显存 `b` 和 `MemAvailable` 都遵循此规则；`MemTotal`
是容量而不是可用内存，不能在 `MemAvailable` unknown 时替代它。GPU 探测失败产生
provisional CPU plan，并在 GPU gate 上 fail closed；它不能被改写成“已确认无 GPU”。

- 只有显式提供正的 `a_mem_mb` 时，`N*a=n*b` 才是带实测/估计单任务显存的数值规划；
  例如 `a_mem_mb=512` 的 `format_plan` 报告 `a=512 MB/task`。
- 未提供 `a_mem_mb`（或值为 0）表示 `a` **未知**，不是单任务占用 0 MB。此时规划器
  默认 `Y=1/GPU`，`format_plan` 必须报告 `a=not provided (default Y=1/GPU)`，不得打印
  `a=0 MB/task`。
- `n_gpu`（包括显式 override）必须是非负整数；负值必须在规划和任何目录/设备副作用前
  拒绝，不能把负数解释成 CPU 请求。已知 `n_gpu=0` 与 unknown GPU 数量必须在报告中可
  区分。`b=0` 是已知显存耗尽，`b=None` 是未知；二者都不能伪装成可执行 GPU 计划。
- `mem_avail_mb=None` 时 `mem_ok=unknown`；`mem_avail_mb=0` 等已知不足时必须保留
  `mem_ok=False`。最终 `cpu_ok=False`、`mem_ok=False` 或 `gpu_ok=False` 的计划都必须在
  preflight 拒绝，不能先创建 run directory 再失败。
- `--dry-run` 只验证规划、设备分配与报告文本，不构成单任务显存实测。

```bash
python -m pyqcd.parallel --dry-run --confs 6250,6450
mpirun -np N python -m pyqcd.parallel --confs 6250,6450
```

## Collective preflight

多 rank 运行在任何 `run_dir`/`data`/配置文件创建前完成一次 collective preflight：

- `steps` 只能包含 `vertex/2pt/ope/3pt/4pt` 与 rank 0 的
  `env/analysis/plots/report`；未知步骤必须同步拒绝。`tmd` 当前没有安全的逐 rank 输出
  契约，必须明确拒绝，不能空跑后返回成功。
- 已提供的 `plan['N']` 必须严格等于 `comm.size`；不一致时所有 rank 收到同一失败，不能
  让 `mpirun -np` 与规划结果各自继续。
- `env` 在 MPI 中是 rank 0 的报告元数据阶段（组态、精度、格点和 gauge 目录），不是
  每个 rank 各写一份环境文件；真正的环境快照和文件归档由 `pyqcd-pipeline` 负责。
- preflight、规划日志和后续 setup/step/log 阶段均捕获 `BaseException`，通过
  `allgather` rendezvous 后再统一抛出聚合 `RuntimeError`；任何 rank 失败都不能让其他 rank
  进入下一个 collective。
- 不得使用未经保护的 `comm.Barrier()`。所有可能发生 rank 分歧的阶段都必须先收集本地
  失败摘要，再通过有界的 `allgather` rendezvous；单 rank 异常时其余 rank 也必须得到同一
  个聚合失败，而不是永久等待 barrier。

`--dry-run` 在 preflight/规划报告后结束，不创建运行目录、配置快照或数据文件；size=1
时也不调用串行 `run_pipeline`，并返回未经串行 fallback 改写的推荐计划。它不证明 GPU
显存实测、输入存在或计算成功。

## 运行目录契约

- 显式传入 `run_dir` 时，rank 0 原样广播该路径，不改名、不追加后缀；其余 rank 以广播值
  为准；默认目录工厂的 `tag` 规则不改写也不限制这个显式路径。
- 共享默认目录工厂的可选 `tag` 只接受 `None`、空串或不含路径组件/NUL 的单个字符串
  basename；路径型或非字符串 `tag` 在创建 `output_root` 之前即抛 `ValueError`。
- `run_dir=None` 时，仅 rank 0 通过共享目录工厂生成默认路径，再向当前 communicator
  广播。默认 basename 采用
  `output_<秒级时间>_<9位纳秒片段>_p<PID>_<64-bit安全随机后缀>`；候选目录以
  `os.makedirs(..., exist_ok=False)` 原子预留，碰撞时最多重试 16 次。同一作业所有 rank
  因此使用同一路径。
- broadcast 只证明**当前 MPI 作业内一致性**，不连接两个独立 communicator，也不提供
  跨文件系统全局锁。高分辨率时间、PID 与安全随机后缀减少同秒独立作业选中同一候选的
  概率，原子 `mkdir` 只在共享的 `output_root` 文件系统上裁决竞争；不能据此声称跨主机、
  跨文件系统绝对零碰撞。

## 后端与 collective 失败契约

- MPI setup 在每个 rank 上先确定请求的 `backend/device`（torch GPU 时设备为
  `cuda:{rank % n_gpu}`），再统一调用 `set_backend(backend, device=device)`；该调用必须
  发生在首个元任务之前。不能把 OPE 元任务内的局部后端设置当作整个并行作业的初始化。
- 会让 rank 控制流分叉的阶段按同一模式处理：本地代码捕获 `BaseException`，生成只含
  `phase/rank/type/message` 的失败摘要，然后所有 rank 都调用
  `_collective_failure_rendezvous`。真实 communicator 通过 `allgather` 汇总；任一摘要非空
  时，各 rank 收到同一条聚合 `RuntimeError`，而不是一部分继续到下一个 collective。
- 当前同步边界是 `preflight`（含规划与 plan logger）、`run-dir`、`setup`、每个
  `meta-task:<step>`、`step-log:<step>`、rank 0 `postprocess:<step>` 和
  `completion-log`。尤其日志回调也属于运行阶段：preflight、step 完成日志或 completion
  日志在单 rank 失败时都必须先 rendezvous，不能局部抛出后让同伴阻塞。
- `BaseException` 契约覆盖 `KeyboardInterrupt`、`SystemExit` 等中断；测试验收的是所有
  rank 有界地得到相同聚合失败，不是静默吞掉中断，也不是仓库当前未实现的自定义
  failpoint、`--test-mode` 或 `MPI.Abort` 协议。
- `get_mpi_context()` 只有在没有 MPI launcher 标记时，才允许 `mpi4py` 初始化失败后返回
  `(None, 0, 1)`。若环境含 `_MPI_LAUNCH_ENV` 中任一标记（例如
  `OMPI_COMM_WORLD_SIZE`、`PMI_SIZE`、`PMIX_RANK` 或 `MV2_COMM_WORLD_SIZE`），初始化失败
  必须抛 `RuntimeError`；禁止把多个 launcher rank 伪装成彼此独立的串行任务。

对应的真实验证入口是：

```bash
python -m pyqcd.testing._pipeline_runtime_contract
python -m pyqcd.testing._mpi_reliability_contract
python -c "from pyqcd.testing._mpi_failure_contract import test_mpi_collective_failure_contracts; test_mpi_collective_failure_contracts(); print('PASS test_mpi_collective_failure_contracts')"
```

第三条会在依赖可用时启动真实双 rank，逐项覆盖 run-dir、setup、普通/`BaseException`
元任务、step-log、completion-log、preflight plan-log 和 rank 0 postprocess；外层 timeout
负责把旧实现的永久阻塞判为失败。不要用文档中不存在的 CLI flag 替代这些契约入口。

## 调度与验收

- `run_parallel_pipeline` 按 `(step, conf)` round-robin，rank `mod n` 绑定 GPU。
- 验收默认目录时分别检查两条性质：同一 communicator 各 rank 路径完全相同；两个独立
  作业即使同秒启动也得到不同 basename。显式 `run_dir` 必须逐字节保持不变。
- MPI `2pt` 元任务复用 `step_2pt` 的精确数组完成门：每个请求通道的 `P0/P2` 文件必须
  是有限 `float64`、shape 为 `(NT,)`；HDF5 顶层只能有 `data`，空、损坏、缺 dataset、
  额外 dataset、非数值、错误 dtype 或错误 shape 按未完成处理。`.h5` 优先，损坏的首选
  HDF5 不能由同名旧 `.npy` 静默掩盖。通用 vertex resume 也严格检查唯一 `data`、精确
  precision/dtype、shape 和 finite；大型数组有限性检查按首轴分块。该数组门已验证完整性，
  但仍与 test9 canonical 的 JSON/SHA 身份门分层；默认命中完整缓存即跳过，
  `recompute_2pt=True`/`--recompute-2pt` 才强制重算。
- 每个元任务无论成功或异常都在 `finally` 调用后端感知的 `free_gpu_memory` 和垃圾回收；
  两项清理各自吞掉 `BaseException`，前一项失败不阻止后一项；清理失败不能覆盖主计算
  异常，成功计算也不能因清理失败变成失败。这是 best-effort 契约，numpy/cupy 路径不应
  为清理强制导入 torch。analysis/plots/report 仅 rank 0。
- size=1 的 MPI/串行 fallback 必须真实调用串行 `run_pipeline`，并逐字透传
  `recompute_2pt=True/False` 及相关配置；不得静默忽略强制重算。dry-run 保留推荐的
  `N/X/Y` 计划、不调用串行计算；只有实际运行才把执行视图收敛为 `N=1`。
- 正式运行前保存 dry-run、设备/显存和任务清单；运行后核对每个任务退出码、rank0
  产物和模板/大小守卫。
- OOM 时按公式降低批次/进程数，或回退 CPU；不要静默改变任务集合。

MPI 规划/CLI 的当前 contract 证据：

```bash
python -B -m pyqcd.testing._mpi_planning_contract
# 38 passed, 0 failed
```

该入口同时覆盖负 `--n-gpu` 的 argparse `rc=2`、三态 GPU/RAM 资源、`cpu_ok=False` preflight、
size=1 真实串行透传与 dry-run 计划语义；CLI 非法输入不应产生 traceback。
