# einsum tune 排查手册（2026-11-19 / 11-20）

面向**使用者**：`python -m pyqcd.einsum.test --tune` 与 `python -m pyqcd --gpu-probe`
出问题时的查法。内部口径见 `logs/einsum-tune-20261119-facts.md`。

## 1. 读懂每一组的四种结局

`--tune` 的每一组只有四种结局，全部是**实测**得出的（没有任何预测/估算）：

| 状态 | 含义 | 你该做什么 |
|---|---|---|
| `ok` | 该组跑完，`wall_ms` 是实测中位 | 无需动作；它在参与选优 |
| `pruned` | 某一次调用耗时超过**截止**（`2 × 当前最快中位`） | 无需动作；它只是比最快慢 |
| `timeout` | 某一次调用超过 `--tune-timeout × 调用次数` 的组级窗口 | 把 `--tune-timeout` 调大，或先 `--tune pass` |
| `error` | 该组抛了异常，`原因=` 是**异常原文** | 看原文字面；见下面第 2、4 节 |

`skip` 是第五种标号，但它不属于遍历：那是**遍历前**就判为"这台机器给不了这条路"
的内核（例如没有 GPU 时的 `dedicated`），整档一行、`组数=` 说明被去掉多少组。

## 2. 为什么一组要跑几十秒，而不是 30 秒就死

`--tune-timeout` 说的是**单次** `contract()` 调用的上限，而窗口按**组**执行：

```
组级窗口 = --tune-timeout × (1 + warmup + rounds)
```

缺省 `30 × (1+1+3) = 150 s`（加速设备上 warmup 被抬到 2，于是 `180 s`）。所以：

* 看到一组在 60 s、80 s 返回，**不是**超时失效——它自己结束了；
* 真正的上限要等窗口用完，那时会记 `timeout` 并打印父进程 kill 子进程的事实。

`--tune-timeout 0` = 不设上限（组级窗口也不设）。`--tune-budget` 是整轮上限
（缺省 0 = 不限），用尽即停并报「已测 N/M」。

## 3. 默认行为：第一次调用会先 tune 再执行

配置表里没有"这次调用的键"时，缺省会**先跑一整轮 `--tune`**，再按胜出配置执行，
并打印一行说明。关掉：

```bash
export PYQCD_EINSUM_AUTOTUNE=0
```

同一个键在一个进程里只会自动 tune 一次（失败也记账，不会每次调用重演）。
`--profile false` / `PYQCD_EINSUM_PROFILE=0` 时不自动 tune（用户说过"别读写表"）。

## 4. 原生 `.so` 过期怎么判断与重建

症状：拓扑/专用内核"明明编译了却一次都没跑"，原生 trace 里一行
`topology-specialized` 都没有；契约静默失败。

```bash
# 判断：符号里的表项数必须等于源码里的 kTopoRecordCount
nm -D pyqcd/einsum/kupl/libpyqcdeinsum_kupl.so | grep kTopoRecords
grep -n 'kTopoRecordCount' cpp/kupl/einsum/src/topology_kernels.inc | head -1

# 重建（两步都是）
bash build-kupl.sh                 # KUPL（CPU 侧 ctypes 桥）
bash build-nv.sh                   # CUDA（需要 nvcc）
PYQCD_EINSUM_NATIVE_ARCH=0 bash build-kupl.sh   # 可移植构建（不带 -march=native）
```

CUDA 侧的步表在 `cpp/cuda/einsum/src/topology_steps.h`（8638 条表项 / 3477 条不同
步序），由 `python -m pyqcd.einsum.topology_steps_gen --target cuda` 生成，随
sdist 发布，编进 `.so`。

## 5. GPU 计时为什么必须同步

`cupy` / `torch(cuda)` 异步下发，不同步就只量到入队时间。0.5.9 之前的全部 GPU 数字
因此**作废**（配置表里 `wall_ms=0.320` 而脚本实测 900 ms 量级）。现在 `_timed` 在
每个样本**前后**各 `_sync.synchronize()` 一次；CPU 上是零依赖空操作。

## 6. 容差怎么选

不写 `--atol` = 自动档：64 位 `1e-12`、单精度 `7.63e-6`（该 dtype 机器精度 × 64）。
一条 10 算子收缩在单精度上实测累积约 16 eps，拿 `1e-12` 判单精度会让每次运行都失败。
写了数值就**原样生效**（`--atol 1e-30` 就该判失败）。

## 7. `--gpu-probe` 现在会说什么

```
kernel status  : ok — topology entry returned the reference result for complex128;
                 dtypes: complex128=ok complex64=ok float64=ok float32=ok
dtypes         : complex128=ok complex64=ok float64=ok float32=ok
verdict        : CUDA fast path is active
```

* 四类 dtype **平权**，逐类给结论；
* 某一类失败时原因带**真实阶段**（`stage=native: NativeUnavailable: ...`），不再
  统一说成"数值不一致"；
* 部分可用时 verdict 写 `partially active — dtype(s) X fall back`，不说 active 也
  不说 NOT active。

## 8. 全量记录表与"更省内存的配置"

`--tune` 除了写配置表（每个键只留胜出那一行），还整表写一份
`pyqcd.einsum.record.csv`（同目录）：

* 列与配置表相同，多一列 `peak_memory_mb`（单位 MB = 1024² 字节）；
* 记录**每一组**，`note` 列写 `status=… best` 等；
* **从不参与选路**——它不是缓存。

命中配置表时，若记录表里同一键下存在"峰值不到当前一半、且耗时不超过当前 1.2 倍"
的配置，会多打一行 `存在更省内存的配置：…`，并指明"把 record.csv 中那一行直接替换
profile.csv 的对应项"。

## 9. 配置表放哪

`$XDG_CACHE_HOME/pyqcd.einsum/`，未设时 `~/.cache/pyqcd.einsum/`。
`PYQCD_EINSUM_PROFILE_DIR` 或 `--profile-dir` 可覆盖。目录按**用户**算，不按工作
目录——tune 一次，所有项目共用。旧位置（`./.cache/pyqcd.einsum.profile.csv`）仍然
可读，只读地当作来源，不会自动搬动。
