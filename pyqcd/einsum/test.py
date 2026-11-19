"""``qcd-einsum`` 单方程测试的**唯一合法接口**（强制，2026-11-15 起）。

强制规定的原文
--------------
本仓库对 "单条方程 / 这组形状 / 这个 dtype / 这块设备，算得对不对、多快" 的验证，
**只认这一个入口**。禁止用 ``/tmp/*.py``、scratchpad 脚本、一次性 ``python -c``、
或 ``pyqcd/testing/`` 下的临时脚本来替代；这类脚本产出的结论不予采信，也不得写进
报告、文档或 tag 消息。唯一例外是整库回归 ``python -m pyqcd.testing``——它答
"整库有没有退化"，本模块答"这一条方程对不对、多快"，二者互补，没有第三个入口。

为什么必须统一（三次实测教训，不是形式主义）
--------------------------------------------
1. **warmup**：临时脚本普遍测单次，而首调含 plan 缓存 + 原生库绑定 + JIT；
   实测 seq=2583 首调 45 ms vs 热态 0.56 ms（**80×**），单次测量完全失真。
   本模块强制 ``--warmup >= 1``，并把 ``--warmup 0`` 直接判为用法错误。
2. **容差**：``np.allclose`` 的逐元素绝对阈值会误判接近零的元素；本模块统一用
   **相对**容差（按 ``max|expected|`` 归一），见 ``--atol``。
3. **参照**：有人比 ``np.einsum``、有人比 ``opt_einsum``、有人不比，结论无法互比。
   本模块用 ``--compare {numpy,opt_einsum,backend,none}`` 把参照固定成枚举。

规范由 ``pyqcd/testing/_einsum_test_cli_contract.py`` 护航（形状解析、参数校验、
相对容差口径、缺省值、``--help`` 完整性、退出码语义、内存采样不在计时循环内）。
改本模块必须同步该契约，否则套件红。

调用样式
--------
::

    # 你给的标准形态
    python -m pyqcd.einsum.test --device gpu --dtype complex128 \\
        --equation 'ambn,cgdh,ekfl,oipj,ce,mo,gi,bdf,np,hjl->ak' \\
        --shape '[400,4,1,1],[4,4,9,1],[4,400,400,400],[4,4,9,9],[4,4],[4,4],[4,4],[1,9,400],[1,9],[1,9,400]->[400,400]' \\
        --atol 1e-12 --backend cupy --verbose true --compare opt_einsum

    # 不写 --shape 时按语料里的 roles 反推生产尺寸（seq 自动匹配）
    python -m pyqcd.einsum.test --equation 'cadb,menf,ac,em,bd,fn->' --scene distill

    # 看全部选项与缺省值
    python -m pyqcd.einsum.test --help

口径约定（改这里就是改规范，不要在临时脚本里另立一套）
------------------------------------------------------
* **容差**：``--atol`` 是**相对**容差（按 ``max|expected|`` 归一），不是逐元素
  ``np.allclose`` 的绝对阈值——后者对接近零的元素会误判。缺省 ``1e-12``。
* **计时**：先丢弃一次（plan 缓存 + 原生库绑定 + JIT），再取 ``--rounds`` 次中位。
  单次测量会被首调开销完全支配（实测 seq=2583 首调 45 ms vs 热态 0.56 ms），
  所以 **warmup 不可关闭时不给 0**：``--warmup 0`` 会被拒绝。
* **参照**：``--compare`` 决定与谁比。``numpy`` 用 ``np.einsum(optimize=...)``；
  ``opt_einsum`` 用 ``oe.contract(optimize=...)``；``backend`` 用同一批张量上的
  **另一个后端**（CPU↔GPU 互证）；``none`` 只跑不比对。
* **主路径的优化器**：``--optimize`` 只认 **7 个叶子**（``greedy``/``optimal``/``dp``/
  ``branch_1``/``branch_2``/``branch_all``/``none``，``choices`` 由
  ``_profile_cache.OPTIMIZE_LEAVES`` 给）。``auto`` **不是可写取值**——它是
  ``contract(optimize=...)`` 的签名缺省值、由内置选择器实现，所以该选项的缺省值是
  ``None``（"未指定"），``main`` 再还原成 ``auto`` 交给 ``run()``；显式写 ``auto``
  以退出码 2 结束。
* **参照的优化器**：``--compare-optimize`` 只管**参照**，与主路径的 ``--optimize``
  互不影响。缺省 ``auto`` = 沿用旧行为（numpy→``greedy``；opt_einsum→``greedy``
  或跟随 ``--optimize``）。两边的名字不同——numpy 实测只认 ``greedy``/``optimal``
  （``dp``/``branch-*``/``auto`` 全是 ``KeyError``），opt_einsum 则按自己的路径注册表
  认 ``dp``/``branch-*``/``auto`` 等。**不做跨库重映射**：不支持就报用法错误。
* **形状**：``--shape`` 与 ``--equation`` 的操作数个数必须一致，且同一标签在各
  操作数里长度必须相同；不符直接报错，**不做广播猜测**。
* **专用路径**：``--specialize auto``（缺省）跟随环境变量
  ``PYQCD_EINSUM_SPECIALIZE``；``false`` **强制通用路径**，是排查原生内核问题的
  第一手段；``true`` 强制走特化路径，不让它静默回落。三态的选择与"实际生效"的
  布尔值都会写进报告与 ``--verbose true`` 的明细。
* **强制内核**：``--kernel-force generic|dedicated|topology|backend``（**全部是强制值，
  没有 auto**）。``generic``（缺省）完全跳过专用内核；``dedicated`` / ``topology``
  只允许**当前后端的**那一个（numpy → ``kupl_*``，cupy → ``cuda_*``），**只接受单个值**
  （逗号链式已取消，写多个直接报用法错误），强制**真试一次**，跑不成就抛
  ``KernelUnavailableError`` 并以非零退出码结束——**绝不**静默回落；``backend``
  直接调后端自己的 ``einsum``（与其它标签互斥）。写 ``auto``（或任何未知取值）报用法错误
  （退出码 2），消息点名交给 ``--tune`` 并列出合法集合——自动选路只由 tune 的实测结论决定。
  显式 ``--kernel-force`` 压过环境变量 ``PYQCD_EINSUM_KERNEL_FORCE``（未设 = ``generic``）。
  明细里的 ``kernel :`` 行给出 ``requested`` / ``ran`` / ``attempted``。
* **缓存**：``--cache auto|true|false``。``auto``（缺省）跟随环境变量
  ``PYQCD_EINSUM_CACHE``（只有 ``0/false/off/no`` 才关闭）；显式 ``true``/``false``
  压过环境变量，后者让底层**既不读也不写**任何 einsum 缓存（不留半热状态、不污染
  统计）。显式给出 ``true``/``false`` 时，``--verbose true`` 的报告**追加**一行
  ``cache    : enabled=... hits=... misses=... hit_rate=...``；``auto`` 与省略该
  选项等价，不写这一行——既有输出逐字节不变。
* **首次详解**：``--explain auto|true|false``。``auto``（缺省）跟随 ``--verbose``；
  ``true`` 即使不开 ``--verbose`` 也打印；``false`` 一律不打印。本次进程**第一次**
  且画像未命中时，在既有明细**之后追加**三段（``plan`` → ``kernel`` → ``why``，
  用 ``→`` 串因果）；之后每次只加一行画像状态（命中 / 未命中已记录 / 关闭）。
  默认（非 verbose、非 explain）一个字都不多打。
* **配置画像**：``--profile auto|true|false`` 与 ``--profile-dir DIR`` 原样交给
  ``contract(profile=..., profile_dir=...)``。``auto`` 跟随
  ``PYQCD_EINSUM_PROFILE``；``false`` 让本次**既不读也不写**画像 CSV（连目录都不建）。
  画像只**建议**配置：方程、dtype、形状与数值一概不改；报告里给出的 ms 一律是
  **记录值**，不是本次实测。缺表时的 ``需要先 tune：`` 提示用**本次命令行的实参**
  构造（``--device``/``--dtype``/``--equation``/``--shape``（含 ``->[…]`` 输出段）/
  ``--backend``），``--compare`` 只在用户原命令给过时才写——提示必须能原样复制执行，
  而不是把人引到另一条配置上（2026-11-18 用户实测：GPU 跑却提示 ``--device cpu``）。
* **自动施加等价写法**：``--as-equivalence auto|true|false|force``。前三态同
  ``--equivalence``；``force`` 是第四态，**忽略负标记**：画像里"该方程查过、无法
  等价改写"的跨进程标记不再拦住这次查询，而是真的再查一遍并刷新结论（``true``
  会尊重该标记）。字符串 ``"force"`` 原样交给 ``contract(as_equivalence=...)``。
* **结果行**：``PASS <equation>  <实际耗时>ms  min_wall_ms=<最优值>ms
  best=kernel_force=… as_equivalence=… optimize=…  vram+…``。``--tune`` 时把**剪枝上界**
  与**最终 best 配置**一并写进这一行——数值相同也要都显示，因为"上界确实来自最优组"
  正是靠这两个数相等来确认的（2026-11-18 用户要求）。
* **``--tune`` 的取值**（2026-11-18 用户裁定）：它**带取值**，不是开关。留空 /
  ``true`` / ``all`` = ``all``（下面各条描述的完整模式）；``pass`` = **快速通过模式**
  ——强制 B3 的 21 组（无 ``topology``）、``as_equivalence`` 固定 ``false``、跳过分支
  探测与全部 ``--compare``/数值校验（``rel`` 一律 ``-``、不做 MISMATCH 判定），并关闭
  提示/详解等无关功能，表尾注 ``pass 模式：跳过数值校验``；``false`` / 空串 = 不跑。
  非法取值退出码 2，绝不静默当成 off（那会让拼写错误看起来像"跑了没输出"）。
* **初始化过程输出**：遍历开始之前按阶段逐行打印 ``tune init :``（mode/形状规模/
  分支探测**起止**/内核准入**起止**/session/首组**起止**）。字段名与取值是**机器可读**
  的稳定 token、不跟 ``--language`` 翻译，耗时以 ``ms`` 取整——目的只有一个：
  一旦卡住，立刻看得出卡在哪一步（2026-11-19 用户要求）。
* **遍历调优**：``--tune``（缺省关）在**同一条**方程/形状/dtype/设备/后端上遍历
  三轴 ``kernel_force × as_equivalence × optimize``。**不是笛卡尔积**：清单按分支
  给出且**逐项有序**（共同前缀 ``dedicated`` → ``generic`` → 分支专属 ``topology``
  一条 → 尾部 ``backend``；每个内核内 ``greedy/branch_1/branch_2/dp/branch_all/
   optimal/none``），B1/B2 = 22 组、B3 = 21 组（硬上限 30，超了报用法错误而不是
   截断）。分组判据只看两件事：原方程能否特化、以及等价写法能否特化——只探测、
   不计时、不写表。``--compare`` **缺省关闭**（不给就不跑参考基准；显式给出时
   主路径照旧跑一次，遍历本身不为它多算一遍）。这三轴由 ``--tune`` 自己给，
   所以同时再显式给 ``--kernel-force`` / ``--as-equivalence`` / ``--optimize``
   （其中 ``--kernel-force`` 非 ``generic``、``--as-equivalence`` 非 ``auto``、
   ``--optimize`` 只要写了就与网格不同——它的 ``auto`` 根本不在可写集合里）
   时**各打一行 stderr
   警告后忽略它**（每个参数一行），遍历照常进行、
   退出码仍为 0。每组沿用 ``--warmup``/``--rounds`` 取中位；``--verbose`` 时另按中位
   升序给前 5 行 + 每个 ``kernel_force`` 取值的最快一行，既有输出行一字不动。报的是
   **单次运行的实测，不是基准**。
* **``min_wall_ms`` 剪枝**：上界初值 = ``--tune-timeout``；任一组的单次 ``contract()``
  耗时超过当前上界即**终止该组**（原文 ``耗时大于min_wall_ms,此组被终止``，并附
  ``wall_ms`` 与 ``min_wall_ms`` 两个数值）、继续下一组、**不写缓存**；某组测到更小耗时
  则**立刻**把上界降为该值（后续所有组都用新上界剪枝）。``best`` 就是记录的最小耗时
  那一组。剪枝不豁免任何组——包括清单最前面的那几组。
* **遍历前先探测再裁剪**（规格 §18，2026-11-18）：``dedicated`` / ``topology`` 会在
  跑任何一组之前查一次**准入**——``dedicated`` 复用内核自己的 ``enabled()`` /
  ``dedicated_dtypes()`` / ``_dedicated_cached()`` / ``*_plan_fallback_reason()``
  （**计划 arena 超预算**的原文与阈值都来自内核，不在本模块另立数字），``topology``
  用 ``_specialize.match(方程或等价改写)``。判为不可用的内核**整档**不跑，表里给一行
  ``状态=skip``（**不是** ``error``：``error`` 只留给真正的意外），表尾另有
  ``skipped-kernels:`` 逐条点名；``1/N`` 的 ``N`` 是**裁剪后**实际遍历的组数，
  ``tune hdr`` / ``tune plan`` 两行写明"计划多少 → 实际多少"。
* **tune 的输出必须完整**（规格 §12/§13，2026-11-17）：遍历期间**每一组都给一行**
  ``tune comb : 序号/总数  kernel_force=…  as_equivalence=…  optimize=…  backend=…
  dtype=…  device=…  wall_ms=…  状态=<ok|timeout|error|mismatch|pruned|skip>  原因=…``
  （后三项是该组 **实际下发**的性能配置），与 ``--verbose`` 无关、不去重、不截断。
  ``timeout`` / ``error`` / ``mismatch`` / ``pruned`` 的**原因原文**逐字打出：原文含
  换行时逐行续打（``tune due  :``），绝不折叠成"…等 N 项"，也不只报首个。保存信息
  同样完整：配置表
  **绝对路径**（``tune file :``）、写入的每一行（键各字段 + 配置字段 + ``wall_ms`` +
  ``note``，``tune wrote:``）、删除的行数与**每一被删行的键**（``tune del  :`` /
  ``tune delrow:``），以及是否因"同键多行"做过清理（``tune dup  :``）。临时关闭缓存、
  缺文件警告、冲突警告一律逐条完整输出（tune 模式下不受任何 quiet/notice 去重控制）。
  **末尾**再给一张对齐的汇总表（列固定 ``# kernel_force as_equivalence optimize
  wall_ms 状态``，配置名不截断），表尾 ``best: …`` 与 ``saved: <绝对路径>
  rows=+N deleted=-M``。
  非 tune 的普通调用仍只有一行简洁提示，详细输出不扩散到日常路径。
* **tune 是配置表的唯一写者**：整轮遍历包在 ``_profile_cache.tune_session()`` 里——
  期间配置表**既不读也不写**（读会把这一组量成"历史命中耗时"，写会留下没被选中的垃圾
  行）。遍历结束只写**胜出的那一组**，``wall_ms`` 就是整次 ``contract()`` 的墙钟
  中位（:data:`TUNE_EQUIVALENCE_NOTE` 那条备注除外，见下）：
  ``as_equivalence=false`` 胜出 → 以**原方程**写一条完整性能配置；``as_equivalence=true``
  胜出 → 以 ``target_equation`` 为方程写完整性能配置，**另**以原方程写一条**只含等价
  载荷**（``operand_order`` / ``operand_permutations`` / ``output_permutation``）+ 备注
  "走等价转换后的配置性能更佳"的记录。两种情形都**删除该键的其它所有行**（同键只允许
  1 行），删除条数如实打印。普通 ``contract()`` 调用**只读**，一个字节都不写。
* **超时与预算**：``--tune-timeout``（缺省 30 秒）是**每一次** ``contract()`` 调用的
  上限（warmup 与每一轮都算）。**缺省每组跑在一个全新解释器里**，超时由父进程 kill
  子进程强制生效——这是**硬上限**，能打断原生 C/CUDA 代码，代价是每组一次进程启动
  （实测 ``import pyqcd.einsum`` 中位 126 ms、``import cupy`` 另约 244 ms），这段开销
  **不计入** ``wall_ms``，单独累计在 ``tune note :`` 那一行与 ``isolation.startup`` 里。
  ``PYQCD_EINSUM_TUNE_INPROCESS=1`` 退回进程内路线，那时上限是**尽力而为
  （best-effort）**：能打断 Python 侧长循环（例如 ``optimal`` 的穷举规划，实测 6.5 s），
  **打不断原生 C/CUDA 代码**——信号只在 Python 字节码之间派发，原生代码不返回解释器时
  处理器永不执行，那一组只能等调用返回后才被记 ``timeout``（实测 ``--tune`` 在 CPU
  100%、GPU 0% 状态下挂了 3 分钟，超时完全失效；本机 2026-11-19 另测：单次 2.34 s 的
  原生 BLAS、上限 0.3 s，实际 2.29 s 才抛，即**超限 1.99 s**）。
  子进程走 ``subprocess`` 而不是 ``fork``：CUDA 上下文不跨 ``fork`` 复用，fork 出来的
  子进程必然 ``cudaErrorInitializationError``（实测 22 组全 error）。GPU 上请把
  ``--tune-timeout`` 取足够大，或先用 ``--tune pass`` 快速通过（跳过 compare 与
  等价改写）。
  **窗口粒度是整组**：一组有 ``1+warmup+rounds`` 次调用（缺省 7 次），实际装的是
  ``limit × 7`` 的**组级预算**——按单次设限会把 ``7 × limit`` 全花光，且与
  ``min_wall_ms`` 剪枝（比整组中位）不同口径。
  **预判剪枝**另有两条，只看规划器、不跑收缩：规划出来的 FLOPs 按已测吞吐折算后
  明显超过上界（**估算**，原因里写 ``plan-only estimate, not measured``），或规划
  **自己**就超过上界（规划是一次调用的一部分，这条是**实测**）。它们针对
  ``optimize=none``（生产方程单次 1e17 次浮点）与 ``optimal``（规划 38 s）：
  硬超时也能挡住，但要先把整组额度花光。命中即 ``状态=pruned``、``tune predict:``
  单独报计数——它们**没被测过**，不许混进"测出来慢"。
  任一次超过上限 → 该组记 ``timeout``、继续下一组，中位只由全部成功且未超时的调用
  构成（不拿部分样本凑）。``--tune-budget``（缺省 ``0`` = 不限）是整轮上限，用尽即
  停止遍历并报"已测 N/M"。坏组合——超时、异常、``rel > atol+rtol``——
  **只打印、不中断遍历**，并且一律不参与选优、不写进配置表。
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import statistics
import sys
import threading
import time
import unicodedata
from typing import Any, Sequence

import numpy as np

from . import _tune_axes, _tune_runner

__all__ = ["main", "run", "parse_shape_spec", "DTYPES", "COMPARE_MODES",
           "COMPARE_OPTIMIZERS", "NUMPY_COMPARE_OPTIMIZERS", "DEVICES",
           "SPECIALIZE_MODES", "parse_specialize", "resolve_compare_optimize",
           "translate", "resolve_language", "LANGUAGE_ENV", "LANGUAGE_MODES",
           "DEFAULT_LANGUAGE", "EQUIVALENCE_MODES", "EQUIVALENCE_ENV",
           "AS_EQUIVALENCE_MODES", "AS_EQUIVALENCE_ENV", "DEFAULT_AS_EQUIVALENCE",
           "parse_as_equivalence",
           "DEFAULT_EQUIVALENCE", "parse_equivalence", "equivalence_enabled",
           "DEFAULT_KERNEL", "KERNEL_LABELS", "KERNEL_FORCE_ENV",
           "kernel_labels", "kernel_requested", "kernel_force_line",
           "CACHE_MODES", "CACHE_ENV", "DEFAULT_CACHE", "parse_cache",
           "cache_enabled", "cache_line",
           "EXPLAIN_MODES", "DEFAULT_EXPLAIN", "parse_explain",
           "resolve_explain", "explain_kernel_line",
           "PROFILE_MODES", "DEFAULT_PROFILE", "PROFILE_ENV", "PROFILE_DIR_ENV",
           "parse_profile",
           "TUNE_KERNEL_FORCE", "TUNE_AS_EQUIVALENCE", "TUNE_OPTIMIZERS",
           "TUNE_MAX_COMBOS", "TUNE_AXES", "tune_combos", "tune_conflicts",
           "DEFAULT_TUNE_TIMEOUT", "DEFAULT_TUNE_BUDGET",
           "TUNE_STATUSES", "TUNE_STATUS_OK", "TUNE_STATUS_TIMEOUT",
           "TUNE_STATUS_ERROR", "TUNE_STATUS_MISMATCH", "TUNE_STATUS_SKIP",
           "tune_status", "tune_comb_lines", "tune_skip_line",
           "TUNE_BRANCHES", "TUNE_BRANCH_RAW", "TUNE_BRANCH_EQUIVALENT",
           "TUNE_BRANCH_NONE", "tune_branch", "tune_plan",
           "tune_unavailable_kernels", "tune_plan_trimmed",
           "tune_plan_sizes_trimmed",
           "TERMINATION_MESSAGE",
           "TUNE_MODE_ALL", "TUNE_MODE_PASS", "TUNE_MODES", "parse_tune_mode",
           "tune_mode_enabled", "TUNE_PASS_SKIP_NOTE", "TUNE_INIT_PREFIX",
           "run_tune", "print_tune", "tune_best"]

class UsageError(SystemExit):
    """参数或环境错误。

    继承 ``SystemExit`` 以便顶层直接终止，但把退出码固定成 **2**，与 ``--help`` 里
    写的"0 通过 / 1 数值或形状不符 / 2 参数或环境错误"一致。裸
    ``SystemExit(str)`` 的退出码是 1，会与"数值失败"混淆——脚本据此判断是
    "结果不对"还是"命令写错了"。

    **消息必须自己打印**：解释器只在 ``SystemExit.code`` 是 **str** 时把它写进
    stderr；这里 code 是 int 2，于是 ``args`` 里的文案会被**静默丢弃**——
    21 处用法错误全都只返回 2、一个字都不说（2026-11-15 第三轮巡查实证：
    ``--shape`` 与方程不符时 rc=2 且 stdout/stderr 皆空）。所以这里显式打到
    stderr，并且**保留文案**在 ``args[0]`` 里供进程内捕获。
    """

    code = 2

    def __init__(self, message: str) -> None:
        super().__init__(2, message)
        self.message = message
        self.code = 2


#: 四类 dtype 平权（与 ``_kernel_kupl.TOPOLOGY_DTYPES`` 同集合）。
DTYPES = ("complex128", "complex64", "float64", "float32")
#: ``--device``：决定数组落在哪里。``cpu``→numpy，``gpu``/``cuda``→cupy。
DEVICES = ("cpu", "gpu", "cuda")
#: ``--backend``：原样传给 ``contract(backend=...)``。``auto`` 让库自己判。
#: 只有计算库（numpy/cupy/torch）是合法后端；KUPL/CUDA 原生特化走
#: ``--kernel-force dedicated|topology``（正交维度），不是后端。
BACKENDS = ("auto", "numpy", "cupy", "torch")
#: ``--compare``：参照实现。``backend`` 以 ``--backend`` 的解析结果为参照
#: （原 ``--compare numpy`` 的等价写法是 ``--backend numpy --compare backend``）。
COMPARE_MODES = ("opt_einsum", "backend", "none")
#: ``--compare-optimize``：**只看参照**的收缩序优化器。``auto`` 沿用旧行为。
#: 名字不跨库通用：``numpy`` 实测只认 ``greedy``/``optimal``（其余 ``KeyError``），
#: ``opt_einsum`` 按自己的 ``paths._PATH_OPTIONS`` 注册表认 ``dp``/``branch-*`` 等。
COMPARE_OPTIMIZERS = ("auto", "greedy", "dp", "optimal",
                      "branch-all", "branch-2", "branch-1")
#: numpy 的 ``np.einsum(optimize=...)`` 实测接受的名字（源码是硬编码两个分支，
#: 别处一律 ``KeyError("Path name %s not found", ...)``）。
NUMPY_COMPARE_OPTIMIZERS = ("greedy", "optimal")
#: ``--specialize``：三态。``auto``→(None) 跟随环境变量，``true``/``false`` 强制。
SPECIALIZE_MODES = ("auto", "true", "false")
#: ``--equivalence``：三态。``auto`` 跟随 ``PYQCD_EINSUM_EQUIVALENCE``（缺省开启）。
#: **只影响报告**，绝不改 ``contract()`` 的行为、耗时或任何数值。
EQUIVALENCE_MODES = ("auto", "true", "false")
#: 等价映射报告的开关。缺省**开启**（与 ``_specialize.NOTICE_ENV`` 同口径：
#: 只有 ``0/false/off/no`` 才关闭）。
EQUIVALENCE_ENV = "PYQCD_EINSUM_EQUIVALENCE"
#: ``--cache``：三态。``auto`` 跟随 ``PYQCD_EINSUM_CACHE``（缺省开启）。
#: 与前两个三态不同，它**真的改变底层行为**：``false`` 时缓存既不读也不写。
CACHE_MODES = ("auto", "true", "false")
#: 缓存总开关的环境变量。与 ``_cache.CACHE_ENV`` 同值；写字面量是为了 ``_cache``
#: 缺席时 ``--help`` 仍能说清 `auto` 跟随什么。
CACHE_ENV = "PYQCD_EINSUM_CACHE"
#: ``--explain``：三态。``auto``（缺省）跟随 ``--verbose``；``true`` 强制详解，
#: ``false`` 一律不打印。它只决定**输出**，不改任何执行路径。
EXPLAIN_MODES = ("auto", "true", "false")
DEFAULT_EXPLAIN = "auto"
#: ``--profile``：三态，直传 ``contract(profile=...)``。``auto`` = ``None``
#: （跟随 ``PYQCD_EINSUM_PROFILE``）；``false`` 让底层既不读也不写画像 CSV。
PROFILE_MODES = ("auto", "true", "false")
DEFAULT_PROFILE = "auto"
#: 画像缓存的开关与目录环境变量（与 ``_profile_cache.PROFILE_ENV``/``DIR_ENV``
#: 同值）。写字面量是为了该模块缺席时 ``--help`` 仍说得清 `auto` 跟随什么。
PROFILE_ENV = "PYQCD_EINSUM_PROFILE"
PROFILE_DIR_ENV = "PYQCD_EINSUM_PROFILE_DIR"
#: ``--as-equivalence``：**四态**。``auto`` 跟随 ``PYQCD_EINSUM_AS_EQUIVALENCE``
#: （缺省开启）。与 ``--equivalence`` 不同，它**真的改变执行路径**：开启时命中映射就
#: 改写方程跑，再把结果转回原输出轴序（返回数组仍是原方程的答案）。``force`` 是
#: ``true`` 之上多出的一档：**忽略负标记**（画像里"查过、无法等价改写"的跨进程标记），
#: 真的再查一次并刷新该标记——字符串 ``"force"`` 原样交给
#: ``contract(as_equivalence="force")``。
AS_EQUIVALENCE_MODES = ("auto", "true", "false", "force")
#: 自动施加等价映射的开关，与 ``_equivalence.AS_EQUIVALENCE_ENV`` 同值。
AS_EQUIVALENCE_ENV = "PYQCD_EINSUM_AS_EQUIVALENCE"

DEFAULT_DEVICE = "cpu"
DEFAULT_DTYPE = "complex128"
DEFAULT_BACKEND = "auto"
DEFAULT_SPECIALIZE = "auto"
DEFAULT_EQUIVALENCE = "auto"
DEFAULT_CACHE = "auto"
DEFAULT_AS_EQUIVALENCE = "auto"
DEFAULT_ATOL = 1e-12
DEFAULT_RTOL = 0.0
DEFAULT_ROUNDS = 5
DEFAULT_WARMUP = 1
DEFAULT_COMPARE = "none"
DEFAULT_COMPARE_OPTIMIZE = "auto"
DEFAULT_OPTIMIZE = "auto"
DEFAULT_SCENE = "distill"
DEFAULT_VERBOSE = False
#: ``--language``：输出语言。``auto`` 见 :func:`resolve_language`。
LANGUAGE_MODES = ("auto", "en", "zh-cn")
DEFAULT_LANGUAGE = "auto"

# ── 遍历调优（--tune） ──────────────────────────────────────────────────────
#: 三轴的取值与**有序清单**都由 :mod:`pyqcd.einsum._tune_axes` 给出（唯一声明处）：
#: 顺序即执行顺序（共同前缀 → 分支专属 → 尾部），不是笛卡尔积。
TUNE_KERNEL_FORCE = _tune_axes.TUNE_KERNEL_FORCE
TUNE_AS_EQUIVALENCE = _tune_axes.TUNE_AS_EQUIVALENCE
TUNE_OPTIMIZERS = _tune_axes.TUNE_OPTIMIZERS
#: 组合数**硬上限**：超了直接报用法错误，**绝不静默截断**——截断会让"最优"变成
#: "某个被截断的网格里最好"，而报告字面上完全看不出这一点。
TUNE_MAX_COMBOS = 30
#: 被 ``--tune`` 自行遍历的三维（报告与警告文案要用它说明"为什么你给的值被忽略"）。
TUNE_AXES = "kernel_force × as_equivalence × optimize"
#: ``--tune-timeout``：**每一次** ``contract()`` 调用（warmup 与每一轮都算）的上限
#: 秒数。``<= 0`` = 不设上限。超时的那一组记 ``timeout``、只打印、继续下一组。
#: 它是**尽力而为（best-effort）上限**，不是硬超时：能打断 Python 侧长循环，
#: 打不断原生 C/CUDA 代码（详见 :func:`_guarded_call` 与模块 docstring）。
#: 缺省 **30**：一轮 22/21 组（B1/B2、B3）每组 ``1+warmup+rounds`` 次调用，
#: 上限取得过大时"某组卡死"就会把一次遍历拖成半小时；30 秒足够覆盖本仓库已实测的
#: 生产尺寸（最大一组约 3 s），而真正打不断的原生调用无论取多小都得等它返回。
DEFAULT_TUNE_TIMEOUT = 30.0
#: ``--tune-budget``：整轮遍历的上限秒数，``0`` = 不限。用尽即停止遍历并报"已测 N/M"。
DEFAULT_TUNE_BUDGET = 0.0
#: 预判剪枝时**问规划器**的时间预算（秒）：`contract_path` 只规划、不收缩，但
#: ``optimal`` 的穷举是纯 Python，本机这条 10 算子方程实测 **38 s**。问代价不该比
#: 被测对象还贵，超过这个预算就放弃预判、照旧跑那一组（旧行为）。
#: 2 秒足够覆盖 greedy/dp/branch_* 的全部实测值（最大 18 ms），又远小于一次收缩。
PLAN_PROBE_SECONDS = 2.0
#: ``--tune`` 的**取值**（2026-11-18 用户裁定）：它带取值，不是开关。``all`` = 完整模式
#: （分支判定 + 22/21 组遍历 + 剪枝 + 参照校验），``pass`` = **快速通过模式**（强制
#: ``B3`` 的 21 组、无 ``topology``、``as_equivalence`` 全 ``false``、不跑参照、
#: 不判 MISMATCH）。
TUNE_MODE_ALL = "all"
TUNE_MODE_PASS = "pass"
TUNE_MODE_OFF = "false"
TUNE_MODES = (TUNE_MODE_ALL, TUNE_MODE_PASS)
#: ``pass`` 模式的表尾注记：**固定原文**，两种语言逐字相同（它是"这一轮没做数值校验"
#: 的唯一标记，翻译掉脚本就抓不到）。
TUNE_PASS_SKIP_NOTE = "pass 模式：跳过数值校验"
#: 初始化过程输出的前缀与方程显示上限（**机器可读**、不跟 ``--language`` 翻译）。
TUNE_INIT_PREFIX = "tune init : "
TUNE_INIT_EQUATION_CHARS = 60


# ── 输出语言（--language） ──────────────────────────────────────────────────
#: 与 ``_specialize.LANGUAGE_ENV`` 同一个环境变量名——库内只有一套语言开关。
LANGUAGE_ENV = "PYQCD_EINSUM_LANG"

#: 文案表：``key -> {"en": ..., "zh": ...}``。缺 key 或缺语言**直接报错**，
#: 不静默返回 key 本身——那样会以"看起来很正常的英文"混进中文输出里。
MESSAGES: dict[str, dict[str, str]] = {
    # --help：description
    "help.description": {
        "en": ("The only sanctioned entry point for single-equation qcd-einsum tests "
               "(mandatory, since 2026-11-15).\n"
               "Give an equation + shapes + dtype + device; get: whether it is within "
               "tolerance,\n"
               "the timing, the reference timing, memory/VRAM footprint, and which "
               "native route ran.\n"
               "Do not replace this entry with /tmp scripts or one-off python -c; "
               "conclusions from those\n"
               "are not accepted. The only exception is the full-suite regression "
               "python -m pyqcd.testing."),
        "zh": ("qcd-einsum 单方程测试的唯一合法接口（强制，2026-11-15 起）。\n"
               "给方程 + 形状 + dtype + 设备，输出：是否在容差内、耗时、参照耗时、\n"
               "内存/显存占用、走的哪条原生路线。\n"
               "禁止用 /tmp 脚本或一次性 python -c 替代本入口；违反者的结论不予采信。\n"
               "唯一例外是整库回归 python -m pyqcd.testing。"),
    },
    # --help：epilog
    "help.epilog": {
        "en": ("Examples:\n"
               "  python -m pyqcd.einsum.test --device gpu --dtype complex128 \\\n"
               "      --equation 'ambn,cgdh,ekfl,oipj,ce,mo,gi,bdf,np,hjl->ak' \\\n"
               "      --shape '[400,4,1,1],[4,4,9,1],[4,400,400,400],[4,4,9,9],"
               "[4,4],[4,4],[4,4],[1,9,400],[1,9],[1,9,400]->[400,400]' \\\n"
               "      --atol 1e-12 --backend cupy --verbose true --compare opt_einsum\n\n"
               "  # without --shape, production sizes are inferred from corpus roles\n"
               "  python -m pyqcd.einsum.test --equation 'cadb,menf,ac,em,bd,fn->' "
               "--scene distill\n\n"
               "Calibration: --atol is a **relative** tolerance (normalised by "
               "max|expected|); timing\n"
               "discards --warmup calls then takes the median of --rounds; warmup 0 is "
               "refused.\n"
               "Specialisation: --specialize auto (follows PYQCD_EINSUM_SPECIALIZE) / "
               "true (force\n"
               "specialised) / false (force generic — the first tool when debugging "
               "native kernels).\n"
               "Reference optimiser: --compare-optimize affects the reference only (the "
               "main path follows\n"
               "--optimize); default auto = legacy behaviour (numpy→greedy, "
               "opt_einsum→greedy or follow\n"
               "--optimize). numpy accepts only greedy/optimal, opt_einsum accepts "
               "dp/branch-* etc.; names\n"
               "are never remapped across libraries, unsupported values exit with "
               "code 2.\n"
               "Output: one PASS line with pyqcd timing, reference timing, ratio and "
               "memory delta;\n"
               "--verbose true adds a per-item breakdown (specialize effective value / "
               "ref time / memory /\n"
               "peak / pyqcd origin).\n"
               "Memory is sampled only outside the timed loop (before/after operand "
               "construction, at the\n"
               "end), so it never perturbs timings.\n"
               "Language: --language auto (follows PYQCD_EINSUM_LANG; Chinese when it "
               "is unset) / en / zh-cn.\n"
               "Equivalence report: --equivalence auto (follows "
               "PYQCD_EINSUM_EQUIVALENCE, on by default) / true / false. When the "
               "equation misses\nevery specialised path but an equivalent spelling hits "
               "one, --verbose true lists the tensor-side\nand result-side adjustments "
               "needed to use it. Report only: contract() behaviour, timings and\n"
               "numbers are never changed.\n"
               "Exit codes: 0 pass / 1 numeric or shape mismatch / 2 usage or "
               "environment error.\n"),
        "zh": ("示例:\n"
               "  python -m pyqcd.einsum.test --device gpu --dtype complex128 \\\n"
               "      --equation 'ambn,cgdh,ekfl,oipj,ce,mo,gi,bdf,np,hjl->ak' \\\n"
               "      --shape '[400,4,1,1],[4,4,9,1],[4,400,400,400],[4,4,9,9],"
               "[4,4],[4,4],[4,4],[1,9,400],[1,9],[1,9,400]->[400,400]' \\\n"
               "      --atol 1e-12 --backend cupy --verbose true --compare opt_einsum\n\n"
               "  # 不写 --shape 时按语料 roles 反推生产尺寸\n"
               "  python -m pyqcd.einsum.test --equation 'cadb,menf,ac,em,bd,fn->' "
               "--scene distill\n\n"
               "口径: --atol 是**相对**容差（按 max|expected| 归一）；耗时先丢 --warmup "
               "次再取 --rounds 次中位，warmup 不允许 0。\n"
               "对照: --specialize auto（跟随 PYQCD_EINSUM_SPECIALIZE）/ true（强制特化）"
               "/ false（强制通用，排查原生内核问题的第一手段）。\n"
               "参照优化器: --compare-optimize 只管参照（主路径看 --optimize）；"
               "缺省 auto = 沿用旧行为（numpy→greedy，opt_einsum→greedy 或跟随 "
               "--optimize）。numpy 只认 greedy/optimal，opt_einsum 认 "
               "dp/branch-* 等；名字不跨库重映射，不支持即退出码 2。\n"
               "输出: 默认一行 PASS 里带 pyqcd 耗时、参照耗时、比值与内存增量；"
               "--verbose true 另给逐项明细（含 specialize 实际取值 / ref time / memory / "
               "peak / pyqcd 来源）。\n"
               "内存采样只在计时循环之外（构造前/构造后/收尾），不影响计时。\n"
               "语言: --language auto（跟随 PYQCD_EINSUM_LANG，未设时为中文）/ en / "
               "zh-cn。\n"
               "等价映射: --equivalence auto（跟随 PYQCD_EINSUM_EQUIVALENCE，缺省开启）"
               "/ true / false。方程未命中任何特化路径、但存在能命中"
               "的等价写法时，--verbose true 会给出改用该写法所需"
               "的输入侧张量调整与输出侧结果回填。"
               "只做报告：不改 contract() 行为、不改耗时、不改任何数值。\n"
               "退出码: 0 通过 / 1 数值或形状不符 / 2 参数或环境错误。\n"),
    },
    # --help：各选项
    "help.equation": {
        "en": "Einstein equation; the explicit '->' is required",
        "zh": "Einstein 方程，必须带显式 '->'",
    },
    "help.shape": {
        "en": ("Shapes like '[a..],[b..],...->[c..]'. When omitted, production sizes "
               "are inferred from the corpus roles via --scene (an equation absent "
               "from the corpus is an error)"),
        "zh": ("形状，形如 '[a..],[b..],...->[c..]'。省略时按 --scene "
               "从语料 roles 反推生产尺寸（语料里没有该方程则报错）"),
    },
    "help.scene": {
        "en": ("Size profile used when --shape is omitted (distill / distill_curr / "
               "scene / full), default {default}"),
        "zh": ("省略 --shape 时的尺寸档（distill / distill_curr / "
               "scene / full），缺省 {default}"),
    },
    "help.device": {
        "en": ("Where the arrays live: cpu→numpy, gpu/cuda→cupy; "
               "`cuda` here is a DEVICE name (= gpu), not a backend; "
               "backends are chosen by --backend. Default {default}"),
        "zh": ("数组落在哪：cpu→numpy，gpu/cuda→cupy；此处 cuda 是设备名"
               "（等价于 gpu），不是后端名；后端由 --backend 选择。缺省 {default}"),
    },
    "help.dtype": {
        "en": "Four dtypes on equal footing, default {default}",
        "zh": "四类平权，缺省 {default}",
    },
    "help.backend": {
        "en": ("Passed straight to contract(backend=...). auto lets the library "
               "decide; numpy/cupy/torch pick the array library. Native "
               "specialisation is selected by --kernel-force "
               "(dedicated/topology), NOT by --backend. Default {default}"),
        "zh": ("传给 contract(backend=...)。auto 让库自判；numpy/cupy/torch 指定"
               "数组库。原生特化由 --kernel-force（dedicated/topology）选择，"
               "并非 --backend。缺省 {default}"),
    },
    "help.compare": {
        "en": ("Reference implementation: numpy=np.einsum(optimize=...), "
               "opt_einsum=oe.contract(optimize=...), backend=cross-check on the other "
               "backend, none=run without comparing. Default {default}"),
        "zh": ("参照实现：numpy=np.einsum(optimize=...)、"
               "opt_einsum=oe.contract(optimize=...)、"
               "backend=另一后端互证、none=只跑不比。"
               f"缺省 {DEFAULT_COMPARE}"),
    },
    "help.compare_optimize": {
        "en": ("Contraction-order optimiser for **the reference only** (the main path "
               "still follows --optimize). auto=legacy behaviour: numpy uses greedy, "
               "opt_einsum uses greedy or follows --optimize (default); explicit values "
               "are handed to the reference as-is. The two reference paths use "
               "different namespaces and are **never remapped across libraries**: "
               "numpy accepts only greedy/optimal (dp/branch-* is a usage error), "
               "opt_einsum accepts dp/greedy/optimal/branch-all/branch-2/branch-1 "
               "(validated against its paths registry). Default {default}"),
        "zh": ("**只影响参照实现**的收缩序优化器（主路径仍由 --optimize "
               "决定）。auto=沿用旧行为：numpy 用 greedy、opt_einsum 用 "
               "greedy 或跟随 --optimize（缺省）；显式值原样交给参照。"
               "两条参照路径名字不同且**不做跨库重映射**："
               "numpy 只认 greedy/optimal（dp/branch-* 会报用法错误），"
               "opt_einsum 认 dp/greedy/optimal/branch-all/branch-2/"
               "branch-1（按其 paths 注册表校验）。"
               "缺省 {default}"),
    },
    "help.specialize": {
        "en": ("Tri-state specialised kernel. auto=follow PYQCD_EINSUM_SPECIALIZE "
               "(current behaviour, default); true=force the specialised path, never "
               "fall back silently; false=force the generic path, the first tool when "
               "debugging native kernels (kernel should be empty under --verbose true). "
               "Default {default}; applies to both the main path and the "
               "--compare backend reference path"),
        "zh": ("专用内核三态。auto=跟随环境变量 "
               "PYQCD_EINSUM_SPECIALIZE（现有行为，缺省）；"
               "true=强制走特化路径，不让它静默回落；"
               "false=强制通用路径，是排查原生内核问题的第一手段"
               "（--verbose true 中 kernel 应为空）。缺省 "
               "{default}；主路径与 --compare backend 的"
               "参照路径同时生效"),
    },
    "help.kernel": {
        "en": ("Which kernel to FORCE (there is no 'auto' value: the route is chosen by "
               "measurement — --tune writes the winner into the profile table — not "
               "guessed at dispatch time). generic = skip the specialised kernels "
               "entirely (no warning), and skip the corpus shape lookup and the "
               "equivalence rewrite with them — both exist only to reach a specialised "
               "form, so the generic notice is not printed either; dedicated / topology "
               "= allow exactly that kernel of the operand backend (numpy -> kupl, cupy "
               "-> cuda), comma separated to allow both, tried in the order written; "
               "backend = call the backend's own einsum and skip the specialised "
               "dispatch (cannot be combined with other labels). Forcing a kernel makes "
               "it really try once (its negative cache is bypassed) and, when it does "
               "not run, raises KernelUnavailableError with a non-zero exit code — it "
               "never falls back to the generic path. An unknown value (including the "
               "removed 'auto') exits with code 2 and names --tune plus the legal "
               "values. An explicit --kernel-force wins over the {env} environment "
               "variable; leaving the option out runs with generic but is NOT the same "
               "as writing generic (the kernel layer still sees 'not given', so a tuned "
               "profile row such as kernel=backend is applied). Main path only: the "
               "--compare backend "
               "reference keeps its own dispatch (forcing a kernel on the other backend "
               "cannot work). Default {default}"),
        "zh": ("强制跑哪个内核（没有 auto：路线由实测决定——--tune 把胜出的那一组"
               "写进配置表，而不是运行时猜分派次序）。generic = "
               "完全跳过专用内核（不产生 warning），并连语料形态匹配与等价改写一起"
               "跳过——两者都只为命中特化形态，因此也不再打通用路径提示；dedicated / "
               "topology = 只允许"
               "当前后端的对应内核（numpy → kupl，cupy → cuda），可用逗号给两个、"
               "按书写次序尝试；backend = 直接调后端自己的 einsum、不走专用分派"
               "（不可与其它标签混用）。"
               "强制内核时会真试一次（旁路负缓存），跑不成即抛 "
               "KernelUnavailableError 并以非零退出码结束——绝不回落成通用路径。"
               "未知取值（含已删除的 auto）以退出码 2 结束，消息点名 --tune 并列出"
               "合法集合。显式 --kernel-force 优先于环境变量 {env}；不写该选项时按 "
               "generic 执行，但那不等于显式写了 generic（内核层仍视为未指定，已 tune 的"
               "配置表里 kernel=backend 这类行因此照常套用）。"
               "只作用于主路径：--compare backend 的参照仍走自己的分派"
               "（在另一个后端上强制内核必然不可用）。缺省 {default}"),
    },
    "help.optimize": {
        "en": ("Contraction-order optimiser: one of {values}. 'auto' is not a "
               "value - it is the built-in default selector (used when the option "
               "is omitted), so writing it is a usage error."),
        "zh": "收缩序优化器：{values} 之一。auto 不是可写取值——它是内置的缺省选择器"
              "（不写该选项时就用它），显式写它属于用法错误。",
    },
    "help.atol": {
        "en": "**Relative** tolerance (normalised by max|expected|). Default {default}",
        "zh": "**相对**容差（按 max|expected| 归一）。缺省 {default}",
    },
    "help.rtol": {
        "en": "Extra relative margin. Default {default}",
        "zh": "附加的相对余量。缺省 {default}",
    },
    "help.rounds": {
        "en": "Timing rounds (median is taken). Default {default}",
        "zh": "计时轮数（取中位）。缺省 {default}",
    },
    "help.warmup": {
        "en": "Warmup calls to discard (0 is not allowed). Default {default}",
        "zh": "丢弃的预热次数（不可为 0）。缺省 {default}",
    },
    "help.seed": {
        "en": "Random seed for the operands (every path sees the same data). Default 0",
        "zh": "操作数随机种子（各路径看到同一批数据）。缺省 0",
    },
    "help.verbose": {
        "en": "When true, print the per-item breakdown to stderr. Default false",
        "zh": "true 时打印逐项明细到 stderr。缺省 false",
    },
    "help.json": {
        "en": "Write the full result to this JSON path",
        "zh": "把完整结果写到该 JSON 路径",
    },
    "help.language": {
        "en": ("Output language of this entry point: auto (follow {env}; Chinese when "
               "it is unset), en, zh-cn. Translation only — it never changes results, "
               "the --compare choice, or any environment variable. Default {default}"),
        "zh": ("本入口的输出语言：auto（跟随 {env}，未设时为中文）、en、zh-cn。"
               "只影响输出文案——不改任何计算结果、不改 --compare 的选择、"
               "不写任何环境变量。缺省 {default}"),
    },
    "help.as_equivalence": {
        "en": ("Apply an equivalent spelling automatically. Kinds of value: auto=follow "
               "{env} (on by default, only 0/false/off/no turns it off); true=rewrite "
               "when a mapping exists; false=never; force=rewrite as true does, but "
               "**ignore the negative mark**.\n"
               "The four states are not three: a lookup that found nothing leaves a "
               "cross-process mark on this equation, and plain true honours it (so the "
               "mapping is not looked up again for that equation). force ignores that "
               "mark, looks the mapping up again and refreshes the mark with this "
               "lookup's verdict — use it when the mapping tables changed and the old "
               "'nothing here' verdict is stale.\n"
               "When the rewrite happens, the operands are permuted, the equivalent "
               "spelling is contracted and the result is transposed back — the returned "
               "array is still the answer to the ORIGINAL equation (same shape, same "
               "dtype). Skipped for out=, extra kwargs, element-wise equations and "
               "explicit optimizer paths. Default {default}"),
        "zh": ("自动施加等价写法。取值有四档：auto=跟随 {env}（缺省开启，只有 "
               "0/false/off/no 才关闭）；true=存在映射就改写；false=不改写；"
               "**force=与 true 一样改写，但忽略负标记**。\n"
               "四档不是三档：查过却没有映射时会在该方程上留下跨进程负标记，"
               "普通 true 会尊重它（此后同一方程不再重查）；force 无视这个标记，"
               "真的再查一次并用本次结论刷新标记——映射表更新过、旧的"
               "「这里没有」已过期时用它。\n"
               "改写发生时：重排操作数、按等价写法收缩、再把结果转回原输出轴序，"
               "返回的数组仍是**原方程**的答案（形状与 dtype 都不变）。"
               "out=、额外 kwargs、逐元素方程与显式优化器路径不施加。缺省 {default}"),
    },
    "help.tune": {
        "en": ("Mode argument, not a switch. Leave it empty (bare --tune), or write "
               "true, True or all = the FULL sweep described below (branch probe + 22/21 "
               "combinations + min_wall_ms pruning + numeric self-check). Write pass = "
               "FAST PASS mode: the branch is forced to B3 (21 combinations, no "
               "topology), as_equivalence is pinned to false, every --compare feature is "
               "disabled (no reference is computed, rel stays '-', no MISMATCH verdict; "
               "the table ends with 'pass mode: numerical check skipped'), and the side "
               "features (--explain/--verbose breakdowns, the specialize notices, the "
               "extra backend/dtype/device probe, the reference) are skipped so the run "
               "is fast and never blocks. Leave it out (or write false / an empty string) "
               "= run no sweep. Any other value is a usage error (exit code 2), never a "
               "silent 'off'.\n"
               "Before the sweep starts, progress lines prefixed 'tune init :' are "
               "printed one per stage (mode/shapes, branch probe start+done, kernel "
               "admission start+done, session start, first group start+done), each with "
               "its own start and end and with timings in whole ms, so a hang shows "
               "exactly which stage it is stuck in. In pass mode the branch probe lines "
               "read 'skipped (pass mode)'.\n"
               "Sweep kernel_force × as_equivalence × optimize and report the median of "
               "THIS run for every combination. The equation, shapes, dtype, device and "
               "backend are FIXED (they are this run's invariants); --compare defaults "
               "to OFF under --tune (no reference is timed unless you pass --compare "
               "explicitly, in which case the main path times it once). The three swept "
               "axes are supplied by "
               "--tune itself, so an explicit --kernel-force / --as-equivalence / "
               "--optimize outside the grid (--kernel-force other than generic, the "
               "other two other than auto) is IGNORED with one warning line per option "
               "on stderr (the sweep still runs and the exit code stays 0). The "
               "combinations are NOT a cartesian product: they form an ordered list per "
               "branch (common prefix dedicated -> generic, then one branch-specific "
               "topology combination, then the backend tail; inside each kernel "
               "greedy/branch_1/branch_2/dp/branch_all/optimal/none) with 22 "
               "combinations for B1/B2 and 21 for B3 (the branch is probed from whether "
               "the original equation and its equivalent spelling can be specialised; "
               "probing does not time anything). Hard cap 30; more is a usage error, "
               "never a silent truncation. Each combination reuses --warmup / --rounds "
               "and the median is what was measured now — a single-run measurement, NOT "
               "a benchmark; that median IS the wall_ms of the winning row.\n"
               "min_wall_ms pruning: the bound starts at --tune-timeout; a combination "
               "whose single contract() call is slower than the current bound is "
               "TERMINATED (reported as 'pruned', never written) and the sweep "
               "continues; any combination that measures faster lowers the bound "
               "immediately, so every later combination is pruned against it. The best "
               "row is the smallest recorded wall_ms.\n"
               "The whole sweep runs inside the cache's tune session, so the "
               "configuration table is neither read nor written while measuring. After "
               "the sweep ONLY the winning combination is written: if it came from "
               "as_equivalence=false its configuration is written under the original "
               "equation; if it came from as_equivalence=true a full configuration row "
               "is written under target_equation AND a second row under the original "
               "equation carries only the equivalence payload (operand_order / "
               "operand_permutations / output_permutation) plus a note that the "
               "rewritten configuration is faster. Each write deletes every other row "
               "of that key (one row per key), and the number of rows written and "
               "deleted is printed.\n"
               "The table goes to stderr (existing lines unchanged): header lines + the "
               "5 fastest rows + the fastest row of EVERY kernel_force value + the "
               "written/deleted rows + the best row; without --verbose only the header, "
               "the written/deleted rows and the best line. A combination that raises is "
               "marked error=<type>, one that exceeds --tune-timeout is marked timeout, "
               "one whose rel exceeds --atol/--rtol is marked MISMATCH: all three are "
               "printed, the sweep continues, and none of them can win or be written. "
               "Default off"),
        "zh": ("**带取值的模式参数，不是开关**。留空（裸 --tune）或写 true / True / "
               "all = 下面的**完整模式**（分支判定 + 22/21 组遍历 + min_wall_ms 剪枝 + "
               "数值自检）；写 **pass = 快速通过模式**：强制 B3（21 组、无 topology）、"
               "as_equivalence 固定 false、禁用 --compare 的**全部**功能（不解析参照、"
               "不算 np.einsum、rel 一律 '-'、不做 MISMATCH 判定，表尾注"
               "「pass 模式：跳过数值校验」），并跳过与 tune 本身无关的功能"
               "（--explain/--verbose 的详解、专用路径提示、额外的 backend/dtype/device "
               "探测、参照）——专注快速且无阻塞。不给该选项（或写 false / 空串）= "
               "不跑遍历。其它取值一律用法错误（退出码 2），**绝不**静默当成 off。\n"
               "遍历开始之前先逐阶段打印 'tune init :' 过程行（mode 与规模、分支探测"
               "起止、内核准入起止、session 起止、首组起止），每步都有起有止、耗时用"
               "整数 ms——卡住时一眼看出卡在哪一步；pass 模式的分支探测行写 "
               "'skipped (pass mode)'。\n"
               "遍历 kernel_force × as_equivalence × optimize，逐组给出**本次**实测中位。"
               "方程、形状、dtype、设备与后端**固定不动**（它们是本次的不变量）；"
               "--compare 在 --tune 下**缺省关闭**（不给就不跑参考基准；显式给出时由"
               "主路径照旧跑一次）。被遍历的三维由 --tune 自己"
               "提供：显式 --kernel-force / --as-equivalence / --optimize 只要不在网格里"
               "（--kernel-force 非 generic，另两轴非 auto），"
               "就会**被忽略**，每个参数在 stderr 上给一行警告（遍历照常进行，退出码仍为 "
               "0）。组合**不是笛卡尔积**：按分支给出**有序清单**（共同前缀 dedicated "
               "→ generic，再一条分支专属 topology，最后 backend 尾部；每个内核内 "
               "greedy/branch_1/branch_2/dp/branch_all/optimal/none），B1/B2 = 22 组、"
               "B3 = 21 组（分支由「原方程能否特化、等价写法能否特化」探测得出，"
               "探测不计时）。硬上限 30，超出即用法错误，**绝不静默截断**。"
               "每组沿用 --warmup / --rounds，中位就是此刻测到的值——"
               "**单次运行的实测，不是基准**，那个中位**就是**胜出行里的 wall_ms。\n"
               "min_wall_ms 剪枝：上界初值 = --tune-timeout；某组的单次 contract() 耗时"
               "超过当前上界即**终止该组**（标 pruned、绝不写表）并继续下一组；任何一组"
               "测到更小耗时都**立刻**压低上界，后续所有组都用它剪枝。最优行就是记录到的"
               "最小 wall_ms。\n"
               "整轮遍历在缓存的 tune session 里跑，期间配置表**既不读也不写**。遍历结束"
               "只写**胜出的那一组**：胜出组 as_equivalence=false → 以原方程写该配置；"
               "胜出组 as_equivalence=true → 以 target_equation 为方程写一条完整性能配置，"
               "**另**以原方程写一条只含等价载荷（operand_order / operand_permutations / "
               "output_permutation）+ 备注「走等价转换后的配置性能更佳」的记录。每次写都会"
               "**删除该键的其它所有行**（同键只允许 1 行），写入与删除的行数都会打印。\n"
               "表格打到 stderr（既有输出行不动）：表头 + 最快 5 行 + 每个 kernel_force "
               "取值的最快一行 + 写入/删除行 + 最优行；不加 --verbose 时只给表头、"
               "写入/删除行与最优行。抛异常的组合标 error=<类型>，超过 --tune-timeout 的"
               "标 timeout，rel 超 --atol/--rtol 的标 MISMATCH：三者只打印、遍历继续，"
               "都不参与选优、也不写进配置表。缺省关闭"),
    },
    "help.tune_timeout": {
        "en": ("Upper bound, in seconds, for EACH contract() call of --tune (warmup and "
               "every round count). By default each combination runs in a FRESH "
               "interpreter, so this is a HARD bound: the parent kills the child, which "
               "means native C/CUDA code can really be interrupted (SIGALRM is only "
               "delivered between Python bytecodes, so in-process it cannot). The price "
               "is one process startup per combination (~126 ms; ~244 ms more when cupy "
               "is imported), which is NOT counted in wall_ms and is reported separately "
               "on the 'tune note :' line. PYQCD_EINSUM_TUNE_INPROCESS=1 goes back to "
               "the in-process best-effort route, which can interrupt long Python-side "
               "loops (e.g. the optimal planner's exhaustive search) but CANNOT "
               "INTERRUPT NATIVE C/CUDA code: a call stuck inside a native kernel runs "
               "to completion and is judged timeout only after it returns. On GPU make "
               "the bound large enough, or use --tune pass first (skips compare and "
               "equivalence rewriting). A combination whose call exceeds the bound is "
               "marked timeout, is reported and skipped, and the sweep continues: a "
               "timeout never aborts the run. The median is taken over all calls that "
               "stayed within the bound (a partial sample set never becomes a median). "
               "0 or negative disables the bound. A combination whose PLANNED flops "
               "already dwarf the current bound is reported as pruned WITHOUT being "
               "run, with 'plan-only estimate, not measured' in its reason. "
               "Default {default}"),
        "zh": ("--tune 的**每一次** contract() 调用的上限秒数（warmup 与每一轮都算）。"
               "**缺省每组跑在一个全新解释器里**，所以这是**硬上限**：父进程 kill 子进程，"
               "原生 C/CUDA 代码真的能被打断（SIGALRM 只在 Python 字节码之间派发，进程内"
               "做不到）。代价是每组一次进程启动（约 126 ms；导入 cupy 再加约 244 ms），"
               "这段开销**不计入 wall_ms**，单独报在 tune note 那一行。"
               "设 PYQCD_EINSUM_TUNE_INPROCESS=1 退回进程内尽力而为路线：那时能打断 "
               "Python 侧的长循环（例如 optimal 的穷举规划），但**打不断原生 C/CUDA 代码**"
               "——卡在原生内核里的调用会跑完，返回之后才被判为 timeout。"
               "GPU 上请把 --tune-timeout 取足够大，或先用 --tune pass 快速通过"
               "（跳过 compare 与等价改写）。任一次超过它 → 该组记 timeout、只打印并"
               "跳过、**绝不写配置表**，遍历继续（超时永远不中止整轮）。"
               "中位只由全部在上限内成功的调用构成（不拿部分样本凑中位）。"
               "0 或负数 = 不设上限。**规划出来的 FLOPs 已远超当前上界的组不会跑**，"
               "状态记 pruned，原因里写明 plan-only estimate, not measured。"
               "缺省 {default}"),
    },
    "help.tune_budget": {
        "en": ("Upper bound, in seconds, for the WHOLE --tune sweep. When it is used up "
               "the remaining combinations are skipped and the report says how many were "
               "measured out of the total. 0 = no bound. Default {default}"),
        "zh": ("整轮 --tune 遍历的上限秒数。用尽即跳过剩余组合，并在报告里给出"
               "「已测 N/M」。0 = 不限。缺省 {default}"),
    },
    "help.equivalence": {
        "en": ("Equivalence report tri-state. auto=follow {env} (on by default, which "
               "only 0/false/off/no turns off); true=always look for an equivalent "
               "spelling; false=never. When the equation misses every specialised path "
               "but an equivalent spelling hits one, --verbose true shows the "
               "tensor-side adjustments AND the result-side permutation needed to use "
               "it. Report only: it never changes contract() behaviour, timings or any "
               "number. Default {default}"),
        "zh": ("等价映射报告三态。auto=跟随 {env}（缺省开启，只有 "
               "0/false/off/no 才关闭）；true=总是查等价写法；false=不查。"
               "方程未命中任何特化路径、但存在能命中的等价写法时，"
               "--verbose true 会同时给出**输入侧的张量调整**与**输出侧的结果回填**。"
                "只做报告：不改 contract() 行为、不改耗时、不改任何数值。"
                "缺省 {default}"),
    },
    "help.cache": {
        "en": ("Einsum cache tri-state. auto=follow {env} (on by default; only "
               "0/false/off/no turns it off, the same rule as the library); true=force "
               "the cache on for this call; false=force it off, in which case the "
               "matchers and the planner neither read nor write any cache, so nothing "
               "is warmed up or poisoned. An explicit true/false wins over {env}. "
               "Caching only reuses pure-function results: numbers, shapes and routes "
               "are never changed. With an explicit true/false, --verbose true adds one "
               "machine-readable line 'cache    : enabled=<true|false>  hits=<n>  "
               "misses=<n>  hit_rate=<0.00|none>' (auto is identical to leaving the "
               "option out and prints no such line). Default {default}"),
        "zh": ("einsum 缓存三态。auto=跟随环境变量 {env}（缺省开启，与库同口径："
               "只有 0/false/off/no 才关闭）；true=本次强制开启；false=本次强制关闭，"
               "此时匹配器与规划器**既不读也不写**任何缓存，不会预热也不会污染统计。"
               "显式 true/false 压过 {env}。缓存只复用纯函数的结果：数值、形状与"
               "实际走的路线一概不变。显式给出 true/false 时，--verbose true 会追加"
               "一行机器可读统计 'cache    : enabled=<true|false>  hits=<n>  "
               "misses=<n>  hit_rate=<0.00|none>'（auto 与省略该选项等价，不写这一行）。"
               "缺省 {default}"),
    },
    "help.explain": {
        "en": ("First-call explanation: auto / true / false.\n"
               "auto follows --verbose (default); true prints the explanation even "
               "without --verbose; false never prints it.\n"
               "The first call in this process that does not hit the configuration "
               "profile appends three sections\n"
               "after the existing lines, chained with '->': plan (which form was hit, "
               "or which equivalent\n"
               "spelling exists), kernel (requested / attempted / ran) and why (the "
               "abandon reason, one of\n"
               "contiguity / dtype-policy / kernel-unavailable / profile-backend, or "
               "'kept: <label>').\n"
               "Every later call adds exactly one '[pyqcd.einsum] profile ...' line. "
               "The ms values shown\n"
               "are recorded values, never a measurement of this run. "
               "Default {default}"),
        "zh": ("首次详解：auto / true / false。\n"
               "auto 跟随 --verbose（缺省）；true 即使不开 --verbose 也打印；"
               "false 一律不打印。\n"
               "本次进程第一次、且未命中配置画像时，在既有行之后追加三段并用 → "
               "串因果：\n"
               "plan（命中了哪个形态，或存在哪条等价写法）、kernel（requested / "
               "attempted / ran）、\n"
               "why（放弃原因，四类之一：contiguity / dtype-policy / "
               "kernel-unavailable / profile-backend，\n"
               "或最终保留专用内核时的 kept: <具体标签>）。\n"
               "之后每次只加一行 [pyqcd.einsum] profile …。其中出现的 ms 一律是"
               "记录值，不是本次实测。缺省 {default}"),
    },
    "help.profile": {
        "en": ("Configuration-profile cache: auto / true / false, passed straight to "
               "contract(profile=...).\n"
               "auto follows {env} (on by default); true forces it on for this call; "
               "false forces it off,\n"
               "in which case nothing is read or written and no directory is created. "
               "The profile only\n"
               "proposes a configuration: the equation, dtype, shapes and every number "
               "stay untouched. Default {default}"),
        "zh": ("配置画像缓存：auto / true / false，原样交给 contract(profile=...)。\n"
               "auto 跟随 {env}（缺省开启）；true 本次强制开启；false 本次强制关闭，"
               "此时既不读也不写、\n"
               "连目录都不建。画像只**建议**配置：方程、dtype、形状与任何数值都不改。"
               "缺省 {default}"),
    },
    "help.profile_dir": {
        "en": ("Directory of the configuration profile (passed to "
               "contract(profile_dir=...)); default {env} or ./.cache"),
        "zh": ("配置画像目录（传给 contract(profile_dir=...)）；"
               "缺省 {env} 或 ./.cache"),
    },
    # --verbose true 的明细
    "detail.operands": {
        "en": "{n} operand(s), {elements:,} elements in total; output {out_shape} "
              "({out_elements:,} elements)",
        "zh": "{n} 个，共 {elements:,} 元素；输出 {out_shape} ({out_elements:,} 元素)",
    },
    "detail.specialize": {
        "en": "{requested} (effective {effective})",
        "zh": "{requested}（实际 {effective}）",
    },
    "detail.shape": {
        "en": "{n} operand shape(s), left to right, in the equation's own order",
        "zh": "共 {n} 个操作数形状（按方程中从左到右的顺序）",
    },
    "detail.shape_empty": {
        "en": "(no operands)",
        "zh": "（无操作数）",
    },
    "detail.route_generic": {
        "en": "generic (no native hit: Python steps / numpy.einsum / BLAS)",
        "zh": "通用（无原生命中：Python 分步 / numpy.einsum / BLAS）",
    },
    "detail.route_specialized": {
        "en": "specialized - native {symbol} (path={path})",
        "zh": "特化 · 原生 {symbol}（path={path}）",
    },
    "detail.route_backend": {
        "en": ("backend einsum (--kernel-force backend: the specialised dispatch "
               "never ran)"),
        "zh": ("后端 einsum（--kernel-force backend：未走专用内核分派）"),
    },
    "detail.route_joiner": {
        "en": " + ",
        "zh": " + ",
    },
    "detail.time": {
        "en": "{ms:.3f} ms (warmup={warmup}, rounds={rounds}, median)",
        "zh": "{ms:.3f} ms （warmup={warmup}，rounds={rounds}，中位）",
    },
    "detail.ref_opt": {
        "en": "requested {requested} → reference actually uses {shown}",
        "zh": "要 {requested} → 参照实际用 {shown}",
    },
    "detail.ref_opt_not_applicable": {
        "en": "not applicable",
        "zh": "不适用",
    },
    "detail.ratio": {
        "en": "  (pyqcd/reference = {ratio:.2f}x)",
        "zh": "  （pyqcd/参照 = {ratio:.2f}×）",
    },
    "detail.memory_inputs_rss": {
        "en": "input tensors +{mib:.0f} MiB RSS",
        "zh": "输入张量 +{mib:.0f} MiB RSS",
    },
    "detail.memory_inputs_gpu": {
        "en": "+{mib:.0f} MiB VRAM",
        "zh": "+{mib:.0f} MiB 显存",
    },
    "detail.memory_peak_rss_after": {
        "en": " (process total {mib:.0f} MiB)",
        "zh": "（进程共 {mib:.0f} MiB）",
    },
    "detail.memory_peak_gpu": {
        "en": "VRAM +{mib:.0f} MiB",
        "zh": "显存 +{mib:.0f} MiB",
    },
    "detail.memory_peak_pool": {
        "en": "(pool reserved {mib:.0f} MiB)",
        "zh": "（池保留 {mib:.0f} MiB）",
    },
    "detail.memory_peak_total": {
        "en": "total VRAM {mib:.0f} MiB",
        "zh": "总显存 {mib:.0f} MiB",
    },
    "detail.kernel_none": {
        "en": "no native hit (generic path)",
        "zh": "无原生命中（通用路径）",
    },
    "detail.joiner": {
        "en": ", ",
        "zh": "，",
    },
    # 等价映射（--equivalence）：只在「未命中特化 + 存在等价写法」时出现。
    # 键名固定 `equiv`，值按语言翻译；输入侧与输出侧**都必须出现**，
    # 缺失的一侧明确写 `-`——只报输入侧会让用户漏掉结果回填而算错。
    "detail.equiv_header": {
        "en": "an equivalent spelling exists ({kind})",
        "zh": "存在等价写法（{kind}）",
    },
    "detail.equiv_none": {
        "en": "none (this equation has no equivalent spelling known to land on a "
              "specialised form)",
        "zh": "无（该方程不存在可落到已知形式的等价写法）",
    },
    "detail.equiv_equation": {
        "en": "use this equation: {equation}",
        "zh": "方程：{equation}",
    },
    "detail.equiv_tensors": {
        "en": "tensors: {adjustments}   (per operand, in the rewritten equation's order)",
        "zh": "张量：{adjustments}（按改写后方程的操作数次序）",
    },
    "detail.equiv_result": {
        "en": "result: {perm}   (apply to the rewritten result to get the original "
              "answer; '-' means the rewritten axis order already matches)",
        "zh": "结果：{perm}（对改写后的结果做此转置才是原方程的答案；"
              "'-' 表示改写后轴序已一致）",
    },
    "detail.equiv_note": {
        "en": "note: convert the tensors exactly as above, then call contract() with the "
              "equivalent equation to reach the specialised path; the result is "
              "element-wise identical to the direct spelling",
        "zh": "说明：按上述调整后用等价方程调用，可走上特化路径；"
              "结果与直接写法逐元素相同",
    },
    "detail.equiv_unavailable": {
        "en": "equivalence layer unavailable ({exc})",
        "zh": "等价映射层不可用（{exc}）",
    },
    # 首次详解（--explain）：只在「本进程第一次且画像未命中」时出现的三段。
    # 段间用 `→` 串因果：形态 → 尝试 → 结论。`kept`/`contiguity`/`dtype-policy`/
    # `kernel-unavailable` 是**稳定前缀**（脚本可解析），两种语言逐字相同。
    "explain.plan_shape": {
        "en": "shape-specialised {name}  [{kind}] → declared kernels {kernels}",
        "zh": "形状特化 {name}  [{kind}] → 声明内核 {kernels}",
    },
    "explain.plan_corpus": {
        "en": "corpus form {name}  [{kind}] → declared kernels {kernels}",
        "zh": "语料形态 {name}  [{kind}] → 声明内核 {kernels}",
    },
    "explain.plan_none": {
        "en": "no specialised form (generic step-by-step path) → nothing to dispatch",
        "zh": "未命中任何特化形态（走通用分步路径） → 没有可下发的原生内核",
    },
    "explain.plan_equiv": {
        "en": "equivalent spelling exists ({kind}): {equation}   tensors: {adjustments}"
              "   result: {perm}",
        "zh": "存在等价写法（{kind}）：{equation}   张量：{adjustments}"
              "   结果：{perm}",
    },
    "explain.plan_equiv_unavailable": {
        "en": "equivalence layer unavailable ({exc})",
        "zh": "等价映射层不可用（{exc}）",
    },
    "explain.why_kept": {
        "en": "kept: {label}",
        "zh": "kept: {label}",
    },
    "explain.why_contiguity": {
        "en": "contiguity: {reason}",
        "zh": "contiguity: {reason}",
    },
    "explain.why_dtype": {
        "en": "dtype-policy: real dtype declined",
        "zh": "dtype-policy: real dtype declined",
    },
    "explain.why_unavailable": {
        "en": "kernel-unavailable: {reasons}",
        "zh": "kernel-unavailable: {reasons}",
    },
    "explain.why_profile": {
        "en": ("profile-backend: profile suggestion (not measured in this run) — the "
               "cache records backend faster ({backend_ms} vs {kernel_ms})"),
        "zh": ("profile-backend: profile 建议（非本次实测）——缓存记录 backend 更快"
               "（{backend_ms} vs {kernel_ms}）"),
    },
    "explain.why_none": {
        "en": "none: no specialised kernel was attempted in this call "
              "(requested={requested})",
        "zh": "none：本次没有尝试专用内核（requested={requested}）",
    },
    # 之后每次只打一行（`[pyqcd.einsum]` 前缀与 `→` 是稳定形态，脚本可解析）
    "explain.status_hit": {
        "en": "[pyqcd.einsum] profile hit → backend={backend} kernel={kernel}  "
              "(wall_ms={wall_ms})",
        "zh": "[pyqcd.einsum] profile 命中 → backend={backend} kernel={kernel} "
              "（wall_ms={wall_ms}）",
    },
    "explain.status_hit_inert": {
        "en": "[pyqcd.einsum] profile hit but not adopted → kernel={kernel} "
              "(not replayable; this call kept the original dispatch)",
        "zh": "[pyqcd.einsum] profile 命中但未采纳 → kernel={kernel} "
              "（该取值不可重放，本次仍走原分派）",
    },
    "explain.status_miss": {
        "en": "[pyqcd.einsum] profile miss → recorded (rows={rows})",
        "zh": "[pyqcd.einsum] profile 未命中 → 已记录（rows={rows}）",
    },
    "explain.status_off": {
        "en": "[pyqcd.einsum] profile off → replanned on every call",
        "zh": "[pyqcd.einsum] profile 关闭 → 每次重新规划",
    },
    "explain.status_unknown": {
        "en": "[pyqcd.einsum] profile unknown → key not reproducible "
              "(execution unaffected)",
        "zh": "[pyqcd.einsum] profile 未知 → 无法复现键（不影响本次执行）",
    },
    # 结果行与警告
    "result.fail_rel": {
        "en": "  rel={rel:.3e} > atol={atol:g}",
        "zh": "  rel={rel:.3e} > atol={atol:g}",
    },
    "result.fail_shape": {
        "en": "  actual shape={actual} expected={expected}",
        "zh": "  actual shape={actual} expected={expected}",
    },
    "result.error_line": {
        "en": "  error: {message}",
        "zh": "  error: {message}",
    },
    "result.warning": {
        "en": "warning: {message}",
        "zh": "warning: {message}",
    },
    "result.error": {
        "en": "error: {message}",
        "zh": "错误：{message}",
    },
    "result.warning_compare_optimize_inert": {
        "en": ("--compare-optimize {value} has no effect for --compare {compare} "
               "(this mode has no optimiser to set)"),
        "zh": ("--compare-optimize {value!r} 对 "
               "--compare {compare} 无效（该模式没有可设的优化器）"),
    },
    # --tune 的表格。表头/字段名与 `tune best :` 行是**机器可读**的稳定形态
    # （同 `kernel :`），不翻译；只有说明性文字（这一行）跟 --language 走。
    "tune.caption": {
        "en": ("{combos} combinations on the SAME equation/shape/dtype/device/backend; "
               "compare={compare} (off by default under --tune; given explicitly it is "
               "timed once on the main path); warmup={warmup}, rounds={rounds}, median "
               "of THIS run; rel vs {reference}. Single-run measurement of this "
               "process, NOT a benchmark."),
        "zh": ("{combos} 组，方程/形状/dtype/设备/后端全部固定；compare={compare}"
               "（--tune 下缺省关闭，显式给出时由主路径跑一次）；"
               "warmup={warmup}、rounds={rounds}，取**本次**中位；rel 对 {reference}。"
               "这是**单次运行的实测，不是基准**。"),
    },
    "tune.skip": {
        "en": ("excluded from best: MISMATCH={mismatch} error={error} timeout={timeout} "
               "(a mismatched, failed or timed-out combination never wins)"),
        "zh": ("不参与选优：MISMATCH={mismatch} error={error} timeout={timeout}"
               "（数值不符、跑不成或超时的组合一律不参与最优选择）"),
    },
    # ``min_wall_ms`` 剪枝的一句话摘要：逐组的原文本已由 ``tune comb`` 的
    # ``原因=`` 逐字给出（含 ``耗时大于min_wall_ms,此组被终止``），这里只报总数与
    # 当时的最终上界——"哪些组被砍了"在明细里，不在这里重算一遍。
    "tune.prune": {
        "en": ("{pruned} combination(s) terminated early (slower than min_wall_ms); "
               "final min_wall_ms={min_wall_ms}; set by {config}"),
        "zh": ("{pruned} 组因超过 min_wall_ms 被提前终止；最终上界 min_wall_ms={min_wall_ms}，"
               "由 {config} 设定"),
    },
    "tune.none": {
        "en": "no usable combination: every one errored, timed out or mismatched",
        "zh": "没有任何可用组合：全部报错、超时或数值不符",
    },
    "tune.ignored": {
        "en": ("--tune sweeps {swept} itself, so the value you gave is ignored: {given!r}"),
        "zh": "--tune 会自行遍历 {swept}，已忽略你给的 {given!r}",
    },
    # pass 模式禁用 --compare 时的那一行警告（`--compare` 与 pass 天然冲突）。
    "tune.pass_compare_disabled": {
        "en": ("--tune pass disables every --compare feature (no reference, rel stays "
               "'-', no MISMATCH verdict), so the given --compare {value} is ignored"),
        "zh": ("--tune pass 禁用 --compare 的全部功能（不算参照、rel 一律 '-'、"
               "不判 MISMATCH），已忽略你给的 --compare {value}"),
    },
    "tune.signal_note": {
        "en": ("in-process route (PYQCD_EINSUM_TUNE_INPROCESS=1): --tune-timeout is a "
               "best-effort upper bound here: it can interrupt long Python-side loops but "
               "cannot interrupt native C/CUDA code (a call stuck inside a native kernel "
               "runs to completion and is judged timeout only after it returns); on GPU "
               "make the bound large enough, or use --tune pass first (skips compare and "
               "equivalence rewriting). Drop the environment variable to get the hard "
               "subprocess bound back"),
        "zh": ("进程内路线（PYQCD_EINSUM_TUNE_INPROCESS=1）：--tune-timeout 是尽力而为上限"
               "（best-effort），能打断 Python 侧长循环，但打不断原生 C/CUDA 代码（卡在"
               "原生内核里的调用会跑完，返回之后才被判为 timeout）；GPU 上请把该上限取"
               "足够大，或先用 --tune pass 快速通过（跳过 compare 与等价改写）。"
               "去掉该环境变量即可回到子进程硬上限"),
    },
    "tune.posthoc_note": {
        "en": ("no timer available here: each call is timed first and judged "
               "afterwards, so a slow one still runs to completion"),
        "zh": ("本环境装不上定时器：每次调用先跑完再按耗时判定，超时的那次仍会跑完"),
    },
    "tune.subprocess_note": {
        "en": ("isolation=subprocess: each combination runs in a FRESH interpreter, so "
               "--tune-timeout really interrupts native C/CUDA code (the parent kills "
               "the child); unlike the in-process SIGALRM route this is a hard bound. "
               "The price is measurement overhead that is NOT part of wall_ms: "
               "{groups} process startup(s), median {median} ms / total {total} ms, "
               "plus a one-off {stage_mib} MiB operand staging in {stage_s} s. Compare "
               "wall_ms only with other subprocess runs, never with an in-process "
               "measurement. Set PYQCD_EINSUM_TUNE_INPROCESS=1 to go back to the "
               "best-effort in-process route (cannot interrupt native code)"),
        "zh": ("isolation=subprocess：每一组都在**全新解释器**里跑，所以 --tune-timeout "
               "能真正打断原生 C/CUDA 代码（父进程 kill 子进程），这是硬上限，与进程内 "
               "SIGALRM 路线不同。代价是**不计入 wall_ms** 的测量开销：{groups} 次进程"
               "启动，中位 {median} ms / 合计 {total} ms，另有一次 {stage_mib} MiB 的"
               "操作数暂存（{stage_s} s）。wall_ms 只能与其它子进程运行比较，不能与"
               "进程内测量混着比。设 PYQCD_EINSUM_TUNE_INPROCESS=1 可退回进程内"
               "尽力而为路线（打不断原生代码）"),
    },
    "tune.predicted": {
        "en": ("{count} combination(s) were NOT run: planned FLOPs alone already exceed "
               "the current min_wall_ms by a wide margin (rate {rate} flop/s from "
               "{source}). These are estimates, not measurements: they are reported as "
               "pruned but no wall_ms was taken"),
        "zh": ("{count} 组**没有跑**：规划出来的 FLOPs 已明显超过当前 min_wall_ms"
               "（吞吐 {rate} flop/s，取自 {source}）。这是**估算不是实测**：状态记为 "
               "pruned 但没有取到任何 wall_ms"),
    },
    "tune.stop": {
        "en": ("measured {measured}/{total} ({skipped} combination(s) skipped): "
               "--tune-budget {budget:g}s used up"),
        "zh": ("已测 {measured}/{total}（跳过 {skipped} 组）：--tune-budget {budget:g}s 已用尽"),
    },
    "tune.api_missing": {
        "en": ("_profile_cache.write_winner() is unavailable, so the winning "
               "configuration was NOT written (the sweep itself is unaffected)"),
        "zh": ("_profile_cache.write_winner() 不可用，胜出配置**没有**写进配置表"
               "（遍历本身不受影响）"),
    },
    "tune.profile_off": {
        "en": ("the configuration table is off for this run (--profile false or "
               "PYQCD_EINSUM_PROFILE=0): the winning configuration was NOT written "
               "(this run neither reads nor writes it)"),
        "zh": ("本次运行配置表关闭（--profile false 或 PYQCD_EINSUM_PROFILE=0）："
               "胜出配置**没有**写进配置表（本次运行既不读也不写它）"),
    },
    "tune.write_error": {
        "en": "writing the configuration table failed: {message}",
        "zh": "写配置表失败：{message}",
    },
    # §12.3：临时关闭缓存的提示必须逐条完整输出（tune 模式不受任何去重控制）。
    "tune.session_on": {
        "en": ("the configuration table is temporarily closed for the whole sweep: "
               "nothing is read or written while measuring (tune_session is active)"),
        "zh": ("整轮遍历期间配置表**已临时关闭**：测量时既不读也不写"
               "（tune_session 生效）"),
    },
    "tune.session_off": {
        "en": ("tune_session() is unavailable, so the configuration table was NOT "
               "closed while measuring: a historical hit can still be applied to a "
               "combination (the sweep itself is unaffected)"),
        "zh": ("tune_session() 不可用，测量期间配置表**没有**关闭：历史命中仍可能被"
               "施加到某一组上（遍历本身不受影响）"),
    },
    # §12.2：是否因"同键多行"做过清理（旧行全部删除，只留写下的那 1 行）。
    "tune.duplicate_cleanup": {
        "en": ("duplicate rows for one key: {rows} row(s) sat on the same key; "
               "cleaned up to one-row-per-key ({removed} old row(s) deleted)"),
        "zh": ("同键多行：同一个键上有 {rows} 行，已按「同键一行」清理"
               "（删除 {removed} 行旧记录，只保留写下的 1 行）"),
    },
    # 用法/环境错误
    "error.shape_empty": {
        "en": "--shape must not be empty",
        "zh": "--shape 不能为空",
    },
    "error.shape_count": {
        "en": "--shape gave {given} operand shape(s) but the equation has "
              "{expected} operand(s)",
        "zh": "--shape 给了 {given} 个操作数形状，但方程有 {expected} 个操作数",
    },
    "error.shape_rank": {
        "en": "--shape operand {term!r} has {rank} label(s) but the given shape "
              "is {shape} (rank {given})",
        "zh": "--shape 中操作数 {term!r} 有 {rank} 个标签，但给定形状是 "
              "{shape}（秩 {given}）",
    },
    "error.shape_conflict": {
        "en": "label {label!r} has conflicting lengths across operands: "
              "{first} vs {second}",
        "zh": "标签 {label!r} 在各操作数里的长度冲突：{first} vs {second}",
    },
    "error.json_unwritable": {
        "en": "cannot write --json to {path!r}: {exc}",
        "zh": "无法把 --json 写到 {path!r}：{exc}",
    },
    "error.shape_equation_unparsable": {
        "en": "cannot derive the output shape from the equation {equation!r}: {exc}",
        "zh": "无法从方程 {equation!r} 推导输出形状：{exc}",
    },
    "error.bracket_nested": {
        "en": "{where}: nested '[' found, {text!r}",
        "zh": "{where}: 出现嵌套的 '[', {text!r}",
    },
    "error.bracket_unmatched": {
        "en": "{where}: unmatched ']', {text!r}",
        "zh": "{where}: 未配对的 ']', {text!r}",
    },
    "error.bracket_unclosed": {
        "en": "{where}: unclosed '[', {text!r}",
        "zh": "{where}: 未闭合的 '[', {text!r}",
    },
    "error.bracket_non_integer": {
        "en": "{where}: non-integer axis length {body!r}",
        "zh": "{where}: 非整数的轴长 {body!r}",
    },
    "error.bracket_stray_char": {
        "en": "{where}: stray character outside the brackets {char!r}",
        "zh": "{where}: 方括号外的多余字符 {char!r}",
    },
    "error.bracket_none_parsed": {
        "en": "{where}: no shape parsed, {text!r}",
        "zh": "{where}: 没有解析到任何形状, {text!r}",
    },
    "error.shape_inputs_label": {
        "en": "the input part of --shape",
        "zh": "--shape 的输入部分",
    },
    "error.shape_outputs_label": {
        "en": "the output part of --shape",
        "zh": "--shape 的输出部分",
    },
    "error.shape_output_count": {
        "en": "--shape output must be exactly one bracketed list, got {n}",
        "zh": "--shape 输出必须恰好一个方括号，得到 {n}",
    },
    "error.equation_needs_arrow": {
        "en": ("--equation must contain an explicit '->' (an implicit output would "
               "clash with the output section of --shape)"),
        "zh": "--equation 必须带显式 '->'（隐式输出会与 --shape 的输出段冲突）",
    },
    "error.operand_count": {
        "en": "--equation has {n_terms} operand(s) but --shape gave {n_shapes}",
        "zh": "--equation 有 {n_terms} 个操作数，--shape 给了 {n_shapes} 个",
    },
    "error.operand_rank": {
        "en": "operand {term!r} has rank {term_rank} but shape {shape} has rank "
              "{shape_rank}",
        "zh": "操作数 {term!r} 的秩是 {term_rank}，形状 {shape} 的秩是 {shape_rank}",
    },
    "error.extent_positive": {
        "en": "operand {term!r} axis lengths must be positive, got {extent}",
        "zh": "操作数 {term!r} 的轴长必须为正，得到 {extent}",
    },
    "error.label_extent_clash": {
        "en": ("label {label!r} has inconsistent lengths across operands "
               "({known} vs {extent}) — this tool never guesses broadcasting"),
        "zh": ("标签 {label!r} 在不同操作数里长度不一致"
               "（{known} vs {extent}）——本工具不做广播猜测"),
    },
    "error.output_label_missing": {
        "en": "output label {label!r} does not appear in any input",
        "zh": "输出标签 {label!r} 未在任何输入里出现",
    },
    "error.cupy_import": {
        "en": ("--device {device!r} needs cupy, but the import failed: "
               "{exc_type}: {exc}\n  hint: --device cpu works without a GPU"),
        "zh": (f"--device {{device!r}} 需要 cupy，但导入失败：{{exc_type}}: "
               f"{{exc}}\n  提示: --device cpu 可在无 GPU 时运行"),
    },
    "error.cupy_enumerate": {
        "en": "cupy cannot enumerate CUDA devices: {exc_type}: {exc}",
        "zh": "cupy 无法枚举 CUDA 设备：{exc_type}: {exc}",
    },
    "error.cupy_no_device": {
        "en": ("--device gpu but this process sees no CUDA device "
               "(check CUDA_VISIBLE_DEVICES / the driver)"),
        "zh": ("--device gpu 但本进程看不到任何 CUDA 设备"
               "（检查 CUDA_VISIBLE_DEVICES / 驱动）"),
    },
    "error.dtype": {
        "en": "--dtype must be one of {values}, got {value!r}",
        "zh": "--dtype 必须是 {values} 之一，得到 {value!r}",
    },
    "error.compare": {
        "en": "--compare must be one of {values}, got {value!r}",
        "zh": "--compare 必须是 {values} 之一，得到 {value!r}",
    },
    "error.warmup_zero": {
        "en": ("--warmup must not be 0: the first call carries plan caching / native "
               "library binding / JIT, so not discarding it makes the timing "
               "meaningless"),
        "zh": ("--warmup 不能为 0：首调包含 plan 缓存/原生库绑定/JIT，"
               "不丢弃会让耗时完全失真"),
    },
    "error.rounds": {
        "en": "--rounds must be >= 1",
        "zh": "--rounds 必须 ≥ 1",
    },
    "error.output_shape_mismatch": {
        "en": ("the equation output disagrees with the output section of --shape: "
               "{out_shape} vs {expected_shape}"),
        "zh": "方程输出与 --shape 的输出段不一致：{out_shape} vs {expected_shape}",
    },
    "error.specialize": {
        "en": "--specialize must be one of {values}, got {value!r}",
        "zh": "--specialize 必须是 {values} 之一，得到 {value!r}",
    },
    "error.kernel": {
        "en": ("--kernel-force must be one of {labels} (comma separated for several; "
               "backend cannot be combined with other labels), got {value!r}. "
               "kernel_force only accepts forced values: automatic dispatch has been "
               "replaced by tune (run --tune and pin the winner)"),
        "zh": ("--kernel-force 必须是 {labels} 之一"
               "（多个用逗号分隔；backend 不可与其它标签混用），得到 {value!r}。"
               "kernel_force 只接受强制值，自动分派已由 tune 取代"
               "（用 --tune 实测出最快路线后再写死）"),
    },
    "error.as_equivalence": {
        "en": "--as-equivalence must be one of {values}, got {value!r}",
        "zh": "--as-equivalence 必须是 {values} 之一，得到 {value!r}",
    },
    "error.equivalence": {
        "en": "--equivalence must be one of {values}, got {value!r}",
        "zh": "--equivalence 必须是 {values} 之一，得到 {value!r}",
    },
    "error.cache": {
        "en": "--cache must be one of {values}, got {value!r}",
        "zh": "--cache 必须是 {values} 之一，得到 {value!r}",
    },
    "error.explain": {
        "en": "--explain must be one of {values}, got {value!r}",
        "zh": "--explain 必须是 {values} 之一，得到 {value!r}",
    },
    "error.profile": {
        "en": "--profile must be one of {values}, got {value!r}",
        "zh": "--profile 必须是 {values} 之一，得到 {value!r}",
    },
    "error.tune_mode": {
        "en": ("--tune must be one of {values} (or empty / omitted = no sweep), got "
               "{value!r}; 'pass' is the fast pass mode, everything else that is truthy "
               "means the full mode"),
        "zh": ("--tune 必须是 {values} 之一（留空或不给 = 不跑遍历），得到 {value!r}；"
               "pass 是快速通过模式，其余真值一律是完整模式"),
    },
    "error.tune_timeout": {
        "en": "--tune-timeout must be >= 0 seconds (0 disables the bound), got {value!r}",
        "zh": "--tune-timeout 必须 ≥ 0 秒（0 = 不设上限），得到 {value!r}",
    },
    "error.tune_budget": {
        "en": "--tune-budget must be >= 0 seconds (0 disables the bound), got {value!r}",
        "zh": "--tune-budget 必须 ≥ 0 秒（0 = 不限），得到 {value!r}",
    },
    "error.tune_grid": {
        "en": ("--tune would run {count} combinations, above the hard cap of {limit}; "
               "refusing to truncate silently (a truncated grid would make 'best' "
               "meaningless)"),
        "zh": ("--tune 需要跑 {count} 组，超过硬上限 {limit}；拒绝静默截断"
               "（截断后的网格会让「最优」失去意义）"),
    },
    "error.compare_optimize": {
        "en": "--compare-optimize must be one of {values}, got {value!r}",
        "zh": "--compare-optimize 必须是 {values} 之一，得到 {value!r}",
    },
    "error.language": {
        "en": "--language must be one of {values}, got {value!r}",
        "zh": "--language 必须是 {values} 之一，得到 {value!r}",
    },
    "error.numpy_compare_optimize": {
        "en": ("--compare numpy does not accept --compare-optimize {requested!r}: "
               "np.einsum(optimize=...) accepts only {values} (measured), anything "
               "else raises KeyError('Path name %s not found').\n"
               "  hint: for {requested!r} use --compare opt_einsum (that name exists "
               "in opt_einsum's path registry); do not hard-map it onto a "
               "semantically different numpy name"),
        "zh": ("--compare numpy 不接受 --compare-optimize {requested!r}："
               "np.einsum(optimize=...) 实测只认 "
               "{values} 两个名字，其余抛 "
               "KeyError('Path name %s not found')。"
               "\n  提示: 想要 {requested!r} 请用 --compare opt_einsum"
               "（该名字在 opt_einsum 的路径注册表里），"
               "不要把它硬映射成一个语义不同的 numpy 名字"),
    },
    "error.opt_einsum_compare_optimize": {
        "en": ("--compare opt_einsum does not recognise --compare-optimize "
               "{requested!r}: {exc_type}: {exc}"),
        "zh": ("--compare opt_einsum 不认识 --compare-optimize {requested!r}："
               "{exc_type}: {exc}"),
    },
    "error.corpus_miss": {
        "en": ("without --shape the equation must exist in the corpus (sizes are "
               "inferred from roles), but {equation!r} is not in the {scene} corpus. "
               "Please pass --shape explicitly."),
        "zh": ("省略 --shape 时方程必须在语料里（用于按 roles 反推生产尺寸），"
               "但 {equation!r} 不在 {scene} 语料中。请显式给 --shape。"),
    },
    # 参照实现
    "reference.opt_einsum_import_failed": {
        "en": "opt_einsum import failed: {exc_type}",
        "zh": "opt_einsum 导入失败：{exc_type}",
    },
    "reference.opt_einsum_failed": {
        "en": "opt_einsum execution failed: {exc_type}: {exc}",
        "zh": "opt_einsum 执行失败：{exc_type}: {exc}",
    },
    "reference.other_backend_unavailable": {
        "en": "the other backend is unavailable: {exc}",
        "zh": "对方后端不可用：{exc}",
    },
    "reference.other_backend_failed": {
        "en": "the other backend failed: {exc_type}: {exc}",
        "zh": "对方后端执行失败：{exc_type}: {exc}",
    },
    "reference.unknown_compare": {
        "en": "unknown compare mode {compare!r}",
        "zh": "未知 compare 模式 {compare!r}",
    },
    "reference.unavailable": {
        "en": "reference implementation {compare} is unavailable: {note}",
        "zh": "参照实现 {compare} 不可用：{note}",
    },
    # 诊断
    "diagnostic.specialize_not_forwarded": {
        "en": ("specialize=false should force the generic path, yet native hits were "
               "still observed for {hits} — --specialize did not reach the underlying "
               "contract()"),
        "zh": ("specialize=false 本应强制通用路径，但仍观察到原生命中 "
               "{hits}——--specialize 没有传到底层 contract()"),
    },
    # 形状来源（写进 JSON 报告）
    "note.shape_from_corpus": {
        "en": "shapes inferred from corpus roles (scene={scene})",
        "zh": "形状由语料 roles 反推（scene={scene}）",
    },
    "note.shape_from_flag": {
        "en": "shapes come from --shape",
        "zh": "形状来自 --shape",
    },
}


def resolve_language(requested: str | None = None) -> str:
    """把 ``--language`` 解析成实际输出语言（``"en"`` / ``"zh"``）。

    与 ``_specialize.notice_language()`` 是**同一套语言开关**，只是 ``auto`` 的
    缺省值不同——这里刻意不直接等于 ``notice_language()``：

    * ``en`` / ``zh-cn``（也接受 ``zh``）显式指定，一直用它；
    * ``auto``（缺省）：只有 ``PYQCD_EINSUM_LANG`` **显式设了**才跟随它；未设时
      保持本入口的历史行为（中文）。直接跟随 locale 会让 ``LANG=C.UTF-8`` 的机器
      上不传选项的输出从中文变英文，那是无声的破坏性变更。
    """
    text = DEFAULT_LANGUAGE if requested is None else str(requested).strip().lower()
    if text in ("en", "english"):
        return "en"
    if text in ("zh-cn", "zh", "cn", "chinese"):
        return "zh"
    if text != "auto":
        # 语言无法解析时按 env 的 auto 判定报错，别让英文用户收到中文错误
        raise UsageError(translate("error.language", _env_language(),
                                   values=LANGUAGE_MODES, value=requested))
    return _env_language()


def _env_language() -> str:
    """``auto`` 的判定：只在 ``PYQCD_EINSUM_LANG`` 显式设了时跟随，否则中文。"""
    explicit = os.environ.get(LANGUAGE_ENV, "").strip().lower()
    if explicit.startswith("en"):
        return "en"
    if explicit.startswith("zh"):
        return "zh"
    return "zh"


def translate(key: str, language: str, **kwargs: Any) -> str:
    """查文案表并格式化。缺 key / 缺语言 / 缺占位符**直接抛 KeyError**。

    静默返回 key 本身会让漏翻的字符串以"看似正常的英文"混进输出里，比直接崩更难
    发现；本入口宁可 fail loudly。``zh-cn``/``zh`` 都是 ``"zh"`` 的别名，与
    :data:`LANGUAGE_MODES` 一致。
    """
    lang = str(language).strip().lower()
    if lang.startswith("zh"):
        lang = "zh"
    elif lang.startswith("en"):
        lang = "en"
    try:
        variants = MESSAGES[key]
    except KeyError:
        raise KeyError(f"missing i18n key {key!r}") from None
    try:
        template = variants[lang]
    except KeyError:
        raise KeyError(
            f"i18n key {key!r} has no {lang!r} variant (have "
            f"{sorted(variants)})") from None
    return template.format(**kwargs) if kwargs else template


# ── 形状解析 ────────────────────────────────────────────────────────────────
def parse_shape_spec(spec: str, *, language: str = "zh",
                     equation: str | None = None
                     ) -> tuple[list[tuple[int, ...]], tuple[int, ...] | None]:
    """把 ``'[..],[..]->[..]'`` 解析成 (输入形状列表, 输出形状)。

    只接受你给的这种纯数字形态；任何解析失败都**直接报错**，不猜。

    **省略 ``->`` 时不再直接当成整式归约**（返回 ``None``，由调用方按方程输出段
    推导）。这是 2026-11-15 用户实测暴露的缺陷：他用 ``--shape "[4,4,1,1],..."`
    （16 个输入形状、没写输出段）跑 ``...->es``，方程明明指定了输出 ``e`` 与 ``s``，
    却被判成标量而报 ``actual shape=[100, 100] expected=[]``。给了 ``equation``
    时按它的输出段从输入形状里查长度；查不到就报错，不猜。

    ``language`` 只影响错误文案（``"en"``/``"zh"``），缺省 ``"zh"`` 保持旧行为。
    """
    text = spec.strip().replace(" ", "")
    if not text:
        raise ValueError(translate("error.shape_empty", language))
    if "->" in text:
        left, right = text.split("->", 1)
    else:
        left, right = text, ""
    inputs = _parse_bracket_list(
        left, translate("error.shape_inputs_label", language), language)
    if right:
        outputs = _parse_bracket_list(
            right, translate("error.shape_outputs_label", language), language)
        if len(outputs) != 1:
            raise ValueError(translate("error.shape_output_count", language,
                                       n=len(outputs)))
        out_shape: tuple[int, ...] | None = outputs[0]
    elif equation is not None:
        out_shape = _out_shape_from_equation(equation, inputs, language)
    else:
        out_shape = ()
    return inputs, out_shape


def _out_shape_from_equation(equation: str, inputs: Sequence[tuple[int, ...]],
                             language: str) -> tuple[int, ...]:
    """按方程输出段，从各输入形状里查出每个输出标签的长度。

    **用库自己的子脚本解析器，不要用更严的 ``parse_equation``。**
    ``_dedicated_kernels.parse_equation`` 是给原生下发的保守校验：它拒绝
    "同一操作数内重复标签"（对角 ``ii->i``）、拒绝省略号、且不做去空格。
    而 ``contract()`` 本身支持这三种写法（``parser.py`` 先 ``replace(" ","")``，
    再按需要展开 ellipsis）。原先这里用它，导致**同一个方程**省略输出段时 rc=2、
    补上 ``->[...]`` 却 PASS——测试入口比被测库更严，属功能缺口
    （2026-11-15 第三轮巡查实证：``ii->i`` 配 ``[4,4]`` rc=2，配 ``[4,4]->[4]`` PASS）。

    这里走 ``parser.parse_einsum_input``：它做去空格与 ellipsis 展开，返回
    ``(输入子脚本串, 输出子脚本串, 形状)``，与 ``contract`` 同一口径。对角标签在
    一个操作数内重复出现时，按**该操作数自己的轴长**校验一致性（``[4,5]`` 配
    ``ii`` 必须报错，因为它要求 4==5）。
    """
    from pyqcd.einsum import parser as _parser

    try:
        lhs, rhs, parsed_shapes = _parser.parse_einsum_input(
            [equation] + [tuple(int(d) for d in shape) for shape in inputs],
            shapes=True)
    except (ValueError, TypeError) as exc:
        raise ValueError(translate("error.shape_equation_unparsable", language,
                                   equation=equation, exc=exc)) from exc
    terms = lhs.split(",") if lhs else []
    # ``parser.parse_einsum_input`` 只对**整串**去空格，逗号两侧的空格会留在每个
    # 子脚本首尾（``"ij, jk -> ik"`` 的第二个是 ``" jk "``）。原先直接 ``split(",")``
    # 就把空格算进秩里，报"秩是 4"（2026-11-15 第三轮巡查实证 A-2）。
    terms = [term.strip() for term in terms]
    if len(terms) != len(inputs):
        raise ValueError(translate("error.shape_count", language,
                                   given=len(inputs), expected=len(terms)))
    label_sizes: dict[str, int] = {}
    for term, shape in zip(terms, inputs):
        if len(term) != len(shape):
            raise ValueError(translate("error.shape_rank", language,
                                       term=term, rank=len(term),
                                       shape=list(shape), given=len(shape)))
        for ch, extent in zip(term, shape):
            previous = label_sizes.get(ch)
            if previous is not None and previous != extent:
                # 同一标签在不同位置长度必须一致（含 ``ii`` 这种操作数内重复）。
                raise ValueError(translate("error.shape_conflict", language,
                                           label=ch, first=previous,
                                           second=extent))
            label_sizes[ch] = int(extent)
    try:
        return tuple(label_sizes[ch] for ch in rhs)
    except KeyError as missing:
        raise ValueError(translate("error.shape_equation_unparsable", language,
                                   equation=equation,
                                   exc=f"output label {missing.args[0]!r} "
                                       f"has no length")) from None


def _parse_bracket_list(text: str, where: str, language: str = "zh"
                        ) -> list[tuple[int, ...]]:
    """解析 ``[1,2],[3,4]`` 形式的方括号列表。"""
    shapes: list[tuple[int, ...]] = []
    depth = 0
    current: list[str] = []
    for char in text:
        if char == "[":
            if depth != 0:
                raise ValueError(translate("error.bracket_nested", language,
                                           where=where, text=text))
            depth = 1
            current = []
        elif char == "]":
            if depth != 1:
                raise ValueError(translate("error.bracket_unmatched", language,
                                           where=where, text=text))
            depth = 0
            body = "".join(current).strip()
            if body == "":
                shapes.append(())
            else:
                try:
                    shapes.append(tuple(int(item) for item in body.split(",")))
                except ValueError as exc:
                    raise ValueError(translate("error.bracket_non_integer", language,
                                               where=where, body=body)) from exc
        elif char == "," and depth == 0:
            continue
        else:
            if depth != 1:
                raise ValueError(translate("error.bracket_stray_char", language,
                                           where=where, char=char))
            current.append(char)
    if depth != 0:
        raise ValueError(translate("error.bracket_unclosed", language,
                                   where=where, text=text))
    if not shapes:
        raise ValueError(translate("error.bracket_none_parsed", language,
                                   where=where, text=text))
    return shapes


def _validate_against_equation(equation: str, shapes: Sequence[tuple[int, ...]],
                               language: str = "zh") -> None:
    """操作数个数、秩、同标签等长——任一不符直接报错。

    **方程侧要先按库的口径归一**（去空格 + 展开 ellipsis），否则同一个方程会因为
    多写了个空格就被判"秩是 4"（``"ij, jk -> ik"`` 的第二个子脚本是 ``" jk "``，
    长度 4 而形状是 ``(4,4)``），而它在 ``contract()`` 里完全合法。
    2026-11-15 第三轮巡查实证 A-2 / C-1。

    归一用 ``parser.parse_einsum_input``——与 ``contract`` 同一实现，因此对角
    （``ii->i``）、带空格、ellipsis 三种写法都与被测库一致。归一失败时按原样
    校验，让更具体的错误（缺 ``->``、个数不符）先报出来。
    """
    if "->" not in equation:
        raise UsageError(translate("error.equation_needs_arrow", language))
    lhs, rhs = equation.split("->", 1)
    try:
        from pyqcd.einsum import parser as _parser
        normalized_lhs, normalized_rhs, _ = _parser.parse_einsum_input(
            [equation] + [tuple(int(d) for d in shape) for shape in shapes],
            shapes=True)
        lhs, rhs = normalized_lhs, normalized_rhs
    except (ValueError, TypeError):
        pass                        # 保持原样，交给下面的逐项校验报具体原因
    terms = [term.strip() for term in lhs.split(",")]
    if len(terms) != len(shapes):
        raise UsageError(translate("error.operand_count", language,
                                   n_terms=len(terms), n_shapes=len(shapes)))
    sizes: dict[str, int] = {}
    for term, shape in zip(terms, shapes):
        if len(term) != len(shape):
            raise UsageError(translate("error.operand_rank", language,
                                       term=term, term_rank=len(term),
                                       shape=shape, shape_rank=len(shape)))
        for label, extent in zip(term, shape):
            if extent <= 0:
                raise UsageError(translate("error.extent_positive", language,
                                           term=term, extent=extent))
            known = sizes.setdefault(label, extent)
            if known != extent:
                raise UsageError(translate("error.label_extent_clash", language,
                                           label=label, known=known, extent=extent))
    for label in rhs:
        if label not in sizes:
            raise UsageError(translate("error.output_label_missing", language,
                                       label=label))


# ── 设备与张量 ──────────────────────────────────────────────────────────────
def _xp_for_device(device: str, language: str = "zh"):
    """返回该设备对应的数组模块（numpy 或 cupy）。"""
    if device == "cpu":
        return np
    try:
        import cupy
    except Exception as exc:                    # noqa: BLE001
        raise UsageError(translate(
            "error.cupy_import", language, device=device,
            exc_type=type(exc).__name__, exc=exc)) from exc
    try:
        count = int(cupy.cuda.runtime.getDeviceCount())
    except Exception as exc:                    # noqa: BLE001
        raise UsageError(translate("error.cupy_enumerate", language,
                                   exc_type=type(exc).__name__, exc=exc)) from exc
    if count == 0:
        raise UsageError(translate("error.cupy_no_device", language))
    return cupy


def _make_operands(xp, shapes, dtype: np.dtype, seed: int):
    """按固定 seed 生成操作数（同一 seed 下各路径看到同一批数据）。"""
    rng = np.random.default_rng(seed)
    host = []
    for shape in shapes:
        if dtype.kind == "c":
            real = rng.standard_normal(shape)
            imag = rng.standard_normal(shape)
            host.append((real + 1j * imag).astype(dtype))
        else:
            host.append(rng.standard_normal(shape).astype(dtype))
    if xp is np:
        return host, host
    return [xp.asarray(item) for item in host], host


# ── 计时 ────────────────────────────────────────────────────────────────────
def _timed(call, warmup: int, rounds: int) -> float:
    """丢弃 ``warmup`` 次后取 ``rounds`` 次中位（秒）。"""
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(rounds):
        start = time.perf_counter()
        call()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def _relative_error(actual, expected) -> float:
    """按 ``max|expected|`` 归一的相对误差（与仓内其他基准同口径）。"""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        return float("inf")
    scale = float(np.max(np.abs(expected))) if expected.size else 0.0
    if scale == 0.0:
        scale = 1.0
    return float(np.max(np.abs(actual - expected))) / scale


# ── 运行 ────────────────────────────────────────────────────────────────────
def run(*, equation: str, shapes: Sequence[tuple[int, ...]], out_shape: tuple[int, ...],
        device: str = DEFAULT_DEVICE, dtype: str = DEFAULT_DTYPE,
        backend: str = DEFAULT_BACKEND, atol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL, compare: str = DEFAULT_COMPARE,
        optimize: Any = DEFAULT_OPTIMIZE, rounds: int = DEFAULT_ROUNDS,
        warmup: int = DEFAULT_WARMUP, seed: int = 0, verbose: bool = DEFAULT_VERBOSE,
        specialize: bool | None = None,
        compare_optimize: str = DEFAULT_COMPARE_OPTIMIZE,
        equivalence: bool | None = None,
        as_equivalence: bool | None | str = None,
        kernel_force: str | None = None,
        cache: bool | None = None,
        explain: bool | None = None,
        profile: bool | None = None,
        profile_dir: str | None = None,
        expected_shape: tuple[int, ...] | None = None,
        language: str | None = None) -> dict[str, Any]:
    """跑一条方程的完整检查，返回结构化结果（供 CLI 与外部脚本消费）。

    ``specialize`` 三态直传 ``contract(specialize=...)``：``None`` 跟随环境变量
    ``PYQCD_EINSUM_SPECIALIZE``，``False`` 强制通用路径，``True`` 强制特化路径。

    ``compare_optimize`` **只影响参照实现**（``--compare numpy/opt_einsum``），
    主路径的收缩序仍由 ``optimize`` 决定。缺省 ``auto`` = 沿用旧行为。

    ``equivalence`` 三态**只影响报告**：``None`` 跟随 ``PYQCD_EINSUM_EQUIVALENCE``，
    ``True``/``False`` 强制。开启且方程未命中任何特化路径时，报告多一个
    ``equivalence`` 字段（``_equivalence.explain()`` 的返回值），供 verbose 明细与
    ``--json`` 使用。它**不改** ``contract()`` 的入参、耗时与任何数值，且在计时
    结束之后才算，不进入计时区域。

    ``language`` 只影响错误文案与 ``verbose`` 明细（``None`` = 走
    :func:`resolve_language`，即 ``--language auto``），不改任何计算结果。

    ``kernel`` 是 ``--kernel-force`` 的原始取值（``None`` = 用户没指定）。**显式性原样
    传下去**：``None`` 一路传到 ``contract(kernel_force=None)``，由 ``_api`` 自己回落
    （环境变量 > ``generic``）——它用 ``kernel_force is not None`` 判断"用户真的给过"，
    CLI 若在这里填上 ``generic``，每次不写该选项的调用都会被判成显式冲突，配置表里
    ``kernel=backend`` 这类命中行就再也套不上。解析出的名字只用于报告（``requested=``）。
    强制标签而没跑成时，``KernelUnavailableError``
    **不被吞掉**：它带着 requested/attempted/逐标签原因向上抛，由 :func:`main`
    打印正文并非零退出——没有"warning + 回落"这一档。

    ``cache`` 三态直传 ``contract(cache=...)``：``None``（``auto``）跟随
    ``PYQCD_EINSUM_CACHE``，``False`` 让底层缓存**既不读也不写**。与 ``equivalence``
    不同，它**真的改变底层行为**，所以报告里既记生效布尔值 ``"cache"``，也记
    ``_cache.stats()`` 的原始快照 ``"cache_stats"``（取不到就是 ``None``）。

    ``explain`` 三态（``None`` = ``auto``）只决定**输出**：``auto`` 跟随
    ``verbose``，``True`` 在不开 verbose 时也打详解，``False`` 一律不打。详解是
    **追加**在既有明细之后的（既有行一字不动）：本进程第一次且画像未命中时给三段
    ``plan`` / ``kernel`` / ``why``，之后每次只给一行画像状态。

    ``profile`` / ``profile_dir`` 三态直传 ``contract(profile=..., profile_dir=...)``；
    ``False`` 时底层既不读也不写画像 CSV、连目录都不建。画像状态在**主调用之前**
    取快照（``_profile_snapshot``），否则读到的会是本次调用自己刚记进去的"命中"。
    """
    import pyqcd.einsum as pe
    from pyqcd.einsum import _specialize

    lang = resolve_language(language)
    # 把本次的语言**推进库内**：库自己的提示（``_specialize.announce`` /
    # ``announce_generic`` / 画像命中行）与报告正文必须同语言，否则一次运行里会同时
    # 出现英文标签与中文解释（2026-11-17 用户实测）。它只改进程内状态、**不写任何
    # 环境变量**（``--language`` 承诺只影响输出）。
    _specialize.set_language(lang)
    if dtype not in DTYPES:
        raise UsageError(translate("error.dtype", lang, values=DTYPES, value=dtype))
    if compare not in COMPARE_MODES:
        raise UsageError(translate("error.compare", lang, values=COMPARE_MODES,
                                   value=compare))
    specialize = parse_specialize(specialize)
    equivalence = parse_equivalence(equivalence, lang)
    as_equivalence_value = parse_as_equivalence(as_equivalence, lang)
    # ``--cache`` 是唯一改变底层行为的三态：显式 ``true``/``false`` 压过环境变量。
    # ``cache_requested``（bool | None）**原样**交给 ``contract(cache=...)``；
    # ``cache_effective`` 是本次真正生效的布尔值，报告用它而不是 ``stats()['enabled']``
    # ——后者是环境变量口径，``--cache false`` 时它会说 True，那样报告就自相矛盾。
    cache_requested = parse_cache(cache, lang)
    cache_effective = cache_enabled(cache_requested)
    cache_explicit = cache_requested is not None
    # ``--explain`` 只决定输出；``--profile`` 才真的改变底层行为（直传 contract）。
    explain_on = resolve_explain(parse_explain(explain, lang), bool(verbose))
    # 完整详解档：``--explain true`` / ``--verbose true`` 恢复库内的解释性提示
    # （画像命中时那唯一一行之外的部分）；两者都没给时回到环境变量口径，
    # 因此 ``PYQCD_EINSUM_SPECIALIZE_VERBOSE=1`` 仍然有效。
    _specialize.set_detail(True if (verbose or explain_on) else None)
    profile_requested = parse_profile(profile, lang)
    profile_kwargs = _contract_profile_kwargs(profile_requested, profile_dir)
    # ``--kernel-force`` 先解析：非法标签必须在构造任何操作数之前就报成用法错误。
    # 规范化后的字符串只用于报告（``requested=``）；**传给 contract 的是显式性本身**
    # ——见下面的 ``kernel_passthrough``。
    kernel_value = ",".join(kernel_labels(kernel_force, lang))
    # "用户没指定"必须以 ``None`` 抵达 ``_api``：它用 ``kernel_force is not None`` 取
    # ``caller_force_given``，填上 ``generic`` 就等于声称"用户点名了 generic"，配置表
    # 里 ``kernel=backend`` 的命中行会被判成显式冲突（``ProfileConflict``）。回落值
    # （环境变量 > ``generic``）由 ``_api`` 自己算，与 CLI 侧解析同源。
    kernel_passthrough = None if kernel_force is None else (kernel_value or None)
    # ``generic`` / ``backend`` 是**路线**不是内核，原样交给 contract()：前者在库里解析
    # 成空强制集，于是 ``_specialize.match`` 与等价改写整段都不必做——这两件事的唯一
    # 目的都是命中特化形态，而调用方已经明确不要形态；后者更早返回，一次原生入口都不碰。
    # 跳过的是**耗时与提示**（一次语料查询 + 一次改写查询，外加"通用路径/专用路径"
    # 两条自相矛盾的提示），方程、dtype、形状与数值一概不变；``route`` 行仍按
    # ``kernel_force`` 如实报出路线，所以"没查"不等于"少了一条证据"。
    ref_optimize = resolve_compare_optimize(compare, optimize, compare_optimize,
                                            language=lang)
    if warmup < 1:
        raise UsageError(translate("error.warmup_zero", lang))
    if rounds < 1:
        raise UsageError(translate("error.rounds", lang))
    if expected_shape is not None and tuple(expected_shape) != tuple(out_shape):
        raise UsageError(translate("error.output_shape_mismatch", lang,
                                   out_shape=tuple(out_shape),
                                   expected_shape=tuple(expected_shape)))

    _validate_against_equation(equation, shapes, lang)
    xp = _xp_for_device(device, lang)
    np_dtype = np.dtype(dtype)
    # 构造前先拍一次快照，用来算"本次运行新增了多少内存/显存"。
    memory_before = _memory_snapshot(xp)
    operands, host = _make_operands(xp, shapes, np_dtype, seed)
    memory_loaded = _memory_snapshot(xp)

    report: dict[str, Any] = {
        "equation": equation,
        "shapes": [list(item) for item in shapes],
        "out_shape": list(out_shape),
        "device": device,
        "dtype": dtype,
        "backend": backend,
        "compare": compare,
        "atol": atol,
        "rounds": rounds,
        "warmup": warmup,
        "optimize": str(optimize),
        "compare_optimize": str(compare_optimize),
        "compare_optimize_effective": ref_optimize,
        "specialize": _specialize_label(specialize),
        "specialize_effective": _specialize_effective(specialize),
        "kernel_force": kernel_value or DEFAULT_KERNEL,
        "cache": cache_effective,
        "cache_explicit": cache_explicit,
        "n_operands": len(shapes),
        "input_elements": int(sum(int(np.prod(item, dtype=np.int64)) for item in shapes)),
        "output_elements": int(np.prod(out_shape, dtype=np.int64)) if out_shape else 1,
    }

    # 主路径
    def call():
        return pe.contract(equation, *operands, backend=backend, optimize=optimize,
                           specialize=specialize, kernel_force=kernel_passthrough,
                           as_equivalence=as_equivalence_value,
                           cache=cache_requested, **profile_kwargs)

    # 画像状态必须在**第一次 contract() 之前**取：本次调用一旦跑完，它自己刚记下
    # 的观测就在表里，之后再看必然是"命中"——那是自证，不是缓存给的。
    # 只在真的要打详解时取：关闭详解时**一次都不碰**画像层（不 configure、不 load、
    # 不建目录），既有调用路径的副作用因此逐字不变。
    profile_state = (_profile_snapshot(
        report, profile_requested, profile_dir,
        supported="profile" in profile_kwargs) if explain_on else {})

    _clear_kernel_info()
    _reset_last_kernel()
    try:
        got = call()
    except Exception as exc:                    # noqa: BLE001
        # 强制内核没跑成必须**透传**：这是本入口的本职（告知"跑不了"），把它降级
        # 成报告里的 ``ok=False`` 会让 CLI 只打印一行 FAIL，用户看不到 requested /
        # attempted / 逐标签原因。其余异常仍按原样收进报告。
        force = _kernel_force_module()
        unavailable = getattr(force, "KernelUnavailableError", None) if force else None
        if unavailable is not None and isinstance(exc, unavailable):
            raise
        # 配置表的**数据/冲突**错误同样透传：``ProfileError`` 说的是"表里的记录与你的
        # 显式输入冲突"或"同键多行"，不是"数值不对"。折进 ``ok=False`` 会让 CLI 打一行
        # ``FAIL``（退出码 1），把一次**用法/配置**失败说成算法失败；``main`` 据类型
        # 打印正文并以 2 退出（§5 的"报错并退出"）。
        if isinstance(exc, _profile_error()):
            raise
        report["ok"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["cache_stats"] = _cache_stats()
        return report
    seconds = _timed(call, warmup, rounds)
    got_host = got if xp is np else _to_host(got)
    report["seconds"] = seconds
    report["actual_shape"] = list(np.shape(got_host))
    report["kernel_info"] = _kernel_info()
    report["kernel_force"] = kernel_force_fields(
        _last_kernel(), report["kernel_force"], kernel_labels(kernel_force, lang))
    report["install"] = _install_info()
    report["memory"] = {
        "inputs_delta": _memory_delta(memory_before, memory_loaded),
        "peak": _memory_delta(memory_before, _memory_snapshot(xp)),
    }
    _check_specialize_took_effect(report, lang)

    # 参照
    if compare != "none":
        reference, ref_seconds, ref_note = _reference(
            compare, equation, host, operands, out_shape, np_dtype, xp,
            rounds, warmup, device, specialize, ref_optimize, language=lang,
            cache=cache_requested)
        report["reference"] = ref_note
        report["reference_seconds"] = ref_seconds
        if reference is None:
            report["ok"] = False
            report["error"] = translate("reference.unavailable", lang,
                                        compare=compare, note=ref_note)
            report["cache_stats"] = _cache_stats()
            return report
        report["rel"] = _relative_error(got_host, reference)
        report["max_abs_diff"] = float(np.max(
            np.abs(np.asarray(got_host) - np.asarray(reference)))) if np.size(got_host) else 0.0
        # 判定式必须是 ``rel <= atol + rtol``（两者同量纲，都是**相对**量）。
        #
        # 原先是 ``rel <= atol + rtol * rel``，即 ``rel * (1 - rtol) <= atol``：
        # **``rtol >= 1`` 时左边恒 <= 0，判定无条件通过**——一个能造出假 PASS 的
        # 缺陷（2026-11-15 第三轮巡查实证：complex64 语料方程 rel=1.82e-07，
        # ``--rtol 0.999`` FAIL 而 ``--rtol 1.0`` PASS）。而且 rtol 原先不进报告，
        # 同一个 PASS 无法复现。现在 atol/rtol/合成阈值三项都进报告。
        report["atol"] = float(atol)
        report["rtol"] = float(rtol)
        report["tolerance"] = float(atol + rtol)
        report["ok"] = bool(
            report["actual_shape"] == list(out_shape)
            and report["rel"] <= report["tolerance"])
        if report["rel"] != report["rel"]:      # NaN
            report["ok"] = False
    else:
        report["ok"] = bool(report["actual_shape"] == list(out_shape))

    # 等价映射报告：**在全部计时与判定之后**才查，且只写报告字段，
    # 不进 contract() 的入参。关闭（--equivalence false / 环境变量 0）时不写该字段。
    equivalent = _equivalence_of(equation, equivalence, cache=cache_requested)
    if equivalent is not None:
        report["equivalence"] = equivalent

    # 命中统计必须在**全部** contract() 调用之后取快照，否则本次运行刚挣到的命中
    # 不在里面，"缓存到底开没开"就还是看不出。
    report["cache_stats"] = _cache_stats()

    if verbose:
        _print_detail(report, lang)
    # 详解**只追加**在既有明细之后（既有行顺序与内容一字不动，契约逐字符比对）。
    # 首次三段 / 之后一行由 `_take_explain_first()` 决定；画像命中时只出一行。
    if explain_on:
        _print_explain(report, profile_state, lang, first=_take_explain_first())
    return report


def _to_host(value):
    """把 cupy/torch 张量转回 numpy（只用于比对与打印）。"""
    module = type(value).__module__
    if module.startswith("cupy"):
        return value.get()
    if module.startswith("torch"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


# ── 专用路径（--specialize） ────────────────────────────────────────────────
def parse_specialize(value: bool | None | str, language: str = "zh") -> bool | None:
    """把 ``auto|true|false``（或 ``None``/``bool``）规范成 ``bool | None``。

    ``None`` 与 ``auto`` 等价（跟随 ``PYQCD_EINSUM_SPECIALIZE``），**不做**环境变量
    读取——是否真正启用由 ``contract()`` 决定，本入口只负责传递意图。
    """
    if value is None or value is False or value is True:
        return value
    text = str(value).strip().lower()
    if text == "auto":
        return None
    if text in ("true", "1", "yes", "on"):
        return True
    if text in ("false", "0", "no", "off"):
        return False
    raise UsageError(translate("error.specialize", language,
                               values=SPECIALIZE_MODES, value=value))


def _specialize_label(specialize: bool | None) -> str:
    """``bool | None`` → CLI 口径的三态名。"""
    if specialize is None:
        return "auto"
    return "true" if specialize else "false"


# ── 强制内核（--kernel-force） ─────────────────────────────────────────────────────
#: 缺省：强制通用路径。与 ``_kernel_force.DEFAULT_LABEL`` 同值；本模块不 import 它，
#: 免得 ``_kernel_force`` 缺席时整个入口起不来。
DEFAULT_KERNEL = "generic"
#: ``last_kernel()`` 取不到时的占位。**机器可读**，不随 ``--language`` 变——
#: 它说的是"没观测到"，不是一句说明文字。
KERNEL_UNKNOWN = "unknown"
#: 可强制的内核标签（与 ``_kernel_force.KERNEL_LABELS`` 同集合：**逻辑名**，运行时按
#: 操作数后端解析成 ``*_dedicated`` / ``*_topology``）。写字面量是为了该模块缺席时
#: ``--help`` 仍能列出合法集合。
KERNEL_LABELS = frozenset({
    "dedicated",
    "topology",
})
#: 不产生强制、只切换路线的两个取值（与 ``_kernel_force`` 同名）。
GENERIC_LABEL = "generic"
BACKEND_LABEL = "backend"
#: 强制内核的环境变量（与 ``_kernel_force.KERNEL_FORCE_ENV`` 同名）。
KERNEL_FORCE_ENV = "PYQCD_EINSUM_KERNEL_FORCE"


def _kernel_force_module():
    """取 ``_kernel_force`` 模块；取不到返回 ``None``（降级成旧行为，不半强制）。"""
    try:
        from pyqcd.einsum import _kernel_force as force
    except ImportError:                         # pragma: no cover - 旧版库
        return None
    return force


def _kernel_label_names() -> tuple[str, ...]:
    """合法取值全集：优先用 ``_kernel_force`` 的权威定义，取不到用本模块字面量。

    报错文案要列的是**能写什么**：四个强制值（``generic`` / ``dedicated`` /
    ``topology`` / ``backend``），不含已删除的 ``auto``。
    """
    force = _kernel_force_module()
    live = getattr(force, "ALLOWED_LABELS", None) if force is not None else None
    names = set(live or (KERNEL_LABELS | {GENERIC_LABEL, BACKEND_LABEL}))
    return tuple(sorted(names))


class _NeverRaised(Exception):
    """``_kernel_force`` 缺席时占据 ``except`` 子句的占位类型（永不匹配）。"""


def _kernel_unavailable_error() -> type[BaseException]:
    """``except`` 用的类型：``KernelUnavailableError``，取不到就是永不匹配的占位。

    必须在 ``except`` 里用一个**表达式**而不是直接 import：``_kernel_force`` 是
    延迟取用的（与 ``_api`` 分属不同改动批次），它缺席时本入口仍要能跑。
    """
    force = _kernel_force_module()
    error = getattr(force, "KernelUnavailableError", None) if force else None
    return error if isinstance(error, type) and issubclass(
        error, BaseException) else _NeverRaised


def kernel_labels(value: str | None, language: str = "zh") -> tuple[str, ...]:
    """解析生效的 ``--kernel-force``：显式取值优先于 ``PYQCD_EINSUM_KERNEL_FORCE``。

    返回 ``_kernel_force.resolve`` 的原始元组（``("generic",)`` = 只走通用路径，
    其余为要强制的**逻辑名**：``dedicated`` / ``topology`` / ``backend``）。非法标签抛
    ``UsageError``（退出码 2）并列出合法集合——静默忽略拼写错误会让用户以为
    强制生效了。
    """
    force = _kernel_force_module()
    if force is None:
        text = DEFAULT_KERNEL if value is None else str(value).strip().lower()
        return (GENERIC_LABEL,) if text in ("", DEFAULT_KERNEL) else (text,)
    try:
        return tuple(force.resolve(value))
    except ValueError as exc:
        raise UsageError(translate("error.kernel", language, value=value,
                                   labels=", ".join(_kernel_label_names()))) from exc


def kernel_requested(value: str | None, language: str = "zh") -> str:
    """报告里 ``requested=`` 的取值：``generic`` 或 ``label,...``。

    与真正传给 ``contract(kernel_force=...)`` 的口径同源（都从这里解析出来），所以这一格
    回答的是"这次到底要求了什么"，而不是"命令行上写了什么"——没给时它同样反映
    ``PYQCD_EINSUM_KERNEL_FORCE`` 或缺省 ``generic``。
    """
    labels = kernel_labels(value, language)
    return ",".join(labels) if labels else DEFAULT_KERNEL


def _reset_last_kernel() -> None:
    """把 ``_api`` 的"最近一次内核"记录清空，让报告只反映**本次**调用。

    与 :func:`_clear_kernel_info` 同口径、同理由：``contract()`` 只在确实走进专用
    分支时才写记录，``--specialize false`` 这类调用一条都不写，残留的上一次记录
    会被读成"这次跑了它"。只清**存在**的字段，旧版 ``_api`` 没有它就不管。
    """
    try:
        from pyqcd.einsum import _api
    except ImportError:                         # pragma: no cover - 旧版库
        return
    if getattr(_api, "last_kernel", None) is None:
        return
    if hasattr(_api, "_LAST_KERNEL"):
        try:
            _api._LAST_KERNEL = None
        except Exception:                       # noqa: BLE001 - 清不掉就靠下面的校验
            pass


def _last_kernel() -> dict[str, Any] | None:
    """读 ``_api.last_kernel()``；接口缺失或状态不对时返回 ``None``。

    **必须容忍缺失**：本入口与 ``_api`` 分属不同改动批次，取不到就走退化显示
    （``ran=unknown attempted=unknown``）。绝不凭 ``LAST_INFO`` / ``kernel_info``
    反推一个"看起来合理"的 ran——那是编造观测。
    """
    try:
        from pyqcd.einsum import _api
    except ImportError:                         # pragma: no cover - 旧版库
        return None
    probe = getattr(_api, "last_kernel", None)
    if probe is None:
        return None
    try:
        record = probe()
    except Exception:                           # noqa: BLE001 - 报告层不许影响主流程
        return None
    return record if isinstance(record, dict) else None


def _forced_of(labels: tuple[str, ...]) -> tuple[str, ...]:
    """``resolve`` 的元组 → ``last_kernel()['requested']`` 的形状。

    ``generic`` → ``()``（不进专用分派、没有尝试可记）；其余原样。自己算而不调
    ``_kernel_force.forced_labels``，是为了该模块缺席时也能校验记录归属。
    """
    if labels == (GENERIC_LABEL,):
        return ()
    return labels


def kernel_force_fields(record: dict[str, Any] | None, requested: str,
                        expected: tuple[str, ...]) -> dict[str, Any]:
    """把 ``last_kernel()`` 的记录整理成报告要的三段；取不到的一律 ``None``。

    ``expected`` 是本次请求的标签元组，用来挡掉**不属于本次调用**的记录：记录的
    ``requested`` 与它对不上就不认。宁可显示 ``unknown``，也不显示一条没法归因的
    观测。
    """
    fields: dict[str, Any] = {"requested": requested, "ran": None, "attempted": None}
    if record is None:
        return fields
    recorded = record.get("requested")
    recorded = None if recorded is None else tuple(str(item) for item in recorded)
    if recorded != _forced_of(expected):
        return fields
    if "ran" in record:
        ran = record.get("ran")
        fields["ran"] = str(ran) if ran else "none"
    attempted = record.get("attempted")
    if attempted is not None:
        labels = [str(item) for item in attempted if str(item)]
        fields["attempted"] = ",".join(labels) if labels else "none"
    return fields


def kernel_force_line(fields: dict[str, Any]) -> str:
    """拼 ``kernel : requested=... ran=... attempted=...`` 那一行。

    这一行是**机器可读**的（同 ``route :``/``kernel   :``），不进 :data:`MESSAGES`：
    三个字段名与 ``none``/``unknown`` 都是稳定取值，跟着 ``--language`` 翻译会让
    脚本解析不到。三段各自独立退化——``requested`` 总会有值（来自本入口的解析），
    ``ran`` / ``attempted`` 取不到时写 ``unknown``。
    """
    return ("kernel : requested={}  ran={}  attempted={}".format(
        fields.get("requested") or DEFAULT_KERNEL,
        fields.get("ran") or KERNEL_UNKNOWN,
        fields.get("attempted") or KERNEL_UNKNOWN))


# ── 等价映射报告（--equivalence） ───────────────────────────────────────────
def parse_equivalence(value: bool | None | str,
                      language: str = "zh") -> bool | None:
    """把 ``auto|true|false``（或 ``None``/``bool``）规范成 ``bool | None``。

    与 :func:`parse_specialize` 同口径：``None`` 与 ``auto`` 等价（跟随
    ``PYQCD_EINSUM_EQUIVALENCE``），**不做**环境变量读取——是否真正开启由
    :func:`equivalence_enabled` 决定。
    """
    if value is None or value is False or value is True:
        return value
    text = str(value).strip().lower()
    if text == "auto":
        return None
    if text in ("true", "1", "yes", "on"):
        return True
    if text in ("false", "0", "no", "off"):
        return False
    raise UsageError(translate("error.equivalence", language,
                               values=EQUIVALENCE_MODES, value=value))


def parse_as_equivalence(value: bool | None | str,
                         language: str = "zh") -> bool | None | str:
    """把 ``auto|true|false|force``（或 ``None``/``bool``）规范成取值。

    ``None`` 与 ``auto`` 等价（跟随 ``PYQCD_EINSUM_AS_EQUIVALENCE``），本函数**不读**
    环境变量——真正是否施加由 :func:`pyqcd.einsum._equivalence.as_equivalence_enabled`
    在调用时决定。

    ``force`` 是**第四态**，不是布尔值：它被**原样**返回（字符串 ``"force"``），由
    ``contract(as_equivalence="force")`` 解释成"忽略跨进程负标记、真的再查一次并
    刷新标记"。压成 ``True`` 会让它退化成 ``true``（尊重负标记），那正是它要绕过的那
    道门，所以这里绝不归一成布尔。
    """
    if value is None or value is False or value is True:
        return value
    text = str(value).strip().lower()
    if text == "auto":
        return None
    if text in ("true", "1", "yes", "on"):
        return True
    if text in ("false", "0", "no", "off"):
        return False
    if text == "force":
        return "force"
    raise UsageError(translate("error.as_equivalence", language,
                               values=", ".join(AS_EQUIVALENCE_MODES),
                               value=value))


def equivalence_enabled(requested: bool | None = None) -> bool:
    """本次调用是否报告等价写法。``None`` 时跟随 ``PYQCD_EINSUM_EQUIVALENCE``。

    写法照抄 ``_specialize.notice_enabled()``：缺省开启，只有 ``0/false/off/no``
    关闭。这是**纯报告开关**——关闭时只是不查等价写法、不打印那一段，
    ``contract()`` 的行为、耗时与数值一概不变。
    """
    if requested is not None:
        return bool(requested)
    return os.environ.get(EQUIVALENCE_ENV, "1").strip().lower() not in (
        "0", "false", "off", "no")


def _equivalence_of(equation: str, requested: bool | None = None,
                    cache: bool | None = None) -> dict[str, Any] | None:
    """查该方程的等价写法，供报告使用；关闭时返回 ``None``。

    ``requested`` 是 ``--equivalence`` 的三态（``None`` = ``auto``）。返回
    ``_equivalence.explain()`` 的原始字典——这是该模块的稳定公开形态，本函数
    只读，不碰它的内部实现，也**绝不**替换或改写 ``contract()`` 收到的方程。

    **只在未命中任何特化路径时才查**：已命中的方程没有必要看等价写法（那一段
    也不该出现，以免干扰正常输出），且能省掉这次查询。``_specialize.match``
    的判据就是"是否命中特化路径"，与 ``route`` 行同源。

    ``cache`` 与主路径同源并原样转发给 ``match(use=...)``：``--cache false`` 时
    报告层**也不许**读/写缓存，否则统计会被"只是查一下报告"的调用污染。
    """
    if not equivalence_enabled(requested):
        return None
    try:
        from pyqcd.einsum import _specialize

        if _specialize.match(equation, use=cache) is not None:
            # 命中形态就到此为止：改写只为命中形态，已命中再改写只是把同一个答案换个
            # 写法报告出来，还会多出一段 equiv 干扰正常输出（本函数只读，绝不改方程）。
            return None
        from pyqcd.einsum import _equivalence

        return _equivalence.explain(equation)
    except Exception as exc:                    # noqa: BLE001 - 报告层不许影响主流程
        return {"equation": equation, "rewritten": False,
                "unavailable": f"{type(exc).__name__}: {exc}"}


def _format_operand_adjustments(equivalent: dict[str, Any]) -> str:
    """把 ``operand_order`` + ``operand_permutations`` 渲染成一行张量调整说明。

    输出形如 ``transpose(1, 0, 2, 3), -, -, -, -, -``：逐操作数一条，按**改写后**
    方程的操作数次序，不需要动的那一路明确写 ``-``。``operand_order`` 非空时
    前置 ``order=[...]``，说明张量列表本身要怎么重排。
    """
    parts: list[str] = []
    order = equivalent.get("operand_order")
    if order:
        parts.append(f"order={list(order)}")
    permutations = equivalent.get("operand_permutations") or ()
    rendered = [
        "-" if perm is None else "transpose(" + ", ".join(str(i) for i in perm) + ")"
        for perm in permutations
    ]
    if rendered:
        parts.append(", ".join(rendered))
    return "  ".join(parts) if parts else "-"


def _format_output_adjustment(equivalent: dict[str, Any]) -> str:
    """输出侧回填：``transpose(1, 2, 0, 3)``，没有则 ``-``。

    这一侧**不许省略**：输入调整后不改结果轴序，用户会拿到一个形状/顺序不对的
    答案。``output_permutation`` 缺字段（旧版 ``_equivalence``）时同样写 ``-``，
    绝不猜。
    """
    perm = equivalent.get("output_permutation")
    if not perm:
        return "-"
    return "transpose(" + ", ".join(str(i) for i in perm) + ")"


# ── 缓存开关与命中统计（--cache） ───────────────────────────────────────────
def parse_cache(value: bool | None | str, language: str = "zh") -> bool | None:
    """把 ``auto|true|false``（或 ``None``/``bool``）规范成 ``bool | None``。

    与 :func:`parse_specialize` / :func:`parse_equivalence` 同口径：``None`` 与
    ``auto`` 等价（跟随 ``PYQCD_EINSUM_CACHE``），**不做**环境变量读取——真正的
    生效值由 :func:`cache_enabled` 决定。非法取值抛 ``UsageError``（退出码 2）。
    """
    if value is None or value is False or value is True:
        return value
    text = str(value).strip().lower()
    if text == "auto":
        return None
    if text in ("true", "1", "yes", "on"):
        return True
    if text in ("false", "0", "no", "off"):
        return False
    raise UsageError(translate("error.cache", language,
                               values=CACHE_MODES, value=value))


def cache_enabled(requested: bool | None = None) -> bool:
    """本次调用实际是否启用缓存。``None`` 时跟随 ``PYQCD_EINSUM_CACHE``。

    判定**委托给库**（``_cache.enabled``）：报告里的 ``enabled=`` 必须与底层真正用的
    那一套口径一致，本模块不另立一份环境变量解释。``_cache`` 缺席（旧版库）时才退回
    同一规则的本地实现（缺省开启，只有 ``0/false/off/no`` 关闭）。
    """
    if requested is not None:
        return bool(requested)
    try:
        from pyqcd.einsum import _cache
    except ImportError:                         # pragma: no cover - 旧版库
        return os.environ.get(CACHE_ENV, "1").strip().lower() not in (
            "0", "false", "off", "no")
    return bool(_cache.enabled(None))


def _cache_stats() -> dict[str, Any] | None:
    """读 ``pyqcd.einsum._cache.stats()`` 的原始快照；取不到返回 ``None``。

    **绝不伪造**：模块、函数或返回形态任一不符就如实给 ``None``（报告里显示
    ``unknown``），不凭命中提示反推一个"看起来合理"的数字。异常同样吞掉——报告层
    的任何问题都不许影响主流程。
    """
    try:
        from pyqcd.einsum import _cache
    except ImportError:                         # pragma: no cover - 旧版库
        return None
    probe = getattr(_cache, "stats", None)
    if probe is None:
        return None
    try:
        snapshot = probe()
    except Exception:                           # noqa: BLE001 - 报告层不许影响主流程
        return None
    return snapshot if isinstance(snapshot, dict) else None


def _cache_totals(stats: dict[str, Any] | None
                  ) -> tuple[int | None, int | None, float | None]:
    """把 ``stats()['caches']`` 汇总成 ``(hits, misses, hit_rate)``。

    统计按**全部已注册缓存**求和：用户问的是"这个入口到底命中了没有"，不是某一个
    内部缓存的名字。没有任何查询时 ``hit_rate`` 为 ``None``（不写 ``0.00``——那会把
    "没问过"说成"问过但一次没中"）。``stats`` 缺失或不完整时三项都是 ``None``。
    """
    if not isinstance(stats, dict):
        return None, None, None
    per_cache = stats.get("caches")
    if not isinstance(per_cache, dict):
        return None, None, None
    hits = misses = 0
    for info in per_cache.values():
        if not isinstance(info, dict):
            continue
        hits += int(info.get("hits") or 0)
        misses += int(info.get("misses") or 0)
    queries = hits + misses
    return hits, misses, (hits / queries if queries else None)


def cache_line(report: dict[str, Any]) -> str:
    """拼 ``cache    : enabled=... hits=... misses=... hit_rate=...`` 那一行。

    这一行是**机器可读**的（同 ``kernel :``），字段名与 ``none``/``unknown`` 都是稳定
    取值，不进 :data:`MESSAGES`——跟着 ``--language`` 翻译会让脚本解析不到。``enabled``
    取 ``report['cache']``（本次**生效**的布尔值），不是 ``stats()['enabled']``：后者
    是环境变量口径，``--cache false`` 时它会如实说 ``true``，用错就是自相矛盾的报告。
    """
    enabled = report.get("cache")
    enabled_text = "unknown" if enabled is None else ("true" if enabled else "false")
    if enabled is False:
        # 本次关闭时**本次调用没有发生任何缓存读写**；``cache_stats()`` 里那些数字
        # 来自报告层自己的只读查询（例如 ``explain()`` 的匹配器调用），把它们印在
        # ``enabled=false`` 后面会让人以为"关了还在命中"，所以这里如实报 0。
        return "cache    : enabled=false  hits=0  misses=0  hit_rate=none"
    hits, misses, rate = _cache_totals(report.get("cache_stats"))
    return ("cache    : enabled={}  hits={}  misses={}  hit_rate={}".format(
        enabled_text,
        "unknown" if hits is None else hits,
        "unknown" if misses is None else misses,
        "none" if rate is None else f"{rate:.2f}"))


# ── 首次详解与画像状态（--explain / --profile） ──────────────────────────────
#: 「首次详解是否已经打过」：进程级。首次 = 本进程第一次进入详解**且**画像未命中。
_EXPLAIN_SHOWN = False


def parse_explain(value: bool | None | str, language: str = "zh") -> bool | None:
    """把 ``auto|true|false``（或 ``None``/``bool``）规范成 ``bool | None``。

    与 :func:`parse_cache` 同口径：``None`` 与 ``auto`` 等价（跟随 ``--verbose``，
    见 :func:`resolve_explain`），**不做**环境变量读取。非法取值抛 ``UsageError``
    （退出码 2）——静默忽略拼写错误会让用户以为详解生效了。
    """
    if value is None or value is False or value is True:
        return value
    text = str(value).strip().lower()
    if text == "auto":
        return None
    if text in ("true", "1", "yes", "on"):
        return True
    if text in ("false", "0", "no", "off"):
        return False
    raise UsageError(translate("error.explain", language,
                               values=EXPLAIN_MODES, value=value))


def resolve_explain(requested: bool | None = None,
                    verbose: bool = False) -> bool:
    """本次是否打印详解：``None``（``auto``）跟随 ``--verbose``，显式值压过它。"""
    return bool(verbose) if requested is None else bool(requested)


def parse_profile(value: bool | None | str, language: str = "zh") -> bool | None:
    """把 ``auto|true|false``（或 ``None``/``bool``）规范成 ``bool | None``。

    ``None`` 表示"用户没给"，交给 :mod:`pyqcd.einsum._profile_cache` 跟随
    ``PYQCD_EINSUM_PROFILE``；``False`` 让底层**既不读也不写**画像 CSV、连目录都
    不建。非法取值抛 ``UsageError``（退出码 2）。
    """
    if value is None or value is False or value is True:
        return value
    text = str(value).strip().lower()
    if text == "auto":
        return None
    if text in ("true", "1", "yes", "on"):
        return True
    if text in ("false", "0", "no", "off"):
        return False
    raise UsageError(translate("error.profile", language,
                               values=PROFILE_MODES, value=value))


def _contract_profile_kwargs(profile: bool | None,
                             profile_dir: str | None) -> dict[str, Any]:
    """只在底层 ``contract()`` 真的接这两个参数时才传（旧版库上退化，不伪造）。

    本入口与 ``_api`` 分属不同改动批次：``profile=``/``profile_dir=`` 是后加的
    关键字参数。硬传会在旧版库上抛 ``TypeError``；这里按实际签名过滤，缺席就当
    "没有画像层"（详解里的画像状态随之降级成 ``unknown``，绝不假装命中/未命中）。
    """
    import inspect

    import pyqcd.einsum as pe

    try:
        names = inspect.signature(pe.contract).parameters
    except (TypeError, ValueError):             # pragma: no cover - 装饰器/扩展
        return {}
    kwargs: dict[str, Any] = {}
    if "profile" in names:
        kwargs["profile"] = profile
    if "profile_dir" in names and profile_dir is not None:
        kwargs["profile_dir"] = profile_dir
    return kwargs


def _profile_module():
    """取 ``_profile_cache`` 模块；取不到返回 ``None``（退回"未知"，绝不编）。"""
    try:
        from pyqcd.einsum import _profile_cache as cache
    except ImportError:                         # pragma: no cover - 旧版库
        return None
    return cache


def _optimize_choices() -> tuple[str, ...] | None:
    """``--optimize`` 的**可写取值**：配置表的 7 个叶子（唯一来源，不抄第二份）。

    ``auto`` **不在此列**：它是 ``contract(optimize=...)`` 的签名缺省值、也是
    ``paths._IMPLICIT_DEFAULT_NAMES`` 里的内部归一结果，``paths.get_path_fn("auto")``
    照旧抛 ``KeyError``。于是 CLI 层不接受用户写 ``auto``，而 API 的缺省行为一个字
    都不用改——``--optimize`` 的缺省值是 ``None``（"未指定"），``main`` 把它还原成
    :data:`DEFAULT_OPTIMIZE` 再交给 ``run()``。

    旧版库没有这个常量时返回 ``None``：argparse 不设 ``choices``（与旧行为一致），
    不做任何本地硬编码的"七叶子"副本。
    """
    cache = _profile_module()
    leaves = getattr(cache, "OPTIMIZE_LEAVES", None) if cache is not None else None
    return tuple(leaves) if leaves else None


def _profile_error() -> type[BaseException]:
    """配置表的**数据/冲突**错误基类（``ProfileError``）；库缺席时给一个永不命中的占位。

    ``run`` 据此把"表里同键多行"与"命中行和显式输入冲突"从"数值不符"里分出来
    向上透传，``main`` 打印正文并以退出码 2 结束（缺表 ``ProfileMissing`` 走的是
    "警告 + 继续"那一路，不会到这里）。返回值**总是**一个类：``except`` 与
    ``isinstance`` 都要它，``None`` 在这两处都不可用。
    """
    cache = _profile_module()
    error = getattr(cache, "ProfileError", None) if cache is not None else None
    return error if isinstance(error, type) else _NoProfileError


class _NoProfileError(Exception):
    """``_profile_cache`` 缺席时的占位异常类（永远不会被抛出）。"""


def _profile_file() -> str | None:
    """配置表的绝对路径：优先 ``profile_path()``，旧版退回 ``summary()['file']``。

    只问库、不自己拼路径：文件名（``pyqcd.einsum.profile.csv``）是 ``_profile_cache``
    的常量，报告层抄一份就等着两边对不上。
    """
    cache = _profile_module()
    if cache is None:
        return None
    probe = getattr(cache, "profile_path", None)
    if callable(probe):
        try:
            path = str(probe() or "")
        except Exception:                       # noqa: BLE001 - 报告层不许影响主流程
            path = ""
        if path:
            return path
    probe = getattr(cache, "summary", None)
    if callable(probe):
        try:
            snapshot = probe()
        except Exception:                       # noqa: BLE001
            return None
        if isinstance(snapshot, dict):
            return str(snapshot.get("file") or "") or None
    return None


def _profile_summary() -> dict[str, Any] | None:
    """配置表当前状态的最小快照；取不到就给 ``None``（报告写 unknown，绝不编）。

    优先 ``summary()``（旧版接口）。A 路按 §7 重写后只剩 ``enabled()`` /
    ``profile_path()`` / ``last_error()``，这时拼一个最小快照并把 ``rows`` / ``keys``
    留成 **缺席**：数文件行数是读侧的口径，报告层不另立一份。
    """
    cache = _profile_module()
    if cache is None:
        return None
    probe = getattr(cache, "summary", None)
    if callable(probe):
        try:
            snapshot = probe()
        except Exception:                       # noqa: BLE001 - 报告层不许影响主流程
            snapshot = None
        if isinstance(snapshot, dict):
            return snapshot
    enabled = getattr(cache, "enabled", None)
    if not callable(enabled):
        return None
    try:
        on = bool(enabled())
    except Exception:                           # noqa: BLE001
        return None
    snapshot = {"enabled": on, "file": _profile_file()}
    last = getattr(cache, "last_error", None)
    if callable(last):
        try:
            snapshot["last_error"] = last()
        except Exception:                       # noqa: BLE001
            pass
    return snapshot


def _profile_key(report: dict[str, Any], cache) -> dict[str, Any] | None:
    """复现 ``_api`` 构造的画像键；复现不了（GPU/原生后端）返回 ``None``。

    ``_api`` 用**操作数对象**的 ``device`` 建键（cupy 只有设备编号、torch 是
    ``cuda:0``），而 CLI 在收缩之后已经拿不到那个对象。所以这里只复现**确定的
    CPU 口径**：``device=cpu`` 且后端是 numpy。其余一律交回 ``unknown``——宁可说
    不知道，也不能凭猜测报"命中/未命中"。
    """
    device = str(report.get("device") or "")
    backend = str(report.get("backend") or DEFAULT_BACKEND).strip().lower()
    if backend in ("", "auto"):
        backend = "numpy" if device == "cpu" else ""
    if device != "cpu" or backend not in ("numpy", "cpu"):
        return None
    formatter = getattr(cache, "format_shape", None)
    shapes = [tuple(int(item) for item in shape)
              for shape in (report.get("shapes") or [])]
    return {
        "user_device": "cpu",
        "backend": backend,
        "dtype": str(report.get("dtype") or ""),
        "equation": "".join(str(report.get("equation") or "").split()),
        "shape": formatter(shapes) if callable(formatter) else "",
        "device": "cpu",
    }


def _profile_snapshot(report: dict[str, Any], requested: bool | None,
                      profile_dir: str | None,
                      supported: bool = True) -> dict[str, Any]:
    """本次调用**开工前**的画像状态：``off`` / ``hit`` / ``miss`` / ``unknown``。

    必须在 ``contract()`` **之前**取。底层在第一次调用里没命中就把观测记进内存，
    之后再查必然是"命中"——那是本次调用自己刚写进去的，不是缓存给的。CLI 的
    "命中/未命中"说的是"开工时这条键在不在表里"，与调用方看到的行为一致。

    ``profile=False``（或环境变量关闭）在第一个判断就返回，**不 configure、不读表、
    不建目录**——与 ``_api._profile_begin`` 的零副作用口径一致。
    ``supported=False``（底层 ``contract()`` 没有 ``profile=``）时同样返回
    ``unknown``：参数都递不进去，就不能声称自己知道画像状态。

    读入口是 §7 的 ``read_winner``（**绝不写**），三种情形分得很清，这里照实回答：
    文件不存在（``ProfileMissing``）= **未命中**（本该先 tune）；同键多行
    （``DuplicateProfileEntry``）= **不知道**（没有"该信哪一条"，真值由 ``_api`` 硬报错）；
    其他读失败 = ``unknown`` + 原因。
    """
    state: dict[str, Any] = {"status": "unknown", "row": None, "rows": None,
                             "last_error": None}
    if not supported:
        return state
    cache = _profile_module()
    if cache is None:
        return state                            # 旧版库：没有画像层，如实说不知道
    try:
        if not cache.enabled(requested):
            state["status"] = "off"
            return state
    except Exception as exc:                    # noqa: BLE001 - 报告层不许影响主流程
        state["last_error"] = f"{type(exc).__name__}: {exc}"
        return state
    key = _profile_key(report, cache)
    if key is None:
        return state                            # GPU/原生后端：键不可复现
    missing = getattr(cache, "ProfileMissing", None)
    duplicate = getattr(cache, "DuplicateProfileEntry", None)
    row: dict[str, Any] | None = None
    try:
        if profile_dir:
            cache.configure(profile_dir)
        row = cache.read_winner(key)
    except Exception as exc:                    # noqa: BLE001
        if isinstance(duplicate, type) and isinstance(exc, duplicate):
            state["last_error"] = str(exc)      # 同键多行：状态就是"不知道"
            return state
        if not (isinstance(missing, type) and isinstance(exc, missing)):
            state["last_error"] = f"{type(exc).__name__}: {exc}"
            return state
        state["status"] = "miss"                # 缺表 = 还没 tune 过
    summary = _profile_summary()
    if isinstance(summary, dict):
        state["rows"] = summary.get("rows")
        state["last_error"] = summary.get("last_error") or state["last_error"]
        if not summary.get("enabled", True):
            state["status"] = "off"
            return state
    if state["status"] == "miss":
        return state
    if row is None:
        state["status"] = "miss"
    else:
        state["status"] = "hit"
        state["row"] = dict(row)
    return state


def _profile_rows() -> list[dict[str, str]]:
    """读配置表的原始行（**只读**，绝不写）。取不到就给空表。

    只用于 ``why`` 的第 ④ 类：把"缓存记录 backend 更快"写成**建议**。字段名来自
    ``_profile_cache.PROFILE_COLUMNS`` 那一套（``kernel``/``wall_ms``/...），
    这里按列名取值，不依赖列序；路径问 ``profile_path()``，文件名不在这里抄一份。
    """
    import csv

    cache = _profile_module()
    enabled = getattr(cache, "enabled", None) if cache is not None else None
    if callable(enabled):
        try:
            if not enabled():
                return []
        except Exception:                       # noqa: BLE001 - 报告层不许影响主流程
            return []
    path = _profile_file()
    if not path:
        return []
    try:
        with open(path, encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle) if row]
    except OSError:
        return []


def _ms_of(row: dict[str, Any]) -> float:
    """``wall_ms`` 列的数值（缺失/坏值按"无穷慢"处理，选优时永远排最后）。

    只认 ``wall_ms``：列名由 §8 冻结（覆盖 §4 的旧名），表文件名也一起换过，旧格式的
    表在新读侧根本走不到，多留一个旧名字只是让人以为还存在两套表。
    """
    try:
        value = float(row.get("wall_ms") or "")
    except (TypeError, ValueError):
        return float("inf")
    return value if value == value else float("inf")


def _ms_text(row: dict[str, Any]) -> str:
    """行里的耗时原文（``wall_ms``）；没有就空串。"""
    return str(row.get("wall_ms") or "")


def _profile_backend_advice(report: dict[str, Any]) -> tuple[str, str | None] | None:
    """画像里**同方程**的历史记录：backend 是否比专用内核更快？是就给两个记录值。

    只读 CSV，不碰执行路径。数字是**记录值**，不是本次实测——报告里必须这么写
    （``profile 建议（非本次实测）``）。没有 backend 记录、或 backend 并不更快时返回
    ``None``：不为了"有话说"编一条结论。
    """
    equation = "".join(str(report.get("equation") or "").split())
    rows = [row for row in _profile_rows()
            if "".join(str(row.get("equation") or "").split()) == equation]
    backend_rows = [row for row in rows
                    if str(row.get("kernel") or "").strip() == BACKEND_LABEL]
    if not backend_rows:
        return None
    kernel_rows = [row for row in rows
                   if str(row.get("kernel") or "").strip()
                   not in ("", BACKEND_LABEL, GENERIC_LABEL)]
    best_backend = min(backend_rows, key=_ms_of)
    if not kernel_rows:
        return _ms_text(best_backend), None
    best_kernel = min(kernel_rows, key=_ms_of)
    if _ms_of(best_backend) >= _ms_of(best_kernel):
        return None
    return _ms_text(best_backend), _ms_text(best_kernel)


def _classify_reasons(reasons: dict[str, Any] | None) -> tuple[str, str] | None:
    """把逐标签失败原因归到 ``why`` 的四类之一（互斥，只出命中的那类）。

    ``contiguity`` → ``dtype-policy`` → ``kernel-unavailable``。**兜底类必须最后判**：
    先判它会让前两类细因永远出不来（它们本身就是"内核不可用"的一种），四类也就
    退化成一类。判定用的是内核自己写的失败原文（``LAST_FAILURE['reason']`` /
    异常文本），不做二次解释。
    """
    items = [(str(label), str(text)) for label, text in (reasons or {}).items()
             if str(text)]
    for label, text in items:
        if "contiguous" in text.lower():
            return "contiguity", f"{label}: {text}"
    for label, text in items:
        lowered = text.lower()
        if ("real dtype" in lowered and "declin" in lowered) \
                or "dtype-policy" in lowered:
            return "dtype-policy", f"{label}: {text}"
    if items:
        return "kernel-unavailable", "; ".join(f"{label}: {text}"
                                               for label, text in items)
    return None


def _plan_of(report: dict[str, Any], language: str = "zh") -> str:
    """``plan`` 段：命中了哪个形态（或哪条等价写法），以及它声明了哪些内核。

    ``_shape_paths`` 先查（形状特化比语料更具体），再查语料索引
    （``_specialize.match``，与 ``route`` 行同源）。都未命中时才看等价写法——
    这一段的字段与 ``equiv`` 段同源（``report['equivalence']``），不做二次查询。
    """
    equation = str(report.get("equation") or "")
    entry: dict[str, Any] | None = None
    source = ""
    try:
        from pyqcd.einsum import _shape_paths
        entry = _shape_paths.entry(equation)
        source = "shape"
    except Exception:                           # noqa: BLE001 - 报告层不许影响主流程
        entry, source = None, ""
    if entry is None:
        try:
            from pyqcd.einsum import _specialize
            entry = _specialize.match(equation, use=report.get("cache"))
            source = "corpus"
        except Exception:                       # noqa: BLE001
            entry = None
    if isinstance(entry, dict):
        kernels = ", ".join(str(item) for item in (entry.get("kernels") or ())) or "-"
        return translate("explain.plan_shape" if source == "shape"
                         else "explain.plan_corpus", language,
                         name=entry.get("name", "-"),
                         kind=entry.get("kind", "-"), kernels=kernels)
    line = translate("explain.plan_none", language)
    equivalent = report.get("equivalence")
    if isinstance(equivalent, dict):
        if equivalent.get("unavailable"):
            line += " → " + translate("explain.plan_equiv_unavailable", language,
                                      exc=equivalent["unavailable"])
        elif equivalent.get("rewritten"):
            line += " → " + translate(
                "explain.plan_equiv", language,
                kind=equivalent.get("kind", "-"),
                equation=equivalent.get("equation", "-"),
                adjustments=_format_operand_adjustments(equivalent),
                perm=_format_output_adjustment(equivalent))
    return line


def _explain_kernel_fields(report: dict[str, Any]) -> dict[str, Any]:
    """详解用的三段：优先用 ``run()`` 整理好的 ``report['kernel_force']``。

    早退路径没有那个字段，此时退到 ``last_kernel()``；再取不到就写 ``unknown``——
    绝不从 ``kernel_info`` 反推一个"看起来合理"的 ran。
    """
    fields = report.get("kernel_force")
    if isinstance(fields, dict):
        return fields
    requested = report.get("kernel_force")
    result: dict[str, Any] = {
        "requested": requested if isinstance(requested, str) else DEFAULT_KERNEL,
        "ran": None, "attempted": None}
    record = _last_kernel()
    if isinstance(record, dict):
        ran = record.get("ran")
        result["ran"] = str(ran) if ran else "none"
        attempted = [str(item) for item in (record.get("attempted") or ())
                     if str(item)]
        result["attempted"] = ",".join(attempted) or "none"
    return result


def explain_kernel_line(fields: dict[str, Any]) -> str:
    """``kernel : requested=... → attempted=... → ran=...``（选路的因果链）。

    与 :func:`kernel_force_line` 同源（同一份 ``last_kernel()`` 整理结果），但用
    ``→`` 表达"要求了什么 → 真试了哪些 → 最后跑成哪个"。字段名与 ``none``/
    ``unknown`` 是稳定取值，不进 :data:`MESSAGES`——跟着 ``--language`` 翻译会让
    脚本解析不到。
    """
    return ("kernel : requested={}  → attempted={}  → ran={}".format(
        fields.get("requested") or DEFAULT_KERNEL,
        fields.get("attempted") or KERNEL_UNKNOWN,
        fields.get("ran") or KERNEL_UNKNOWN))


def _why_line(report: dict[str, Any], state: dict[str, Any],
              language: str = "zh") -> str:
    """``why`` 段：本次为什么没走（或最终保留了）专用内核。

    优先级：**最终跑成了专用内核**（``kept: <具体标签>``）> 四类放弃原因
    （``contiguity`` / ``dtype-policy`` / ``kernel-unavailable`` / ``profile-backend``）
    > ``none``（本次根本没尝试）。原因只读 ``last_kernel()['reasons']``，
    不重新跑、不重新解释。
    """
    record = _last_kernel() or {}
    ran = record.get("ran")
    if ran:
        return translate("explain.why_kept", language, label=str(ran))
    classified = _classify_reasons(record.get("reasons"))
    if classified is not None:
        kind, detail = classified
        if kind == "contiguity":
            return translate("explain.why_contiguity", language, reason=detail)
        if kind == "dtype-policy":
            return translate("explain.why_dtype", language)
        return translate("explain.why_unavailable", language, reasons=detail)
    advice = (_profile_backend_advice(report)
              if state.get("status") in ("hit", "miss") else None)
    if advice is not None:
        return translate("explain.why_profile", language,
                         backend_ms=advice[0] or KERNEL_UNKNOWN,
                         kernel_ms=advice[1] or KERNEL_UNKNOWN)
    requested = report.get("kernel_force")
    return translate("explain.why_none", language,
                     requested=requested if isinstance(requested, str)
                     else DEFAULT_KERNEL)


def _profile_replayable(kernel: Any) -> bool:
    """画像该行的 ``kernel`` 能否被 ``contract()`` 重放（建议是谁的裁决归谁）。

    判据**不在这里另立一份**：直接问 ``_api._profile_force_label``（它决定命中的
    ``kernel`` 能不能变成 ``kernel_force=``）。取不到该函数时才退回 ``_profile_cache``
    的 "不可重放" 集合口径（空 / ``equivalence``）。

    这里曾经把 ``generic`` 硬编码成"不可重放"，于是画像真的按 ``generic`` 执行了
    （命中行写着"已套用"），下一行却说"命中但未采纳"——两句自相矛盾
    （2026-11-17 用户实测）。
    """
    text = str(kernel or "")
    try:
        from pyqcd.einsum import _api

        resolve = getattr(_api, "_profile_force_label", None)
        if callable(resolve):
            return resolve(text) is not None
    except Exception:                           # noqa: BLE001 - 报告层不许影响主流程
        pass
    return text.strip().lower() not in ("", "equivalence")


def _profile_status_line(state: dict[str, Any], language: str = "zh") -> str:
    """之后每次那**一行**：命中 / 未命中已记录 / 关闭 / 未知。"""
    status = state.get("status")
    if status == "hit":
        row = state.get("row") or {}
        # 建议能不能重放由 ``_api`` 裁决（``generic``/``backend`` 都可重放），
        # 否则"已套用"的命中会被下一行说成"未采纳"。
        if not _profile_replayable(row.get("kernel")):
            return translate("explain.status_hit_inert", language,
                             kernel=row.get("kernel") or KERNEL_UNKNOWN)
        return translate("explain.status_hit", language,
                         backend=row.get("backend") or KERNEL_UNKNOWN,
                         kernel=row.get("kernel") or KERNEL_UNKNOWN,
                         wall_ms=_ms_text(row) or KERNEL_UNKNOWN)
    if status == "miss":
        rows = state.get("rows")
        return translate("explain.status_miss", language,
                         rows=KERNEL_UNKNOWN if rows is None else rows)
    if status == "off":
        return translate("explain.status_off", language)
    return translate("explain.status_unknown", language)


def _take_explain_first() -> bool:
    """本进程第一次详解返回 ``True``（并置位），之后一律 ``False``。"""
    global _EXPLAIN_SHOWN
    if _EXPLAIN_SHOWN:
        return False
    _EXPLAIN_SHOWN = True
    return True


def _print_explain(report: dict[str, Any], state: dict[str, Any],
                   language: str = "zh", first: bool = True) -> None:
    """``--explain`` 的输出：首次三段（``plan`` → ``kernel`` → ``why``），之后一行。

    全部写 stderr（与既有明细同流），**只追加**、不改既有行：调用点在
    ``_print_detail`` 之后。画像命中时**不**打三段——这是"之后只打一行"的口径，
    命中本身已经说明了配置来自缓存。
    """
    out = sys.stderr
    if first and state.get("status") != "hit":
        print("plan   : " + _plan_of(report, language), file=out)
        print(explain_kernel_line(_explain_kernel_fields(report)), file=out)
        print("why    : " + _why_line(report, state, language), file=out)
    # 命中时只留这一行：① 记录定位已经由库提示的 ``<pyqcd.einsum.profile.csv>:<数据行号>``
    # （``state['row']['line']``，1-based）钉住——同一个 ``kernel=`` 会出现在许多行里，
    # 没有行号用户翻不到具体哪一条在生效；② ``route :`` / ``kernel :`` 这类**结果**行
    # 不在此列：它们是本次实测的证据、不是对分派的解释，由 ``_print_detail`` 照旧输出；
    # ③ 详解档（``--explain true`` / ``--verbose true``）恢复的是库内提示，不是给一条
    # 由缓存给定、本次没有规划过的路线补理由。
    print(_profile_status_line(state, language), file=out)


# ── 参照的优化器（--compare-optimize） ─────────────────────────────────────
def resolve_compare_optimize(compare: str, optimize: Any,
                             compare_optimize: str | None,
                             language: str = "zh") -> str | None:
    """把 ``--compare-optimize`` 解析成**参照实际要用的**优化器名。

    返回 ``None`` 表示该模式没有优化器可设（``backend``/``none``）。

    两条参照路径的**名字空间不同**，本函数刻意不做跨库重映射：

    * ``numpy``：``np.einsum`` 源码只认 ``"greedy"`` 与 ``"optimal"``，其余值一律
      ``KeyError("Path name %s not found", ...)``（numpy 2.5.4 实测）。所以 ``dp`` /
      ``branch-*`` / ``auto`` 在这里是**明确的用法错误**，不是"退回 greedy"。
      ``auto`` 的旧行为是 ``greedy``，保持不变。
    * ``opt_einsum``：``auto`` 的旧行为是"``--optimize`` 是 auto/None 就用 greedy，
      否则跟随 ``--optimize``"——本条逐字保留，否则会悄悄改掉别人已有的对照结论。
      显式给了值就用该值，但先按 ``oe.paths`` 注册表校验，不接受未知名字。
    * ``backend`` / ``none``：优化器不参与，非 ``auto`` 值只在 verbose 里提示无效。
    """
    requested = DEFAULT_COMPARE_OPTIMIZE if compare_optimize is None else str(
        compare_optimize).strip().lower()
    if requested not in COMPARE_OPTIMIZERS:
        raise UsageError(translate("error.compare_optimize", language,
                                   values=COMPARE_OPTIMIZERS,
                                   value=compare_optimize))

    if compare in ("backend", "none"):
        return None

    if compare == "numpy":
        if requested == "auto":
            return "greedy"                     # 旧行为，逐字保留
        if requested not in NUMPY_COMPARE_OPTIMIZERS:
            raise UsageError(translate("error.numpy_compare_optimize", language,
                                       requested=requested,
                                       values=NUMPY_COMPARE_OPTIMIZERS))
        return requested

    # compare == "opt_einsum"
    if requested == "auto":
        path = "greedy" if optimize in (None, "auto") else optimize
        return path if isinstance(path, str) else "greedy"
    try:
        from opt_einsum import paths as _oe_paths
        _oe_paths.get_path_fn(requested)
    except Exception as exc:                    # noqa: BLE001
        raise UsageError(translate("error.opt_einsum_compare_optimize", language,
                                   requested=requested,
                                   exc_type=type(exc).__name__, exc=exc)) from exc
    return requested


def _specialize_effective(specialize: bool | None) -> bool:
    """本次调用实际是否启用专用路径（``auto`` 时读库的判定，不重复解释环境变量）。"""
    from pyqcd.einsum import _specialize as specialize_module
    return bool(specialize_module.enabled(specialize))


def _clear_kernel_info() -> None:
    """清空原生入口的 ``LAST_INFO``，让报告只反映**本次**调用。

    原生路径本身在每次下发前会 ``clear()``，但 ``specialize=False`` 根本不会进原生
    路径，残留的上一次记录会被误读成"关掉专用路径后仍然命中"。
    """
    from pyqcd.einsum import _kernel_cuda, _kernel_kupl

    for module in (_kernel_kupl, _kernel_cuda):
        info = getattr(module, "LAST_INFO", None)
        if isinstance(info, dict):
            info.clear()


def _check_specialize_took_effect(report: dict[str, Any], language: str = "zh") -> None:
    """``specialize=false`` 却留下原生命中 = 参数没传下去，必须显式记下。

    这条只在 ``--specialize false`` 时有意义：关闭专用路径的调用不该碰任何原生
    入口，因此 ``kernel`` 字段必须为空。
    """
    if report.get("specialize_effective") or not report.get("kernel_info"):
        return
    report["specialize_warning"] = translate(
        "diagnostic.specialize_not_forwarded", language,
        hits=sorted(report["kernel_info"]))


def _kernel_info() -> dict[str, Any]:
    """读 KUPL/CUDA 入口最近一次调用的路线信息（不触发加载）。"""
    info: dict[str, Any] = {}
    from pyqcd.einsum import _kernel_cuda, _kernel_kupl

    for name, module in (("kupl", _kernel_kupl), ("cuda", _kernel_cuda)):
        last = getattr(module, "LAST_INFO", None)
        if last:
            info[name] = {key: last[key] for key in
                          ("path", "specialized", "steps", "dtype", "copied")
                          if key in last}
    return info


def _reference(compare, equation, host, operands, out_shape, dtype, xp,
               rounds, warmup, device, specialize=None,
               ref_optimize=None, language: str = "zh", cache=None):
    """按 ``--compare`` 取参照结果，返回 ``(数组或 None, 秒数或 None, 说明)``。

    ``ref_optimize`` 是 ``resolve_compare_optimize`` 解析出的**参照实际用的**优化器
    （``None`` = 该模式不设优化器）。本函数刻意**不读主路径的 ``optimize``**——参照
    借用主路径的选择会让对照实验失去独立性。

    ``cache`` 与主路径同源（``--compare backend`` 也会调用 ``contract()``）：
    ``--cache false`` 不该被参照这一路偷偷预热或污染统计。

    ``language`` 只影响失败说明的文案。
    """
    if compare == "numpy":
        path = "greedy" if ref_optimize is None else ref_optimize

        def call():
            return np.einsum(equation, *host, optimize=path)
        return call(), _timed(call, warmup, rounds), f"np.einsum(optimize={path!r})"
    if compare == "opt_einsum":
        try:
            import opt_einsum as oe
        except Exception as exc:                # noqa: BLE001
            return None, None, translate("reference.opt_einsum_import_failed",
                                         language, exc_type=type(exc).__name__)
        path = "greedy" if ref_optimize is None else ref_optimize

        def call():
            return oe.contract(equation, *host, optimize=path)
        try:
            seconds = _timed(call, warmup, rounds)
            return call(), seconds, f"opt_einsum.contract(optimize={path!r})"
        except Exception as exc:                # noqa: BLE001
            return None, None, translate("reference.opt_einsum_failed", language,
                                         exc_type=type(exc).__name__, exc=exc)
    if compare == "backend":
        other = "numpy" if xp is not np else ("cupy" if device == "cpu" else "numpy")
        try:
            other_xp = np if other == "numpy" else _xp_for_device("gpu", language)
        except SystemExit as exc:
            return None, None, translate("reference.other_backend_unavailable",
                                         language, exc=exc)
        other_ops = (host if other == "numpy"
                     else [other_xp.asarray(item) for item in host])

        def call():
            return pe_contract(equation, other_ops, other, specialize, cache)
        try:
            seconds = _timed(call, warmup, rounds)
            result = _to_host(call())
            return result, seconds, (
                f"pyqcd.contract(backend={other!r}, {other}, "
                f"specialize={_specialize_label(specialize)})")
        except Exception as exc:                # noqa: BLE001
            return None, None, translate("reference.other_backend_failed", language,
                                         exc_type=type(exc).__name__, exc=exc)
    return None, None, translate("reference.unknown_compare", language,
                                 compare=compare)


def pe_contract(equation, operands, backend, specialize=None, cache=None):
    """``compare='backend'`` 用：在指定后端上跑同一批张量。

    ``specialize`` 与 ``cache`` 都与主路径同源——否则 ``--compare backend`` 会把
    "专用路径 vs 通用路径"当成跨后端差异来比较，结论失真；``cache=false`` 也会被
    参照这一路绕过。
    """
    import pyqcd.einsum as pe
    return pe.contract(equation, *operands, backend=backend, specialize=specialize,
                       cache=cache)


def _install_info() -> dict[str, Any]:
    """报告**实际被加载的** pyqcd 版本与包路径。

    为什么需要它：``python -m pyqcd.einsum.test`` 用哪一个 pyqcd 由 ``sys.path``
    决定，而 conda 环境里可能同时存在 ``site-packages/pyqcd``（旧版本普通安装）
    与仓库源码。用户实际踩过：base 环境装的是 pyqcd 0.1.3，在仓库目录外运行
    ``-m pyqcd.einsum.test`` 直接 ``No module named``。版本与路径打到 verbose 里，
    这类问题一眼可见。
    """
    info: dict[str, Any] = {}
    try:
        import pyqcd
        info["version"] = getattr(pyqcd, "__version__", "?")
        info["package"] = pyqcd.__file__
    except Exception as exc:                    # noqa: BLE001
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def _memory_snapshot(xp) -> dict[str, Any]:
    """采集进程内存与显存占用。

    只在**计时循环之外**调用（操作数构造后、收尾时各一次），因此不进入任何
    measured region，不污染 ``--rounds`` 的中位数。三项都是 O(1) 查询或一次
    ``/proc`` 读取，开销在微秒量级：

    * ``rss_mib``：进程常驻集（``/proc/self/status`` 的 VmRSS，不受 allocator
      缓存假象影响）；
    * ``gpu_used_mib``：cupy 显存池**已分配**总量（``memory_pool().used_bytes()``）；
    * ``gpu_total_mib``：该设备总显存。
    """
    info: dict[str, Any] = {}
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    info["rss_mib"] = round(int(line.split()[1]) / 1024.0, 1)
                    break
    except OSError:
        pass
    if xp is not np:
        try:
            import cupy
            pool = cupy.get_default_memory_pool()
            info["gpu_used_mib"] = round(pool.used_bytes() / 2 ** 20, 1)
            info["gpu_reserved_mib"] = round(pool.total_bytes() / 2 ** 20, 1)
            free, total = cupy.cuda.runtime.memGetInfo()
            info["gpu_total_mib"] = round(total / 2 ** 20, 1)
            info["gpu_free_mib"] = round(free / 2 ** 20, 1)
        except Exception as exc:                # noqa: BLE001
            info["gpu_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _memory_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """前后快照相减，得到本次运行的内存占用增量，并保留绝对值。

    为什么两者都要：小规模张量（如 256×256 = 1 MiB）的 RSS 增量可能是 **0**——
    Python 进程的 arena 早已持有足够内存，分配器不需要向 OS 再要一页，``VmRSS``
    就不会涨。只看增量会误以为"没占内存"。绝对值 ``*_after`` 让读者仍能看到规模。
    """
    delta: dict[str, Any] = {}
    for key in ("rss_mib", "gpu_used_mib"):
        if before.get(key) is not None and after.get(key) is not None:
            delta[key] = round(after[key] - before[key], 1)
            delta[f"{key}_after"] = after[key]
    for key in ("gpu_total_mib", "gpu_free_mib"):
        if after.get(key) is not None:
            delta[key] = after[key]
    return delta


def _print_detail(report: dict[str, Any], language: str = "zh") -> None:
    """``--verbose true`` 时的人类可读明细。

    键名（``equation``/``device``/``time``/``rel``…）保留英文——它们是稳定的机器
    可读标签；只翻译值里的说明文字。``language`` 取 ``"en"``/``"zh"``。
    """
    out = sys.stderr
    join = translate("detail.joiner", language)
    print(f"equation : {report['equation']}", file=out)
    print(f"device   : {report['device']}  dtype={report['dtype']}  "
          f"backend={report['backend']}", file=out)
    print("operands : " + translate(
        "detail.operands", language, n=report["n_operands"],
        elements=report["input_elements"], out_shape=report["out_shape"],
        out_elements=report["output_elements"]), file=out)
    # 每个操作数的**完整形状**单独一行：``operands`` 那行只给总数与输出，
    # 排查"哪一路的轴长不对"时必须看到逐个形状（2026-11-15 用户要求）。
    shapes = report.get("shapes") or []
    if shapes:
        # 用 ", " 分隔（不随语言变），便于直接复制粘贴回 --shape。
        rendered = ", ".join("[" + ",".join(str(int(d)) for d in shape) + "]"
                             for shape in shapes)
        print("shape    : " + translate("detail.shape", language, n=len(shapes)), file=out)
        print("           " + rendered, file=out)
    else:
        print("shape    : " + translate("detail.shape_empty", language), file=out)
    if "specialize" in report:
        effective = "unknown" if "specialize_effective" not in report else str(
            bool(report["specialize_effective"]))
        print("specialize: " + translate(
            "detail.specialize", language, requested=report["specialize"],
            effective=effective), file=out)
    if "seconds" in report:
        print("time     : " + translate(
            "detail.time", language, ms=report["seconds"] * 1000,
            warmup=report["warmup"], rounds=report["rounds"]), file=out)
    if report.get("compare_optimize") is not None:
        effective = report.get("compare_optimize_effective")
        shown = translate("detail.ref_opt_not_applicable", language) \
            if effective is None else repr(effective)
        print("ref opt  : " + translate(
            "detail.ref_opt", language, requested=report["compare_optimize"],
            shown=shown), file=out)
    if report.get("reference"):
        print(f"reference: {report['reference']}", file=out)
    ref_seconds = report.get("reference_seconds")
    own_seconds = report.get("seconds")
    if ref_seconds is not None:
        ratio = ""
        if own_seconds:
            ratio = translate("detail.ratio", language,
                              ratio=own_seconds / ref_seconds)
        print(f"ref time : {ref_seconds * 1000:.3f} ms{ratio}", file=out)
    if "rel" in report:
        print(f"rel      : {report['rel']:.3e}  (atol={report['atol']:g})", file=out)
    memory = report.get("memory") or {}
    inputs_delta = memory.get("inputs_delta") or {}
    peak = memory.get("peak") or {}
    parts = []
    if inputs_delta.get("rss_mib") is not None:
        parts.append(translate("detail.memory_inputs_rss", language,
                               mib=inputs_delta["rss_mib"]))
    if inputs_delta.get("gpu_used_mib") is not None:
        parts.append(translate("detail.memory_inputs_gpu", language,
                               mib=inputs_delta["gpu_used_mib"]))
    if parts:
        print(f"memory   : {join.join(parts)}", file=out)
    parts = []
    if peak.get("rss_mib") is not None:
        after = peak.get("rss_mib_after")
        suffix = (translate("detail.memory_peak_rss_after", language, mib=after)
                  if after is not None else "")
        parts.append(f"RSS +{peak['rss_mib']:.0f} MiB{suffix}")
    if peak.get("gpu_used_mib") is not None:
        parts.append(translate("detail.memory_peak_gpu", language,
                               mib=peak["gpu_used_mib"]))
    if peak.get("gpu_reserved_mib") is not None:
        parts.append(translate("detail.memory_peak_pool", language,
                               mib=peak["gpu_reserved_mib"]))
    if peak.get("gpu_total_mib") is not None:
        parts.append(translate("detail.memory_peak_total", language,
                               mib=peak["gpu_total_mib"]))
    if parts:
        print(f"peak     : {join.join(parts)}", file=out)
    install = report.get("install") or {}
    if install.get("version"):
        print(f"pyqcd    : {install['version']}  {install.get('package', '?')}",
              file=out)
    if report.get("specialize_warning"):
        print(f"warning  : {report['specialize_warning']}", file=out)
    if "kernel_info" in report:
        kernel = report["kernel_info"] or translate("detail.kernel_none", language)
        print(f"kernel   : {kernel}", file=out)
        # 把"实际走的入口"单独拎成一行：``kernel`` 那个嵌套 dict 要读两层才看得懂，
        # 而形态提示（``topology:xxxx``）说的是**形态**、不承诺本次走哪条入口。
        # 这一行才是"到底跑了什么"的权威答案（2026-11-15 用户要求）。
        route = _route_of(report.get("kernel_info"), language,
                          report.get("kernel_force"))
        if route:
            print(f"route    : {route}", file=out)
    # 强制内核的三段当然也要在：``kernel`` 行说的是原生入口留下的信息，
    # 分不清"要求了什么/到底跑没跑成"。这一行**追加**在 route 之后，不动既有行序。
    if "kernel_force" in report:
        print(kernel_force_line(report["kernel_force"]), file=out)
    equivalent = report.get("equivalence")
    if equivalent is not None:
        _print_equivalence(equivalent, language)
    # 缓存统计**追加**在最后：既有行的顺序与内容一律不动（契约逐字符比对）。
    # 只在 ``--cache true|false`` 显式给出时出现——``auto``/省略时多一行就是破坏。
    if report.get("cache_explicit"):
        print(cache_line(report), file=out)


def _print_equivalence(equivalent: dict[str, Any], language: str = "zh") -> None:
    """打印``--equivalence`` 的明细段（只在未命中特化时由 ``run()`` 写入）。

    键名固定 ``equiv``，值按语言翻译。**输入侧与输出侧必须同时报出**：只给
    "怎么改张量"而不给"结果怎么回填"，用户会拿到轴序错了的答案。某一侧为空时
    明确写 ``-``（不省略整行）。
    """
    out = sys.stderr
    indent = " " * 11
    # 该模块不可用时（例如旧版 _equivalence 尚未提供 explain）如实报告，
    # 不让报告层的问题伪装成"没有等价写法"。
    if equivalent.get("unavailable"):
        print("equiv    : " + translate("detail.equiv_unavailable", language,
                                        exc=equivalent["unavailable"]), file=out)
        return
    if not equivalent.get("rewritten"):
        print("equiv    : " + translate("detail.equiv_none", language), file=out)
        return
    print("equiv    : " + translate("detail.equiv_header", language,
                                    kind=equivalent.get("kind", "-")), file=out)
    print(indent + translate("detail.equiv_equation", language,
                             equation=equivalent.get("equation", "-")), file=out)
    print(indent + translate("detail.equiv_tensors", language,
                             adjustments=_format_operand_adjustments(equivalent)),
          file=out)
    print(indent + translate("detail.equiv_result", language,
                             perm=_format_output_adjustment(equivalent)), file=out)
    print(indent + translate("detail.equiv_note", language), file=out)


def _route_of(kernel_info: Any, language: str = "zh",
              force_fields: Any = None) -> str:
    """把 ``kernel_info`` 翻译成一句人话：**通用内核还是特化内核**，以及哪个原生入口。

    用户要的是"到底跑了什么"，而 ``kernel`` 那个嵌套 dict 要读两层才看得懂，
    形态提示（``topology:xxxx``）说的又是**形态**、不承诺本次走哪条入口。
    这里按 ``LAST_INFO['path']`` 给出确定答案：

    =========================  ========================  ==========================
    ``path``                   执行层                     底层符号
    =========================  ========================  ==========================
    ``kupl_dedicated``         原生 · 特化（展开步表）    ``qcd_kupl_dedicated_apply``
    ``kupl_topology``          原生 · 特化（v3 步表）     ``qcd_kupl_topology_apply``
    ``kupl_topology_v2``       原生 · 特化（v2 回落）     ``qcd_kupl_topology_apply``
    ``cuda_dedicated``         原生 · 特化（CUDA 步表）   ``qcd_cuda_dedicated_apply``
    ``cuda_topology``          原生 · 特化（CUDA 拓扑）   ``qcd_cuda_topology_apply``
    （无 path）                 通用（Python/numpy/BLAS）  ``numpy.einsum``/BLAS
    =========================  ========================  ==========================

    返回形如 ``特化 · 原生 qcd_kupl_dedicated_apply`` 或 ``通用（无原生命中）``。

    ``force_fields`` 是 ``report['kernel_force']`` 整理后的三段（可选）。它只在
    ``--kernel-force backend`` 那一支改变结论：那一路**直接**调后端 ``einsum``、
    一次原生入口都不碰，``kernel_info`` 自然为空——只按 ``kernel_info`` 判会把它
    说成"通用（无原生命中）"，而真实路线是**后端 einsum**。这不是新增一行，是把
    既有 ``route`` 行的结论在"确实走了 backend"时说准。
    """
    if isinstance(force_fields, dict) and force_fields.get("ran") == BACKEND_LABEL:
        return translate("detail.route_backend", language)
    if not isinstance(kernel_info, dict) or not kernel_info:
        return translate("detail.route_generic", language)
    native_symbols = {
        "kupl_dedicated": "qcd_kupl_dedicated_apply",
        "kupl_topology": "qcd_kupl_topology_apply",
        "kupl_topology_v2": "qcd_kupl_topology_apply",
        "cuda_dedicated": "qcd_cuda_dedicated_apply",
        "cuda_topology": "qcd_cuda_topology_apply",
    }
    joiner = translate("detail.route_joiner", language)
    labels: list[str] = []
    for _backend, info in kernel_info.items():
        path = info.get("path") if isinstance(info, dict) else info
        if not path:
            continue
        symbol = native_symbols.get(str(path), str(path))
        labels.append(translate("detail.route_specialized", language,
                                symbol=symbol, path=path))
    if not labels:
        return translate("detail.route_generic", language)
    return joiner.join(labels)
    return "; ".join(labels) if labels else "通用（无原生命中）"


# ── 遍历调优（--tune） ──────────────────────────────────────────────────────
#: 三个分支的稳定名（**机器可读**，不翻译），见 ``_tune_axes``。
TUNE_BRANCH_RAW = _tune_axes.TUNE_BRANCH_RAW
TUNE_BRANCH_EQUIVALENT = _tune_axes.TUNE_BRANCH_EQUIVALENT
TUNE_BRANCH_NONE = _tune_axes.TUNE_BRANCH_NONE
TUNE_BRANCHES = _tune_axes.TUNE_BRANCHES
#: 剪枝时的固定原文（用户口径，逐字不可改写）。
TERMINATION_MESSAGE = _tune_axes.TERMINATION_MESSAGE
#: ``原因=`` 的拼装：固定原文 + ``wall_ms`` 与 ``min_wall_ms`` 两个数值。
prune_reason = _tune_axes.prune_reason
#: ``分支 → 有序清单``（组数与逐项顺序的唯一声明处）。
tune_plan = _tune_axes.tune_plan


#: ``--tune`` 取值的**稳定名**（机器可读，不翻译）：``all`` / ``pass`` 两种模式，
#: ``off`` 表示"不跑遍历"（不给该选项，或显式 ``false`` / 空串）。
TUNE_MODE_NAMES = (TUNE_MODE_ALL, TUNE_MODE_PASS, TUNE_MODE_OFF)


def parse_tune_mode(value: bool | None | str,
                    language: str = "zh") -> str:
    """``--tune`` 的取值 → ``"all"`` / ``"pass"`` / ``"off"``（**唯一解析点**）。

    用户 2026-11-18 的裁定把 ``--tune`` 从开关变成了**带取值的模式参数**：

    ======================  ==================================================
    输入                   语义
    ======================  ==================================================
    留空 / ``None`` / ``True`` / ``true`` / ``True`` / ``all``   ``all``（完整模式）
    ``pass``                ``pass``（快速通过模式）
    ``false`` / 空串 / ``"0"`` / ``False``                      ``off``（不跑）
    ======================  ==================================================

    "留空"在 argparse 层是 ``const="true"``（``nargs="?"``），API 层是 ``None`` 或
    ``True``——三者在**同一个**出口归一成 ``"all"``，下游只看模式名，不再各自判布尔。

    未知取值抛 ``UsageError``（退出码 2）而不是静默当 ``off``：拼错一个字母就得到
    "跑了一遍却没有输出"比直接报错更难查（tune 关闭时字面上什么都没有）。
    """
    if value is None or value is True:
        return TUNE_MODE_ALL
    if value is False:
        return TUNE_MODE_OFF
    text = str(value).strip().lower()
    if text in ("", "false", "0", "no", "off"):
        return TUNE_MODE_OFF
    if text in ("true", "1", "yes", "on", TUNE_MODE_ALL):
        return TUNE_MODE_ALL
    if text == TUNE_MODE_PASS:
        return TUNE_MODE_PASS
    raise UsageError(translate("error.tune_mode", language,
                               values=", ".join(TUNE_MODES), value=value))


def tune_mode_enabled(value: bool | None | str,
                      language: str = "zh") -> str | None:
    """:func:`parse_tune_mode` 的"要不要跑"版本：``off`` → ``None``，其余原样。"""
    mode = parse_tune_mode(value, language)
    return None if mode == TUNE_MODE_OFF else mode


def _tune_init(line: str) -> None:
    """打一行初始化过程输出（**机器可读**、不跟 ``--language`` 翻译、只走 stderr）。

    为什么必须逐阶段打：``--tune`` 在长方程上可能很久没有一行输出，卡住时无法区分
    "还在探测"与"已经死在里面"。这七行把初始化切成有起有止的阶段，最后一行
    ``first group start`` 一旦出现，说明遍历真的开始了。
    """
    print(TUNE_INIT_PREFIX + line, file=sys.stderr, flush=True)


def _tune_init_ms(started: float) -> str:
    """阶段耗时：整毫秒（0 位小数），**同一步的起止**用同一个口径。"""
    return f"{int(round((time.perf_counter() - started) * 1000.0))}ms"


def _tune_init_equation(equation: str) -> str:
    """方程的过程输出摘要：前 :data:`TUNE_INIT_EQUATION_CHARS` 个字符（单行、不截断词）。"""
    text = "".join(str(equation).split())
    if len(text) <= TUNE_INIT_EQUATION_CHARS:
        return text
    return text[:TUNE_INIT_EQUATION_CHARS]


def _tune_init_shapes(shapes: Sequence[Sequence[int]]) -> tuple[int, float]:
    """``(输入元素总数, 估计字节 MiB)``：规模大到什么程度，先摆出来再开始跑。

    字节按 ``complex128``（16 B/元素）估：它是本入口的缺省 dtype，也是**最大**的一档
    ——给出的数因此是"至多"，不会把一个大形状说小。真实 dtype 由下一行自己带出来。
    """
    total = 0
    for shape in shapes:
        elements = 1
        for axis in shape:
            elements *= int(axis)
        total += elements
    return total, float(total) * 16.0 / (1024.0 * 1024.0)


def tune_branch(equation: str, language: str = "zh") -> dict[str, Any]:
    """**只探测、不计时**：这条方程该走哪一条遍历清单（规格 §2）。

    三条判据都用库自己的两个入口，不自己解析方程：

    * ``raw_ok = _specialize.match(equation) is not None`` —— 原方程的形态能否特化；
    * ``mapping = _equivalence.normalize_equation(equation)``、
      ``equiv_eq = mapping.equation``、
      ``equiv_ok = mapping is not None and _specialize.match(equiv_eq) is not None``.

    两个都为假说明这条方程既没有特化形态、也没有能落到特化形态的等价写法，于是
    ``topology`` 那一组根本不该试（试了也只能报不可用）。
    探测**不写配置表、不构造张量、不计时**——它是判定，不是测量。
    """
    from pyqcd.einsum import _equivalence, _specialize

    raw_ok = _specialize.match(equation) is not None
    mapping = None
    equiv_eq = None
    equiv_ok = False
    try:
        mapping = _equivalence.normalize_equation(equation)
    except Exception:                           # noqa: BLE001 - 探测失败按"无映射"
        mapping = None
    if mapping is not None:
        equiv_eq = str(getattr(mapping, "equation", "") or "")
        try:
            equiv_ok = bool(equiv_eq) and _specialize.match(equiv_eq) is not None
        except Exception:                       # noqa: BLE001 - 同上
            equiv_ok = False
    if raw_ok:
        branch = TUNE_BRANCH_RAW
    elif equiv_ok:
        branch = TUNE_BRANCH_EQUIVALENT
    else:
        branch = TUNE_BRANCH_NONE
    return {
        "branch": branch,
        "raw_ok": raw_ok,
        "equiv_ok": equiv_ok,
        "target_equation": equiv_eq or None,
    }


def tune_combos(language: str = "zh", *,
                equation: str | None = None) -> list[tuple[str, str, str]]:
    """``--tune`` 的**有序**组合清单：``(kernel_force, as_equivalence, optimize)``。

    不再是笛卡尔积：清单按"最可能快"排序（共同前缀 → 分支专属 → ``backend`` 尾部），
    好让 :class:`_tune_axes.MinWall` 的上界尽早被压低、后面的格子一超界就被终止。
    顺序与组数是冻结口径，见 :mod:`pyqcd.einsum._tune_axes`。

    ``equation`` 省略时按"两轴都不可特化"的分支（``B3``）给清单——这是**不依赖任何
    方程**的最小清单，供只需要"有哪些格子"的调用方使用；真跑 tune 一律显式传方程。

    组数超过 :data:`TUNE_MAX_COMBOS` 时抛 ``UsageError``（退出码 2），**绝不**截断
    ——截断后的"最优"不再是这张清单里的最优，而报告里看不出来。
    """
    if equation is None:
        combos = _tune_axes.tune_plan(TUNE_BRANCH_NONE)
    else:
        combos = _tune_axes.tune_plan(
            tune_branch(equation, language)["branch"])
    if len(combos) > TUNE_MAX_COMBOS:
        raise UsageError(translate("error.tune_grid", language,
                                   count=len(combos), limit=TUNE_MAX_COMBOS))
    return combos


def _tune_combos_for(equation: str,
                     language: str) -> list[tuple[str, str, str]]:
    """本轮的清单来源：等价写法是 ``tune_combos(language, equation=equation)``。

    为什么单独一层：既有契约用 ``cli.tune_combos`` 的**一元**替身
    （``lambda language="zh": [...]``）注入小清单，好让"坏组合不留配置行"那几条断言
    跑在 3 组上而不是 21 组上。替身不收 ``equation``，于是这里退回只传语言——注入的
    清单与真清单走的是同一条路径（都是"一串三元组"），不改变遍历语义。
    """
    try:
        return tune_combos(language, equation=equation)
    except TypeError:
        return tune_combos(language)


#: 一组的"**遍历前就判不可用**"状态（机器可读、**不翻译**）。它与 ``error`` 是两件
#: 事：``error`` 说的是"这一组跑出了意外"，而 ``skip`` 说的是"该内核对这个形状本来
#: 就没有路"——后者逐组跑一遍只会得到同一句错误的 N 份副本（2026-11-18 用户实测：
#: 22 组里 8 组是这种噪音，真正的问题被淹掉）。
TUNE_STATUS_SKIP = "skip"


def _dedicated_unavailable_reason(equation: str, shapes: Sequence[Sequence[int]],
                                 dtype: Any, backend: str,
                                 device: str = "cpu") -> str | None:
    """``dedicated`` 在**这条方程/形状/dtype/后端**上能不能用（只查准入，不执行）。

    判据全部复用内核自己的函数，**不新造阈值**：``enabled()``（原生库/开关）、
    ``dedicated_dtypes()``（dtype 有没有步表路径）、``_dedicated_cached()``（这条
    收缩能不能编码成固定步表，只规划不下发）、
    ``_cpu_plan_fallback_reason`` / ``_plan_fallback_reason``（**计划 arena 超预算**
    这一类拒绝，原文里就带着阈值与两个数字，报告里逐字打出来）。

    返回 ``None`` = 可用；否则返回内核自己的**原文**（用户能与内核日志里同一句话对上）。
    探测自身出错时同样返回 ``None``：宁可照旧逐组跑（看得见错误），也不凭一次失败
    的探测把一个其实可用的内核整组裁掉。
    """
    try:
        from pyqcd.einsum import _kernel_cuda, _kernel_kupl

        name = str(backend or "").strip().lower()
        if name in ("", "auto"):
            # ``--backend auto`` 时真正生效的后端由设备定（与 ``_tune_key`` 的口径一致）：
            # 拿错后端等于用另一个内核的预算去判它（CPU 8 GiB vs CUDA 3 GiB）。
            gpu = str(device or "").strip().lower() in ("gpu", "cuda")
            name = "cupy" if gpu else "numpy"
        cupy = name in ("cupy", "cuda", "gpu")
        if not cupy and name not in ("numpy", "cpu", "torch"):
            return None                   # 未知后端：不猜，交给逐组实测
        module = _kernel_cuda if cupy else _kernel_kupl
        if cupy:
            try:
                import cupy  # noqa: F401 - 只探"装没装"
            except ImportError:
                return "cupy is not installed"
        enabled = getattr(module, "enabled", None)
        if not callable(enabled) or not enabled():
            return ("kernel disabled (native library missing, or "
                    "PYQCD_EINSUM_KERNEL_* = 0)")
        if not cupy and name == "torch":
            # ``try_dedicated`` 只吃 numpy / cupy 数组，torch 后端的步表路径因此恒不可用。
            return "operands must be numpy arrays"
        np_dtype = np.dtype(dtype)
        dedicated_dtypes = getattr(module, "dedicated_dtypes", None)
        supported = tuple(dedicated_dtypes()) if callable(dedicated_dtypes) else ()
        if supported and np_dtype not in supported:
            return ("dtype {} has no dedicated step-table path (supported: {}); "
                    "no implicit downcast is performed".format(
                        np_dtype, ", ".join(str(item) for item in supported)))
        optimize = _kernel_kupl._effective_dedicated_optimizer("auto")
        entry = _kernel_kupl._dedicated_cached(
            equation, tuple(tuple(int(axis) for axis in shape) for shape in shapes),
            np_dtype, optimize, "cuda" if cupy else "kupl")
        if entry is None:
            return "the contraction path is not expressible as a fixed step table"
        stats = _kernel_kupl._plan_statistics(entry[0])
        gate = (module._plan_fallback_reason(np_dtype, stats) if cupy
                else module._cpu_plan_fallback_reason(stats))
        return None if gate is None else str(gate)
    except Exception:                     # noqa: BLE001 - 探测失败不许裁掉可用内核
        return None


def _topology_unavailable_reason(equation: str,
                                 branch: dict[str, Any] | None) -> str | None:
    """``topology`` 只在这条方程（或其等价改写）能特化时可用。

    判据就是分支探测用的那两个 ``_specialize.match``：原方程能特化 → ``B1`` 的
    ``as_equivalence=false`` 那组可用；只有等价写法能特化 → ``B2`` 的
    ``as_equivalence=true`` 那组可用；两者都不行 → 整组跳过（试了也只能报不可用）。

    **命中语料 ≠ 能执行**：``match`` 只算拓扑签名、查语料索引，而原生侧另有两道门
    （编译期步表的覆盖、运行时的 ``status 4`` / ``QCD_TOPOLOGY_UNSUPPORTED``）。所以
    "``match`` 命中"只是本函数能给出的**上界**；真不可执行时那一组仍会落到 ``error``。
    要让这类失败也变 ``skip``，唯一可靠办法是准入时真跑一次，代价见
    ``docs/einsum_tune_timeout_20261119.md`` §7 结尾。
    """
    try:
        from pyqcd.einsum import _specialize
    except ImportError:                   # pragma: no cover - 旧版库
        return None
    probe = branch or tune_branch(equation)
    try:
        if probe.get("raw_ok"):
            return None
        target = str(probe.get("target_equation") or "")
        if target and _specialize.match(target) is not None:
            return None
    except Exception:                     # noqa: BLE001 - 探测失败不裁
        return None
    return ("the equation is not specializable and neither is its equivalent "
            "spelling (_specialize.match missed)")


def tune_unavailable_kernels(*, equation: str,
                             shapes: Sequence[Sequence[int]] | None = None,
                             dtype: Any = None, backend: str = "auto",
                             device: str = "cpu",
                             branch: dict[str, Any] | None = None
                             ) -> dict[str, str]:
    """遍历**前**的准入探测：``{kernel_force: 不可用原因}``（不执行任何收缩）。

    ``generic`` / ``backend`` 恒可用（它们就是通用路径与后端 einsum），所以这里只探
    ``dedicated`` 与 ``topology``，别的内核名不猜。``shapes`` 缺失时只探 topology
    ——``dedicated`` 的准入必须知道形状（步表按形状编码）。``backend`` 是 ``auto``
    时由 ``device`` 定实际内核（GPU/``cuda`` → CUDA，其余 → KUPL），否则两个内核的
    预算（3 GiB / 8 GiB）会张冠李戴。
    """
    reasons: dict[str, str] = {}
    for kernel in TUNE_KERNEL_FORCE:
        reason: str | None = None
        if kernel == "dedicated":
            if shapes is not None and dtype is not None:
                reason = _dedicated_unavailable_reason(
                    equation, shapes, dtype, backend, device)
        elif kernel == "topology":
            reason = _topology_unavailable_reason(equation, branch)
        if reason:
            reasons[kernel] = str(reason)
    return reasons


def tune_plan_trimmed(branch: str,
                      unavailable: Sequence[str] | None = None
                      ) -> list[tuple[str, str, str]]:
    """:func:`tune_plan` **裁剪后**的清单（去掉 ``unavailable`` 里那些内核的组）。"""
    kept, _dropped = _tune_axes.plan_without(
        _tune_axes.tune_plan(branch), unavailable)
    return kept


def tune_plan_sizes_trimmed(unavailable: Sequence[str] | None = None
                            ) -> dict[str, int]:
    """三个分支**裁剪后**的组数（``B1``/``B2``/``B3`` 各自算，不共享一个数）。"""
    return {branch: _tune_axes.trimmed_plan_size(branch, unavailable)
            for branch in TUNE_BRANCHES}


def tune_conflicts(*, kernel_force: str | None = None,
                   as_equivalence: bool | None | str = None,
                   optimize: Any = None, language: str = "zh") -> list[str]:
    """``--tune`` 与"它自己会遍历的三维"撞车时：**每个参数打一行警告，然后忽略它**。

    为什么是警告而不是用法错误：``--tune`` 的价值就在于"别再让我手动挑内核/等价/
    收缩序"，用户顺手写上一维（尤其脚本里统一传 ``--kernel-force dedicated``）不该
    让整轮遍历以退出码 2 收场；而**悄悄**忽略又会让用户以为那一维真的生效了。折中：
    每个撞车的参数在 stderr 上给**一行**（``--tune 会自行遍历 …，已忽略你给的 '<值>'``），
    遍历照常进行、退出码仍为 0，报告里也能看到网格覆盖了哪些取值。

    ``kernel_force`` 只有解析出**具体内核**才算撞车：``generic`` 是缺省值、且网格本身
    就遍历它，显式写它不改变任何东西，也就没有可警告的。非法标签仍按原来的用法错误
    报（退出码 2）——那是拼写问题，不是"被忽略"。

    返回被忽略的三轴名（``["--kernel-force"]`` 之类），供调用方/测试断言。
    """
    ignored: list[str] = []
    kernel_labels_value = (kernel_labels(kernel_force, language)
                           if kernel_force is not None else ())
    if kernel_labels_value and kernel_labels_value != (GENERIC_LABEL,):
        ignored.append("--kernel-force")
        _warn_tune_ignored(translate("tune.ignored", language, swept=TUNE_AXES,
                                     given=f"--kernel-force={kernel_force}"),
                           language)
    if as_equivalence is not None:
        if as_equivalence is True:
            text = "true"
        elif as_equivalence is False:
            text = "false"
        else:
            text = str(as_equivalence).strip().lower()
        if text != "auto":
            ignored.append("--as-equivalence")
            _warn_tune_ignored(translate("tune.ignored", language, swept=TUNE_AXES,
                                         given=f"--as-equivalence={text}"), language)
    if optimize is not None and str(optimize).strip().lower() != "auto":
        ignored.append("--optimize")
        _warn_tune_ignored(translate("tune.ignored", language, swept=TUNE_AXES,
                                     given=f"--optimize={optimize}"), language)
    return ignored


def _warn_tune_ignored(message: str, language: str = "zh") -> None:
    """一行 stderr 警告（与 ``result.warning`` 同一前缀，既有输出行不动）。"""
    print(translate("result.warning", language, message=message), file=sys.stderr)


#: ``--tune`` 里真实参照（一次完整 ``np.einsum``）的**输入元素总数上限**。
#: 超过就跳过参照、改用"首个成功组合"的自洽校验：生产形状上的完整收缩可能几分钟
#: 且不可中断，而 tune 不该为了一次参照把整轮拖死（2026-11-18 用户实测卡死）。
#: 200 万元素（complex128 约 32 MB）足以覆盖契约与中小方程，同时把大方程挡在外面。
#: 用户那条 10 算子生产方程实测 **256,010,297** 个输入元素（本门限的 **128 倍**），
#: 即它必然走自洽参照那条路——这正是"卡死"与"校验仍在"能同时成立的唯一办法。
TUNE_REFERENCE_MAX_ELEMENTS = 2_000_000


def _tune_reference(equation: str, host: Sequence[Any]) -> tuple[Any, str]:
    """``--tune`` 的唯一参照：``np.einsum(optimize="greedy")``（numpy 参照的等价写法）。

    ``--compare`` 在 ``--tune`` 下缺省关闭，所以参照**只在这里算一次**，不进任何一组的
    计时区——否则 22 组就要各付一次完整参照的耗时。算不出来时返回 ``(None, 原因)``。

    **跳过参照不能让数值校验消失**：``rel`` 依赖参照，参照为 ``None`` 时所有组
    ``rel=None``、``mismatch`` 恒假，契约里"``rel`` 超容差必须判 MISMATCH"会静默变成
    永真。所以遍历循环对大形状改用"**首个成功组合**的结果"做自洽校验（成本为零，仍能
    抓住"不同内核算出不同数"）。改这里时只允许两种写法：小形状真参照，或自洽参照；
    **不能**只把参照设成 ``None`` 就完事。判据与实测见
    ``docs/einsum_tune_timeout_20261119.md`` §5。
    """
    # 大规模形状上**不跑**参照：这是一次完整的 ``np.einsum`` 收缩，生产尺寸上可能
    # 算几分钟且**不可中断**（SIGALRM 打不断原生代码）。它就发生在任何超时保护之外，
    # 是"tune 卡死无输出"的直接原因。规模上限按**输入元素总数**判，小形状仍做真参照
    # （数值校验因此保留；大形状退化成首个成功组合的自洽校验，见遍历循环）。
    total = 0
    for item in host:
        size = getattr(item, "size", None)
        total += int(size) if size is not None else 0
    if total > TUNE_REFERENCE_MAX_ELEMENTS:
        return None, (f"n/a (inputs {total} elements > "
                      f"{TUNE_REFERENCE_MAX_ELEMENTS}, reference skipped)")
    try:
        return np.einsum(equation, *host, optimize="greedy"), "np.einsum(optimize='greedy')"
    except Exception as exc:                    # noqa: BLE001 - 参照失败不该中断遍历
        return None, f"n/a ({type(exc).__name__}: {exc})"


def _plan_probe(equation: str, shapes: Sequence[Sequence[int]],
                optimize: str) -> dict[str, Any]:
    """问规划器：这一组**要跑多少运算**、以及**规划本身**花了多久。

    返回 ``{"flops": float|None, "seconds": float|None, "limit_hit": bool,
    "error": str|None}``：``flops`` 取不到就是 ``None``；``limit_hit`` 表示规划超过了
    :data:`PLAN_PROBE_SECONDS`（此时 ``seconds`` 只是"至少这么久"的下界）。

    为什么要先问规划器再跑：``--tune`` 里最贵的不是收缩本身，而是"跑一个明知不可行的
    配置"——``optimize=none`` 在生产方程上是**单次 10 元朴素收缩**（本机实测
    ``naive_cost = 7.6441e+17``），硬超时能挡住它，但要先把整组额度花光
    （缺省 ``30s × 7 = 210s``）；而 :func:`pyqcd.einsum.contract_path` 给出这个数只要
    **0.07 ms**。

    为什么不用**朴素**位次代替各优化器自己的位次：那样 ``optimal``/``dp`` 会被误判成
    "要跑 40 小时"（它们的规划结果比朴素小两个量级）。错剪会让真正的 ``best`` 永远
    不出现，而报告里看不出来，所以必须按**这一组自己的**优化器问。

    为什么 ``limit_hit`` 也能剪枝：``--tune-timeout`` 的上限量的是**整组**的
    ``contract()``，而规划是那次调用的一部分。规划**单独**就超过当前 ``min_wall_ms``
    时，这一组必然超过上界——这是**实测**结论（不是估算），只是被测到的是规划段。
    ``optimal`` 在生产方程上规划要 38 s（本机实测），按这一条能在 2 s 内剪掉它。

    只用 ``shapes=True`` 的 ``contract_path``，**不建张量、不跑收缩**，并显式
    ``cache=False``：这一问只为预判，不该污染规划器缓存。
    """
    def probe() -> dict[str, Any]:
        import pyqcd.einsum as pe

        started = time.perf_counter()
        _path, info = pe.contract_path(
            equation, *[tuple(int(axis) for axis in shape) for shape in shapes],
            shapes=True, optimize=_contract_optimize(optimize), cache=False)
        return {"flops": float(info.opt_cost),
                "seconds": time.perf_counter() - started,
                "limit_hit": False, "error": None}

    if not _tune_timer_available():
        try:
            return probe()
        except Exception as exc:                # noqa: BLE001 - 预判失败不影响遍历
            return {"flops": None, "seconds": None, "limit_hit": False,
                    "error": f"{type(exc).__name__}: {exc}"}
    # ``_guarded_call`` **没有返回值**（它只负责装上上限并调用）：结果必须自己收进闭包，
    # 写 ``return _guarded_call(...) or {...}`` 会让每一次预判都退化成"问不出来"，
    # 于是 FLOPs 判据永远不生效——本机实测就是这么漏掉 ``none`` 那一组的。
    box: list[dict[str, Any]] = []

    def record() -> None:
        box.append(probe())

    try:
        _guarded_call(record, PLAN_PROBE_SECONDS, None)
        return box[0]
    except _TuneTimeout:
        # 规划本身太贵：这**已经是**一条可用结论（见 docstring），但这里不替调用方
        # 决定要不要剪——只如实报"至少花了 PLAN_PROBE_SECONDS"。
        return {"flops": None, "seconds": PLAN_PROBE_SECONDS,
                "limit_hit": True, "error": None}
    except Exception as exc:                    # noqa: BLE001 - 预判失败不影响遍历
        return {"flops": None, "seconds": None, "limit_hit": False,
                "error": f"{type(exc).__name__}: {exc}"}


def _tune_route(fields: dict[str, Any], kernel_info: Any) -> str:
    """本组实际走的路线：**稳定短标签**（``generic`` / ``backend`` / 原生 path / ``unavailable``）。

    这一列是机器可读的（同 ``kernel :`` 行），**不跟 ``--language`` 翻译**——表格要能
    用脚本筛。取值只来自 ``last_kernel()`` 与原生入口的 ``LAST_INFO['path']``：
    跑成了就给具体标签（``kupl_dedicated`` 之类），没跑成但有尝试记录给
    ``unavailable``，两者都没有才是 ``generic``——绝不从报告字段反推。
    """
    ran = str(fields.get("ran") or "").strip()
    if ran and ran != "none":
        return ran
    paths = [str(info.get("path") or "")
             for info in (kernel_info or {}).values() if isinstance(info, dict)]
    paths = [path for path in paths if path]
    if paths:
        return "+".join(paths)
    attempted = str(fields.get("attempted") or "").strip()
    if attempted and attempted != "none":
        return "unavailable"
    return "generic"


def _tune_ms(row: dict[str, Any]) -> str:
    """表格里的中位列：没测到就写 ``-``（不写 0，0 会被读成"极快"）。"""
    value = row.get("median_ms")
    return "-" if value is None else f"{value:.6f}ms"


def _tune_rel(row: dict[str, Any]) -> str:
    """表格里的 rel 列：没参照或没测到就写 ``-``。"""
    value = row.get("rel")
    return "-" if value is None else f"{value:.3e}"


#: 一组的分类取值（规格 §12.1 冻结，机器可读、**不翻译**）。``pruned`` 是
#: ``min_wall_ms`` 剪枝：这一组**没被测完**（它比当前上界慢），因此同样不参与选优。
#: ``skip`` 是**遍历前**就判定"该内核对这个形状没有路"（计划 arena 超预算、方程不可
#: 特化……）：它既不是异常也不是"太慢"，不该混进 ``error``。
TUNE_STATUS_OK = "ok"
TUNE_STATUS_TIMEOUT = "timeout"
TUNE_STATUS_ERROR = "error"
TUNE_STATUS_MISMATCH = "mismatch"
TUNE_STATUS_PRUNED = "pruned"
TUNE_STATUSES = (TUNE_STATUS_OK, TUNE_STATUS_TIMEOUT, TUNE_STATUS_ERROR,
                 TUNE_STATUS_MISMATCH, TUNE_STATUS_PRUNED, TUNE_STATUS_SKIP)


def tune_status(row: dict[str, Any]) -> str:
    """一组的状态：``ok`` / ``timeout`` / ``error`` / ``mismatch`` / ``pruned`` / ``skip``。

    ``budget`` 归 ``timeout``：那一组是被**整轮**时间上限截断的（不是它自己太慢），
    而 §12.1 只给了四个取值；记成别的字面量会让"为什么这一格没有数"看不出是时间
    原因。``pruned`` 单独成一档（不折进 ``timeout``）：它说的是"比当前最快的那一组
    还慢"，与"这一组自己超过 ``--tune-timeout``"是两件不同的事，混在一起就没法回答
    "到底是卡死了还是单纯不快"。``skip`` 同样单独成一档：它是**遍历前**的准入结论
    （不是测量结果），折进 ``error`` 会让一次正常遍历看起来像 8 次意外。

    ``predicted``（只凭规划器 FLOPs 就判定不值得跑）**也归 ``pruned``**：两者都是
    "这一格没有被测完"，区别只在依据是估算还是实测，而 ``原因=`` 原文已经写明
    ``plan-only estimate, not measured``。不折进去的话，一行没有 ``median_ms`` 的
    记录会被读成 ``error``——那等于把"我们没跑"说成"它跑崩了"。
    """
    if row.get("pruned") or row.get("predicted"):
        return TUNE_STATUS_PRUNED
    if row.get("skip"):
        return TUNE_STATUS_SKIP
    if row.get("timeout") or row.get("budget"):
        return TUNE_STATUS_TIMEOUT
    if row.get("error"):
        return TUNE_STATUS_ERROR
    if row.get("mismatch"):
        return TUNE_STATUS_MISMATCH
    return TUNE_STATUS_OK if row.get("median_ms") is not None else TUNE_STATUS_ERROR


def tune_comb_lines(index: int, total: int, row: dict[str, Any]) -> list[str]:
    """一组的**完整**一行（规格 §12.1 + §13.2）：序号/总数 + 三轴 + 实际下发的
    ``backend`` / ``dtype`` / ``device`` + ``wall_ms`` + 状态 + 原因原文。

    为什么一行都不能少：``--tune`` 的价值就是"这张网格上哪一格跑得怎么样"，只打
    "最快 5 行"会让"某一格到底跑没跑、为什么没跑"无从回答（实测有方程被某个
    ``kernel_force`` 取值整体占满前 5 行）。§13.2 再要求把该组**实际下发**的性能
    配置印在同一行里：只看 ``kernel_force=dedicated`` 却不看后端，就分不清"内核
    跑不起来"是这一档的问题，还是后端本来就给不了这个内核。

    原因**不截断**：``reason`` 含换行时，其余各行逐字跟在 ``tune due  :`` 之后打出
    （原文一个字不改，也不折叠成"…等 N 项"）。字段名与状态取值是稳定 token，
    **不跟 --language 翻译**——脚本要能用它筛。
    """
    wall = row.get("median_ms")
    wall_text = "-" if wall is None else f"{float(wall):.6f}"
    head = ("{}/{}  kernel_force={}  as_equivalence={}  optimize={}  "
            "backend={}  dtype={}  device={}  "
            "wall_ms={}  状态={}  原因=").format(
        index, total, row.get("kernel_force") or "-",
        row.get("as_equivalence") or "-", row.get("optimize") or "-",
        row.get("backend") or "-", row.get("dtype") or "-",
        row.get("device") or "-", wall_text, tune_status(row))
    reason = str(row.get("reason") or "-")
    lines = reason.splitlines() or ["-"]
    return (["tune comb : " + head + lines[0]]
            + ["tune due  : " + extra for extra in lines[1:]])


def tune_skip_line(row: dict[str, Any],
                   dispatched: dict[str, Any] | None = None) -> list[str]:
    """被**遍历前**排除的内核：一行说清"哪一档、几组、为什么"（状态是 ``skip``）。

    与 :func:`tune_comb_lines` 同一套字段名（``kernel_force`` / ``backend`` / ``dtype``
    / ``device`` / ``wall_ms`` / ``状态`` / ``原因``），区别只有两处：``序号`` 写 ``skip``
    （它不在 ``1/N`` 的编号里——那 N 是裁剪后**实际遍历**的组数），``组数=`` 给这一档
    被去掉多少组。原因保持内核原文，一行一行打出去（多行原文逐行续打）。

    为什么整档一行而不是每组一行：同一句 "plan arena 超预算" 重复 7 遍只会把真正的
    异常挤出屏幕，而"这一档跑没跑"照样一眼可见。
    """
    fields = dispatched or {}
    head = ("skip  kernel_force={}  组数={}  backend={}  dtype={}  device={}  "
            "wall_ms=-  状态={}  原因=").format(
        row.get("kernel_force") or "-", int(row.get("combos") or 0),
        fields.get("backend") or "-", fields.get("dtype") or "-",
        fields.get("device") or "-", tune_status(row))
    lines = str(row.get("reason") or "-").splitlines() or ["-"]
    return (["tune comb : " + head + lines[0]]
            + ["tune due  : " + extra for extra in lines[1:]])


def tune_best(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """最快且**可用**的那一组。

    ``error``（跑不成）、``timeout``（单次超上限）、``budget``（被整轮预算截断）、
    ``pruned``（超过当时的 ``min_wall_ms``、被终止）与 ``mismatch``（数值不符）一律
    不参选：它们的 ``median_ms`` 要么不存在、要么不来自一份完整且合格的样本集。
    """
    usable = [row for row in rows
              if row.get("median_ms") is not None
              and not row.get("error") and not row.get("mismatch")
              and not row.get("timeout") and not row.get("budget")
              and not row.get("pruned")]
    return min(usable, key=lambda row: row["median_ms"]) if usable else None


def _absolute_path(path: Any) -> str | None:
    """把配置表路径钉成**绝对路径**（规格 §13.3 的 ``saved:`` 要求）；空值返回 ``None``。"""
    text = str(path or "").strip()
    return os.path.abspath(text) if text else None


def _display_width(text: Any) -> int:
    """终端**显示**宽度：CJK 全角字符占 2 列。

    ``len("状态")`` 是 2，但等宽字体里它占 4 列；按 ``len`` 补空格，表格就会在含中文的
    那一列整体错位（§13.3 要求"列必须对齐"）。
    """
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
               for ch in str(text))


def _pad_cell(text: str, width: int, *, right: bool) -> str:
    """按**显示宽度**补齐一格（右对齐补在左，左对齐补在右）。"""
    fill = " " * max(width - _display_width(text), 0)
    return fill + text if right else text + fill


def tune_table_lines(tune: dict[str, Any],
                     language: str = "zh") -> list[str]:
    """规格 §13.3 的**末尾对齐表格**：列固定 ``# kernel_force as_equivalence optimize
    wall_ms 状态``，表尾 ``best:`` 与 ``saved:``。

    列宽按本表**实际内容**算，配置名一个字符都不截断——``as_equivalence`` 本身就有 14
    个字符，写死一个更小的宽度会把 ``false`` 截成 ``fals``，读者分不清那一格到底是什么。
    逐组一行的详细版仍在上面（``tune comb``，带 backend/dtype/device 与原因原文）；
    这里给的是"一张能一眼扫完的汇总"，两者内容同源（都读 ``tune['rows']``），不另算一份。

    表尾两行是**稳定形态**（``best:`` / ``saved:`` 不跟 ``--language`` 翻译）：
    ``best:`` 给出胜出组与它的 ``wall_ms``（没有可用组合时写 ``none``），``saved:``
    给出配置表**绝对路径**与本次 ``rows=+N deleted=-M``（§12.2 的写入/删除计数）。
    """
    rows = list(tune.get("rows") or [])
    headers = ("#", "kernel_force", "as_equivalence", "optimize", "wall_ms",
               "状态")
    body: list[tuple[str, ...]] = []
    total = len(rows)
    for index, row in enumerate(rows, 1):
        wall = row.get("median_ms")
        body.append((
            f"{index}/{total}",
            str(row.get("kernel_force") or "-"),
            str(row.get("as_equivalence") or "-"),
            str(row.get("optimize") or "-"),
            "-" if wall is None else f"{float(wall):.6f}",
            tune_status(row),
        ))
    widths = [max([_display_width(headers[col])]
                  + [_display_width(item[col]) for item in body])
              for col in range(len(headers))]
    # ``#`` 与 ``wall_ms`` 右对齐（数字竖排好看），名字列左对齐（不东倒西歪）。
    right = {0, 4}

    def line(cells: Sequence[str]) -> str:
        return "  ".join(
            _pad_cell(cell, widths[col], right=col in right)
            for col, cell in enumerate(cells))

    lines = [line(headers), line(tuple("-" * width for width in widths))]
    lines.extend(line(item) for item in body)
    best = tune.get("best")
    if best is None:
        lines.append("best: none  (" + translate("tune.none", language) + ")")
    else:
        wall = best.get("median_ms")
        lines.append(
            "best: kernel_force={}  as_equivalence={}  optimize={}  wall_ms={}"
            .format(best.get("kernel_force") or "-",
                    best.get("as_equivalence") or "-",
                    best.get("optimize") or "-",
                    "-" if wall is None else f"{float(wall):.6f}"))
    write = tune.get("write") or {}
    saved = _absolute_path(_profile_file()) or KERNEL_UNKNOWN
    footer = "saved: {} rows=+{} deleted=-{}".format(
        saved, len(write.get("rows") or []), int(write.get("deleted") or 0))
    reason = str(write.get("reason") or "")
    if reason:
        footer += "  reason=" + reason
    lines.append(footer)
    # ``pass`` 模式：表尾**必须**有一行明说"没做数值校验"。整张表的 ``rel`` 列全是
    # ``-``、也没有任何 MISMATCH 判定——不写这一行，读者只会以为"全都没问题"。
    if tune.get("pass_mode"):
        lines.append("mode: " + TUNE_PASS_SKIP_NOTE)
    # 遍历前被排除的内核：**表尾**必须逐条点名（内核 + 被去掉的组数 + 内核原文）。
    # 它们不在上面的表格里（那 N 是裁剪后实际遍历的组数），所以不写这一段，"为什么
    # 少了 8 组"就永远查不到。
    for item in tune.get("skipped_kernels") or []:
        lines.append("skipped-kernels: kernel_force={}  组数={}  原因={}".format(
            item.get("kernel_force") or "-", int(item.get("combos") or 0),
            str(item.get("reason") or "-").replace("\n", " ")))
    return lines


# ── 单次超时与整轮预算（--tune-timeout / --tune-budget） ─────────────────────
class _TuneTimeout(BaseException):
    """某一次 ``contract()`` 调用超过了 ``--tune-timeout``。

    刻意继承 ``BaseException`` 而不是 ``Exception``：库里有若干 ``except Exception``
    的"丢掉这次尝试、回退再跑一遍"（例如计划缓存的 memo 失效分支）。若把超时降级成
    普通异常，那些分支会把它吞掉再跑一遍，超时形同虚设。``KeyboardInterrupt`` 也是
    同一个理由。
    """


class _TuneBudget(BaseException):
    """整轮 ``--tune-budget`` 用尽（同样必须穿透库内的 ``except Exception``）。"""


class _TunePruned(BaseException):
    """这一组的单次耗时超过了当前 ``min_wall_ms``：**终止该组**，继续下一组。

    同样继承 ``BaseException``：剪枝必须穿透库内那些 ``except Exception`` 的
    "丢掉这次尝试、回退再跑一遍"，否则被终止的那一组会以通用路径重跑一遍，
    "终止"就成了空话。它带两个数值，供 ``原因=`` 原样打出（不留一个只能靠猜的标记）。
    """

    def __init__(self, wall_ms: float, min_wall_ms: float | None) -> None:
        super().__init__(prune_reason(wall_ms, min_wall_ms))
        self.wall_ms = float(wall_ms)
        self.min_wall_ms = None if min_wall_ms is None else float(min_wall_ms)


def _measure(call, limit: float | None, deadline: float | None,
             min_wall: Any) -> None:
    """跑一次 ``call()``，并把这**一次**的耗时与当前 ``min_wall_ms`` 比较。

    单次上限（``--tune-timeout``）与整轮预算（``--tune-budget``）由
    :func:`_guarded_call` 负责；这里只加一条用户规则：这一次调用比当前上界还慢
    → 抛 :class:`_TunePruned`，**终止该组**（不再跑完它的 warmup/rounds）。

    为什么按**单次**剪枝却按**中位**抬高上界：规则原文两句说的就是这两件事——
    "每一组的单次 contract() 调用耗时与 min_wall_ms 比较，超过即终止该组"，而
    "某组某次调用得到更小 wall_ms → 立即更新"。上界取该组的实测中位（= 报表与配置表
    里的 ``wall_ms``），于是"最终 best"与"记录的最小 wall_ms"是同一个数，不会出现
    "best 说 A、上界说 B"的两套账。
    """
    started = time.perf_counter()
    _guarded_call(call, limit, deadline)
    wall_ms = (time.perf_counter() - started) * 1000.0
    if min_wall is not None and min_wall.exceeds(wall_ms):
        raise _TunePruned(wall_ms, min_wall.ms)


def _contract_optimize(optimize: str) -> Any:
    """清单标签 → ``contract(optimize=...)`` 的取值。

    ``none`` 在**配置表**里是稳定写法（读侧按它比对），但库要的是 ``False``
    （``"none"`` 是 ``opt_einsum`` 的路径名，不是"不做收缩序优化"）。这两件事必须
    分开：清单与报表用字符串，调用用布尔。
    """
    text = str(optimize).strip().lower()
    return False if text == "none" else text


def _tune_timer_available() -> bool:
    """能不能给单次调用装 ``SIGALRM`` 定时器（``setitimer`` 只在主线程可用）。"""
    return (hasattr(signal, "setitimer") and hasattr(signal, "SIGALRM")
            and threading.current_thread() is threading.main_thread())


class _AlarmWindow:
    """一次受保护调用的窗口：抛什么、以及这次调用**还在不在**窗口里。

    ``armed`` 在收尾时先置 ``False``，之后才换回上一层窗口——这样"信号已生成、配送
    落在收尾之后"的那一次会被安静地忽略，而不是打到下一次调用头上。注意**不能**拿
    ``getitimer()`` 当判据：一次性定时器触发后剩余时间本来就归零，用它会连真正该抛
    的那一次也吞掉（实测过：0.5 ms 上限、300 次调用一次都没抛）。
    """

    __slots__ = ("error", "message", "armed")

    def __init__(self, error: type[BaseException], message: str) -> None:
        self.error = error
        self.message = message
        self.armed = True


#: 当前窗口；``None`` = 没有调用在受保护。处理器读它而不是闭包捕获——这样一个
#: 处理器就能服务所有调用，不必每次换（换处理器会留出"信号已在路上、处理器却已
#: 恢复默认"的窗口，而 ``SIGALRM`` 的默认动作是终止进程）。
_ALARM_WINDOW: _AlarmWindow | None = None


def _alarm_handler(_signum, _frame):
    """``SIGALRM`` 的唯一处理器：窗口内抛对应异常，其余一律**忽略**。

    忽略而不是回落到默认动作，是因为迟到的一次闹钟不该杀掉进程（``SIGALRM`` 的默认
    动作就是终止）。处理器装上就不摘，代价见 :func:`_call_alarm`。
    """
    window = _ALARM_WINDOW
    if window is None or not window.armed:
        return
    raise window.error(window.message)


@contextlib.contextmanager
def _call_alarm(seconds: float, error: type[BaseException]):
    """给**一次**调用装 ``SIGALRM`` 上限，退出时无条件解除定时器。

    解除是硬要求：一次超时之后本进程还要继续跑剩下的组合（以及调用方自己的代码），
    残留的 itimer 会在几十秒后的某个无关调用里抛异常，那时没人能把它归因到这里。

    处理器**装一次就不再摘**（见 :func:`_alarm_handler`）：每次调用都换回旧处理器会
    留下一段"信号已在路上、处理器却已恢复默认"的窗口，而 ``SIGALRM`` 的默认动作是
    终止进程。代价是 tune 之后本进程里没有受保护调用时 ``SIGALRM`` 被忽略而不是
    终止——对一个自己装闹钟的宿主程序来说，这个差异是刻意选择的（宁可忽略一次，
    也不要莫名其妙地被杀掉）。
    """
    global _ALARM_WINDOW
    message = (f"单次 contract() 超过 --tune-timeout {seconds:g}s"
               if error is _TuneTimeout
               else f"整轮超过 --tune-budget（本次调用被截断于 {seconds:g}s）")
    if signal.getsignal(signal.SIGALRM) is not _alarm_handler:
        signal.signal(signal.SIGALRM, _alarm_handler)
    window = _AlarmWindow(error, message)
    previous_window = _ALARM_WINDOW
    _ALARM_WINDOW = window
    interval, remaining = signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        try:
            if remaining > 0.0:                 # 调用前就存在的定时器原样还回去
                signal.setitimer(signal.ITIMER_REAL, interval, remaining)
        finally:
            # 关窗口这一句就是"迟到的信号算不算数"的分界；放进内层 finally，
            # 中途被信号打断也不会让一次超时永久留在模块状态里。
            window.armed = False
            _ALARM_WINDOW = previous_window


def _guarded_call(call, limit: float | None, deadline: float | None) -> None:
    """跑一次 ``call()``；超过 ``limit`` 秒（单次上限）或整轮 ``deadline`` 即中止。

    这是**尽力而为（best-effort）上限**，不是硬超时：装得上定时器、且调用一直在
    Python 里跑，就**当场中断**；调用落进原生 C/CUDA 代码时信号派发不进去，只能等它
    返回再判超时；装不上定时器（非主线程 / 无 ``SIGALRM``）则退化成"先跑完、再按耗时
    判定"——仍然如实判超时，只是没能省下那段墙钟。预算比单次上限更紧时按预算截断，
    并抛 :class:`_TuneBudget`（它说明的是"整轮没时间了"，不是"这一组太慢"）。
    """
    seconds = limit
    error: type[BaseException] = _TuneTimeout
    if deadline is not None:
        left = deadline - time.monotonic()
        if left <= 0.0:
            raise _TuneBudget("整轮 --tune-budget 已用尽")
        if seconds is None or left < seconds:
            seconds, error = left, _TuneBudget
    if seconds is None:
        call()
        return
    # 上限**不是硬超时**：SIGALRM 只在 Python 字节码之间派发，一旦调用进入原生
    # C/CUDA 代码（不返回解释器）就永远不触发——实测 10 算子方程的 `--tune` 会
    # 在 CPU 100%、GPU 0% 状态下挂 3 分钟以上，`--tune-timeout` 完全没生效。
    # 子进程隔离（fork）试过并**已回退**：CUDA 上下文不跨 fork 复用，GPU 上必然
    # ``cudaErrorInitializationError``（实测 22 组全 error）。因此这里仍是 SIGALRM——
    # 它能打断 Python 长循环，但打不断原生 C/CUDA 代码，这一点在 --help 与
    # ``tune note :`` 行里如实写明。
    #
    # 分档只看"能否回到解释器"，不看"耗时长不长"（本机 2026-11-19 实测）：
    #   纯 Python 5 秒循环、上限 0.3 s       → 0.30 s 抛，准；
    #   单次 5000³ 原生 BLAS、上限 0.3 s     → 2.29 s 才抛（该次调用本身 2.34 s），
    #                                          即**超限 1.99 s**。
    # 所以"上限"这个词对外只能写 best-effort；写成硬超时会把上面第二种情况说成保证。
    # 另注意本函数在 tune 里收到的是**组级**上限（``limit × (1+warmup+rounds)``），
    # 见 ``run_tune`` 的 ``group_limit`` 与 ``docs/einsum_tune_timeout_20261119.md`` §4。
    if not _tune_timer_available():
        started = time.perf_counter()
        call()
        if time.perf_counter() - started > seconds:
            raise error(f"单次调用耗时超过上限 {seconds:g}s（未能中断，已跑完）")
        return
    with _call_alarm(seconds, error):
        call()


#: ``as_equivalence=true`` 胜出时写在**原方程**那一行上的备注（规格 §3.3 b 的冻结文案）。
#: 它是**写进配置表的数据**，不跟 ``--language`` 变：同一个键的备注在两种语言下必须是
#: 同一串文字，否则读侧还得按语言挑一行。
TUNE_EQUIVALENCE_NOTE = "走等价转换后的配置性能更佳"

#: ``tune write:`` 行里 ``reason=`` 的稳定取值（机器可读，**不翻译**）。
TUNE_NO_USABLE = "no-usable"          # 全部 error / timeout / MISMATCH，没有胜出者
TUNE_API_MISSING = "api-missing"      # _profile_cache.write_winner() 不存在
TUNE_PROFILE_OFF = "profile-off"      # --profile false
TUNE_WRITE_FAILED = "failed"          # write_winner() 抛异常


@contextlib.contextmanager
def _tune_cache_session():
    """``with _profile_cache.tune_session():`` —— 遍历期间配置表既不读也不写。

    规格 §3.1 的唯一实现方式：**读**会把这一组量成"历史命中后的耗时"（于是 auto 那几
    组与显式 dedicated/topology 不可比），**写**会给未被选中的组合留下垃圾行。

    ``tune_session`` 是 §7 的冻结接口，由 A 路（``_profile_cache`` 重写）提供。它尚未
    落地时这里退化成"空窗口"并让调用方知道（yield ``False``）——报告里如实写"缓存**没有**
    真正关闭"，绝不假装关过。
    """
    cache = _profile_module()
    factory = getattr(cache, "tune_session", None) if cache is not None else None
    if not callable(factory):
        yield False
        return
    try:
        window = factory()
    except Exception:                           # noqa: BLE001 - 开不出窗口就退化成空窗口
        yield False
        return
    with window:
        yield True


def _fallback_shape_text(operands: Sequence[Any]) -> str:
    """``format_shape`` 的本地等价实现（``x`` 连轴、``|`` 连操作数）。

    只在 ``_profile_cache.format_shape`` 缺席时用。形状是键的一部分，两种写法必须落到
    同一个键上（``[2,3]`` / ``2x3`` / ``(2, 3)`` 是同一把键），所以口径照抄库里的那份。
    """
    parts: list[str] = []
    for operand in operands:
        try:
            axes = list(operand.shape)
        except AttributeError:
            axes = list(operand)
        parts.append("x".join(str(int(axis)) for axis in axes))
    return "|".join(parts)


def _tune_key(cache, operands: Sequence[Any], *, equation: str, backend: str,
              device: str, dtype: str) -> dict[str, Any]:
    """复现 ``_api._profile_begin`` 建的那把键（键 = shape/dtype/device/equation/backend）。

    规格 §0：``host`` 不参与匹配。``dtype`` 与两个 device 列**只能从操作数对象取**
    （cupy 的 device 是设备编号、torch 是 ``cuda:0``，与命令行上的 ``gpu`` 不是一回事）；
    自己按命令行猜，就会写下一把读侧永远匹配不上的键——那比不写更坏。取不到时才退回
    命令行口径，并且形状仍然按库的规范化口径生成。
    """
    shape_text = ""
    formatter = getattr(cache, "format_shape", None) if cache is not None else None
    try:
        shape_text = (formatter([operand.shape for operand in operands])
                      if callable(formatter) else _fallback_shape_text(operands))
    except Exception:                           # noqa: BLE001 - 键构造不许影响主流程
        shape_text = _fallback_shape_text(operands)
    fallback_backend = (str(backend or DEFAULT_BACKEND).strip().lower() or "numpy")
    if fallback_backend in ("", "auto"):
        fallback_backend = "numpy" if device == "cpu" else ""
    key: dict[str, Any] = {
        "user_device": "cpu" if device == "cpu" else str(device),
        "backend": fallback_backend,
        "dtype": str(dtype),
        "equation": "".join(str(equation).split()),
        "shape": shape_text,
        "device": "cpu" if device == "cpu" else str(device),
    }
    try:
        from pyqcd.einsum import _api

        name = _api.parse_backend(operands, backend)
        device_text, user_device = _api._profile_devices(operands, name)
        key["backend"] = str(name)
        key["device"] = str(device_text)
        key["user_device"] = str(user_device)
        key["dtype"] = str(_api._profile_dtype(operands))
    except Exception:                           # noqa: BLE001 - 旧版库/怪后端：用命令行口径
        pass
    return key


def _tune_kernel_label(row: dict[str, Any]) -> str:
    """写进配置表的 ``kernel``：优先**实际跑成的**路线，其次请求的逻辑名。

    配置表里的 ``kernel`` 要能被 ``contract()`` 重放（``_api._profile_force_label`` 认得
    ``kupl_dedicated`` 这类具体名、``generic`` 与 ``backend``），所以先把路线过一遍那张
    映射表——``kupl_topology_v2`` 这类带版本后缀的名字自己映射不出来。连路线都取不到时
    才写下缺省 ``generic``（没有专用内核可记）。
    """
    route = str(row.get("route") or "").strip()
    requested = str(row.get("kernel_force") or "").strip()
    unusable = ("", "none", KERNEL_UNKNOWN, "unavailable")
    try:
        from pyqcd.einsum import _api

        replay = getattr(_api, "_profile_force_label", None)
    except ImportError:                         # pragma: no cover - 旧版库
        replay = None
    if callable(replay):
        for candidate in (route, requested):
            if candidate in unusable:
                continue
            try:
                mapped = replay(candidate)
            except Exception:                   # noqa: BLE001 - 映射失败按原值处理
                mapped = None
            if mapped:
                return str(mapped)
    if route not in unusable:
        return route
    return requested if requested not in unusable else DEFAULT_KERNEL


def _tune_config(row: dict[str, Any], *, backend: str) -> dict[str, Any]:
    """胜出组的**性能配置**（写进配置表的那几列）。

    ``wall_ms`` 是本次实测中**整次 ``contract()`` 的墙钟**中位（规格 §4 的口径：含
    规划、匹配、等价转换与缓存读写，那是必要成本），不是任何拆分出来的小数字。
    ``init_ms`` 不写：本入口不单独测它，凭报告里的 ``route`` 反推一个"看起来合理"的
    值就是编造。列名与 ``_profile_cache.PERFORMANCE_FIELDS`` 同源——写方按名字取值，
    这里换一个名字就等于把数字静默丢掉。
    """
    median = row.get("median_ms")
    value = float(median) if median is not None else 0.0
    return {
        "backend": str(backend or ""),
        "kernel": _tune_kernel_label(row),
        "optimize": str(row.get("optimize") or ""),
        "wall_ms": value,
    }


def _tune_equivalence_payload(equation: str) -> dict[str, Any] | None:
    """该方程的等价改写载荷；没有可复用的改写就 ``None``。

    只在 ``as_equivalence=true`` 胜出时才需要：胜出组跑的就是这条改写，把它写进配置表
    才能让下次直接按它改道。``rewritten=false``（identity / 查不到）一律 ``None``——
    §3.3 b 要求"另以原 equation 写一行等价信息"，没有信息可写时就不写第二行。

    返回的键**就是**规格 §3.3 b 允许写进那一行的全部内容：``target_equation`` 加
    ``operand_order`` / ``operand_permutations`` / ``output_permutation``。别的一概
    不带——多写的键虽然会被 ``write_winner`` 丢掉，但读者会以为那行还带别的信息。
    """
    try:
        from pyqcd.einsum import _equivalence

        explained = _equivalence.explain(equation)
    except Exception:                           # noqa: BLE001 - 报告层不许影响主流程
        return None
    if not isinstance(explained, dict) or not explained.get("rewritten"):
        return None
    target = str(explained.get("equation") or "").strip()
    if not target or "".join(target.split()) == "".join(str(equation).split()):
        return None
    return {
        "target_equation": target,
        "operand_order": explained.get("operand_order"),
        "operand_permutations": explained.get("operand_permutations"),
        "output_permutation": explained.get("output_permutation"),
    }


#: 配置表键字段的兜底顺序（``_profile_cache.KEY_FIELDS`` 缺席时用；与规格 §0 一致）。
_KEY_FIELD_ORDER = ("shape", "dtype", "device", "equation", "backend")


def _key_field_names(cache) -> tuple[str, ...]:
    """键字段名：问 ``_profile_cache.KEY_FIELDS``（列定义只有一份），缺席才用兜底。"""
    names = getattr(cache, "KEY_FIELDS", None) if cache is not None else None
    return tuple(str(name) for name in names) if names else _KEY_FIELD_ORDER


def _key_text(cache, key: dict[str, Any]) -> str:
    """键的逐字段呈现：``shape=…  dtype=…  device=…  equation=…  backend=…``。

    规格 §12.2 要的是"被删行的**键**"，所以五个键字段一个都不省。取值用 ``repr``
    让空白与空串可见（``shape`` 的 ``|``/``x`` 是库的规范写法），``user_device`` 这类
    非键列不进这一行——它不是键的一部分，混进来会让"哪一把键"变成两说。
    """
    return "  ".join(f"{name}={key.get(name, '')!r}"
                     for name in _key_field_names(cache))


def _profile_rows_for(cache, key: dict[str, Any],
                      notes: list[str] | None = None
                      ) -> list[dict[str, Any]] | None:
    """该键在配置表里**现在有哪些行**（报告"删除了哪些键"用）；判不出返回 ``None``。

    只走冻结读接口：``read_winner`` 精确匹配 → 那一行；同键多行时抛
    ``DuplicateProfileEntry(rows)`` → **全部**候选行（每行带 ``line``，那些正是这次写
    要删掉的行）；文件不存在时抛 ``ProfileMissing`` → 0 行，并把它的**完整提示**（含
    现成的 tune 命令）记进 ``notes``——规格 §12.3 要求缺文件警告逐条完整输出。
    **不碰文件格式**：列名与列序归 ``_profile_cache`` 所有，这里一个都不硬编码，
    接口改名也只是退化成 ``None``（报告写 unknown 而不是猜）。
    """
    reader = getattr(cache, "read_winner", None) if cache is not None else None
    if not callable(reader):
        return None
    duplicate = getattr(cache, "DuplicateProfileEntry", None)
    missing = getattr(cache, "ProfileMissing", None)
    try:
        row = reader(dict(key))
    except Exception as exc:                    # noqa: BLE001 - 数不出就不报数字
        if isinstance(duplicate, type) and isinstance(exc, duplicate):
            rows = getattr(exc, "rows", None)
            if rows is None and exc.args:
                rows = exc.args[0]
            try:
                return [dict(item) for item in rows]
            except TypeError:
                return None
        if isinstance(missing, type) and isinstance(exc, missing):
            if notes is not None:
                notes.append(str(exc))
            return []
        return None
    return [] if row is None else [dict(row)]


def _tune_write_one(cache, key: dict[str, Any], config: dict[str, Any],
                    equivalence: dict[str, Any] | None, note: str, kind: str,
                    report: dict[str, Any], target: str | None = None) -> bool:
    """写一行并记账：**先取同键旧行**（那些会被 ``write_winner`` 删掉），再写。

    旧行只能这样取：``write_winner`` 的冻结语义是"删同键所有行后只写 1 行"，它自己
    不回报删了哪几条。这里把**每一被删行的键与识别列**都记下来（规格 §12.2 要求
    "删除的行数与被删行的键"），判不出（接口缺席/异常类型不符）时记 ``unknown``，
    **不猜数字**；同键多行（写前就 >1 行）另记一笔，报告里明说做过清理。
    """
    writer = getattr(cache, "write_winner", None) if cache is not None else None
    existing = _profile_rows_for(cache, key, report["notes"])
    try:
        writer(dict(key), dict(config), equivalence=equivalence, note=note)
    except Exception as exc:                    # noqa: BLE001 - 写失败只打印，不退出
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        return False
    report["rows"].append({
        "kind": kind,
        "key": {name: key.get(name) for name in _key_field_names(cache)},
        "equation": key.get("equation"),
        "target": target,
        "kernel": config.get("kernel"),
        "optimize": config.get("optimize"),
        "wall_ms": config.get("wall_ms"),
        "equivalence": dict(equivalence) if equivalence else None,
        "note": note,
    })
    if existing is None:
        report["unknown"] += 1
    else:
        report["deleted"] += len(existing)
        report["deletions"].append({
            "equation": key.get("equation"),
            "key": {name: key.get(name) for name in _key_field_names(cache)},
            "deleted": len(existing),
            # 被删的**每一行**（同键多行时唯一能区分它们的就是 line 与配置列）。
            "rows": existing,
        })
        if len(existing) > 1:                   # 同键多行 → 这次写做了清理
            report["duplicates"].append({
                "key": {name: key.get(name) for name in _key_field_names(cache)},
                "equation": key.get("equation"),
                "rows": len(existing),
            })
    return True


def _tune_persist(*, best: dict[str, Any] | None, equation: str,
                  operands: Sequence[Any], backend: str, device: str, dtype: str,
                  profile: bool | None) -> dict[str, Any]:
    """按规格 §3.3 把**胜出**的那一组写进配置表，返回可打印的报告字典。

    三种情形逐字对应规格：

    * 胜出组来自 ``as_equivalence=false`` → 以**原方程**写一条完整性能配置；
    * 胜出组来自 ``as_equivalence=true`` → 以 ``target_equation`` 为方程写完整性能配置，
      并且**另**以原方程写一条只含等价载荷（``operand_order`` /
      ``operand_permutations`` / ``output_permutation``）+ 备注的记录；
    * 没有可用组合（全部 ``error`` / ``timeout`` / ``MISMATCH``）→ **什么都不写**。

    §3.3 的"删除同键其它所有行"由 ``write_winner`` 的冻结语义完成，这里只负责把**真的**
    删掉了几行数出来（:func:`_profile_rows_for`）并如实回报。
    """
    report: dict[str, Any] = {
        "rows": [], "deleted": 0, "deletions": [], "unknown": 0,
        # 同键多行的清理记录：写前 >1 行的键各记一笔（§12.2 要"是否做过清理"）。
        "duplicates": [],
        "errors": [], "notes": [], "reason": None, "target": None,
        "session": None,
    }
    cache = _profile_module()
    if best is None:
        report["reason"] = TUNE_NO_USABLE
        return report
    # 判据用**本次生效值**而不是命令行字面量：``--profile false`` 与
    # ``PYQCD_EINSUM_PROFILE=0`` 都会让 ``_profile_cache.write_winner`` 静默不写，
    # 若只看 ``profile is False``，账上会记"写了 1 行"而表里一个字都没变。
    if _profile_effective(profile) is False:
        report["reason"] = TUNE_PROFILE_OFF
        return report
    writer = getattr(cache, "write_winner", None) if cache is not None else None
    if not callable(writer):
        report["reason"] = TUNE_API_MISSING
        return report

    key = _tune_key(cache, operands, equation=equation, backend=backend,
                    device=device, dtype=dtype)
    config = _tune_config(best, backend=str(key.get("backend") or backend))
    as_equiv = str(best.get("as_equivalence") or "false").strip().lower() == "true"
    payload = _tune_equivalence_payload(equation) if as_equiv else None
    if payload is None:
        if as_equiv:
            report["notes"].append(
                "as_equivalence=true won but the equation has no reusable rewrite; "
                "the plain configuration was written under the original equation")
        _tune_write_one(cache, key, config, None, "", "config", report)
    else:
        target = str(payload["target_equation"])
        report["target"] = target
        target_key = dict(key)
        target_key["equation"] = "".join(target.split())
        _tune_write_one(cache, target_key, config, None, "", "config", report,
                        target=target)
        _tune_write_one(cache, key, {}, payload, TUNE_EQUIVALENCE_NOTE,
                        "equivalence", report, target=target)
    if not report["rows"] and report["errors"]:
        report["reason"] = TUNE_WRITE_FAILED
    return report


def _profile_effective(requested: bool | None) -> bool | None:
    """本次调用画像是否真的生效：``False`` 一律不生效，其余看库的有效可用性。

    与 :func:`cache_line` 同一条原则——报**本次生效值**，不报 ``summary()['enabled']``
    那个环境变量口径。否则 ``--profile false`` 会印出 ``enabled=true``（表里有行、
    环境变量也开着），与本次"既不读也不写"自相矛盾。
    """
    if requested is False:
        return False
    cache = _profile_module()
    probe = getattr(cache, "enabled", None) if cache is not None else None
    if not callable(probe):
        return None
    try:
        return bool(probe())
    except Exception:                           # noqa: BLE001 - 报告层不许影响主流程
        return None


def _dispatched_config(operands: Sequence[Any], *, equation: str, backend: str,
                       device: str, dtype: str) -> dict[str, str]:
    """本次遍历**实际下发**的 ``backend`` / ``dtype`` / ``device``（规格 §13.2 的报表列）。

    取值问 :func:`_tune_key`——写配置表用的就是这把键，所以 ``--backend auto`` 报的是
    真正跑起来的后端（``numpy`` 等）、``--device gpu`` 报的是库认到的设备型号，而不是
    命令行上的字面量。三者对整轮遍历都是**不变量**，但它们必须与"哪一组跑了什么"
    印在同一行里：读者据此判断"这次 dedicated 跑不起来是不是因为后端本来就不支持"。
    """
    key = _tune_key(_profile_module(), operands, equation=equation,
                    backend=backend, device=device, dtype=dtype)
    return {
        "backend": str(key.get("backend") or backend or "-"),
        "dtype": str(key.get("dtype") or dtype or "-"),
        "device": str(key.get("device") or device or "-"),
    }


def run_tune(*, equation: str, shapes: Sequence[tuple[int, ...]],
             device: str = DEFAULT_DEVICE, dtype: str = DEFAULT_DTYPE,
             backend: str = DEFAULT_BACKEND, atol: float = DEFAULT_ATOL,
             rtol: float = DEFAULT_RTOL, rounds: int = DEFAULT_ROUNDS,
             warmup: int = DEFAULT_WARMUP, seed: int = 0,
             specialize: bool | None = None, cache: bool | None = None,
             profile: bool | None = None,
             profile_kwargs: dict[str, Any] | None = None,
             timeout: float = DEFAULT_TUNE_TIMEOUT,
             budget: float = DEFAULT_TUNE_BUDGET,
             compare: str = "none",
             mode: bool | str | None = TUNE_MODE_ALL,
             isolate: bool | None = None,
             language: str | None = None) -> dict[str, Any]:
    """遍历网格并返回 ``{"rows", "best", "write", ...}``（表格由 :func:`print_tune` 打）。

    ``mode`` 是 ``--tune`` 的取值（:func:`parse_tune_mode`）：``all``（缺省）跑完整
    模式，``pass`` 跑**快速通过模式**——分支直接钉成 ``B3``（21 组、无 ``topology``）、
    ``as_equivalence`` 全 ``false``、不探测准入、不算参照、不判 MISMATCH，并静音专用
    路径提示。``off`` 传到这里没有意义（调用方不该调用），按 ``all`` 处理而不是静默
    什么都不做。

    ``isolate`` 决定**这一组跑在哪里**：``None``（缺省）跟随
    ``PYQCD_EINSUM_TUNE_INPROCESS``——未设时每组跑在**全新解释器**里，超时由父进程
    kill 子进程强制生效，这是唯一能打断原生 C/CUDA 代码的办法（``SIGALRM`` 只在
    字节码之间派发，实测超限 1.99 s 才抛）；显式 ``True``/``False`` 压过环境变量，
    ``False`` 是进程内 ``SIGALRM`` 路线（打不断原生代码，只作对照实验与契约注入）。

    子进程路线的两条**测量口径**必须记住：
    * ``wall_ms`` 只装受测时间区（``1+warmup+rounds`` 次 ``contract()``），**不含**
      每组一次的解释器启动与操作数载入，那部分单独累计进 ``isolation.startup``；
    * 操作数按固定 ``seed`` 生成**一次**、落盘一次（``isolation.stage``），全部组共用
      ——每组各落一份在生产形状上就是 21 × 3.8 GiB。

    初始化阶段（遍历开始之前）逐行打印 ``tune init :``：mode 与规模、分支探测**起止**、
    内核准入**起止**、session **起止**、首组**起止**。这些行**机器可读**、不跟
    ``--language`` 翻译，耗时取整毫秒——卡住时直接看得出停在哪一步（2026-11-19 用户
    要求"方便判断进度与健康性"）。

    **不变量**：方程、形状、dtype、设备、后端都只在这里取一次；每组只换
    ``kernel_force`` / ``as_equivalence`` / ``optimize``。``compare`` 不影响遍历的
    计时——参照要么在调用方那一侧跑一次（规格 §13.1：``--tune`` 下缺省关闭、
    显式给出才执行），要么根本不跑；这里只把它记进报告，好让输出不说假话。

    每组直接调 ``pe.contract()``（不走 :func:`run`）：``run`` 每组都会再算一遍参照与
    等价报告，30 组就是 30 份无用功；而这一层要的只是"这一组配置跑一次要多久"。

    整轮遍历包在 ``_profile_cache.tune_session()`` 里（规格 §3.1）：配置表**既不读也不
    写**。读会把这一组量成"历史命中后的耗时"，让 ``auto`` 与显式 ``dedicated`` 不可比；
    写会给 29 组没被选中的组合留下垃圾行。遍历结束后只写胜出的那一组
    （:func:`_tune_persist`，规格 §3.3）。

    遍历的是**有序清单**而不是笛卡尔积（:func:`tune_combos` + :mod:`._tune_axes`）：
    分支（原方程可特化 / 只有等价写法可特化 / 都不可特化）在**探测阶段**定下来，
    只探测、不计时、不写表；``topology`` 那一组只在分支允许时才出现。

    ``min_wall_ms`` 剪枝（用户规则）：初值 = ``timeout``（``--tune-timeout``，秒→毫秒），
    每一组的单次 ``contract()`` 耗时与它比较，**超过即终止该组**（原文
    ``耗时大于min_wall_ms,此组被终止``）+ 继续下一组 + 该组**不写缓存**；某组测到更小
    耗时则**立刻**把 ``min_wall_ms`` 降为该值（后续所有组都用新上界剪枝）。``best`` 就是
    记录的最小耗时那一组。

    ``timeout`` 是**每一次** ``contract()`` 调用（warmup 与每一轮都算）的上限秒数，
    ``budget`` 是整轮上限（``0`` = 不限）。任一次调用超时 → 该组记 ``timeout``；
    中位只由全部成功且未超时的调用构成（``_timed`` 一抛出来就没有中位，不拿部分样本凑）。
    预算用尽 → 立刻停止遍历，返回值里 ``stopped``/``measured`` 给出"已测 N/M"。

    任一组合抛异常（含强制内核的 ``KernelUnavailableError``）只在该行记
    ``error=<类型>`` 并继续——一次遍历不该因为一个不可用的内核整段作废。超时、异常与
    数值不符（``rel > atol+rtol``）三类组合**只打印**：既不改退出码，也一律不参与选优
    （:func:`tune_best`），因此永远不会被写进配置表。
    """
    global _ALARM_WINDOW

    import pyqcd.einsum as pe
    from pyqcd.einsum import _specialize

    lang = resolve_language(language)
    # 遍历期间库内的提示也必须与报告同语言（见 :func:`run` 处的说明）。
    _specialize.set_language(lang)
    if warmup < 1:
        raise UsageError(translate("error.warmup_zero", lang))
    if rounds < 1:
        raise UsageError(translate("error.rounds", lang))
    if float(timeout) < 0.0:
        raise UsageError(translate("error.tune_timeout", lang, value=timeout))
    if float(budget) < 0.0:
        raise UsageError(translate("error.tune_budget", lang, value=budget))
    # 模式解析放在**最前**（早于任何探测与张量构造）：非法取值必须立刻以退出码 2
    # 结束，而不是先跑一段再报错。
    tune_mode = parse_tune_mode(mode, lang)
    pass_mode = tune_mode == TUNE_MODE_PASS
    # ① mode 与本次不变量（方程 / 操作数个数 / dtype / 设备 / 后端）。
    _tune_init("mode={}  equation={}  operands={}  dtype={}  device={}  backend={}".format(
        tune_mode, _tune_init_equation(equation), len(shapes), dtype, device, backend))
    total_elements, est_mib = _tune_init_shapes(shapes)
    _tune_init("shapes ok  total_elements={}  est_bytes={}MiB".format(
        total_elements, int(round(est_mib))))
    # ② 分支探测**起止**。``pass`` 模式跳过整段探测（``_specialize.match`` 与
    # ``normalize_equation`` 在长方程上可能很慢——那正是 pass 要避免的），直接把
    # 分支钉成 ``B3``（既不改写、也不要求特化），清单因此天然没有 ``topology`` 组。
    probe_started = time.perf_counter()
    if pass_mode:
        _tune_init("branch probe start  skipped (pass mode)")
        branch: dict[str, Any] = {
            "branch": TUNE_BRANCH_NONE, "raw_ok": False, "equiv_ok": False,
            "target_equation": None}
        _tune_init("branch probe done  skipped (pass mode)  raw_ok=false  "
                   "equiv_ok=false  branch={}  {}".format(
                       TUNE_BRANCH_NONE, _tune_init_ms(probe_started)))
        combos = tune_combos(lang)
    else:
        _tune_init("branch probe start")
        branch = tune_branch(equation, lang)
        _tune_init("branch probe done  raw_ok={}  equiv_ok={}  branch={}  {}".format(
            str(bool(branch.get("raw_ok"))).lower(),
            str(bool(branch.get("equiv_ok"))).lower(),
            branch.get("branch") or "-", _tune_init_ms(probe_started)))
        combos = _tune_combos_for(equation, lang)  # 超上限在构造任何张量之前就报错
    np_dtype = np.dtype(dtype)
    # ③ 内核准入**起止**。``pass`` 模式整段跳过：清单里没有 ``topology`` 组，而
    # ``dedicated`` 的准入本身要规划一次步表——两者都属于"与快速通过无关"的成本。
    planned = len(combos)
    admit_started = time.perf_counter()
    if pass_mode:
        _tune_init("kernel admission start  skipped (pass mode)")
        unavailable: dict[str, str] = {}
        _tune_init("kernel admission done  skipped (pass mode)  unavailable=-  "
                   "{}".format(_tune_init_ms(admit_started)))
    else:
        _tune_init("kernel admission start")
        # 规格 §18「先探测再裁剪」：该内核对这个形状**本质上没有路**时（计划 arena 超
        # 预算、方程不可特化……），整组跳过并在报告里写成 ``skip``——逐组跑一遍只会得到
        # 同一句错误的 N 份副本，把真正的异常淹掉（2026-11-18 用户实测）。
        unavailable = tune_unavailable_kernels(
            equation=equation, shapes=shapes, dtype=np_dtype, backend=backend,
            device=device, branch=branch)
        _tune_init("kernel admission done  unavailable={}  {}".format(
            ",".join(sorted(unavailable)) or "-", _tune_init_ms(admit_started)))
    combos, dropped = _tune_axes.plan_without(combos, unavailable)
    skipped_rows: list[dict[str, Any]] = [
        {"kernel_force": kernel, "skip": True, "status": TUNE_STATUS_SKIP,
         "combos": int(count),
         "reason": f"skip: {kernel} 不可用（{unavailable.get(kernel) or '未知原因'}）"}
        for kernel, count in dropped.items()]
    xp = _xp_for_device(device, lang)
    # 操作数只构造一次：每组看到的必须是同一批数据，否则"哪组更快"混进了数据差异。
    operands, host = _make_operands(xp, shapes, np_dtype, seed)
    # 实际下发的三列（§13.2）：整轮不变，但每组那行都要带着印出来。
    # ``pass`` 模式跳过这把键的额外探测（``_tune_key`` 要跑 ``_api.parse_backend`` /
    # ``_profile_devices``）：直接报命令行上的不变量，不为了三列诊断多花时间。
    dispatched = ({"backend": str(backend or "-"), "dtype": str(dtype or "-"),
                   "device": str(device or "-")} if pass_mode
                  else _dispatched_config(operands, equation=equation, backend=backend,
                                          device=device, dtype=dtype))
    # 参照**总是尝试**，但由 :data:`TUNE_REFERENCE_MAX_ELEMENTS` 挡住大形状：
    # 小形状照旧做真实 ``np.einsum``（数值校验保留），大形状跳过（那一次完整收缩
    # 可能几分钟且不可中断——2026-11-18 用户实测"tune 卡死无输出"的根因）。
    # ``--compare`` 只影响**普通调用**要不要另跑参照基准，不是 tune 的数值校验开关。
    # ``pass`` 模式**完全不算参照**：那是 pass 的语义（跳过数值校验），不是省事。
    if pass_mode:
        reference, reference_note = None, "skipped (pass mode)"
    else:
        reference, reference_note = _tune_reference(equation, host)
    extra = dict(profile_kwargs or {})
    tolerance = float(atol) + float(rtol)
    # ``limit`` = 每一次 contract() 调用的上限；``deadline`` = 整轮的上限。
    # ``0``/负数都是"不设上限"（CLI 已校验非负，这里的 0 是合法取值）。
    limit = float(timeout) if float(timeout) > 0.0 else None
    deadline = (time.monotonic() + float(budget)) if float(budget) > 0.0 else None
    # 组级上限仍是 ``limit × (1+warmup+rounds)``：它与 ``min_wall_ms`` 同口径（都看整组），
    # 换成子进程之后这条不变量不变，只是"到点"由父进程 kill 而不是等 SIGALRM 派发。
    group_limit = (None if limit is None
                   else float(limit) * float(1 + warmup + rounds))
    # 隔离方式：缺省**子进程**（能打断原生 C/CUDA），``PYQCD_EINSUM_TUNE_INPROCESS=1``
    # 或显式 ``isolate=False`` 才退回进程内 SIGALRM（打不断原生代码，只作对照实验）。
    isolated = (_tune_runner.isolation_enabled() if isolate is None
                else bool(isolate))
    # ``min_wall_ms`` 初值 = ``--tune-timeout``（秒 → 毫秒）；``0`` = 不设上界。
    min_wall = _tune_axes.MinWall(
        ms=None if limit is None else limit * 1000.0,
        source="--tune-timeout")
    # 预判剪枝的吞吐估计：只由**成功且未超时**的组更新（与 min_wall 同口径）。
    rate = _tune_axes.Rate()
    startup_s: list[float] = []
    staged: dict[str, Any] = {}
    stage_root: str | None = None
    rows: list[dict[str, Any]] = []
    measured = 0
    stopped = False
    # tune_session 是否真的开起来了（A 路接口缺席时是 False，报告里如实写）。
    session = False
    # ``pass`` 模式静音库内提示（``_specialize.announce`` / ``announce_generic``）与
    # 完整详解：这两者是"过程解说"，与"哪一组多快"无关，正是 pass 要关掉的东西。
    # 只在本次调用期间生效，退出时**原样还回去**（不改调用方的进程状态）。
    notice_saved: str | None = None
    detail_saved: Any = None
    set_detail = getattr(_specialize, "set_detail", None)
    if pass_mode:
        notice_saved = os.environ.get(_specialize.NOTICE_ENV)
        os.environ[_specialize.NOTICE_ENV] = "0"
        if callable(set_detail):
            detail_saved = set_detail(False)
    _tune_init("session start  profile_disabled=true")
    first_combo = combos[0] if combos else None
    first_started = time.perf_counter()
    first_done = False
    if isolated:
        _tune_init("isolation start  mode=subprocess  profile_disabled=true  "
                   "groups={}".format(len(combos)))
    try:
        # 规格 §3.1：遍历期间配置表**既不读也不写**。写侧由这里的 tune_session 负责，
        # 读侧也一并关掉——否则 ``auto`` 那几组会被历史命中劫持，量到的是"命中耗时"。
        with _tune_cache_session() as session:
            if first_combo is None:
                _tune_init("first group start  -  (no combination to run)")
            else:
                _tune_init("first group start  {}/{}/{}".format(*first_combo))
            for kernel, as_equiv, optimize in combos:
                if deadline is not None and time.monotonic() >= deadline:
                    stopped = True
                    break
                row: dict[str, Any] = {
                    "kernel_force": kernel,
                    "as_equivalence": as_equiv,
                    "optimize": optimize,
                    # §13.2：该组**实际下发**的 backend / dtype / device。
                    "backend": dispatched["backend"],
                    "dtype": dispatched["dtype"],
                    "device": dispatched["device"],
                    "median_ms": None,
                    # ``wall_ms`` 是同一数值的**对外**名字（规格 §8/§12.1 的列名）：
                    # 内部键 ``median_ms`` 另有既有读侧在用，两个都留着不丢信息。
                    "wall_ms": None,
                    "route": KERNEL_UNKNOWN,
                    "rel": None,
                    "status": None,
                    # 原因**原文**：成功组为 ``None``；坏组原样保留异常正文（不截断）。
                    "reason": None,
                }
                holder: list[Any] = []

                def call(kernel=kernel, as_equiv=as_equiv, optimize=optimize,
                         holder=holder):
                    got = pe.contract(equation, *operands, backend=backend,
                                      optimize=_contract_optimize(optimize),
                                      specialize=specialize,
                                      kernel_force=kernel,
                                      as_equivalence=(as_equiv == "true"),
                                      cache=cache, **extra)
                    # 只留**最近一次**结果：append 会把 warmup+rounds 份中间结果全留活，
                    # 生产尺寸上那是把峰值内存乘上轮数。
                    holder[:] = [got]
                    return got

                _clear_kernel_info()
                _reset_last_kernel()
                bad = False
                seconds: float | None = None
                observation: dict[str, Any] | None = None
                # 本组规划出来的运算量（只在预判剪枝与吞吐记账时用）。
                plan = (_plan_probe(equation, shapes, optimize) if isolated
                        else {"flops": None, "seconds": None,
                              "limit_hit": False, "error": None})
                group_flops = plan.get("flops")
                # 预判剪枝：这一组**不跑**，依据全部来自规划器，两条判据都写进 ``原因``。
                # ① 规划出来的 FLOPs 按已测吞吐折算，明显超过当前上界（**估算**）；
                # ② 规划**自己**就超过了当前上界（**实测**：规划是那次调用的一部分）。
                # 针对的正是本例那个 ``optimize=none``（单次 1e17 次浮点）与
                # ``optimal``（规划 38 s）：硬超时也能挡住，但要先白等整组额度。
                skip_reason = ""
                if isolated:
                    limit_hit = bool(plan.get("limit_hit"))
                    should_skip, estimate_reason = rate.should_skip(
                        group_flops, min_wall.ms)
                    plan_ms = plan.get("seconds")
                    slower_plan = (limit_hit or (
                        plan_ms is not None and min_wall.ms is not None
                        and float(plan_ms) * 1000.0 > float(min_wall.ms)))
                    if slower_plan:
                        skip_reason = (
                            "planning alone exceeded min_wall_ms  plan_ms={}{}  "
                            "min_wall_ms={:.6f}  (planned, not contracted)".format(
                                "-" if plan_ms is None else f"{float(plan_ms) * 1000.0:.6f}",
                                " (>probe budget)" if limit_hit else "",
                                float(min_wall.ms)))
                    elif should_skip:
                        skip_reason = estimate_reason
                    if skip_reason:
                        row["predicted"] = True
                        row["reason"] = skip_reason
                        row["status"] = TUNE_STATUS_PRUNED
                        rows.append(row)
                        if not first_done:
                            first_done = True
                            _tune_init("first group done  {}/{}/{}  状态={}  {}".format(
                                kernel, as_equiv, optimize, row["status"],
                                _tune_init_ms(first_started)))
                        continue
                try:
                    if isolated:
                        # **子进程**路线（缺省）：父进程 kill 子进程才能打断原生
                        # C/CUDA 代码。组级上限与 min_wall 都在子进程里生效
                        # （后者必须能在两次调用之间判定，只有子进程做得到）。
                        started = time.perf_counter()
                        if stage_root is None:      # 操作数只落盘一次，21 组共用
                            stage_root = _tune_runner.stage_dir()
                            staged = _tune_runner.stage_operands(host, stage_root)
                            _tune_init("stage done  files={}  bytes={}  {}".format(
                                int(staged["count"]), int(staged["bytes"]),
                                _tune_init_ms(started)))
                        group_dir = os.path.join(
                            stage_root, f"group_{len(rows):03d}")
                        output_path = os.path.join(group_dir, "output.npy")
                        # 数值校验仍然保留：小形状（有真参照）时让子进程把结果落回来，
                        # 父进程照旧算 ``rel``。大形状本来就没有参照，不为它多写一次盘。
                        want_output = (not pass_mode) and reference is not None
                        observation = _tune_runner.run_group(
                            {
                                "equation": equation,
                                "device": device,
                                "backend": backend,
                                "dtype": dtype,
                                "warmup": warmup,
                                "rounds": rounds,
                                "min_wall_ms": min_wall.ms,
                                "operands_dir": staged["dir"],
                                "kernel_force": kernel,
                                "as_equivalence": (as_equiv == "true"),
                                # ``none`` → ``False``：清单用字符串，调用用布尔。
                                "optimize": _contract_optimize(optimize),
                                "contract_kwargs": {
                                    "specialize": specialize,
                                    "cache": cache,
                                },
                                "profile_kwargs": dict(extra),
                                # 子进程里的 ``tune_session`` 不生效（``_TUNING`` 是进程内
                                # 计数器），必须显式让每一组都 ``profile=False``，
                                # 否则读表会把命中行的三轴盖到被测格子上，并抛出
                                # ``ProfileConflict``（2026-11-19 用户实测 9 组 error）。
                                "profile_off": True,
                                "output_operand": output_path if want_output else None,
                                "output_max_bytes": _tune_runner.OUTPUT_MAX_BYTES,
                            },
                            workdir=group_dir,
                            group_limit_s=group_limit,
                            deadline=deadline)
                        status = str(observation.get("status") or "ok")
                        if observation.get("startup_s") is not None:
                            startup_s.append(float(observation["startup_s"]))
                        if status == "ok":
                            seconds = float(observation["median_s"])
                            # 子进程的选路记录随结果文件回来：``holder`` 里放它，
                            # 下面 ``_tune_route`` 与报告都从这一处取，不再读本进程
                            # 的 ``LAST_INFO``（那是父进程自己的，本组根本没在这里跑）。
                            holder[:] = [observation.get("last_kernel")]
                            if (want_output and observation.get("saved_output")
                                    and os.path.exists(output_path)):
                                row["_output"] = output_path
                            if observation.get("output_skipped_bytes"):
                                row["output_skipped_bytes"] = int(
                                    observation["output_skipped_bytes"])
                        elif status == "pruned":
                            row["pruned"] = True
                            row["wall_ms"] = round(
                                float(observation.get("pruned_wall_ms") or 0.0), 6)
                            row["reason"] = prune_reason(
                                row["wall_ms"], observation.get("pruned_min_wall_ms"))
                            bad = True
                        elif status == "timeout":
                            row["timeout"] = True
                            row["reason"] = str(observation.get("reason") or "timeout")
                            bad = True
                        elif status == "budget":
                            row["budget"] = True
                            row["reason"] = str(observation.get("reason") or "budget")
                            bad = True
                            stopped = True
                        else:
                            row["error"] = str(observation.get("error_type")
                                               or "TuneWorkerError")
                            row["reason"] = str(observation.get("reason")
                                                or observation.get("error") or "error")
                            bad = True
                    else:
                        # **进程内**路线（``PYQCD_EINSUM_TUNE_INPROCESS=1``）：老行为，
                        # SIGALRM 打不断原生 C/CUDA 代码，只作对照实验。
                        seconds = _timed(
                            lambda: _guarded_call(
                                lambda: _measure(call, None, deadline, min_wall),
                                group_limit, deadline),
                            warmup, rounds)
                except _TunePruned as exc:
                    # 用户规则 §4：超过当前 ``min_wall_ms`` 的这一组**终止**、继续下一组，
                    # 并且不写缓存。原文与两个数值都进 ``reason``，一字不改地打出来。
                    row["pruned"] = True
                    row["wall_ms"] = round(float(exc.wall_ms), 6)
                    row["reason"] = prune_reason(exc.wall_ms, exc.min_wall_ms)
                    bad = True
                except _TuneTimeout as exc:
                    # 超时只打印：遍历继续，这一组不参与选优、也不会被写进配置表。
                    # ``reason`` 存**原文**（类型 + 异常正文），报告里一字不改地打出来。
                    row["timeout"] = True
                    row["reason"] = f"{type(exc).__name__}: {exc}"
                    bad = True
                except _TuneBudget as exc:
                    row["budget"] = True
                    row["reason"] = f"{type(exc).__name__}: {exc}"
                    bad = True
                    stopped = True
                except Exception as exc:        # noqa: BLE001 - 单组失败不中断遍历
                    # ``error`` 只留**类型名**（机器可读的稳定 token，既有读侧按它比对），
                    # 完整正文进 ``reason``——两件事分开，谁也不吞掉谁。
                    row["error"] = type(exc).__name__
                    row["reason"] = f"{type(exc).__name__}: {exc}"
                    bad = True
                row["route"] = _tune_route(
                    kernel_force_fields(
                        (observation.get("last_kernel")
                         if observation and observation.get("status") == "ok"
                         else None) or _last_kernel(),
                        kernel, kernel_labels(kernel, lang)),
                    (observation.get("kernel_info") if observation else None)
                    or _kernel_info())
                if not bad and seconds is not None:
                    row["median_ms"] = round(seconds * 1000.0, 6)
                    row["wall_ms"] = row["median_ms"]
                    measured += 1
                    # 吞吐记账：供**后续组**的预判剪枝用。只用成功组（与 min_wall 同口径）。
                    if group_flops:
                        rate.observe(group_flops, seconds, (kernel, as_equiv, optimize))
                    # 参照：``compare`` 显式给出时用真实参照；``none``（``--tune``
                    # 缺省）时退化为**首个成功组合的结果**做自洽校验。后者成本为零，
                    # 却仍能检出"不同内核/规划器算出不同数"这一类真实分歧——否则
                    # 跳过参照会让 MISMATCH 永远检不出来（2026-11-18 实测：契约
                    # ``rel 超容差必须判 MISMATCH`` 变成恒假）。
                    #
                    # ``pass`` 模式整段不做：连"首个成功组当参照"也不做，否则
                    # ``rel`` 会有一个数而表尾同时写着"跳过数值校验"——两句话互相
                    # 矛盾，读者无从判断到底校验过没有。
                    #
                    # 子进程路线下结果在**文件**里（``row["_output"]``）；进程内路线下
                    # 它在 ``holder`` 里。两条路都归到同一个 ``got``，数值校验因此
                    # 不因换了执行位置而消失——那正是"升级超时机制"最容易踩的坑。
                    if not pass_mode:
                        cached_output = row.pop("_output", None)
                        got = None
                        if cached_output and os.path.exists(cached_output):
                            got = np.load(cached_output)
                        elif not isolated and holder:
                            got = _to_host(holder[-1])
                        if got is not None:
                            if reference is None:
                                reference = np.asarray(got)
                                reference_note = "self(first ok combination)"
                            row["rel"] = _relative_error(got, reference)
                            if not (row["rel"] <= tolerance):
                                row["mismatch"] = True
                                row["reason"] = (
                                    "rel={:.3e} > atol+rtol={:g} (atol={:g}, rtol={:g}, "
                                    "reference={})".format(
                                        row["rel"], tolerance, float(atol), float(rtol),
                                        reference_note))
                                bad = True
                    # 用户规则 §4：这一组的中位比当前上界更小 → **立即**压低上界，
                    # 后续所有组都用它剪枝。只在"可用"的组上压低：数值不符/跑不成
                    # 的那些不该定义"要多快才算值得跑"。
                    if not bad and min_wall.observe(
                            row["median_ms"], (kernel, as_equiv, optimize)):
                        row["min_wall"] = True
                row["status"] = tune_status(row)
                holder.clear()                  # 别把上一组的结果留到下一组
                rows.append(row)
                # 首组**收尾**行：它与上面的 ``first group start`` 成对，是"遍历真的
                # 开始了、而且第一组已经有结论"的直接证据（卡在首组时不会有这一行）。
                if not first_done:
                    first_done = True
                    _tune_init("first group done  {}/{}/{}  状态={}  {}".format(
                        kernel, as_equiv, optimize, row["status"],
                        _tune_init_ms(first_started)))
                if stopped:                     # 预算用尽：不再开新的一组
                    break
    finally:
        # 闹钟窗口是模块级状态：单次调用若正好在收尾时被迟到的信号打断，那句"关窗口"
        # 可能少走一次。整轮结束时无条件清空，别把这个状态留给后面的人。
        _ALARM_WINDOW = None
        # 操作数暂存目录是**本次遍历**的临时物：不删会在 /tmp 里留下 3.8 GiB 的 .npy
        # （本机生产形状实测），下次 `--tune` 又会各留一份。删失败不抛（临时目录残留
        # 不该让一次成功的遍历变成失败）。
        if stage_root is not None:
            _tune_runner.cleanup(stage_root)
            stage_root = None
        # pass 模式静音的提示/详解**原样还回去**：这是本次调用的输出选择，不该改
        # 调用方进程的状态（同语言设置与缓存开关的口径）。
        if pass_mode:
            if notice_saved is None:
                os.environ.pop(_specialize.NOTICE_ENV, None)
            else:
                os.environ[_specialize.NOTICE_ENV] = notice_saved
            if callable(set_detail):
                set_detail(detail_saved)

    best = tune_best(rows)
    # 规格 §3.3：只写胜出的那一组（可能是一行配置，也可能是"target 配置 + 原键等价行"）。
    write = _tune_persist(best=best, equation=equation, operands=operands,
                          backend=backend, device=device, dtype=dtype,
                          profile=profile)
    write["session"] = session
    return {
        "rows": rows,
        "best": best,
        "write": write,
        "profile": _profile_summary(),
        "profile_requested": "auto" if profile is None else (
            "true" if profile else "false"),
        "profile_enabled": _profile_effective(profile),
        "reference": reference_note,
        "combos": len(combos),
        "measured": measured,
        "total": len(combos),
        "planned": planned,
        "trimmed": planned - len(combos),
        # 遍历前被排除的内核（每条一行：内核 + 被去掉的组数 + 内核自己的原文）。
        "skipped_kernels": skipped_rows,
        "stopped": stopped,
        "skipped": len(combos) - measured,
        "timeout": limit,
        # ``min_wall_ms`` 剪枝的记账：初值来源、当前上界、压低它的那一组。
        "min_wall_ms": min_wall.ms,
        "min_wall_source": min_wall.source,
        "min_wall_config": (None if min_wall.config is None
                            else list(min_wall.config)),
        "min_wall_ms_initial": None if limit is None else float(limit) * 1000.0,
        "budget": float(budget),
        # 超时机制的**实际执行位置**（机器可读）：``subprocess`` = 父进程 kill 子进程，
        # 能打断原生 C/CUDA；``sigalrm`` = 进程内尽力而为；``posthoc`` = 装不上定时器；
        # ``off`` = 没设上限。
        "timer": ("subprocess" if isolated else
                  ("sigalrm" if (limit is not None and _tune_timer_available())
                   else ("posthoc" if limit is not None else "off"))),
        # 子进程路线的**测量开销**（不混进 wall_ms）：每组一次解释器启动 + 导入 + 载入。
        "isolation": {
            "mode": "subprocess" if isolated else "inprocess",
            "startup": _tune_runner.summarize_startup(startup_s),
            "stage": {
                "files": int(staged.get("count") or 0),
                "bytes": int(staged.get("bytes") or 0),
                "seconds": round(float(staged.get("seconds") or 0.0), 6),
            } if staged else None,
        },
        # 预判剪枝的记账：命中了几组、当时的吞吐估计来自哪一组。
        "predicted": sum(1 for row in rows if row.get("predicted")),
        "rate_flops_per_s": rate.flops_per_s,
        "rate_source": rate.source,
        "warmup": warmup,
        "rounds": rounds,
        "compare": str(compare or "none"),
        # ``--tune`` 的取值：``all``（完整模式）或 ``pass``（快速通过模式）。表尾的
        # "跳过数值校验"注记由它决定，JSON 消费方也据此判断 rel 为什么整列是 ``-``。
        "mode": tune_mode,
        "pass_mode": bool(pass_mode),
        "skip_note": TUNE_PASS_SKIP_NOTE if pass_mode else None,
        # 本次遍历实际下发的三列（§13.2）：表格与 JSON 共用同一份取值。
        "dispatched": dict(dispatched),
        # 分支判定（只探测、不计时）：B1 原方程可特化 / B2 只有等价写法可特化 /
        # B3 都不可特化。它决定清单里有没有 ``topology`` 那一组。
        "branch": branch,
        "axes": {
            "kernel_force": list(TUNE_KERNEL_FORCE),
            "as_equivalence": list(TUNE_AS_EQUIVALENCE),
            "optimize": list(TUNE_OPTIMIZERS),
        },
    }


def _bold(text: str) -> str:
    """最优行的加粗标记：只在 stderr 是终端时加 ANSI，重定向到日志时保持纯文本。"""
    if not sys.stderr.isatty():
        return text
    return f"\033[1m{text}\033[0m"


def _tune_line(row: dict[str, Any], best: dict[str, Any] | None) -> str:
    """一行结果。``*`` 是**稳定**的最优标记（终端里另加粗），便于脚本筛。

    ``MISMATCH`` / ``timeout`` / ``error=<类型>`` 是三类"不参与选优"的稳定标号，
    与 ``tune hdr`` 里的字段名一样**不跟 --language 翻译**（脚本要能筛）。
    """
    marker = "*" if row is best else " "
    text = ("{} wall_ms={:>11s}  kernel_force={:<9s}  as_equivalence={:<5s}  "
            "optimize={:<7s}  rel={:>9s}  route={}").format(
        marker, _tune_ms(row), row["kernel_force"], row["as_equivalence"],
        row["optimize"], _tune_rel(row), row.get("route") or KERNEL_UNKNOWN)
    if row.get("mismatch"):
        text += "  MISMATCH"
    if row.get("pruned"):
        text += "  pruned"
    if row.get("timeout"):
        text += "  timeout"
    if row.get("budget"):
        text += "  budget"
    if row.get("error"):
        text += f"  error={row['error']}"
    return text


def _print_multiline(prefix: str, text: Any, out=None) -> None:
    """把一段**原文**逐行打印：首行带 ``prefix``，其余行原样续打（对齐前缀宽度）。

    §12 反复强调"原因原文必须完整打印"：异常正文里常有换行（原生库的多行报错、
    ``ProfileMissing`` 自带的 tune 命令模板），截断或折叠成"…等 N 项"正是要禁止的
    那种"看不见"。续行只加缩进、一个字都不改，复制粘贴回命令行仍然有效。
    """
    out = sys.stdout if out is None else out
    lines = str(text).splitlines() or [""]
    print(prefix + lines[0], file=out)
    for extra in lines[1:]:
        print(" " * len(prefix) + extra, file=out)


def _print_tune_write(write: dict[str, Any], language: str = "zh") -> None:
    """``tune write:`` 那一段：**存哪、写了哪几行、删了哪几行**（规格 §12.2）。

    输出顺序（前缀都是**机器可读**的稳定形态，同 ``tune hdr`` / ``tune best``；
    字段名与 ``reason=`` 的取值不跟 ``--language`` 翻译，脚本要能筛）：

    * ``tune file :`` —— 配置表**绝对路径**。写没写成都要先给这一行，否则"存哪了"
      无从查证（尤其 ``--profile-dir`` 是相对路径时，读者看到的是 CWD 下的结果）。
    * ``tune write:`` —— 写入行数、删除行数、判不出旧行数的键数、``reason=``。
    * ``tune wrote:`` —— **每一写入行**：五个键字段 + 配置字段（``kernel`` /
      ``optimize`` / ``wall_ms``）或等价载荷（``target_equation`` + 三个置换字段）
      + ``note``。缺哪一列都不算"完整"（§12.2 ①）。
    * ``tune del  :`` / ``tune delrow:`` —— 每个键删了几行，以及**每一被删行**的键与
      识别列（``line`` + 配置列）。同键多行时这是唯一能区分它们的证据。
    * ``tune dup  :`` —— 该键写前就有多行，本次写顺带做了"同键一行"清理。
    * ``tune error:`` / ``tune warn :`` —— 写失败原因与退化提示，**逐条完整**输出
      （多行原文照原样续打，不截断、不折叠）。

    为什么必须报删除条数与被删的键：同键只允许 1 行，写一行就必然删掉旧行；只报
    "写成功"会让"我上次那套配置去哪了"无从回答。
    """
    out = sys.stderr
    cache = _profile_module()
    rows = list(write.get("rows") or [])
    reason = str(write.get("reason") or "")
    # ① 配置表的绝对路径（窗口期外取：``profile_path()`` 只算字符串，不碰磁盘）。
    print("tune file : " + (_profile_file() or KERNEL_UNKNOWN), file=out)
    summary = "rows={}  deleted={}".format(len(rows), int(write.get("deleted") or 0))
    if write.get("unknown"):
        # 判不出旧行数的键数：接口缺席/异常类型不符时如实记 unknown，不猜。
        summary += "  deleted_unknown={}".format(int(write["unknown"]))
    duplicates = list(write.get("duplicates") or [])
    if duplicates:
        summary += "  duplicate_keys={}  duplicate_rows={}".format(
            len(duplicates), sum(int(item.get("rows") or 0) for item in duplicates))
    if reason:
        summary += "  reason={}".format(reason)
    print("tune write: " + summary, file=out)
    # ② 写入的每一行：键各字段 + 配置字段 + wall_ms + note（等价行则给完整载荷）。
    for item in rows:
        key_text = _key_text(cache, item.get("key") or {})
        if item.get("kind") == "equivalence":
            payload = item.get("equivalence") or {}
            print("tune wrote: kind=equivalence  {}  target_equation={}  "
                  "operand_order={}  operand_permutations={}  "
                  "output_permutation={}  note={}".format(
                      key_text,
                      payload.get("target_equation") or item.get("target") or "-",
                      payload.get("operand_order"),
                      payload.get("operand_permutations"),
                      payload.get("output_permutation"),
                      item.get("note") or "-"), file=out)
            continue
        wall_ms = item.get("wall_ms")
        print("tune wrote: kind=config  {}  kernel={}  optimize={}  wall_ms={}  "
              "note={}".format(
                  key_text, item.get("kernel") or "-", item.get("optimize") or "-",
                  "-" if wall_ms is None else f"{wall_ms:.6f}",
                  item.get("note") or "-"), file=out)
    # ③ 删除的行数与**被删行的键**（每一行都给，不只给总数）。
    for item in write.get("deletions") or []:
        print("tune del  : {}  deleted={}".format(
            _key_text(cache, item.get("key") or {}),
            int(item.get("deleted") or 0)), file=out)
        for dropped in item.get("rows") or []:
            print("tune delrow: line={}  {}  kernel={}  optimize={}  wall_ms={}  "
                  "target_equation={}  note={}".format(
                      dropped.get("line") or "?", _key_text(cache, dropped),
                      dropped.get("kernel") or "-",
                      dropped.get("optimize") or "-",
                      dropped.get("wall_ms") or "-",
                      dropped.get("target_equation") or "-",
                      dropped.get("note") or "-"), file=out)
    # ④ 是否因"同键多行"做过清理（写 1 行、删掉其余所有旧行）。
    for item in duplicates:
        print("tune dup  : {}  ".format(
            _key_text(cache, item.get("key") or {})) + translate(
                "tune.duplicate_cleanup", language,
                rows=int(item.get("rows") or 0),
                removed=int(item.get("rows") or 0)), file=out)
    for exc in write.get("errors") or []:
        _print_multiline("tune error: ", exc, out)
    # 退化路径的提示（例如 as_equivalence=true 胜出却没有可复用的改写、缺表警告）：
    # 在**任何**情形下都要露面，不能因为"这一行写成功了"就吞掉。
    for note in write.get("notes") or []:
        _print_multiline("tune warn : ", note, out)
    if rows:
        return
    # 没写成任何一行：给出**翻译过的**原因，机器行里只有稳定 token。
    if reason == TUNE_NO_USABLE:
        print("tune why  : " + translate("tune.none", language), file=out)
    elif reason == TUNE_API_MISSING:
        print("tune why  : " + translate("tune.api_missing", language), file=out)
    elif reason == TUNE_PROFILE_OFF:
        print("tune why  : " + translate("tune.profile_off", language), file=out)
    elif reason == TUNE_WRITE_FAILED:
        print("tune why  : " + translate(
            "tune.write_error", language,
            message="; ".join(str(item) for item in write.get("errors") or []) or "-"),
            file=out)


def print_tune(tune: dict[str, Any], language: str = "zh",
               verbose: bool = DEFAULT_VERBOSE) -> None:
    """把 ``--tune`` 的报告打到 **stderr**（既有输出行一字不动）。

    顺序：说明行 → 超时/缓存关闭的提示 → 遍历覆盖的取值 → 列名 → 分隔线（这几行是
    表头）→ **每一组一行**（``tune comb :``，含 ``wall_ms``、状态与原因原文；规格
    §12.1 要求全列，不随 ``--verbose`` 变化）→ 最快 5 行 + 每个 ``kernel_force``
    取值的最快一行（``--verbose`` 才有）→ 被排除的组合计数 → 画像 rows/keys/
    tune_session → **配置表路径 + 写入行 + 被删行的键**（``tune file:`` /
    ``tune wrote:`` / ``tune del:`` / ``tune delrow:`` / ``tune dup:``）→ 预算截断
    提示 → ``tune best :``（**最后一行**）。

    为什么坏组也逐行打：只给"最快 5 行"时，"某格到底跑没跑、为什么没跑"完全看不见
    （``error`` / ``timeout`` / ``MISMATCH`` 会被前 5 行挤掉）。§12 因此取消了一切
    抑制与截断——详细输出只多在这里，普通调用仍只有一行提示。
    """
    out = sys.stderr
    rows = list(tune.get("rows") or [])
    best = tune.get("best")
    print("tune      : " + translate(
        "tune.caption", language, combos=tune.get("combos"),
        warmup=tune.get("warmup"), rounds=tune.get("rounds"),
        compare=tune.get("compare") or "none",
        reference=tune.get("reference")), file=out)
    # 超时的**真实能力**必须写在报告里。子进程路线的能力更强（父进程 kill 子进程，
    # 能打断原生 C/CUDA），但**测量开销**也更高——两件事都要说，否则读者会拿
    # ``wall_ms`` 去和进程内口径比较。进程内路线仍只能尽力而为，装不上定时器时更要说，
    # 否则"没超时"会被误读成"够快"。
    timer = tune.get("timer")
    if timer == "subprocess":
        isolation = tune.get("isolation") or {}
        startup = isolation.get("startup") or {}
        stage = isolation.get("stage") or {}
        print("tune note : " + translate(
            "tune.subprocess_note", language,
            groups=int(startup.get("count") or 0),
            median=("-" if startup.get("median_ms") is None
                    else f"{float(startup['median_ms']):.1f}"),
            total=("-" if startup.get("total_ms") is None
                   else f"{float(startup['total_ms']):.1f}"),
            stage_mib=f"{float(stage.get('bytes') or 0) / (1024.0 * 1024.0):.1f}",
            stage_s=f"{float(stage.get('seconds') or 0.0):.2f}"), file=out)
    elif timer == "sigalrm":
        print("tune note : " + translate("tune.signal_note", language), file=out)
    elif timer == "posthoc":
        print("tune note : " + translate("tune.posthoc_note", language), file=out)
    # 预判剪枝：有几组**没跑**、依据是规划器的 FLOPs 而不是实测。它与 ``tune prune``
    # 那条（比当前上界慢）是两件事，混在一起就会把"没测过"说成"测出来慢"。
    if tune.get("predicted"):
        print("tune predict: " + translate(
            "tune.predicted", language, count=int(tune.get("predicted") or 0),
            rate=("-" if tune.get("rate_flops_per_s") is None else
                  f"{float(tune['rate_flops_per_s']):.6e}"),
            source=tune.get("rate_source") or "-"), file=out)
    # §12.3：临时关闭缓存的提示必须整句给出（不是只有一个 session=true 的字段）。
    # 关不上时更要说明白——那几组量到的是"带历史命中"的耗时，读者必须知道。
    session = (tune.get("write") or {}).get("session")
    if session is True:
        print("tune cache: " + translate("tune.session_on", language), file=out)
    elif session is False:
        print("tune cache: " + translate("tune.session_off", language), file=out)
    # 网格覆盖了哪些取值：用户显式写的 --kernel-force 等被忽略后，光看"最快 5 行"
    # 无法确认那一维到底遍历过没有，这一行把它钉死（机器可读，固定字段名）。
    axes = tune.get("axes") or {}
    print("tune axes : kernel_force={}  as_equivalence={}  optimize={}".format(
        ",".join(str(item) for item in (axes.get("kernel_force") or ())) or "-",
        ",".join(str(item) for item in (axes.get("as_equivalence") or ())) or "-",
        ",".join(str(item) for item in (axes.get("optimize") or ())) or "-"),
        file=out)
    # 分支判定（只探测、不计时）：它决定清单里有没有 ``topology`` 那一组，
    # 以及"为什么这一轮没试它"到底是什么原因。机器可读，字段名不翻译。
    branch = tune.get("branch") or {}
    if branch:
        print("tune branch: branch={}  raw_ok={}  equiv_ok={}  "
              "target_equation={}".format(
                  branch.get("branch") or "-",
                  str(bool(branch.get("raw_ok"))).lower(),
                  str(bool(branch.get("equiv_ok"))).lower(),
                  branch.get("target_equation") or "-"), file=out)
    # 遍历前的准入探测改了**组数**：``1/N`` 的 ``N`` 是裁剪后**实际会被遍历**的组数，
    # 所以这一行必须说清"计划多少、实际多少、哪些内核被排除"（机器可读，字段名不翻译）。
    skipped_kernels = [dict(item) for item in (tune.get("skipped_kernels") or [])]
    planned = tune.get("planned")
    kept = tune.get("total")
    if planned is not None and kept is not None:
        print("tune plan : planned={}  kept={}  trimmed={}  skipped-kernels={}".format(
            int(planned), int(kept), int(tune.get("trimmed") or 0),
            ",".join(str(item.get("kernel_force")) for item in skipped_kernels) or "-"),
            file=out)
    print("tune hdr  : 序号/总数  kernel_force  as_equivalence  optimize  wall_ms  "
          "状态(ok|timeout|error|mismatch|pruned|skip)  原因"
          + ("  （总数 = 裁剪后可用组数：计划 {} → 实际 {}{}）".format(
              int(planned), int(kept),
              "；已排除内核见 skipped-kernels: " + ",".join(
                  str(item.get("kernel_force")) for item in skipped_kernels)
              if skipped_kernels else "")
             if planned is not None and kept is not None else ""), file=out)
    print("tune ---- : " + "-" * 88, file=out)
    # 规格 §12.1：**每组一行、全部列出**——不受 --verbose 控制、不去重、不截断。
    # 坏组（timeout / error / mismatch / pruned）一样在这一段里，原因原文跟在
    # ``原因=`` 之后。
    total = len(rows)
    for index, row in enumerate(rows, 1):
        for line in tune_comb_lines(index, total, row):
            print(line, file=out)
    # 被排除的内核**整组一行**（不是每组一行）：同一个原因重复 7 遍只会把表格淹掉，
    # 而"这一档到底跑没跑、为什么没跑"仍然看得见（状态是 ``skip``，不是 ``error``）。
    for row in skipped_kernels:
        for line in tune_skip_line(row, dispatched=tune.get("dispatched") or {}):
            print(line, file=out)
    if verbose:
        measured = sorted((row for row in rows if row.get("median_ms") is not None),
                          key=lambda row: row["median_ms"])
        for row in measured[:5]:
            print("tune row  : " + (_bold(_tune_line(row, best))
                                    if row is best else _tune_line(row, best)),
                  file=out)
        # 「最快 5 行」会被某一个 kernel_force 取值整体占满（实测有这样的方程：
        # generic 比其它取值快一个量级时，5 行全是 generic，另外四档在表里一次都不
        # 出现——而用户正是想知道"那一档到底跑没跑、跑多快"）。所以再补每个
        # kernel_force 取值的最快一行（已在前 5 行里的不重复打）。
        shown = {id(row) for row in measured[:5]}
        for kernel in TUNE_KERNEL_FORCE:
            candidates = [row for row in measured if row["kernel_force"] == kernel]
            if not candidates or id(candidates[0]) in shown:
                continue
            print("tune row  : " + _tune_line(candidates[0], best), file=out)
        # 坏组合**不必**在这里补打：上面的 ``tune comb`` 已经把每一组连同原因原文
        # 都列出来了（原先这里那段"坏组合也要露脸"因此失效，删掉以免同一行出现两次）。
    skipped = [row for row in rows if row.get("error") or row.get("mismatch")]
    if skipped or any(row.get("timeout") for row in rows):
        print("tune skip : " + translate(
            "tune.skip", language,
            mismatch=sum(1 for row in skipped if row.get("mismatch")),
            error=sum(1 for row in skipped if row.get("error")),
            timeout=sum(1 for row in rows if row.get("timeout"))), file=out)
    pruned = [row for row in rows if row.get("pruned")]
    if pruned:
        print("tune prune: " + translate(
            "tune.prune", language, pruned=len(pruned),
            min_wall_ms=tune.get("min_wall_ms"),
            config="-" if not tune.get("min_wall_config") else (
                "kernel_force={}  as_equivalence={}  optimize={}".format(
                    *tune["min_wall_config"]))), file=out)
    profile = tune.get("profile") or {}
    enabled = tune.get("profile_enabled")
    write = tune.get("write") or {}
    print("tune prof : profile={}  enabled={}  rows={}  keys={}  session={}{}".format(
        tune.get("profile_requested") or KERNEL_UNKNOWN,
        KERNEL_UNKNOWN if enabled is None else str(bool(enabled)).lower(),
        KERNEL_UNKNOWN if profile.get("rows") is None else profile["rows"],
        KERNEL_UNKNOWN if profile.get("keys") is None else profile["keys"],
        # ``session`` = tune_session 是否真的开起来了：False 时上面那几组量的是
        # "带历史命中"的耗时，读者必须知道。
        KERNEL_UNKNOWN if write.get("session") is None
        else str(bool(write.get("session"))).lower(),
        f"  last_error={profile['last_error']}" if profile.get("last_error") else ""),
        file=out)
    # 最终写进配置表的行与删掉的同键行（规格 §3.3 / §7）：机器可读的稳定形态，
    # 与 ``tune hdr`` / ``tune best`` 同口径，字段名不翻译。
    _print_tune_write(write, language)
    # 预算用尽 → 必须报"已测 N/M"：不说清有多少组没测，"最优"就会被读成全局最优。
    if tune.get("stopped"):
        print("tune stop : " + translate(
            "tune.stop", language, measured=tune.get("measured"),
            total=tune.get("total"), skipped=tune.get("skipped"),
            budget=float(tune.get("budget") or 0.0)), file=out)
    if best is None:
        print("tune best : none  (" + translate("tune.none", language) + ")", file=out)
    else:
        print("tune best : kernel_force={}  as_equivalence={}  optimize={}  "
              "wall_ms={:.6f}".format(best["kernel_force"], best["as_equivalence"],
                                      best["optimize"], best["median_ms"]), file=out)
    # 规格 §13.3：**末尾**必须是一张对齐的汇总表，表尾 ``best:`` / ``saved:``。
    # 放在最后一行之后（而不是塞进上面的明细里）：读者扫一眼表尾就能拿到
    # "选谁、存哪、写了几行删了几行"，不必回头在 30 行明细里找。
    for line in tune_table_lines(tune, language):
        print(line, file=out)


# ── 语料反推（不写 --shape 时） ──────────────────────────────────────────────
def restore_original_shapes(mapping: Any,
                            rewritten_shapes: Sequence[tuple[int, ...]],
                            original_terms: Sequence[str],
                            sizes: dict[str, int]
                            ) -> list[tuple[int, ...]] | None:
    """把"等价写法"算出的形状按 ``mapping`` **反向还原**成用户原方程的轴序。

    为什么必须还原：``_equivalence.normalize_equation`` 给出的 ``mapping.equation``
    是**语料那条写法**的轴序（操作数位置、每个操作数的轴序、输出轴序都可能被改），
    ``roles`` 也是配着那条写法的。直接拿它的形状去跑用户的方程，等于把
    ``(100,4,1,1)`` 喂给本该是 ``(4,100,1,100)`` 的操作数——**形状看起来都合法**，
    只有数值悄悄错位。所以这里只给"能证明对得上"的答案。

    三个字段的用法（语义见 :class:`pyqcd.einsum._equivalence.Equivalence`）：

    * ``operand_order[i]`` = 改写后第 i 个操作数取自用户张量的第几个 → 逆映射回位置；
    * ``operand_permutations[i]`` = 该操作数的 ``transpose`` 参数（``new[j]=old[perm[j]]``）
      → 形状按 ``old[k]=new[perm⁻¹[k]]`` 逆推；
    * ``output_permutation`` **不参与**：它只动输出轴序，而输出形状由
      ``main`` 按用户方程的 ``->`` 段从 ``sizes`` 拼，本来就不会错。

    最后再拿 ``sizes`` 按**用户方程**的指标段重算一遍逐轴比对（``sizes`` 的键就是
    用户的字母，映射只在语料内部做字母双射）：两者不逐轴相等说明这条 mapping 与
    方程不自洽（例如被外部改过），此时返回 ``None`` —— 调用方按"未命中"报错，
    **绝不**返回一份错位的形状。``None`` 也覆盖字段缺失/长度不符/``perm`` 不是置换
    这些"无法反向施加"的情形。
    """
    count = len(rewritten_shapes)
    if len(original_terms) != count:
        return None
    order = mapping.operand_order
    if order is None:
        order = tuple(range(count))
    elif len(order) != count or sorted(order) != list(range(count)):
        return None
    # ``_build`` 在"没有任何操作数需要转置"时把该字段留成空元组，两种写法等价。
    perms = tuple(mapping.operand_permutations) or (None,) * count
    if len(perms) != count:
        return None
    restored: list[tuple[int, ...] | None] = [None] * count
    for index, rewritten in enumerate(rewritten_shapes):
        perm = perms[index]
        if perm is None:
            original = tuple(rewritten)
        else:
            if len(perm) != len(rewritten) or sorted(perm) != list(range(len(perm))):
                return None             # 不是置换：反向施加没有定义
            inverse = [0] * len(perm)
            for axis, source in enumerate(perm):
                inverse[source] = axis
            original = tuple(rewritten[axis] for axis in inverse)
        target = original_terms[order[index]]
        if len(original) != len(target):
            return None
        try:
            expected = tuple(sizes[label] for label in target)
        except KeyError:                # mapping 与方程不同字：无从校验
            return None
        if original != expected:
            return None                 # 字段与方程不自洽，宁可不给
        restored[order[index]] = original
    if any(shape is None for shape in restored):
        return None
    return [shape for shape in restored if shape is not None]


def shapes_from_corpus(equation: str, scene: str,
                       detail: dict[str, Any] | None = None
                       ) -> tuple[list[tuple[int, ...]], bool, dict[str, int]]:
    """按语料 ``roles`` + 生产 dims 反推形状。

    返回 ``(每个操作数的形状, 是否命中语料, label → 尺寸表)``。第三项让调用方能
    按方程输出段拼出**正确的输出形状**——少了它，省略 ``--shape`` 的非标量输出
    会被当成 ``()`` 而误判形状不符（见 :func:`main` 里的说明）。

    三阶段匹配（次序固定）
    ----------------------
    1. **直接匹配**：语料里有方程的**逐字**同一条（``task.equation == equation``）；
    2. **等价转换后匹配**：``mapping = _equivalence.normalize_equation(equation)``，
       再用 ``mapping.equation`` 去语料里找。轴序变体（换操作数位置 / 转置某个
       操作数 / 换输出轴序）在这一步命中，形状再按 :func:`restore_original_shapes`
       **反向还原**成用户原方程的轴序——还原不出来就退回第 3 步的报错；
    3. 两者都没有 → ``matched=False``，由 :func:`main` 报 ``error.corpus_miss``。

    ``detail`` 是**出参**（省略即不需要）：命中后写入
    ``{"matched_via": "direct"|"equivalence", "target_equation": 命中的那条语料方程}``，
    供 ``--explain``/``--verbose`` 打印"形状是从哪来的"。返回值保持三元组不变——
    既有调用方与契约按三元组解包，加第四个元素会把它们一起弄红。

    形状口径
    --------
    标签 → 尺寸仍由 :func:`pyqcd.einsum.bench_specialized._label_sizes` 按
    ``task.roles`` + 场景 dims 给出（本函数只决定用**哪条 roles**）。
    口径本身的切换（``_topology_gen_common.ROLE_SHAPES``）由那一侧负责，不在本函数。

    **不得 import ``pyqcd.testing``**（本库反模式：生产代码不与测试包耦合，见
    ``AGENTS.md`` 反模式节）。这里原先是
    ``from pyqcd.testing._topology_kernel_gen import parse``，而
    ``pyqcd/testing/__init__.py`` 顶层 ``from scipy.linalg import expm``——于是在
    没装 scipy 的运行环境里，一个**不需要 scipy 的纯 einsum 方程反推**会直接
    ``ModuleNotFoundError: No module named 'scipy'``（2026-11-15 用户实测）。
    改用生产侧既有的 :func:`pyqcd.einsum._dedicated_kernels.parse_equation`：
    它比测试侧那个两行 ``split`` 更严格（拒绝重复标签、输出标签缺失、省略号），
    正是这里该有的校验。
    """
    from pyqcd.einsum import _bench_scene
    from pyqcd.einsum import _equivalence
    from pyqcd.einsum import _wick_tasklist as tasklist
    from pyqcd.einsum._dedicated_kernels import parse_equation
    from pyqcd.einsum.bench_specialized import _label_sizes

    table = _bench_scene.dims(scene)

    def find_task(text: str) -> Any:
        for seq in range(1, len(tasklist.by_seq()) + 1):
            task = tasklist.lookup(seq)
            if task is not None and task.equation == text:
                return task
        return None

    def shapes_of(text: str, task: Any
                  ) -> tuple[list[tuple[int, ...]], dict[str, int]]:
        terms, _out = parse_equation(text)
        sizes = _label_sizes(terms, task.roles, table)
        return [tuple(sizes[label] for label in term) for term in terms], sizes

    # ① 直接匹配（命中口径与旧实现完全相同：逐字相等）
    task = find_task(equation)
    if task is not None:
        shapes, sizes = shapes_of(equation, task)
        if detail is not None:
            detail.update(matched_via="direct", target_equation=task.equation)
        return shapes, True, {str(k): int(v) for k, v in sizes.items()}

    # ② 等价转换后匹配：`normalize_equation` 自己带开关，闸关时只给 None。
    mapping = _equivalence.normalize_equation(equation)
    if mapping is None:
        return [], False, {}
    task = find_task(mapping.equation)
    if task is None:
        return [], False, {}
    rewritten_shapes, sizes = shapes_of(mapping.equation, task)
    original_terms, _out = parse_equation(equation)
    shapes = restore_original_shapes(mapping, rewritten_shapes, original_terms, sizes)
    if shapes is None:
        return [], False, {}            # 还原不可靠：按未命中报错，不给错位形状
    if detail is not None:
        detail.update(matched_via="equivalence", target_equation=task.equation)
    return shapes, True, {str(k): int(v) for k, v in sizes.items()}


# ── CLI ─────────────────────────────────────────────────────────────────────
def build_parser(language: str = "zh") -> argparse.ArgumentParser:
    """构造参数解析器。缺省值全部写在这里，``--help`` 即规范。

    ``language`` 取 ``"en"``/``"zh"``（缺省 ``"zh"`` 保持旧行为），只影响
    ``description``/``epilog``/``help`` 的文案。argparse 在构造时求值这些字符串，
    所以语言必须在调用本函数时定下来，不能在 import 时算。
    """
    lang = "en" if str(language).strip().lower().startswith("en") else "zh"

    def t(key: str, **kwargs: Any) -> str:
        return translate(key, lang, **kwargs)

    parser = argparse.ArgumentParser(
        prog="python -m pyqcd.einsum.test",
        description=t("help.description"),
        epilog=t("help.epilog"),
        # 排版由这段文案自己负责：argparse **不渲染** markdown，`**x**` 与双反引号
        # 都会原样打进终端（实测 `--help` 里仍有 22 处 `**`）。所以 MESSAGES 里只准用
        # 纯文本换行缩进；改完文案务必实跑一次 `--help` 看渲染，别只读源码。
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--equation", required=True, help=t("help.equation"))
    parser.add_argument("--shape", default=None, help=t("help.shape"))
    parser.add_argument("--scene", default=DEFAULT_SCENE,
                        help=t("help.scene", default=DEFAULT_SCENE))
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=DEVICES,
                        help=t("help.device", default=DEFAULT_DEVICE))
    parser.add_argument("--dtype", default=DEFAULT_DTYPE, choices=DTYPES,
                        help=t("help.dtype", default=DEFAULT_DTYPE))
    parser.add_argument("--backend", default=DEFAULT_BACKEND, choices=BACKENDS,
                        help=t("help.backend", default=DEFAULT_BACKEND))
    # ``default=None`` 是刻意的：只有这样才能区分"用户没给 --compare"与"用户给了
    # numpy"。规格 §13.1 要求 ``--tune`` 下 compare 默认关闭（不跑参考基准），
    # 而普通调用仍缺省 ``DEFAULT_COMPARE``；把缺省写死成 ``"numpy"`` 会让这两件事
    # 无法区分，于是 tune 每次都要多跑一遍参照。
    parser.add_argument("--compare", default=None, choices=COMPARE_MODES,
                        help=t("help.compare", default=DEFAULT_COMPARE))
    parser.add_argument("--compare-optimize", default=DEFAULT_COMPARE_OPTIMIZE,
                        choices=COMPARE_OPTIMIZERS,
                        help=t("help.compare_optimize",
                               default=DEFAULT_COMPARE_OPTIMIZE))
    parser.add_argument("--specialize", default=DEFAULT_SPECIALIZE,
                        choices=SPECIALIZE_MODES,
                        help=t("help.specialize", default=DEFAULT_SPECIALIZE))
    # ``default=None`` 是刻意的：``None`` = "用户没指定"，不是"显式 generic"。
    # ``_api._profile_begin`` 用 ``kernel_force is not None`` 判断显式性，argparse 缺省
    # 一填 ``DEFAULT_KERNEL``，每一次不写该选项的调用都会被判成显式 generic，配置表里
    # ``kernel=backend`` 的命中行就抛 ``ProfileConflict``（2026-11-18 用户实测）。
    # 真正生效的回落值（``generic``）仍显示在帮助文案里——它是"未指定时按 generic 执行"，
    # 不是"用户点名了 generic"。
    parser.add_argument("--kernel-force", default=None,
                        help=t("help.kernel", default=DEFAULT_KERNEL,
                               env=KERNEL_FORCE_ENV,
                               labels=", ".join(_kernel_label_names())))
    parser.add_argument("--as-equivalence", default=DEFAULT_AS_EQUIVALENCE,
                        choices=AS_EQUIVALENCE_MODES,
                        help=t("help.as_equivalence", env=AS_EQUIVALENCE_ENV,
                               default=DEFAULT_AS_EQUIVALENCE))
    # ``--tune`` 是**带取值的模式参数**（不是开关，2026-11-18 用户裁定）：
    # ``nargs="?"`` 让 ``--tune``（留空 → ``all``）、``--tune pass``、``--tune false``
    # 三种写法都成立；``const`` 是"留空"的取值，``default`` 是"没给"（= 不跑）。
    # 取值本身的解析在 :func:`parse_tune_mode`（唯一出口），这里不做校验：argparse 的
    # ``choices`` 会让 ``--tune pass`` 与 `--tune <其它选项>` 的报错文案变成 argparse
    # 自带的英文列表，而本入口的错误文案要跟 ``--language`` 走。
    parser.add_argument("--tune", nargs="?", const="true", default="false",
                        help=t("help.tune"))
    # 超时/预算只在 --tune 下有语义：缺省值写进 --help，用户不必猜"没写会跑多久"。
    parser.add_argument("--tune-timeout", type=float, default=DEFAULT_TUNE_TIMEOUT,
                        help=t("help.tune_timeout",
                               default=f"{DEFAULT_TUNE_TIMEOUT:g}"))
    parser.add_argument("--tune-budget", type=float, default=DEFAULT_TUNE_BUDGET,
                        help=t("help.tune_budget",
                               default=f"{DEFAULT_TUNE_BUDGET:g}"))
    parser.add_argument("--equivalence", default=DEFAULT_EQUIVALENCE,
                        choices=EQUIVALENCE_MODES,
                        help=t("help.equivalence", env=EQUIVALENCE_ENV,
                               default=DEFAULT_EQUIVALENCE))
    # ``default=None`` 与 ``--kernel-force`` 同理：``None`` 代表"用户没给"，
    # 只有显式 ``true``/``false`` 才写 ``cache :`` 报告行（``auto`` 与省略等价）。
    parser.add_argument("--cache", default=None, choices=CACHE_MODES,
                        help=t("help.cache", env=CACHE_ENV,
                               default=DEFAULT_CACHE))
    # 可写取值 = 配置表的 7 个叶子（``auto`` 不在其中，见 :func:`_optimize_choices`）。
    # ``default=None`` 是"用户没给"：``main`` 再把它还原成 ``DEFAULT_OPTIMIZE``，
    # 与 ``--kernel-force``/``--cache`` 同款三态，API 的缺省行为因此不变。
    optimize_choices = _optimize_choices()
    parser.add_argument("--optimize", default=None, choices=optimize_choices,
                        help=t("help.optimize",
                               values=" / ".join(optimize_choices or ())))
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL,
                        help=t("help.atol", default=f"{DEFAULT_ATOL:g}"))
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL,
                        help=t("help.rtol", default=f"{DEFAULT_RTOL:g}"))
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS,
                        help=t("help.rounds", default=DEFAULT_ROUNDS))
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                        help=t("help.warmup", default=DEFAULT_WARMUP))
    parser.add_argument("--seed", type=int, default=0, help=t("help.seed"))
    parser.add_argument("--verbose", default="false", help=t("help.verbose"))
    # ``auto`` 是 ``None``（跟随 --verbose），显式 true/false 压过它；缺省值写进
    # help，用户不必猜"为什么不开 verbose 也打印"。
    parser.add_argument("--explain", default=DEFAULT_EXPLAIN, choices=EXPLAIN_MODES,
                        help=t("help.explain", default=DEFAULT_EXPLAIN))
    # ``default='auto'`` = 没给：底层跟随 PYQCD_EINSUM_PROFILE。写成 ``None`` 会与
    # "用户显式关掉"混成一个值，所以这里用三态名，和 --specialize/--cache 同口径。
    parser.add_argument("--profile", default=DEFAULT_PROFILE, choices=PROFILE_MODES,
                        help=t("help.profile", env=PROFILE_ENV,
                               default=DEFAULT_PROFILE))
    parser.add_argument("--profile-dir", default=None,
                        help=t("help.profile_dir", env=PROFILE_DIR_ENV))
    parser.add_argument("--json", default=None, help=t("help.json"))
    parser.add_argument("--language", default=DEFAULT_LANGUAGE,
                        choices=LANGUAGE_MODES,
                        help=t("help.language", env=LANGUAGE_ENV,
                               default=DEFAULT_LANGUAGE))
    return parser


def _as_bool(text: str) -> bool:
    return str(text).strip().lower() in ("1", "true", "yes", "on")


def main(argv: Sequence[str] | None = None) -> int:
    # 语言必须在调用时解析（不能在 import 时），否则同进程内改语言会失效。
    # 两遍解析：先只取 --language，再按该语言构造 parser，让 --help/错误文案同语言。
    # 非法值不进第一遍——留给下面 parse_args 的 choices 校验，否则会先抛
    # UsageError 而把错误信息（int 型退出码不打印消息）与 --help 一起吞掉。
    peeked = _peek_language(argv)
    language = resolve_language(peeked if peeked in LANGUAGE_MODES else None)
    parser = build_parser(language=language)
    args = parser.parse_args(argv)
    language = resolve_language(args.language)
    # ``--tune`` 的取值 → 模式名（``all`` / ``pass`` / ``off``）。解析在这里**只做一次**：
    # 后面每一处判定都看 ``tune_mode``，不再各自把字符串翻成布尔（``_as_bool("pass")``
    # 是 ``False``——那会把 pass 模式静默变成"不跑"）。
    tune_mode = parse_tune_mode(args.tune, language)
    tune_on = tune_mode != TUNE_MODE_OFF
    pass_mode = tune_mode == TUNE_MODE_PASS
    # ``--tune`` 自己会遍历这三轴：显式非 auto 取值只**警告 + 忽略**（每个参数一行），
    # 遍历照常进行、退出码仍为 0——顺手写上的一维不该让整轮 tune 变成用法错误。
    # ``pass`` 模式也一样（它把 as_equivalence 钉成 false，同样不采纳用户给的值）。
    if tune_on:
        tune_conflicts(kernel_force=args.kernel_force,
                       as_equivalence=args.as_equivalence,
                       optimize=args.optimize, language=language)
    # 规格 §13.1：``--tune`` 下 compare **默认关闭**（不给 --compare 就不跑参考基准），
    # 显式给出时照旧执行；普通调用仍缺省 ``DEFAULT_COMPARE``。这只决定"要不要另跑
    # 参照"，不改遍历维度（tune 自己仍遍历 {auto,none,greedy}）。
    # ``pass`` 模式**禁用 --compare 的一切**（主路径与遍历都不跑参照）：用户显式给了
    # 就制裁一行警告再忽略——静默忽略会让人以为数值真的校验过了。
    if pass_mode and args.compare is not None and args.compare != "none":
        _warn_tune_ignored(translate("tune.pass_compare_disabled", language,
                                     value=args.compare), language)
    compare_value = args.compare if args.compare is not None else (
        "none" if tune_on else DEFAULT_COMPARE)
    if pass_mode:
        compare_value = "none"
    # ``--optimize`` 的 ``None`` 是"用户没给"（``auto`` 已从 CLI 可写集合里删除）：
    # 这里还原成签名缺省值，报告行与 ``run()`` 看到的值与上一版逐字相同。
    optimize_value = args.optimize if args.optimize is not None else DEFAULT_OPTIMIZE

    if args.shape is None:
        # ``corpus_detail`` 是出参：命中来源（direct/equivalence）与命中的那条语料
        # 方程。返回值仍是三元组，见 ``shapes_from_corpus`` 的说明。
        corpus_detail: dict[str, Any] = {}
        shapes, matched, label_sizes = shapes_from_corpus(
            args.equation, args.scene, corpus_detail)
        if not matched:
            raise UsageError(translate("error.corpus_miss", language,
                                       equation=args.equation, scene=args.scene))
        # 形状来自等价写法的反演时**必须说清**：形状是它给的，读者要知道那条
        # 语料方程是谁（否则形状与自己的方程轴序看着对不上，无从核对）。
        if corpus_detail.get("matched_via") == "equivalence" and resolve_explain(
                parse_explain(args.explain, language), _as_bool(args.verbose)):
            print(f"shape source: corpus(equivalence) via "
                  f"{corpus_detail['target_equation']}")
        # 输出形状必须从**反推出的形状**推导，不能留空。
        #
        # 原先把 ``out_shape`` 设成 ``None``、下面再兜成 ``()``，于是任何省略
        # ``--shape`` 且输出非标量的方程都会误报
        # ``actual shape=[100, 100] expected=[]``（2026-11-15 用户实测
        # ``...vx->es`` 命中）。这里按方程输出段从 label→尺寸表拼出来。
        _, rhs = args.equation.split("->", 1)
        out_shape = tuple(label_sizes[ch] for ch in rhs.replace(" ", ""))
        report_note = translate("note.shape_from_corpus", language,
                                scene=args.scene)
    else:
        # 传 equation 进去：``--shape`` 省略 ``->…`` 时按方程输出段推导输出形状，
        # 而不是一律当标量（否则 ``...->es`` 这类会误报 expected=[]）。
        try:
            shapes, declared_out = parse_shape_spec(
                args.shape, language=language, equation=args.equation)
        except ValueError as exc:
            # 参数错误必须走 UsageError（退出码 2），不能裸 ValueError（退出码 1）。
            # 否则脚本里"改命令行"与"查算法"两类失败无法区分（--help 承诺 2）。
            raise UsageError(str(exc)) from exc
        report_note = translate("note.shape_from_flag", language)
        out_shape = declared_out
        if out_shape is None:                     # 没有 equation 时的旧语义
            out_shape = ()

    # 这三个取值先在本地解析一次：主路径与 ``--tune`` 必须看到**同一套**语义
    # （尤其 ``--cache false`` 不许被遍历那一路绕过）。
    specialize_value = parse_specialize(args.specialize, language)
    cache_value = parse_cache(args.cache, language)
    profile_value = parse_profile(args.profile, language)
    # 缺表提示要用**本次命令行**的真实参数（提示里原先按键重建，GPU 下会退成
    # ``--device cpu``、形状也没有输出段——2026-11-18 用户实测）。
    _install_cli_hint(args, compare_given=args.compare is not None)
    try:
        result = run(
            equation=args.equation, shapes=shapes,
            out_shape=tuple(out_shape),
            device=args.device, dtype=args.dtype, backend=args.backend,
            atol=args.atol, rtol=args.rtol, compare=compare_value,
            optimize=optimize_value, rounds=args.rounds, warmup=args.warmup,
            seed=args.seed, verbose=_as_bool(args.verbose),
            specialize=specialize_value,
            compare_optimize=args.compare_optimize,
            equivalence=parse_equivalence(args.equivalence, language),
            as_equivalence=args.as_equivalence,
            kernel_force=args.kernel_force,
            cache=cache_value,
            explain=parse_explain(args.explain, language),
            profile=profile_value,
            profile_dir=args.profile_dir,
            language=language,
        )
    except _kernel_unavailable_error() as exc:
        # 强制内核没跑成：这是**环境**问题（原生库/驱动/形态不支持），不是数值不符，
        # 因此按 ``--help`` 的承诺用退出码 2。异常正文已经写清 requested / attempted /
        # 逐标签原因，**原样**打出来即可——降级成一行 warning + 通用路径会让用户以为
        # 强制生效了。
        print(translate("result.error", language, message=exc), file=sys.stderr)
        return 2
    except _profile_error() as exc:
        # 配置表的数据/冲突错误（§4 同键多行、§5 命中行与显式输入冲突）：正文已经由
        # ``_api`` 拼好（键、表中值、显式值、记录的文件行号、怎么处理，以及**全部**
        # 候选行），这里只**原样**打印，绝不加 traceback、不改写成 ``FAIL``。
        # 退出码 2：脚本据此区分"改命令行/改表"与"算法结果不对"（后者是 1）。
        # 同键多行那份正文 ``_api`` 已经打过一遍（列全部候选是它的职责），这里不再重复
        # 第二遍——两段一模一样的话只会让人以为是两次独立的失败。
        cache = _profile_module()
        duplicate = getattr(cache, "DuplicateProfileEntry", None) if cache else None
        if not (isinstance(duplicate, type) and isinstance(exc, duplicate)):
            print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        # CLI 上下文只服务**本次**调用：同进程的下一次调用（测试、脚本循环）不该
        # 继承上一条命令行的 device/shape。
        _clear_cli_hint()
    result["shape_source"] = report_note
    if result.get("specialize_warning"):
        print(translate("result.warning", language,
                        message=result["specialize_warning"]), file=sys.stderr)
    if result.get("compare_optimize") not in (None, DEFAULT_COMPARE_OPTIMIZE) and \
            result.get("compare_optimize_effective") is None:
        print(translate("result.warning", language, message=translate(
            "result.warning_compare_optimize_inert", language,
            value=result["compare_optimize"], compare=compare_value)),
            file=sys.stderr)

    # ``--tune``：在主路径既有输出之后**追加**遍历（表格走 stderr，既有行不动）。
    # 放在 ``--json`` 之前，``report['tune']`` 才会一起落进 JSON。
    tune_summary: dict[str, Any] | None = None
    if tune_on:
        tune_summary = run_tune(
            equation=args.equation, shapes=shapes,
            device=args.device, dtype=args.dtype, backend=args.backend,
            atol=args.atol, rtol=args.rtol, rounds=args.rounds,
            warmup=args.warmup, seed=args.seed, specialize=specialize_value,
            cache=cache_value, profile=profile_value,
            profile_kwargs=_contract_profile_kwargs(profile_value, args.profile_dir),
            timeout=args.tune_timeout, budget=args.tune_budget,
            compare=compare_value, mode=tune_mode,
            language=language,
        )
        result["tune"] = tune_summary["rows"]
        # ``pass`` 模式关掉过程详解（``--verbose`` 里的"最快 5 行 / 每个内核最快一行"
        # 与 ``--explain`` 的三段因果）：那是解说，不是"快速通过"要的东西。
        print_tune(tune_summary, language,
                   verbose=_as_bool(args.verbose) and not pass_mode)

    if args.json:
        # 写不进去属于**参数/环境错误**（退出码 2），不是数值失败（1）。
        # 原先裸 ``open()`` 让 ``FileNotFoundError``/``PermissionError`` 冒泡，
        # traceback + rc=1，脚本会以为是"结果不对"（2026-11-15 第六轮复查实证）。
        try:
            with open(args.json, "w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=1, ensure_ascii=False)
        except OSError as exc:
            raise UsageError(translate("error.json_unwritable", language,
                                       path=args.json,
                                       exc=f"{type(exc).__name__}: {exc}")) from exc
        print(f"wrote {args.json}")

    if not result.get("ok"):
        print(f"FAIL {args.equation}")
        if result.get("error"):
            print(translate("result.error_line", language,
                            message=result["error"]))
        if "rel" in result:
            print(translate("result.fail_rel", language,
                            rel=result["rel"], atol=args.atol))
            print(translate("result.fail_shape", language,
                            actual=result.get("actual_shape"),
                            expected=result["out_shape"]))
        return 1
    detail = ""
    if "rel" in result:
        detail = f" rel={result['rel']:.2e} vs {result['reference']}"
    if "seconds" in result:
        detail += f"  {result['seconds'] * 1000:.3f}ms"
        ref_seconds = result.get("reference_seconds")
        if ref_seconds:
            ref_opt = result.get("compare_optimize_effective")
            tag = f"@{ref_opt}" if ref_opt else ""
            detail += (f" (ref {ref_seconds * 1000:.3f}ms{tag}, "
                       f"{result['seconds'] / ref_seconds:.2f}x)")
        peak = (result.get("memory") or {}).get("peak") or {}
        if tune_summary is not None:
            detail += _tune_pass_detail(tune_summary)
        if peak.get("gpu_used_mib") is not None:
            detail += f"  vram+{peak['gpu_used_mib']:.0f}MiB"
        elif peak.get("rss_mib") is not None:
            detail += f"  rss+{peak['rss_mib']:.0f}MiB"
    print(f"PASS {args.equation}{detail}")
    return 0


def _tune_pass_detail(tune: dict[str, Any]) -> str:
    """``PASS`` 行里追加的 ``min_wall_ms=…`` 与最终 ``best`` 配置（用户要求，2026-11-18）。

    为什么两个都写：``min_wall_ms`` 是**剪枝上界**（遍历中被最小组压低到哪），``best``
    是**最终选中的配置**。两者数值相同也要并列显示——那正是"上界确实来自最优组"的
    唯一直接证据；只写一个，读者无法分辨"上界被压低"与"碰巧选到"。

    ``--tune-timeout 0``（不设上界）时 ``min_wall_ms`` 是 ``None``，此时退回 ``best``
    的实测中位：报告里给的是"这个数"，而不是一个 ``None`` 让人以为没测到。
    """
    best = tune.get("best")
    bound = tune.get("min_wall_ms")
    if bound is None and isinstance(best, dict):
        bound = best.get("median_ms")
    text = "  min_wall_ms=" + ("-" if bound is None else f"{float(bound):.6f}") + "ms"
    if not isinstance(best, dict):
        return text + "  best=none"
    return text + "  best=kernel_force={} as_equivalence={} optimize={}".format(
        best.get("kernel_force") or "-",
        best.get("as_equivalence") or "-",
        best.get("optimize") or "-")


def _install_cli_hint(args: Any, *, compare_given: bool) -> None:
    """把**本次命令行**的实参交给缺表提示（``需要先 tune：`` 那一段）。

    为什么不让提示按键重建：键里没有"用户给的 ``--shape``（含 ``->[…]`` 输出段）"与
    "用户给没给 ``--compare``"，而 GPU 下 ``device`` 列是 ``cupy:0``。只按键重建，
    用户在 GPU 上跑却收到一条 ``--device cpu``、形状还少一段的命令（2026-11-18 实测）。
    这里递过去的全是 argparse 的实参，提示因此与"他刚才怎么跑的"逐字对得上。

    环境变量/旧版库没有这个接口时**静默跳过**：提示退化成键重建（仍可用），
    绝不因为一个提示接口缺失而挡住主流程。
    """
    cache = _profile_module()
    install = getattr(cache, "set_cli_hint_context", None) if cache is not None else None
    if not callable(install):
        return
    try:
        install(device=str(args.device or ""), dtype=str(args.dtype or ""),
                equation=str(args.equation or ""), shape=args.shape,
                backend=str(args.backend or ""),
                compare=(str(args.compare) if compare_given else None))
    except Exception:                       # noqa: BLE001 - 提示层不许影响主流程
        return


def _clear_cli_hint() -> None:
    """丢掉 CLI 上下文（一次 CLI 调用结束；同进程的下一次调用不该继承它）。"""
    cache = _profile_module()
    clear = getattr(cache, "clear_cli_hint_context", None) if cache is not None else None
    if not callable(clear):
        return
    try:
        clear()
    except Exception:                       # noqa: BLE001 - 同上
        return


def _peek_language(argv: Sequence[str] | None) -> str | None:
    """在完整解析之前抽出 ``--language``（两种写法都认），用于选 help 文案。

    解析失败（如 ``--language fr``）交给第二次完整解析去报错；此处只做保守取值。
    """
    items = list(sys.argv[1:] if argv is None else argv)
    found: str | None = None
    for index, item in enumerate(items):
        if item == "--language" and index + 1 < len(items):
            found = items[index + 1]
        elif item.startswith("--language="):
            found = item.split("=", 1)[1]
    return found


if __name__ == "__main__":
    # ``UsageError`` 的退出码是 int 2，解释器不会打印 ``args`` 里的文案，所以
    # 在这里统一把它写进 stderr（21 处 raise 点无需各自打印）。
    try:
        raise SystemExit(main())
    except UsageError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        raise SystemExit(2) from None
