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
* **强制内核**：``--kernel-force auto|generic|dedicated|topology|backend``。``auto``
  （缺省）保持原分派次序，失败只 ``warning`` + 回落；``generic`` 完全跳过专用内核；
  ``dedicated`` / ``topology`` 只允许**当前后端的**那一个（numpy → ``kupl_*``，
  cupy → ``cuda_*``），可逗号分隔多个、按书写次序尝试，强制**真试一次**，跑不成就抛
  ``KernelUnavailableError`` 并以非零退出码结束——**绝不**静默回落；``backend``
  直接调后端自己的 ``einsum``（与其它标签互斥）。
  显式 ``--kernel-force`` 压过环境变量 ``PYQCD_EINSUM_KERNEL_FORCE``。明细里的
  ``kernel :`` 行给出 ``requested`` / ``ran`` / ``attempted``。
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
  **记录值**，不是本次实测。
* **自动施加等价写法**：``--as-equivalence auto|true|false|force``。前三态同
  ``--equivalence``；``force`` 是第四态，**忽略负标记**：画像里"该方程查过、无法
  等价改写"的跨进程标记不再拦住这次查询，而是真的再查一遍并刷新结论（``true``
  会尊重该标记）。字符串 ``"force"`` 原样交给 ``contract(as_equivalence=...)``。
* **遍历调优**：``--tune``（缺省关）在**同一条**方程/形状/dtype/设备/后端上遍历
  ``kernel_force × as_equivalence × optimize``（5×2×3 = 30 组，硬上限 30，超了报
  用法错误而不是截断），``--compare`` 固定 ``none``。这三轴由 ``--tune`` 自己给，
  所以同时再显式给 ``--kernel-force`` / ``--as-equivalence`` / ``--optimize``
  （非 ``auto``）时**各打一行 stderr 警告后忽略它**（每个参数一行），遍历照常进行、
  退出码仍为 0。每组沿用 ``--warmup``/``--rounds`` 取中位，结果全部写进配置画像并在
  打印前 ``flush()``；表格按中位升序（前 5 行 + 最优行标记）打到 stderr，既有输出行
  一字不动。报的是**单次运行的实测，不是基准**。
* **超时与预算**：``--tune-timeout``（缺省 100 秒）是**每一次** ``contract()`` 调用的
  上限（warmup 与每一轮都算）：任一次超时 → 该组记 ``timeout``、**不写画像**、继续下一
  组，中位只由全部成功且未超时的调用构成（不拿部分样本凑）。``--tune-budget``（缺省
  ``0`` = 不限）是整轮上限，用尽即停止遍历并报"已测 N/M"。实现用
  ``signal.setitimer`` + ``SIGALRM``：**可能打断原生内核，建议在新进程里跑 tune**
  （信号只在主线程可用；装不上定时器时退化成"跑完再按耗时判定"）。
  坏组合——超时、抛异常、``rel > atol+rtol``——都会被 ``_profile_cache.drop()`` 从
  内存索引与待写队列里撤回（落盘层是追加写，历史行删不掉，那种情形如实标注）。
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
from typing import Any, Sequence

import numpy as np

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
BACKENDS = ("auto", "numpy", "cupy", "torch", "kupl", "cuda")
#: ``--compare``：参照实现。
COMPARE_MODES = ("numpy", "opt_einsum", "backend", "none")
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
DEFAULT_COMPARE = "numpy"
DEFAULT_COMPARE_OPTIMIZE = "auto"
DEFAULT_OPTIMIZE = "auto"
DEFAULT_SCENE = "distill"
DEFAULT_VERBOSE = False
#: ``--language``：输出语言。``auto`` 见 :func:`resolve_language`。
LANGUAGE_MODES = ("auto", "en", "zh-cn")
DEFAULT_LANGUAGE = "auto"

# ── 遍历调优（--tune） ──────────────────────────────────────────────────────
#: ``--tune`` 遍历的三个轴，**顺序即报告顺序**。``as_equivalence`` 只遍历
#: ``true``/``false``：``auto`` 是"跟随环境变量"、``force`` 是"不信负标记"，两者都
#: 不是可比的性能档。
TUNE_KERNEL_FORCE = ("auto", "dedicated", "topology", "generic", "backend")
TUNE_AS_EQUIVALENCE = ("true", "false")
TUNE_OPTIMIZERS = ("auto", "greedy", "optimal")
#: 组合数**硬上限**：超了直接报用法错误，**绝不静默截断**——截断会让"最优"变成
#: "某个被截断的网格里最好"，而报告字面上完全看不出这一点。
TUNE_MAX_COMBOS = 30
#: 被 ``--tune`` 自行遍历的三维（报告与警告文案要用它说明"为什么你给的值被忽略"）。
TUNE_AXES = "kernel_force × as_equivalence × optimize"
#: ``--tune-timeout``：**每一次** ``contract()`` 调用（warmup 与每一轮都算）的上限
#: 秒数。``<= 0`` = 不设上限。超时的那一组记 ``timeout``、不写画像、继续下一组。
DEFAULT_TUNE_TIMEOUT = 100.0
#: ``--tune-budget``：整轮遍历的上限秒数，``0`` = 不限。用尽即停止遍历并报"已测 N/M"。
DEFAULT_TUNE_BUDGET = 0.0


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
        "en": ("Where the arrays live: cpu→numpy, gpu/cuda→cupy. "
               "Default {default}"),
        "zh": "数组落在哪：cpu→numpy，gpu/cuda→cupy。缺省 {default}",
    },
    "help.dtype": {
        "en": "Four dtypes on equal footing, default {default}",
        "zh": "四类平权，缺省 {default}",
    },
    "help.backend": {
        "en": ("Passed straight to contract(backend=...). auto lets the library "
               "decide; numpy/cupy/torch pick the array library; kupl/cuda pick the "
               "native entry. Default {default}"),
        "zh": ("传给 contract(backend=...)。auto 让库自判；"
               "numpy/cupy/torch 指定数组库；kupl/cuda 指定原生入口。"
               "缺省 {default}"),
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
        "en": ("Which specialised kernel may run. Kinds of value: auto = keep the "
               "original dispatch (dedicated, then topology) and, when none runs, "
               "only warn and fall back to the generic path (default); generic = skip "
               "the specialised kernels entirely (no warning); dedicated / topology = "
               "allow exactly that kernel of the operand backend (numpy -> kupl, "
               "cupy -> cuda), comma separated to allow both, tried in the order "
               "written; backend = call the backend's own einsum and skip the "
               "specialised dispatch (cannot be combined with other labels). "
               "An explicit --kernel-force wins over the {env} environment variable. "
               "Forcing a kernel makes it really try once (its negative cache "
               "is bypassed) and, when it does not run, raises KernelUnavailableError "
               "with a non-zero exit code — it never falls back to the generic path. "
               "Main path only: the --compare backend reference keeps its own dispatch "
               "(forcing a kernel on the other backend cannot work). A kernel that is "
               "not available is reported as an error, never as a warning. An unknown "
               "value exits with code 2. Default {default}"),
        "zh": ("允许跑哪个专用内核。取值有几种：auto = 保持原分派次序（dedicated → "
               "topology），跑不成只 warning 后回落到通用路径（缺省）；generic = "
               "完全跳过专用内核（不产生 warning）；dedicated / topology = 只允许"
               "当前后端的对应内核（numpy → kupl，cupy → cuda），可用逗号给两个、"
               "按书写次序尝试；backend = 直接调后端自己的 einsum、不走专用分派"
               "（不可与其它标签混用）。"
               "显式 --kernel-force 优先于环境变量 {env}。"
               "强制具体内核时会**真试一次**（旁路负缓存），跑不成即抛 "
               "KernelUnavailableError 并以非零退出码结束——绝不回落成通用路径。"
               "只作用于主路径：--compare backend 的参照仍走自己的分派"
               "（在另一个后端上强制内核必然不可用）。"
               "强制跑不成一律报错，不会降级成 warning。未知取值以退出码 2 结束。"
               "缺省 {default}"),
    },
    "help.optimize": {
        "en": ("Contraction-order optimiser (auto/optimal/dp/greedy/branch-*, or an "
               "explicit path). Default {default}"),
        "zh": "收缩序优化器（auto/optimal/dp/greedy/branch-*，或显式 path）。缺省 {default}",
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
        "en": ("Sweep kernel_force × as_equivalence × optimize and report the median of "
               "THIS run for every combination. The equation, shapes, dtype, device and "
               "backend are FIXED (they are this run's invariants); --compare is fixed "
               "to none, so no reference is timed. The three swept axes are supplied by "
               "--tune itself, so an explicit --kernel-force / --as-equivalence / "
               "--optimize other than auto is IGNORED with one warning line per option "
               "on stderr (the sweep still runs and the exit code stays 0). The grid is "
               "5 × 2 × 3 = 30 combinations (hard cap 30; more is a usage error, never a "
               "silent truncation). Each combination reuses --warmup / --rounds and the "
               "median is what was measured now — a single-run measurement, NOT a "
               "benchmark. Every combination is recorded into the configuration profile "
               "(force_write), and a bad one — timeout, exception, or rel above "
               "--atol/--rtol — is withdrawn from the profile again; the table goes to "
               "stderr (existing lines unchanged): header lines + the 5 fastest rows + "
               "the fastest row of EVERY kernel_force value + the best row, and without "
               "--verbose only the header and the best line. A "
               "combination that raises is marked error=<type>, one that exceeds "
               "--tune-timeout is marked timeout, one whose rel exceeds --atol/--rtol is "
               "marked MISMATCH, and none of them can win. Default off"),
        "zh": ("遍历 kernel_force × as_equivalence × optimize，逐组给出**本次**实测中位。"
               "方程、形状、dtype、设备与后端**固定不动**（它们是本次的不变量）；"
               "--compare 固定为 none，因此不额外计时参照。被遍历的三维由 --tune 自己"
               "提供：显式 --kernel-force / --as-equivalence / --optimize 只要不是 auto，"
               "就会**被忽略**，每个参数在 stderr 上给一行警告（遍历照常进行，退出码仍为 "
               "0）。网格为 5 × 2 × 3 = 30 组（硬上限 30，超出即用法错误，"
               "**绝不静默截断**）。每组沿用 --warmup / --rounds，中位就是此刻测到的值——"
               "**单次运行的实测，不是基准**。每组都会写进配置画像（force_write），"
               "坏组合——超时、抛异常、rel 超 --atol/--rtol——会被重新撤回；"
               "表格打到 stderr（既有输出行不动）：表头 + 最快 5 行 + 每个 kernel_force 取值的最快一行 + 最优行；"
               "不加 --verbose 时只给表头与最优行。抛异常的组合标 error=<类型>，"
               "超过 --tune-timeout 的标 timeout，rel 超 --atol/--rtol 的标 MISMATCH，"
               "三者都不参与选优。缺省关闭"),
    },
    "help.tune_timeout": {
        "en": ("Upper bound, in seconds, for EACH contract() call of --tune (warmup and "
               "every round count). A combination whose call exceeds it is marked "
               "timeout, is NOT written to the profile, and the sweep continues; the "
               "median is taken over all calls that succeeded within the bound (a "
               "partial sample set never becomes a median). 0 or negative disables the "
               "bound. Implemented with signal.setitimer + SIGALRM, which MAY INTERRUPT "
               "A NATIVE KERNEL — prefer running tune in a fresh process. Signals are "
               "only available on the main thread; where no timer can be armed the call "
               "is measured first and judged afterwards. Default {default}"),
        "zh": ("--tune 的**每一次** contract() 调用的上限秒数（warmup 与每一轮都算）。"
               "任一次超过它 → 该组记 timeout、**不写画像**、继续下一组；中位只由全部"
               "在上限内成功的调用构成（不拿部分样本凑中位）。0 或负数 = 不设上限。"
               "实现用 signal.setitimer + SIGALRM：**可能打断原生内核**，建议在新进程里"
               "跑 tune。信号只在主线程可用；装不上定时器时退化成「先跑完、再按耗时判定」。"
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
              "(compute_ms={compute_ms})",
        "zh": "[pyqcd.einsum] profile 命中 → backend={backend} kernel={kernel} "
              "（compute_ms={compute_ms}）",
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
               "compare=none (fixed); warmup={warmup}, rounds={rounds}, median of THIS "
               "run; rel vs {reference}. Single-run measurement of this process, "
               "NOT a benchmark."),
        "zh": ("{combos} 组，方程/形状/dtype/设备/后端全部固定；compare=none（固定）；"
               "warmup={warmup}、rounds={rounds}，取**本次**中位；rel 对 {reference}。"
               "这是**单次运行的实测，不是基准**。"),
    },
    "tune.skip": {
        "en": ("excluded from best: MISMATCH={mismatch} error={error} timeout={timeout} "
               "(a mismatched, failed or timed-out combination never wins)"),
        "zh": ("不参与选优：MISMATCH={mismatch} error={error} timeout={timeout}"
               "（数值不符、跑不成或超时的组合一律不参与最优选择）"),
    },
    "tune.none": {
        "en": "no usable combination: every one errored, timed out or mismatched",
        "zh": "没有任何可用组合：全部报错、超时或数值不符",
    },
    "tune.ignored": {
        "en": ("--tune sweeps {swept} itself, so the value you gave is ignored: {given!r}"),
        "zh": "--tune 会自行遍历 {swept}，已忽略你给的 {given!r}",
    },
    "tune.signal_note": {
        "en": ("per-call timeout uses SIGALRM — it may interrupt a native kernel; "
               "prefer running tune in a fresh process"),
        "zh": ("单次超时用 SIGALRM 实现——可能打断原生内核，建议在新进程里跑 tune"),
    },
    "tune.posthoc_note": {
        "en": ("no timer available here: each call is timed first and judged "
               "afterwards, so a slow one still runs to completion"),
        "zh": ("本环境装不上定时器：每次调用先跑完再按耗时判定，超时的那次仍会跑完"),
    },
    "tune.stop": {
        "en": ("measured {measured}/{total} ({skipped} combination(s) skipped): "
               "--tune-budget {budget:g}s used up"),
        "zh": ("已测 {measured}/{total}（跳过 {skipped} 组）：--tune-budget {budget:g}s 已用尽"),
    },
    "tune.history_kept": {
        "en": ("{n} bad combination(s) could not be erased from disk: profile.csv is "
               "append-only and keeps historical rows (in-memory index and pending queue "
               "were withdrawn)"),
        "zh": ("{n} 组坏组合没能从磁盘上抹掉：profile.csv 是追加写、会保留历史行"
               "（内存索引与待写队列已撤回）"),
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
               "backend cannot be combined with other labels), got {value!r}"),
        "zh": ("--kernel-force 必须是 {labels} 之一"
               "（多个用逗号分隔；backend 不可与其它标签混用），得到 {value!r}"),
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

    ``kernel`` 是 ``--kernel-force`` 的原始取值（``None`` = 缺省 ``auto``），解析后原样
    传给 ``contract(kernel_force=...)``。强制具体标签而没跑成时，``KernelUnavailableError``
    **不被吞掉**：它带着 requested/attempted/逐标签原因向上抛，由 :func:`main`
    打印正文并非零退出。``auto`` 模式仍只 warning + 回落，行为逐字不变。

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

    lang = resolve_language(language)
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
    profile_requested = parse_profile(profile, lang)
    profile_kwargs = _contract_profile_kwargs(profile_requested, profile_dir)
    # ``--kernel-force`` 先解析：非法标签必须在构造任何操作数之前就报成用法错误。
    # 规范化后的字符串**原样**给 ``contract(kernel_force=...)``，报告里的 ``requested=``
    # 也用同一处解析结果，二者不会各说各话。
    kernel_value = ",".join(kernel_labels(kernel_force, lang))
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
                           specialize=specialize, kernel_force=kernel_value or None,
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
#: 缺省：保持原分派次序（失败只 warning + 回落）。与 ``_kernel_force.AUTO_LABEL``
#: 同值；本模块不 import 它，免得 ``_kernel_force`` 缺席时整个入口起不来。
DEFAULT_KERNEL = "auto"
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

    含 ``auto`` / ``generic`` / ``backend`` —— 报错文案要列的是**能写什么**，不是
    "可强制的内核标签"子集，只列 ``dedicated, topology`` 会漏掉两个合法取值。
    """
    force = _kernel_force_module()
    live = getattr(force, "KERNEL_LABELS", None) if force is not None else None
    names = set(live or KERNEL_LABELS)
    names |= {DEFAULT_KERNEL, GENERIC_LABEL, BACKEND_LABEL}
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

    返回 ``_kernel_force.resolve`` 的原始元组（``()`` = 自动分派，
    ``("generic",)`` = 只走通用路径，其余为要强制的**逻辑名**：``dedicated`` /
    ``topology`` / ``backend``）。非法标签抛
    ``UsageError``（退出码 2）并列出合法集合——静默忽略拼写错误会让用户以为
    强制生效了，而他其实还在跑 auto。
    """
    force = _kernel_force_module()
    if force is None:
        text = DEFAULT_KERNEL if value is None else str(value).strip()
        return () if text in ("", DEFAULT_KERNEL) else (text,)
    try:
        return tuple(force.resolve(value))
    except ValueError as exc:
        raise UsageError(translate("error.kernel", language, value=value,
                                   labels=", ".join(_kernel_label_names()))) from exc


def kernel_requested(value: str | None, language: str = "zh") -> str:
    """报告里 ``requested=`` 的取值：``auto`` / ``generic`` / ``label,...``。

    与真正传给 ``contract(kernel_force=...)`` 的口径同源（都从这里解析出来），所以这一格
    回答的是"这次到底要求了什么"，而不是"命令行上写了什么"——``auto`` 时它还会
    跟随 ``PYQCD_EINSUM_KERNEL_FORCE``，报告必须反映那一刻的生效值。
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


def _forced_of(labels: tuple[str, ...]) -> tuple[str, ...] | None:
    """``resolve`` 的元组 → ``last_kernel()['requested']`` 的形状。

    ``None`` 自动分派；``()`` 只走通用路径；其余为强制标签。自己算而不调
    ``_kernel_force.forced_labels``，是为了该模块缺席时也能校验记录归属。
    """
    if not labels:
        return None
    if labels == ("generic",):
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


def _profile_summary() -> dict[str, Any] | None:
    """读 ``_profile_cache.summary()`` 的原始快照；模块/函数/形态不符就给 ``None``。"""
    cache = _profile_module()
    probe = getattr(cache, "summary", None) if cache is not None else None
    if probe is None:
        return None
    try:
        snapshot = probe()
    except Exception:                           # noqa: BLE001 - 报告层不许影响主流程
        return None
    return snapshot if isinstance(snapshot, dict) else None


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

    ``profile=False``（或环境变量关闭）在第一个判断就返回，**不 configure、不
    load、不建目录**——与 ``_api._profile_begin`` 的零副作用口径一致。
    ``supported=False``（底层 ``contract()`` 没有 ``profile=``）时同样返回
    ``unknown``：参数都递不进去，就不能声称自己知道画像状态。
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
    try:
        if profile_dir:
            cache.configure(profile_dir)
        row = cache.best(key)
        summary = cache.summary()
    except Exception as exc:                    # noqa: BLE001
        state["last_error"] = f"{type(exc).__name__}: {exc}"
        return state
    if isinstance(summary, dict):
        state["rows"] = summary.get("rows")
        state["last_error"] = summary.get("last_error") or state["last_error"]
        if not summary.get("enabled", True):
            state["status"] = "off"
            return state
    if row is None:
        state["status"] = "miss"
    else:
        state["status"] = "hit"
        state["row"] = dict(row)
    return state


def _profile_rows() -> list[dict[str, str]]:
    """读画像 CSV 的原始行（**只读**，绝不写）。取不到就给空表。

    只用于 ``why`` 的第 ④ 类：把"缓存记录 backend 更快"写成**建议**。字段名来自
    ``_profile_cache.PROFILE_COLUMNS`` 那一套（``kernel``/``compute_ms``/...），
    这里按列名取值，不依赖列序。
    """
    import csv

    summary = _profile_summary()
    if not summary or not summary.get("enabled"):
        return []
    path = summary.get("file")
    if not path:
        return []
    try:
        with open(path, encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle) if row]
    except OSError:
        return []


def _ms_of(row: dict[str, Any]) -> float:
    """``compute_ms`` 列的数值（缺失/坏值按"无穷慢"处理，选优时永远排最后）。"""
    try:
        value = float(row.get("compute_ms") or "")
    except (TypeError, ValueError):
        return float("inf")
    return value if value == value else float("inf")


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
        return str(best_backend.get("compute_ms") or ""), None
    best_kernel = min(kernel_rows, key=_ms_of)
    if _ms_of(best_backend) >= _ms_of(best_kernel):
        return None
    return (str(best_backend.get("compute_ms") or ""),
            str(best_kernel.get("compute_ms") or ""))


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


def _profile_status_line(state: dict[str, Any], language: str = "zh") -> str:
    """之后每次那**一行**：命中 / 未命中已记录 / 关闭 / 未知。"""
    status = state.get("status")
    if status == "hit":
        row = state.get("row") or {}
        kernel = str(row.get("kernel") or "").strip().lower()
        # ``generic`` 不可重放：`_api._profile_force_label` 不会采纳它，这一行若还写
        # "命中 →" 就会让人以为换了路（用户实跑里正是这么误解的）。
        if kernel in ("", "generic"):
            return translate("explain.status_hit_inert", language,
                             kernel=row.get("kernel") or KERNEL_UNKNOWN)
        return translate("explain.status_hit", language,
                         backend=row.get("backend") or KERNEL_UNKNOWN,
                         kernel=row.get("kernel") or KERNEL_UNKNOWN,
                         compute_ms=row.get("compute_ms") or KERNEL_UNKNOWN)
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
def tune_combos(language: str = "zh") -> list[tuple[str, str, str]]:
    """``--tune`` 的完整网格：``(kernel_force, as_equivalence, optimize)`` 三元组。

    组合数超过 :data:`TUNE_MAX_COMBOS` 时抛 ``UsageError``（退出码 2），**绝不**截断
    ——截断后的"最优"不再是这张网格里的最优，而报告里看不出来。
    """
    combos = [(kernel, as_equiv, optimize)
              for kernel in TUNE_KERNEL_FORCE
              for as_equiv in TUNE_AS_EQUIVALENCE
              for optimize in TUNE_OPTIMIZERS]
    if len(combos) > TUNE_MAX_COMBOS:
        raise UsageError(translate("error.tune_grid", language,
                                   count=len(combos), limit=TUNE_MAX_COMBOS))
    return combos


def tune_conflicts(*, kernel_force: str | None = None,
                   as_equivalence: bool | None | str = None,
                   optimize: Any = None, language: str = "zh") -> list[str]:
    """``--tune`` 与"它自己会遍历的三维"撞车时：**每个参数打一行警告，然后忽略它**。

    为什么是警告而不是用法错误：``--tune`` 的价值就在于"别再让我手动挑内核/等价/
    收缩序"，用户顺手写上一维（尤其脚本里统一传 ``--kernel-force dedicated``）不该
    让整轮遍历以退出码 2 收场；而**悄悄**忽略又会让用户以为那一维真的生效了。折中：
    每个撞车的参数在 stderr 上给**一行**（``--tune 会自行遍历 …，已忽略你给的 '<值>'``），
    遍历照常进行、退出码仍为 0，报告里也能看到网格覆盖了哪些取值。

    ``kernel_force`` 只有解析出**具体标签**才算撞车：``auto``（以及 ``None``）与网格的
    第一档同义，显式写它不改变任何东西，也就没有可警告的。非法标签仍按原来的用法错误
    报（退出码 2）——那是拼写问题，不是"被忽略"。

    返回被忽略的三轴名（``["--kernel-force"]`` 之类），供调用方/测试断言。
    """
    ignored: list[str] = []
    if kernel_force is not None and kernel_labels(kernel_force, language):
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


def _tune_reference(equation: str, host: Sequence[Any]) -> tuple[Any, str]:
    """``--tune`` 的唯一参照：``np.einsum(optimize="greedy")``（与缺省 ``--compare numpy`` 同源）。

    只算一次、且不进任何一组的计时区：``--tune`` 把 ``compare`` 固定成 ``none``，
    为的就是别把 30 份参照耗时再算一遍。算不出来时返回 ``(None, 原因)``——此时所有
    组合的 ``rel`` 都是 ``None``，也就不做 MISMATCH 判定（如实标注，不猜）。
    """
    try:
        return np.einsum(equation, *host, optimize="greedy"), "np.einsum(optimize='greedy')"
    except Exception as exc:                    # noqa: BLE001 - 参照失败不该中断遍历
        return None, f"n/a ({type(exc).__name__}: {exc})"


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


def tune_best(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """最快且**可用**的那一组。

    ``error``（跑不成）、``timeout``（单次超上限）、``budget``（被整轮预算截断）与
    ``mismatch``（数值不符）一律不参选：它们的 ``median_ms`` 要么不存在、要么不来自
    一份完整且合格的样本集。
    """
    usable = [row for row in rows
              if row.get("median_ms") is not None
              and not row.get("error") and not row.get("mismatch")
              and not row.get("timeout") and not row.get("budget")]
    return min(usable, key=lambda row: row["median_ms"]) if usable else None


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

    装得上定时器就**中断**；装不上（非主线程 / 无 ``SIGALRM``）就退化成"先跑完、再按
    耗时判定"——仍然如实判超时，只是没能省下那段墙钟。预算比单次上限更紧时按预算截断，
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
    if not _tune_timer_available():
        started = time.perf_counter()
        call()
        if time.perf_counter() - started > seconds:
            raise error(f"单次调用耗时超过上限 {seconds:g}s（未能中断，已跑完）")
        return
    with _call_alarm(seconds, error):
        call()


def _profile_tune(on: bool) -> bool | None:
    """把"tune 期间强制记录"的开关设成 ``on``，返回**设置前**的取值。

    底层是 ``_api._PROFILE_TUNE``（``_api._profile_record`` 每次记录时**现读**它，
    所以模块级赋值就够了，与 :func:`_reset_last_kernel` 写 ``_api._LAST_KERNEL``
    同一手法）。库若提供 ``set_profile_tune()`` 就优先用它。返回 ``None`` = 这个开关
    根本不存在（旧版库）：tune 仍会跑，但记录退回稀疏口径，报告里照实标注。
    """
    try:
        from pyqcd.einsum import _api
    except ImportError:                         # pragma: no cover - 旧版库
        return None
    setter = getattr(_api, "set_profile_tune", None)
    if callable(setter):
        setter(bool(on))
        return not bool(on)                     # 没有读值接口：按二态恢复
    if hasattr(_api, "_PROFILE_TUNE"):
        previous = bool(getattr(_api, "_PROFILE_TUNE"))
        _api._PROFILE_TUNE = bool(on)
        return previous
    return None


class _ProfileRecordProbe:
    """截获 ``_profile_cache.record``，记下**这一组**到底把哪几条写进了画像。

    为什么必须截获而不是事后反推：``_api`` 记录的 ``kernel`` 是**具体标签**
    （``kupl_dedicated`` / ``backend`` / ``generic`` / ``equivalence``），而这一组请求
    的是逻辑名（``dedicated`` / ``auto`` …）。超时或数值不符时手上只剩一个结果，
    靠 ``last_kernel()`` 反推在 auto 分派与等价改写下会指错行；包装一层直接拿到实参，
    撤回时一一对得上。

    只在 :func:`run_tune` 期间生效（CLI 单线程路径），退出时**原样恢复**：``_api``
    是 ``import`` 模块对象后调 ``_profile_cache.record``，替换模块属性即可生效。
    """

    def __init__(self) -> None:
        self.cache = _profile_module()
        self.original = (getattr(self.cache, "record", None)
                         if self.cache is not None else None)
        self.rows: list[tuple[dict[str, Any], str, str]] = []
        self._active = False

    def __enter__(self) -> "_ProfileRecordProbe":
        if self.cache is None or not callable(self.original):
            return self          # 旧版库：没有可记的东西，也不假装记了

        def wrapper(row, *, force_write: bool = False):
            stored = self.original(row, force_write=force_write)
            label = str(row.get("kernel") or "").strip().lower()
            # 不可重放的标签（空 / equivalence）在 record 里**早退**、根本不进索引，
            # 收进来只会让撤回多报一条"没撤回"。
            if stored or label not in ("", "equivalence"):
                self.rows.append((dict(row), str(row.get("kernel") or ""),
                                  str(row.get("optimize") or "")))
            return stored

        self.cache.record = wrapper
        self._active = True
        return self

    def __exit__(self, *_exc_info) -> bool:
        if self._active and self.cache is not None:
            self.cache.record = self.original
            self._active = False
        return False

    def pairs(self) -> list[tuple[dict[str, Any], str, str]]:
        """去重后的 ``(键, kernel, optimize)`` 列表（保持首次出现次序）。"""
        seen: set[tuple[Any, ...]] = set()
        unique: list[tuple[dict[str, Any], str, str]] = []
        for key, kernel, optimize in self.rows:
            token = (key.get("host"), key.get("user_device"), key.get("backend"),
                     key.get("dtype"), key.get("equation"), key.get("shape"),
                     key.get("device"), kernel, optimize)
            if token in seen:
                continue
            seen.add(token)
            unique.append((key, kernel, optimize))
        return unique

    def configs(self) -> set[tuple[str, str]]:
        """本组写过的 ``(kernel, optimize)`` 集合。"""
        return {(kernel, optimize) for _key, kernel, optimize in self.rows}


def _withdraw_records(probe: _ProfileRecordProbe,
                      good: set[tuple[str, str]]) -> tuple[int, int]:
    """撤回本组写进画像的坏记录，返回 ``(撤掉, 未撤回)``。

    ``good`` 是**已成功且合格**的组合写过的 ``(kernel, optimize)``：同一个
    ``(键, kernel, optimize)`` 也可能由另一组写下来（``auto`` 分派跑成
    ``kupl_dedicated`` 与显式 ``dedicated`` 会落到同一行），那种行绝不能删——它背后
    有一次合格的实测。

    ``drop()`` 返回 ``False`` 有两种含义：本来就没有这一条，或盘上仍有历史行
    （``profile.csv`` 是追加写、没有删改接口）。这里都记作"未撤回"，由报告如实说明。
    """
    cache = probe.cache
    drop = getattr(cache, "drop", None) if cache is not None else None
    if not callable(drop):
        return 0, 0
    removed = kept = 0
    for key, kernel, optimize in probe.pairs():
        if (kernel, optimize) in good:
            continue
        try:
            ok = bool(drop(key, kernel, optimize))
        except Exception:                       # noqa: BLE001 - 撤回失败不该中断遍历
            ok = False
        if ok:
            removed += 1
        else:
            kept += 1
    return removed, kept


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
             language: str | None = None) -> dict[str, Any]:
    """遍历网格并返回 ``{"rows", "best", "profile", "flushed", ...}``（表格由 :func:`print_tune` 打）。

    **不变量**：方程、形状、dtype、设备、后端都只在这里取一次；每组只换
    ``kernel_force`` / ``as_equivalence`` / ``optimize``。``compare`` 不参与（调用方
    已固定 ``none``）。

    每组直接调 ``pe.contract()``（不走 :func:`run`）有两个理由：①``run`` 每组都会再
    算一遍参照与等价报告，30 组就是 30 份无用功；②画像观测本来就是 ``contract()``
    自己记的，调它才是"真的产生了观测"，不需要也不应该由这里手工写行。

    ``timeout`` 是**每一次** ``contract()`` 调用（warmup 与每一轮都算）的上限秒数，
    ``budget`` 是整轮上限（``0`` = 不限）。任一次调用超时 → 该组记 ``timeout``；中位
    只由全部成功且未超时的调用构成（``_timed`` 一抛出来就没有中位，不拿部分样本凑）。
    预算用尽 → 立刻停止遍历，返回值里 ``stopped``/``measured`` 给出"已测 N/M"。

    任一组合抛异常（含强制内核的 ``KernelUnavailableError``）只在该行记
    ``error=<类型>`` 并继续——一次遍历不该因为一个不可用的内核整段作废。超时、异常与
    数值不符（``rel > atol+rtol``）三类组合写进画像的记录都会被撤回（``drop()``）；
    记录在 tune 期间是**强制**的（``force_write``），否则"坏的没留下"就无从证明。
    """
    global _ALARM_WINDOW

    import pyqcd.einsum as pe

    lang = resolve_language(language)
    if warmup < 1:
        raise UsageError(translate("error.warmup_zero", lang))
    if rounds < 1:
        raise UsageError(translate("error.rounds", lang))
    if float(timeout) < 0.0:
        raise UsageError(translate("error.tune_timeout", lang, value=timeout))
    if float(budget) < 0.0:
        raise UsageError(translate("error.tune_budget", lang, value=budget))
    combos = tune_combos(lang)          # 超上限在构造任何张量之前就报错
    xp = _xp_for_device(device, lang)
    np_dtype = np.dtype(dtype)
    # 操作数只构造一次：30 组看到的必须是同一批数据，否则"哪组更快"混进了数据差异。
    operands, host = _make_operands(xp, shapes, np_dtype, seed)
    reference, reference_note = _tune_reference(equation, host)
    extra = dict(profile_kwargs or {})
    tolerance = float(atol) + float(rtol)
    # ``limit`` = 每一次 contract() 调用的上限；``deadline`` = 整轮的上限。
    # ``0``/负数都是"不设上限"（CLI 已校验非负，这里的 0 是合法取值）。
    limit = float(timeout) if float(timeout) > 0.0 else None
    deadline = (time.monotonic() + float(budget)) if float(budget) > 0.0 else None
    rows: list[dict[str, Any]] = []
    # 已成功且合格的组合写过哪些 (kernel, optimize)：坏组合撤回时不能误删这些行。
    good: set[tuple[str, str]] = set()
    measured = 0
    unwithdrawn = 0
    stopped = False
    # tune 期间记录是**强制**的（整张地图才有比较价值），坏组合随后由 drop() 撤回。
    # 这个开关是过程级状态，``finally`` 里原样还回去，别让后面的调用继续强制写盘。
    previous_tune = _profile_tune(True)
    try:
        for kernel, as_equiv, optimize in combos:
            if deadline is not None and time.monotonic() >= deadline:
                stopped = True
                break
            row: dict[str, Any] = {
                "kernel_force": kernel,
                "as_equivalence": as_equiv,
                "optimize": optimize,
                "median_ms": None,
                "route": KERNEL_UNKNOWN,
                "rel": None,
            }
            holder: list[Any] = []

            def call(kernel=kernel, as_equiv=as_equiv, optimize=optimize,
                     holder=holder):
                got = pe.contract(equation, *operands, backend=backend,
                                  optimize=optimize, specialize=specialize,
                                  kernel_force=kernel,
                                  as_equivalence=(as_equiv == "true"),
                                  cache=cache, **extra)
                # 只留**最近一次**结果：append 会把 warmup+rounds 份中间结果全留活，
                # 生产尺寸上那是把峰值内存乘上轮数。
                holder[:] = [got]
                return got

            _clear_kernel_info()
            _reset_last_kernel()
            probe = _ProfileRecordProbe()
            bad = False
            seconds: float | None = None
            # 记录本组到底往画像里写了哪几条：超时/异常/数值不符时按它撤回。
            with probe:
                try:
                    seconds = _timed(
                        lambda: _guarded_call(call, limit, deadline), warmup, rounds)
                except _TuneTimeout:
                    row["timeout"] = True
                    bad = True
                except _TuneBudget:
                    row["budget"] = True
                    bad = True
                    stopped = True
                except Exception as exc:        # noqa: BLE001 - 单组失败不中断遍历
                    row["error"] = type(exc).__name__
                    bad = True
            row["route"] = _tune_route(
                kernel_force_fields(_last_kernel(), kernel,
                                    kernel_labels(kernel, lang)),
                _kernel_info())
            if not bad and seconds is not None:
                row["median_ms"] = round(seconds * 1000.0, 6)
                measured += 1
                if reference is not None and holder:
                    row["rel"] = _relative_error(_to_host(holder[-1]), reference)
                    if not (row["rel"] <= tolerance):
                        row["mismatch"] = True
                        bad = True
            holder.clear()                      # 别把上一组的结果留到下一组
            if bad:
                # 超时 / 异常 / MISMATCH：这三类**不能**留在画像里。
                removed, kept = _withdraw_records(probe, good)
                row["profile_dropped"] = removed
                row["profile_kept"] = kept
                unwithdrawn += kept
            else:
                good |= probe.configs()
            rows.append(row)
            if stopped:                         # 预算用尽：不再开新的一组
                break
    finally:
        if previous_tune is not None:
            _profile_tune(previous_tune)
        # 闹钟窗口是模块级状态：单次调用若正好在收尾时被迟到的信号打断，那句"关窗口"
        # 可能少走一次。整轮结束时无条件清空，别把这个状态留给后面的人。
        _ALARM_WINDOW = None

    # 写回画像：``contract()`` 已经把每组的观测记进内存，这里只负责落盘并如实报告
    # 它自己的统计（rows/keys）。flush 失败/模块缺席都只降级，不影响本次结论。
    flushed: int | None = None
    cache_module = _profile_module()
    flush = getattr(cache_module, "flush", None) if cache_module is not None else None
    if callable(flush):
        try:
            flushed = int(flush())
        except Exception:                       # noqa: BLE001 - 报告层不许影响主流程
            flushed = None
    return {
        "rows": rows,
        "best": tune_best(rows),
        "profile": _profile_summary(),
        "profile_requested": "auto" if profile is None else (
            "true" if profile else "false"),
        "profile_enabled": _profile_effective(profile),
        "profile_forced": previous_tune is not None,
        "flushed": flushed,
        "reference": reference_note,
        "combos": len(combos),
        "measured": measured,
        "total": len(combos),
        "stopped": stopped,
        "skipped": len(combos) - measured,
        "unwithdrawn": unwithdrawn,
        "timeout": limit,
        "budget": float(budget),
        "timer": "sigalrm" if (limit is not None and _tune_timer_available())
                 else ("posthoc" if limit is not None else "off"),
        "warmup": warmup,
        "rounds": rounds,
        "compare": "none",
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
    text = ("{} median={:>11s}  kernel_force={:<9s}  as_equivalence={:<5s}  "
            "optimize={:<7s}  rel={:>9s}  route={}").format(
        marker, _tune_ms(row), row["kernel_force"], row["as_equivalence"],
        row["optimize"], _tune_rel(row), row.get("route") or KERNEL_UNKNOWN)
    if row.get("mismatch"):
        text += "  MISMATCH"
    if row.get("timeout"):
        text += "  timeout"
    if row.get("budget"):
        text += "  budget"
    if row.get("error"):
        text += f"  error={row['error']}"
    return text


def print_tune(tune: dict[str, Any], language: str = "zh",
               verbose: bool = DEFAULT_VERBOSE) -> None:
    """把 ``--tune`` 的表打到 **stderr**（既有输出行一字不动）。

    顺序：说明行 → 超时实现说明 → 遍历覆盖的取值 → 列名 → 分隔线（这几行是表头）
    → 最快 5 行 + 每个 ``kernel_force`` 取值的最快一行（``--verbose`` 才有）→
    被排除的组合计数 → 画像 rows/keys → 预算截断/撤回提示 → ``tune best :``
    （**最后一行**）。不加 ``--verbose`` 时只有表头、画像行与最优行，避免 30 行刷屏。
    """
    out = sys.stderr
    rows = list(tune.get("rows") or [])
    best = tune.get("best")
    print("tune      : " + translate(
        "tune.caption", language, combos=tune.get("combos"),
        warmup=tune.get("warmup"), rounds=tune.get("rounds"),
        reference=tune.get("reference")), file=out)
    # 超时是怎么实现的必须写在报告里：SIGALRM 可能打断原生内核，用户据此决定是否
    # 换个干净进程再跑。装不上定时器时同样要说，否则"没超时"会被误读成"够快"。
    timer = tune.get("timer")
    if timer == "sigalrm":
        print("tune note : " + translate("tune.signal_note", language), file=out)
    elif timer == "posthoc":
        print("tune note : " + translate("tune.posthoc_note", language), file=out)
    # 网格覆盖了哪些取值：用户显式写的 --kernel-force 等被忽略后，光看"最快 5 行"
    # 无法确认那一维到底遍历过没有，这一行把它钉死（机器可读，固定字段名）。
    axes = tune.get("axes") or {}
    print("tune axes : kernel_force={}  as_equivalence={}  optimize={}".format(
        ",".join(str(item) for item in (axes.get("kernel_force") or ())) or "-",
        ",".join(str(item) for item in (axes.get("as_equivalence") or ())) or "-",
        ",".join(str(item) for item in (axes.get("optimize") or ())) or "-"),
        file=out)
    print("tune hdr  : median_ms  kernel_force  as_equivalence  optimize  rel  "
          "route", file=out)
    print("tune ---- : " + "-" * 88, file=out)
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
        # 坏组合也要露脸（哪怕不带 --verbose 时无人可见）：只把它们打在这里，
        # 30 行全打会刷屏，但"哪个组合被排除、为什么"必须能查到。
        for row in rows:
            if row.get("error") or row.get("timeout") or row.get("budget"):
                print("tune row  : " + _tune_line(row, best), file=out)
    skipped = [row for row in rows if row.get("error") or row.get("mismatch")]
    if skipped or any(row.get("timeout") for row in rows):
        print("tune skip : " + translate(
            "tune.skip", language,
            mismatch=sum(1 for row in skipped if row.get("mismatch")),
            error=sum(1 for row in skipped if row.get("error")),
            timeout=sum(1 for row in rows if row.get("timeout"))), file=out)
    profile = tune.get("profile") or {}
    enabled = tune.get("profile_enabled")
    print("tune prof : profile={}  enabled={}  rows={}  keys={}  flushed={}{}".format(
        tune.get("profile_requested") or KERNEL_UNKNOWN,
        KERNEL_UNKNOWN if enabled is None else str(bool(enabled)).lower(),
        KERNEL_UNKNOWN if profile.get("rows") is None else profile["rows"],
        KERNEL_UNKNOWN if profile.get("keys") is None else profile["keys"],
        KERNEL_UNKNOWN if tune.get("flushed") is None else tune["flushed"],
        f"  last_error={profile['last_error']}" if profile.get("last_error") else ""),
        file=out)
    # 坏组合没能从磁盘上抹掉时如实说明：profile.csv 是追加写，历史行删不掉。
    if tune.get("unwithdrawn"):
        print("tune cache: " + translate("tune.history_kept", language,
                                        n=tune["unwithdrawn"]), file=out)
    # 预算用尽 → 必须报"已测 N/M"：不说清有多少组没测，"最优"就会被读成全局最优。
    if tune.get("stopped"):
        print("tune stop : " + translate(
            "tune.stop", language, measured=tune.get("measured"),
            total=tune.get("total"), skipped=tune.get("skipped"),
            budget=float(tune.get("budget") or 0.0)), file=out)
    if best is None:
        print("tune best : none  (" + translate("tune.none", language) + ")", file=out)
        return
    print("tune best : kernel_force={}  as_equivalence={}  optimize={}  "
          "median={:.6f}ms".format(best["kernel_force"], best["as_equivalence"],
                                   best["optimize"], best["median_ms"]), file=out)


# ── 语料反推（不写 --shape 时） ──────────────────────────────────────────────
def shapes_from_corpus(equation: str, scene: str
                       ) -> tuple[list[tuple[int, ...]], bool, dict[str, int]]:
    """按语料 ``roles`` + 生产 dims 反推形状。

    返回 ``(每个操作数的形状, 是否命中语料, label → 尺寸表)``。第三项让调用方能
    按方程输出段拼出**正确的输出形状**——少了它，省略 ``--shape`` 的非标量输出
    会被当成 ``()`` 而误判形状不符（见 :func:`main` 里的说明）。

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
    from pyqcd.einsum import _wick_tasklist as tasklist
    from pyqcd.einsum._dedicated_kernels import parse_equation
    from pyqcd.einsum.bench_specialized import _label_sizes

    table = _bench_scene.dims(scene)
    for seq in range(1, len(tasklist.by_seq()) + 1):
        task = tasklist.lookup(seq)
        if task is None or task.equation != equation:
            continue
        terms, _out = parse_equation(equation)
        sizes = _label_sizes(terms, task.roles, table)
        shapes = [tuple(sizes[ch] for ch in term) for term in terms]
        return shapes, True, {str(k): int(v) for k, v in sizes.items()}
    return [], False, {}


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
    parser.add_argument("--compare", default=DEFAULT_COMPARE, choices=COMPARE_MODES,
                        help=t("help.compare", default=DEFAULT_COMPARE))
    parser.add_argument("--compare-optimize", default=DEFAULT_COMPARE_OPTIMIZE,
                        choices=COMPARE_OPTIMIZERS,
                        help=t("help.compare_optimize",
                               default=DEFAULT_COMPARE_OPTIMIZE))
    parser.add_argument("--specialize", default=DEFAULT_SPECIALIZE,
                        choices=SPECIALIZE_MODES,
                        help=t("help.specialize", default=DEFAULT_SPECIALIZE))
    # ``default=None`` 是刻意的，不是 ``DEFAULT_KERNEL``：``_kernel_force`` 里
    # **显式参数优先于环境变量**，把缺省写成 ``"auto"`` 就等于每次都显式给了 auto，
    # ``PYQCD_EINSUM_KERNEL_FORCE`` 会被永久压住。``None`` 才代表"用户没给"。
    parser.add_argument("--kernel-force", default=None,
                        help=t("help.kernel", default=DEFAULT_KERNEL,
                               env=KERNEL_FORCE_ENV,
                               labels=", ".join(_kernel_label_names())))
    parser.add_argument("--as-equivalence", default=DEFAULT_AS_EQUIVALENCE,
                        choices=AS_EQUIVALENCE_MODES,
                        help=t("help.as_equivalence", env=AS_EQUIVALENCE_ENV,
                               default=DEFAULT_AS_EQUIVALENCE))
    # ``--tune`` 是开关：``nargs="?"`` 让 ``--tune``、``--tune true``、``--tune false``
    # 三种写法都成立（与 ``--verbose`` 同风格），缺省关闭。
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
    parser.add_argument("--optimize", default=DEFAULT_OPTIMIZE,
                        help=t("help.optimize", default=DEFAULT_OPTIMIZE))
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
    # ``--tune`` 自己会遍历这三轴：显式非 auto 取值只**警告 + 忽略**（每个参数一行），
    # 遍历照常进行、退出码仍为 0——顺手写上的一维不该让整轮 tune 变成用法错误。
    tune_on = _as_bool(args.tune)
    if tune_on:
        tune_conflicts(kernel_force=args.kernel_force,
                       as_equivalence=args.as_equivalence,
                       optimize=args.optimize, language=language)

    if args.shape is None:
        shapes, matched, label_sizes = shapes_from_corpus(args.equation, args.scene)
        if not matched:
            raise UsageError(translate("error.corpus_miss", language,
                                       equation=args.equation, scene=args.scene))
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
    try:
        result = run(
            equation=args.equation, shapes=shapes,
            out_shape=tuple(out_shape),
            device=args.device, dtype=args.dtype, backend=args.backend,
            atol=args.atol, rtol=args.rtol, compare=args.compare,
            optimize=args.optimize, rounds=args.rounds, warmup=args.warmup,
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
    result["shape_source"] = report_note
    if result.get("specialize_warning"):
        print(translate("result.warning", language,
                        message=result["specialize_warning"]), file=sys.stderr)
    if result.get("compare_optimize") not in (None, DEFAULT_COMPARE_OPTIMIZE) and \
            result.get("compare_optimize_effective") is None:
        print(translate("result.warning", language, message=translate(
            "result.warning_compare_optimize_inert", language,
            value=result["compare_optimize"], compare=args.compare)),
            file=sys.stderr)

    # ``--tune``：在主路径既有输出之后**追加**遍历（表格走 stderr，既有行不动）。
    # 放在 ``--json`` 之前，``report['tune']`` 才会一起落进 JSON。
    if tune_on:
        tune = run_tune(
            equation=args.equation, shapes=shapes,
            device=args.device, dtype=args.dtype, backend=args.backend,
            atol=args.atol, rtol=args.rtol, rounds=args.rounds,
            warmup=args.warmup, seed=args.seed, specialize=specialize_value,
            cache=cache_value, profile=profile_value,
            profile_kwargs=_contract_profile_kwargs(profile_value, args.profile_dir),
            timeout=args.tune_timeout, budget=args.tune_budget,
            language=language,
        )
        result["tune"] = tune["rows"]
        print_tune(tune, language, verbose=_as_bool(args.verbose))

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
        if peak.get("gpu_used_mib") is not None:
            detail += f"  vram+{peak['gpu_used_mib']:.0f}MiB"
        elif peak.get("rss_mib") is not None:
            detail += f"  rss+{peak['rss_mib']:.0f}MiB"
    print(f"PASS {args.equation}{detail}")
    return 0


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
