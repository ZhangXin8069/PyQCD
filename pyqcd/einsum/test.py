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
* **强制内核**：``--kernel-force auto|generic|<label>[,...]``。``auto``（缺省）保持原
  分派次序，失败只 ``warning`` + 回落；``generic`` 完全跳过专用内核；具体标签
  （``kupl_*`` / ``cuda_*``，可逗号分隔多个、按书写次序尝试）强制**真试一次**，
  跑不成就抛 ``KernelUnavailableError`` 并以非零退出码结束——**绝不**静默回落。
  显式 ``--kernel-force`` 压过环境变量 ``PYQCD_EINSUM_KERNEL_FORCE``。明细里的
  ``kernel :`` 行给出 ``requested`` / ``ran`` / ``attempted``。
* **缓存**：``--cache auto|true|false``。``auto``（缺省）跟随环境变量
  ``PYQCD_EINSUM_CACHE``（只有 ``0/false/off/no`` 才关闭）；显式 ``true``/``false``
  压过环境变量，后者让底层**既不读也不写**任何 einsum 缓存（不留半热状态、不污染
  统计）。显式给出 ``true``/``false`` 时，``--verbose true`` 的报告**追加**一行
  ``cache    : enabled=... hits=... misses=... hit_rate=...``；``auto`` 与省略该
  选项等价，不写这一行——既有输出逐字节不变。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
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
           "cache_enabled", "cache_line"]

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
#: ``--as-equivalence``：三态。``auto`` 跟随 ``PYQCD_EINSUM_AS_EQUIVALENCE``（缺省开启）。
#: 与 ``--equivalence`` 不同，它**真的改变执行路径**：开启时命中映射就改写方程跑，
#: 再把结果转回原输出轴序（返回数组仍是原方程的答案）。
AS_EQUIVALENCE_MODES = ("auto", "true", "false")
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
        "en": ("Which specialised kernel may run. Three kinds of value: auto = keep "
               "the original dispatch (dedicated, then topology) and, when none runs, "
               "only warn and fall back to the generic path (default); generic = skip "
               "the specialised kernels entirely (no warning); <label> = force exactly "
               "these ({labels}, comma separated, tried in the order written). "
               "An explicit --kernel-force wins over the {env} environment variable. "
               "Forcing a label makes the kernel really try once (its negative cache "
               "is bypassed) and, when it does not run, raises KernelUnavailableError "
               "with a non-zero exit code — it never falls back to the generic path. "
               "Main path only: the --compare backend reference keeps its own dispatch "
               "(forcing a label on the other backend cannot work). A kernel that is "
               "not available is reported as an error, never as a warning. An unknown "
               "label exits with code 2. Default {default}"),
        "zh": ("允许跑哪个专用内核。三种取值：auto = 保持原分派次序（dedicated → "
               "topology），跑不成只 warning 后回落到通用路径（缺省）；generic = "
               "完全跳过专用内核（不产生 warning）；<label> = 强制只跑这些"
               "（{labels}，可用逗号给多个、按书写次序尝试）。"
               "显式 --kernel-force 优先于环境变量 {env}。"
               "强制具体内核时会**真试一次**（旁路负缓存），跑不成即抛 "
               "KernelUnavailableError 并以非零退出码结束——绝不回落成通用路径。"
               "只作用于主路径：--compare backend 的参照仍走自己的分派"
               "（在另一个后端上强制标签必然不可用）。"
               "强制跑不成一律报错，不会降级成 warning。未知标签以退出码 2 结束。"
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
        "en": ("Apply an equivalent spelling automatically: auto=follow {env} "
               "(on by default, only 0/false/off/no turns it off); true=rewrite when a "
               "mapping exists; false=never. When on and the equation misses every "
               "specialised path, the operands are permuted, the equivalent spelling is "
               "contracted, and the result is transposed back — the returned array is "
               "still the answer to the ORIGINAL equation (same shape, same dtype). "
               "Skipped for out=, extra kwargs, element-wise equations and explicit "
               "optimizer paths. Default {default}"),
        "zh": ("自动施加等价写法：auto=跟随 {env}（缺省开启，只有 0/false/off/no 才关闭）；"
               "true=存在映射就改写；false=不改写。开启且方程未命中任何特化路径时，"
               "会重排操作数、按等价写法收缩、再把结果转回原输出轴序——"
               "返回的数组仍是**原方程**的答案（形状与 dtype 都不变）。"
               "out=、额外 kwargs、逐元素方程与显式优化器路径不施加。缺省 {default}"),
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
        "en": ("--kernel-force must be 'auto', 'generic', or one of {labels} "
               "(comma separated for several), got {value!r}"),
        "zh": ("--kernel-force 必须是 'auto'、'generic' 或 {labels} 之一"
               "（多个用逗号分隔），得到 {value!r}"),
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
                           cache=cache_requested)

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
#: 可强制的内核标签（与 ``_kernel_force.KERNEL_LABELS`` 同集合）。写字面量是为了
#: 该模块缺席时 ``--help`` 仍能列出合法集合。
KERNEL_LABELS = frozenset({
    "cuda_dedicated",
    "cuda_topology",
    "kupl_dedicated",
    "kupl_topology",
})
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
    """合法标签集合：优先用 ``_kernel_force`` 的权威定义，取不到用本模块字面量。"""
    force = _kernel_force_module()
    live = getattr(force, "KERNEL_LABELS", None) if force is not None else None
    return tuple(sorted(live or KERNEL_LABELS))


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
    ``("generic",)`` = 只走通用路径，其余为要强制的标签）。非法标签抛
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
                         language: str = "zh") -> bool | None:
    """把 ``auto|true|false``（或 ``None``/``bool``）规范成 ``bool | None``。

    与 :func:`parse_equivalence` 同口径：``None`` 与 ``auto`` 等价（跟随
    ``PYQCD_EINSUM_AS_EQUIVALENCE``），本函数**不读**环境变量——真正是否施加由
    :func:`pyqcd.einsum._equivalence.as_equivalence_enabled` 在调用时决定。
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
        route = _route_of(report.get("kernel_info"), language)
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


def _route_of(kernel_info: Any, language: str = "zh") -> str:
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
    """
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

    try:
        result = run(
            equation=args.equation, shapes=shapes,
            out_shape=tuple(out_shape),
            device=args.device, dtype=args.dtype, backend=args.backend,
            atol=args.atol, rtol=args.rtol, compare=args.compare,
            optimize=args.optimize, rounds=args.rounds, warmup=args.warmup,
            seed=args.seed, verbose=_as_bool(args.verbose),
            specialize=parse_specialize(args.specialize, language),
            compare_optimize=args.compare_optimize,
            equivalence=parse_equivalence(args.equivalence, language),
            as_equivalence=args.as_equivalence,
            kernel_force=args.kernel_force,
            cache=parse_cache(args.cache, language),
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
