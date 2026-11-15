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
           "DEFAULT_LANGUAGE"]

class UsageError(SystemExit):
    """参数或环境错误。

    继承 ``SystemExit`` 以便顶层直接终止，但把退出码固定成 **2**，与 ``--help`` 里
    写的"0 通过 / 1 数值或形状不符 / 2 参数或环境错误"一致。裸
    ``SystemExit(str)`` 的退出码是 1，会与"数值失败"混淆——脚本据此判断是
    "结果不对"还是"命令写错了"。
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
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

DEFAULT_DEVICE = "cpu"
DEFAULT_DTYPE = "complex128"
DEFAULT_BACKEND = "auto"
DEFAULT_SPECIALIZE = "auto"
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
def parse_shape_spec(spec: str, *, language: str = "zh"
                     ) -> tuple[list[tuple[int, ...]], tuple[int, ...]]:
    """把 ``'[..],[..]->[..]'`` 解析成 (输入形状列表, 输出形状)。

    只接受你给的这种纯数字形态；任何解析失败都**直接报错**，不猜。
    也允许省略 ``->…``（此时输出形状为 ``()``，表示整式归约）。

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
        out_shape = outputs[0]
    else:
        out_shape = ()
    return inputs, out_shape


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
    """操作数个数、秩、同标签等长——任一不符直接报错。"""
    if "->" not in equation:
        raise UsageError(translate("error.equation_needs_arrow", language))
    lhs, rhs = equation.split("->", 1)
    terms = lhs.split(",")
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
        expected_shape: tuple[int, ...] | None = None,
        language: str | None = None) -> dict[str, Any]:
    """跑一条方程的完整检查，返回结构化结果（供 CLI 与外部脚本消费）。

    ``specialize`` 三态直传 ``contract(specialize=...)``：``None`` 跟随环境变量
    ``PYQCD_EINSUM_SPECIALIZE``，``False`` 强制通用路径，``True`` 强制特化路径。

    ``compare_optimize`` **只影响参照实现**（``--compare numpy/opt_einsum``），
    主路径的收缩序仍由 ``optimize`` 决定。缺省 ``auto`` = 沿用旧行为。

    ``language`` 只影响错误文案与 ``verbose`` 明细（``None`` = 走
    :func:`resolve_language`，即 ``--language auto``），不改任何计算结果。
    """
    import pyqcd.einsum as pe

    lang = resolve_language(language)
    if dtype not in DTYPES:
        raise UsageError(translate("error.dtype", lang, values=DTYPES, value=dtype))
    if compare not in COMPARE_MODES:
        raise UsageError(translate("error.compare", lang, values=COMPARE_MODES,
                                   value=compare))
    specialize = parse_specialize(specialize)
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
        "n_operands": len(shapes),
        "input_elements": int(sum(int(np.prod(item, dtype=np.int64)) for item in shapes)),
        "output_elements": int(np.prod(out_shape, dtype=np.int64)) if out_shape else 1,
    }

    # 主路径
    def call():
        return pe.contract(equation, *operands, backend=backend, optimize=optimize,
                           specialize=specialize)

    _clear_kernel_info()
    try:
        got = call()
    except Exception as exc:                    # noqa: BLE001
        report["ok"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report
    seconds = _timed(call, warmup, rounds)
    got_host = got if xp is np else _to_host(got)
    report["seconds"] = seconds
    report["actual_shape"] = list(np.shape(got_host))
    report["kernel_info"] = _kernel_info()
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
            rounds, warmup, device, specialize, ref_optimize, language=lang)
        report["reference"] = ref_note
        report["reference_seconds"] = ref_seconds
        if reference is None:
            report["ok"] = False
            report["error"] = translate("reference.unavailable", lang,
                                        compare=compare, note=ref_note)
            return report
        report["rel"] = _relative_error(got_host, reference)
        report["max_abs_diff"] = float(np.max(
            np.abs(np.asarray(got_host) - np.asarray(reference)))) if np.size(got_host) else 0.0
        report["ok"] = bool(
            report["actual_shape"] == list(out_shape)
            and report["rel"] <= atol + rtol * report["rel"])
        if report["rel"] != report["rel"]:      # NaN
            report["ok"] = False
    else:
        report["ok"] = bool(report["actual_shape"] == list(out_shape))

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
               ref_optimize=None, language: str = "zh"):
    """按 ``--compare`` 取参照结果，返回 ``(数组或 None, 秒数或 None, 说明)``。

    ``ref_optimize`` 是 ``resolve_compare_optimize`` 解析出的**参照实际用的**优化器
    （``None`` = 该模式不设优化器）。本函数刻意**不读主路径的 ``optimize``**——参照
    借用主路径的选择会让对照实验失去独立性。

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
            return pe_contract(equation, other_ops, other, specialize)
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


def pe_contract(equation, operands, backend, specialize=None):
    """``compare='backend'`` 用：在指定后端上跑同一批张量。

    ``specialize`` 与主路径同源——否则 ``--compare backend`` 会把"专用路径 vs 通用
    路径"当成跨后端差异来比较，结论失真。
    """
    import pyqcd.einsum as pe
    return pe.contract(equation, *operands, backend=backend, specialize=specialize)


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
        shapes, declared_out = parse_shape_spec(args.shape, language=language)
        report_note = translate("note.shape_from_flag", language)
        out_shape = declared_out

    result = run(
        equation=args.equation, shapes=shapes,
        out_shape=tuple(out_shape),
        device=args.device, dtype=args.dtype, backend=args.backend,
        atol=args.atol, rtol=args.rtol, compare=args.compare,
        optimize=args.optimize, rounds=args.rounds, warmup=args.warmup,
        seed=args.seed, verbose=_as_bool(args.verbose),
        specialize=parse_specialize(args.specialize, language),
        compare_optimize=args.compare_optimize,
        language=language,
    )
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
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1, ensure_ascii=False)
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
    raise SystemExit(main())
