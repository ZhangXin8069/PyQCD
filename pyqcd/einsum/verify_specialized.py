"""特化路径查询：给定一个 einsum 方程，报告它在语料中的归属与执行契约。

只做**查询**，不做数值计算：不生成操作数、不调用 ``contract()``、不改任何
全局状态。因此它既能当 CLI 用，也能在生产代码里安全调用。

    python -m pyqcd.einsum.verify_specialized "agbh,cmdn,eqfr,ospt,ikjl,ce,mo,gi,kq,bdf,np,hj,lrt->as"
    python -m pyqcd.einsum.verify_specialized "<eq>" --json
    python -m pyqcd.einsum.verify_specialized "<eq>" --language en

判据（任一不满足即非零退出）：

1. **命中**：先查形状特化路径（``_shape_paths``，如 ``x,ax,bx,cx->abc``），
   再查语料索引（``_specialize.match``，8638 条 wick 拓扑）；
2. **可解析**：``canonical_signature`` 非空；
3. **未命中**时退出码为 1，但输出仍说明这是语料之外的方程、会走通用路径。

字段口径：

* ``name``：与运行时提示里打印的 ``topology:<摘要前 8 位>`` 完全一致；
* ``digest``：完整 96 位摘要（``_specialize._signature_digest``，blake2b-128 截断）；
* ``kernels``：该形态**声明**可用的原生内核标签。声明不等于实际执行——
  实际下发受 ``out=``、dtype、原生库可用性影响，判据见
  ``skills/pyqcd-einsum/SKILL.md`` 的「命中/回落判据」；
* ``corpus``：语料里同签名的代表写法；语料外的方程没有这一段；
* ``equivalence``：未命中且等价映射开启时，``_equivalence.explain()`` 的原始字典
  （``rewritten`` 为假也照写，与 ``test.py`` 的 ``equivalence`` 字段同口径）；
  关闭时**不出现**该字段；
* ``reason``：**历史中文文案**（逐字保留，等价于对 ``reason_key`` 取 zh 渲染）；
* ``reason_key`` / ``reason_args``：语言无关的稳定形式。``--json`` 的消费者要
  本地化就直接查本模块的 :data:`MESSAGES`，不要把 ``reason`` 当机器字段解析。

数值验证（结果是否正确、是否真跑原生内核）需要实际执行，属于测试侧职责：
``python -m pyqcd.testing.einsum.verify_specialized``。本模块刻意不重复那条路径，
因为生产代码不得 import ``pyqcd.testing``。

输出语言（``--language {auto,en,zh-cn}``，缺省 ``auto``）：只影响人读的文案，
不改本模块任何判定、退出码或 ``--json`` 的结构化字段。``auto`` **只在**
``PYQCD_EINSUM_LANG`` 显式设了时跟随它（``en*``→英文，``zh*``→中文），未设时保持
历史行为中文；它不跟随系统 locale（本机 ``LANG=C.UTF-8``，跟随 locale 会让不传
选项的输出从中文静默变成英文）。与 ``_specialize.notice_language()`` 共用同一个
环境变量名，不另立开关。

退出码：``0`` 命中语料；``1`` 语料之外（走通用路径）；``2`` 参数或方程非法。

等价映射（``--equivalence {auto,true,false}``，缺省 ``auto``）：``auto`` 跟随
``PYQCD_EINSUM_EQUIVALENCE``（未设即开启），``true``/``false`` 强制开关并**压过**
环境变量。开启时，未命中的方程会多给一段「等价写法 + 张量调整（输入侧）+
结果调整（输出侧）」——两侧**必须**同时在场，只按输入侧改写会拿到轴序错了的结果；
关闭时该段完全消失，MISS 的 ``原因`` 回到引入本功能之前的原文。它只是**附加信息**：
不改 ``matched``、不改退出码、不改任何执行路径。

强制内核（``--kernel-force {generic,dedicated,topology,backend}``，缺省 ``generic``）：
取值语义与 ``python -m pyqcd.einsum.test --kernel-force`` 完全一致——``generic`` 跳过专用
内核、``dedicated`` / ``topology`` 只允许**当前后端的**那一个（numpy → ``kupl_*``，cupy →
``cuda_*``）、``backend`` 直接调后端自己的 ``einsum``。只接受**单个**取值，逗号链式已取消；
显式取值压过 ``PYQCD_EINSUM_KERNEL_FORCE``；非法取值退出码 ``2`` 并列出合法集合。

**执行边界**：本模块是纯查询（不生成操作数、不调 ``contract()``），所以它**不跑**
任何内核，强制失败也不可能在这里发生——``--kernel-force`` 只决定报告里出不出 ``kernel :``
那一行，``ran`` / ``attempted`` 只读 ``_api.last_kernel()``，本进程没跑过就如实写
``unknown``（不猜）。不给 ``--kernel-force`` 时该行**不出现**，缺省输出与改造前逐字节相同。
真要强制跑内核请用 ``python -m pyqcd.einsum.test --kernel-force <label>``（那条路径失败时
打印 ``KernelUnavailableError`` 正文并非零退出）。

**详解（``--explain {auto,true,false}``，缺省 ``auto``）**：本模块不收缩，所以它给不出
"实测"的放弃原因，只把**会怎样**写进报告——``plan``（命中了哪个形态 / 声明了哪些内核 /
存在哪条等价写法）、``kernel``（``requested`` / ``attempted`` / ``ran``，只认本进程真跑过
的记录）、``why``（有记录时按 ``kept`` / ``contiguity`` / ``dtype-policy`` /
``kernel-unavailable`` 归类，没记录时如实写 ``not-run``）。``auto`` 等于关闭：缺省输出
与改造前**逐字节相同**（``auto`` 不跟随任何环境变量，本模块也没有 ``--verbose``）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Sequence

from . import _specialize

__all__ = [
    "query",
    "translate",
    "resolve_language",
    "resolve_equivalence",
    "resolve_kernel",
    "resolve_explain",
    "kernel_requested",
    "kernel_force_section",
    "last_kernel",
    "render_kernel_force",
    "render_explain_kernel",
    "build_parser",
    "MESSAGES",
    "LANGUAGE_MODES",
    "DEFAULT_LANGUAGE",
    "LANGUAGE_ENV",
    "EQUIVALENCE_MODES",
    "DEFAULT_EQUIVALENCE",
    "EQUIVALENCE_ENV",
    "EXPLAIN_MODES",
    "DEFAULT_EXPLAIN",
    "DEFAULT_KERNEL",
    "KERNEL_LABELS",
    "KERNEL_FORCE_ENV",
]

#: ``--language``：输出语言。``auto`` 见 :func:`resolve_language`。
LANGUAGE_MODES = ("auto", "en", "zh-cn")
DEFAULT_LANGUAGE = "auto"
#: 与 ``_specialize.LANGUAGE_ENV`` 同一个环境变量名——库内只有一套语言开关。
LANGUAGE_ENV = _specialize.LANGUAGE_ENV

#: ``--equivalence``：等价映射三态。``auto`` 见 :func:`resolve_equivalence`。
EQUIVALENCE_MODES = ("auto", "true", "false")
DEFAULT_EQUIVALENCE = "auto"
#: 与 ``_equivalence.EQUIVALENCE_ENV`` 同一个环境变量名——库内只有一套等价开关。
#: 这里写字面量而不是 import：``_equivalence`` 是**延迟导入**的（只在真正要查等价
#: 写法时才碰它），契约用例钉住两者相等。
EQUIVALENCE_ENV = "PYQCD_EINSUM_EQUIVALENCE"

#: ``--kernel-force``：缺省 ``generic``（与 ``_kernel_force.DEFAULT_LABEL`` 同值）。
#: 没有"未指定 → 自动分派"的档：自动选路交给 ``--tune`` 的实测结论。
DEFAULT_KERNEL = "generic"
#: 可强制的内核标签（与 ``_kernel_force.KERNEL_LABELS`` 同集合：**逻辑名**，运行时按
#: 操作数后端解析成 ``*_dedicated`` / ``*_topology``）。
KERNEL_LABELS = frozenset({
    "dedicated",
    "topology",
})
#: 不产生强制、只切换路线的两个取值（与 ``_kernel_force`` 同名）。
GENERIC_LABEL = "generic"
BACKEND_LABEL = "backend"
#: 与 ``_kernel_force.KERNEL_FORCE_ENV`` 同名的环境变量。
KERNEL_FORCE_ENV = "PYQCD_EINSUM_KERNEL_FORCE"
#: ``last_kernel()`` 取不到时的占位：说的是"没观测到"，不是一句说明文字。
KERNEL_UNKNOWN = "unknown"

#: ``--explain``：三态。``auto``（缺省）**等于关闭**——本模块没有 ``--verbose``，
#: 缺省输出必须与改造前逐字节相同；``true`` 才把"会怎样"写进报告。
EXPLAIN_MODES = ("auto", "true", "false")
DEFAULT_EXPLAIN = "auto"

#: 文案表：``key -> {"en": ..., "zh": ...}``。缺 key 或缺语言**直接报错**，
#: 不静默返回 key 本身——那样会以"看起来很正常的英文"混进中文输出里。
#:
#: ``field.*`` 的文案**自带对齐空格**（冒号前统一到 12 列，全角字符算 2 列），
#: 这样 zh 分支与改造前的输出逐字节相同，不必在代码里做宽度计算。
MESSAGES: dict[str, dict[str, str]] = {
    # --help
    "help.description": {
        "en": "Report whether an einsum equation hits a corpus-specialised path, "
              "and print its execution contract",
        "zh": "查询一条 einsum 方程是否命中语料特化路径，并给出执行契约",
    },
    "help.equation": {
        "en": 'einsum equation, e.g. "agbh,cmdn,...,lrt->as"',
        "zh": 'einsum 方程，例如 "agbh,cmdn,...,lrt->as"',
    },
    "help.json": {
        "en": "print a machine-readable report (never affected by --language)",
        "zh": "输出机器可读结果（不受 --language 影响）",
    },
    "help.language": {
        "en": ("output language: auto / en / zh-cn.\n"
               "auto follows {env}; when it is unset the report stays Chinese "
               "(it never follows the system locale).\n"
               "Translation only: it changes no verdict, no exit code, no "
               "--json field, and no environment variable.\n"
               "Default {default}"),
        "zh": ("输出语言：auto / en / zh-cn。\n"
               "auto 跟随 {env}；未设时报告保持中文（不跟随系统 locale）。\n"
               "只影响文案：不改判定、不改退出码、不改 --json 字段、不写环境变量。\n"
               "缺省 {default}"),
    },
    "help.equivalence": {
        "en": ("equivalence report: auto / true / false.\n"
               "auto follows {env} (unset = on); true/false force it and "
               "override that variable.\n"
               "A MISS then also lists the equivalent spelling plus the input-side "
               "tensor adjustments\n"
               "and the output-side result adjustment (both lines always shown; "
               "'-' when a side needs nothing).\n"
               "false removes that section and restores the pre-equivalence "
               "wording of the reason.\n"
               "Report only: it changes no verdict and no exit code.\n"
               "Default {default}"),
        "zh": ("等价映射：auto / true / false。\n"
               "auto 跟随 {env}（未设即开启）；true/false 强制开关并压过该变量。\n"
               "开启时未命中的方程会多给「等价写法 + 输入侧张量调整 + 输出侧结果"
               "调整」\n"
               "（两行恒在，某一侧无需调整时写 -）。\n"
               "false 时该段完全不出现，原因回到引入等价映射之前的原文。\n"
               "只影响报告：不改判定、不改退出码。\n"
               "缺省 {default}"),
    },
    "help.explain": {
        "en": ("explain what WOULD happen: auto / true / false.\n"
               "auto (default) is off — this module has no --verbose, so the default "
               "report stays byte-identical;\n"
               "true adds three lines to the report: plan (which form is hit, or which "
               "equivalent spelling exists),\n"
               "kernel (requested / attempted / ran — read from a real run in this "
               "process only) and why\n"
               "(kept / contiguity / dtype-policy / kernel-unavailable when a record "
               "exists, otherwise not-run).\n"
               "This module never contracts: every value here is a plan or a recorded "
               "observation, never a\n"
               "measurement of this query. It changes no verdict and no exit code. "
               "Default {default}"),
        "zh": ("报告「会怎样」：auto / true / false。\n"
               "auto（缺省）等于关闭——本模块没有 --verbose，缺省报告必须与改造前"
               "逐字节相同；\n"
               "true 时报告多三行：plan（命中了哪个形态，或存在哪条等价写法）、"
               "kernel（requested / attempted /\n"
               "ran，只认本进程真跑过的记录）、why（有记录时按 kept / contiguity / "
               "dtype-policy /\n"
               "kernel-unavailable 归类，没记录时如实写 not-run）。\n"
               "本模块不做任何收缩：这里的每个值都是「计划」或「记录值」，不是本次"
               "查询的实测。不改判定、不改退出码。缺省 {default}"),
    },
    "help.epilog": {
        "en": ("Exit codes:\n"
               "  0  hit a specialised (corpus/shape) path\n"
               "  1  not hit — the generic step-by-step path runs\n"
               "  2  equation unparsable, or bad arguments\n"
               "\n"
               "Declared kernels only mean the contract allows them; whether they "
               "actually run depends\n"
               "on out=, dtype and native-library availability.\n"
               "This module never contracts, so --kernel-force only records what was "
               "requested; use\n"
               "python -m pyqcd.einsum.test --kernel-force <value> to actually force one.\n"
               "Numeric verification: python -m pyqcd.testing.einsum.verify_specialized"),
        "zh": ("退出码：\n"
               "  0  命中特化路径（语料 / 形状特化）\n"
               "  1  未命中——走通用分步路径\n"
               "  2  方程无法解析，或参数非法\n"
               "\n"
               "「声明内核」只表示契约允许；实际是否下发取决于 out=、dtype、"
               "原生库可用性。\n"
               "本模块不做任何收缩，所以 --kernel-force 只记录「要求了什么」；"
               "真要强制跑内核请用\n"
               "python -m pyqcd.einsum.test --kernel-force <value>。\n"
               "数值验证：python -m pyqcd.testing.einsum.verify_specialized"),
    },
    "help.kernel": {
        "en": ("which specialised kernel is requested: generic / dedicated / topology / "
               "backend. "
               "generic skips the specialised kernels entirely; "
               "dedicated / topology allow exactly that kernel of the operand backend "
               "(numpy -> kupl, cupy -> cuda) and report a failure instead of falling "
               "back, never a warning; backend calls the backend's own einsum and skips "
               "the specialised dispatch. Exactly one label, never several. Same "
               "meaning as test.py --kernel-force; the explicit value wins over the "
               "{env} environment variable, and an unknown value exits with code 2. "
               "This module never contracts, so it cannot run a kernel: the kernel line "
               "shows requested= from this value, and ran=/attempted= only when a "
               "kernel actually ran in this process (otherwise 'unknown', never a "
               "guess). Omit the option and that line is absent, so the default report "
               "stays byte-identical. To really force one use "
               "python -m pyqcd.einsum.test --kernel-force <value>. Default {default}"),
        "zh": ("要求哪个专用内核：generic / dedicated / topology / backend。"
               "generic 完全跳过专用内核；dedicated / topology 只允许当前后端的对应内核"
               "（numpy → kupl，cupy → cuda），跑不成即报错而不回落，不会降级成 "
               "warning；backend 直接调后端自己的 einsum、不走专用分派。"
               "只接受单个取值，逗号链式已取消。"
               "与 test.py --kernel-force 同义；显式取值优先于环境变量 {env}，未知取值"
               "退出码 2。"
               "本模块不做任何收缩、跑不了内核：kernel 行里 requested= 来自该取值，"
               "ran=/attempted= 只有本进程真的跑过内核时才给"
               "（否则写 unknown，绝不猜）。不给该选项时整行不出现，"
               "缺省报告保持逐字节不变。真要强制跑内核请用 "
               "python -m pyqcd.einsum.test --kernel-force <value>。缺省 {default}"),
    },
    # 字段标签（自带对齐空格，冒号前 12 列）
    "field.equation": {"en": "equation    ", "zh": "方程        "},
    "field.normalized": {"en": "normalized  ", "zh": "归一化      "},
    "field.result": {"en": "result      ", "zh": "结果        "},
    "field.reason": {"en": "  reason    ", "zh": "  原因      "},
    "field.signature": {"en": "topology    ", "zh": "拓扑签名    "},
    "field.arity_slots": {"en": "arity/slots ", "zh": "元数/槽位   "},
    "field.digest": {"en": "digest      ", "zh": "摘要        "},
    "field.specialize": {"en": "specialise  ", "zh": "特化开关    "},
    "field.path": {"en": "  path      ", "zh": "  特化路径  "},
    "field.kernels": {"en": "  kernels   ", "zh": "  声明内核  "},
    "field.corpus": {"en": "  corpus    ", "zh": "  语料出处  "},
    "field.canonical": {"en": "  spelling  ", "zh": "  代表写法  "},
    "field.roles": {"en": "  roles     ", "zh": "  操作数角色"},
    "field.caveat": {"en": "  caveat    ", "zh": "  说明      "},
    "field.rewrite": {"en": "  rewrite   ", "zh": "  等价写法  "},
    "field.rewrite_kind": {"en": "  kind      ", "zh": "  变换类型  "},
    "field.rewrite_detail": {"en": "  detail    ", "zh": "  变换说明  "},
    "field.operand_perms": {"en": "  operands  ", "zh": "  张量调整  "},
    "field.operand_order": {"en": "  ordering  ", "zh": "  张量次序  "},
    "field.output_perm": {"en": "  output    ", "zh": "  结果调整  "},
    "field.hint": {"en": "  hint      ", "zh": "  提示      "},
    # 强制内核：``requested`` / ``ran`` / ``attempted`` 三段恒在同一行。
    # 标签自带对齐空格（冒号前 12 列），与其余 ``field.*`` 同宽。
    "field.kernel_force": {"en": "kernel      ", "zh": "强制内核    "},
    # 详解（--explain true）：本模块只报"会怎样"，不执行收缩。
    "field.explain_plan": {"en": "explain/plan", "zh": "详解/计划   "},
    "field.explain_kernel": {"en": "explain/kern", "zh": "详解/内核   "},
    "field.explain_why": {"en": "explain/why ", "zh": "详解/原因   "},
    # 值
    "value.explain_source_shape": {
        "en": "shape-specialised",
        "zh": "形状特化",
    },
    "value.explain_source_corpus": {
        "en": "corpus form",
        "zh": "语料形态",
    },
    "value.explain_plan_hit": {
        "en": "{source} {name}  [{kind}] → declared kernels {kernels}",
        "zh": "{source} {name}  [{kind}] → 声明内核 {kernels}",
    },
    "value.explain_plan_none": {
        "en": "no specialised form (generic step-by-step path) → nothing to dispatch",
        "zh": "未命中任何特化形态（走通用分步路径） → 没有可下发的原生内核",
    },
    "value.explain_plan_equiv": {
        "en": " → equivalent spelling exists ({kind}): {equation}   tensors: "
              "{adjustments}   result: {perm}",
        "zh": " → 存在等价写法（{kind}）：{equation}   张量：{adjustments}   结果：{perm}",
    },
    "value.explain_plan_equiv_unavailable": {
        "en": " → equivalence layer unavailable ({exc})",
        "zh": " → 等价映射层不可用（{exc}）",
    },
    "value.explain_plan_no_equiv": {
        "en": " → no equivalent spelling reaches a known form",
        "zh": " → 不存在可落到已知形式的等价写法",
    },
    "value.explain_why_kept": {
        "en": "kept: {detail} (recorded by a real run in this process)",
        "zh": "保留：{detail}（本进程真的跑过的记录）",
    },
    "value.explain_why_contiguity": {
        "en": "contiguity: {detail}",
        "zh": "连续性问题：{detail}",
    },
    "value.explain_why_dtype": {
        "en": "dtype-policy: real dtype declined ({detail})",
        "zh": "dtype 策略：实型被拒绝（{detail}）",
    },
    "value.explain_why_kernel_unavailable": {
        "en": "kernel-unavailable: {detail}",
        "zh": "内核不可用：{detail}",
    },
    "value.explain_why_not_run": {
        "en": ("not run: this module never contracts, so this query has no abandon "
               "reason (use python -m pyqcd.einsum.test --explain true for a measured "
               "one)"),
        "zh": ("未执行：本模块不收缩张量，因此本次查询没有放弃原因"
               "（实测请用 python -m pyqcd.einsum.test --explain true）"),
    },
    # 值
    "value.arity_slots": {
        "en": "{arity} operands / {slots} index slots (output rank {output_rank})",
        "zh": "{arity} 个操作数 / {slots} 个指标槽（输出秩 {output_rank}）",
    },
    "value.specialize_on": {
        "en": "on (PYQCD_EINSUM_SPECIALIZE)",
        "zh": "开启（PYQCD_EINSUM_SPECIALIZE）",
    },
    "value.specialize_off": {
        "en": "off (PYQCD_EINSUM_SPECIALIZE)",
        "zh": "关闭（PYQCD_EINSUM_SPECIALIZE）",
    },
    "value.operand_order": {
        "en": "reorder operands as {order}",
        "zh": "按 {order} 重排操作数",
    },
    "value.output_perm": {
        "en": "transpose{perm} the result afterwards",
        "zh": "拿到结果后需 transpose{perm}",
    },
    # 输出侧**不需要**回填时的显式文案：这一行不许省略——只报输入侧会让人以为
    # 改写后的结果可以直接用，实际轴序却是错的（或者反过来白转一次）。
    "value.output_perm_none": {
        "en": "- (the rewritten result already has the required axis order)",
        "zh": "-（改写后输出轴序已一致）",
    },
    # 说明 / 提示
    "note.caveat": {
        "en": "declared kernels only mean the contract allows them; whether they "
              "actually run depends on out=, dtype and native-library availability.",
        "zh": "声明内核只表示契约允许；实际是否下发取决于 out=、dtype、原生库可用性。",
    },
    "note.caveat_cont": {
        "en": "numeric verification: python -m "
              "pyqcd.testing.einsum.verify_specialized",
        "zh": "数值验证见 python -m pyqcd.testing.einsum.verify_specialized",
    },
    "hint.equivalence": {
        "en": "the rewrite only changes how the equation is written, never any "
              "execution path; convert the tensors exactly as listed above,",
        "zh": "等价改写只改「方程怎么写」，不改任何执行路径；按上面的张量调整同步转换后，",
    },
    "hint.equivalence_cont": {
        "en": "then call contract() with the equivalent equation to take the "
              "specialised path — the result is element-wise identical.",
        "zh": "用等价方程调用 contract() 即可走上特化路径，结果与直接写法逐元素相同。",
    },
    # 等价映射说明（英文侧按结构化字段自行渲染；中文侧沿用 _equivalence 的原话）
    "detail.operand_order": {
        "en": "operand order {order}",
        "zh": "操作数按 {order} 重排",
    },
    "detail.operand_axes": {
        "en": "operand {index} axes transpose{perm}",
        "zh": "操作数 {index} 轴序 transpose{perm}",
    },
    "detail.output_axes": {
        "en": "output axes transpose{perm}",
        "zh": "输出轴序 transpose{perm}",
    },
    "detail.rewrite_kind": {
        "en": "rewrite kind {kind!r}",
        "zh": "改写类型 {kind!r}",
    },
    # query() 的 reason
    "reason.not_a_string": {
        "en": "the equation must be a string, got {type_name}",
        "zh": "方程必须是字符串，收到 {type_name}",
    },
    "reason.unparsable": {
        "en": "equation is not parsable (illegal character, empty operand, more than "
              "one ->, or an invalid output index)",
        "zh": "方程无法解析（非法字符、空操作数、多重 -> 或输出指标不合法）",
    },
    # 参数错误：非法 --kernel-force 取值（退出码 2，与 argparse 的 choices 同档）
    "error.kernel": {
        "en": ("--kernel-force must be one of {labels} (exactly one label), "
               "got {value!r}"),
        "zh": ("--kernel-force 必须是 {labels} 之一（只接受单个取值），得到 {value!r}"),
    },
    "reason.equivalent_rewrite": {
        "en": "the literal spelling is not in the corpus, but an equivalent spelling "
              "does hit a specialised path ({description}); convert the tensors as "
              "listed, then compute — the result is element-wise identical to the "
              "literal spelling",
        "zh": "直接写法不在语料里，但存在等价写法可命中特化路径（{description}）；"
              "需按给出的张量调整同步转换后再算，结果与直接写逐元素相同",
    },
    "reason.no_specialized_path": {
        "en": "not in the wick_12cases corpus, no shape-specialised path, and no "
              "equivalent spelling reaches a known form — the generic step-by-step "
              "path runs (no specialised kernel)",
        "zh": "既不在 wick_12cases 语料里，也没有形状特化路径，且不存在可落到已知"
              "形式的等价写法，走通用分步路径（无特化内核）",
    },
    # 等价映射**关闭**（`PYQCD_EINSUM_EQUIVALENCE=0`）时的原版措辞：与引入该功能
    # 之前逐字相同，证明"关闭即无此功能"。这段里不许出现任何"等价写法"字样。
    "reason.no_specialized_path_disabled": {
        "en": "not in the wick_12cases corpus and no shape-specialised path — "
              "the generic step-by-step path runs (no specialised kernel)",
        "zh": "既不在 wick_12cases 语料里，也没有形状特化路径，走通用分步路径"
              "（无特化内核）",
    },
}


def _language_name(language: str | None) -> str:
    """``zh-cn``/``zh`` → ``"zh"``，``en*`` → ``"en"``；其余按中文处理。"""
    text = "zh" if language is None else str(language).strip().lower()
    if text.startswith("zh"):
        return "zh"
    if text.startswith("en"):
        return "en"
    return "zh"


def translate(key: str, language: str = "zh", **kwargs: Any) -> str:
    """查文案表并格式化。缺 key / 缺语言 / 缺占位符**直接抛异常**。

    静默返回 key 本身会让漏翻的字符串以"看似正常的英文"混进输出里，比直接崩更难
    发现；本模块宁可 fail loudly。
    """
    lang = _language_name(language)
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


def resolve_language(requested: str | None = None) -> str:
    """把 ``--language`` 解析成实际输出语言（``"en"`` / ``"zh"``）。

    * ``en`` / ``zh-cn``（也接受 ``zh``）显式指定，一直用它；
    * ``auto``（缺省）：只有 ``PYQCD_EINSUM_LANG`` **显式设了**才跟随它；未设时
      保持本入口的历史行为（中文）。**不跟随系统 locale**——直接跟随
      ``notice_language()`` 会让 ``LANG=C.UTF-8`` 的机器上不传选项的输出从中文
      变英文，那是无声的破坏性变更。

    非法值抛 ``ValueError``；CLI 侧由 argparse 的 ``choices`` 拦成退出码 2。
    """
    text = DEFAULT_LANGUAGE if requested is None else str(requested).strip().lower()
    if text in ("en", "english"):
        return "en"
    if text in ("zh-cn", "zh", "cn", "chinese"):
        return "zh"
    if text != "auto":
        raise ValueError(
            f"--language must be one of {LANGUAGE_MODES}, got {requested!r}")
    return _env_language()


def _env_language() -> str:
    """``auto`` 的判定：只在 ``PYQCD_EINSUM_LANG`` 显式设了时跟随，否则中文。"""
    explicit = os.environ.get(LANGUAGE_ENV, "").strip().lower()
    if explicit.startswith("en"):
        return "en"
    if explicit.startswith("zh"):
        return "zh"
    return "zh"


def resolve_equivalence(requested: str | bool | None = None) -> bool:
    """把 ``--equivalence`` 解析成本次是否查等价写法（``True`` = 查）。

    * ``true`` / ``false``（也接受 ``bool`` 与 ``1/0/yes/no/on/off``）**强制**开关，
      压过 ``PYQCD_EINSUM_EQUIVALENCE``；
    * ``auto``（缺省，``None`` 等价）跟随 :func:`_equivalence_enabled`，即
      ``_equivalence.enabled()`` 的环境变量口径（未设 = 开启）。

    非法值抛 ``ValueError``；CLI 侧由 argparse 的 ``choices`` 拦成退出码 2。
    """
    if requested is None:
        return _equivalence_enabled()
    if isinstance(requested, bool):
        return requested
    text = str(requested).strip().lower()
    if text == "auto":
        return _equivalence_enabled()
    if text in ("true", "1", "yes", "on"):
        return True
    if text in ("false", "0", "no", "off"):
        return False
    raise ValueError(
        f"--equivalence must be one of {EQUIVALENCE_MODES}, got {requested!r}")


def resolve_explain(requested: str | bool | None = None) -> bool:
    """把 ``--explain`` 解析成本次是否把"会怎样"写进报告。

    ``auto``（缺省，``None`` 等价）**恒为关闭**：本模块没有 ``--verbose`` 可跟随，
    而缺省报告由 ``_einsum_verify_specialized_contract`` **逐字节**钉住，无条件多
    三行就是破坏性变更。``true``/``false`` 强制开关（也接受 ``bool`` 与
    ``1/0/yes/no/on/off``）。非法值抛 ``ValueError``；CLI 侧由 argparse 的
    ``choices`` 拦成退出码 2。
    """
    if requested is None:
        return False
    if isinstance(requested, bool):
        return requested
    text = str(requested).strip().lower()
    if text == "auto":
        return False
    if text in ("true", "1", "yes", "on"):
        return True
    if text in ("false", "0", "no", "off"):
        return False
    raise ValueError(
        f"--explain must be one of {EXPLAIN_MODES}, got {requested!r}")


def _forced_on(requested: str | bool | None) -> bool:
    """``--equivalence true`` 的**显式**强制开启（区别于 ``auto`` 跟随环境变量）。

    只有这一支会去压 ``_equivalence.enabled()``——``auto`` 读到的就是它，没有冲突。
    """
    if isinstance(requested, bool):
        return requested
    return str(requested).strip().lower() in ("true", "1", "yes", "on")


def _equivalence_enabled() -> bool:
    """``auto`` 的判定：交给 ``_equivalence.enabled()``，不在这里另解析环境变量。

    旧版 ``_equivalence`` 没有 ``enabled`` 时按**开启**处理——那正是引入开关之前的
    行为（等价段一直是开的）。降级只会"关不掉"，不会把功能静默关死。
    """
    from . import _equivalence

    probe = getattr(_equivalence, "enabled", None)
    return True if probe is None else bool(probe())


def _kernel_force_module():
    """取 ``_kernel_force`` 模块；取不到返回 ``None``（降级成旧行为，不半强制）。"""
    try:
        from . import _kernel_force as force
    except ImportError:                         # pragma: no cover - 旧版库
        return None
    return force


def _kernel_label_names() -> tuple[str, ...]:
    """合法取值全集：优先用 ``_kernel_force`` 的权威定义，取不到用本模块字面量。

    含 ``auto`` / ``generic`` / ``backend`` —— 报错文案要列的是**能写什么**。
    """
    force = _kernel_force_module()
    live = getattr(force, "KERNEL_LABELS", None) if force is not None else None
    names = set(live or KERNEL_LABELS)
    names |= {DEFAULT_KERNEL, GENERIC_LABEL, BACKEND_LABEL}
    return tuple(sorted(names))


def resolve_kernel(requested: str | None = None) -> tuple[str, ...]:
    """把 ``--kernel-force`` 解析成标签元组：``()`` 自动 / ``("generic",)`` / 具体标签。

    语义完全交给 :func:`pyqcd.einsum._kernel_force.resolve`（显式取值优先于
    ``PYQCD_EINSUM_KERNEL_FORCE``），本模块只负责把非法取值变成 ``ValueError``
    ——CLI 侧由 :func:`main` 转成退出码 2 并打印合法集合。该模块暂时缺席时按
    ``auto`` 处理：失败关闭，降级成旧行为。
    """
    force = _kernel_force_module()
    if force is None:
        text = DEFAULT_KERNEL if requested is None else str(requested).strip()
        return () if text in ("", DEFAULT_KERNEL) else (text,)
    try:
        return tuple(force.resolve(requested))
    except ValueError as exc:
        raise ValueError(str(exc)) from exc


def kernel_requested(requested: str | None = None) -> str:
    """报告里 ``requested=`` 的取值：``auto`` / ``generic`` / ``label,...``。

    与真正会传给 ``contract(kernel_force=...)`` 的口径同源（同一处解析），所以 ``auto``
    时会反映那一刻 ``PYQCD_EINSUM_KERNEL_FORCE`` 的生效值。
    """
    labels = resolve_kernel(requested)
    return ",".join(labels) if labels else DEFAULT_KERNEL


def last_kernel() -> dict[str, Any] | None:
    """读 ``_api.last_kernel()``；接口缺失或状态不对时返回 ``None``。

    **必须容忍缺失**：本模块与 ``_api`` 分属不同改动批次，取不到就走退化显示
    （``ran=unknown attempted=unknown``）。绝不凭声明内核或 ``LAST_INFO`` 反推一个
    "看起来合理"的 ran——本模块压根不执行，那是编造观测。
    """
    try:
        from . import _api
    except ImportError:                         # pragma: no cover - 旧版库
        return None
    probe = getattr(_api, "last_kernel", None)
    if probe is None:
        return None
    try:
        record = probe()
    except Exception:                           # noqa: BLE001 - 报告层不许影响查询
        return None
    return record if isinstance(record, dict) else None


def _forced_of(labels: tuple[str, ...]) -> tuple[str, ...] | None:
    """``resolve`` 的元组 → ``last_kernel()['requested']`` 的形状。

    ``None`` 自动分派；``()`` 只走通用路径；其余为强制标签。
    """
    if not labels:
        return None
    if labels == ("generic",):
        return ()
    return labels


def kernel_force_fields(requested: str, labels: tuple[str, ...],
                        record: dict[str, Any] | None = None
                        ) -> dict[str, Any]:
    """整理报告三段：``requested`` 来自本模块解析，``ran``/``attempted`` 来自记录。

    ``labels`` 用来挡掉**不属于本次请求**的残留记录（记录的 ``requested`` 对不上
    就不认）：宁可写 ``unknown``，也不显示一条无法归因的观测。
    """
    fields: dict[str, Any] = {"requested": requested, "ran": None, "attempted": None}
    if record is None:
        return fields
    recorded = record.get("requested")
    recorded = None if recorded is None else tuple(str(item) for item in recorded)
    if recorded != _forced_of(labels):
        return fields
    if "ran" in record:
        ran = record.get("ran")
        fields["ran"] = str(ran) if ran else "none"
    attempted = record.get("attempted")
    if attempted is not None:
        names = [str(item) for item in attempted if str(item)]
        fields["attempted"] = ",".join(names) if names else "none"
    return fields


def kernel_force_section(kernel_force: str | None = None) -> dict[str, Any] | None:
    """本次查询要给出 ``kernel :`` 段吗？给就返回三段，否则 ``None``。

    **只在 ``--kernel-force`` 被显式给出时才出现**，理由是两条都不能让步：

    * 本模块的缺省输出是历史原文，由 ``_einsum_verify_specialized_contract``
      逐字节钉住，无条件多一行就是破坏性变更；
    * 本模块**不执行**任何内核，所以 ``PYQCD_EINSUM_KERNEL_FORCE`` 在这里完全
      不起作用——把它写进 ``requested=`` 会让人以为"有东西被强制了"，而实际上
      什么都没跑。要强制就用 ``python -m pyqcd.einsum.test --kernel-force <label>``。

    值方面：``requested`` 是本模块解析出的取值；``ran`` / ``attempted`` 只认
    :func:`last_kernel`，取不到写 ``unknown``——本模块不跑内核，绝不反推。
    """
    if kernel_force is None:
        return None
    labels = resolve_kernel(kernel_force)
    requested = ",".join(labels) if labels else DEFAULT_KERNEL
    return kernel_force_fields(requested, labels, last_kernel())


def render_kernel_force(fields: dict[str, Any]) -> str:
    """``requested=... ran=... attempted=...`` 的值（字段名一律不翻译）。

    这一行是机器可读的：三个字段名与 ``none``/``unknown`` 都是稳定取值，跟着
    ``--language`` 翻译会让脚本解析不到；只有左边的标签走 :data:`MESSAGES`。
    """
    return "requested={}  ran={}  attempted={}".format(
        fields.get("requested") or DEFAULT_KERNEL,
        fields.get("ran") or KERNEL_UNKNOWN,
        fields.get("attempted") or KERNEL_UNKNOWN)


def _explain(subscripts: str, forced_on: bool) -> dict[str, Any]:
    """调 ``_equivalence.explain()``；``--equivalence true`` 时压过它的环境开关。

    优先级要求"显式 ``true`` 压过 ``PYQCD_EINSUM_EQUIVALENCE``"，而
    ``explain()`` 内部还会再查一次 ``enabled()``——环境变量为 0 时它会直接回
    ``rewritten=False``，光在本模块里决定"要不要查"是不够的。这里在**调用期间**
    把该变量摆成开启、``finally`` 原样恢复：本模块是纯查询入口，不并发写全局状态，
    且只在"显式强制开启 + 环境变量关闭"这一个分支发生。

    仍然只走 ``explain()`` 这个公开入口，不碰 ``_equivalence`` 的内部实现。
    """
    from . import _equivalence

    if not forced_on or _equivalence_enabled():
        return _equivalence.explain(subscripts)

    saved = os.environ.get(EQUIVALENCE_ENV)
    os.environ[EQUIVALENCE_ENV] = "1"
    try:
        return _equivalence.explain(subscripts)
    finally:
        if saved is None:
            os.environ.pop(EQUIVALENCE_ENV, None)
        else:
            os.environ[EQUIVALENCE_ENV] = saved


def _explain_why(fields: dict[str, Any],
                 reasons: dict[str, Any]) -> dict[str, str]:
    """``why`` 段：本模块**不执行**，所以只有真跑过的记录才配给"放弃原因"。

    四类判据与 ``python -m pyqcd.einsum.test`` 完全同源（``contiguity`` →
    ``dtype-policy`` → ``kernel-unavailable``；兜底类必须最后判，否则前两类细因
    永远出不来）。没有记录时如实写 ``not-run``——绝不把"没查过"说成"没命中"。
    """
    ran = fields.get("ran")
    if ran:
        return {"kind": "kept", "detail": str(ran)}
    items = [(str(label), str(text)) for label, text in (reasons or {}).items()
             if str(text)]
    for label, text in items:
        if "contiguous" in text.lower():
            return {"kind": "contiguity", "detail": f"{label}: {text}"}
    for label, text in items:
        lowered = text.lower()
        if ("real dtype" in lowered and "declin" in lowered) \
                or "dtype-policy" in lowered:
            return {"kind": "dtype-policy", "detail": f"{label}: {text}"}
    if items:
        return {"kind": "kernel-unavailable",
                "detail": "; ".join(f"{label}: {text}" for label, text in items)}
    return {"kind": "not-run", "detail": ""}


def _explain_section(report: dict[str, Any], kernel_force: str | None,
                     source: str) -> dict[str, Any]:
    """``--explain`` 的报告段：``plan`` / ``kernel`` / ``why``（全部只读）。

    ``source`` 是命中的来源（``shape`` / ``corpus`` / ``none``），由 :func:`query`
    在判定时给出——报告层不再重新查一遍（两次查询说两套话的坑不值得踩）。
    ``kernel`` 的三段与 ``--kernel-force`` 那一行**同源**（:func:`kernel_force_fields`），
    因此 ``ran`` / ``attempted`` 只可能来自本进程真跑过的记录；取不到写 ``unknown``。
    """
    labels = resolve_kernel(kernel_force)
    requested = ",".join(labels) if labels else DEFAULT_KERNEL
    record = last_kernel()
    fields = kernel_force_fields(requested, labels, record)
    reasons = dict((record or {}).get("reasons") or {}) if fields.get("ran") else {}
    return {
        "plan": {
            "matched": bool(report.get("matched")),
            "source": source,
            "name": report.get("name"),
            "kind": report.get("kind"),
            "kernels": list(report.get("kernels") or ()),
            "equivalence": report.get("equivalence"),
        },
        "kernel": {
            "requested": requested,
            "ran": fields.get("ran"),
            "attempted": fields.get("attempted"),
            "reasons": reasons,
        },
        "why": _explain_why(fields, reasons),
    }


def render_explain_kernel(kernel: dict[str, Any]) -> str:
    """``requested=... → attempted=... → ran=...``（字段名一律不翻译）。

    与 ``test.py`` 的同名行同口径：三段稳定可解析，跟着 ``--language`` 翻译会让
    脚本解析不到。缺值写 ``unknown``/``none``，绝不空着。
    """
    return "requested={}  → attempted={}  → ran={}".format(
        kernel.get("requested") or DEFAULT_KERNEL,
        kernel.get("attempted") or KERNEL_UNKNOWN,
        kernel.get("ran") or KERNEL_UNKNOWN)


def _explain_plan_text(plan: dict[str, Any], language: str = "zh") -> str:
    """``plan`` 段的一行人话：命中了什么 / 没命中时有没有等价写法。"""
    if plan.get("matched"):
        source = plan.get("source")
        source_key = ("value.explain_source_shape" if source == "shape"
                      else "value.explain_source_corpus")
        return translate("value.explain_plan_hit", language,
                         source=translate(source_key, language),
                         name=plan.get("name") or "-",
                         kind=plan.get("kind") or "-",
                         kernels=", ".join(plan.get("kernels") or ()) or "-")
    line = translate("value.explain_plan_none", language)
    equivalent = plan.get("equivalence")
    if isinstance(equivalent, dict):
        if equivalent.get("unavailable"):
            line += translate("value.explain_plan_equiv_unavailable", language,
                              exc=equivalent["unavailable"])
        elif equivalent.get("rewritten"):
            line += translate("value.explain_plan_equiv", language,
                              kind=equivalent.get("kind", "-"),
                              equation=equivalent.get("equation", "-"),
                              adjustments=_format_operand_adjustments(equivalent),
                              perm=_format_output_adjustment(equivalent, language))
        else:
            line += translate("value.explain_plan_no_equiv", language)
    return line


def _explain_why_text(why: dict[str, Any], language: str = "zh") -> str:
    """``why`` 段的一行人话；四类 + ``not-run``，未知类别也明确写出来。"""
    kind = str(why.get("kind") or "not-run")
    key = {
        "kept": "value.explain_why_kept",
        "contiguity": "value.explain_why_contiguity",
        "dtype-policy": "value.explain_why_dtype",
        "kernel-unavailable": "value.explain_why_kernel_unavailable",
    }.get(kind, "value.explain_why_not_run")
    return translate(key, language, detail=why.get("detail") or "-")


def _reason(key: str, **args: Any) -> dict[str, Any]:
    """``reason`` 相关字段：历史中文文案 + 语言无关的 key/参数。"""
    return {
        "reason": translate(key, "zh", **args),
        "reason_key": key,
        "reason_args": dict(args),
    }


def query(subscripts: str, equivalence: str | bool | None = None,
          kernel_force: str | None = None,
          explain: str | bool | None = None) -> dict[str, Any]:
    """返回该方程的语料归属与执行契约（纯查询，无副作用）。

    ``equivalence`` 是 ``--equivalence`` 的三态（``None`` = ``auto``，见
    :func:`resolve_equivalence`）。它只决定"未命中时要不要查等价写法"，
    **不参与** ``matched`` 的判定，因此也不影响退出码。

    ``kernel`` 是 ``--kernel-force`` 的原始取值（``None`` = 未给）。给了就在报告里多一个
    ``kernel_force`` 字段（``requested`` / ``ran`` / ``attempted``）；不给则**没有**
    该字段，报告与改造前逐字段相同。它同样不参与判定、不影响退出码。

    ``explain`` 是 ``--explain`` 的三态（``None`` = ``auto`` = 关闭，见
    :func:`resolve_explain`）。开启时报告多一个 ``explain`` 字段（``plan`` /
    ``kernel`` / ``why``），描述"**会怎样**"——本模块不执行收缩，所以那里没有任何
    实测数字；``ran``/``attempted`` 只可能来自本进程真跑过的记录。它同样不参与
    判定、不影响退出码，也不改变 ``--json`` 之外的任何字段。
    """
    explain_on = resolve_explain(explain)
    force_section = kernel_force_section(kernel_force)
    if not isinstance(subscripts, str):
        report: dict[str, Any] = {
            "equation": subscripts,
            "equation_normalized": "",
            "parsable": False,
            "matched": False,
            "specialize_enabled": _specialize.enabled(),
            **_reason("reason.not_a_string", type_name=type(subscripts).__name__),
        }
        if force_section is not None:
            report["kernel_force"] = force_section
        return report
    normalized = _specialize.normalize(subscripts)
    signature = _specialize.canonical_signature(subscripts)
    report = {
        "equation": subscripts,
        "equation_normalized": normalized,
        "parsable": signature is not None,
        "matched": False,
        "specialize_enabled": _specialize.enabled(),
    }
    if force_section is not None:
        report["kernel_force"] = force_section
    if signature is None:
        report.update(_reason("reason.unparsable"))
        return report

    report["signature"] = repr(signature)
    report["arity"] = len(signature) - 1
    report["slots"] = sum(len(term) for term in signature)
    report["digest"] = _specialize._signature_digest(signature)
    report["output_rank"] = len(signature[-1])

    from . import _shape_paths

    shape = _shape_paths.entry(subscripts)
    if shape is not None:
        report.update({
            "matched": True,
            "name": shape["name"],
            "kind": shape["kind"],
            "kernels": list(shape.get("kernels", ())),
            "note": shape.get("note", ""),
        })
        if explain_on:
            report["explain"] = _explain_section(report, kernel_force, "shape")
        return report

    entry = _specialize.match(subscripts)
    if entry is None:
        # 未直接命中：查是否存在**数学等价**的写法能落到语料/形状特化上。
        # 只做轴序 / 操作数位置 / 输出轴序 / 复合变换（逐元素等价，2026-11-16 实测），
        # 不改任何执行路径——本模块仍只输出报告。
        # `--equivalence false`（或 `PYQCD_EINSUM_EQUIVALENCE=0`）时既不查也不打印，
        # 并回到**没有本功能时**的原版措辞，报告里也没有 `equivalence` 字段。
        if not resolve_equivalence(equivalence):
            report.update(_reason("reason.no_specialized_path_disabled"))
            if explain_on:
                report["explain"] = _explain_section(report, kernel_force, "none")
            return report

        equivalent = _explain(subscripts, _forced_on(equivalence))
        # 与 `test.py` 的 `equivalence` 字段同口径：开启时**总是**写（`rewritten`
        # 为假也写，消费者据此区分"查过了没有等价写法"与"根本没查"）；关闭时不写。
        report["equivalence"] = equivalent
        if equivalent.get("rewritten"):
            report.update(_reason("reason.equivalent_rewrite",
                                  description=equivalent["description"]))
        else:
            report.update(_reason("reason.no_specialized_path"))
        if explain_on:
            report["explain"] = _explain_section(report, kernel_force, "none")
        return report

    report.update({
        "matched": True,
        "name": entry["name"],
        "kind": entry["kind"],
        "kernels": list(entry.get("kernels", ())),
    })
    corpus = _corpus_entry(signature)
    if corpus is not None:
        report["corpus"] = corpus
    if explain_on:
        report["explain"] = _explain_section(report, kernel_force, "corpus")
    return report


def _corpus_entry(signature: tuple) -> dict[str, Any] | None:
    """语料里同签名的代表条目。

    首次调用建一次 ``摘要 -> Task`` 索引（8638 条，约 0.13 s），之后 O(1)。
    逐次线性扫描要 12.9 ms/次，查询工具会被反复调用，不值得。
    """
    index = _corpus_index()
    task = index.get(_specialize._signature_digest(signature))
    if task is None:
        return None
    return {
        "seq": task.seq,
        "tid": task.tid,
        "case": task.case,
        "subtable": task.subtable,
        "equation": task.equation,
        "roles": list(task.roles),
    }


_CORPUS_INDEX: dict[str, Any] | None = None


def _corpus_index() -> dict[str, Any]:
    """``摘要 -> Task`` 索引，第一次用到时构建并缓存。"""
    global _CORPUS_INDEX
    if _CORPUS_INDEX is None:
        from . import _wick_tasklist as tasklist

        index: dict[str, Any] = {}
        for task in tasklist.TASKS:
            task_signature = _specialize.canonical_signature(task.equation)
            if task_signature is None:
                continue
            index.setdefault(_specialize._signature_digest(task_signature), task)
        _CORPUS_INDEX = index
    return _CORPUS_INDEX


def _describe_equivalence(equivalent: dict[str, Any], language: str) -> str:
    """等价映射的一句话说明。

    中文侧**逐字沿用** ``_equivalence`` 给的 ``description``（改造前的输出就是它，
    换一种写法就等于改了默认输出）；英文侧按结构化字段自己渲染，因为
    ``_equivalence`` 目前只产出中文说明。渲染只读 ``operand_order`` /
    ``operand_permutations`` / ``output_permutation`` / ``kind`` 四个字段——这些是
    该模块的稳定公开形态，不依赖它的内部实现。
    """
    if _language_name(language) == "zh":
        return str(equivalent.get("description", ""))
    parts: list[str] = []
    order = equivalent.get("operand_order")
    if order:
        parts.append(translate("detail.operand_order", language, order=list(order)))
    for index, perm in enumerate(equivalent.get("operand_permutations") or ()):
        if perm is not None:
            parts.append(translate("detail.operand_axes", language,
                                   index=index, perm=tuple(perm)))
    output_perm = equivalent.get("output_permutation")
    if output_perm:
        parts.append(translate("detail.output_axes", language,
                               perm=tuple(output_perm)))
    if not parts:
        parts.append(translate("detail.rewrite_kind", language,
                               kind=equivalent.get("kind")))
    return "; ".join(parts)


def _format_operand_adjustments(equivalent: dict[str, Any]) -> str:
    """输入侧一行：逐操作数给 ``transpose(...)``，不需要动的那一路写 ``-``。

    按**改写后**方程的操作数次序渲染。``operand_order`` 不并进这一行——它有自己的
    标签行（``张量次序``），混进来会让"哪一路要转置"变得难读。
    """
    rendered = [
        "-" if perm is None else f"transpose{tuple(perm)}"
        for perm in (equivalent.get("operand_permutations") or ())]
    return ", ".join(rendered) if rendered else "-"


def _format_output_adjustment(equivalent: dict[str, Any], language: str) -> str:
    """输出侧一行：``transpose(...)``，无需回填时给**显式**文案而不是省略整行。

    只报输入侧会让人以为改写后的结果可以直接用，实际轴序却是错的。两侧必须同时
    在场（2026-11-16 用户要求"转换输入的同时也有对应变换输出"）。
    """
    perm = equivalent.get("output_permutation")
    if not perm:
        return translate("value.output_perm_none", language)
    return translate("value.output_perm", language, perm=tuple(perm))


def _reason_text(report: dict[str, Any], language: str) -> str:
    """把 ``reason_key``/``reason_args`` 渲染成目标语言；缺字段时退回 ``reason``。"""
    key = report.get("reason_key")
    if key is None:
        return str(report.get("reason", ""))
    args = dict(report.get("reason_args") or {})
    equivalent = report.get("equivalence")
    if equivalent is not None and "description" in args:
        args["description"] = _describe_equivalence(equivalent, language)
    return translate(key, language, **args)


def _print_report(report: dict[str, Any], language: str = "zh") -> None:
    """打印人读报告。``language`` 只影响文案；缺省 ``"zh"`` 与改造前逐字节相同。"""
    def t(key: str, **kwargs: Any) -> str:
        return translate(key, language, **kwargs)

    print(f"{t('field.equation')}: {report['equation']}")
    print(f"{t('field.normalized')}: {report['equation_normalized']}")
    if not report["parsable"]:
        print(f"{t('field.result')}: FAIL")
        print(f"{t('field.reason')}: {_reason_text(report, language)}")
        return

    print(f"{t('field.signature')}: {report['signature']}")
    print(f"{t('field.arity_slots')}: "
          + t("value.arity_slots", arity=report['arity'], slots=report['slots'],
              output_rank=report['output_rank']))
    print(f"{t('field.digest')}: {report['digest']}")
    print(f"{t('field.specialize')}: "
          + t("value.specialize_on" if report['specialize_enabled']
              else "value.specialize_off"))
    # ``--kernel-force`` 只在显式给出时才有这一段（``query()`` 的 kernel_force 字段）：
    # 缺省报告必须与改造前逐字节相同，所以这一行不能无条件出现。
    kernel_force = report.get("kernel_force")
    if kernel_force is not None:
        print(f"{t('field.kernel_force')}: {render_kernel_force(kernel_force)}")
    # ``--explain true`` 的三段：本模块不执行，所以这里全是"会怎样"而不是实测。
    # 缺省（``auto``）没有该字段，报告与改造前逐字节相同。
    explain = report.get("explain")
    if explain is not None:
        plan = explain.get("plan") or {}
        print(f"{t('field.explain_plan')}: {_explain_plan_text(plan, language)}")
        print(f"{t('field.explain_kernel')}: "
              + render_explain_kernel(explain.get("kernel") or {}))
        print(f"{t('field.explain_why')}: "
              + _explain_why_text(explain.get("why") or {}, language))

    if not report["matched"]:
        print(f"{t('field.result')}: MISS")
        print(f"{t('field.reason')}: {_reason_text(report, language)}")
        equivalent = report.get("equivalence")
        if equivalent and equivalent.get("rewritten"):
            print(f"{t('field.rewrite')}: {equivalent['equation']}")
            print(f"{t('field.rewrite_kind')}: {equivalent['kind']}")
            print(f"{t('field.rewrite_detail')}: "
                  + _describe_equivalence(equivalent, language))
            print(f"{t('field.operand_perms')}: "
                  + _format_operand_adjustments(equivalent))
            order = equivalent.get("operand_order")
            if order:
                print(f"{t('field.operand_order')}: "
                      + t("value.operand_order", order=order))
            print(f"{t('field.output_perm')}: "
                  + _format_output_adjustment(equivalent, language))
            print(f"{t('field.hint')}: {t('hint.equivalence')}")
            print(" " * 14 + t("hint.equivalence_cont"))
        return

    print(f"{t('field.result')}: HIT")
    print(f"{t('field.path')}: {report['name']}  [{report['kind']}]")
    print(f"{t('field.kernels')}: {', '.join(report['kernels']) or '-'}")
    corpus = report.get("corpus")
    if corpus is not None:
        print(f"{t('field.corpus')}: seq={corpus['seq']}  {corpus['tid']}  "
              f"{corpus['case']}"
              f"{' / ' + corpus['subtable'] if corpus['subtable'] else ''}")
        print(f"{t('field.canonical')}: {corpus['equation']}")
        print(f"{t('field.roles')}: {', '.join(corpus['roles'])}")
    print(f"{t('field.caveat')}: {t('note.caveat')}")
    print(" " * 14 + t("note.caveat_cont"))


def build_parser(language: str = "zh") -> argparse.ArgumentParser:
    """构造参数解析器。``language`` 只影响 ``description``/``epilog``/``help``。

    argparse 在构造时求值这些字符串，所以语言必须在调用本函数时定下来，不能在
    import 时算。缺省 ``"zh"``，与改造前一致。
    """
    lang = _language_name(language)

    def t(key: str, **kwargs: Any) -> str:
        return translate(key, lang, **kwargs)

    parser = argparse.ArgumentParser(
        description=t("help.description"),
        epilog=t("help.epilog"),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("equation", help=t("help.equation"))
    parser.add_argument("--json", action="store_true", help=t("help.json"))
    parser.add_argument("--language", default=DEFAULT_LANGUAGE,
                        choices=LANGUAGE_MODES,
                        help=t("help.language", env=LANGUAGE_ENV,
                               default=DEFAULT_LANGUAGE))
    parser.add_argument("--equivalence", default=DEFAULT_EQUIVALENCE,
                        choices=EQUIVALENCE_MODES,
                        help=t("help.equivalence", env=EQUIVALENCE_ENV,
                               default=DEFAULT_EQUIVALENCE))
    # 故意不设 ``choices``：取值允许逗号分隔的多个标签，合法性由
    # ``resolve_kernel``（即 ``_kernel_force.parse``）统一判，报错能列出合法集合。
    # ``default=None`` 是刻意的：``None`` 表示"用户没给"，报告里就不出 kernel 行，
    # 缺省输出才与改造前逐字节相同。
    parser.add_argument("--kernel-force", default=None,
                        help=t("help.kernel", default=DEFAULT_KERNEL,
                               env=KERNEL_FORCE_ENV,
                               labels=", ".join(_kernel_label_names())))
    # ``default='auto'`` = 关闭：本模块没有 --verbose 可跟随，缺省报告必须逐字节不变。
    parser.add_argument("--explain", default=DEFAULT_EXPLAIN, choices=EXPLAIN_MODES,
                        help=t("help.explain", default=DEFAULT_EXPLAIN))
    return parser


def _peek_language(argv: Sequence[str] | None) -> str | None:
    """在完整解析之前抽出 ``--language``（两种写法都认），用于选 help 文案。

    解析失败（如 ``--language fr``）交给完整解析去报错；此处只做保守取值。
    """
    items = list(sys.argv[1:] if argv is None else argv)
    found: str | None = None
    for index, item in enumerate(items):
        if item == "--language" and index + 1 < len(items):
            found = items[index + 1]
        elif item.startswith("--language="):
            found = item.split("=", 1)[1]
    return found


def main(argv: list[str] | None = None) -> int:
    # 语言在调用时解析（不在 import 时）：先只取 --language 决定 help 文案，
    # 再完整解析一次。非法值不进第一遍，留给 argparse 的 choices 报成退出码 2。
    peeked = _peek_language(argv)
    parser = build_parser(
        language=resolve_language(peeked if peeked in LANGUAGE_MODES else None))
    args = parser.parse_args(argv)
    language = resolve_language(args.language)

    # --kernel-force 在这里就校验：非法标签属于**参数错误**（退出码 2），必须在做任何
    # 查询之前报出来，并把合法集合一并写上——否则用户只会拿到一句 argparse 的
    # "invalid choice"，不知道能填什么。
    if args.kernel_force is not None:
        try:
            resolve_kernel(args.kernel_force)
        except ValueError:
            print(translate("error.kernel", language, value=args.kernel_force,
                            labels=", ".join(_kernel_label_names())),
                  file=sys.stderr)
            return 2

    report = query(args.equation, equivalence=args.equivalence,
                   kernel_force=args.kernel_force, explain=args.explain)
    if args.json:
        # 机器读：query() 的结果与语言无关，--language 影响不到任何结构化字段。
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_report(report, language)

    if not report["parsable"]:
        return 2
    return 0 if report["matched"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
