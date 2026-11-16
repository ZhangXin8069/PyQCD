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
    "build_parser",
    "MESSAGES",
    "LANGUAGE_MODES",
    "DEFAULT_LANGUAGE",
    "LANGUAGE_ENV",
    "EQUIVALENCE_MODES",
    "DEFAULT_EQUIVALENCE",
    "EQUIVALENCE_ENV",
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
    "help.epilog": {
        "en": ("Exit codes:\n"
               "  0  hit a specialised (corpus/shape) path\n"
               "  1  not hit — the generic step-by-step path runs\n"
               "  2  equation unparsable, or bad arguments\n"
               "\n"
               "Declared kernels only mean the contract allows them; whether they "
               "actually run depends\n"
               "on out=, dtype and native-library availability.\n"
               "Numeric verification: python -m pyqcd.testing.einsum.verify_specialized"),
        "zh": ("退出码：\n"
               "  0  命中特化路径（语料 / 形状特化）\n"
               "  1  未命中——走通用分步路径\n"
               "  2  方程无法解析，或参数非法\n"
               "\n"
               "「声明内核」只表示契约允许；实际是否下发取决于 out=、dtype、"
               "原生库可用性。\n"
               "数值验证：python -m pyqcd.testing.einsum.verify_specialized"),
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


def _reason(key: str, **args: Any) -> dict[str, Any]:
    """``reason`` 相关字段：历史中文文案 + 语言无关的 key/参数。"""
    return {
        "reason": translate(key, "zh", **args),
        "reason_key": key,
        "reason_args": dict(args),
    }


def query(subscripts: str, equivalence: str | bool | None = None) -> dict[str, Any]:
    """返回该方程的语料归属与执行契约（纯查询，无副作用）。

    ``equivalence`` 是 ``--equivalence`` 的三态（``None`` = ``auto``，见
    :func:`resolve_equivalence`）。它只决定"未命中时要不要查等价写法"，
    **不参与** ``matched`` 的判定，因此也不影响退出码。
    """
    if not isinstance(subscripts, str):
        return {
            "equation": subscripts,
            "equation_normalized": "",
            "parsable": False,
            "matched": False,
            "specialize_enabled": _specialize.enabled(),
            **_reason("reason.not_a_string", type_name=type(subscripts).__name__),
        }
    normalized = _specialize.normalize(subscripts)
    signature = _specialize.canonical_signature(subscripts)
    report: dict[str, Any] = {
        "equation": subscripts,
        "equation_normalized": normalized,
        "parsable": signature is not None,
        "matched": False,
        "specialize_enabled": _specialize.enabled(),
    }
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

    report = query(args.equation, equivalence=args.equivalence)
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
