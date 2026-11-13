"""特化路径查询：给定一个 einsum 方程，报告它在语料中的归属与执行契约。

只做**查询**，不做数值计算：不生成操作数、不调用 ``contract()``、不改任何
全局状态。因此它既能当 CLI 用，也能在生产代码里安全调用。

    python -m pyqcd.einsum.verify_specialized "agbh,cmdn,eqfr,ospt,ikjl,ce,mo,gi,kq,bdf,np,hj,lrt->as"
    python -m pyqcd.einsum.verify_specialized "<eq>" --json

判据（任一不满足即非零退出）：

1. **命中**：拓扑签名在语料索引里（``_specialize.match``）；
2. **可解析**：``canonical_signature`` 非空；
3. **未命中**时退出码为 1，但输出仍说明这是语料之外的方程、会走通用路径。

字段口径：

* ``name``：与运行时提示里打印的 ``topology:<摘要前 8 位>`` 完全一致；
* ``digest``：完整 96 位摘要（``_specialize._signature_digest``，blake2b-128 截断）；
* ``kernels``：该形态**声明**可用的原生内核标签。声明不等于实际执行——
  实际下发受 ``out=``、dtype、原生库可用性影响，判据见
  ``skills/pyqcd-einsum/SKILL.md`` 的「命中/回落判据」；
* ``corpus``：语料里同签名的代表写法；语料外的方程没有这一段。

数值验证（结果是否正确、是否真跑原生内核）需要实际执行，属于测试侧职责：
``python -m pyqcd.testing.einsum.verify_specialized``。本模块刻意不重复那条路径，
因为生产代码不得 import ``pyqcd.testing``。

退出码：``0`` 命中语料；``1`` 语料之外（走通用路径）；``2`` 参数或方程非法。
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from . import _specialize

__all__ = ["query"]


def query(subscripts: str) -> dict[str, Any]:
    """返回该方程的语料归属与执行契约（纯查询，无副作用）。"""
    if not isinstance(subscripts, str):
        return {
            "equation": subscripts,
            "equation_normalized": "",
            "parsable": False,
            "matched": False,
            "specialize_enabled": _specialize.enabled(),
            "reason": f"方程必须是字符串，收到 {type(subscripts).__name__}",
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
        report["reason"] = "方程无法解析（非法字符、空操作数、多重 -> 或输出指标不合法）"
        return report

    report["signature"] = repr(signature)
    report["arity"] = len(signature) - 1
    report["slots"] = sum(len(term) for term in signature)
    report["digest"] = _specialize._signature_digest(signature)
    report["output_rank"] = len(signature[-1])

    entry = _specialize.match(subscripts)
    if entry is None:
        report["reason"] = "不在 wick_12cases 语料里，走通用分步路径（无特化内核）"
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


def _print_report(report: dict[str, Any]) -> None:
    print(f"方程        : {report['equation']}")
    print(f"归一化      : {report['equation_normalized']}")
    if not report["parsable"]:
        print(f"结果        : FAIL")
        print(f"  原因      : {report['reason']}")
        return

    print(f"拓扑签名    : {report['signature']}")
    print(f"元数/槽位   : {report['arity']} 个操作数 / {report['slots']} 个指标槽"
          f"（输出秩 {report['output_rank']}）")
    print(f"摘要        : {report['digest']}")
    print(f"特化开关    : {'开启' if report['specialize_enabled'] else '关闭'}"
          f"（PYQCD_EINSUM_SPECIALIZE）")

    if not report["matched"]:
        print(f"结果        : MISS")
        print(f"  原因      : {report['reason']}")
        return

    print(f"结果        : HIT")
    print(f"  特化路径  : {report['name']}  [{report['kind']}]")
    print(f"  声明内核  : {', '.join(report['kernels']) or '-'}")
    corpus = report.get("corpus")
    if corpus is not None:
        print(f"  语料出处  : seq={corpus['seq']}  {corpus['tid']}  {corpus['case']}"
              f"{' / ' + corpus['subtable'] if corpus['subtable'] else ''}")
        print(f"  代表写法  : {corpus['equation']}")
        print(f"  操作数角色: {', '.join(corpus['roles'])}")
    print("  说明      : 声明内核只表示契约允许；实际是否下发取决于 out=、dtype、"
          "原生库可用性。")
    print("              数值验证见 python -m pyqcd.testing.einsum.verify_specialized")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="查询一条 einsum 方程是否命中语料特化路径，并给出执行契约")
    parser.add_argument("equation", help='einsum 方程，例如 "agbh,cmdn,...,lrt->as"')
    parser.add_argument("--json", action="store_true", help="输出机器可读结果")
    args = parser.parse_args(argv)

    report = query(args.equation)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_report(report)

    if not report["parsable"]:
        return 2
    return 0 if report["matched"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
