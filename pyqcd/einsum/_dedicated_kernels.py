"""给原生下发的保守方程校验：把 ``a,b,c->out`` 拆成 (terms, output)。

比库自己的子脚本解析器更严：拒绝省略号、空操作数、操作数内重复指标、输出里
重复或来自输入之外的指标。原生入口拿到的方程必须先过这一道，否则越界的那
部分错误会推迟到 C++ 侧才暴露。
"""

from __future__ import annotations

__all__ = ["parse_equation"]


def parse_equation(equation: str) -> tuple[tuple[str, ...], str]:
    """把 ``a,b,c->out`` 拆成 (terms, output)；拒绝省略号与非法字符。"""
    if not isinstance(equation, str):
        raise TypeError(f"equation must be a string, got {type(equation).__name__}")
    text = "".join(equation.split())
    if text.count("->") != 1:
        raise ValueError(f"equation must contain exactly one '->': {equation!r}")
    lhs, output = text.split("->")
    terms = tuple(lhs.split(","))
    if any(not term for term in terms):
        raise ValueError(f"equation has an empty operand: {equation!r}")
    if "..." in lhs or "..." in output:
        raise ValueError(f"ellipsis is not supported: {equation!r}")
    seen: set[str] = set()
    for term in terms:
        if len(set(term)) != len(term):
            raise ValueError(f"repeated label inside one operand: {term!r}")
        seen.update(term)
    if len(set(output)) != len(output):
        raise ValueError(f"repeated label in output: {output!r}")
    missing = set(output) - seen
    if missing:
        raise ValueError(f"output labels absent from the inputs: {sorted(missing)!r}")
    return terms, output


if __name__ == "__main__":
    print(parse_equation("ac,cd,de->ae"))
