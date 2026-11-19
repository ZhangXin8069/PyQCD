"""向 ``_wick_extra.jsonl`` 追加一条特化方程，并重建 :mod:`pyqcd.einsum._wick_tasklist`。

:mod:`pyqcd.einsum._wick_tasklist` 是图纸 ``refer/sush/wick_12cases.md`` 的**生成物**，
本模块是它的唯一追加入口——那份文件请勿手改。追加条目**不写入**它，而是写进独立文件
``pyqcd/einsum/_wick_extra.jsonl``，再由 :mod:`pyqcd.einsum.wick_tasklist_gen` 合并
生成（本工具写完会自动调用生成器重建，``--no-regen`` 可关闭）。用法::

    python -m pyqcd.einsum.wick_tasklist_add \\
        "<描述> <case 名> <equation> <roles 逗号分隔>"

实例如下（``->`` 之前是"描述 + case 名"，之后是 roles）::

    python -m pyqcd.einsum.wick_tasklist_add "动量指标与流分量指标 五夸克到三夸克四点函数 cwdx,eufv,msnt,okpl,igjh,qarb,ac,mo,Ggi,Hkq,su,bd,fnp,Mhj,Mlr,tvx->MGHew peram_u,peram_u,peram_u,peram_u,peram_u,peram_u,gamma_M,C*gamma_B,gamma_J1,gamma_J2,C*gamma_BH,VDV_0,VVV_1,VDV_2,VDV_3,VVV_4"

整串按空白切分后：

* 含 ``->`` 的那个 token 是 ``equation``；
* 它后面唯一的 token 是 ``roles``（逗号分隔，个数必须等于输入数）；
* 它前面最后一个 token 是 case 名，写入 ``subtable`` 字段；
* 更前面的全部是描述，写入 ``section`` 字段（可为空）。

``_wick_extra.jsonl`` 每行一个 JSON 对象::

    {"seq": 8639, "tid": "T08639", "line": 0, "section": "…",
     "subtable": "…", "equation": "ij,jk->ik", "roles": ["…", "…"]}

``line`` 固定 **0**，表示"非图纸来源"：图纸行号从 1 起，``0`` 不冒充任何图纸行，
且 :func:`pyqcd.einsum._wick_tasklist.base_table_count` 的判据是 ``0 < line <= 2805``，
所以 ``0`` 不会被算进基础表。``seq``/``tid`` 由生成器按合并顺序重算，本工具写盘时
按现有条数顺延，生成器再以同一规则复算（同一份文件，两个入口结果一致）。

**不重排已有条目**，``seq`` 保持稳定。

形状口径分两层，别把第一层当实际操作数形状：下表的 ``peram``/``gamma``/``VDV``/``VVV``
是**角色的标称配置**；实际操作数形状取自 equation 该位置的指标段长度。等长角色
（``gamma``/``VDV``/``VVV``，其配置各轴同长）把该指标段的**每个**标签都绑到这个长度，
因此 ``gamma_J1`` 配 ``Ggi`` 得到 ``[4, 4, 4]``；``peram`` 四轴不等长，按四位置绑定。

实测语料（8638 条）各角色的标签个数：``peram`` 恒 4；基础表 ``gamma`` 2、
``VDV`` 2、``VVV`` 3；附录 ``gamma`` 2–3、``VDV`` 2–3、``VVV`` 3–4。所以标签个数
必须由 equation 决定，不能按标称轴数硬卡——否则用户样例的 ``VVV_1``/``VVV_4``
（各 3 个标签）会被直接判死。

幂等：同一 ``equation`` + 同 ``section``（case）已存在时不重复添加，打印既有条目
并退出 0。校验失败（缺 ``->``、输入数与 roles 数不符、角色前缀不在下表）退出 2
且不写盘。

形状口径（用户给定，生成器默认配置）::

    peram → [4, 4, 100, 100]
    gamma → [4, 4]                （含 C*gamma_* 前缀形式，同样 [4, 4]）
    VDV   → [100, 100]
    VVV   → [100, 100, 100, 100]

``roles`` 按**前缀**匹配（``peram_u`` / ``gamma_M`` / ``C*gamma_B`` / ``VDV_0`` /
``VVV_1`` 等）。输出形状由这些信息自动推出。绑定次序与
:func:`pyqcd.einsum.bench_specialized._label_sizes` 同构——**先小张量后 peram**：

1. ``gamma`` 系列先把该指标段**每个**标签钉到 4（``Ggi`` → ``G=g=i=4``）；
   ``VDV``/``VVV`` 把指标段的标签钉到其配置的轴长（各轴同长，故逐标签取该长度，
   因此 ``VDV`` 带 3 个标签、``VVV`` 带 3 个标签都合法——语料附录形态正是如此；
   基础表才是 ``VDV`` 2 个 / ``VVV`` 3–4 个）；
2. ``peram`` 只补**尚未被钉住**的标签，长度取 ``peram`` 配置的末轴（``Nev``）；
3. 同一步内两个标签要不同长度、或同一标签被钉两次不同长度 → 报错退出 2。

第 2 步是"补余"而不是"覆盖"，与用户样例强相关：它的 ``igjh``（peram）与
``Ggi``（gamma）共享 ``g``/``i``，前者必须是 ``Nev`` 才有意义，而 ``gamma`` 侧
要的是旋量维 4——两条要求只有"先小张量、peram 补余"能同时满足。

注意这套固定配置与 :func:`pyqcd.einsum.bench_specialized._label_sizes` 的**场景口径**
不同：后者按 ``Lt``/``Mom`` 等真实生产轴长绑定（``distill`` 档下 ``vdv`` 首两轴是
``Lt=1``/``Mom=1`` 占位），两者不可互相搬用。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXTRA = Path(__file__).with_name("_wick_extra.jsonl")
NON_BLUEPRINT_LINE = 0

ROLE_SHAPES: dict[str, tuple[int, ...]] = {
    "peram": (4, 4, 100, 100),
    "gamma": (4, 4),
    "VDV": (100, 100),
    "VVV": (100, 100, 100, 100),
}


class SpecError(ValueError):
    """命令行串不合约定（退出码 2，不写盘）。"""


def parse_spec(text: str) -> tuple[str, str, str, tuple[str, ...]]:
    """把整串拆成 ``(section, subsection, equation, roles)``。"""
    tokens = text.split()
    index = next((i for i, token in enumerate(tokens) if "->" in token), None)
    if index is None:
        raise SpecError("找不到含 '->' 的 equation")
    if index == 0:
        raise SpecError("equation 之前缺少 case 名")
    if index + 1 >= len(tokens):
        raise SpecError("equation 之后缺少 roles")
    if index + 2 != len(tokens):
        raise SpecError(
            f"equation 之后只允许一个 roles 串，多出 {len(tokens) - index - 2} 个 token")

    equation = tokens[index]
    roles = tuple(tokens[index + 1].split(","))
    if equation.count("->") != 1:
        raise SpecError(f"equation 必须恰好含一个 '->'：{equation!r}")
    if not all(roles):
        raise SpecError("roles 里有空项")
    return " ".join(tokens[:index - 1]), tokens[index - 1], equation, roles


def role_shape(role: str) -> tuple[int, ...]:
    """角色前缀 → 轴长。``C*gamma_B`` 与 ``gamma_M`` 同属 ``gamma``。

    语料里前缀写作 ``C * gamma_B``（带空格），本工具两种拼写都接受。
    """
    name = role.replace(" ", "")
    name = name[2:] if name.startswith("C*") else name
    prefix = name.split("_", 1)[0]
    if prefix not in ROLE_SHAPES:
        raise SpecError(
            f"未知角色前缀 {role!r}（合法前缀：{', '.join(ROLE_SHAPES)}，"
            f"可带 C* 前缀）")
    return ROLE_SHAPES[prefix]


def _is_peram(role: str) -> bool:
    return role.replace(" ", "").startswith("peram")


def infer_shapes(equation: str, roles: tuple[str, ...]
                 ) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """返回每个操作数的形状与输出形状；标签冲突或未绑定即报错。"""
    terms = equation.split("->")[0].split(",")
    if len(roles) != len(terms):
        raise SpecError(f"{len(roles)} 个 roles 对 {len(terms)} 个输入")

    sizes: dict[str, int] = {}

    def bind(label: str, extent: int, role: str) -> None:
        known = sizes.setdefault(label, extent)
        if known != extent:
            raise SpecError(
                f"标签 {label!r} 在别处是 {known}、在 {role!r} 里是 {extent}")

    # 小张量先钉：等长角色（配置各轴同长）对指标段每个标签生效，标签个数由
    # equation 决定（语料基础表 VDV 2 个、附录 3 个），不按标称轴数硬卡。
    for term, role in zip(terms, roles):
        if _is_peram(role):
            continue
        if not term:
            raise SpecError(f"角色 {role!r} 的指标段为空")
        shape = role_shape(role)
        if len(set(shape)) != 1:
            raise SpecError(f"角色 {role!r} 的标称轴长 {list(shape)} 不等长，无法按标签绑定")
        for label in term:
            bind(label, shape[0], role)

    # peram 只补**余**：已被小张量钉住的标签沿用原值，不覆盖、不报冲突。
    # 语料里 peram 的指标段与 gamma/VDV/VVV 大面积共享（如 ``ac`` 同时是
    # ``gamma_M`` 与 ``peram_u``），"覆盖"会让同一条方程自相矛盾。
    for term, role in zip(terms, roles):
        if not _is_peram(role):
            continue
        shape = role_shape(role)
        if not term:
            raise SpecError(f"角色 {role!r} 的指标段为空")
        if len(term) != len(shape):
            raise SpecError(
                f"角色 {role!r} 的标称轴长 {list(shape)} 有 {len(shape)} 个轴，"
                f"指标段 {term!r} 有 {len(term)} 个标签")
        for label in term:
            sizes.setdefault(label, shape[-1])

    output = equation.split("->")[1]
    if output and not output.isalpha():
        raise SpecError(f"输出段只允许标签字母：{output!r}")
    missing = [label for label in output if label not in sizes]
    if missing:
        raise SpecError(f"输出标签 {missing} 无法由 roles 推出轴长")
    shapes = tuple(tuple(sizes[label] for label in term) for term in terms)
    return shapes, tuple(sizes[label] for label in output)


def _entries(path: Path) -> list[dict[str, object]]:
    """读 ``_wick_extra.jsonl``；文件不存在即无追加条目。"""
    if not path.exists():
        return []
    entries: list[dict[str, object]] = []
    for line_no, raw in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SpecError(f"{path.name} 第 {line_no} 行不是合法 JSON：{exc}")
        if not isinstance(entry, dict):
            raise SpecError(f"{path.name} 第 {line_no} 行不是 JSON 对象")
        entries.append(entry)
    return entries


def _base_count() -> int:
    """图纸条目数，用于把追加条目的 ``seq`` 标成合并后的真实序号。"""
    from . import wick_tasklist_gen

    return len(wick_tasklist_gen.parse_blueprint(wick_tasklist_gen.DEFAULT_SOURCE))


def _regen(extra: Path) -> int:
    """调用生成器重建 ``_wick_tasklist.py``（同一份追加文件）。"""
    from . import wick_tasklist_gen

    return wick_tasklist_gen.main(["--extra", str(extra)])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pyqcd.einsum.wick_tasklist_add",
        description="向 _wick_extra.jsonl 追加一条特化方程并重建 _wick_tasklist.py")
    parser.add_argument("spec", help="<描述> <case 名> <equation> <roles>")
    parser.add_argument("--extra", type=Path, default=EXTRA,
                        help=f"追加文件（缺省 {EXTRA.name}；演练时可指向副本）")
    parser.add_argument("--dry-run", action="store_true", help="只打印，不写盘")
    parser.add_argument("--no-regen", action="store_true",
                        help="只写追加文件，不调用生成器重建 _wick_tasklist.py")
    args = parser.parse_args(argv)

    try:
        section, subsection, equation, roles = parse_spec(args.spec)
        shapes, output_shape = infer_shapes(equation, roles)
        entries = _entries(args.extra)
        seq = _base_count() + len(entries) + 1
    except (SpecError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    print(f"equation : {equation}")
    print(f"section  : {section!r}")
    print(f"case     : {subsection!r}")
    print("roles    : " + ", ".join(
        f"{role}=[标称 {list(role_shape(role))} 实取 {list(shape)}]"
        for role, shape in zip(roles, shapes)))
    print(f"输出形状 : {list(output_shape)}")

    for entry in entries:
        if entry.get("equation") == equation and entry.get("section") == section:
            print(f"已存在：seq={entry.get('seq')} tid={entry.get('tid')} "
                  f"line={entry.get('line')}（equation + case 相同，未写入）")
            return 0

    entry = {"seq": seq, "tid": f"T{seq:05d}", "line": NON_BLUEPRINT_LINE,
             "section": section, "subtable": subsection, "equation": equation,
             "roles": list(roles)}
    line = json.dumps(entry, ensure_ascii=False)
    print(f"seq/tid  : {seq} {entry['tid']} line={NON_BLUEPRINT_LINE}"
          f"（追加条目 {len(entries)} → {len(entries) + 1}）")
    print(f"新增行   : {line}")
    if args.dry_run:
        print("dry-run：未写盘")
        return 0

    prefix = args.extra.read_text(encoding="utf-8") if args.extra.exists() else ""
    if prefix and not prefix.endswith("\n"):
        prefix += "\n"
    args.extra.write_text(prefix + line + "\n", encoding="utf-8")
    print(f"已写入 {args.extra}")

    if args.no_regen:
        print("未重建：需要时运行 python -m pyqcd.einsum.wick_tasklist_gen")
        return 0

    if args.extra.resolve() != EXTRA.resolve():
        print(f"未重建：追加文件不是缺省 {EXTRA.name}，请自行运行 "
              f"python -m pyqcd.einsum.wick_tasklist_gen --extra {args.extra}")
        return 0

    status = _regen(args.extra)
    if status != 0:
        print("错误：生成器重建失败", file=sys.stderr)
        return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
