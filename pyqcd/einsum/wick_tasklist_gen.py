"""从图纸生成 :mod:`pyqcd.einsum._wick_tasklist`，并合并 ``_wick_extra.jsonl``。

:mod:`pyqcd.einsum._wick_tasklist` 是**生成物**：它的大块 ``_BLOB`` 由本模块从图纸
``refer/sush/wick_12cases.md`` 解析重建。用法::

    python -m pyqcd.einsum.wick_tasklist_gen            # 重建 _wick_tasklist.py
    python -m pyqcd.einsum.wick_tasklist_gen --check    # 只比对，不写盘
    python -m pyqcd.einsum.wick_tasklist_gen --dry-run  # 打印统计

图纸解析
--------

任务行形态（两张表——「基础表」与「附录」——同格式）::

    | T00001 | 0 | `cadb,menf,ac,em,bd,fn->` | `peram_u, peram_u, gamma_M, gamma_MH, VDV_0, VDV_1` |

即 ``| tid | 图号 | `equation` | `roles` |``。两条关键事实：

* ``T`` 号在两张表里**各自从 T00001 起编号**（720 个重复），所以主键只能用连续
  ``seq``（1…N），不能由 T 号推；
* ``line`` 是**图纸文件的物理行号**（1-based），表格间夹着标题/说明/空行，故跳号。

``section`` / ``subsection`` 取自表格前的标题层级，不做任何自造命名：

* 基础表：``## N. <情形名>`` → ``section``；``subsection`` 为空；
* 附录：``## 动量指标与流分量指标`` → ``section``，其下 ``#### 情形 N：<情形名>``
  → ``subsection``。

``roles`` 在图纸里写作 ``, `` 分隔（``peram_u, peram_u``），落表时改为 ``,``
分隔（``peram_u,peram_u``），**角色名内部空格原样保留**（``C * gamma_B``）——这样
生成结果与图纸的字段值一一对应，``--check`` 才有意义。

``--normalize`` 另有一套**去空格**规范（``1. 介子两点函数`` → ``1-介子两点函数``、
``情形 7：五夸克…`` → ``情形7-五夸克…``、``C * gamma_B`` → ``C*gamma_B``），写盘前
逐字段断言"不含空格、``roles`` 每个 role 匹配 ``^(C\\*)?\\w+$``"。它**默认开启**
（``--no-normalize`` 关掉；工作区的 ``_BLOB`` 就是规范化后的形态）：整表本来就都要
改格式，``role_name`` 又已经 ``replace(" ", "")``，两条拼写消费侧都接受。

**"整表都改了格式"时的验收判据**是 :func:`normalize_row` 的**逐行不动点**——
``normalize(旧行) == 新行`` 对**全部 8638 行**成立（实测 8638/8638）。不要用"逐字节
等于图纸"来判：规范化正是为了不等；也不要用 ``旧行.replace(" ", "") == 新行``——
那条规则会把 ``. `` 与 ``：`` 一起删掉，与"编号符换 ``-``"不是同一个变换。

追加条目
--------

``_wick_extra.jsonl`` 每行一个 JSON 对象::

    {"seq": 8639, "tid": "T08639", "line": 0, "section": "…",
     "subtable": "…", "equation": "ij,jk->ik", "roles": ["…", "…"]}

生成时 = 「图纸条目」+「追加条目」，``seq`` 从 ``N_base + 1`` 顺延重算，``line`` 用
``0`` 作哨兵（图纸行号从 1 起，``0`` 不冒充任何图纸行，也不进
:func:`pyqcd.einsum._wick_tasklist.base_table_count` 的基础表判据）。

生成只替换 ``_BLOB`` 的内容与文档里的两处条数，**其余逐字保留**——docstring 的
说明段落由人维护，生成器不改写它。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE.parents[1]
DEFAULT_SOURCE = REPO_ROOT / "refer" / "sush" / "wick_12cases.md"
DEFAULT_TARGET = PACKAGE / "_wick_tasklist.py"
DEFAULT_EXTRA = PACKAGE / "_wick_extra.jsonl"

BLOB_ANCHOR = '_BLOB = """'
FIELD_COUNT = 7

_HEADING = re.compile(r"^(#{2,4})\s+(\S.*?)\s*$")
_TASK_ROW = re.compile(
    r"^\|\s*(T\d+)\s*\|\s*(\d+)\s*\|\s*`([^`]*)`\s*\|\s*`([^`]*)`\s*\|\s*$")

_COUNT_PATTERNS = (
    (re.compile(r"共 \*\*\d+\*\* 条"), "共 **{total}** 条"),
    (re.compile(r"（1…\d+）"), "（1…{total}）"),
)
# 注意**不要**把"但只有 9142 行"那句加进来：它讲的是**图纸的物理行号**
# （表格间夹标题/空行，故跳号），与条目数无关；早期版本替换过它，属于改错。

# ``--normalize`` 的规范化规则（见 ``logs/kf/TASKLIST_GEN.md`` 的 ★★★★ 节）：
# ``1. 介子两点函数`` → ``1-介子两点函数``、``情形 7：五夸克…`` → ``情形7-五夸克…``、
# ``C * gamma_B`` → ``C*gamma_B``。默认**开启**（``--no-normalize`` 关），
# 现行 ``_BLOB`` 即规范化后的形态。
_SECTION_RULE = re.compile(r"^(\d+)\. ")
_SUBSECTION_RULE = re.compile(r"^情形 (\d+)：")
_ROLE_RULE = re.compile(r"^(C\*)?\w+$")


class BlueprintError(ValueError):
    """图纸或追加文件不合约定（退出码 2，不写盘）。"""


class NormalizationError(BlueprintError):
    """``--normalize`` 后的字段仍含空格（违反硬性规范）。"""


def parse_blueprint(source: Path) -> list[tuple[str, int, str, str, str, str]]:
    """解析图纸，返回 ``(tid, line, section, subsection, equation, roles)``。"""
    rows: list[tuple[str, int, str, str, str, str]] = []
    section = ""
    subsection = ""
    for line_no, raw in enumerate(
            source.read_text(encoding="utf-8").splitlines(), 1):
        heading = _HEADING.match(raw)
        if heading is not None:
            level = len(heading.group(1))
            title = heading.group(2)
            if level == 2:
                section, subsection = title, ""
            elif level == 4:
                subsection = title
            else:
                subsection = ""
            continue
        row = _TASK_ROW.match(raw)
        if row is None:
            continue
        equation = row.group(3).strip()
        if equation.count("->") != 1:
            raise BlueprintError(f"图纸第 {line_no} 行的 equation 不合法：{equation!r}")
        roles = ",".join(part.strip() for part in row.group(4).split(","))
        rows.append((row.group(1), line_no, section, subsection, equation, roles))
    if not rows:
        raise BlueprintError(f"{source} 里没有解析到任何任务行")
    return rows


def load_extra(path: Path) -> list[dict[str, object]]:
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
            raise BlueprintError(f"{path.name} 第 {line_no} 行不是合法 JSON：{exc}")
        if not isinstance(entry, dict):
            raise BlueprintError(f"{path.name} 第 {line_no} 行不是 JSON 对象")
        entries.append(entry)
    return entries


def _extra_row(entry: dict[str, object], seq: int) -> str:
    equation = str(entry.get("equation", "")).strip()
    if equation.count("->") != 1:
        raise BlueprintError(f"追加条目 seq={seq} 的 equation 不合法：{equation!r}")
    roles = entry.get("roles", [])
    if isinstance(roles, str):
        pieces = [part.strip() for part in roles.split(",")]
    else:
        pieces = [str(role).strip() for role in roles]           # type: ignore[union-attr]
    if not pieces or not all(pieces):
        raise BlueprintError(f"追加条目 seq={seq} 的 roles 为空或有空项")
    tid = str(entry.get("tid") or f"T{seq:05d}")
    return "\t".join([str(seq), tid, "0", str(entry.get("section", "")),
                      str(entry.get("subtable", "")), equation, ",".join(pieces)])


def normalize_row(fields: list[str]) -> list[str]:
    """按 ``--normalize`` 规范去空格（``seq``/``tid``/``line``/``equation`` 不变）。"""
    fields = list(fields)
    fields[3] = _SECTION_RULE.sub(r"\1-", fields[3])
    if fields[4]:
        fields[4] = _SUBSECTION_RULE.sub(r"情形\1-", fields[4])
    fields[6] = ",".join(role.replace(" ", "") for role in fields[6].split(","))
    for name, value in zip(("section", "subsection", "equation", "roles"), fields[3:]):
        if " " in value:
            raise NormalizationError(
                f"seq={fields[0]} 的 {name} 取值为 {value!r}，规范化后仍含空格")
    for role in fields[6].split(","):
        if not _ROLE_RULE.match(role):
            raise NormalizationError(f"seq={fields[0]} 的 role {role!r} 不合规范")
    return fields


def build_rows(source: Path, extra: Path, normalize: bool = True) -> list[str]:
    """图纸条目 + 追加条目 → ``_BLOB`` 的行列表。"""
    blueprint = parse_blueprint(source)
    # ``seq`` 只能由 enumerate 连续给出：图纸的 T 号在「基础表」与「附录」里各自从
    # T00001 起编（720 个重复），拿 T 号推序号必然撞主键。追加条目从 ``N_base+1``
    # 顺延，同样不借 T 号。
    rows = ["\t".join([str(seq), tid, str(line_no), section, subsection,
                       equation, roles])
            for seq, (tid, line_no, section, subsection, equation, roles)
            in enumerate(blueprint, 1)]
    for offset, entry in enumerate(load_extra(extra), 1):
        rows.append(_extra_row(entry, len(blueprint) + offset))
    if normalize:
        rows = ["\t".join(normalize_row(row.split("\t"))) for row in rows]
    for row in rows:
        if len(row.split("\t")) != FIELD_COUNT:
            raise BlueprintError(f"字段数 {len(row.split(chr(9)))} != {FIELD_COUNT}：{row!r}")
    return rows


def _blob_parts(text: str) -> tuple[str, str, str]:
    """把目标文件切成 ``(前, _BLOB 内容, 后)``，``前`` 含起始锚点。"""
    if BLOB_ANCHOR not in text:
        raise BlueprintError(f"目标文件缺少 {BLOB_ANCHOR!r}，不是本生成器的产物")
    start = text.index(BLOB_ANCHOR) + len(BLOB_ANCHOR)
    end = text.index('"""', start)
    return text[:start], text[start:end], text[end:]


def _bump_counts(head: str, total: int) -> str:
    for pattern, template in _COUNT_PATTERNS:
        head = pattern.sub(template.format(total=total), head, count=1)
    return head


def render(text: str, rows: list[str]) -> str:
    """用 ``rows`` 替换 ``_BLOB`` 并同步文档里的条数，其余逐字保留。

    **不要**改成"按模板重新渲染整个文件"：``Task`` 类、``_tasks()``、``lookup`` 这些
    骨架是手写维护的，重渲染会引入"骨架漂移"这一整类失败（生成器写出与模板不同的
    代码，而 diff 里看不出来）。这里只做"切开 → 换中间 → 拼回"。
    """
    head, _blob, tail = _blob_parts(text)
    return _bump_counts(head, len(rows)) + "\n".join(rows) + tail


def generate(source: Path, target: Path, extra: Path,
             normalize: bool = True) -> tuple[str, int]:
    """生成目标文件内容（不写盘），返回 ``(新内容, 条目数)``。"""
    rows = build_rows(source, extra, normalize)
    return render(target.read_text(encoding="utf-8"), rows), len(rows)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pyqcd.einsum.wick_tasklist_gen",
        description="从图纸重建 _wick_tasklist.py（合并 _wick_extra.jsonl 追加条目）")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help=f"图纸（缺省 {DEFAULT_SOURCE}）")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET,
                        help=f"生成目标（缺省 {DEFAULT_TARGET.name}）")
    parser.add_argument("--extra", type=Path, default=DEFAULT_EXTRA,
                        help=f"追加条目（缺省 {DEFAULT_EXTRA.name}；不存在即无追加）")
    parser.add_argument("--check", action="store_true",
                        help="只比对目标文件与生成结果，不一致时退出 1")
    parser.add_argument("--dry-run", action="store_true", help="只打印，不写盘")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false",
                        help="保留图纸原样（含空格）；默认**开启**规范化："
                             "section `1. X`→`1-X`、subsection `情形 N：X`→`情形N-X`、"
                             "roles 去空格（硬性规范，见 logs/kf/TASKLIST_GEN.md ★★★★）")
    args = parser.parse_args(argv)

    try:
        new_text, total = generate(args.source, args.target, args.extra,
                                    args.normalize)
    except (BlueprintError, OSError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    old_text = args.target.read_text(encoding="utf-8")
    n_extra = len(load_extra(args.extra))
    print(f"图纸     : {args.source}")
    print(f"追加     : {args.extra}（{n_extra} 条）")
    print(f"条目数   : {total}（图纸 {total - n_extra} + 追加 {n_extra}）")
    print(f"规范     : {'去空格（--normalize）' if args.normalize else '原样（与图纸一致）'}")
    print(f"sha256   : 现有 {_digest(old_text)}")
    print(f"           生成 {_digest(new_text)}")
    if new_text == old_text:
        print("一致     : 生成结果与现有文件逐字节相同")
        return 0
    print("不一致   : 生成结果与现有文件不同")
    if args.check:
        return 1
    if args.dry_run:
        print("dry-run：未写盘")
        return 0
    args.target.write_text(new_text, encoding="utf-8")
    print(f"已写入 {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
