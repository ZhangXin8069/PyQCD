#!/usr/bin/env python3
"""用户文档十轮巡查：验证文档与实现一致、无弃用残留、示例可执行。

十轮设计：
  D1  页面集合与 toctree 一致（en/zh 两侧）
  D2  双语代码块结构等价 + 英文页无 CJK
  D3  文档示例可执行（docs_examples）
  D4  文档引用的 API 真实存在（_docs_api_contract）
  D5  文档中的命令入口真实存在（模块/脚本可解析）
  D6  弃用内容无残留（einsums / init+run / plan_einsums / bench-collective / mpi）
  D7  特化路径附录：条数与实现一致、覆盖 8638 条
  D8  内核首次命中提醒：两类提示都存在且可关闭（NOTICE=0）
  D9  基准脚本可运行且删除的历史用例不再出现
  D10 Sphinx -W 双站构建零告警

用法：python -m pyqcd.testing._inspect10_docs [--fast]
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path("/home/zhangxin/PyQCD")
RESULTS: list[dict] = []


def record(round_no: str, name: str, ok: bool, detail: str) -> None:
    RESULTS.append({"round": round_no, "name": name, "ok": ok, "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def run(cmd: str, timeout: int = 300, env: dict | None = None) -> tuple[int, str]:
    merged = dict(os.environ, PYTHONPATH=str(ROOT))
    if env:
        merged.update(env)
    try:
        done = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True,
                              text=True, timeout=timeout, env=merged)
        return done.returncode, (done.stdout + done.stderr)
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


# ── D1 页面集合与 toctree ───────────────────────────────────────────
def d1() -> None:
    def toc(path):
        text = pathlib.Path(path).read_text(encoding="utf-8")
        match = re.search(r"\.\. toctree::\n(?:[ \t]*:[^\n]+\n)*\n((?:[ \t]+\S+\n)+)", text)
        return [line.strip() for line in match.group(1).strip().splitlines()]

    en, zh = toc(ROOT / "docs/en/index.rst"), toc(ROOT / "docs/zh/index.rst")
    disk_en = {p.stem for p in (ROOT / "docs/en").glob("*.md")}
    disk_zh = {p.stem for p in (ROOT / "docs/zh").glob("*.md")}
    record("D1", "en/zh toctree 一致", en == zh, f"en={en} zh={zh}")
    record("D1", "toctree 覆盖磁盘页面", set(en) == disk_en == disk_zh,
           f"缺 {sorted((set(en) ^ disk_en) | (set(en) ^ disk_zh))}")


# ── D2 双语代码块 ───────────────────────────────────────────────────
def d2() -> None:
    import ast

    block = re.compile(r"```[a-z]*\n(.*?)```", re.S)
    cjk = re.compile(r"[\u4e00-\u9fff]")

    def signature(source):
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
                body = getattr(node, "body", None)
                if (body and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    node.body = body[1:] or [ast.Pass()]
        return ast.dump(tree, annotate_fields=True, include_attributes=False)

    bad = []
    # 只比较 Python 代码块：shell 命令快的注释天然是中英对照，不要求 AST 等价。
    py_block = re.compile(r"```python\n(.*?)```", re.S)
    for page in sorted(p.stem for p in (ROOT / "docs/en").glob("*.md")):
        en = py_block.findall((ROOT / "docs/en" / f"{page}.md").read_text(encoding="utf-8"))
        zh_path = ROOT / "docs/zh" / f"{page}.md"
        if not zh_path.is_file():
            bad.append(f"{page}: 中文缺页")
            continue
        zh = py_block.findall(zh_path.read_text(encoding="utf-8"))
        same = sum(signature(a) == signature(b) for a, b in zip(en, zh))
        localized = not any(cjk.search(block_) for block_ in en)
        if not (len(en) == len(zh) and same == len(en) and localized):
            bad.append(f"{page}: en={len(en)} zh={len(zh)} same={same} "
                       f"localized={localized}")
    record("D2", "双语代码块等价 + 英文页无 CJK", not bad,
           "; ".join(bad[:3]) if bad else "全部页面一致")


# ── D3 文档示例可执行 ───────────────────────────────────────────────
def d3(fast: bool) -> None:
    rc, out = run("python -m pyqcd.testing.docs_examples --keep-going", timeout=900)
    match = re.search(r"total: ok=(\d+) skipped=(\d+) failed=(\d+)", out)
    record("D3", "英文页示例可执行", bool(match) and match.group(3) == "0",
           match.group(0) if match else out.strip()[-100:])
    rc, out = run("python -m pyqcd.testing.docs_examples --lang zh --keep-going", timeout=900)
    match = re.search(r"total: ok=(\d+) skipped=(\d+) failed=(\d+)", out)
    record("D3", "中文页示例可执行", bool(match) and match.group(3) == "0",
           match.group(0) if match else out.strip()[-100:])


# ── D4 文档引用的 API 存在 ──────────────────────────────────────────
def d4() -> None:
    rc, out = run("python -m pyqcd.testing._docs_api_contract", timeout=300)
    record("D4", "文档 API 引用真实存在", rc == 0,
           out.strip().splitlines()[-1][:120] if out.strip() else "no output")


# ── D5 文档命令入口存在 ─────────────────────────────────────────────
def d5() -> None:
    commands = set()
    for page in list((ROOT / "docs/en").glob("*.md")) + list((ROOT / "docs/zh").glob("*.md")):
        text = page.read_text(encoding="utf-8", errors="ignore")
        commands.update(re.findall(r"python -m (pyqcd[\w.]*)", text))
    broken = []
    for module in sorted(commands):
        rc, _ = run(f"python -c \"import importlib,sys;"
                    f"m=importlib.import_module('{module}');"
                    f"print('OK')\"", timeout=120)
        if rc != 0:
            broken.append(module)
    record("D5", "文档命令入口可导入", not broken,
           f"失效 {broken[:5]}" if broken else f"{len(commands)} 个入口全部可导入")


# ── D6 弃用内容无残留 ───────────────────────────────────────────────
def d6() -> None:
    patterns = {
        # 只禁 pyqcd 自己的 API 名；第三方库 Einsums 与其桥目录 cpu-einsums 保留
        "einsums / plan_einsums":
            r"pe\.einsums|pyqcd.*einsums\(|plan_einsums|"
            r"`einsums`|parallel_einsums|bench-einsums",
        "init + run / plan_tasks": r"plan_tasks|`init`\s*\+\s*`run`",
        "bench-collective": r"bench-collective|bench_collective",
        "MPI 调度入口": r"plan_parallel|run_meta_task|run_parallel_pipeline",
    }
    for label, pattern in patterns.items():
        rc, out = run(f"grep -rnE '{pattern}' --include='*.md' docs/en/ docs/zh/ "
                      f"2>/dev/null | grep -v 'appendix.md' | head -5")
        hits = [line for line in out.strip().splitlines() if line.strip()]
        record("D6", f"无 {label} 残留", not hits,
               f"{len(hits)} 处: {hits[:2]}" if hits else "0 处")


# ── D7 特化路径附录 ─────────────────────────────────────────────────
def d7() -> None:
    import pyqcd.einsum._wick_tasklist as tasks
    from pyqcd.einsum import _topology_registry

    for language in ("en", "zh"):
        page = ROOT / "docs" / language / "appendix_specialized.md"
        if not page.is_file():
            record("D7", f"{language} 特化路径附录存在", False, "缺文件")
            continue
        text = page.read_text(encoding="utf-8")
        has_total = str(tasks.N_TASKS) in text
        has_digest = str(_topology_registry.N_ENTRIES) in text
        has_arity = all(str(arity) in text
                        for arity in sorted({t.arity for t in tasks.TASKS}))
        record("D7", f"{language} 附录数字与实现一致",
               has_total and has_digest and has_arity,
               f"total={tasks.N_TASKS} digests={_topology_registry.N_ENTRIES} "
               f"arity覆盖={has_arity}")


# ── D8 首次命中提醒 ─────────────────────────────────────────────────
def d8() -> None:
    probe = ROOT / "_d8probe.py"
    probe.write_text('''
import io, contextlib
import numpy as np
import pyqcd.einsum as pe

a = np.ones((4, 4)); b = np.ones((4, 4))
eq_spec = ("cwdx,eufv,msnt,okpl,igjh,qarb,ac,mo,Ggi,Hkq,su,bd,fnp,Mhj,Mlr,tvx->MGHew")
ops_spec = [np.random.default_rng(i).standard_normal(
                tuple(2 if c.isupper() else 3 for c in t))
            for i, t in enumerate(eq_spec.split("->")[0].split(","))]

# 1) 通用路径首次命中必须提醒
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    pe.contract("ij,jk->ik", a, b)
generic_text = buf.getvalue()
print("GENERIC", "通用路径" in generic_text or "generic" in generic_text.lower())

# 2) 特化首次命中必须提醒
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    pe.contract(eq_spec, *ops_spec)
special_text = buf.getvalue()
print("SPECIAL", ("专用路径" in special_text)
      or ("specialized path" in special_text))
print("KIND", "kernel" in special_text or "plan-cache" in special_text)
''', encoding="utf-8")
    rc, out = run(f"python {probe}")
    probe.unlink(missing_ok=True)
    record("D8", "通用内核首次命中提醒", "GENERIC True" in out,
           "有提醒" if "GENERIC True" in out else out.strip()[-100:])
    record("D8", "特化内核首次命中提醒", "SPECIAL True" in out and "KIND True" in out,
           "有提醒且标注 kernel/plan-cache")

    # 3) NOTICE=0 必须全部静音
    probe2 = ROOT / "_d8probe2.py"
    probe2.write_text('''
import io, contextlib
import numpy as np
import pyqcd.einsum as pe
a = np.ones((4, 4)); b = np.ones((4, 4))
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    pe.contract("ij,jk->ik", a, b)
print("SILENT", buf.getvalue().strip() == "")
''', encoding="utf-8")
    rc, out = run(f"python {probe2}", env={"PYQCD_EINSUM_NOTICE": "0"})
    probe2.unlink(missing_ok=True)
    record("D8", "NOTICE=0 可完全关闭", "SILENT True" in out,
           "已静音" if "SILENT True" in out else out.strip()[-100:])


# ── D9 基准脚本 ─────────────────────────────────────────────────────
def d9() -> None:
    for language in ("en", "zh"):
        text = (ROOT / "docs" / language / "performance.md").read_text(encoding="utf-8")
        removed = re.search(r"nucleon 2pt \(kernel\)|gamma sandwich \(kernel\)", text)
        record("D9", f"{language} 性能页无已删用例", removed is None,
               "无" if removed is None else f"仍存在: {removed.group(0)}")
    # 附录里保留的历史行必须带标注
    for language, needle in (("en", "Historic data"), ("zh", "历史数据")):
        text = (ROOT / "docs" / language / "appendix.md").read_text(encoding="utf-8")
        record("D9", f"{language} 附录历史数据已标注", needle in text,
               "已标注" if needle in text else "缺标注")
    # 文档里的可复现命令必须真能跑（抽 3 个）
    for cmd in ("python -m pyqcd.einsum._specialize",
                "python -m pyqcd.testing._topology_verify_snapshot --help",
                "python -m pyqcd.testing._topology_perf_multidim --help"):
        rc, _ = run(cmd, timeout=120)
        record("D9", f"入口可运行: {cmd.split()[-1]}", rc == 0, f"exit={rc}")


# ── D10 Sphinx -W 双站 ──────────────────────────────────────────────
def d10(fast: bool) -> None:
    for language in ("en", "zh"):
        rc, out = run(
            f"python -m sphinx -b html -d /tmp/ins_{language}/doctrees -W "
            f"docs/{language} /tmp/ins_{language}/html", timeout=600)
        ok = rc == 0 and "build succeeded" in out
        tail = [l for l in out.splitlines() if "build succeeded" in l or "ERROR" in l]
        record("D10", f"sphinx -W ({language})", ok,
               tail[-1][:100] if tail else out.strip()[-100:])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    rounds = [("D1", d1), ("D2", d2), ("D3", lambda: d3(args.fast)),
              ("D4", d4), ("D5", d5), ("D6", d6), ("D7", d7),
              ("D8", d8), ("D9", d9), ("D10", lambda: d10(args.fast))]
    for name, function in rounds:
        print(f"\n=== {name} ===", flush=True)
        try:
            function()
        except Exception as exc:                       # noqa: BLE001
            record(name, "执行异常", False, f"{type(exc).__name__}: {exc}")

    failed = [item for item in RESULTS if not item["ok"]]
    print(f"\n{'=' * 60}")
    print(f"用户文档十轮巡查：{len(RESULTS)} 项，"
          f"{len(RESULTS) - len(failed)} PASS / {len(failed)} FAIL")
    for item in failed:
        print(f"  FAIL [{item['round']}] {item['name']}: {item['detail']}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
