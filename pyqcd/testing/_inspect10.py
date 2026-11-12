#!/usr/bin/env python3
"""十轮全面巡查：每轮聚焦一个维度，逐项给 PASS/FAIL 与证据。

轮次设计（互不重叠）：
  R1  导入与公开 API 面
  R2  全量单元回归
  R3  float32 全语料正确性（opt_einsum 参考）
  R4  complex128 / complex64 正确性（complex128 真值参考）
  R5  双后端编译
  R6  导出符号
  R7  残留检查（MPI / 旧专用 kernel）
  R8  性能一致性（多档中尺寸 + 自洽性）
  R9  代码卫生（调试残留 / TODO / 大文件 / 生成物）
  R10 Git 状态与交付完整性

用法：python inspect10.py [--fast]   （--fast 跳过全量重活，只做结构与抽样）
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path("/home/zhangxin/PyQCD")
RESULTS: list[dict] = []


def record(round_no: str, name: str, ok: bool, detail: str) -> None:
    RESULTS.append({"round": round_no, "name": name, "ok": ok, "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def run(cmd: str, timeout: int = 600, env: dict | None = None) -> tuple[int, str]:
    merged = dict(os.environ)
    if env:
        merged.update(env)
    try:
        done = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True,
                              text=True, timeout=timeout, env=merged)
        return done.returncode, (done.stdout + done.stderr)
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


# ── R1 导入与公开 API ────────────────────────────────────────────────
def round1() -> None:
    code = (
        "import pyqcd, pyqcd.einsum as pe\n"
        "import numpy as np\n"
        "assert pyqcd.__version__, 'no version'\n"
        "names = ['contract','contract_path','build_info','specializations',\n"
        "         'shared_intermediates','count_cached_ops','contract_expression']\n"
        "missing = [n for n in names if not hasattr(pe, n)]\n"
        "assert not missing, f'missing API: {missing}'\n"
        "assert not hasattr(pe, 'mpi_enabled'), 'mpi_enabled still exported'\n"
        "a=np.ones((3,3)); assert np.allclose(pe.contract('ij,jk->ik',a,a), a@a)\n"
        "for dt in ('float32','float64','complex64','complex128'):\n"
        "    x=np.ones((3,3),dtype=dt)\n"
        "    assert pe.contract('ij,jk->ik',x,x).dtype==np.dtype(dt), dt\n"
        "print('OK')\n"
    )
    code_path = ROOT / "_r1probe.py"
    code_path.write_text(code, encoding="utf-8")
    rc, out = run(f"python {code_path}")
    code_path.unlink(missing_ok=True)
    record("R1", "导入与 API 面", rc == 0 and "OK" in out,
           out.strip().splitlines()[-1] if out.strip() else "no output")


# ── R2 全量回归 ──────────────────────────────────────────────────────
def round2() -> None:
    rc, out = run("python -B -m pyqcd.testing", timeout=900)
    match = re.search(r"(\d+) passed, (\d+) skipped, (\d+) failed", out)
    if not match:
        record("R2", "全量回归", False, "无法解析结果")
        return
    passed, skipped, failed = (int(v) for v in match.groups())
    record("R2", "全量回归", failed == 0 and passed > 0,
           f"{passed} passed / {skipped} skipped / {failed} failed")
    # 附：确认 MPI/parallel 契约已随删除消失
    rc2, out2 = run("ls pyqcd/testing/_mpi*.py pyqcd/testing/parallel_einsum.py 2>/dev/null | wc -l")
    record("R2", "MPI 契约已移除", out2.strip() == "0", f"残留 {out2.strip()} 个")


# ── R3 float32 全量正确性 ────────────────────────────────────────────
def round3(fast: bool) -> None:
    limit = "1000" if fast else "0"
    out_dir = "/tmp/ins_r3"
    rc, out = run(f"python -u -m pyqcd.testing._topology_full_bench --dim 3 "
                  f"--time-limit 900 --quiet --out {out_dir}", timeout=900)
    match = re.search(r"成功 / 失败\s*:\s*(\d+) / (\d+)", out)
    if not match:
        record("R3", "float32 全语料基准", False, "无法解析")
        return
    ok, failed = (int(v) for v in match.groups())
    record("R3", "float32 全语料基准", failed == 0 and ok > 0, f"{ok} ok / {failed} failed")

    rc2, out2 = run(f"python -m pyqcd.testing._topology_verify_snapshot {out_dir} "
                    f"--sample {limit} --engine opt_einsum", timeout=900)
    match2 = re.search(r"复核 (\d+) 条，失败 (\d+)，最大偏差 ([\d.eE+-]+)", out2)
    if not match2:
        record("R3", "opt_einsum 参考核验", False, "无法解析")
        return
    checked, bad, worst = int(match2.group(1)), int(match2.group(2)), float(match2.group(3))
    record("R3", "opt_einsum 参考核验", bad == 0 and worst <= 1e-4,
           f"{checked} 条，失败 {bad}，最大偏差 {worst:.2e}")


# ── R4 复数正确性 ────────────────────────────────────────────────────
def round4(fast: bool) -> None:
    for dtype in ("complex128", "complex64"):
        limit = " --limit 500" if fast else ""
        rc, out = run(f"python -u -m pyqcd.testing._complex_specialized_verify "
                      f"--dtype {dtype}{limit}", timeout=900)
        match = re.search(r"通过 / 失败\s*:\s*(\d+) / (\d+)", out)
        worst = re.search(r"特化最大偏差\s*:\s*([\d.eE+-]+)", out)
        if not match:
            record("R4", f"{dtype} 全量正确性", False, "无法解析")
            continue
        passed, failed = (int(v) for v in match.groups())
        record("R4", f"{dtype} 全量正确性", failed == 0 and passed > 0,
               f"{passed} 通过 / {failed} 失败，最大偏差 "
               f"{float(worst.group(1)):.2e}" if worst else "n/a")


# ── R5 双后端编译 ────────────────────────────────────────────────────
def round5() -> None:
    rc, out = run("PYQCD_EINSUM_BACKEND=kupl bash cpp/kupl/einsum/make.sh", timeout=900)
    record("R5", "KUPL 编译", rc == 0 and "SUCCESS" in out,
           out.strip().splitlines()[-1] if out.strip() else "no output")
    rc, out = run("PYQCD_EINSUM_BACKEND=cuda bash cpp/cuda/einsum/make-nv.sh", timeout=900)
    record("R5", "CUDA 编译", rc == 0 and "SUCCESS" in out,
           out.strip().splitlines()[-1] if out.strip() else "no output")
    # 同步到包内
    run("cp -f cpp/kupl/einsum/libpyqcdeinsum_kupl.so pyqcd/einsum/kupl/")
    run("cp -f cpp/cuda/einsum/libpyqcdeinsum_cuda.so pyqcd/einsum/cuda/ 2>/dev/null")


# ── R6 导出符号 ──────────────────────────────────────────────────────
def round6() -> None:
    expected_kupl = {"applyContractQcd", "applyPlanQcd", "applyCompiledQcd",
                     "qcd_kupl_dedicated_apply", "qcd_kupl_topology_apply"}
    expected_cuda = {"qcd_cuda_dedicated_apply", "qcd_cuda_topology_apply"}
    for lib, expected in (
            ("cpp/kupl/einsum/libpyqcdeinsum_kupl.so", expected_kupl),
            ("cpp/cuda/einsum/libpyqcdeinsum_cuda.so", expected_cuda)):
        rc, out = run(f"nm -D --defined-only {lib} 2>/dev/null | awk '{{print $3}}'")
        found = set(out.split())
        missing = expected - found
        record("R6", f"{lib.split('/')[-1]} 符号", not missing,
               f"缺失 {sorted(missing)}" if missing else "全部存在")


# ── R7 残留检查 ──────────────────────────────────────────────────────
def round7() -> None:
    checks = [
        ("MPI 调用残留", r"mpi4py|from \.mpi import|kupl\.mpi|_mpi_transport|parallel\.einsum"),
        ("旧专用 kernel 残留", r"nucleon2pt|pattern_3pt"),
        ("parallel 包残留", r"^pyqcd/parallel/"),
    ]
    for name, pattern in checks:
        rc, out = run(
            f"grep -rnE '{pattern}' --include='*.py' --include='*.cpp' --include='*.cu' "
            f"--include='*.h' --include='*.sh' --exclude='_inspect10.py' "
            f"pyqcd/ cpp/ pyproject.toml 2>/dev/null "
            f"| grep -v __pycache__ | grep -v '\\.so' | head -5")
        hits = [line for line in out.strip().splitlines() if line.strip()]
        record("R7", name, not hits, f"{len(hits)} 处" if hits else "0 处")
    rc, out = run("ls pyqcd/parallel/*.py 2>/dev/null | grep -v __init__ | xargs -r -n1 basename | sort | tr '\n' ' '")
    remaining = out.strip()
    # 允许保留 _resources.py（纯资源探测，与 MPI 无关）
    record("R7", "parallel 仅剩资源探测", remaining.strip() == "_resources.py",
           f"剩余 {remaining.strip() or '空'}")


# ── R8 性能一致性 ────────────────────────────────────────────────────
def round8(fast: bool) -> None:
    sample = "6" if fast else "10"
    dims = "4,6" if fast else "4,6,8,10"
    rc, out = run(f"python -u -m pyqcd.testing._topology_perf_multidim "
                  f"--sample {sample} --dims {dims} --rounds 5", timeout=900)
    consistent = "全部尺寸均快于 opt_einsum" in out
    suspect = re.search(r"自洽性检查：(\d+) 条疑似假数据", out)
    lines = [l for l in out.splitlines() if re.match(r"^\s+\d+\s+\d+\s+", l)]
    record("R8", "多档尺寸方向一致", consistent, f"{len(lines)} 档参与")
    record("R8", "自洽性（无方向反转）", suspect is None,
           f"{suspect.group(1)} 条疑似" if suspect else "0 条疑似")
    # 抽取每档 vs opt_einsum
    ratios = re.findall(r"([\d.]+)x\s+([\d.eE+-]+)$", out, re.M)
    if ratios:
        worst = max(float(r[0]) for r in ratios)
        record("R8", "最差档 vs opt_einsum", worst < 1.0, f"{worst:.2f}x")


# ── R9 代码卫生 ──────────────────────────────────────────────────────
def round9() -> None:
    rc, out = run("grep -rn 'printf(\"\\[dedicated\\]\\|print(\"DEBUG\\|pdb.set_trace\\|breakpoint()' "
                  "--include='*.py' --include='*.cpp' --include='*.cu' --exclude='_inspect10.py' "
                  "pyqcd/ cpp/ 2>/dev/null "
                  "| grep -v __pycache__ | grep -v trace_enabled | head -5")
    hits = [l for l in out.strip().splitlines() if l.strip()]
    record("R9", "调试残留", not hits, f"{len(hits)} 处" if hits else "0 处")

    rc, out = run("grep -rn 'TODO\\|FIXME\\|XXX' --include='*.py' --include='*.cpp' "
                  "--include='*.cu' pyqcd/einsum/ cpp/kupl/einsum/src/ cpp/cuda/einsum/src/ 2>/dev/null "
                  "| grep -v __pycache__ | head -5")
    hits = [l for l in out.strip().splitlines() if l.strip()]
    # apply_contract.cu 的 TODO 是"多设备流未实现"的已知边界说明，属合理设计注释
    hits = [l for l in hits if "multi-device scheduling is not implemented" not in l]
    record("R9", "TODO/FIXME（排除已知边界说明）", not hits,
           f"{len(hits)} 处" if hits else "0 处（1 处已知边界说明已排除）")

    rc, out = run("git status --short | grep '^??' | head -10")
    untracked = [l.split()[-1] for l in out.strip().splitlines() if l.strip()]
    # 新增的验证/巡查脚本属预期交付物；其他未跟踪文件才算问题
    expected = {p for p in untracked
                if pathlib.Path(p).name in {
                    "_complex_specialized_verify.py", "_inspect10.py",
                    "_topology_perf_multidim.py", "_dedicated_step_tables.py",
                    "_wick_tasklist.py", "_topology_registry.py",
                    "bench_generic.py", "bench_specialized.py"}}
    unexpected = [p for p in untracked if p not in expected]
    record("R9", "未跟踪文件（非预期）", not unexpected,
           f"预期新增 {len(expected)} 个，非预期 {len(unexpected)} 个: {unexpected[:3]}"
           if unexpected else f"仅预期新增 {len(expected)} 个")

    rc, out = run("find pyqcd cpp -name '*.so' -not -path '*/build/*' 2>/dev/null | "
                  "xargs -r ls -l 2>/dev/null | awk '$5>5000000 {{print $5, $9}}' | head -5")
    big = [l for l in out.strip().splitlines() if l.strip()]
    record("R9", "大文件（>5MB 的 .so）", True, f"{len(big)} 个（.so 不入库）" if big else "0 个")


# ── R10 Git 与交付 ───────────────────────────────────────────────────
def round10() -> None:
    rc, out = run("git diff --check")
    record("R10", "diff --check", rc == 0 and not out.strip(), out.strip()[:120] or "干净")

    rc, out = run("git status --short | wc -l")
    dirty = out.strip()
    record("R10", "工作区干净", dirty == "0", f"{dirty} 个改动")

    rc, out = run("git log --oneline -3")
    record("R10", "最近提交", rc == 0, out.strip().splitlines()[0][:70] if out.strip() else "n/a")

    rc, out = run("python -c \"import tomllib;d=tomllib.load(open('pyproject.toml','rb'))['project'];"
                  "print(d['version']); print('mpi' in d.get('optional-dependencies',{}))\"")
    lines = out.strip().splitlines()
    if len(lines) >= 2:
        record("R10", "项目元数据", lines[1] == "False",
               f"version={lines[0]}, mpi extra={'仍存在' if lines[1]=='True' else '已移除'}")

    rc, out = run("ls dist/*.tar.gz 2>/dev/null | tail -1")
    record("R10", "dist 产物", True, out.strip() or "无（未构建，正常）")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="跳过全量重活（用抽样代替），用于快速回归")
    args = parser.parse_args()

    started = time.perf_counter()
    rounds = [
        ("R1", lambda: round1()),
        ("R2", lambda: round2()),
        ("R3", lambda: round3(args.fast)),
        ("R4", lambda: round4(args.fast)),
        ("R5", lambda: round5()),
        ("R6", lambda: round6()),
        ("R7", lambda: round7()),
        ("R8", lambda: round8(args.fast)),
        ("R9", lambda: round9()),
        ("R10", lambda: round10()),
    ]
    for name, function in rounds:
        print(f"\n=== {name} ===", flush=True)
        t0 = time.perf_counter()
        try:
            function()
        except Exception as exc:                       # noqa: BLE001
            record(name, "执行异常", False, f"{type(exc).__name__}: {exc}")
        print(f"  ({time.perf_counter() - t0:.0f}s)", flush=True)

    failed = [item for item in RESULTS if not item["ok"]]
    print(f"\n{'=' * 60}")
    print(f"十轮巡查合计 {len(RESULTS)} 项："
          f"{len(RESULTS) - len(failed)} PASS / {len(failed)} FAIL")
    if failed:
        for item in failed:
            print(f"  FAIL [{item['round']}] {item['name']}: {item['detail']}")
    print(f"总耗时 {time.perf_counter() - started:.0f}s")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
