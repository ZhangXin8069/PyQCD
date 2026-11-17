"""验收：画像学会 ``generic``，默认调用命中并施加，``--tune`` 真的写多行。

复现用户实跑的两条命令（真实方程是 10 操作数的语料外方程，这里取结构同形的 7 操作数
方程 ``ambn,codp,egfh,ce,mo,bdf,nph->ga``、形状全 2 做等价类比）：

1. **种子**：``contract(..., profile=True, profile_dir=tmp, kernel_force="generic",
   as_equivalence=True)`` → ``tmp/profile.csv`` 存在且至少一行 ``kernel=generic``；
2. **命中并施加**：同一方程再调一次（不带 ``kernel_force``）→ ``last_kernel()`` 报
   ``requested=()``（generic 被施加的标志；``ran=None`` 是正常的，generic 不跑专用
   内核），数值与 ``np.einsum`` 对拍 ``rel <= 1e-12``；
3. **输出形态**：命中提示**恰好一行**，含 ``profile.csv:<数据行号>``，且不含
   ``specialized path:``（generic 下不该做特化匹配）；
4. **反向证明**：不带 ``kernel_force`` 且缓存为空时 ``generic path:`` 照样出现
   ——第 3 步的安静来自"命中已施加"，不是提示被全局关掉；
5. **``--tune`` 写多行**：以 CLI 子进程跑 ``python -m pyqcd.einsum.test --tune``
   （``--profile-dir tmp2 --rounds 1 --warmup 1``），``profile.csv`` 数据行数 > 1
   （修前是 1/30）。

全程只用一次性临时目录（不碰仓库根的 ``./.cache``），也**不做任何计时断言**：这里判的
是"是否写入 / 是否命中 / 输出形态 / 数值一致"。断言失败即实现缺口，本脚本不放宽、不删
步骤。

    python -m pyqcd.testing.einsum.acceptance_cache_tune
"""

from __future__ import annotations

import contextlib
import csv
import io
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

import pyqcd.einsum as pe
from pyqcd.einsum import _api, _profile_cache, _specialize

#: 语料之外的方程（``_specialize.match`` 不命中），但存在等价改写（输出换序）——
#: 于是"命中施加"与"通用路径提示"两条路都能在同一个方程上验到。
EQUATION = "ambn,codp,egfh,ce,mo,bdf,nph->ga"
SHAPES = ((2, 2, 2, 2), (2, 2, 2, 2), (2, 2, 2, 2), (2, 2), (2, 2), (2, 2, 2), (2, 2, 2))
SHAPE_SPEC = ",".join("[" + ",".join(str(dim) for dim in shape) + "]" for shape in SHAPES)
SHAPE_SPEC += "->[2,2]"

#: 数值对拍门限（与库里其它对拍脚本同口径）。
TOL = 1e-12

#: CLI 子进程的工作目录基座：``-m pyqcd.einsum.test`` 要能 import 到本仓库的包。
REPO_ROOT = Path(__file__).resolve().parents[3]

#: ``--tune`` 子进程的上限（30 组 × (warmup 1 + rounds 1) 次小方程调用）。
TUNE_TIMEOUT_S = 900


def _read_profile(path: str) -> list[dict[str, str]]:
    """读画像主表的数据行（文件不存在即"没有行"）。"""
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _relative(got: Any, expected: Any) -> float:
    """相对差（Frobenius，按参照范数归一）；参照为零时退化成绝对差。"""
    delta = float(np.linalg.norm(np.asarray(got) - np.asarray(expected)))
    scale = float(np.linalg.norm(np.asarray(expected)))
    return delta / scale if scale else delta


class _Recorder:
    """逐项断言记录器：打印 ``PASS``/``FAIL`` 并累计失败数。"""

    def __init__(self) -> None:
        self.total = 0
        self.failed = 0

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.total += 1
        if not ok:
            self.failed += 1
        suffix = f"  {detail}" if detail else ""
        print(f"{'PASS' if ok else 'FAIL'} {name}{suffix}", flush=True)
        return bool(ok)


def _only_stdout(call) -> tuple[Any, str]:
    """跑一次调用并只捕获 stdout（提示是 ``print(flush=True)`` 出去的）。"""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = call()
    return result, buffer.getvalue()


def _step_seed(rec: _Recorder, operands: list[Any], table_dir: str) -> list[dict[str, str]]:
    """步骤 1：``--kernel-force generic`` 必须把 ``kernel=generic`` 写进画像。"""
    print("[1] 种子：--kernel-force generic 记录到画像", flush=True)
    pe.contract(EQUATION, *operands, profile=True, profile_dir=table_dir,
                kernel_force="generic", as_equivalence=True)
    # 两条命令是**两个进程**：第一条命令的观测靠退出时的 flush 落盘。这里同进程，
    # 所以显式落盘一次，才谈得上"下一条命令读得到"。
    flushed = _profile_cache.flush()
    profile_csv = os.path.join(table_dir, _profile_cache.PROFILE_NAME)
    exists = os.path.exists(profile_csv)
    rows = _read_profile(profile_csv)
    generic = [row for row in rows
               if str(row.get("kernel", "")).strip().lower() == "generic"]
    rec.check("1a profile.csv 落盘", exists, f"{profile_csv} flushed={flushed}")
    rec.check("1b 至少一行 kernel=generic", bool(generic),
              f"rows={len(rows)} generic={len(generic)}")
    return rows


def _step_hit_and_output(rec: _Recorder, operands: list[Any], expected: Any,
                         table_dir: str, rows: list[dict[str, str]]) -> None:
    """步骤 2 + 3：默认调用命中并把 generic 施加下去，且只打一行带行号的提示。

    ``requested=()`` 是"画像把 generic 真的施加了"的痕迹：施加后它的取值会下发成
    本次调用的 ``kernel_force``（走等价改写时则是内层那次调用，它是最后写
    ``last_kernel()`` 的一方）；调用方没要求、画像也没命中时这个字段是 ``None``。
    """
    print("[2] 命中并施加：不带 kernel_force 的默认调用", flush=True)
    # 索引里的行要带上 profile.csv 的数据行号，命中提示才报得出 ``profile.csv:<行号>``；
    # 而"内存里刚记下、还没进过文件"的行没有行号。重读一次文件 = 第二条命令的视角。
    _profile_cache.load(force=True)
    # 让下面那条判定真的来自本次调用，而不是步骤 1 留下的记录（generic 已经写过一次）。
    _api._LAST_KERNEL = None
    result, captured = _only_stdout(
        lambda: pe.contract(EQUATION, *operands, profile=True, profile_dir=table_dir))

    record = _api.last_kernel()
    rec.check("2a 命中后施加 generic（requested=()）",
              record is not None and record.get("requested") == (),
              f"last_kernel={record}")
    rec.check("2b 未跑专用内核（ran=None）",
              record is not None and record.get("ran") is None,
              f"ran={(record or {}).get('ran')!r}")
    rel = _relative(result, expected)
    rec.check("2c 数值对拍 rel<=1e-12", rel <= TOL, f"rel={rel:.3e}")

    print("[3] 输出形态：命中提示恰好一行、带行号、无特化提示", flush=True)
    lines = [line for line in captured.splitlines() if line.strip()]
    where = re.search(r"profile\.csv:(\d+)", captured)
    line_no = int(where.group(1)) if where else 0
    rec.check("3a 命中提示恰好一行", len(lines) == 1,
              f"lines={len(lines)} {lines[:2]}")
    rec.check("3b 提示含 profile.csv:<行号>",
              1 <= line_no <= max(len(rows), 1),
              f"line={line_no} rows={len(rows)}")
    rec.check("3c generic 下无 specialized 提示",
              "specialized path:" not in captured and "专用路径:" not in captured,
              "不含 specialized path:")


def _step_reverse(rec: _Recorder, operands: list[Any], empty_dir: str) -> None:
    """步骤 4：缓存为空时通用路径提示照打（证明步骤 3 的安静是命中造成的）。"""
    print("[4] 反向证明：空缓存 + 不强制内核时 generic path 仍在", flush=True)
    # "清空缓存"在用户那里是"换个新进程"。提示去重是**进程级**的，这里把它复原成
    # 初见状态，否则同一进程里第二次通用命中会被静音，判的就不再是同一件事。
    announced = getattr(_specialize, "_announced", None)
    if announced is not None:
        announced.discard("__generic__")
    _, captured = _only_stdout(
        lambda: pe.contract(EQUATION, *operands, profile=True, profile_dir=empty_dir))
    rec.check("4 无缓存时通用路径提示仍在",
              "generic path:" in captured or "通用路径:" in captured,
              captured.splitlines()[0][:120] if captured.strip() else "（无输出）")


def _step_tune(rec: _Recorder, work_root: str) -> None:
    """步骤 5：``python -m pyqcd.einsum.test --tune`` 必须往画像里写多行。"""
    print("[5] --tune：CLI 子进程写多行", flush=True)
    run_dir = os.path.join(work_root, "cli")
    os.makedirs(run_dir, exist_ok=True)
    tune_dir = os.path.join(run_dir, "table")
    command = [sys.executable, "-B", "-m", "pyqcd.einsum.test",
               "--equation", EQUATION, "--shape", SHAPE_SPEC,
               "--tune", "--profile-dir", tune_dir,
               "--rounds", "1", "--warmup", "1"]
    env = dict(os.environ)
    # 子进程的工作目录是临时目录：显式给 PYTHONPATH，避免依赖"恰好装在 site-packages"；
    # 同时任何默认目录也只会落在临时目录里，不会在仓库根留下 .cache。
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(REPO_ROOT), env.get("PYTHONPATH")) if part)
    completed = subprocess.run(command, cwd=run_dir, env=env, text=True,
                               capture_output=True, timeout=TUNE_TIMEOUT_S)
    tail = (completed.stderr or completed.stdout or "").strip().splitlines()
    rec.check("5a CLI --tune 退出码 0", completed.returncode == 0,
              f"returncode={completed.returncode}"
              + (f" tail={tail[-1][:160]}" if tail else ""))
    tune_csv = os.path.join(tune_dir, _profile_cache.PROFILE_NAME)
    tune_rows = _read_profile(tune_csv)
    rec.check("5b tune 写入多行", len(tune_rows) > 1,
              f"rows={len(tune_rows)} file={tune_csv}")


def main() -> int:
    # 提示语言钉在英文：断言的是"哪一条提示出现了"，不该随跑测机器的 locale 漂移。
    _specialize.set_language("en")
    rec = _Recorder()
    rng = np.random.default_rng(0)
    operands = [rng.standard_normal(shape) for shape in SHAPES]
    expected = np.einsum(EQUATION, *operands)

    with tempfile.TemporaryDirectory(prefix="pyqcd-accept-cache-") as table_dir, \
            tempfile.TemporaryDirectory(prefix="pyqcd-accept-tune-") as work_root:
        rows = _step_seed(rec, operands, table_dir)
        _step_hit_and_output(rec, operands, expected, table_dir, rows)
        _step_reverse(rec, operands, os.path.join(work_root, "empty-table"))
        _step_tune(rec, work_root)

    passed = rec.total - rec.failed
    print(f"{passed}/{rec.total} PASS, {rec.failed} FAIL", flush=True)
    return 0 if rec.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
