"""蒸馏生产的实战基准：按 ``refer/sush/sush-einsum.py`` 的真实尺寸跑 ``pyqcd.einsum``。

这个脚本照抄参考脚本的**协议**，不是照抄它的依赖：

* **尺寸**逐字取自 ``refer/sush/sush-einsum.py``（见 ``_bench_scene.PRODUCTION_DISTILL``）：
  ``Nev_src=100``/``Nev_curr=400``、``Mom_src=1``/``Mom_curr=9``、``Ns=4``、``Nc=3``、
  ``dtype=complex64``。**没有空间轴**（该脚本 ``Nx``/``Lt``/``lattice`` 匹配数为 0）。
* **计时协议**照抄参考脚本 ``main()`` 的循环（见 ``:256-320``）：
  ``plan_seconds`` 是**首次调用**（含路径搜索、原生步表编译）的耗时，
  ``result_seconds`` 是随后每次执行的耗时；本脚本先跑 ``--warmup`` 次丢弃，
  再取 ``--rounds`` 次的中位数/均值/最小值。
* **操作数**按参考脚本的字面张量构造：``peram[tq,ta]=(Ns,Ns,nev[tq],nev[ta])``
  —— 16 个组合给出 ``(4,4,100,100)``/``(4,4,100,400)``/``(4,4,400,100)``/
  ``(4,4,400,400)`` **四种异形轴**；``VVV=(Mom,Nev,Nev,Nev)``、``VDV=(Mom,Nev,Nev)``、
  ``gamma=(Ns,Ns)``。

参考脚本自己的方程来自 ``lqcddb.wick_contraction`` 的 ``result_indx``，那条路径需要
``/public/home/sush/distillation/lqcddb_{new,update}/src``（本机不存在）。因此方程用
本仓库**已核验**的 sush 生产方程（标签→形状映射见
:func:`pyqcd.testing.einsum.bench_distillation.build_operands`），只把尺寸换成参考值；
``--with-lqcddb`` 在源码树存在时改为直接调用参考脚本自己的 wick 输出。

为什么每个引擎都要单独起子进程
----------------------------

``opt_einsum:optimal`` 的路径搜索是**穷举**（``n`` 个算子的搜索空间 ~ ``(2n-3)!!``），
15 算子的 ``P1``/``P2`` 单次就要**几十分钟往上**——它曾经让整份基准在外部
``timeout`` 下静默死掉（280 s 无任何输出）。同类风险还有：BLAS 调用、原生步表
编译一旦卡住，**线程和信号都中断不了**（C 代码里没有安全中断点）。

因此每个 ``(scene, 方程, 引擎)`` 都跑在**独立子进程**里（``subprocess.run`` +
``timeout``，跨平台、无死锁、可读），并由子进程把每个阶段的结论**立刻 fsync 进
一份账本**（``.ledger.jsonl``）：

* ``plan`` 阶段：首次调用（含路径搜索/编译）的耗时与 ``rel``；
* ``hot`` 阶段：``--warmup`` 次 + ``--rounds`` 次计时执行的样本。

父进程在 ``2 * --engine-timeout + 启动余量`` 后硬杀子进程，再读账本判断卡在
**plan** 还是**hot**，打印 ``TIMEOUT`` 并继续下一个引擎——整次运行不会再被
外部 ``timeout`` 杀掉。

用法::

    python -m pyqcd.testing.einsum.bench_distill_production
    python -m pyqcd.testing.einsum.bench_distill_production --rounds 10 --json out.json
    python -m pyqcd.testing.einsum.bench_distill_production --scene distill --equations S1
    # 只跑几个引擎，单阶段预算 60 s
    python -m pyqcd.testing.einsum.bench_distill_production --scene distill --equations P2 \
        --rounds 2 --warmup 1 --optimize greedy --engine-timeout 60 \
        --engines pyqcd:auto,opt_einsum:dp,opt_einsum:greedy
    # 明知要等才跑穷举路径（15 算子请按小时计）
    python -m pyqcd.testing.einsum.bench_distill_production --equations S2 \
        --engines opt_einsum:optimal --engine-timeout 600

退出码：0 = 全部数值与形状断言通过；1 = 有对照失败；2 = 生产规模装不下（内存守卫）；
3 = 有引擎 ``TIMEOUT``（如实标 unverified，不当作通过）。
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from ...einsum import _bench_scene
from ...einsum import contract as pyqcd_contract

# sush 生产方程（标签→形状映射已由 bench_distillation.build_operands 核验）。
# 每条：(名字, 方程, 说明, 每算子形状键)。形状键在 _shapes_for 里解析。
EQUATIONS = {
    "S1": {
        "eq": "eofp,ambn,cgdh,ce,mo,Nbdf,Mnph,ga->MN",
        "desc": "sush NN 2pt（8 算子）",
        # `ga` 是 4x4 的 gamma（参考实现 _projector() 返回 (gamma0+gamma4)/2），
        # 不是 P1/P2 里那种切成 (4,2) 的投影。
        "ops": ("peram", "peram", "peram", "gamma", "gamma",
                "vvv", "vvv", "gamma"),
    },
    "S2": {
        "eq": "gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG",
        "desc": "sush 3pt（9 算子，含 mu-nu 插入）",
        "ops": ("peram", "peram", "peram", "nv2", "vdv", "vdv",
                "gamma", "gam4", "gam4"),
    },
    "S3": {
        "eq": "manb,cedf,bd,Mfn,ac,MFem->MF",
        "desc": "sush 2pt（6 算子，含 mu 插入）",
        "ops": ("peram", "peram", "nv2", "vdv", "gamma", "gam4"),
    },
    "P1": {
        "eq": ("agbh,eqfr,ospt,cmdn,ikjl,ce,Gmo,gi,kq,"
               "Nbdf,NLnp,Mhj,Mlrt,ax,sy->MNGLxy"),
        "desc": "sush 3pt 主项（15 算子，含 gamma_curr 与投影）",
        "ops": ("peram", "peram", "peram", "peram", "peram", "gamma", "gam3",
                "gamma", "gamma", "vvv", "vvv_link", "vdv", "vvv",
                "proj2", "proj2"),
    },
    "P2": {
        "eq": ("agbh,eqfr,ospt,ckdl,imjn,ce,Gmo,gi,kq,"
               "Nbdf,NLnp,Mhj,Mlrt,ax,sy->MNGLxy"),
        "desc": "sush 3pt 主项（15 算子，置换型）",
        "ops": ("peram", "peram", "peram", "peram", "peram", "gamma", "gam3",
                "gamma", "gamma", "vvv", "vvv_link", "vdv", "vvv",
                "proj2", "proj2"),
    },
}

# 代表参考脚本时间端点的 peram 选择不再需要：见 `_shapes_for` 的单一 Nev 说明。
TOLERANCE = {np.dtype(np.complex128): 1e-12, np.dtype(np.float64): 1e-12,
             np.dtype(np.complex64): 1e-4, np.dtype(np.float32): 1e-4}

#: 操作数种子：固定值保证父子进程、各引擎看到的是同一批张量。
SEED = 20261114

#: 默认引擎集合。``opt_einsum:optimal`` **故意不在里面**：它的路径搜索是穷举，
#: ``n`` 个算子的搜索空间 ~ ``(2n-3)!!``，15 算子单次要几十分钟，只有
#: ``--engines`` 里点名才跑。
DEFAULT_ENGINES = ("pyqcd:generic", "pyqcd:auto", "pyqcd:dedicated",
                   "pyqcd:topology", "opt_einsum:dp", "opt_einsum:greedy")
KNOWN_ENGINES = DEFAULT_ENGINES + ("opt_einsum:optimal",)
ENGINE_NOTES = {
    "pyqcd:generic": "specialize=False 的通用分步收缩",
    "pyqcd:auto": "默认分派（特化→原生→通用）",
    "pyqcd:dedicated": "原生固定步表（try_dedicated）",
    "pyqcd:topology": "原生拓扑下发（try_topology_apply）",
    "opt_einsum:dp": "opt_einsum 动态规划路径",
    "opt_einsum:greedy": "opt_einsum 贪心路径",
    "opt_einsum:optimal": ("opt_einsum 穷举最优路径——**指数级**搜索空间，"
                           "15 算子请按几十分钟到几小时计，默认不跑"),
}

_MODULE = "pyqcd.testing.einsum.bench_distill_production"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKER_FLAG = "--_worker"
#: 子进程启动 + 操作数构造 + 参考结果留出的余量（不算进阶段预算）。
_STARTUP_ALLOWANCE = 60.0


def _parse_dtype(name: str) -> np.dtype:
    try:
        dtype = np.dtype(name)
    except TypeError as exc:
        raise SystemExit(f"unknown dtype {name!r}") from exc
    if dtype not in TOLERANCE:
        raise SystemExit(
            f"{dtype} is outside the native contract (complex128/complex64/"
            f"float64/float32 only); refusing to benchmark it")
    return dtype


def _shapes_for(equation: str, kinds: tuple[str, ...],
                table: dict[str, int]) -> list[tuple[int, ...]]:
    """按算子类型给出形状。**同一方程只有一个 ``Nev``**，理由见下。

    本仓库这 5 条方程是 sush 单 ``Nev`` 口径下核验过的：``S1`` 的 ``f,d,b,n,p,h``
    六个标签同时出现在不同 peram 与 ``VVV`` 里（``Nbdf``/``Mnph``），必须等长；
    ``S2``/``P1`` 同理。也就是说**参考脚本那种「同一张图里 ``Nev`` 既是 100 又是
    400」无法用这批方程表达** —— 那需要参考脚本自己的 wick 方程（依赖
    ``lqcddb``，本机不可用）。

    因此这里分两档跑，各自都是参考脚本的真实生产尺寸：
    ``distill`` 用 ``Nev=100, Mom=1``（sink/source 端），
    ``distill_curr`` 用 ``Nev=400, Mom=9``（current 端）。
    张量内部仍是异形轴（``(4,4,100,100)`` vs ``(1,100,100,100)``），只是不跨端点混排。
    """
    ns = table["Ns"]
    nev = table["Nev"]
    mom = max(table["Mom"], 1)
    shapes: list[tuple[int, ...]] = []
    for kind in kinds:
        if kind == "peram":
            shapes.append((ns, ns, nev, nev))
        elif kind == "gamma":
            shapes.append((ns, ns))
        elif kind == "gam3":
            shapes.append((ns, ns, ns))
        elif kind == "gam4":
            shapes.append((mom, mom, ns, ns))
        elif kind == "proj2":
            shapes.append((ns, ns // 2))
        elif kind == "nv2":
            shapes.append((nev, nev))
        elif kind == "vvv":
            shapes.append((mom, nev, nev, nev))
        elif kind == "vvv_link":
            shapes.append((mom, 2, nev, nev))
        elif kind == "vdv":
            shapes.append((mom, nev, nev))
        else:                                                # pragma: no cover
            raise SystemExit(f"unknown operand kind {kind!r}")
    del equation                                            # 仅为可读性保留参数
    return shapes


def _build_operands(shapes, dtype, seed):
    rng = np.random.default_rng(seed)
    out = []
    for shape in shapes:
        real = rng.standard_normal(shape)
        if dtype.kind == "c":
            value = (real + 1j * rng.standard_normal(shape)) / math.sqrt(2.0)
        else:
            value = real
        out.append(np.ascontiguousarray(value.astype(dtype)))
    return out


def _flops(equation, shapes, optimize):
    """FLOPs 用 opt_einsum 的 ``opt_cost``（BLAS 口径 2*M*N*K），与引擎无关。"""
    import opt_einsum as oe
    try:
        _, info = oe.contract_path(equation, *shapes, shapes=True,
                                   optimize=optimize)
        return int(info.opt_cost)
    except Exception:                                        # noqa: BLE001
        return 0


def _summary(samples):
    return {"median_ms": statistics.median(samples) * 1e3,
            "mean_ms": statistics.fmean(samples) * 1e3,
            "min_ms": min(samples) * 1e3}


def _rel_error(value, reference):
    value = np.asarray(value)
    reference = np.asarray(reference)
    if value.shape != reference.shape:
        return float("inf")
    if value.dtype != reference.dtype:
        return float("inf")
    scale = float(np.max(np.abs(reference))) if reference.size else 0.0
    if scale == 0.0:
        scale = 1.0
    return float(np.max(np.abs(value - reference))) / scale


def _engine_call(engine, equation, operands):
    """返回该引擎的一次调用（可调用对象）。不可用引擎由调用点拿异常。"""
    from ...einsum import _kernel_kupl as kupl
    import opt_einsum as oe

    if engine == "pyqcd:generic":
        return lambda: pyqcd_contract(equation, *operands, specialize=False)
    if engine == "pyqcd:auto":
        return lambda: pyqcd_contract(equation, *operands)

    if engine == "pyqcd:dedicated":
        def dedicated():
            result = kupl.try_dedicated(equation, operands)
            if result is None:
                raise RuntimeError(f"dedicated miss: {kupl.LAST_FAILURE['reason']}")
            return result
        return dedicated

    if engine == "pyqcd:topology":
        def topology():
            result = kupl.try_topology_apply(equation, operands)
            if result is None:
                raise RuntimeError(f"topology miss: {kupl.LAST_FAILURE['reason']}")
            return result
        return topology

    if engine.startswith("opt_einsum:"):
        name = engine.split(":", 1)[1]
        return lambda: oe.contract(equation, *operands, optimize=name)
    raise SystemExit(f"unknown engine {engine!r}; choose from {list(KNOWN_ENGINES)}")


def _path_from_output(text: str) -> str:
    """从 ``pyqcd.einsum`` 的提示里取路径证据（HIT=特化命中，MISS=通用路径）。

    提示语言跟随 locale（``[pyqcd.einsum] specialized path: ...`` /
    ``[pyqcd.einsum] 特化路径: ...``），两种都认。
    """
    for line in text.splitlines():
        for marker in ("specialized path:", "特化路径:"):
            if marker in line:
                return "HIT " + line.split(marker, 1)[1].strip()[:70]
        for marker in ("generic path:", "通用路径:", "通用内核"):
            if marker in line:
                return "MISS(generic)"
    return ""


def _miss_kind(reason: str) -> str:
    """把 ``LAST_FAILURE['reason']`` 分成「策略拒绝」与「真 miss」。"""
    if not reason:
        return "MISS"
    lowered = reason.lower()
    if "declined by policy" in lowered:
        return "MISS(策略拒绝：实数走 dedicated，属正常)"
    if "kernel disabled" in lowered:
        return "MISS(kernel 被禁用/库缺失)"
    return f"MISS({reason[:80]})"


# --------------------------------------------------------------------------- #
# 子进程侧：一个 (scene, 方程, 引擎) 的阶段账本
# --------------------------------------------------------------------------- #
def _append_ledger(path: Path, payload: dict) -> None:
    """追加并 fsync：父进程硬杀之后必须还能读到已完成的阶段。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:                          # pragma: no cover
            continue
    return rows


def _worker_main(spec_path: Path) -> int:
    """子进程入口：先 plan 阶段，再 hot 阶段，每阶段结论立刻落账本。"""
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    ledger = Path(spec["ledger"])
    engine = spec["engine"]
    # 提示必须打开才能拿到 HIT/MISS/path 证据；它是被 redirect 到账本里的，
    # 不会污染父进程的输出。
    os.environ["PYQCD_EINSUM_NOTICE"] = "1"

    dtype = np.dtype(spec["dtype"])
    table = _bench_scene.dims(spec["scene"])
    equation_spec = EQUATIONS[spec["equation"]]
    shapes = _shapes_for(equation_spec["eq"], equation_spec["ops"], table)
    equation = equation_spec["eq"]

    start = time.perf_counter()
    operands = _build_operands(shapes, dtype, spec["seed"])
    _append_ledger(ledger, {
        "phase": "build", "seconds": time.perf_counter() - start,
        "elements": sum(int(np.prod(shape)) for shape in shapes),
        "footprint_mib": (sum(int(np.prod(shape)) for shape in shapes)
                          * dtype.itemsize / 2**20)})

    call = _engine_call(engine, equation, operands)

    reference = None
    reference_seconds = None
    if spec["reference"]:
        import opt_einsum as oe
        start = time.perf_counter()
        reference = oe.contract(equation, *operands, optimize=spec["reference"])
        reference_seconds = time.perf_counter() - start

    from ...einsum import _kernel_kupl as kupl

    # ---- plan 阶段：首次调用（含路径搜索 / 原生步表编译） ----
    notice = io.StringIO()
    plan_start = time.perf_counter()
    try:
        with contextlib.redirect_stdout(notice):
            value = call()
    except BaseException as exc:                              # noqa: BLE001
        _append_ledger(ledger, {
            "phase": "plan", "ok": False, "seconds": time.perf_counter() - plan_start,
            "error": f"{type(exc).__name__}: {exc}",
            "notice": notice.getvalue()[-1500:],
            "last_failure": str(kupl.LAST_FAILURE.get("reason", "")),
            "reference_seconds": reference_seconds})
        return 1
    plan_seconds = time.perf_counter() - plan_start
    rel = _rel_error(value, reference) if reference is not None else None
    output = np.asarray(value)
    payload = {"phase": "plan", "ok": True, "plan_seconds": plan_seconds,
               "seconds": plan_seconds, "rel": rel,
               "dtype_out": str(output.dtype), "shape_out": list(output.shape),
               "notice": notice.getvalue()[-1500:],
               "path": _path_from_output(notice.getvalue()),
               "last_failure": str(kupl.LAST_FAILURE.get("reason", "")),
               "reference_seconds": reference_seconds}
    _append_ledger(ledger, payload)
    del value, output, reference

    # ---- hot 阶段：warmup 丢弃 + rounds 计时（口径 = result_seconds） ----
    hot = io.StringIO()
    samples = []
    try:
        with contextlib.redirect_stdout(hot):
            for _ in range(spec["warmup"]):
                call()
            for _ in range(spec["rounds"]):
                start = time.perf_counter()
                call()
                samples.append(time.perf_counter() - start)
    except BaseException as exc:                              # noqa: BLE001
        _append_ledger(ledger, {"phase": "hot", "ok": False, "samples": samples,
                                "error": f"{type(exc).__name__}: {exc}"})
        return 1
    _append_ledger(ledger, {"phase": "hot", "ok": True, "samples": samples})
    return 0


# --------------------------------------------------------------------------- #
# 父进程侧：起子进程 + 硬超时 + 读账本
# --------------------------------------------------------------------------- #
def _worker_env() -> dict[str, str]:
    env = dict(os.environ)
    root = str(_REPO_ROOT)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root if not existing else root + os.pathsep + existing
    return env


def _measure(engine, equation, scene, dtype, optimize, rounds, warmup,
             engine_timeout, startup_allowance, workdir: Path,
             reference) -> dict:
    """跑一个 (scene, 方程, 引擎)：plan 阶段 + hot 阶段，各带超时保护。"""
    tag = f"{scene}__{equation}__{engine.replace(':', '-')}"
    spec_path = workdir / f"{tag}.spec.json"
    ledger_path = workdir / f"{tag}.ledger.jsonl"
    spec = {"engine": engine, "equation": equation, "scene": scene,
            "dtype": str(dtype), "seed": SEED, "rounds": rounds,
            "warmup": warmup, "reference": reference,
            "ledger": str(ledger_path)}
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    # 每个阶段各有一份 --engine-timeout 的预算；父进程的硬杀窗口是两份预算
    # 再加上进程启动/操作数构造/参考结果的固定余量（不能把启动时间算成引擎耗时）。
    budget = 2.0 * engine_timeout + startup_allowance
    command = [sys.executable, "-m", _MODULE, _WORKER_FLAG, str(spec_path)]
    started = time.perf_counter()
    timed_out = False
    stderr = ""
    try:
        completed = subprocess.run(command, cwd=str(_REPO_ROOT),
                                   env=_worker_env(), capture_output=True,
                                   text=True, timeout=budget)
        stderr = (completed.stderr or "")[-1200:]
    except subprocess.TimeoutExpired:
        timed_out = True
    wall = time.perf_counter() - started

    rows = _read_ledger(ledger_path)
    by_phase = {}
    for row in rows:
        by_phase.setdefault(row.get("phase"), row)
    plan = by_phase.get("plan")
    hot = by_phase.get("hot")
    build = by_phase.get("build")

    if plan is None:
        status = "TIMEOUT" if timed_out else "CRASH"
    elif not plan.get("ok"):
        status = "ERROR"
    elif hot is None:
        status = "TIMEOUT-HOT" if timed_out else "CRASH"
    elif not hot.get("ok"):
        status = "ERROR"
    else:
        status = "OK"

    return {"engine": engine, "equation": equation, "scene": scene,
            "status": status, "wall_seconds": wall, "budget_seconds": budget,
            "timed_out": timed_out,
            "build_seconds": (build or {}).get("seconds"),
            "footprint_mib": (build or {}).get("footprint_mib"),
            "plan": plan, "hot": hot, "stderr": stderr}


def _format_row(engine: str, result: dict, flops: int, tolerance: float,
                timeout: float) -> tuple[str, dict]:
    """把一次测量渲染成一行表格，并返回写进 JSON 的记录。"""
    status = result["status"]
    plan = result["plan"] or {}
    hot = result["hot"] or {}
    base = {"scene": result["scene"], "equation": result["equation"],
            "engine": engine, "status": status,
            "wall_seconds": result["wall_seconds"],
            "build_seconds": result["build_seconds"],
            "footprint_mib": result["footprint_mib"]}
    if plan:
        base["plan_seconds"] = plan.get("plan_seconds")
        base["rel"] = plan.get("rel")
        base["path"] = plan.get("path")
        base["last_failure"] = plan.get("last_failure")
    if hot:
        base["samples"] = hot.get("samples")

    prefix = f"      {engine:<18s}"
    if status == "TIMEOUT":
        base["timeout"] = True
        note = (f"TIMEOUT: 首次调用（含 plan+执行）超过预算 {timeout:g}s"
                f"（硬杀窗口 {result['budget_seconds']:g}s）")
        return (f"{prefix} {'TIMEOUT':>10s} {'-':>10s} {'-':>10s} {'-':>10s} "
                f"{'-':>9s} {'-':>9s}  {note}"), base
    if status == "TIMEOUT-HOT":
        plan_ms = (plan.get("plan_seconds") or 0.0) * 1e3
        base["timeout"] = True
        return (f"{prefix} {plan_ms:10.2f} {'TIMEOUT':>10s} {'-':>10s} {'-':>10s} "
                f"{'-':>9s} {(plan.get('rel') or 0.0):9.1e}  "
                f"TIMEOUT: 计时循环（warmup+rounds）超过预算 {timeout:g}s"), base
    if status in ("ERROR", "CRASH"):
        error = (plan.get("error") or hot.get("error")
                 or result.get("stderr", "").strip().splitlines()[-1:]
                 or [status])
        if isinstance(error, list):
            error = error[0] if error else status
        base["error"] = str(error)
        return (f"{prefix} {'-':>10s} {'-':>10s} {'-':>10s} {'-':>10s} "
                f"{'-':>9s} {'-':>9s}  不可用/出错"
                f"（{str(error)[:70]}）"), base

    samples = hot.get("samples") or []
    if not samples:                                           # pragma: no cover
        base["status"] = "ERROR"
        base["error"] = "no samples"
        return f"{prefix} {'-':>10s} ...  不可用（没有任何计时样本）", base
    stats = _summary(samples)
    rate = flops / (stats["median_ms"] * 1e-3) / 1e9 if stats["median_ms"] else 0.0
    rel = plan.get("rel")
    ok = rel is not None and rel <= tolerance
    base.update(stats)
    base["gflops"] = rate
    base["tolerance"] = tolerance
    base["ok"] = ok
    plan_ms = (plan.get("plan_seconds") or 0.0) * 1e3
    evidence = plan.get("path") or ""
    if not evidence:
        # dedicated/topology 返回了结果就是命中（它们的 LAST_FAILURE 为空）；
        # specialize=False 的通用路径本来就没有提示，如实标注。
        if engine in ("pyqcd:dedicated", "pyqcd:topology"):
            evidence = "HIT"
        elif engine == "pyqcd:generic":
            evidence = "generic(specialize=False)"
    row = (f"{prefix} {plan_ms:10.2f} {stats['median_ms']:10.2f} "
           f"{stats['mean_ms']:10.2f} {stats['min_ms']:10.2f} "
           f"{rate:9.1f} {rel:9.1e}  {'OK' if ok else 'FAIL'}")
    if evidence:
        row += f"  {evidence}"
    return row, base


def run(equations, scene_names, dtype, optimize, engines, rounds, warmup,
        max_bytes, engine_timeout, startup_allowance, json_path):
    from .._production_perf_gate_contract import classify_measurement, claim_basis

    print("=" * 96)
    print("蒸馏生产实战基准（协议照抄 refer/sush/sush-einsum.py）")
    print(f"  dtype={dtype}  optimize={optimize}  rounds={rounds}  warmup={warmup}")
    print(f"  引擎 {len(engines)} 个: {', '.join(engines)}")
    print(f"  单阶段预算 --engine-timeout={engine_timeout:g}s"
          f"（首次调用含 plan / 计时循环各一份；超时标 TIMEOUT 并继续）")
    print(f"  子进程硬杀窗口 = 2*{engine_timeout:g}s"
          f" + 启动余量 {startup_allowance:g}s = "
          f"{2 * engine_timeout + startup_allowance:g}s")
    if "opt_einsum:optimal" in engines:
        print(f"  ! opt_einsum:optimal: {ENGINE_NOTES['opt_einsum:optimal']}")
    print(f"  尺寸来源: {Path(_bench_scene.__file__).name} · PRODUCTION_DISTILL")
    for key, value in _bench_scene.PRODUCTION_DISTILL.items():
        print(f"    {key:<10s} = {value}")
    print(f"  参考脚本 HEAD 变体（仅核对，不作结论）: "
          f"{_bench_scene.DISTILL_HEAD_VARIANT}")
    print("=" * 96)

    records = []
    failures = 0
    budget_hit = 0
    timed_out = []
    total_cells = len(equations) * len(scene_names)
    workdir = Path(tempfile.mkdtemp(prefix="pyqcd-distill-bench-"))
    try:
        for scene in scene_names:
            table = _bench_scene.dims(scene)
            verdict = classify_measurement(scene, None, rounds)
            basis_ok = claim_basis(scene, "distill")
            print(f"\n### scene={scene}  dims={ {k: v for k, v in table.items()} }")
            print(f"    证据等级={verdict}  可支撑 distill 口径={basis_ok}")

            for name in equations:
                spec = EQUATIONS[name]
                shapes = _shapes_for(spec["eq"], spec["ops"], table)
                footprint = (sum(int(np.prod(shape)) for shape in shapes)
                             * dtype.itemsize)
                if footprint > max_bytes:
                    budget_hit += 1
                    print(f"  [skip] {name}: 操作数 {footprint / 2**30:.2f} GiB "
                          f"> 预算 {max_bytes / 2**30:.2f} GiB"
                          f"（硬性规则：不得用缩小尺寸替代，如实跳过）")
                    continue
                elements = sum(int(np.prod(shape)) for shape in shapes)
                flops = _flops(spec["eq"], shapes, optimize)

                print(f"\n  --- {name}: {spec['desc']}")
                print(f"      方程 {spec['eq']}")
                print(f"      操作数 {elements:,} 元素"
                      f" = {footprint / 2**20:.2f} MiB；FLOPs={flops:.3e}；"
                      f"轴长={sorted({e for s in shapes for e in s})}")
                print(f"      {'引擎':<18s} {'plan ms':>10s} {'结果ms':>10s} "
                      f"{'均值ms':>10s} {'最小ms':>10s} {'GFLOP/s':>9s} "
                      f"{'rel':>9s}  判定")
                print(f"      （plan ms = 首次调用含路径搜索/编译，对应参考脚本 "
                      f"plan_seconds；结果ms = 热态中位数，对应 result_seconds）")

                for engine in engines:
                    result = _measure(engine, name, scene, dtype, optimize,
                                      rounds, warmup, engine_timeout,
                                      startup_allowance, workdir, optimize)
                    row, record = _format_row(engine, result, flops,
                                              TOLERANCE[dtype], engine_timeout)
                    record.update({"elements": elements,
                                   "footprint_mib": footprint / 2**20,
                                   "flops": flops, "evidence": verdict,
                                   "distill_basis": basis_ok})
                    print(row, flush=True)
                    records.append(record)
                    if record["status"] == "OK":
                        failures += 0 if record.get("ok") else 1
                    elif record["status"].startswith("TIMEOUT"):
                        timed_out.append(f"{scene}/{name}/{engine}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print("\n" + "=" * 96)
    pyqcd_rows = [r for r in records if r.get("engine", "").startswith("pyqcd:")]
    oe_rows = [r for r in records if r.get("engine", "").startswith("opt_einsum:")]
    if pyqcd_rows and oe_rows:
        best_py = {}
        best_oe = {}
        for row in pyqcd_rows:
            if "median_ms" not in row:
                continue
            key = (row["scene"], row["equation"])
            best_py[key] = min(best_py.get(key, float("inf")), row["median_ms"])
        for row in oe_rows:
            if "median_ms" not in row:
                continue
            key = (row["scene"], row["equation"])
            best_oe[key] = min(best_oe.get(key, float("inf")), row["median_ms"])
        print("最佳引擎对照（pyqcd 最快路径 vs opt_einsum 最快路径）")
        print(f"  {'scene':<14s} {'方程':<6s} {'pyqcd ms':>10s} "
              f"{'oe ms':>10s} {'pyqcd/oe':>10s}")
        for key in sorted(best_py):
            if key not in best_oe:
                continue
            ratio = best_py[key] / best_oe[key] if best_oe[key] else float("nan")
            print(f"  {key[0]:<14s} {key[1]:<6s} {best_py[key]:10.2f} "
                  f"{best_oe[key]:10.2f} {ratio:10.2f}")
    print("=" * 96)
    print(f"数值失败 {failures} 项；因内存预算跳过 {budget_hit} 项")
    if timed_out:
        print(f"TIMEOUT 未验证 {len(timed_out)} 项（不计入通过，也不当作失败）:")
        for item in timed_out:
            print(f"  - {item}")

    if json_path:
        Path(json_path).write_text(
            json.dumps({"dtype": str(dtype), "optimize": optimize,
                        "engines": list(engines),
                        "engine_timeout": engine_timeout,
                        "rounds": rounds, "warmup": warmup,
                        "production": _bench_scene.PRODUCTION_DISTILL,
                        "head_variant": _bench_scene.DISTILL_HEAD_VARIANT,
                        "records": records}, indent=2, ensure_ascii=False),
            encoding="utf-8")
        print(f"明细已写入 {json_path}")

    if failures:
        return 1
    if budget_hit == total_cells:
        return 2
    if timed_out:
        return 3
    return 0


def _parse_engines(text: str) -> list[str]:
    names = [item.strip() for item in text.split(",") if item.strip()]
    for name in names:
        if name not in KNOWN_ENGINES:
            raise SystemExit(f"unknown engine {name!r}; choose from "
                             f"{list(KNOWN_ENGINES)}")
    if not names:
        raise SystemExit("--engines 为空")
    if len(set(names)) != len(names):
        raise SystemExit("--engines 里有重复项")
    return names


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default="distill,distill_curr",
                        help="逗号分隔（distill/distill_curr）")
    parser.add_argument("--equations", default="S1,S2,S3,P1,P2",
                        help=f"逗号分隔，可选 {','.join(EQUATIONS)}")
    parser.add_argument("--dtype", default="complex64",
                        help="complex64/complex128/float64/float32")
    parser.add_argument("--optimize", default="dp",
                        choices=("dp", "greedy", "auto"),
                        help="参考结果与 FLOPs 的 opt_einsum 口径"
                             "（只接受 dp/greedy/auto：optimal 的穷举路径搜索"
                             "是指数级的，想要它请用 --engines opt_einsum:optimal）")
    parser.add_argument("--engines", default=",".join(DEFAULT_ENGINES),
                        help="逗号分隔的引擎名，默认 "
                             + ",".join(DEFAULT_ENGINES)
                             + f"。可选: {','.join(KNOWN_ENGINES)}。"
                               "注意 opt_einsum:optimal 是**指数级**穷举路径搜索"
                               "（n 个算子约 (2n-3)!! 条候选），15 算子单次要"
                               "几十分钟到几小时，因此**默认排除**，只有在这里"
                               "点名才跑")
    parser.add_argument("--engine-timeout", type=float, default=120.0,
                        help="每个引擎每个阶段的预算（秒，默认 120）：首次调用"
                             "（含 plan/路径搜索/编译）一份、计时循环一份；"
                             "超时打印 TIMEOUT 并继续下一个引擎，不会让整次运行"
                             "被外部 timeout 杀掉。子进程硬杀窗口 = "
                             "2*该值 + --startup-allowance")
    parser.add_argument("--startup-allowance", type=float,
                        default=_STARTUP_ALLOWANCE,
                        help="子进程启动 + 操作数构造 + 参考结果留出的余量（秒，"
                             f"默认 {_STARTUP_ALLOWANCE:g}）；不算进引擎耗时。"
                             "distill_curr 的 P1/P2 操作数有 9.35 GiB，构造本身就要"
                             "近一分钟，不要把它调到这以下")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-bytes", type=float, default=15.0,
                        help="操作数足迹上限（GiB）；超了如实跳过，不缩小替代")
    parser.add_argument("--json", default=None)
    parser.add_argument("--with-lqcddb", action="store_true",
                        help="源码树可用时改为调用参考脚本自己的 wick 方程")
    parser.add_argument(_WORKER_FLAG, dest="worker", default=None,
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker is not None:
        return _worker_main(Path(args.worker))

    scenes = [item.strip() for item in args.scene.split(",") if item.strip()]
    for scene in scenes:
        if scene not in ("distill", "distill_curr"):
            raise SystemExit(
                f"{scene!r} 不是蒸馏生产口径；只接受 distill/distill_curr"
                f"（其它档位的倍率不得当蒸馏结论）")
    equations = [item.strip() for item in args.equations.split(",") if item.strip()]
    for name in equations:
        if name not in EQUATIONS:
            raise SystemExit(f"unknown equation {name!r}; choose from {list(EQUATIONS)}")
    engines = _parse_engines(args.engines)

    if args.with_lqcddb:
        raise SystemExit(
            "参考脚本的 wick 方程需要 /public/home/sush/distillation/"
            "lqcddb_{new,update}/src（本机不存在）或已安装的 lqcddb；"
            "当前用本仓库已核验的 sush 生产方程代替，去掉 --with-lqcddb 即可运行")

    if args.rounds < 1:
        raise SystemExit("--rounds must be >= 1")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")
    if args.engine_timeout <= 0:
        raise SystemExit("--engine-timeout must be > 0")
    if args.startup_allowance < 0:
        raise SystemExit("--startup-allowance must be >= 0")

    return run(equations, scenes, _parse_dtype(args.dtype), args.optimize,
               engines, args.rounds, args.warmup, args.max_bytes * 2**30,
               args.engine_timeout, args.startup_allowance, args.json)


if __name__ == "__main__":
    sys.exit(main())
