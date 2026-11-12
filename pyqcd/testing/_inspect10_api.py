#!/usr/bin/env python3
"""用户接口十轮巡查：从「使用者视角」验证公开面，而不是内部实现。

十轮设计（每轮一个接口维度）：
  U1  包级导入与子包可发现性（__all__ / dir / 延迟加载）
  U2  pyqcd.einsum 公开 API 逐项可调用（含 opt_einsum 兼容面）
  U3  dtype 矩阵：4 种 dtype × 基本方程，dtype/形状/数值
  U4  kwargs 契约：optimize / backend / out / specialize / dtype 保真
  U5  错误路径：非法输入必须给明确异常（不是静默 None 或崩溃）
  U6  可发现性：help()/docstring/签名可读，`python -m pyqcd` CLI 可用
  U7  跨后端一致性：numpy vs torch vs cupy（同一方程同结果）
  U8  上下文管理：shared_intermediates / count_cached_ops 语义
  U9  兼容性回退：opt_einsum 不装时是否仍可用（模拟 ImportError）
  U10 用户体验：错误消息含可操作信息；无内部术语泄漏；性能提示正确

用法：python -m pyqcd.testing._inspect10_api
"""
from __future__ import annotations

import inspect
import os
import pathlib
import re
import subprocess
import sys
import traceback

ROOT = pathlib.Path("/home/zhangxin/PyQCD")
RESULTS: list[dict] = []


def record(round_no: str, name: str, ok: bool, detail: str) -> None:
    RESULTS.append({"round": round_no, "name": name, "ok": ok, "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def run(code: str, timeout: int = 120, env: dict | None = None) -> tuple[int, str]:
    """在子进程里跑一段 Python（隔离副作用，模拟真实用户）。"""
    merged = dict(os.environ)
    merged["PYTHONPATH"] = str(ROOT)
    if env:
        merged.update(env)
    try:
        done = subprocess.run([sys.executable, "-c", code], cwd="/tmp",
                              capture_output=True, text=True, timeout=timeout,
                              env=merged)
        return done.returncode, (done.stdout + done.stderr)
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def expect_ok(round_no: str, name: str, code: str, needle: str = "OK") -> None:
    rc, out = run(code)
    ok = rc == 0 and needle in out
    detail = out.strip().splitlines()[-1][:120] if out.strip() else "no output"
    record(round_no, name, ok, detail)


# ── U1 包级导入与可发现性 ───────────────────────────────────────────
def u1() -> None:
    expect_ok("U1", "import pyqcd 与子包", """
import importlib.util
import pyqcd
missing = [n for n in pyqcd.__all__
           if importlib.util.find_spec(f"pyqcd.{n}") is None]
assert not missing, f"子包缺失: {missing}"
assert dir(pyqcd), "dir(pyqcd) 为空（PEP 562 未生效）"
print("OK")
""")
    # 删掉 parallel 的 einsum 后，访问 pyqcd.parallel 不应崩
    expect_ok("U1", "pyqcd.parallel 仍可导入（仅资源探测）", """
import pyqcd.parallel as P
assert hasattr(P, "detect_resources"), "资源探测入口丢失"
assert hasattr(P, "cpu_threads")
print("OK")
""")
    # 已删除的接口必须明确 AttributeError，而不是静默
    expect_ok("U1", "已删接口给 AttributeError", """
import pyqcd.parallel as P
for gone in ("init", "run", "plan_tasks", "einsums", "plan_parallel", "run_parallel_pipeline"):
    try:
        getattr(P, gone)
    except AttributeError:
        continue
    raise AssertionError(f"{gone} 仍可访问（应已删除）")
print("OK")
""")


# ── U2 einsum 公开 API ──────────────────────────────────────────────
def u2() -> None:
    expect_ok("U2", "核心 API 全部可调用", """
import pyqcd.einsum as pe
import numpy as np
a = np.ones((3, 3)); b = np.ones((3, 3))
assert np.allclose(pe.contract("ij,jk->ik", a, b), a @ b)
path, info = pe.contract_path("ij,jk->ik", a, b)
assert float(info.opt_cost) == 54.0, info.opt_cost
expr = pe.contract_expression("ij,jk->ik", a.shape, b.shape)
assert np.allclose(expr(a, b), a @ b)
assert pe.shape_only(((3, 3), (3, 3))) is not None
assert pe.specializations()
assert set(("backend", "native_available", "array_backend")) <= set(pe.build_info())
print("OK")
""")
    expect_ok("U2", "opt_einsum 兼容面", """
import pyqcd.einsum as pe
for name in ("get_func", "get_path_fn", "register_path_fn", "has_backend",
             "has_einsum", "parse_backend", "infer_backend", "PATH_OPTIONS",
             "PathInfo", "PathOptimizer", "format_const_einsum_str"):
    assert hasattr(pe, name), name
assert isinstance(pe.PATH_OPTIONS, (list, tuple, set))
print("OK")
""")


# ── U3 dtype 矩阵 ───────────────────────────────────────────────────
def u3() -> None:
    expect_ok("U3", "4 dtype × 形状/数值保真", """
import numpy as np
import pyqcd.einsum as pe
rng = np.random.default_rng(0)
for dt in (np.float32, np.float64, np.complex64, np.complex128):
    a = rng.standard_normal((4, 5)).astype(dt)
    b = rng.standard_normal((5, 6)).astype(dt)
    got = pe.contract("ij,jk->ik", a, b)
    assert got.dtype == np.dtype(dt), (dt, got.dtype)
    assert got.shape == (4, 6), got.shape
    ref = (a.astype(np.complex128) @ b.astype(np.complex128))
    tol = 1e-5 if np.dtype(dt).itemsize <= 8 else 1e-12
    assert np.allclose(got, ref, rtol=tol, atol=tol), dt
print("OK")
""")


# ── U4 kwargs 契约 ──────────────────────────────────────────────────
def u4() -> None:
    expect_ok("U4", "optimize 各档均可用", """
import numpy as np, pyqcd.einsum as pe
a = np.arange(9.).reshape(3,3); b = np.eye(3)
for opt in (False, True, "greedy", "optimal", "dp", "auto"):
    got = pe.contract("ij,jk->ik", a, b, optimize=opt)
    assert np.allclose(got, a @ b), opt
print("OK")
""")
    expect_ok("U4", "out= 写入调用方缓冲区", """
import numpy as np, pyqcd.einsum as pe
a = np.arange(9.).reshape(3,3); b = np.eye(3)
out = np.zeros((3,3))
got = pe.contract("ij,jk->ik", a, b, out=out)
assert got is out or np.shares_memory(got, out), "out= 未被复用"
assert np.allclose(out, a @ b)
print("OK")
""")
    expect_ok("U4", "specialize=False 可关闭特化", """
import numpy as np, pyqcd.einsum as pe
a = np.ones((6,6)); b = np.ones((6,6))
on = pe.contract("ij,jk->ik", a, b)
off = pe.contract("ij,jk->ik", a, b, specialize=False)
assert np.allclose(on, off)
print("OK")
""")


# ── U5 错误路径 ─────────────────────────────────────────────────────
def u5() -> None:
    expect_ok("U5", "非法输入给明确异常", """
import numpy as np, pyqcd.einsum as pe
a = np.ones((3,3)); b = np.ones((4,4))
cases = [
    (lambda: pe.contract("ij,jk->ik", a), "少操作数"),
    (lambda: pe.contract("ij,jk->ik", a, b), "维度冲突"),
    (lambda: pe.contract("ij,jk->iq", a, a), "输出标签未知"),
    (lambda: pe.contract("ij->ii", a), "输出标签重复"),
]
for fn, label in cases:
    try:
        fn()
    except (ValueError, TypeError):
        continue
    except Exception as exc:
        raise AssertionError(f"{label}: 抛了非预期异常 {type(exc).__name__}: {exc}")
    raise AssertionError(f"{label}: 未报错（静默返回）")
print("OK")
""")
    expect_ok("U5", "错误消息具可操作性", """
import numpy as np, pyqcd.einsum as pe
a = np.ones((3,3)); b = np.ones((4,4))
try:
    pe.contract("ij,jk->ik", a, b)
except Exception as exc:
    msg = str(exc)
    assert len(msg) > 10, f"消息过短: {msg!r}"
    assert any(word in msg.lower() for word in
               ("size", "label", "dimension", "shape", "operand", "indices", "下标", "维度")), msg
else:
    raise AssertionError("未报错")
print("OK")
""")


# ── U6 可发现性与 CLI ───────────────────────────────────────────────
def u6() -> None:
    expect_ok("U6", "docstring 与签名可读", """
import pyqcd.einsum as pe
doc = (pe.contract.__doc__ or "")
assert len(doc) > 100, f"contract docstring 过短 ({len(doc)})"
# 关键：docstring 必须说明核心 kwargs
for key in ("optimize", "out"):
    assert key in doc, f"docstring 未提及 {key}"
sig = __import__("inspect").signature(pe.contract)
params = list(sig.parameters)
assert any("optimize" in p or p == "kwargs" for p in params), params
print("OK")
""")
    rc, out = run("import subprocess,sys;print(subprocess.run([sys.executable,'-m','pyqcd'],capture_output=True,text=True).stdout)")
    ok = rc == 0 and "pyqcd" in out and "backend" in out
    record("U6", "python -m pyqcd CLI", ok, out.strip()[:100] or "无输出")


# ── U7 跨后端一致性 ─────────────────────────────────────────────────
def u7() -> None:
    expect_ok("U7", "numpy vs torch 一致", """
import numpy as np, pyqcd.einsum as pe
a = np.random.default_rng(0).standard_normal((8,8))
b = np.random.default_rng(1).standard_normal((8,8))
base = pe.contract("ij,jk->ik", a, b)
try:
    import torch
except ImportError:
    print("OK (torch 未安装，跳过)")
else:
    ta, tb = torch.from_numpy(a.astype(np.float64)), torch.from_numpy(b.astype(np.float64))
    got = pe.contract("ij,jk->ik", ta, tb)
    assert np.allclose(np.asarray(got), base, rtol=1e-10), "torch 与 numpy 不一致"
    print("OK")
""")
    expect_ok("U7", "numpy vs cupy 一致", """
import numpy as np, pyqcd.einsum as pe
a = np.random.default_rng(0).standard_normal((8,8)).astype(np.float32)
b = np.random.default_rng(1).standard_normal((8,8)).astype(np.float32)
base = pe.contract("ij,jk->ik", a, b)
try:
    import cupy as cp
except ImportError:
    print("OK (cupy 未安装，跳过)")
else:
    if cp.cuda.runtime.getDeviceCount() == 0:
        print("OK (无 GPU，跳过)")
    else:
        got = pe.contract("ij,jk->ik", cp.asarray(a), cp.asarray(b))
        scaled = max(float(np.max(np.abs(base))), 1e-30)
        dev = float(np.max(np.abs(cp.asnumpy(got) - base))) / scaled
        assert dev <= 1e-5, f"cupy 与 numpy 偏差 {dev:.2e}（float32 判据 1e-5）"
        print("OK")
""")


# ── U8 上下文管理 ───────────────────────────────────────────────────
def u8() -> None:
    expect_ok("U8", "shared_intermediates 缓存生效", """
import numpy as np, pyqcd.einsum as pe
a = np.ones((4,4)); b = np.ones((4,4)); c = np.ones((4,4))
with pe.shared_intermediates() as cache:
    pe.contract("ij,jk->ik", a, b)
    pe.contract("ij,jk->ik", a, b)
    ops = pe.count_cached_ops(cache)
    assert sum(ops.values()) >= 1, f"缓存为空: {ops}"
print("OK")
""")
    expect_ok("U8", "线程配置接口可用", """
import pyqcd.einsum as pe
info = pe.thread_info()
assert isinstance(info, dict) and info, info
assert isinstance(pe.thread_profiles(), dict)
pe.configure_threads(omp=1, blas=1)
print("OK")
""")


# ── U9 兼容性回退 ───────────────────────────────────────────────────
def u9() -> None:
    # 模拟 opt_einsum 不可用
    rc, out = run("""
import sys
class _Block:
    def find_module(self, name, path=None):
        if name == "opt_einsum" or name.startswith("opt_einsum."):
            return self
    def load_module(self, name):
        raise ImportError("blocked for test")
sys.meta_path.insert(0, _Block())
sys.modules.pop("opt_einsum", None)
import numpy as np, pyqcd.einsum as pe
a = np.ones((4,4)); b = np.ones((4,4))
got = pe.contract("ij,jk->ik", a, b)
assert np.allclose(got, a @ b), "opt_einsum 缺失时结果错误"
print("OK")
""")
    record("U9", "opt_einsum 缺失时仍可用", rc == 0 and "OK" in out,
           out.strip().splitlines()[-1][:120] if out.strip() else "no output")
    expect_ok("U9", "纯安装环境关键路径不依赖重依赖", """
import sys
import pyqcd.einsum as pe
heavy = [m for m in ("torch", "cupy", "matplotlib") if m in sys.modules]
assert not heavy, f"顶层 import 拉入了重依赖: {heavy}"
print("OK")
""")


# ── U10 用户体验 ────────────────────────────────────────────────────
def u10() -> None:
    expect_ok("U10", "特化命中提示含可操作信息", """
import numpy as np, pyqcd.einsum as pe
import io, contextlib
eq = "cwdx,eufv,msnt,okpl,igjh,qarb,ac,mo,Ggi,Hkq,su,bd,fnp,Mhj,Mlr,tvx->MGHew"
ops = [np.random.default_rng(i).standard_normal(tuple(2 if c.isupper() else 3 for c in t))
       for i, t in enumerate(eq.split("->")[0].split(","))]
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    pe.contract(eq, *ops)
text = buf.getvalue()
# 提示语言跟随 locale（zh_* 说中文，其余说英文），两版都必须出现
assert ("专用路径" in text) or ("specialized path" in text), text[:200]
assert "kernel" in text or "plan-cache" in text, text[:200]
print("OK")
""")
    expect_ok("U10", "build_info 字段语义清晰", """
import pyqcd.einsum as pe
info = pe.build_info()
for key in ("backend", "native_available", "array_backend"):
    assert key in info, key
assert info["backend"] in ("python", "kupl", "cuda"), info["backend"]
assert isinstance(info["native_available"], bool)
print("OK")
""")


def main() -> int:
    rounds = [
        ("U1", u1), ("U2", u2), ("U3", u3), ("U4", u4), ("U5", u5),
        ("U6", u6), ("U7", u7), ("U8", u8), ("U9", u9), ("U10", u10),
    ]
    for name, function in rounds:
        print(f"\n=== {name} ===", flush=True)
        try:
            function()
        except Exception as exc:                       # noqa: BLE001
            record(name, "执行异常", False,
                   f"{type(exc).__name__}: {exc}")
            traceback.print_exc()

    failed = [item for item in RESULTS if not item["ok"]]
    print(f"\n{'=' * 60}")
    print(f"用户接口十轮巡查：{len(RESULTS)} 项，"
          f"{len(RESULTS) - len(failed)} PASS / {len(failed)} FAIL")
    for item in failed:
        print(f"  FAIL [{item['round']}] {item['name']}: {item['detail']}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
