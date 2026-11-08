"""方程形态专用优化路径（pattern specialization）。

格点 QCD 的收缩形式高度固定。对已知形态，本模块给出**专用执行路径**，
并在命中时打印提示，便于与通用路径做性能对照。

开关优先级（高 → 低）：

1. 调用参数 ``contract(..., specialize=False)`` —— 单次调用关闭；
2. 环境变量 ``PYQCD_EINSUM_SPECIALIZE=0`` —— 进程级关闭；
3. 默认开启。

命中时输出（每个进程每个形态一次）：

    [pyqcd.einsum] 专用路径: gauge-propagator(E4) [fused: cuda_e4_fused]

设置 ``PYQCD_EINSUM_SPECIALIZE_VERBOSE=1`` 可每次调用都输出。
"""

from __future__ import annotations

import os
import threading

VERBOSE_ENV = "PYQCD_EINSUM_SPECIALIZE_VERBOSE"
DISABLE_ENV = "PYQCD_EINSUM_SPECIALIZE"
FUSED_DISABLE_ENV = "PYQCD_EINSUM_DISABLE_FUSED"

# 形态键为规范化后的 subscripts（去空白）。
REGISTRY: dict[str, dict[str, str]] = {
    "gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M": {
        "name": "nucleon-2pt(E1)",
        "kind": "kernel(needs-optimization)",
        "note": "8 算子核子 2pt；KUPL/CUDA 专用 kernel（形状不匹配时回落到计划缓存）",
    },
    "Geifj,ckdl,agbh,ce,gi,Mbdf,Mhjl,Gka->GM": {
        "name": "nucleon-3pt(E2)",
        "kind": "plan-cache",
        "note": "8 算子核子 3pt（仅复用计划缓存，无专用 kernel）",
    },
    "zyxcTabn,zyxgTdef,Gae,Tzyxgc->TGdbfn": {
        "name": "gauge-propagator(E4)",
        "kind": "plan-cache",
        "note": "peram × peram × γ × 规范场（本版本无专用 kernel，走计划缓存）",
    },
    "ab,...bce,cd->...dae": {
        "name": "gamma-sandwich(E5)",
        "kind": "plan-cache",
        "note": "γ₅ 夹心（本版本无专用 kernel，走计划缓存）",
    },
    "eofp,ambn,cgdh,ce,mo,Nbdf,Mnph,ga->MN": {
        "name": "sush-2pt(S1)",
        "kind": "plan-cache",
        "note": "sush NN 2pt 生产方程（仅复用计划缓存）",
    },
    "gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG": {
        "name": "sush-3pt-munu(S2)",
        "kind": "plan-cache",
        "note": "sush 3pt 含 μν 插入（仅复用计划缓存）",
    },
    "manb,cedf,bd,Mfn,ac,MFem->MF": {
        "name": "sush-2pt-mu(S3)",
        "kind": "plan-cache",
        "note": "sush 2pt 含 μ 插入（仅复用计划缓存）",
    },
    "agbh,eqfr,ospt,cmdn,ikjl,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt,ax,sy->MNGLxy": {
        "name": "sush-3pt-main(P1)",
        "kind": "kernel(needs-optimization)",
        "note": "sush 3pt 主项；专用 kernel 已接线（CUDA ≈1.8x 于通用；KUPL 待进一步优化）",
    },
    "agbh,eqfr,ospt,ckdl,imjn,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt,ax,sy->MNGLxy": {
        "name": "sush-3pt-main(P2)",
        "kind": "kernel(needs-optimization)",
        "note": "sush 3pt 主项（置换型）；专用 kernel 已接线（CUDA ≈1.8x；KUPL 待优化）",
    },
}

_announced: set[str] = set()
_lock = threading.Lock()


def normalize(subscripts: str) -> str:
    return "".join(subscripts.split())


def enabled(specialize: bool | None = None) -> bool:
    """判定本次调用是否启用专用路径。"""
    if specialize is not None:
        return bool(specialize)
    return os.environ.get(DISABLE_ENV, "1").strip().lower() not in (
        "0", "false", "off", "no")


def match(subscripts: str) -> dict[str, str] | None:
    """返回命中的专用路径描述，未命中返回 None。"""
    if not isinstance(subscripts, str):
        return None
    return REGISTRY.get(normalize(subscripts))


def announce(entry: dict[str, str], kernel: str = "") -> None:
    """命中时打印提示；默认每形态每个进程只打印一次。

    ``kernel`` 是**实际执行**的专用内核名（空串表示只命中计划缓存），
    提示里据实输出，避免把没有 kernel 的形态说成 fused。
    """
    name = entry.get("name", "unknown")
    verbose = os.environ.get(VERBOSE_ENV, "").strip().lower() in (
        "1", "true", "on", "yes")
    with _lock:
        if not verbose and name in _announced:
            return
        _announced.add(name)
    kind = f"kernel: {kernel}" if kernel else "plan-cache"
    note = entry.get("note", "")
    print(f"[pyqcd.einsum] 专用路径: {name} [{kind}] {note}", flush=True)


def describe() -> list[tuple[str, str, str]]:
    """列出所有已注册形态：(name, kind, subscripts)。"""
    return [(item["name"], item["kind"], key) for key, item in REGISTRY.items()]
