"""契约测试：单条方程的特化路径**查询**入口（生产包，纯查询、无副作用）。

守护四件事：

1. **判定正确**：命中语料返回 HIT（含语料出处），语料外返回 MISS，
   非法方程返回不可解析；三者对应 CLI 退出码 0/1/2。
2. **字段口径**：``name`` 必须等于运行时提示里的 ``topology:<摘要前 8 位>``，
   ``digest`` 必须与 ``_specialize._signature_digest`` 一致。
3. **无副作用**：查询不打印、不触碰 ``_specialize._announced``、不调用
   ``contract()``（因此也不产生任何原生调用或计划缓存条目）。
4. **与测试包分工**：生产包只查询、不做数值；数值验证在
   ``pyqcd.testing.einsum.verify_specialized``，那个模块不随 pip 发布。

可直接运行::

    python -m pyqcd.testing._specialized_query_contract
"""
from __future__ import annotations

import contextlib
import io
import os

import numpy as np

from ..einsum import _api as api
from ..einsum import _specialize as spec
from ..einsum import query_specialization
from ..einsum import verify_specialized as query_module

# 目标方程：情形 10「重子四点函数」seq=1650，13 算子、输出秩 2。
EQUATION = "agbh,cmdn,eqfr,ospt,ikjl,ce,mo,gi,kq,bdf,np,hj,lrt->as"
DIGEST = "8006c502f8072c0e0a6db314"


def test_specialized_query_hits_corpus() -> None:
    report = query_specialization(EQUATION)
    assert report["parsable"] and report["matched"], report
    assert report["digest"] == DIGEST, report["digest"]
    assert report["name"] == "topology:" + DIGEST[:8], report["name"]
    assert report["arity"] == 13 and report["output_rank"] == 2, report
    assert report["kernels"] == ["kupl_topology", "cuda_topology"], report["kernels"]
    corpus = report["corpus"]
    assert corpus["seq"] == 1650 and corpus["tid"] == "T00028", corpus
    assert corpus["case"].startswith("10."), corpus["case"]
    assert corpus["equation"] == EQUATION, corpus["equation"]
    assert len(corpus["roles"]) == report["arity"], corpus["roles"]


def test_specialized_query_digest_matches_runtime_notice() -> None:
    """name 必须与 announce() 打印的形态名逐字一致，否则日志对不上查询结果。"""
    signature = spec.canonical_signature(EQUATION)
    entry = spec.match(EQUATION)
    assert entry is not None
    assert entry["name"] == query_specialization(EQUATION)["name"]
    assert entry["name"] == "topology:" + spec._signature_digest(signature)[:8]


def test_specialized_query_misses_outside_corpus() -> None:
    for equation in ("ij,jk->ik", "ab,bc->ac", "ijk->i", "i,j->ij"):
        report = query_specialization(equation)
        assert report["parsable"] is True, equation
        assert report["matched"] is False, equation
        assert "corpus" not in report, equation
        assert report["reason"], equation


def test_specialized_query_rejects_malformed() -> None:
    for bad in ("i->->i", "i,,->i", "ij->ii", "", "   ", None, 123, ["ij"]):
        report = query_specialized_or_direct(bad)
        assert report["parsable"] is False, bad
        assert report["matched"] is False, bad
        assert "corpus" not in report, bad


def query_specialized_or_direct(equation):
    """非字符串经公开 API 会先被类型检查拦下，两者都不得抛异常。"""
    if isinstance(equation, str):
        return query_specialization(equation)
    return query_module.query(equation)


def test_specialized_query_cli_exit_codes() -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        assert query_module.main([EQUATION]) == 0
        assert query_module.main(["ij,jk->ik"]) == 1
        assert query_module.main(["ij->ii"]) == 2


def test_specialized_query_cli_json() -> None:
    import json

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = query_module.main([EQUATION, "--json"])
    assert code == 0
    payload = json.loads(buffer.getvalue())
    assert payload["name"] == "topology:" + DIGEST[:8]
    assert payload["corpus"]["seq"] == 1650


def test_specialized_query_has_no_side_effects() -> None:
    """不打印、不改提示状态、不进入 contract 分派链。"""
    spec._announced.clear()
    before = set(spec._announced)
    buffer = io.StringIO()
    calls: list[tuple] = []

    original_contract = api.contract
    try:
        api.contract = lambda *a, **k: calls.append((a, k))
        with contextlib.redirect_stdout(buffer):
            query_specialization(EQUATION)
            query_specialization("ij,jk->ik")
    finally:
        api.contract = original_contract

    assert buffer.getvalue() == "", buffer.getvalue()
    assert set(spec._announced) == before
    assert calls == [], calls


def test_specialized_query_respects_specialize_switch() -> None:
    """PYQCD_EINSUM_SPECIALIZE=0 时如实报告开关状态（只报状态，不改命中判定）。"""
    saved = os.environ.get(spec.DISABLE_ENV)
    try:
        os.environ[spec.DISABLE_ENV] = "0"
        assert query_specialization(EQUATION)["specialize_enabled"] is False
        os.environ.pop(spec.DISABLE_ENV)
        assert query_specialization(EQUATION)["specialize_enabled"] is True
    finally:
        if saved is None:
            os.environ.pop(spec.DISABLE_ENV, None)
        else:
            os.environ[spec.DISABLE_ENV] = saved


def test_specialized_query_matches_every_corpus_entry() -> None:
    """抽查语料：每条都必须自报 matched 且摘要与注册表一致（不整扫，控制耗时）。"""
    from ..einsum import _wick_tasklist as tasklist

    sample = list(tasklist.TASKS)[::431]          # 约 21 条，覆盖各情形
    assert len(sample) >= 15, len(sample)
    for task in sample:
        report = query_specialization(task.equation)
        assert report["matched"], (task.seq, task.equation)
        assert report["name"] == "topology:" + spec._signature_digest(
            spec.canonical_signature(task.equation))[:8], task.seq
        assert report["corpus"]["seq"] == task.seq, task.seq


def test_specialized_query_does_not_compute() -> None:
    """生产包入口不生成操作数：RNG 未被消费即证明没建数组。"""
    state_before = np.random.get_state()
    query_specialization(EQUATION)
    state_after = np.random.get_state()
    assert state_before[1].tolist() == state_after[1].tolist()
    assert state_before[2] == state_after[2]


if __name__ == "__main__":
    from .conftest import main

    raise SystemExit(main())
