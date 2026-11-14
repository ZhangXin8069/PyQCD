#!/usr/bin/env python3
"""Direct calls to the lqcddb_new and lqcddb_update source trees.

Both source trees expose the top-level import name ``lqcddb``.  They are
therefore loaded here under distinct aliases so that both can be called from
one interpreter without one installation replacing the other.

For production MPI runs, prefer one package per process with the canonical
``lqcddb`` import name.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from itertools import product
from typing import Any

from lqcddb import wick_contraction
PACKAGE_ROOTS = {
    "lqcddb_new": Path("/public/home/sush/distillation/lqcddb_new/src"),
    "lqcddb_update": Path("/public/home/sush/distillation/lqcddb_update/src"),
}
__all__ = ["call_lqcddb_new", "call_lqcddb_update"]


def _configure_runtime() -> None:
    """Keep Matplotlib's cache in the workspace."""
    mpl_config = Path(__file__).resolve().parent / ".test_mpl"
    mpl_config.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))


def _load_lqcddb(alias: str, src_root: Path) -> ModuleType:
    """Load ``src_root/lqcddb`` under a unique module alias."""
    init_file = (src_root / "lqcddb" / "__init__.py").resolve()
    if not init_file.is_file():
        raise FileNotFoundError(f"lqcddb package not found: {init_file}")

    loaded = sys.modules.get(alias)
    if loaded is not None:
        loaded_file = Path(loaded.__file__).resolve()
        if loaded_file != init_file:
            raise RuntimeError(
                f"alias {alias!r} already points to {loaded_file}, not {init_file}"
            )
        return loaded

    spec = importlib.util.spec_from_file_location(
        alias,
        init_file,
        submodule_search_locations=[str(init_file.parent)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import specification for {init_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(alias, None)
        raise
    return module


def _prepare_pyqcd_terms(lqcddb: ModuleType, dynamic: Any,
                         peram: Any, vertices: Any, gammas: Any) -> list[tuple]:
    """Resolve dynamic plan entries for direct ``pyqcd.einsum`` timing.

    The plan, registry data, coefficients, and final einsum labels are kept
    identical to ``dynamic.calculate_all``.  Only the contraction call is
    replaced by ``pyqcd.einsum.contract`` in the benchmark loop.
    """
    dynamic_impl = importlib.import_module(
        f"{lqcddb.__name__}.contraction.dynamic")
    terms = []
    for entry in dynamic.plan:
        total_coeff = sum(coeff for _, _, coeff in entry[0])
        if abs(total_coeff) < 1e-12:
            continue

        _, _, wick, diagram_index, *_ = entry
        equation = wick["result_indx"][diagram_index][0]
        perams = wick["peram"][diagram_index]
        tensors = [peram.resolve(item[2], item[4]) for item in perams]
        tensors.extend(gammas.resolve(item[1]) for item in wick["gamma_pos"])
        v_names = dynamic_impl._per_region_v_names(wick["V"])
        tensors.extend(vertices.resolve(name, time)
                       for name, time in v_names)

        # Match calculate_contraction's optional output-index rewrite.
        if len(entry) > 5:
            lhs, rhs = equation.split("->")
            n_lower = 0
            for label in rhs:
                if label.islower():
                    n_lower += 1
                else:
                    break
            equation = f"{lhs}->{rhs[:n_lower]}{entry[5]}"
        terms.append((total_coeff, equation, tuple(tensors)))
    return terms


def _direct_call(package: str, backend_name: str = "numpy") -> dict[str, Any]:
    """Call the common public API directly from one source tree."""
    if package not in PACKAGE_ROOTS:
        raise ValueError(f"package must be one of {sorted(PACKAGE_ROOTS)}")

    _configure_runtime()
    lqcddb = _load_lqcddb(package, PACKAGE_ROOTS[package])
    lqcddb.set_backend(backend_name)
    backend = lqcddb.get_backend()

    pion_minus = ["|", "d", "u", "gamma_7", "d", "|", "|", "d", "u", "gamma_7", "d", "|"] #, 
    pion_plus = lqcddb.conjugate_operator(pion_minus)
    wick = lqcddb.wick_contraction(
        sink_operators=pion_minus,
        source_operators=pion_plus,
        Vindex = ["N", "N", "M", "M"],
        curr_operators=[],
        Cpt="2pt",
    )
    plan = lqcddb.run_wick_analysis(
        [(pion_minus, pion_plus)],
        Cpt="2pt",
        verbose=False,
    )
    gamma5 = backend.asarray(lqcddb.gamma(5))
    
    return {
        "package": package,
        "module": lqcddb,
        "module_file": Path(lqcddb.__file__).resolve(),
        "version": lqcddb.__version__,
        "backend": backend.__name__,
        "gamma5_shape": tuple(gamma5.shape),
        "source_operator": pion_plus,
        "wick": wick,
        "plan": plan,
    }


def call_lqcddb_new(backend_name: str = "numpy") -> dict[str, Any]:
    """Call lqcddb_new directly."""
    return _direct_call("lqcddb_new", backend_name)


def call_lqcddb_update(backend_name: str = "numpy") -> dict[str, Any]:
    """Call lqcddb_update directly."""
    return _direct_call("lqcddb_update", backend_name)



def main() -> None:
    from opt_einsum import contract as opc
    from opt_einsum import contract_path
    import time
    from pyqcd.einsum import contract as pyc
    import numpy as np

    # ==================== dynamic_contract 可编辑区域 ====================
    # 物理过程: proton sink <- J_4 <- proton source。
    # 直接修改下面的算符、Nev、动量数、时间标签、优化器和张量形状即可。
    backend_name = os.environ.get("LQCDDB_TEST_BACKEND", "numpy")
    nev_src = 100
    nev_curr = 400
    n_mom_src = 1
    n_mom_curr = 3
    
    seed = 20261114
    optimize = "dp"
    dtype = np.complex128
    
    times = ("tsink", "tsrc", "tcur0", "tcur1")
    # The distillation space depends on the time endpoint: sink/source use
    # ``nev_src`` eigenvectors, while each current endpoint uses ``nev_curr``.
    # Keep this mapping explicit so every registered perambulator has the
    # dimensions required by the VVV/VDV tensor at its two endpoints.
    nev_by_time = {
        "tsink": nev_src,
        "tsrc": nev_src,
        "tcur0": nev_curr,
        "tcur1": nev_curr,
    }
    sink_operator = ["|", "u", "u", "gamma_7", "d", "|"]
    source_operator = ["|", "d^d", "gamma_7", "u", "|", "|", "d^d", "u^d", "gamma_7", "d^d", "|"]
    current_operator = ["|", "u^d", "gamma_curr", "d", "|", "|", "u^d", "gamma_curr", "d", "|"]
    vindex = ["", "N", "N", "M", "M"]
    gindex = ["", "G", "G", "", ""]
    
    rng = np.random.default_rng(seed)
    # 所有 host 张量先统一生成，再分别转换到两个包的 backend，确保输入相同。
    host_peram = {
        (tq, ta): np.asarray(
            (rng.standard_normal((4, 4, nev_by_time[tq], nev_by_time[ta]))
             + 1j * rng.standard_normal((4, 4, nev_by_time[tq], nev_by_time[ta])))
            / np.sqrt(2.0),
            dtype=dtype,
        )
        for tq, ta in product(times, repeat=2)
    }
    
    host_vvv_sink = np.asarray(
        (rng.standard_normal((n_mom_src, nev_src, nev_src, nev_src))
         + 1j * rng.standard_normal((n_mom_src, nev_src, nev_src, nev_src))) / np.sqrt(2.0),
        dtype=dtype,
    )
    host_vvv_source = np.asarray(
        (rng.standard_normal((n_mom_src, nev_src, nev_src, nev_src))
         + 1j * rng.standard_normal((n_mom_src, nev_src, nev_src, nev_src))) / np.sqrt(2.0),
        dtype=dtype,
    )
    host_vdv_source = np.asarray(
        (rng.standard_normal((n_mom_src, nev_src, nev_src))
         + 1j * rng.standard_normal((n_mom_src, nev_src, nev_src))) / np.sqrt(2.0),
        dtype=dtype,
    )
    host_vdv_current = np.asarray(
        (rng.standard_normal((n_mom_curr, nev_curr, nev_curr))
         + 1j * rng.standard_normal((n_mom_curr, nev_curr, nev_curr))) / np.sqrt(2.0),
        dtype=dtype,
    )

    dynamic_results = {}
    repeat_count = max(1, int(os.environ.get("DYNAMIC_BENCHMARK_REPEATS", "20")))
    for package in ("lqcddb_new", "lqcddb_update"):
        lqcddb = _load_lqcddb(package, PACKAGE_ROOTS[package])
        lqcddb.set_backend(backend_name)
        backend = lqcddb.get_backend()

        peram = lqcddb.PeramRegistry()
        for (tq, ta), host_value in host_peram.items():
            peram.register("light", (tq, ta), backend.asarray(host_value))

        vertices = lqcddb.VRegistry()
        vertices.register("VVV_0", "tsink", backend.asarray(host_vvv_sink[0]))
        vertices.register("VVV_0", "tsrc", backend.asarray(host_vvv_source))
        vertices.register("VDV_0", "tsrc", backend.asarray(host_vdv_source))
        vertices.register("VDV_0", "tcur0", backend.asarray(host_vdv_current))
        vertices.register("VDV_0", "tcur1", backend.asarray(host_vdv_current))

        gammas = lqcddb.GammaRegistry()
        gammas.register("gamma_4", backend.asarray(lqcddb.gamma(4)), dtype=dtype)
        gammas.register("gamma_7", backend.asarray(lqcddb.gamma(7)), dtype=dtype)
        gammas.register("gamma_curr", backend.asarray([lqcddb.gamma(4), lqcddb.gamma(4), lqcddb.gamma(4), lqcddb.gamma(4)]), dtype=dtype)

        if hasattr(lqcddb, "clear_plan_cache"):
            lqcddb.clear_plan_cache()
        if hasattr(lqcddb, "clear_cache"):
            lqcddb.clear_cache()
        plan_start = time.perf_counter()
        dynamic = lqcddb.dynamic_contraction(
            [(sink_operator, source_operator, current_operator)],
            peram_registry=peram,
            v_registry=vertices,
            gamma_registry=gammas,
            Cpt="3pt",
            Vindex=vindex,
            Gindex = gindex,
            use_equivalence=False,
            ignore_dis=False,
            verbose=False,
            max_detail=0,
            optimize=optimize,
        )
        plan_seconds = time.perf_counter() - plan_start

        direct_terms = _prepare_pyqcd_terms(
            lqcddb, dynamic, peram, vertices, gammas)
        dynamic_timings = []
        direct_timings = []
        result_direct = None
        for _ in range(repeat_count):
            result_start = time.perf_counter()
            result_dynamic = dynamic.calculate_all()
            result_seconds = time.perf_counter() - result_start
            dynamic_timings.append(result_seconds)
            print(
                f"{package}.dynamic_contract: "
                f"plan={plan_seconds:.6f}s "
                f"result={result_seconds:.6f}s"
            )

            result_start = time.perf_counter()
            direct_total = 0
            for coefficient, equation, tensors in direct_terms:
                direct_total += coefficient * pyc(
                    equation, *tensors, optimize=optimize)
            result_direct = direct_total
            direct_seconds = time.perf_counter() - result_start
            direct_timings.append(direct_seconds)
            print(
                f"{package}.pyqcd_contract: "
                f"result={direct_seconds:.6f}s"
            )

        dynamic_results[package] = {
            "module": lqcddb,
            "dynamic": dynamic,
            "result": result_dynamic,
            "direct_result": result_direct,
            "plan_entries": len(dynamic.plan),
            "plan_seconds": plan_seconds,
            "result_seconds": dynamic_timings[-1],
            "direct_seconds": direct_timings[-1],
            "dynamic_mean_seconds": float(np.mean(dynamic_timings[1:] or dynamic_timings)),
            "direct_mean_seconds": float(np.mean(direct_timings[1:] or direct_timings)),
        }
        dynamic_mean = dynamic_results[package]["dynamic_mean_seconds"]
        direct_mean = dynamic_results[package]["direct_mean_seconds"]
        print(
            f"{package}.contract comparison: "
            f"dynamic_mean={dynamic_mean:.6f}s "
            f"pyqcd_mean={direct_mean:.6f}s "
            f"pyqcd/dynamic={direct_mean / max(dynamic_mean, 1e-300):.3f}x "
            f"delta={direct_mean - dynamic_mean:+.6f}s"
        )

    if backend_name == "cupy":
        import cupy as cp
        update_value = cp.asnumpy(dynamic_results["lqcddb_update"]["result"])
        new_value = cp.asnumpy(dynamic_results["lqcddb_new"]["result"])
    else:
        update_value = np.asarray(dynamic_results["lqcddb_update"]["result"])
        new_value = np.asarray(dynamic_results["lqcddb_new"]["result"])
    if update_value.shape != new_value.shape:
        raise AssertionError(
            f"dynamic result shape mismatch: update={update_value.shape}, "
            f"new={new_value.shape}")
    scale = max(
        float(np.max(np.abs(update_value), initial=0.0)),
        float(np.max(np.abs(new_value), initial=0.0)),
        1e-300,
    )
    relative_error = float(np.max(np.abs(update_value - new_value), initial=0.0)) / scale
    print(
        f"dynamic_contract comparison: shape={update_value.shape} "
        f"relative_error={relative_error:.3e} "
        f"equal={relative_error <= 1e-10}"
    )
    for package, values in dynamic_results.items():
        direct_value = values["direct_result"]
        if backend_name == "cupy":
            direct_value = cp.asnumpy(direct_value)
        else:
            direct_value = np.asarray(direct_value)
        if direct_value.shape != np.asarray(values["result"]).shape:
            raise AssertionError(
                f"{package} dynamic/pyqcd shape mismatch: "
                f"dynamic={np.asarray(values['result']).shape}, "
                f"pyqcd={direct_value.shape}")
        direct_scale = max(
            float(np.max(np.abs(np.asarray(values["result"])), initial=0.0)),
            float(np.max(np.abs(direct_value), initial=0.0)),
            1e-300,
        )
        direct_error = float(
            np.max(np.abs(np.asarray(values["result"]) - direct_value), initial=0.0)
        ) / direct_scale
        print(
            f"{package}.dynamic/pyqcd comparison: "
            f"shape={direct_value.shape} "
            f"relative_error={direct_error:.3e} "
            f"equal={direct_error <= 1e-10}"
        )
    
    # 设置 ONLY_DYNAMIC_CONTRACT=1 可只运行上面的动态部分。
    if os.environ.get("ONLY_DYNAMIC_CONTRACT") == "1":
        return
    # ================== dynamic_contract 区域结束 ==================

    peram1 = np.random.random((4, 4, 100, 100)) + 1j*np.random.random((4, 4, 100, 100))
    peram2 = np.random.random((4, 4, 100, 100)) + 1j*np.random.random((4, 4, 100, 100))
    peram3 = np.random.random((4, 4, 100, 100)) + 1j*np.random.random((4, 4, 100, 100))
    peram4 = np.random.random((4, 4, 100, 100)) + 1j*np.random.random((4, 4, 100, 100))
    peram5 = np.random.random((4, 4, 100, 100)) + 1j*np.random.random((4, 4, 100, 100))
    peram6 = np.random.random((4, 4, 100, 100)) + 1j*np.random.random((4, 4, 100, 100))
    peram = [peram1, peram2, peram3, peram4, peram5, peram6]
    
    gamma1 = np.random.random((4, 4)) + 1j*np.random.random((4, 4))
    gamma2 = np.random.random((4, 4)) + 1j*np.random.random((4, 4))
    gamma3 = np.random.random((4, 4)) + 1j*np.random.random((4, 4))
    gamma4 = np.random.random((4, 4)) + 1j*np.random.random((4, 4))
    gamma = [gamma1, gamma2, gamma3, gamma4]
    VVV_shape = (3, 100, 100, 100)
    VDV_shape = (3, 100, 100)
    
    VVV1 = np.random.random(VVV_shape) + 1j*np.random.random(VVV_shape)
    VVV2 = np.random.random(VVV_shape) + 1j*np.random.random(VVV_shape)
    VVV3 = np.random.random(VVV_shape) + 1j*np.random.random(VVV_shape)
    VVV4 = np.random.random(VVV_shape) + 1j*np.random.random(VVV_shape)
    VVV = [VVV1, VVV2, VVV3, VVV4]
    
    VDV1 = np.random.random(VDV_shape) + 1j*np.random.random(VDV_shape)
    VDV2 = np.random.random(VDV_shape) + 1j*np.random.random(VDV_shape)
    VDV = [VDV1, VDV2]
    
    _loop = 6
    result = call_lqcddb_new()
    # pyfc is the contract exported by the loaded lqcddb_new source tree.
    pyfc = result["module"].contract
    # print(result['wick']['result_indx'])
    # exit()
    # for j in range(len(result['wick']['result_indx'])):
    for j in range(4):
    
        print(f"*************************{j}***********************")
        print("*****************pyfc****************")
        A = []
        for i in range(_loop):
            st_opc = time.perf_counter()
            result_pyfc = pyfc(*result['wick']['result_indx'][j], *peram[:6], *gamma, *VVV[:4], optimize = 'auto')
            ed_opc = time.perf_counter()
            A = A + [ed_opc - st_opc]
            print(f'pyfc use time {ed_opc - st_opc} s for loop {i}')
        print(f'+++++++++++++mean pyfc use time {sum(A[1:])/(_loop-1)} s')
        
        print("*****************pyc****************")
        A = []
        for i in range(_loop):
            st_opc = time.perf_counter()
            result_pyc = pyc(*result['wick']['result_indx'][j], *peram[:6], *gamma, *VVV[:4], optimize = 'auto')
            ed_opc = time.perf_counter()
            A = A + [ed_opc - st_opc]
            print(f'pyc use time {ed_opc - st_opc} s for loop {i}')
        print(f'+++++++++++++mean pyc use time {sum(A[1:])/(_loop-1)} s')
        

        
        # print("*****************opc****************")
        # A = []
        # for i in range(_loop):
        #     st_opc = time.perf_counter()
        #     result_opc = opc(*result['wick']['result_indx'][j], *peram[:5], *gamma, VDV1, *VVV[:2], VDV2, optimize = 'auto')
        #     ed_opc = time.perf_counter()
        #     A = A + [ed_opc - st_opc]
        #     print(f'opc use time {ed_opc - st_opc} s for loop {i}')
        # print(f'+++++++++++++mean opc use time {sum(A[1:])/(_loop-1)} s')


if __name__ == "__main__":
    main()
