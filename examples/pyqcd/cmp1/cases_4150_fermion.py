"""4150 费米子 2pt 对照的纯配置与比较基元。

本模块不读取数组，也不依赖参考运行时；它只固定时间对、文件命名和选定切片
比较规则，供 ``run_4150_fermion.py`` 与测试共同使用。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FermionConfig:
    """一条 2pt 实时链所需的可复现参数。"""

    conf_id: str = "4150"
    nx: int = 24
    nt: int = 72
    nev: int = 100
    momentum: tuple[int, int, int] = (0, 0, 0)  # (Pz, Py, Px)
    momentum_smear: int = 0
    variant: str = "Cg5g4"
    t_sources: tuple[int, ...] = (0,)
    delta_t_min: int = 2
    delta_t_max: int = 36

    def __post_init__(self):
        if str(self.conf_id) != "4150":
            raise ValueError(f"本案例只允许组态 4150，收到 {self.conf_id!r}")
        if self.nx <= 0 or self.nt <= 0 or self.nev <= 0:
            raise ValueError("nx、nt、nev 必须为正数")
        if len(self.momentum) != 3:
            raise ValueError("momentum 必须按 (Pz, Py, Px) 给出三分量")
        if not 0 <= self.delta_t_min <= self.delta_t_max < self.nt:
            raise ValueError("delta_t 必须满足 0 <= min <= max < nt")
        sources = tuple(sorted(set(int(t) for t in self.t_sources)))
        if not sources or any(t < 0 or t >= self.nt for t in sources):
            raise ValueError("t_sources 必须是 [0, nt) 内的非空时间片集合")
        object.__setattr__(self, "conf_id", str(self.conf_id))
        object.__setattr__(self, "t_sources", sources)


def selected_pairs(config: FermionConfig) -> tuple[tuple[int, int, int], ...]:
    """返回 ``(t_sink, t_source, delta_t)``，顺序与参考循环一致。"""
    return tuple(
        ((t_source + delta_t) % config.nt, t_source, delta_t)
        for t_source in config.t_sources
        for delta_t in range(config.delta_t_min, config.delta_t_max + 1)
    )


def required_vvv_times(config: FermionConfig) -> tuple[int, ...]:
    """返回本次计算所需的 sink/source VVV 时间片并排序。"""
    return tuple(sorted({t_sink for t_sink, _, _ in selected_pairs(config)} |
                        set(config.t_sources)))


def reference_output_paths(reference_dir: str | Path, config: FermionConfig) -> dict[str, Path]:
    """构造 donghx ``momsmear0_Cg5g4`` 的 contract/nopol 文件名。"""
    pz, py, px = config.momentum
    phase = config.momentum_smear
    # 参考目录 momsmear0_Cg5 的默认 Cg5 文件名不带 ``_Cg5``，其余
    # 显式变体（如 Cg5g4）保留后缀。
    variant_suffix = "" if config.variant == "Cg5" else f"_{config.variant}"
    stem = f"twopt_slice_pp_Px{px}Py{py}Pz{pz}_eginphase{phase}{variant_suffix}"
    root = Path(reference_dir)
    return {
        "contract": root / f"{stem}_contract_conf{config.conf_id}.npy",
        "nopol_pp": root / f"{stem}_nopol_ss_conf{config.conf_id}.npy",
    }


def array_meta(value) -> dict:
    """为结果摘要生成不包含数组内容的形状/dtype/norm 元数据。"""
    arr = np.asarray(value)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "norm": float(np.linalg.norm(arr)),
        "finite": bool(np.isfinite(arr).all()),
    }


def compare_selected(reference, actual, pairs, *, tolerance=1e-10,
                     denominator_floor=1e-300) -> dict:
    """只比较本次真实计算的时间对，返回可序列化的误差证据。"""
    ref = np.asarray(reference)
    got = np.asarray(actual)
    ref_values = np.asarray([ref[t_sink, t_source] for t_sink, t_source, _ in pairs])
    got_values = np.asarray([got[t_sink, t_source] for t_sink, t_source, _ in pairs])
    if ref_values.shape != got_values.shape:
        return {
            "status": "diff",
            "metric": "selected_rel_l2",
            "value": float("inf"),
            "max_abs": float("inf"),
            "tolerance": tolerance,
            "reference_shape": list(ref.shape),
            "actual_shape": list(got.shape),
        }
    delta = got_values - ref_values
    ref_norm = float(np.linalg.norm(ref_values))
    value = float(np.linalg.norm(delta) / max(ref_norm, denominator_floor))
    return {
        "status": "pass" if value <= tolerance else "diff",
        "metric": "selected_rel_l2",
        "value": value,
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
        "tolerance": tolerance,
        "reference_norm": ref_norm,
        "actual_norm": float(np.linalg.norm(got_values)),
        "pair_count": len(pairs),
    }
