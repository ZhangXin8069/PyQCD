from __future__ import annotations

import re
from typing import Any

import yaml


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def resolve_submit_config_presets(
    submit_config: dict[str, Any],
    ensemble_presets: dict[str, Any],
) -> dict[str, Any]:
    resolved = dict(submit_config)
    ensemble_cfg = dict(resolved.get("ensemble", {}) or {})
    solver_cfg = dict(resolved.get("solver", {}) or {})
    runtime_cfg = dict(resolved.get("runtime", {}) or {})

    preset_name = str(ensemble_cfg.get("preset", "")).strip()
    if not preset_name:
        resolved["ensemble"] = ensemble_cfg
        resolved["solver"] = solver_cfg
        resolved["runtime"] = runtime_cfg
        return resolved

    preset = ensemble_presets.get(preset_name)
    if not isinstance(preset, dict):
        raise ValueError(f"unknown ensemble preset: {preset_name}")

    dimensions = preset.get("dimensions")
    if isinstance(dimensions, list) and len(dimensions) == 4:
        _set_default_from_preset(
            ensemble_cfg,
            "geometry",
            [int(dim) for dim in dimensions],
        )

    _set_default_from_preset(ensemble_cfg, "name", preset_name)
    _set_default_from_preset(
        ensemble_cfg,
        "lattice_spacing",
        preset.get("lattice_spacing", preset.get("spacing")),
    )
    temporal_bc = ""
    boundary_conditions = preset.get("boundary_conditions")
    if isinstance(boundary_conditions, dict):
        temporal_bc = str(boundary_conditions.get("temporal", "")).strip().lower()
    if temporal_bc == "anti-periodic":
        _set_default_from_preset(ensemble_cfg, "t_boundary", -1)
    elif temporal_bc == "periodic":
        _set_default_from_preset(ensemble_cfg, "t_boundary", 1)
    _set_default_from_preset(ensemble_cfg, "anisotropy", preset.get("anisotropy"))
    _set_default_from_preset(ensemble_cfg, "process_grid", preset.get("process_grid"))

    _set_default_from_preset(solver_cfg, "xi_0", preset.get("xi_0"))

    clover = preset.get("clover")
    if clover is not None:
        _set_default_from_preset(solver_cfg, "clover_coeff_t", clover)
        _set_default_from_preset(solver_cfg, "clover_coeff_r", clover)

    quark_mass = preset.get("quark_mass")
    if _is_empty_override(solver_cfg.get("mass")):
        if isinstance(quark_mass, dict) and "light" in quark_mass:
            solver_cfg["mass"] = quark_mass["light"]
        elif "l_mass" in preset:
            solver_cfg["mass"] = preset["l_mass"]

    _set_default_from_preset(solver_cfg, "multigrid", preset.get("multigrid"))

    mpi_num = preset.get("mpi_num")
    if mpi_num is not None:
        _set_default_from_preset(runtime_cfg, "mpi_num", int(mpi_num))

    launch = build_runtime_launch(runtime_cfg)
    if launch:
        runtime_cfg["launch"] = launch

    resolved["ensemble"] = ensemble_cfg
    resolved["solver"] = solver_cfg
    resolved["runtime"] = runtime_cfg
    return resolved


def build_prompt_ensemble_yaml(
    raw_config: dict[str, Any],
    ensemble_presets: dict[str, Any],
) -> str:
    if not isinstance(raw_config, dict):
        return ""

    ensemble_cfg = dict(raw_config.get("ensemble", {}) or {})
    runtime_cfg = dict(raw_config.get("runtime", {}) or {})
    preset_name = str(ensemble_cfg.get("preset", "")).strip()
    preset = ensemble_presets.get(preset_name) if preset_name else None
    if not isinstance(preset, dict):
        preset = {}

    ensemble_block = _compact_mapping(
        {
            "preset": preset_name or None,
            "cfg_num": ensemble_cfg.get("cfg_num"),
            "cfg_path": preset.get("cfg_path"),
            "dimensions": preset.get("dimensions"),
            "boundary_conditions": preset.get("boundary_conditions"),
            "lattice_spacing": preset.get("lattice_spacing"),
            "anisotropy": preset.get("anisotropy"),
            "quark_mass": preset.get("quark_mass"),
            "xi_0": preset.get("xi_0"),
            "clover": preset.get("clover"),
            "multigrid": preset.get("multigrid"),
            "mpi_num": preset.get("mpi_num"),
            "process_grid": preset.get("process_grid"),
        }
    )
    payload = {"ensemble": ensemble_block} if ensemble_block else {}
    return yaml.safe_dump(payload, allow_unicode=True, sort_keys=False).strip()


def build_prompt_submit_config_yaml(
    raw_config: dict[str, Any],
    ensemble_presets: dict[str, Any],
) -> str:
    if not isinstance(raw_config, dict):
        return ""

    payload: dict[str, Any] = {}
    for key in ["script", "slurm", "runtime"]:
        value = raw_config.get(key)
        if isinstance(value, dict):
            compact_value = _compact_mapping(dict(value))
            if compact_value:
                payload[key] = compact_value

    ensemble_yaml = build_prompt_ensemble_yaml(raw_config, ensemble_presets)
    if ensemble_yaml:
        ensemble_payload = yaml.safe_load(ensemble_yaml) or {}
        if isinstance(ensemble_payload, dict) and ensemble_payload.get("ensemble"):
            payload["ensemble"] = ensemble_payload["ensemble"]

    return yaml.safe_dump(payload, allow_unicode=True, sort_keys=False).strip()


def build_runtime_launch(runtime_cfg: dict[str, Any]) -> str:
    launch = str(runtime_cfg.get("launch", "")).strip()
    mpi_num = runtime_cfg.get("mpi_num")
    if mpi_num is None:
        return launch

    if launch:
        launch = re.sub(r"(^|\s)-(?:n|np)\s+\d+\b", lambda m: f"{m.group(1)}-n {int(mpi_num)}", launch, count=1)
        if re.search(r"(^|\s)-(?:n|np)\s+\d+\b", launch):
            return launch
        if "mpirun" in launch or "srun" in launch:
            return f"{launch} -n {int(mpi_num)}"
        return launch

    launch_mpi = str(runtime_cfg.get("launch_mpi", "mpirun")).strip() or "mpirun"
    launch_python = str(runtime_cfg.get("launch_python", "python3")).strip() or "python3"
    return f"{launch_mpi} -n {int(mpi_num)} {launch_python}"


def _set_default_from_preset(section: dict[str, Any], key: str, value: Any) -> None:
    if _is_empty_override(section.get(key)):
        section[key] = value


def _is_empty_override(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False


def _compact_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, dict):
            nested = _compact_mapping(value)
            if nested:
                compact[key] = nested
            continue
        if _is_empty_override(value):
            continue
        compact[key] = value
    return compact
