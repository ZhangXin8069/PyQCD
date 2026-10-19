#!/usr/bin/env python3
"""L32x64 raw-v2 audit and separately named Note-basis OPE products.

The compatibility kernel's helicity `combined` slot is TH+SH.  It is never
used to construct the Note target.  Both targets are rebuilt from primitives.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
ENSEMBLE = "beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64"
GAUGE_ROOT = Path("/public/group/lqcd/configurations/CLOVER") / ENSEMBLE
TAUS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.80, 1.00)
CONFS = tuple(range(10000, 10761, 40))
PROJECTIONS = ["legacy_untraced", "traceless"]
ORIENTATIONS = ["plus_z_raw", "minus_z_raw", "even_sum", "odd_difference"]
RAW_AXES = ["field_projection", "channel", "z_orientation", "component", "z", "t"]
RAW_COMPONENTS = ["combined", "Mtiti", "Mijij", "ti", "tj", "ij_single"]
PRIMITIVES = {
    "unpolarized/TU.npy": (0, 1), "unpolarized/SU.npy": (0, 2),
    "helicity/TH.npy": (1, 1), "helicity/SH.npy": (1, 2),
}
TARGETS = {
    "unpolarized/U_TminusS_even.npy": (0, 2, -1),
    "helicity/H_TminusS_odd.npy": (1, 3, -1),
    "helicity/H_TplusS_odd.npy": (1, 3, +1),
    "helicity/H_T_odd.npy": (1, 3, 0),
}
CONVENTIONS = {
    "coordinate_indices_xyzt": [0, 1, 2, 3],
    "gauge_storage_order": ["t", "z", "y", "x", "mu", "color", "color"],
    "epsilon_xyzt": 1,
    "field_strength": "four_leaf_clover_-i_QminusQdagger_over_8",
    "U_munu": "sum_x Tr[F_munu(x+z) W_dagger F_munu(x) W]",
    "H_munu": "sum_x Tr[F_munu(x+z) W_dagger dualF_munu(x) W]",
    "transverse_i_j": "i=(zdir+1)%3; j=(zdir+2)%3",
    "TU": "U_ti+U_tj", "SU": "2*U_ij",
    "TH": "H_ti+H_tj", "SH": "2*H_ij",
    "unpolarized_target": "(TU-SU)_even",
    "helicity_target": "(TH-SH)_odd",
    "helicity_diagnostics": ["(TH+SH)_odd", "TH_odd"],
    "even": "O(+z)+O(-z)", "odd": "O(+z)-O(-z)",
    "orientation_factor_half": False,
    "normalization": "bare_spatial_sum_no_volume_color_g_or_a_factors",
    "wick_phase_helicity": "+i relates Euclidean target to Minkowski; not applied here",
    "complex_projection": "full_complex_no_real_or_imag_projection",
    "field_and_line_flow": "same_Wilson_flowed_gauge_at_same_tau",
    "kernel_precision": "complex64 matrix algebra; complex128 spatial accumulation and output",
    "legacy_raw_helicity_combined": "TH+SH; diagnostic only; ignored for Note target",
    "note_source": "/public/home/donghx/Note/Gradient_Flow/main.tex",
    "note_equations": ["eq:flowed-unpolarized-operator", "eq:flowed-helicity-operator",
                       "eq:helicity-correct-odd-combination"],
}
IDENTITY_KEYS = ("conf_id", "gauge_path", "input_size", "input_mtime_ns",
                 "lime_payload_offset", "lime_payload_bytes", "lattice_xyzt",
                 "a_fm", "tau", "epsilon", "z_dir", "z_count")


def tau_tag(tau):
    return f"tau{float(tau):.3f}".replace(".", "p")


def raw_base(root, req):
    tag = tau_tag(req["tau"])
    return Path(root) / f"conf{req['conf_id']}" / tag / (
        f"conf{req['conf_id']}_{tag}_eps{req['epsilon']:.3f}_zdir{req['z_dir']}")


def output_dir(req):
    return ROOT / "results" / f"conf{req['conf_id']}" / tau_tag(req["tau"]) / f"zdir{req['z_dir']}"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_bytes(json.dumps(value, indent=2, ensure_ascii=False,
                                        allow_nan=False).encode("utf-8") + b"\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_npy(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, data, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(req):
    return {key: req[key] for key in IDENTITY_KEYS}


def assert_input(req):
    stat = Path(req["gauge_path"]).stat()
    require(stat.st_size == req["input_size"] and stat.st_mtime_ns == req["input_mtime_ns"],
            "input gauge size/mtime changed since manifest preparation")


def validate_raw(base, req):
    """Validate raw primitives, legacy algebra and complete physical identity."""
    base = Path(base)
    meta = json.loads(Path(str(base) + ".json").read_text())
    data = np.load(Path(str(base) + ".npy"), mmap_mode="r", allow_pickle=False)
    shape = (2, 2, 4, 6, req["z_count"], req["lattice_xyzt"][3])
    require(meta.get("schema") == "gradient_flow_gluon_ope_v2", "raw schema")
    require(meta.get("status") == "flowed_bare_ope_observable", "raw status")
    require(str(meta.get("conf_id")) == str(req["conf_id"]), "raw conf_id")
    require(meta.get("input_path") == req["gauge_path"], "raw gauge path")
    require(meta.get("input_scheme") == "thin_link_no_hyp_no_smear", "raw thin input")
    require(meta.get("lattice_xyzt") == req["lattice_xyzt"], "raw lattice")
    for key in ("lime_payload_offset", "lime_payload_bytes"):
        require(meta.get(key) == req[key], f"raw {key}")
    flow = meta.get("flow", {})
    require(flow.get("type") == "Wilson", "raw flow type")
    require(flow.get("integrator") == "QUDA_RK3_coefficients_CPU_OpenMP", "raw integrator")
    require(abs(flow.get("tau_t_over_a2", -1) - req["tau"]) < 1e-12, "raw tau")
    require(abs(flow.get("epsilon", -1) - req["epsilon"]) < 1e-12, "raw epsilon")
    require(flow.get("steps") == round(req["tau"] / req["epsilon"]), "raw nstep")
    operator = meta.get("operator", {})
    require(operator.get("z_dir") == req["z_dir"], "raw zdir")
    require(operator.get("z_values_a") == list(range(req["z_count"])), "raw z grid")
    require(operator.get("representation") == "fundamental_closed_trace", "raw color trace")
    require(operator.get("normalization") == "bare_spatial_sum", "raw normalization")
    require(operator.get("sum_kind") == "full_spatial_sum_per_timeslice", "raw spatial sum")
    require(operator.get("field_strength") == CONVENTIONS["field_strength"], "raw Clover")
    require(operator.get("field_projections") == PROJECTIONS, "raw field projections")
    require(operator.get("z_orientations") == ORIENTATIONS, "raw operator orientations")
    require(meta.get("axes") == RAW_AXES, "raw axes")
    labels = meta.get("axis_labels", {})
    require(labels.get("field_projection") == PROJECTIONS, "raw projection labels")
    require(labels.get("channel") == ["unpolarized", "helicity"], "raw channel labels")
    require(labels.get("z_orientation") == ORIENTATIONS, "raw orientation labels")
    require(labels.get("component") == RAW_COMPONENTS, "raw component labels")
    require(data.shape == shape and tuple(meta.get("shape", ())) == shape, "raw shape")
    require(data.dtype == np.dtype("complex128") and meta.get("dtype") == "complex128", "raw dtype")
    require(bool(np.isfinite(data).all()), "raw NaN/Inf")
    for key, tolerance in (("raw_gauge_stats", 5e-5), ("flowed_gauge_stats", 5e-4)):
        stats = meta.get(key, {})
        for field in ("max_unitarity_frobenius", "max_abs_det_minus_1"):
            value = float(stats.get(field, math.inf))
            require(math.isfinite(value) and 0 <= value < tolerance, f"{key} {field}")
    p0 = float(meta["raw_gauge_stats"].get("plaquette", math.nan))
    p1 = float(meta["flowed_gauge_stats"].get("plaquette", math.nan))
    require(math.isfinite(p0) and math.isfinite(p1) and p1 + 1e-7 >= p0, "flow plaquette")
    close = lambda a, b: bool(np.allclose(a, b, rtol=2e-13, atol=1e-10))
    plus, minus, even, odd = (data[:, :, o] for o in range(4))
    require(close(even, plus + minus), "raw even algebra")
    require(close(odd, plus - minus), "raw odd algebra")
    require(close(plus[..., 0, :], minus[..., 0, :]), "raw z=0 orientations")
    require(close(odd[..., 0, :], 0), "raw odd z=0")
    require(close(data[:, :, :, 1], data[:, :, :, 3] + data[:, :, :, 4]), "raw T algebra")
    require(close(data[:, :, :, 2], 2 * data[:, :, :, 5]), "raw S algebra")
    require(close(data[:, 0, :, 0], data[:, 0, :, 1] - data[:, 0, :, 2]), "raw U combined")
    require(close(data[:, 1, :, 0], data[:, 1, :, 1] + data[:, 1, :, 2]), "legacy H combined")
    return data, meta


def arrays_from_raw(data):
    arrays = {name: np.ascontiguousarray(data[:, c, :, k])
              for name, (c, k) in PRIMITIVES.items()}
    for name, (channel, orientation, coefficient) in TARGETS.items():
        arrays[name] = np.ascontiguousarray(data[:, channel, orientation, 1]
                                           + coefficient * data[:, channel, orientation, 2])
    return arrays


def validate_product(destination, req, require_complete=True):
    destination = Path(destination)
    metadata_path = destination / "metadata.json"
    if require_complete:
        complete = json.loads((destination / "complete.json").read_text())
        require(complete.get("status") == "validated_complete", "completion status")
        require(complete.get("metadata_sha256") == sha256(metadata_path), "metadata checksum")
    meta = json.loads(metadata_path.read_text())
    require(meta.get("schema") == "gradient_flow_gluon_TU_SU_TH_SH_v1", "product schema")
    require(meta.get("identity") == identity(req), "product requested identity")
    require(meta.get("conventions") == CONVENTIONS, "product convention contract")
    require(meta.get("axes_primitives") == ["field_projection", "z_orientation", "z", "t"], "primitive axes")
    require(meta.get("axes_targets") == ["field_projection", "z", "t"], "target axes")
    require(meta.get("axis_labels") == {"field_projection": PROJECTIONS,
                                       "z_orientation": ORIENTATIONS}, "product axis labels")
    require(set(meta.get("artifacts", {})) == set(PRIMITIVES) | set(TARGETS), "product file set")
    arrays = {}
    for name in meta["artifacts"]:
        info = meta["artifacts"][name]
        path = destination / name
        require(sha256(path) == info["sha256"], f"checksum {name}")
        data = np.load(path, mmap_mode="r", allow_pickle=False)
        shape = ((2, 4) if name in PRIMITIVES else (2,)) + (req["z_count"], req["lattice_xyzt"][3])
        require(data.shape == shape and tuple(info["shape"]) == shape, f"shape {name}")
        require(data.dtype == np.dtype("complex128") and info["dtype"] == "complex128", f"dtype {name}")
        require(bool(np.isfinite(data).all()), f"NaN/Inf {name}")
        arrays[name] = data
    close = lambda a, b: bool(np.allclose(a, b, rtol=2e-13, atol=1e-10))
    for name in PRIMITIVES:
        data = arrays[name]
        require(close(data[:, 2], data[:, 0] + data[:, 1]), f"even {name}")
        require(close(data[:, 3], data[:, 0] - data[:, 1]), f"odd {name}")
        require(close(data[:, 3, 0], 0), f"z=0 {name}")
    for name, (channel, orientation, coefficient) in TARGETS.items():
        prefix, t_name, s_name = ("unpolarized", "TU", "SU") if channel == 0 else ("helicity", "TH", "SH")
        want = arrays[f"{prefix}/{t_name}.npy"][:, orientation] + coefficient * arrays[f"{prefix}/{s_name}.npy"][:, orientation]
        require(close(arrays[name], want), f"target algebra {name}")
    return meta


def export_product(base, req, destination=None):
    destination = Path(destination) if destination is not None else output_dir(req)
    if (destination / "complete.json").exists():
        validate_product(destination, req)
        print(f"RESUME_SKIP {destination}", flush=True)
        return
    data, raw_meta = validate_raw(base, req)
    arrays = arrays_from_raw(data)
    artifacts = {}
    for name, array in arrays.items():
        path = destination / name
        atomic_npy(path, array)
        artifacts[name] = {"shape": list(array.shape), "dtype": str(array.dtype),
                           "sha256": sha256(path), "bytes": path.stat().st_size}
    source_npy = Path(str(base) + ".npy")
    source_json = Path(str(base) + ".json")
    meta = {
        "schema": "gradient_flow_gluon_TU_SU_TH_SH_v1", "status": "flowed_bare_ope_observable",
        "identity": identity(req), "conventions": CONVENTIONS,
        "axes_primitives": ["field_projection", "z_orientation", "z", "t"],
        "axes_targets": ["field_projection", "z", "t"],
        "axis_labels": {"field_projection": PROJECTIONS, "z_orientation": ORIENTATIONS},
        "z_values_a": list(range(req["z_count"])), "t_values_a": list(range(req["lattice_xyzt"][3])),
        "flow_radius_fm": req["a_fm"] * math.sqrt(8 * req["tau"]),
        "flow_time_fm2": req["tau"] * req["a_fm"] ** 2,
        "nstep": round(req["tau"] / req["epsilon"]),
        "source": {"raw_npy": str(source_npy.resolve()), "raw_npy_sha256": sha256(source_npy),
                   "raw_json": str(source_json.resolve()), "raw_json_sha256": sha256(source_json),
                   "raw_schema": raw_meta["schema"],
                   "legacy_helicity_target_label_is_not_authoritative": True},
        "exporter": {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__)},
        "campaign_kernel_build": json.loads((ROOT / "build_provenance.json").read_text()) if (ROOT / "build_provenance.json").exists() else None,
        "historical_kernel_build": "not recorded by legacy raw schema" if REPO / "output_L32x64_v1" in source_npy.parents else None,
        "raw_gauge_stats": raw_meta["raw_gauge_stats"], "flowed_gauge_stats": raw_meta["flowed_gauge_stats"],
        "artifacts": artifacts,
        "not_applied": ["nucleon_C2C3", "vacuum_subtraction", "renormalization", "matching", "Wick_phase"],
    }
    atomic_json(destination / "metadata.json", meta)
    validate_product(destination, req, require_complete=False)
    atomic_json(destination / "complete.json", {"status": "validated_complete",
                "metadata_sha256": sha256(destination / "metadata.json")})
    print(f"COMPONENTS_VALIDATED {destination}", flush=True)
