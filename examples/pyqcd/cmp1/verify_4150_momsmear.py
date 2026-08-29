"""momentum-smear 4150 最终 2pt 输出级检查器的红绿测试。"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _literal_pplus() -> np.ndarray:
    """独立于被测实现的 DR 基 P+ 矩阵。"""
    return 0.5 * np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ],
        dtype=np.complex128,
    )


def _write_fixture(root: Path) -> tuple[np.ndarray, np.ndarray]:
    contract = np.zeros((4, 4, 4, 4), dtype=np.complex128)
    contract[1, 0, 0, 0] = 2.0 + 1.0j
    contract[0, 1, 1, 0] = -1.0 + 2.0j
    expected = np.einsum("li,yxil->yx", _literal_pplus(), contract)
    yy, xx = np.indices(expected.shape)
    expected = expected.copy()
    expected[yy < xx] *= -1.0
    np.save(
        root / "twopt_slice_pp_Px2Py0Pz0_eginphase2_Cg5g4_contract_conf4150.npy",
        contract,
    )
    np.save(
        root / "twopt_slice_pp_Px2Py0Pz0_eginphase2_Cg5g4_nopol_ss_conf4150.npy",
        expected,
    )
    np.save(
        root / "twopt_slice_pp_Px2Py0Pz0_eginphase2_Cg5g4_pol15_ss_conf4150.npy",
        expected * 0.25,
    )
    return contract, expected


def test_parse_momsmear_filename_preserves_direction_and_phase():
    from inspect_4150_momsmear import parse_output_filename

    parsed = parse_output_filename(
        "twopt_slice_pp_Px0Py0Pz2_eginphase-2_Cg5g4_contract_conf4150.npy"
    )
    assert parsed == {
        "momentum": [2, 0, 0],
        "phase": -2,
        "variant": "Cg5g4",
        "kind": "contract",
    }


def test_projected_nopol_uses_pplus_and_antiperiodic_boundary_sign():
    from inspect_4150_momsmear import compare_contract_nopol

    contract = np.zeros((4, 4, 4, 4), dtype=np.complex128)
    contract[1, 0, 0, 0] = 2.0 + 1.0j
    contract[0, 1, 1, 0] = -1.0 + 2.0j
    expected = np.einsum("li,yxil->yx", _literal_pplus(), contract)
    yy, xx = np.indices(expected.shape)
    expected = expected.copy()
    expected[yy < xx] *= -1.0
    result = compare_contract_nopol(contract, expected)
    assert result["status"] == "pass"
    assert result["max_abs"] == 0.0


def test_inspect_directory_groups_optional_polarizations_and_checks_projection():
    from inspect_4150_momsmear import inspect_directory

    with tempfile.TemporaryDirectory(prefix="pyqcd_momsmear_") as tmp:
        root = Path(tmp)
        _write_fixture(root)
        report = inspect_directory(root)
    assert report["file_count"] == 3
    assert report["group_count"] == 1
    group = report["groups"][0]
    assert group["momentum"] == [0, 0, 2]
    assert group["phase"] == 2
    assert group["files"]["contract"]["shape"] == [4, 4, 4, 4]
    assert group["files"]["nopol_ss"]["shape"] == [4, 4]
    assert group["files"]["pol15_ss"]["shape"] == [4, 4]
    assert group["projection"]["status"] == "pass"


def test_inspect_directory_rejects_malformed_or_foreign_files():
    from inspect_4150_momsmear import inspect_directory

    with tempfile.TemporaryDirectory(prefix="pyqcd_momsmear_") as tmp:
        root = Path(tmp)
        np.save(root / "unrelated.npy", np.zeros(2, dtype=np.float64))
        report = inspect_directory(root)
    assert report["file_count"] == 1
    assert report["unparsed_files"] == ["unrelated.npy"]


def run() -> None:
    tests = [
        test_parse_momsmear_filename_preserves_direction_and_phase,
        test_projected_nopol_uses_pplus_and_antiperiodic_boundary_sign,
        test_inspect_directory_groups_optional_polarizations_and_checks_projection,
        test_inspect_directory_rejects_malformed_or_foreign_files,
    ]
    for test in tests:
        test()
    print(f"verify_4150_momsmear: PASS {len(tests)}/{len(tests)}")


if __name__ == "__main__":
    run()
