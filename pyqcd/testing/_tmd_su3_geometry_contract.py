"""TMD bilocal 的 su(3) 去迹与空间几何行为契约。"""
from __future__ import annotations

import unittest

import numpy as np

from pyqcd.operator import plaquette_clover, staple_operator
from pyqcd.renorm import (
    gluon_tmd_operator,
    tmd_matrix_elements,
    tmd_matrix_elements_time,
)
from pyqcd.renorm._tmd import _matrix_element_from_fields
from pyqcd.testing import _gauge_transform, random_su3_gauge
from pyqcd.tools import set_backend


ERRORS: dict[str, float] = {}


def _manual_traceless(field):
    """测试侧手算 F -> F - Tr(F) I / Nc。"""
    nc = field.shape[-1]
    trace = np.trace(field, axis1=-2, axis2=-1)
    identity = np.eye(nc, dtype=field.dtype)
    return field - trace[..., None, None] * identity / nc


def _manual_shift(field, z, b_perp, z_dir, b_dir):
    endpoint = [0, 0, 0]
    endpoint[z_dir] += z
    endpoint[b_dir] += b_perp
    shifted = field
    for direction, offset in enumerate(endpoint):
        if offset:
            shifted = np.roll(shifted, -offset, axis=3 - direction)
    return shifted


def _manual_bilocal(left, right, wilson, z, b_perp, z_dir, b_dir):
    """手写 Tr[right(x) W left(y) W^dagger]。"""
    shifted = _manual_shift(left, z, b_perp, z_dir, b_dir)
    closed = right @ wilson @ shifted @ wilson.conj().swapaxes(-1, -2)
    return np.trace(closed, axis1=-2, axis2=-1)


def _manual_staple(gauge, z, b_perp, z_dir, b_dir, L):
    """逐点手写 -L*zdir -> b*bdir -> (L+z)*zdir 三段路径。"""
    lattice_shape = gauge.shape[:4]
    nc = gauge.shape[-1]
    out = np.empty(lattice_shape + (nc, nc), dtype=gauge.dtype)
    path = ((z_dir, -L), (b_dir, b_perp), (z_dir, L + z))

    for base in np.ndindex(*lattice_shape):
        pos = list(base)
        transporter = np.eye(nc, dtype=gauge.dtype)
        for direction, signed_length in path:
            axis = 3 - direction
            step = 1 if signed_length >= 0 else -1
            for _ in range(abs(signed_length)):
                if step > 0:
                    link = gauge[tuple(pos)][direction]
                    pos[axis] = (pos[axis] + 1) % lattice_shape[axis]
                else:
                    pos[axis] = (pos[axis] - 1) % lattice_shape[axis]
                    link = gauge[tuple(pos)][direction].conj().T
                transporter = transporter @ link
        out[base] = transporter
    return out


def _constant_link_power(link, length):
    """Return ``U^length`` for a constant link and signed integer length."""
    if length >= 0:
        return np.linalg.matrix_power(link, length)
    return np.linalg.matrix_power(link.conj().T, -length)


def _constant_link_staple(links, z, b_perp, z_dir, b_dir, L):
    """Closed-form constant-link staple, independent of lattice coordinates."""
    return (
        _constant_link_power(links[z_dir], -L)
        @ _constant_link_power(links[b_dir], b_perp)
        @ _constant_link_power(links[z_dir], L + z)
    )


def _relabel_spatial_axes(gauge, new_to_old):
    """同时重标记空间坐标轴与链接方向；new_to_old[new]=old。"""
    q = tuple(new_to_old)
    coordinate_axes = (0, 3 - q[2], 3 - q[1], 3 - q[0], 4, 5, 6)
    relabeled = np.transpose(gauge, coordinate_axes)
    return relabeled[..., list(q) + [3], :, :]


def _relabel_scalar_field(field, new_to_old):
    q = tuple(new_to_old)
    return np.transpose(field, (0, 3 - q[2], 3 - q[1], 3 - q[0]))


class TmdSu3GeometryContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_backend("numpy")

    def tearDown(self):
        set_backend("numpy")

    def test_batch_default_uses_one_fixed_staple_length(self):
        """按 z 改变 staple 臂长会混合不同 rapidity regulator。"""
        gauge = random_su3_gauge(L=3, seed=7300)
        z_list = [0, 1, 2]
        b_list = [1]

        actual = tmd_matrix_elements_time(
            gauge, z_list, b_list, z_dir=2, b_dir=0)
        expected = tmd_matrix_elements_time(
            gauge, z_list, b_list, z_dir=2, b_dir=0, L=2)
        legacy = np.concatenate([
            tmd_matrix_elements_time(
                gauge, [z], b_list, z_dir=2, b_dir=0, L=abs(z))
            for z in z_list
        ], axis=0)

        fixed_error = float(np.max(np.abs(actual - expected)))
        legacy_sensitivity = float(np.max(np.abs(expected - legacy)))
        ERRORS["batch_fixed_staple"] = fixed_error
        ERRORS["batch_legacy_staple_sensitivity"] = legacy_sensitivity
        self.assertGreater(legacy_sensitivity, 1e-6)
        self.assertLess(fixed_error, 1e-10)

    def test_constant_link_path_matches_closed_form_signed_oracle(self):
        """常量非交换链接必须逐项复现闭式三段矩阵乘积。"""
        from pyqcd.renorm import staple_wilson_line

        links = random_su3_gauge(L=1, seed=7307)[0, 0, 0, 0]
        gauge = np.broadcast_to(
            links.reshape((1, 1, 1, 1) + links.shape),
            (1, 2, 2, 2) + links.shape).copy()
        directions = (
            (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1),
        )
        commutator_sensitivity = max(
            float(np.linalg.norm(links[first] @ links[second]
                                 - links[second] @ links[first]))
            for first, second in directions
        )
        max_error = 0.0
        for z_dir, b_dir in directions:
            for z in (-2, -1, 0, 1, 2):
                for b_perp in (-2, -1, 0, 1, 2):
                    for L in (0, 1, 2):
                        with self.subTest(
                                z_dir=z_dir, b_dir=b_dir, z=z,
                                b_perp=b_perp, L=L):
                            expected = _constant_link_staple(
                                links, z, b_perp, z_dir, b_dir, L)
                            actual = np.asarray(staple_wilson_line(
                                gauge, z=z, b_perp=b_perp,
                                z_dir=z_dir, b_dir=b_dir, L=L))
                            error = float(np.max(np.abs(
                                actual - np.broadcast_to(expected, actual.shape))))
                            max_error = max(max_error, error)
                            self.assertLess(error, 2e-12)

        ERRORS["constant_link_signed_path"] = max_error
        ERRORS["constant_link_commutator_sensitivity"] = (
            commutator_sensitivity)
        self.assertGreater(commutator_sensitivity, 1e-3)

    def test_zero_transverse_separation_is_independent_of_staple_length(self):
        """b=0 时同向路径必须全消，固定 L 不得改变 transporter。"""
        from pyqcd.renorm import staple_wilson_line

        gauge = random_su3_gauge(L=3, seed=7308)
        directions = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
        max_error = 0.0
        for z_dir, b_dir in directions:
            for z in (-2, -1, 0, 1, 2):
                baseline = np.asarray(staple_wilson_line(
                    gauge, z=z, b_perp=0, z_dir=z_dir,
                    b_dir=b_dir, L=0))
                for L in (1, 2, 3, 4):
                    with self.subTest(
                            z_dir=z_dir, b_dir=b_dir, z=z, L=L):
                        actual = np.asarray(staple_wilson_line(
                            gauge, z=z, b_perp=0, z_dir=z_dir,
                            b_dir=b_dir, L=L))
                        error = float(np.max(np.abs(actual - baseline)))
                        max_error = max(max_error, error)
                        self.assertLess(error, 2e-12)

        ERRORS["zero_b_staple_length_invariance"] = max_error

    def test_pure_gauge_has_zero_finite_b_tmd_matrix(self):
        """纯规范场 Clover=0，故有限 b TMD 均值与逐时间片矩阵均应为零。"""
        shape = (2, 2, 2, 2)
        identity = np.broadcast_to(
            np.eye(3, dtype=np.complex128), shape + (4, 3, 3)).copy()
        site_transform = random_su3_gauge(L=2, seed=7309)[..., 0, :, :]
        gauge = _gauge_transform(identity, site_transform)
        field = np.asarray(plaquette_clover(gauge, 3, 1))
        averaged = tmd_matrix_elements(
            gauge, [0, 1, 2], [0, 1, 2],
            z_dir=2, b_dir=0, L=2)
        timed = tmd_matrix_elements_time(
            gauge, [0, 1, 2], [0, 1, 2],
            z_dir=2, b_dir=0, L=2)

        field_error = float(np.max(np.abs(field)))
        average_error = float(np.max(np.abs(averaged)))
        time_error = float(np.max(np.abs(timed)))
        ERRORS["pure_gauge_clover"] = field_error
        ERRORS["pure_gauge_finite_b_average"] = average_error
        ERRORS["pure_gauge_finite_b_time"] = time_error
        self.assertEqual(averaged.shape, (3, 3))
        self.assertEqual(timed.shape, (3, 3, shape[0]))
        self.assertLess(field_error, 2e-12)
        self.assertLess(average_error, 2e-12)
        self.assertLess(time_error, 2e-12)

    def test_full_channel_tmd_ope_reduces_to_straight_schema(self):
        """b=0,L=0 必须逐项复现 straight schema-v2 全通道 OPE。"""
        from pyqcd.operator import flowed_gluon_ope
        from pyqcd.renorm import full_channel_tmd_ope_time

        gauge = random_su3_gauge(L=3, seed=7310)
        z_list = [0, 1, 2]
        directions = (
            (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1),
        )
        max_error = 0.0
        for z_dir, b_dir in directions:
            with self.subTest(z_dir=z_dir, b_dir=b_dir):
                actual = full_channel_tmd_ope_time(
                    gauge, z_list, [0], z_dir=z_dir, b_dir=b_dir, L=0)
                expected = flowed_gluon_ope(
                    gauge, z_count=len(z_list), z_dir=z_dir)[1]

                self.assertEqual(
                    actual.shape, (2, 4, 6, 3, 1, gauge.shape[0]))
                self.assertTrue(np.iscomplexobj(actual))
                np.testing.assert_allclose(
                    actual[..., 0, :], expected,
                    rtol=1e-12, atol=1e-10)
                max_error = max(max_error, float(np.max(np.abs(
                    actual[..., 0, :] - expected))))
        ERRORS['full_channel_straight_reduction'] = max_error

    def test_full_channel_tmd_ope_finite_b_identities(self):
        """有限 b 输出必须保持 schema 线性恒等式并兼容现有非极化入口。"""
        from pyqcd.renorm import full_channel_tmd_ope_time

        gauge = random_su3_gauge(L=3, seed=7311)
        z_list = [0, 1]
        b_list = [0, 1]
        actual = full_channel_tmd_ope_time(
            gauge, z_list, b_list, z_dir=2, b_dir=0, L=1)
        self.assertEqual(
            actual.shape, (2, 4, 6, len(z_list), len(b_list), gauge.shape[0]))
        self.assertEqual(actual.dtype, np.dtype(np.complex128))
        self.assertTrue(np.isfinite(actual).all())

        plus = actual[:, 0]
        minus = actual[:, 1]
        np.testing.assert_allclose(
            actual[:, 2], plus + minus, rtol=0, atol=2e-12)
        np.testing.assert_allclose(
            actual[:, 3], plus - minus, rtol=0, atol=2e-12)
        np.testing.assert_allclose(
            actual[0, :, 0], actual[0, :, 1] - actual[0, :, 2],
            rtol=0, atol=2e-12)
        np.testing.assert_allclose(
            actual[1, :, 0], actual[1, :, 1] + actual[1, :, 2],
            rtol=0, atol=2e-12)
        np.testing.assert_allclose(
            actual[:, 3, :, 0, :, :], 0.0, rtol=0, atol=2e-12)

        legacy = tmd_matrix_elements_time(
            gauge, z_list, b_list, z_dir=2, b_dir=0, L=1)
        np.testing.assert_allclose(
            actual[0, 0, 0], legacy,
            rtol=0, atol=2e-10)
        self.assertGreater(
            float(np.max(np.abs(actual[0, :, :, 1, 1]
                                 - actual[0, :, :, 1, 0]))),
            1e-8)
        single64 = full_channel_tmd_ope_time(
            gauge.astype(np.complex64), [0], [0], L=0)
        self.assertEqual(single64.dtype, np.dtype(np.complex64))
        self.assertTrue(np.isfinite(single64).all())

    def test_two_dimensional_staple_closed_form_and_1d_limit(self):
        from pyqcd.renorm import staple_wilson_line, staple_wilson_line_2d

        links = random_su3_gauge(L=1, seed=7312)[0, 0, 0, 0]
        gauge = np.broadcast_to(
            links.reshape((1, 1, 1, 1) + links.shape),
            (1, 2, 2, 2) + links.shape).copy()
        actual = staple_wilson_line_2d(
            gauge, z=1, b_x=1, b_y=2, z_dir=2,
            b_x_dir=0, b_y_dir=1, L=1)
        expected = (
            _constant_link_power(links[2], -1)
            @ _constant_link_power(links[0], 1)
            @ _constant_link_power(links[1], 2)
            @ _constant_link_power(links[2], 2)
        )
        np.testing.assert_allclose(
            actual, np.broadcast_to(expected, actual.shape),
            rtol=0, atol=2e-12)

        one_dimensional = staple_wilson_line(
            gauge, z=1, b_perp=1, z_dir=2, b_dir=0, L=1)
        np.testing.assert_allclose(
            staple_wilson_line_2d(
                gauge, z=1, b_x=1, b_y=0, z_dir=2,
                b_x_dir=0, b_y_dir=1, L=1),
            one_dimensional,
            rtol=0,
            atol=2e-12,
        )
        reversed_path = staple_wilson_line_2d(
            gauge, z=1, b_x=1, b_y=2, z_dir=2,
            b_x_dir=0, b_y_dir=1, L=1,
            transverse_order=(1, 0))
        self.assertGreater(float(np.max(np.abs(reversed_path - actual))), 1e-3)

    def test_two_dimensional_staple_is_gauge_covariant(self):
        from pyqcd.renorm import staple_wilson_line_2d

        gauge = random_su3_gauge(L=3, seed=7314)
        site_transform = random_su3_gauge(L=3, seed=7414)[..., 0, :, :]
        transformed = _gauge_transform(gauge, site_transform)
        kwargs = dict(
            z=1, b_x=1, b_y=2, z_dir=2,
            b_x_dir=0, b_y_dir=1, L=1)
        actual = staple_wilson_line_2d(gauge, **kwargs)
        rotated = staple_wilson_line_2d(transformed, **kwargs)
        endpoint = site_transform
        endpoint = np.roll(endpoint, -1, axis=3 - 2)
        endpoint = np.roll(endpoint, -1, axis=3 - 0)
        endpoint = np.roll(endpoint, -2, axis=3 - 1)
        expected = (
            site_transform @ actual
            @ endpoint.conj().swapaxes(-1, -2))
        np.testing.assert_allclose(rotated, expected, rtol=0, atol=2e-11)

    def test_two_dimensional_full_channel_tmd_ope_reduces_to_1d(self):
        from pyqcd.renorm import (
            full_channel_tmd_ope_time,
            full_channel_tmd_ope_time_2d,
        )

        gauge = random_su3_gauge(L=3, seed=7313)
        z_list = [0, 1]
        b_list = [0, 1]
        actual_2d = full_channel_tmd_ope_time_2d(
            gauge, z_list, b_list, [0],
            z_dir=2, b_x_dir=0, b_y_dir=1, L=1)
        expected_1d = full_channel_tmd_ope_time(
            gauge, z_list, b_list,
            z_dir=2, b_dir=0, L=1)
        self.assertEqual(
            actual_2d.shape, (2, 4, 6, 2, 2, 1, gauge.shape[0]))
        np.testing.assert_allclose(
            actual_2d[..., 0, :], expected_1d, rtol=0, atol=2e-12)
        np.testing.assert_allclose(
            actual_2d[:, 3, :, 0, :, :, :], 0.0, rtol=0, atol=2e-12)
        single64 = full_channel_tmd_ope_time_2d(
            gauge.astype(np.complex64), [0], [0], [0], L=0)
        self.assertEqual(single64.dtype, np.dtype(np.complex64))

    def test_batch_no_spatial_sum_preserves_local_field_shape(self):
        """spatial_sum=False 不得把逐格点张量静默压成单个均值。"""
        gauge = random_su3_gauge(L=2, seed=7306)
        actual = tmd_matrix_elements(
            gauge, [1], [1], z_dir=2, b_dir=0, L=1,
            spatial_sum=False)
        expected = np.asarray(gluon_tmd_operator(
            gauge, z=1, b_perp=1, z_dir=2, b_dir=0, L=1))

        self.assertEqual(actual.shape, (1, 1) + gauge.shape[:4])
        np.testing.assert_allclose(actual[0, 0], expected, rtol=0, atol=2e-12)

    def test_adjoint_color_normalization_is_twice_fundamental_trace(self):
        """标准生成元归一化下 F^a W_adj^ab F^b = 2 Tr[F W F W†]。"""
        field = np.zeros((1, 1, 1, 1, 3, 3), dtype=np.complex128)
        field[..., 0, 0] = 0.5
        field[..., 1, 1] = -0.5
        wilson = np.broadcast_to(
            np.eye(3, dtype=np.complex128), field.shape).copy()

        try:
            fundamental = np.asarray(_matrix_element_from_fields(
                field, field, wilson, z=0, b_perp=0, z_dir=2, b_dir=0,
                color_normalization='fundamental_trace'))
            adjoint = np.asarray(_matrix_element_from_fields(
                field, field, wilson, z=0, b_perp=0, z_dir=2, b_dir=0,
                color_normalization='adjoint'))
        except TypeError as exc:
            self.fail(f'缺少显式颜色归一化接口: {exc}')

        np.testing.assert_allclose(fundamental, 0.5, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(adjoint, 1.0, rtol=0.0, atol=0.0)

    def test_identity_contamination_drops_out_of_bilocal(self):
        """F+c(x)I 不得改变逐点 su(3) bilocal，且保留 complex64。"""
        rng = np.random.default_rng(7301)
        shape = (1, 2, 2, 2, 3, 3)
        left = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(
            np.complex64)
        right = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(
            np.complex64)
        left = _manual_traceless(left)
        right = _manual_traceless(right)
        wilson = random_su3_gauge(L=2, seed=7302)[..., 0, :, :].astype(
            np.complex64)
        identity = np.eye(3, dtype=np.complex64)
        c_left = (rng.normal(size=shape[:4])
                  + 1j * rng.normal(size=shape[:4])).astype(np.complex64)
        c_right = (rng.normal(size=shape[:4])
                   + 1j * rng.normal(size=shape[:4])).astype(np.complex64)

        expected = _manual_bilocal(
            left, right, wilson, z=1, b_perp=1, z_dir=2, b_dir=0)
        clean = np.asarray(_matrix_element_from_fields(
            left, right, wilson, z=1, b_perp=1, z_dir=2, b_dir=0))
        polluted = np.asarray(_matrix_element_from_fields(
            left + c_left[..., None, None] * identity,
            right + c_right[..., None, None] * identity,
            wilson, z=1, b_perp=1, z_dir=2, b_dir=0))

        clean_error = float(np.max(np.abs(clean - expected)))
        pollution_error = float(np.max(np.abs(polluted - expected)))
        ERRORS["identity_clean"] = clean_error
        ERRORS["identity_pollution"] = pollution_error
        self.assertEqual(clean.dtype, np.dtype(np.complex64))
        self.assertEqual(polluted.dtype, np.dtype(np.complex64))
        self.assertLess(clean_error, 2e-5)
        self.assertLess(pollution_error, 2e-5)

    def test_public_staple_uses_traceless_clover(self):
        """operator.staple_operator 必须消费同一逐点去迹 bilocal。"""
        gauge = random_su3_gauge(L=2, seed=7303).astype(np.complex64)
        field_raw = np.asarray(plaquette_clover(gauge, 3, 1))
        field = _manual_traceless(field_raw)
        wilson = _manual_staple(
            gauge, z=1, b_perp=1, z_dir=2, b_dir=0, L=1)
        expected = _manual_bilocal(
            field, field, wilson, z=1, b_perp=1, z_dir=2, b_dir=0)
        raw_result = _manual_bilocal(
            field_raw, field_raw, wilson, z=1, b_perp=1,
            z_dir=2, b_dir=0)
        actual = np.asarray(staple_operator(
            gauge, 3, 1, z=1, b_perp=1, z_dir=2, b_dir=0))

        error = float(np.max(np.abs(actual - expected)))
        fixture_sensitivity = float(np.max(np.abs(raw_result - expected)))
        ERRORS["public_staple_traceless"] = error
        ERRORS["public_staple_trace_sensitivity"] = fixture_sensitivity
        self.assertEqual(actual.dtype, gauge.dtype)
        self.assertGreater(fixture_sensitivity, 1e-5)
        self.assertLess(error, 2e-5)

    def test_spatial_axis_relabeling_is_covariant_for_all_z_dirs(self):
        """空间轴与方向标签同步重标记后，局域标量及批量和必须协变。"""
        gauge = random_su3_gauge(L=2, seed=7304)
        base = np.asarray(gluon_tmd_operator(
            gauge, z=1, b_perp=1, z_dir=2, b_dir=0, L=1))
        base_time = tmd_matrix_elements_time(
            gauge, [1], [1], z_dir=2, b_dir=0, L=1)[0, 0]

        permutations = (
            (0, 1, 2),  # z_dir=2
            (2, 0, 1),  # z_dir=0
            (1, 2, 0),  # z_dir=1
        )
        for q in permutations:
            with self.subTest(new_to_old=q):
                relabeled = _relabel_spatial_axes(gauge, q)
                z_dir = q.index(2)
                b_dir = q.index(0)
                actual = np.asarray(gluon_tmd_operator(
                    relabeled, z=1, b_perp=1,
                    z_dir=z_dir, b_dir=b_dir, L=1))
                expected = _relabel_scalar_field(base, q)
                local_error = float(np.max(np.abs(actual - expected)))

                actual_time = tmd_matrix_elements_time(
                    relabeled, [1], [1], z_dir=z_dir,
                    b_dir=b_dir, L=1)[0, 0]
                batch_error = float(np.max(np.abs(actual_time - base_time)))
                ERRORS[f"axis_relabel_local_z{z_dir}"] = local_error
                ERRORS[f"axis_relabel_batch_z{z_dir}"] = batch_error
                self.assertLess(local_error, 1e-10)
                self.assertLess(batch_error, 1e-10)

    def test_invalid_boolean_and_conflicting_directions_are_rejected(self):
        """方向必须是非布尔空间整数，且 z_dir 与 b_dir 不得相同。"""
        from pyqcd.renorm import full_channel_tmd_ope_time
        from pyqcd.renorm import (
            full_channel_tmd_ope_time_2d, staple_wilson_line_2d,
        )

        gauge = random_su3_gauge(L=2, seed=7305)
        cases = (
            ("gluon_bool_z", lambda: gluon_tmd_operator(
                gauge, 0, 0, z_dir=True, b_dir=0, L=0)),
            ("batch_bool_b", lambda: tmd_matrix_elements(
                gauge, [], [0], z_dir=2, b_dir=False, L=0)),
            ("batch_time_bool_z", lambda: tmd_matrix_elements_time(
                gauge, [], [0], z_dir=np.bool_(False), b_dir=1, L=0)),
            ("operator_float", lambda: staple_operator(
                gauge, 3, 0, 0, 0, z_dir=2.0, b_dir=0)),
            ("invalid_label", lambda: gluon_tmd_operator(
                gauge, 0, 0, z_dir=3, b_dir=0, L=0)),
            ("conflict", lambda: staple_operator(
                gauge, 3, 0, 0, 0, z_dir=1, b_dir=1)),
            ("full_bool_z", lambda: full_channel_tmd_ope_time(
                gauge, [True], [0], L=0)),
            ("full_negative_z", lambda: full_channel_tmd_ope_time(
                gauge, [-1], [0], L=0)),
            ("full_duplicate_b", lambda: full_channel_tmd_ope_time(
                gauge, [0], [1, 1], L=0)),
            ("full_negative_L", lambda: full_channel_tmd_ope_time(
                gauge, [0], [0], L=-1)),
            ("2d_conflict", lambda: staple_wilson_line_2d(
                gauge, 0, 0, 0, z_dir=2, b_x_dir=0, b_y_dir=0)),
            ("2d_bad_order", lambda: full_channel_tmd_ope_time_2d(
                gauge, [0], [0], [0], transverse_order=(0, 0))),
            ("2d_bool_bx", lambda: full_channel_tmd_ope_time_2d(
                gauge, [0], [True], [0])),
        )
        for name, call in cases:
            with self.subTest(case=name):
                try:
                    call()
                except ValueError:
                    continue
                except Exception as exc:
                    self.fail(
                        f"应抛 ValueError，实际 {type(exc).__name__}: {exc}")
                else:
                    self.fail("应抛 ValueError，实际未拒绝")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TmdSu3GeometryContract)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print("numeric errors:")
    for name, value in sorted(ERRORS.items()):
        print(f"  {name}: {value:.3e}")
    raise SystemExit(0 if result.wasSuccessful() else 1)
