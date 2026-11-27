"""`stab56` 默认路径的逐位黄金值（历史产物可复现锁）。

背景
----
`stab57` 给四个 renorm 入口加了可选参数（`hR_PDF(alpha_s)`、
`tmd_matching_hybrid(matching_order)`、`hR_x(tail_model)`、
`match_one(endpoint_order)`）。这些参数**默认不生效**，但"默认不生效"必须由
逐位断言证明，不能靠阅读代码推断：

- 浮点求和顺序、dtype 提升、metadata 键集合的任何变动都可能改变历史产物；
- 数据分析结果一旦被复制进报告，改动默认路径就是**不可追溯的**数值漂移。

本契约把 `stab56` 的输出冻结为黄金值，用 ``np.array_equal``（NaN 位置也须一致）
比较。若某次改动使本契约失败，只有两条路：撤回改动，或**显式**新增
`default_path_revision` 字段并重算全部下游产物——不允许"看起来差不多"就放行。

Golden 提取方式
---------------
从 `git show stab56:pyqcd/renorm/{_hybrid,_matching,_gradient_flow_quasi,
_tmdextract}.py` 取旧实现，注入同一 fixture 求值，`repr()` 全精度写回本文件。
fixture 与 `_production_branch_wiring_contract.py` 完全一致，便于交叉核对。
"""
from __future__ import annotations

import unittest

import numpy as np

from pyqcd.renorm import (
    hR_PDF,
    hR_x,
    match_one,
    tmd_matching_hybrid,
)

# ── fixture（与 _production_branch_wiring_contract 一致，改动需同步两边） ──
_Z = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
_HR_PZ = np.array([2.0, 1.8, 1.5, 1.2, 0.9, 0.7])
_HR_0 = np.array([1.0, 0.95, 0.85, 0.75, 0.65, 0.55])
_ZR = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
_HRX_KW = dict(
    Pz_=4, lambda_extra_=2.0, l1_fit_=1.0, a1_fit_=1.0, lambda0_fit_=0.5,
    zs=0.4, z_data=_Z, hR_pz=_HR_PZ, hR_0=_HR_0, zr_fit=_ZR, conf="L24x72")
_X3 = np.array([0.1, 0.3, 0.5])

_X21 = np.linspace(0.02, 1.48, 21)
_H21 = np.exp(-_X21 ** 2)

_QUASI = np.zeros((1, 2, 4, 6, 3, 2), dtype=np.complex128)
_QUASI[:, :, :, 1, :, :] = np.array([2.0, 4.0, 6.0])[None, None, None, :, None]
_QUASI[:, :, :, 2, :, :] = np.array([1.0, 2.0, 3.0])[None, None, None, :, None]
_QUASI[:, :, :, 0, :, :] = (
    _QUASI[:, :, :, 1, :, :] - _QUASI[:, :, :, 2, :, :])

_XTMD = np.array([0.2, 0.4, 0.6])
_YTMD = np.exp(-_XTMD)

# ── stab56 golden（全精度 repr，逐位） ──
_GOLDEN_HRX_COS = [1.233890973727623, 1.1696153479347091, 1.0471778388275814]
_GOLDEN_HRX_HERMITIAN = [
    1.233890973727623, 1.1696153479347091, 1.0471778388275814]
_GOLDEN_HRPDF_HEAD = [
    -71.6127867338472, 1.6558411668651098,
    1.0163101537400814, 0.7408808722169513]
_GOLDEN_HRPDF_TAIL = [
    0.2437473740509793, 0.1901309396384361, -8.08115839273387]
_GOLDEN_HRPDF_SUM = -70.29012128504898
_GOLDEN_MATCH_ONE_HEAD = [
    0.0, 0.0, 2.443639829555256, 2.443639829555256,
    4.529107332072024, 4.529107332072024]
_GOLDEN_MATCH_ONE_SUM = 447.5121621632456
_GOLDEN_TMD = [
    -62.07001162597276, 0.21353468480563387, -46.95981252924406]


class HRxGoldenContract(unittest.TestCase):
    def test_cos_transform_is_bitwise_frozen(self):
        actual = hR_x(_X3, **_HRX_KW)
        np.testing.assert_array_equal(actual, np.asarray(_GOLDEN_HRX_COS))

    def test_hermitian_transform_is_bitwise_frozen(self):
        actual = hR_x(_X3, transform='hermitian', **_HRX_KW)
        np.testing.assert_array_equal(
            actual, np.asarray(_GOLDEN_HRX_HERMITIAN))

    def test_tail_model_none_equals_the_bare_default(self):
        """显式 `tail_model='none'` 必须与不传该参数完全一致（同一路径）。"""
        np.testing.assert_array_equal(
            hR_x(_X3, tail_model='none', **_HRX_KW),
            np.asarray(_GOLDEN_HRX_COS))

    def test_metadata_flag_does_not_perturb_the_default_path(self):
        with_meta, metadata = hR_x(_X3, return_metadata=True, **_HRX_KW)
        np.testing.assert_array_equal(
            with_meta, np.asarray(_GOLDEN_HRX_COS))
        self.assertFalse(metadata["tail_applied"])
        self.assertEqual(metadata["tail_model"], "none")


class HRPDFGoldenContract(unittest.TestCase):
    def test_default_path_is_bitwise_frozen(self):
        actual = hR_PDF(
            _X21, Pz_=4, conf="L24x72", hR_tilde_data=_H21, mu_=2.0)
        np.testing.assert_array_equal(actual[:4], np.asarray(_GOLDEN_HRPDF_HEAD))
        np.testing.assert_array_equal(actual[-3:], np.asarray(_GOLDEN_HRPDF_TAIL))
        self.assertEqual(float(actual.sum()), _GOLDEN_HRPDF_SUM)

    def test_metadata_flag_does_not_perturb_the_default_path(self):
        with_meta, _ = hR_PDF(
            _X21, Pz_=4, conf="L24x72", hR_tilde_data=_H21, mu_=2.0,
            return_metadata=True)
        np.testing.assert_array_equal(
            with_meta[:4], np.asarray(_GOLDEN_HRPDF_HEAD))

    def test_explicit_coupling_reproduces_the_default_when_equal(self):
        """显式传入 `A_s_run(mu)*4π` 必须与缺省路径逐个比特相同。"""
        from pyqcd.renorm._matching import A_s_run

        coupling = A_s_run(2.0) * 4.0 * np.pi
        actual = hR_PDF(
            _X21, Pz_=4, conf="L24x72", hR_tilde_data=_H21, mu_=2.0,
            alpha_s=coupling)
        np.testing.assert_array_equal(
            actual[:4], np.asarray(_GOLDEN_HRPDF_HEAD))


class MatchOneGoldenContract(unittest.TestCase):
    def test_default_endpoint_order_is_bitwise_frozen(self):
        matched, _ = match_one(_QUASI.copy(), 0.5, 0.0775, 2.0, 0.25)
        head = np.nan_to_num(matched).ravel()[:6].real
        np.testing.assert_array_equal(
            head, np.asarray(_GOLDEN_MATCH_ONE_HEAD))
        self.assertEqual(float(np.nansum(matched.real)),
                         _GOLDEN_MATCH_ONE_SUM)

    def test_metadata_keys_are_a_superset_with_the_same_old_values(self):
        """新键只能是增量；既有键的值不得变。"""
        _, metadata = match_one(_QUASI.copy(), 0.5, 0.0775, 2.0, 0.25)
        self.assertEqual(metadata["scheme_conversion"],
                         "one_loop_endpoint_applied_without_kappa_F")
        details = metadata["scheme_conversion_details"]
        self.assertEqual(details["order"], "one_loop")
        self.assertEqual(details["endpoint_order"], 1)
        self.assertEqual(details["endpoint_form"], "resummed_one_loop")
        self.assertEqual(details["light_cone_matching"], "not_applied")
        self.assertEqual(metadata["z0_status"], "nan_fail_closed")
        self.assertIs(metadata["physical_acceptance"], False)


class TmdHybridGoldenContract(unittest.TestCase):
    def test_default_path_is_bitwise_frozen(self):
        actual = tmd_matching_hybrid(
            _XTMD, cs_kernel=0.0, soft_factor=1.0, x_tmd=_YTMD)[1]
        np.testing.assert_array_equal(actual, np.asarray(_GOLDEN_TMD))

    def test_matching_order_one_is_the_default(self):
        explicit = tmd_matching_hybrid(
            _XTMD, cs_kernel=0.0, soft_factor=1.0, x_tmd=_YTMD,
            matching_order=1)[1]
        np.testing.assert_array_equal(explicit, np.asarray(_GOLDEN_TMD))

    def test_metadata_additions_are_purely_additive(self):
        _, _, metadata = tmd_matching_hybrid(
            _XTMD, cs_kernel=0.0, soft_factor=1.0, x_tmd=_YTMD,
            return_metadata=True)
        # 旧键保留
        self.assertEqual(
            metadata["status"], "one_loop_hybrid_matching_diagnostic")
        self.assertEqual(metadata["light_cone_matching"],
                         "one_loop_gg_only_diagnostic")
        self.assertEqual(metadata["soft_factor_definition"],
                         "S_I(b_perp), consumed as sqrt(S_I)")
        self.assertEqual(metadata["proxy_terms"],
                         ["cs_kernel_zero", "soft_factor_one"])
        # 新键为增量
        self.assertEqual(metadata["matching_order"], 1)
        self.assertEqual(metadata["two_loop_kernel"], "unavailable")


class OptionalPathsAreOptInContract(unittest.TestCase):
    def test_tail_models_change_the_result_only_when_requested(self):
        base = hR_x(_X3, **_HRX_KW)
        for tail_model, extra in (
                ('power_exp', {}),
                ('la_nla', {"tail_params": (1.0, 0.2, 0.5)})):
            with self.subTest(tail_model=tail_model):
                tailed = hR_x(_X3, tail_model=tail_model,
                              **extra, **_HRX_KW)
                self.assertFalse(np.array_equal(base, tailed))

    def test_endpoint_order_two_changes_the_result_only_when_requested(self):
        first, _ = match_one(_QUASI.copy(), 0.5, 0.0775, 2.0, 0.25)
        second, _ = match_one(
            _QUASI.copy(), 0.5, 0.0775, 2.0, 0.25, endpoint_order=2)
        finite = np.isfinite(first) & np.isfinite(second)
        self.assertTrue(finite.any())
        self.assertFalse(np.array_equal(first[finite], second[finite]))


def test_renorm_default_golden_contracts():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for case in (HRxGoldenContract, HRPDFGoldenContract,
                 MatchOneGoldenContract, TmdHybridGoldenContract,
                 OptionalPathsAreOptInContract):
        suite.addTests(loader.loadTestsFromTestCase(case))
    result = unittest.TextTestRunner(verbosity=0).run(suite)
    if not result.wasSuccessful():
        raise AssertionError("renorm default-path golden contracts failed")


if __name__ == "__main__":
    test_renorm_default_golden_contracts()
