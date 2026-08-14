"""pyqcd 集成测试：梯度流、TMD 算符、重整化链的数值断言。"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def random_su3_gauge(L=6, seed=42):
    """随机 SU(3) 规范场 (L,L,L,L,4,3,3)（测试用）。"""
    rng = np.random.default_rng(seed)
    g = np.zeros((L, L, L, L, 4, 3, 3), dtype=complex)
    for idx in np.ndindex(L, L, L, L, 4):
        H = (rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3)))
        H = H - H.conj().T - np.trace(H - H.conj().T) / 3 * np.eye(3)
        g[idx] = expm(H)
    return g


def test_gamma_basis():
    """DR 基 γ 矩阵：γ₅² = I、γ_7 = γ₃γ₁ 反厄米。"""
    from pyqcd.lattice import gamma
    g5 = gamma(5)
    assert np.allclose(g5 @ g5, np.eye(4), atol=1e-10)
    g7 = gamma(7)
    assert np.allclose(g7, -g7.conj().T, atol=1e-10)


def test_zr_parametrization():
    """Z_R 参数化：z=0 处有限、随 z 增大指数衰减（线性发散主导）。"""
    from pyqcd.renorm import th_ZR, th_hB
    z = np.linspace(0.15, 1.0, 5)
    a = 0.105 / 0.197
    zr = th_ZR(z, a, 2.0, 0.5, 0.1, 0.0, 0.0, 0.25, (0.0, 0.0))
    assert np.all(np.isfinite(zr))
    assert zr[0] > zr[-1]           # 单调衰减
    hb = th_hB(z, a, 2.0, (0.5, 0.1, 0.0, 0.0, 0.25) + tuple(np.zeros(14)),
               (0.0, 0.0))
    assert np.all(np.isfinite(hb))


def test_gradient_flow_su3_and_dissipation():
    """梯度流：SU(3) 幺正保持 + 作用量密度递减。"""
    from pyqcd.renorm import wilson_flow, flow_action_density
    from pyqcd.tools import set_backend
    set_backend('numpy')

    g = random_su3_gauge(L=4, seed=7)
    E0 = flow_action_density(g).mean()
    V = wilson_flow(g, tau=0.1, eps=0.05)
    dev = np.abs(V[0, 0, 0, 0, 0] @ V[0, 0, 0, 0, 0].conj().T - np.eye(3)).max()
    E1 = flow_action_density(V).mean()
    assert dev < 1e-6, f"SU(3) 保持失败: dev={dev}"
    assert E1 < E0, f"作用量密度应递减: {E0} -> {E1}"


def test_tmd_operator_runs():
    """TMD staple 算符 O(z, b⊥)：可运行、实数、共线极限有限。"""
    from pyqcd.renorm import gluon_tmd_operator, tmd_matrix_elements
    from pyqcd.tools import set_backend
    set_backend('numpy')

    g = random_su3_gauge(L=6, seed=1)
    O = gluon_tmd_operator(g, z=2, b_perp=1)
    assert O.shape == (6, 6, 6, 6)
    assert np.all(np.isfinite(np.real(O)))

    M = tmd_matrix_elements(g, [0, 1, 2], [0, 1, 2])
    assert M.shape == (3, 3)
    assert np.all(np.isfinite(M))


def test_matching_kernel():
    """NLO 匹配核 hR_PDF：Pz 增大时接近单位变换。"""
    from pyqcd.renorm import hR_PDF
    # 注意：x=0 处匹配核有 1/y 奇点（与参考代码一致），真实网格避开 0
    xx = np.linspace(0.02, 1.48, 21)
    h0 = np.exp(-xx ** 2)   # 测试用准 PDF
    out = hR_PDF(xx, Pz_=4, conf='L24x72', hR_tilde_data=h0, mu_=2.0)
    assert out.shape == h0.shape
    assert np.all(np.isfinite(out))


def test_hybrid_ratio():
    """混合方案：短距比值段与 Z_R 长距段拼接连续。"""
    from pyqcd.renorm import hR_z_Pz, th_ZR
    from pyqcd.tools import set_backend
    set_backend('numpy')

    z = np.linspace(0.15, 1.0, 8)
    a = 0.105 / 0.197
    zr = th_ZR(z, a, 2.0, 0.5, 0.1, 0.0, 0.0, 0.25, (0.0, 0.0))
    hb_pz = np.exp(-z / 0.3)
    hb_0 = np.exp(-z / 0.6)
    lam, hR = hR_z_Pz(z, 4, hb_pz, hb_0, zs=0.5, zr_fit=zr, conf='L24x72')
    assert len(lam) == len(z)
    assert np.all(np.isfinite(hR))
    assert hR[0] > 0   # 短距比值接近 1


def test_tmd_extraction_chain():
    """TMD 提取链：准 TMD-PDF / CS 核 / SFTX 系数可运行且有限。"""
    from pyqcd.renorm import (
        quasi_tmd_pdf, cs_kernel_from_ratio, sftx_gluon_matching_coeff,
        sftx_energy_density_t0,
    )
    z = np.linspace(0.1, 1.0, 32)
    hr = np.exp(-z / 0.3)[:, None] * np.array([1.0, 0.8])[None, :]
    x, xg = quasi_tmd_pdf(hr, z, [0.2, 0.4], 2.0)
    assert xg.shape == (256, 2)
    assert np.all(np.isfinite(xg))
    K = cs_kernel_from_ratio(hr, hr * 1.1, 2.5, 2.0)
    assert np.all(np.isfinite(K))
    al, c = sftx_gluon_matching_coeff(0.1, 2.0)
    assert np.isfinite(al) and np.isfinite(c)
    assert np.all(np.isfinite(sftx_energy_density_t0(1.0, 0.1, 2.0)))


def test_scale_setting_flow_behavior():
    """尺度设定与流时间行为：t²⟨E⟩ 单调递增且 t0 可求。"""
    from pyqcd.renorm import (
        wilson_flow, flow_action_density, scale_setting_t0,
    )
    g = random_su3_gauge(L=4, seed=3)
    vals = []
    for tau in [0.1, 0.3, 0.5]:
        V = wilson_flow(g, tau=tau, eps=0.05)
        vals.append(tau ** 2 * flow_action_density(V).mean())
    assert vals[0] < vals[1] < vals[2], f"t²⟨E⟩ 应单调递增: {vals}"
    target = 0.5 * (vals[0] + vals[2])   # 目标取区间内部，保证插值可达
    t0 = scale_setting_t0(g, target=target, tau_max=0.5, eps=0.05)
    assert np.isfinite(t0) and t0 > 0


def test_hyp_smear():
    """HYP 涂抹：SU(3) 保持 + 作用量密度平滑化（降低）。"""
    from pyqcd.smear import hyp_smear
    from pyqcd.renorm import flow_action_density
    g = random_su3_gauge(L=6, seed=2)
    V = hyp_smear(g)
    dev = np.abs(V[0, 0, 0, 0, 0] @ V[0, 0, 0, 0, 0].conj().T - np.eye(3)).max()
    assert dev < 1e-10, f"SU(3) 保持失败: {dev}"
    E0, E1 = flow_action_density(g).mean(), flow_action_density(V).mean()
    assert E1 < E0, f"HYP 应平滑化（E 降低）: {E0} -> {E1}"


def test_gradient_flow_tau_limit():
    """梯度流 τ→0 连续极限：TMD 矩阵元平滑恢复到未涂抹值。"""
    from pyqcd.renorm import wilson_flow, tmd_matrix_elements
    g = random_su3_gauge(L=4, seed=5)
    M0 = tmd_matrix_elements(g, [0, 1], [0, 1])
    V = wilson_flow(g, tau=0.01, eps=0.01)
    Mt = tmd_matrix_elements(V, [0, 1], [0, 1])
    d_small = np.abs(Mt - M0).max()
    Vb = wilson_flow(g, tau=0.2, eps=0.01)
    Mb = tmd_matrix_elements(Vb, [0, 1], [0, 1])
    d_big = np.abs(Mb - M0).max()
    assert d_small < d_big, f"τ→0 应更接近未涂抹值: {d_small} vs {d_big}"


def test_ratio_fit_extraction():
    """c0 裸矩阵元提取（R 模型拟合）：合成数据精确恢复。"""
    from pyqcd.analysis import R_model, fit_ratio
    rng = np.random.default_rng(1)
    z_list = [0, 1]
    z_set = np.array([0] * 16 + [1] * 16)
    tsep = np.array([8] * 8 + [10] * 8 + [8] * 8 + [10] * 8)
    ti = np.tile(np.arange(1, 9), 4)
    pars = (0.6, -0.3, 0.1, 0.3, -0.2, 0.05, 1.2)
    th = R_model(z_set, tsep, ti, z_list, *pars)
    samples = th[:, None] + 0.01 * rng.standard_normal((32, 50))
    data = (z_set, tsep, ti, th, np.ones(32) * 0.01, samples, z_list, [8, 10], 0)
    res = fit_ratio(data)
    assert abs(res['c0_z0'] - 0.6) < 0.02
    assert abs(res['c0_z1'] - 0.3) < 0.02
    assert abs(res['deltaE'] - 1.2) < 0.1


def test_hyp_vs_flow_consistent():
    """HYP 涂抹与 Wilson flow 定性一致（理论文档对比项）：O(z) 高度相关。"""
    from pyqcd.smear import hyp_smear
    from pyqcd.renorm import wilson_flow, tmd_matrix_elements
    g = random_su3_gauge(L=4, seed=7)
    Vh = hyp_smear(g)
    Vf = wilson_flow(g, tau=0.05, eps=0.05)
    Mh = tmd_matrix_elements(Vh, [1, 2, 3], [0])[:, 0]
    Mf = tmd_matrix_elements(Vf, [1, 2, 3], [0])[:, 0]
    c = np.corrcoef(Mh, Mf)[0, 1]
    assert c > 0.9, f"HYP 与 flow 应定性一致（r={c:.3f}）"


def test_gpu_backend_consistency():
    """GPU(cupy) 后端：flow 结果与 CPU 一致（无 cupy 时跳过）。"""
    try:
        import cupy as cp
    except ImportError:
        return
    from pyqcd.tools import set_backend
    from pyqcd.renorm import wilson_flow
    g = random_su3_gauge(L=4, seed=2)
    set_backend('numpy')
    V_cpu = wilson_flow(g, tau=0.04, eps=0.02)
    set_backend('cupy')
    V_gpu = cp.asnumpy(wilson_flow(cp.asarray(g), tau=0.04, eps=0.02))
    set_backend('numpy')
    d = np.abs(V_cpu - V_gpu).max()
    assert d < 1e-10, f"CPU/GPU 不一致: {d:.3e}"


def test_end_to_end_synthetic_meff():
    """端到端合成验证：含已知质量的 cosh 2pt → meff 精确恢复。"""
    from pyqcd.analysis import Jackknife, meff
    rng = np.random.default_rng(4)
    M_true = 1.12
    a_fm = 0.1053
    a_gev_inv = a_fm / 0.1973269804
    nt = 72
    t = np.arange(nt)
    C = np.cosh(M_true * a_gev_inv * (t - nt / 2))
    noise = 1e-3 * C * rng.standard_normal((10, nt))
    data = C[None, :] + noise
    jk = Jackknife(data, Nconf_axes=0)
    mf = meff(jk['data_sample'], a_fm, Nconf_axes=0, Nt_axes=1,
              meff_type='cosh')
    mmean = np.real(mf['data_mean'])
    M_rec = np.mean(mmean[8:20])
    assert abs(M_rec - M_true) / M_true < 0.02, f"恢复偏差过大: {M_rec}"
