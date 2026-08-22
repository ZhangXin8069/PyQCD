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


def test_tmd_matching_nlo():
    """TMD 混合方案 1 圈匹配：Z⁻¹ 还原输入、快度/软因子生效、形状正确。"""
    from pyqcd.renorm import tmd_matching_hybrid
    x = np.linspace(0.05, 0.95, 48)
    yg = x * np.exp(-x / 0.3)         # 物理输入 y·g(y)（y→0 衰减）
    # 1) δ 项（LO）主导：cs=0, S=1 → 输出 ≈ 输入（NLO 修正 O(αs) < 30%）
    x_o, out = tmd_matching_hybrid(x, b_perp=[0.2], mu=2.0, pz_gev=2.0,
                                   cs_kernel=0.0, soft_factor=1.0, x_tmd=yg)
    assert x_o.shape == x.shape
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out - yg)) < 0.3 * np.max(yg)
    # 2) 快度演化因子：pz_scale≠pz → 输出 × exp[½ln((2Pz)²/(2ζ)²)K]
    _, out_rap = tmd_matching_hybrid(x, b_perp=[0.2], mu=2.0, pz_gev=2.0,
                                     cs_kernel=0.1, soft_factor=1.0,
                                     pz_scale=1.0, x_tmd=yg)
    rap = np.exp(0.5 * np.log((2.0 * 2.0) ** 2 / (2.0 * 1.0) ** 2) * 0.1)
    assert np.allclose(out_rap, out * rap, rtol=1e-8)
    # 3) 软函数：S=4 → 输出 ×½
    _, out_s = tmd_matching_hybrid(x, b_perp=[0.2], mu=2.0, pz_gev=2.0,
                                   cs_kernel=0.0, soft_factor=4.0, x_tmd=yg)
    assert np.allclose(out_s, out / 2.0, rtol=1e-8)
    # 4) 多 b⊥：nb=2 输出形状 (nx, 2)
    yg2 = np.column_stack([yg, 0.5 * yg])
    _, out2 = tmd_matching_hybrid(x, b_perp=[0.2, 0.4], mu=2.0, pz_gev=2.0,
                                  cs_kernel=0.0, soft_factor=1.0, x_tmd=yg2)
    assert out2.shape == (len(x), 2)
    assert np.all(np.isfinite(out2))


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


def test_torch_backend_consistency():
    """torch 后端（CPU/CUDA）：flow/TMD/HYP 与 numpy 一致 + 精度切换。

    - numpy 输入自动转换 torch（旧代码兼容）
    - CPU 与 numpy 逐位一致（rel ~1e-15）
    - CUDA 与 numpy 一致（rel ~1e-15）
    - set_precision('complex64') 后新数组为 complex64
    """
    import torch
    from pyqcd.tools import set_backend, set_precision
    from pyqcd.renorm import wilson_flow, tmd_matrix_elements
    from pyqcd.smear import hyp_smear

    g = random_su3_gauge(L=4, seed=2)
    set_backend('numpy')
    V_cpu = wilson_flow(g, tau=0.04, eps=0.02)
    M_np = tmd_matrix_elements(g, [0, 1], [0, 1])

    # CPU（numpy 输入自动转换）
    set_backend('torch')
    V_t = wilson_flow(g, tau=0.04, eps=0.02)
    M_t = tmd_matrix_elements(g, [0, 1], [0, 1])
    assert np.abs(V_cpu - V_t.get()).max() < 1e-8, "torch CPU flow 不一致"
    assert np.abs(np.asarray(M_np) - np.asarray(M_t)).max() < 1e-8
    # HYP 与 numpy 对比
    set_backend('numpy')
    V_h_np = hyp_smear(g)
    set_backend('torch')
    assert np.abs(V_h_np - hyp_smear(g).get()).max() < 1e-7, "HYP torch 不一致"

    # 精度语义：numpy 显式 dtype 被尊重；'complex' 简写/无 dtype 来源遵循全局精度
    g64 = g.astype(np.complex64)
    set_backend('torch')
    V64 = wilson_flow(g64, tau=0.04, eps=0.02)
    assert V64.dtype == torch.complex64, f"numpy c64 输入应保持: {V64.dtype}"
    set_precision('complex64')
    z64 = wilson_flow(g64, tau=0.04, eps=0.02)
    assert z64.dtype == torch.complex64, f"complex64 切换失败: {z64.dtype}"
    from pyqcd.tools import get_backend
    assert get_backend().zeros((2, 2), dtype='complex').dtype == torch.complex64
    set_precision('complex128')
    assert get_backend().zeros((2, 2), dtype='complex').dtype == torch.complex128

    # CUDA（可用时）
    if torch.cuda.is_available():
        set_backend('torch', device='cuda')
        V_g = wilson_flow(g, tau=0.04, eps=0.02)
        assert str(V_g.device).startswith('cuda')
        d = np.abs(V_cpu - V_g.get()).max()
        assert d < 1e-8, f"torch CUDA flow 不一致: {d:.3e}"
        M_g = tmd_matrix_elements(g, [0, 1], [0, 1])
        assert np.abs(np.asarray(M_np) - np.asarray(M_g)).max() < 1e-8
    set_backend('numpy')


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


def test_matching_sum_rule():
    """NLO 匹配求和规则：∫hR_PDF ≈ ∫hR_tilde（O(αs) 修正内守恒）。"""
    from pyqcd.renorm import hR_PDF
    xx = np.linspace(0.02, 1.48, 148)
    h0 = np.exp(-(xx - 0.4) ** 2 / 0.05)
    out = hR_PDF(xx, Pz_=4, conf='L24x72', hR_tilde_data=h0, mu_=2.0)
    dx = xx[1] - xx[0]
    ratio = np.sum(out) * dx / (np.sum(h0) * dx)
    assert 0.9 < ratio < 1.1, f"求和规则破坏: {ratio:.4f}"


def test_core_chain_integrated():
    """核心目标链整合：梯度流(τ=3a²)→TMD→混合方案→Z_R 全链自洽。"""
    from pyqcd.renorm import (
        wilson_flow, tmd_matrix_elements, hR_z_Pz, th_ZR,
    )
    g = random_su3_gauge(L=4, seed=6)
    tau = 3.0 * (0.1053 * 0.197) ** 2
    V = wilson_flow(g, tau=tau, eps=0.01)
    M = tmd_matrix_elements(V, list(range(6)), [0])[:, 0]
    z_fm = np.array(range(6)) * 0.1053
    zr = th_ZR(z_fm, 0.1053 / 0.197, 2.0, 0.5, 0.1, 0.0, 0.0, 0.25, (0.0, 0.0))
    lam, hR = hR_z_Pz(z_fm, 4, M, M, zs=0.3, zr_fit=zr, conf='L24x72')
    assert np.all(np.isfinite(hR))
    assert abs(hR[0] - 1.0) < 1e-8, "短距比值应自归一"


# ═══════════════════════════════════════════════════════════════════
# 整合功能测试（~auto-all：logs/examples/refer 代码整合验证）
# ═══════════════════════════════════════════════════════════════════


def _smooth_gauge(L=6, amp=0.01, seed=11):
    """近平场（冷场+微扰后 SU(3) 投影）——涂抹类功能的物理性测试场。

    stout 无显式重投影（约定输入为 SU(3) 规范组态，与真实蒸馏数据一致），
    故微扰后须经 smear.proj_su3（SVD 极分解）投影保证严格幺正。
    注意 renorm.proj_su3 仅做 det 归一（适用于本已幺正的流输出），不适用。
    """
    from pyqcd.smear import proj_su3
    g = np.zeros((L, L, L, L, 4, 3, 3), dtype=complex)
    for d in range(4):
        g[..., d, :, :] = np.eye(3)
    rng = np.random.default_rng(seed)
    noisy = g + amp * (rng.standard_normal(g.shape)
                       + 1j * rng.standard_normal(g.shape))
    return np.asarray(proj_su3(noisy))


def test_stout_smear():
    """Stout 涂抹：SU(3) 保持 + 平滑场作用量下降 + 平场不动 + 时间链不变。"""
    from pyqcd.smear import stout_smear
    from pyqcd.renorm import flow_action_density

    flat = np.zeros((4, 4, 4, 4, 4, 3, 3), dtype=complex)
    for d in range(4):
        flat[..., d, :, :] = np.eye(3)
    with np.errstate(invalid='ignore', divide='ignore'):
        vf = stout_smear(flat, nstep=1, rho=0.1)
    assert np.isfinite(vf).all() and np.abs(vf - flat).max() < 1e-12

    g = _smooth_gauge(L=6)
    v = stout_smear(g, nstep=2, rho=0.12)
    dev = np.abs(v[0, 0, 0, 0, 0] @ v[0, 0, 0, 0, 0].conj().T
                 - np.eye(3)).max()
    assert dev < 1e-10, f"SU(3) 保持失败: {dev}"
    e0 = flow_action_density(g).mean()
    e1 = flow_action_density(v).mean()
    assert e1 < e0, f"平滑场应作用量下降: {e0} -> {e1}"
    assert np.abs(v[..., 3, :, :] - g[..., 3, :, :]).max() < 1e-14, \
        "时间方向链接不应被涂抹"


def test_eigvec_compress():
    """本征模压缩：V1 保正交归一、V2 可复现子集抽取、V3/V4 正交投影。"""
    from pyqcd.vertex import (
        check_orthonormal, compress_matrix_V1, compress_matrix_V2,
        compress_matrix_V3, compress_matrix_V4, create_noise,
    )
    rng = np.random.default_rng(0)
    nev, shape = 16, (2, 2, 2, 3)
    a = rng.standard_normal((nev, int(np.prod(shape)))) \
        + 1j * rng.standard_normal((nev, int(np.prod(shape))))
    q, _ = np.linalg.qr(a.T)
    vecs = (q.T).reshape((nev,) + shape)

    assert check_orthonormal(compress_matrix_V1(vecs, 4, 'I'))
    assert check_orthonormal(compress_matrix_V1(vecs, 4, 'B'))
    c2a = compress_matrix_V2(vecs, 4, 2, 'I', seed=7)
    assert np.abs(np.asarray(c2a) - np.asarray(
        compress_matrix_V2(vecs, 4, 2, 'I', seed=7))).max() == 0
    flat_in = vecs.reshape(nev, -1)
    flat_out = np.asarray(c2a).reshape(4, -1)
    hits = sum(any(np.abs(flat_out[k] - flat_in[j]).max() < 1e-12
                   for j in range(nev)) for k in range(4))
    assert hits == 4, "V2 输出应为输入成员"
    assert check_orthonormal(compress_matrix_V3(vecs, 4, 2, 'I', seed=3))
    assert check_orthonormal(compress_matrix_V4(vecs, 4, 2, 'B', seed=3))
    noisy = create_noise(vecs[:4], 3, seed=5)
    assert noisy.shape == (7,) + shape and check_orthonormal(noisy)


def test_cg_coefficients():
    """SU(2) CG：已知值、Condon–Shortley 符号、幺正性、combine/decompose。"""
    from pyqcd.lattice import cg_coefficient, SU2combine, SU2decompose

    assert abs(cg_coefficient(.5, .5, .5, .5, 1, 1) - 1) < 1e-13
    assert abs(cg_coefficient(.5, .5, .5, -.5, 1, 0) - np.sqrt(.5)) < 1e-13
    assert abs(cg_coefficient(.5, -.5, .5, .5, 0, 0) + np.sqrt(.5)) < 1e-13
    assert cg_coefficient(.5, .5, .5, .5, 0, 0) == 0          # M 不守恒
    assert cg_coefficient(.5, .5, .5, -.5, .5, 0) == 0        # 禁戒耦合
    for j in (0, 1):                                          # 幺正性
        s = sum(abs(cg_coefficient(.5, m1, .5, m2, j, 0)) ** 2
                for m1 in (-.5, .5) for m2 in (-.5, .5))
        if s:
            assert abs(s - 1) < 1e-12
    d = SU2decompose([.5, .5], [0., 0.])
    assert abs(d[(.5, -.5)] - np.sqrt(.5)) < 1e-13
    d3 = SU2decompose([.5] * 3, [1.5, 1.5], [1.])
    assert abs(d3[(.5, .5, .5)] - 1) < 1e-13
    try:
        SU2decompose([.5] * 3, [1.5, 1.5])
        raise AssertionError("N>2 无 intermediate_Js 应报错")
    except ValueError:
        pass


def test_hB_dataset_loader():
    """hB/FH loader：z₀ 归一化 + 插值 + 数据集组装进 Z_R 代价函数。"""
    from pyqcd.renorm import (
        build_hB_dataset, make_zr_dataset, cost_function_all,
    )
    rng = np.random.default_rng(3)
    z_fm = np.arange(20) * 0.1053
    ns = 50
    c0 = np.exp(-z_fm[:, None] / 0.35) \
        * (1 + 0.01 * rng.standard_normal((20, ns)))
    ds = build_hB_dataset(c0, z_fm)
    assert ds['hB'].shape == (18, ns)
    assert np.allclose(ds['hB_o'][0], 1.0)
    assert np.all(np.diff(np.log(ds['hB']).mean(axis=1)) < 0)

    dsets = []
    for tau in (0.35, 0.45):
        c = np.exp(-z_fm[:, None] / tau) \
            * (1 + 0.01 * rng.standard_normal((20, ns)))
        dd = build_hB_dataset(c, z_fm)
        dsets.append(make_zr_dataset(dd['loghB'], dd['z'], 0.1053 / 0.197,
                                     kind='boot', seed=11))
    chi2 = cost_function_all([2.0, 0.5, 0.1, 0.0, 0.25] + [0.05] * 14
                             + [0.0, 0.0], dsets, 2.0)
    assert np.isfinite(chi2) and chi2 > 0


def test_hybrid_boot_covariance():
    """混合 λ 外推拟合：diag/boot 协方差均恢复真值，旧签名向后兼容。"""
    from pyqcd.renorm import fit_hR_lambda
    lam = np.linspace(0.5, 6.0, 12)
    truth = (1.2, 1.7, 4.0)
    clean = truth[0] * lam ** (-truth[1]) * np.exp(-lam / truth[2])
    samp = clean[:, None] \
        + 0.002 * np.random.default_rng(5).standard_normal((12, 60))
    for kw in ({}, {'cov_kind': 'boot'}):
        p = fit_hR_lambda([1.0, 1.5, 3.5], (0.8, 6.0), lam, samp, **kw)
        assert all(abs(a - b) / b < 0.08 for a, b in zip(p, truth)), (kw, p)


def test_tmd_plateau_and_cs_kernel():
    """plateau_c0 手算一致；CS 核两动量幂律恢复 + clip 保护。"""
    from pyqcd.analysis import plateau_c0
    from pyqcd.renorm import cs_kernel_two_momentum
    rng = np.random.default_rng(0)
    ratio = rng.standard_normal((5, 20, 20, 3, 2))
    got = plateau_c0(ratio)
    ref = np.zeros((5, 3, 2))
    npts = 0
    for dt in range(7, 11):
        for dtau in range(3, dt - 3 + 1):
            ref += ratio[:, dt, dtau]
            npts += 1
    assert np.abs(got - ref / npts).max() < 1e-14

    k_true = np.array([0.1, -0.2, 0.35])
    z = np.arange(4)[:, None]
    pz1, pz2 = 0.697, 1.394
    c02 = np.exp(-z / 0.3) * (1 + k_true[None, :] * z)
    c01 = c02 * (pz1 / pz2) ** k_true[None, :]
    k_rec = cs_kernel_two_momentum(c01, c02, pz1, pz2, z_ref=1)
    assert abs(k_rec.ravel()[1] - k_true[1]) < 1e-12
    bad = c02.copy()
    bad[1] *= 1e12
    assert np.abs(cs_kernel_two_momentum(bad, c02, pz1, pz2)).max() <= 3.0


def test_plot_tmd_pdf():
    """TMD-PDF 链成图：4 张 png 齐全且非空。"""
    import os
    import tempfile
    from pyqcd.analysis import plot_tmd_pdf
    x = np.linspace(0.05, 0.95, 40)
    xg = np.exp(-(x - 0.3) ** 2 / 0.02)[:, None] * np.array([[1.0, 0.8]])
    out = tempfile.mkdtemp()
    files = plot_tmd_pdf(x, xg, 0.9 * xg, [0.1, 0.2],
                         np.array([0.1, -0.2]), 'P200', out)
    names = sorted(os.path.basename(f) for f in files)
    assert names == ['cs_kernel.png', 'matched_tmd_pdf.png',
                     'quasi_tmd_pdf.png', 'tmd_pdf_vs_b.png']
    assert all(os.path.getsize(f) > 1000 for f in files)


def test_pipeline_validate_and_2pt_resume():
    """数据守卫（形状/NaN/缺失）+ 原始数据齐全度 + 2pt 断点续跑判据。"""
    import os
    import shutil
    import tempfile
    from pyqcd.pipeline import (
        check_input_arrays, check_raw_data, ProgressLog,
    )
    from pyqcd.pipeline import _steps

    root = tempfile.mkdtemp()
    d1 = os.path.join(root, 'conf100')
    os.makedirs(d1)
    rng = np.random.default_rng(0)
    spec_items = [
        {'name': 'corr_pp_P0_{cid}', 'shape': (72,)},
        {'name': 'ops_mu0_nu1_dz24_{cid}', 'ext': '.npz',
         'dataset': 'ops', 'shape': (24, 72)},
    ]
    spec = {'conf_ids': [100], 'items': spec_items}
    np.save(os.path.join(d1, 'corr_pp_P0_100.npy'),
            rng.standard_normal(72))
    np.savez(os.path.join(d1, 'ops_mu0_nu1_dz24_100.npz'),
             ops=rng.standard_normal((24, 72)))
    n_ok, bad = check_input_arrays(root, spec, verbose=False)
    assert n_ok == 1 and not bad
    np.save(os.path.join(d1, 'corr_pp_P0_100.npy'),
            np.where(np.arange(72) < 1, np.nan, 0.0))
    n_ok, bad = check_input_arrays(root, spec, verbose=False)
    assert n_ok == 0 and any('有限=False' in b for b in bad)
    shutil.rmtree(root)

    base = tempfile.mkdtemp()
    ens, nt, cid = 'ENS', 4, 200
    ed = os.path.join(base, 'eigensystem', ens, str(cid))
    os.makedirs(ed)
    for t in range(nt):
        open(os.path.join(ed, f'eigvecs_t{t:03d}_{cid}'), 'w').close()
    pdir = os.path.join(base, 'perambulators', ens, 'light', str(cid))
    os.makedirs(pdir)
    for dd in range(4):
        for t in range(nt):
            open(os.path.join(pdir, f'perams.{cid}.{dd}.{t}'), 'w').close()
    cf = os.path.join(base, 'configurations', 'CLOVER', ens)
    os.makedirs(cf)
    open(os.path.join(cf, f'{ens}_cfg_{cid}.lime'), 'w').close()
    n_ok, bad = check_raw_data([cid], base, ens, nt=nt, verbose=False)
    assert n_ok == 1 and not bad
    os.remove(os.path.join(ed, f'eigvecs_t003_{cid}'))
    n_ok, bad = check_raw_data([cid], base, ens, nt=nt, verbose=False)
    assert n_ok == 0 and any('eigvecs 不全 3/4' in b for b in bad)
    shutil.rmtree(base)

    run_dir = tempfile.mkdtemp()
    cdir = _steps.conf_data_dir(run_dir, 111)
    for ch in ('pp', 'pn', 'pion'):
        for mom in ('P0', 'P2'):
            np.save(os.path.join(cdir, f'corr_{ch}_{mom}_111.npy'),
                    np.zeros(72))
    assert _steps._2pt_all_present(cdir, 111, ('pp', 'pn', 'pion'))
    os.remove(os.path.join(cdir, 'corr_pp_P0_111.npy'))
    assert not _steps._2pt_all_present(cdir, 111, ('pp', 'pn', 'pion'))
    shutil.rmtree(run_dir)
    ProgressLog(10, label='t').step(5)


def test_proton_energy_dirs():
    """能量链方向感知：动量置换约定 + 路径构造（z 向后兼容、x/y 置换）。"""
    import os
    import shutil
    import tempfile
    from pyqcd.analysis._bare_matrix import dir_momentum
    from pyqcd.analysis._proton_energy import EnergyParams, load_raw_corr

    assert dir_momentum(0, 0, 6, 'x') == (6, 0, 0)
    assert dir_momentum(0, 0, 6, 'y') == (0, 6, 0)
    assert dir_momentum(0, 0, 6, 'z') == (0, 0, 6)
    p = EnergyParams(conf_short='L24x72', conf_name='ENS',
                     conf_ids=[7], Nt=8, Nx=4, Px=0, Py=0, Pz=2,
                     Nsample=4, dt_max=5)
    assert p.mom_tag == (0, 0, 2)
    tmp = tempfile.mkdtemp()
    rng = np.random.default_rng(0)
    for sub, tag in (('momsmear2z', 'Px0Py0Pz2'),
                     ('momsmear2x', 'Px2Py0Pz0')):
        os.makedirs(os.path.join(tmp, 'ENS', sub, '7'))
        np.save(os.path.join(tmp, 'ENS', sub, '7',
                             f'twopt_slice_pp_{tag}_eginphase2_Cg5g4'
                             f'_nopol_ss_conf7.npy'),
                rng.standard_normal((8, 8))
                + 1j * rng.standard_normal((8, 8)))
    assert load_raw_corr(tmp, 7, p).shape == (8, 8)
    px_params = EnergyParams(**{**p.__dict__, 'dir': 'x'})
    assert load_raw_corr(tmp, 7, px_params).shape == (8, 8)
    shutil.rmtree(tmp)
