"""
本征模压缩：特征向量代数与 V1–V4 压缩方案（蒸馏 Nev 降维）
============================================================

整合 refer/sush/lqcddb/src/lqcddb/eigvectors/vector.py（vector_creator）
的逻辑，语义等价重构为函数式 API 并增加可复现种子（不 import refer/）。

特征向量布局：(Nev, Nz, Ny, Nx, Nc)（lqcddb 约定，动量 [pz,py,px] 配套）。

    - inner_product / check_orthonormal / normalize / orthnormal_append /
      create_noise：内积、正交归一检查、归一化、Gram–Schmidt 追加、
      正交补随机噪声向量；
    - compress_matrix_V1：求和压缩（组内求和 ÷√组大小，保正交归一）；
    - compress_matrix_V2：随机抽取压缩（组内无放回抽取 N_extract 个本征模）；
    - compress_matrix_V3：随机正交投影压缩（orthonormal 随机基）；
    - compress_matrix_V4：V3 的噪声策略可切换版（'orthnormal' | 'Z_N'）。

分组约定：Ctype='I' 交错流（indices j::N_sum）/ 'B' 连续块（均分切分），
与 lqcddb reshape 语义一致。用途：Nev → N_sum 降维直接降低顶点/2pt 步
显存与耗时（vertex conf6250 GPU 36s、2pt 峰值 570MB 的扩规模增强项）。

与 lqcddb 原版的差异（工程化重构，物理语义不变）：
    - `raise print(...)` 反模式改为 ValueError；check 尾部的字符串比较
      死代码修复为真实断言；
    - 随机性统一由 numpy Generator（seed 可复现、跨后端一致）生成后转换，
      不依赖全局 backend.random；V2 的 while-去重循环等价改写为
      rng.choice(replace=False)；Z_N 取值表向量化等价采样；
    - 输出 dtype 跟随输入（保留 complex64 显存收益），不再硬编码 <c16。
"""
from __future__ import annotations

import numpy as np

from ..tools._backend import get_backend


def _as_rng(seed=None):
    """seed → numpy Generator（None 时用新熵源）。"""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def _flatten_vol(vectors):
    """(N, …) → ((N, V), 原 shape)。"""
    shape = tuple(vectors.shape)
    return vectors.reshape(shape[0], -1), shape


def inner_product(init_vector, test_vector, mode=''):
    """体积分内积 C_n = Σ_v init_n(v)* · test(v)（沿第 0 轴批量）。

    mode='' 返回复内积；mode='abs' 返回 |C|²。
    """
    cp = get_backend()
    a, _ = _flatten_vol(cp.asarray(init_vector))
    b, _ = _flatten_vol(cp.asarray(test_vector))
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"体积不一致: {a.shape[1]} vs {b.shape[1]}")
    c = cp.einsum('nv,nv->n', a.conj(), b)
    if mode == '':
        return c
    if mode == 'abs':
        return c * cp.conj(c)
    raise ValueError(f"未知 mode: {mode}")


def check_orthonormal(eigvecs, tol=1e-10, check_normal=True, verbose=False):
    """正交归一性检查：Gram 矩阵 A=VV† 对角≈1、非对角 < tol。"""
    cp = get_backend()
    vecs = cp.asarray(eigvecs)
    if bool(cp.isnan(vecs).any()):
        if verbose:
            print("eigen have nan")
        return False
    flat, _ = _flatten_vol(vecs)
    gram = cp.einsum('nv,mv->nm', flat, flat.conj())
    gram = gram.get() if hasattr(gram, 'get') else np.asarray(gram)
    ok = True
    off = gram - np.diag(np.diag(gram))
    if check_normal:
        diag_err = np.abs(np.diag(gram) - 1.0).max()
        if diag_err >= tol:
            if verbose:
                bad = int((np.abs(np.diag(gram) - 1.0) >= tol).sum())
                print(f"eigen don't normal ({bad} vectors, max err {diag_err:.2g})")
            ok = False
    if np.abs(off).max() >= tol:
        if verbose:
            pos = np.argwhere(np.abs(off) >= tol)
            print(f"don't orth ({len(pos)} pairs)")
        ok = False
    elif verbose and ok:
        print(f"orthonormal within tol={tol}")
    return ok


def normalize(vectors):
    """逐向量归一化（保持形状与后端）。"""
    cp = get_backend()
    flat, shape = _flatten_vol(cp.asarray(vectors))
    norm = cp.sqrt(cp.einsum('nv,nv->n', flat.conj(), flat).real)
    return (flat / norm[:, None]).reshape(shape)


def orthnormal_append(vectors_init, vector):
    """Gram–Schmidt 追加单个向量（对已有正交集投影去除分量后归一）。"""
    cp = get_backend()
    init = cp.asarray(vectors_init)
    base, shape = _flatten_vol(normalize(init))
    new = cp.asarray(vector).reshape(1, -1)
    new = normalize(new)
    coeff = cp.einsum('nv,mv->mn', base.conj(), new)
    new = normalize(new - cp.einsum('mn,nv->mv', coeff, base))
    out = cp.append(base, new, axis=0)
    return out.reshape((shape[0] + 1,) + shape[1:])


def create_noise(vectors_init, n_extra, dtype='complex', seed=None):
    """在 vectors_init 张成空间的正交补生成 n_extra 个归一化随机向量。

    与原版一致：均匀随机 → 投影去除既有集合分量 → 归一化 → 追加（逐个，
    后续噪声亦与前序噪声正交）。dtype='float' 时取实随机。
    """
    cp = get_backend()
    rng = _as_rng(seed)
    cur, shape = _flatten_vol(normalize(cp.asarray(vectors_init)))
    v_total = cur.shape[1]

    def rand_unit(n):
        z = rng.uniform(-1.0, 1.0, (n, v_total))
        if dtype == 'complex':
            z = z + 1j * rng.uniform(-1.0, 1.0, (n, v_total))
        t = cp.asarray(z)
        return normalize(t)

    for _ in range(int(n_extra)):
        cand = rand_unit(1)
        coeff = cp.einsum('nv,mv->mn', cur.conj(), cand)
        cand = normalize(cand - cp.einsum('mn,nv->mv', coeff, cur))
        cur = cp.append(cur, cand, axis=0)
    return cur.reshape((cur.shape[0],) + shape[1:])


# ═══════════════════════════════════════════════════════════════════
# 分组工具：Ctype='I' 交错流 / 'B' 连续块 → 每组一个索引数组
# ═══════════════════════════════════════════════════════════════════

def _split_indices(n_ev, N_sum, Ctype):
    idx = np.arange(n_ev)
    if Ctype == 'I':
        return [idx[j::N_sum] for j in range(N_sum)]
    if Ctype == 'B':
        return list(np.array_split(idx, N_sum))
    raise ValueError("Ctype 须为 'I'（交错）或 'B'（块）")


def compress_matrix_V1(eigenvectors, N_sum, Ctype='I'):
    """求和压缩：每组求和 ÷√组大小（组内正交 ⇒ 输出仍正交归一）。

    Args:
        eigenvectors: (Nev, Nz, Ny, Nx, Nc)。
        N_sum: 输出向量数（Nev 须能被 N_sum 整除）。
        Ctype: 'I' 交错 / 'B' 连续块。
    Note:
        快速路径：reshape + 单次 sum（交错=reshape(m,N_sum,…).sum(0)，
        块=reshape(N_sum,m,…).sum(1)），与逐组索引求和数值等价。
    """
    cp = get_backend()
    vecs = cp.asarray(eigenvectors)
    n_ev = vecs.shape[0]
    if n_ev % N_sum:
        raise ValueError(f"Nev={n_ev} 不能被 N_sum={N_sum} 整除")
    m = n_ev // N_sum
    tail = tuple(vecs.shape[1:])
    if Ctype == 'I':
        summed = cp.sum(vecs.reshape((m, N_sum) + tail), axis=0)
    elif Ctype == 'B':
        summed = cp.sum(vecs.reshape((N_sum, m) + tail), axis=1)
    else:
        raise ValueError("Ctype 须为 'I'（交错）或 'B'（块）")
    return summed / cp.sqrt(float(m))


def compress_matrix_V2(eigenvectors, N_sum, N_extract, Ctype='I', seed=None):
    """随机抽取压缩：每大组从对应本征模组无放回抽 N_extract 个。

    大组数 = N_sum // N_extract（每大组产出 N_extract 个输出，
    输出为本征模原样成员——非组合）。
    """
    cp = get_backend()
    vecs = cp.asarray(eigenvectors)
    n_ev = vecs.shape[0]
    n_groups = N_sum // N_extract
    if n_groups * N_extract != N_sum:
        raise ValueError("N_sum 必须能被 N_extract 整除")
    rng = _as_rng(seed)
    picks = []
    for grp in _split_indices(n_ev, n_groups, Ctype):
        take = np.sort(rng.choice(len(grp), size=N_extract, replace=False))
        picks.append(grp[take])
    sel = np.concatenate(picks)
    return vecs[list(sel)]


_ZN_VALUES = np.array([1.0 + 0j, -1.0 + 0j, 1j, -1j,
                       np.sqrt(2) + 1j * np.sqrt(2),
                       np.sqrt(2) - 1j * np.sqrt(2),
                       -np.sqrt(2) + 1j * np.sqrt(2),
                       -np.sqrt(2) - 1j * np.sqrt(2)], dtype=complex)


def _random_rows(v_dim, k, scheme, rng):
    """k × v_dim 随机行矩阵。

    'orthnormal'：均匀复随机行经 Gram–Schmidt 正交归一；
    'Z_N'（N=2..8）：lqcddb 相位取值表逐元素采样（含 √2 半径相位点、
    不做行归一，输出一般不正交——沿用原版约定）。
    """
    if scheme == 'orthnormal':
        def rand_unit_row():
            z = rng.uniform(-1, 1, (1, v_dim)) \
                + 1j * rng.uniform(-1, 1, (1, v_dim))
            return (z / np.linalg.norm(z)).reshape(-1)

        rows = rand_unit_row()[None, :]
        for _ in range(k - 1):
            z = rand_unit_row()
            # 正交化：去除沿既有行的分量（系数 = ⟨row_r, z⟩，共轭在行侧）
            z = z - (rows.conj() @ z).ravel() @ rows
            rows = np.vstack([rows, (z / np.linalg.norm(z))[None, :]])
        return rows
    if scheme.startswith('Z_'):
        n_root = min(int(scheme[2:]), len(_ZN_VALUES))
        return _ZN_VALUES[rng.integers(0, n_root, (k, v_dim))]
    raise ValueError(f"未知噪声方案: {scheme}")


def compress_matrix_V3(eigenvectors, N_sum, N_extract, Ctype='I', seed=None,
                       check=True):
    """随机正交投影压缩：每大组用 N_extract 个正交随机基投影。

    输出正交性由随机基正交性 × 输入组内正交性保证（check=True 时校验）。
    """
    return _compress_projection(eigenvectors, N_sum, N_extract, Ctype,
                                scheme='orthnormal', seed=seed, check=check)


def compress_matrix_V4(eigenvectors, N_sum, N_extract, Ctype='I', seed=None,
                       random_type='orthnormal', check=None):
    """V3 噪声策略可切换版：random_type='orthnormal' | 'Z_N'。

    check 默认仅在 orthnormal 下开启（Z_N 行不归一，Gram 校验无意义）。
    """
    do_check = (random_type == 'orthnormal') if check is None else check
    return _compress_projection(eigenvectors, N_sum, N_extract, Ctype,
                                scheme=random_type, seed=seed, check=do_check)


def _compress_projection(eigenvectors, N_sum, N_extract, Ctype, scheme,
                         seed, check):
    cp = get_backend()
    vecs = cp.asarray(eigenvectors)
    n_ev = vecs.shape[0]
    n_groups = N_sum // N_extract
    if n_groups * N_extract != N_sum:
        raise ValueError("N_sum 必须能被 N_extract 整除")
    rng = _as_rng(seed)
    out = np.empty((N_sum,) + tuple(vecs.shape[1:]),
                   dtype=_to_numpy_dtype(vecs))
    pos = 0
    for grp in _split_indices(n_ev, n_groups, Ctype):
        block = vecs[list(grp)]
        v_flat, _ = _flatten_vol(block)
        rows = _random_rows(len(grp), N_extract, scheme, rng)
        proj = cp.asarray(rows) @ v_flat          # (N_extract, V)
        proj = proj.reshape(N_extract, *vecs.shape[1:])
        out[pos:pos + N_extract] = proj.get() if hasattr(proj, 'get') \
            else np.asarray(proj)
        pos += N_extract
    result = cp.asarray(out)
    if check:
        flat = result.get() if hasattr(result, 'get') else np.asarray(result)
        g = flat.reshape(N_sum, -1)
        gram = g @ g.conj().T
        off_err = np.abs(gram - np.diag(np.diag(gram))).max()
        norm_err = np.abs(np.diag(gram) - 1.0).max()
        if off_err > 1e-8 or norm_err > 1e-8:
            raise RuntimeError(
                f"compress 投影失去正交归一: off={off_err:.2g} "
                f"norm={norm_err:.2g}")
    return result


def _to_numpy_dtype(arr):
    """后端数组 → numpy dtype（构造宿主侧容器用）。"""
    a = np.asarray(arr.get() if hasattr(arr, 'get') else arr)
    return a.dtype


# ═══════════════════════════════════════════════════════════════════
# Ω 加速张量（蒸馏重子收缩的方差压缩权重，lqcddb create_omega_accelerate）
# ═══════════════════════════════════════════════════════════════════

def create_omega_accelerate(n_voxel, exact=0, N_eigen=None, N_sum=None,
                            N_extract=None, noise=0, conserved=False,
                            normal=False, dim=2):
    """构建 Ω 权重张量（任意 dim=2/3 维，exact+块压缩+噪声三段分区）。

    照抄 refer/sush lqcddb vertex.py::create_omega_accelerate 的语义
    （不 import 来源）：把用于收缩的 Nev = Σev_sum 个向量按
    [exact] + 压缩块 + noise 分区，Ω[a₁,…,a_dim] 给出逐分区的
    方差压缩权重 w_i^{(j)} = (space_i − j)/(sum_i − j)
    （part i 同时出现在 j 个维度时的重叠修正；conserved 模式退化为
    space_i/sum_i 无对角修正）。exact-only 时 Ω ≡ 1（无压缩无修正）。

    Args:
        n_voxel: 单向量的体素数 V = Nz·Ny·Nx·Nc。
        exact: 精确保留的本征矢数量。
        N_eigen/N_sum/N_extract: 各块压缩前/后数量与每子块抽取数
                （给出 N_eigen 时按 N_sum/N_extract 展开为子块列表，
                与 V1/V2/V3/V4 压缩方案的输出分组一一对应）；
                不做块压缩时传 None 并直接给 N_sum=[各块输出数]。
        noise: 噪声向量数量。
        conserved: 守恒模式（权重恒为 space/sum，dim 强制 2）。
        normal: dim=2 时逐行归一化并对称化下三角。
        dim: 输出张量维度（2 或 3）。
    Returns:
        复数 Ω 张量 (Nev,)*dim（Nev = exact + Σ子块输出 + noise）。
    """
    from itertools import combinations
    cp = get_backend()
    if conserved:
        dim = 2
    n_eigen = list(N_eigen) if N_eigen else []
    n_sum_in = list(N_sum) if N_sum else []

    # 子块展开：每块按 N_sum/N_extract 切成若干 sum=N_extract 的子块。
    # 契约与原版一致：块压缩必须给全 (N_eigen, N_sum, N_extract) 三元组——
    # 原版对"仅 N_sum"输入直接越界崩溃（实跑验证），此处显式拒绝。
    tran_n_sum, tran_n_eigen = [], []
    if n_sum_in and not n_eigen:
        raise ValueError("块压缩须同时提供 N_eigen/N_sum/N_extract "
                         "（原版对仅 N_sum 未定义）")
    if n_eigen:
        for ne, ns in zip(n_eigen, n_sum_in):
            nex = N_extract[len(tran_n_sum)] if N_extract else 1
            ngrp = int(round(ns / nex))
            if abs(ns / nex - ngrp) > 1e-9:
                raise ValueError(
                    f"N_sum={ns} 不能被 N_extract={nex} 整除")
            per = ne / ngrp
            if abs(per - round(per)) > 1e-9:
                raise ValueError(
                    f"块大小 {ne} 不能均分为 {ngrp} 子块")
            for _ in range(ngrp):
                tran_n_sum.append(nex)
                tran_n_eigen.append(int(round(per)))
    else:
        tran_n_sum, tran_n_eigen = n_sum_in, n_sum_in

    ev_space = [x for x in ([exact] + tran_n_eigen
                            + [n_voxel - (exact + sum(tran_n_eigen))])
                if x != 0]
    ev_sum = [x for x in ([exact] + tran_n_sum + [noise]) if x != 0]
    len_space = len(ev_sum)
    nev = sum(ev_sum)

    slices = [slice(sum(ev_sum[:i]), sum(ev_sum[:i + 1]))
              for i in range(len_space)]
    weights = np.empty((len_space, dim), dtype=float)
    for i in range(len_space):
        for j in range(dim):
            # 守卫：sum_i ≤ j 的退化分区（原版会除零崩溃）——
            # 分母下限取 1，语义为"对角重叠修正饱和"
            if conserved:
                weights[i, j] = (ev_space[i]) / (ev_sum[i])
            else:
                weights[i, j] = ((ev_space[i] - j)
                                 / max(ev_sum[i] - j, 1))

    if dim <= 3:
        all_pos = np.unique(np.asarray(
            list(combinations(range(dim * len_space), dim)))
            % len_space, axis=0)
    else:
        raise ValueError("仅支持 dim≤3（原版 4D 分支未纳入）")

    omega = np.empty([nev] * dim, dtype=float)
    for pos in all_pos:
        used = [0] * len_space
        for j in pos:
            used[j] += 1
        position = [slices[p] for p in pos]
        w = 1.0
        for si in range(len_space):
            w *= float(np.prod(weights[si, :used[si]]))
        omega[tuple(position)] = w
        # 同一分区出现在多维度时的对角重叠降权（非守恒模式）
        grid = np.ogrid[[slice(0, s.stop - s.start)
                         for s in position]]
        for extra in range(1, len_space):
            for i in range(used[extra] - 1):
                dims_of_part = np.argwhere(pos == extra).reshape(-1)
                if conserved:
                    break
                for j in list(combinations(dims_of_part.tolist(), 2 + i)):
                    sub = omega[tuple(position)]
                    w_bool = np.ones(tuple(s.stop - s.start
                                           for s in position), dtype=bool)
                    w_diag = w
                    for k in range(len(j) - 1):
                        w_bool &= (grid[j[k]] == grid[j[k + 1]])
                        w_diag /= weights[extra, used[extra] - k - 1]
                    sub[w_bool] = w_diag

    omega_t = cp.asarray(omega)
    if normal and dim == 2:
        for i in range(nev):
            row = omega[i]
            omega[i] = row * nev / row.sum()
        tril = np.tril_indices(nev, -1)
        omega.T[tril] = omega[tril]
        return cp.asarray(omega.astype(complex))
    return omega_t.astype(complex)
