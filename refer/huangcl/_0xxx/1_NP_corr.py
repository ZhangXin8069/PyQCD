# 功能: 计算 2pt_NP 双核子关联函数.
# 输入: 从标准输入按 "key value" 行读取 Nt, Nx, N_dt, conf_id, Nev, light_peram_dir, eigvec_dir, result_dir.
# 输出: 在 result_dir 下保存 NPcorr_conf{conf_id}_P0.npy, 数组形状为 (Nmom // 2, Nmom, N_dt, Nt, 2, 2, 2, 2).


import fileinput
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np
from opt_einsum import contract_expression

from matrixes import NP_coeff, NP_indexes  # noqa: E402

# ---- 从上级目录 98_tools 导入通用函数 ----
# sys.path.insert 必须在 import 之前, 否则运行时 ImportError
sys.path.insert(0, str("/public/home/huangcl/06_deuteron_gluon_pdf/98_tools"))
from distillation_tools import calc_vvv, read_peram_timeslice  # noqa: E402  # type: ignore
from gamma_matrix_DP import gamma  # noqa: E402  # type: ignore
from profiling_tools import print_mem_peak  # noqa: E402  # type: ignore


infile = fileinput.input()
for line in infile:
    tmp = line.split()
    if tmp[0] == "Nt":
        Nt = int(tmp[1])
    if tmp[0] == "Nx":
        Nx = int(tmp[1])
    if tmp[0] == "N_dt":
        N_dt = int(tmp[1])
    if tmp[0] == "conf_id":
        conf_id = int(tmp[1])
    if tmp[0] == "Nev":
        Nev = int(tmp[1])
    if tmp[0] == "light_peram_dir":
        light_peram_dir = tmp[1]
    if tmp[0] == "eigvec_dir":
        eigvec_dir = tmp[1]
    if tmp[0] == "result_dir":
        result_dir = tmp[1]


Px = [0, 0, 0, 1, 0, 0, 0, -1]
Py = [0, 0, 1, 0, 0, 0, -1, 0]
Pz = [0, 1, 0, 0, 0, -1, 0, 0]
Nmom = len(Px)


st_total = time.perf_counter()
# ========== 计算 vvv ==========
st_vvv = time.perf_counter()

# para: p, t, eig1_idx, eig2_idx, eig3_idx
vvv = cp.empty((Nmom, Nt, Nev, Nev, Nev), dtype=cp.complex128)
for p in range(Nmom):
    if p == 4:
        vvv[p] = vvv[0]
        continue

    vvv[p] = calc_vvv(
        eigvec_dir,
        conf_id,
        Nt,
        Nx,
        Nev,
        (Px[p], Py[p], Pz[p]),
        xp=cp,
    )

    cp.get_default_memory_pool().free_all_blocks()
    print(
        f"element p={Px[p]}{Py[p]}{Pz[p]} calc done"
    )
print(f"vvv done, time={time.perf_counter() - st_vvv:.0f}s")
print_mem_peak("after_vvv")


# ========== 收缩路径优化 ==========
Cgamma5 = cp.asarray(gamma(7))
vvv_shape = (Nev, Nev, Nev)

# DP 基下, 正宇称分量是旋量矩阵的前 2 个分量
# 结果只保留 r,q,z,y 的 0,1 分量, 这 4 个自旋指标所在轴截为 2, 减小计算量.
nucl_spin = "rqzy"
st_path = time.perf_counter()
contractions = []
for index, coeff in zip(NP_indexes, NP_coeff):
    peram_shapes = []
    peram_slices = []
    for block in index.split(","):
        cut_spin_sink = block[0] in nucl_spin
        cut_spin_source = block[1] in nucl_spin
        peram_shapes.append(
            (2 if cut_spin_sink else 4, 2 if cut_spin_source else 4, Nev, Nev)
        )
        peram_slices.append(
            (
                slice(0, 2) if cut_spin_sink else slice(None),
                slice(0, 2) if cut_spin_source else slice(None),
            )
        )

# 核子 1: 质子, 核子 2: 中子
    expression = contract_expression(
        index + ",mo,ace,np,bdf,uw,gik,vx,hjl->rqzy",
        *peram_shapes,
        Cgamma5,
        vvv_shape,  # vvv_src1
        Cgamma5,
        vvv_shape,  # vvv_snk1
        Cgamma5,
        vvv_shape,  # vvv_src2
        Cgamma5,
        vvv_shape,  # vvv_snk2
        constants=(6, 8, 10, 12),
        optimize="dp",
    )
    contractions.append((coeff, expression, peram_slices))
print(f"path compile done, time={time.perf_counter() - st_path:.0f}s "
      f"expressions={len(contractions)}")


# ========== 计算收缩 ==========
# DP 基下, 正宇称分量是旋量矩阵的前 2 个分量
# para: p1, p2, dt, t_source, spin_snk1, spin_src1, spin_snk2, spin_src2
corr_NP = cp.zeros((Nmom // 2, Nmom, N_dt, Nt, 2,
                   2, 2, 2), dtype=cp.complex128)

# para: r, q, z, y, 只保留 0,1 分量
tmpNP = cp.empty((2, 2, 2, 2), dtype=cp.complex128)

Path(result_dir).mkdir(parents=True, exist_ok=True)
result_file = f"{result_dir}/NPcorr_conf{conf_id}_P0.npy"

# if test, Nt -> 2
for t_src in range(Nt):
    st_src = time.perf_counter()

    # para: dt, spin_sink, spin_source, eig_sink_idx, eig_source_idx
    peram = read_peram_timeslice(
        light_peram_dir,
        conf_id,
        t_src,
        Nt,
        N_dt,
        Nev,
        representation="DP",
        xp=cp,
    )

    # source 的 vvv 会出现复用的情况, 减少计算量
    vvv_conj_dt = {}

    for dt in range(N_dt):
        t_snk = (t_src + dt) % Nt

        # p_snk1 表示 sink 的核子 1 (质子) 动量, 只考虑 x, y, z 方向的 0,1 动量的情况
        for p_snk1 in range(Nmom // 2):
            P_snk = 0 if p_snk1 == 0 else 1
            p_snk2 = (p_snk1 + Nmom // 2) % Nmom

            # p_src1 表示 source 的核子 1 (质子) 动量, 考虑 x, y, z 方向的 -1, 0, 1 动量的情况, 但不考虑 p1 > p2 的情况
            # 共计算 25 种情况
            for p_src1 in range(Nmom):

                if p_src1 == 4:
                    continue
                P_src = 0 if p_src1 == 0 else 1
                if P_snk > P_src:
                    continue

                # # 复用 source 的 vvv
                p_src2 = (p_src1 + Nmom // 2) % Nmom
                if p_src1 not in vvv_conj_dt:
                    vvv_conj_dt[p_src1] = cp.conj(vvv[p_src1, t_src])
                if p_src2 not in vvv_conj_dt:
                    vvv_conj_dt[p_src2] = cp.conj(vvv[p_src2, t_src])

                vvv_variables = (
                    vvv_conj_dt[p_src1],
                    vvv[p_snk1, t_snk],
                    vvv_conj_dt[p_src2],
                    vvv[p_snk2, t_snk],
                )
                tmpNP.fill(0)
                for coeff, expression, peram_slices in contractions:
                    variables = tuple(
                        peram[dt][slice_sink, slice_source]
                        for slice_sink, slice_source in peram_slices
                    ) + vvv_variables
                    tmpNP += coeff * expression(*variables, backend="cupy")

                corr_NP[p_snk1, p_src1, dt, t_src] += tmpNP

    print(f"t_src {t_src} done, time={time.perf_counter() - st_src:.0f}s")
    print_mem_peak(f"tsrc{t_src}_end")

    np.save(result_file, cp.asnumpy(corr_NP))
    print(f"saved t_src={t_src}: {result_file}")

print(f"saved corr_NP: {result_file} {corr_NP.shape}")
print(f"process done, time={time.perf_counter() - st_total:.0f}s")
