"""
功能: 计算质子的 2pt, 使用动量涂抹.
输入: 组态号.
输出: 01_reslut/<conf_id>/ 下的积分结果.
"""
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np
from opt_einsum import contract

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "98_tools"))

# noqa: E402 抑制 autopep8 的 import 顺序检查
from distillation_tools import (  # noqa: E402  # type: ignore
    calc_vvv,
    read_peram_timeslice,
)
from profiling_tools import (  # noqa: E402  # type: ignore
    print_mem_peak,
    print_var_size,
)
from gamma_matrix_DR import gamma  # noqa: E402  # type: ignore


conf_id = sys.argv[1]
Nt = 72
Nx = 24
Nev1 = 100
Px = -3
Py = 0
Pz = 0
dt_max = 30

eig_dir = (
    "/public/group/lqcd/eigensystem/"
    f"beta6.20_mu-0.2770_ms-0.2400_L24x72/{conf_id}/"
)
peram_dir = (
    "/public/home/sunp/sunpeng_public/mom_smear_perambulators/"
    "beta6.20_mu-0.2770_ms-0.2400_L24x72/output_dir_data/"
    f"mz0_my0_mx2/{conf_id}/"
)
output_dir = Path(__file__).resolve().parent / "01_result" / str(conf_id)
output_dir.mkdir(parents=True, exist_ok=True)

matrix_pplus = cp.asarray(0.5 * (gamma(0) + gamma(4)))
cg5g4 = cp.asarray(gamma(7) @ gamma(4))


def contract_proton(vvv_sink, vvv_source, peram, cg5_peram_cg5):
    result = contract(
        "abc,gjad,gjbe,ilcf,def->il",
        vvv_sink,
        peram,
        cg5_peram_cg5,
        peram,
        vvv_source,
    )
    result -= contract(
        "abc,glaf,gjbe,ijcd,def->il",
        vvv_sink,
        peram,
        cg5_peram_cg5,
        peram,
        vvv_source,
    )
    return result


vvv = calc_vvv(
    eig_dir,
    conf_id,
    Nt,
    Nx,
    Nev1,
    (Px, Py, Pz),
    smear_momentum=(3, 0, 0),
    xp=cp,
)

contrac_nucl_matrix = cp.zeros((Nt, Nt, 4, 4), dtype=complex)
print("\n========== start 2pt contraction ==========")
t_start_all = time.time()
for t_source in range(Nt):
    t_start = time.time()
    vvv_source = cp.conj(vvv[t_source])
    # para: dt, spin_sink, spin_source, eig_sink_idx, eig_source_idx
    peram = read_peram_timeslice(
        peram_dir,
        conf_id,
        t_source,
        Nt,
        dt_max,
        Nev1,
        xp=cp,
    )

    cg5_peram_cg5 = contract("gh,thkbe,jk->tgjbe", cg5g4, peram, cg5g4)

    for dt in range(peram.shape[0]):
        t_sink = (dt + t_source) % Nt
        contrac_nucl_matrix[dt, t_source] = contract_proton(
            vvv[t_sink],
            vvv_source,
            peram[dt],
            cg5_peram_cg5[dt],
        )
    cp._default_memory_pool.free_all_blocks()
    cp.cuda.Device().synchronize()
    print(
        f"t_source={t_source} done, time used: {time.time() - t_start:.3f} s")

print(peram_dir)
print(
    f"2pt contraction done, total time used: {time.time() - t_start_all:.3f} s")
print("========== end 2pt contraction ==========\n")

print_mem_peak("2pt end", use_gpu=True)
print_var_size("2pt end", peram=peram)

# para: dt, t_source
contrac_nucl_pp = contract("li,yxil->yx", matrix_pplus, contrac_nucl_matrix)
output_file = output_dir / "momsmear_pp_Px3Py0Pz0.npy"
np.save(output_file, cp.asnumpy(contrac_nucl_pp))
print(f"saved: {output_file} {contrac_nucl_pp.shape}")
