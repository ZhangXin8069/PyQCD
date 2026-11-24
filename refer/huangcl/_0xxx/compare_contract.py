#!/public/home/huangcl/.venv3_10/bin/python
# 功能: 用同一组随机复数张量比较 pyqcd 和 opt_einsum 的 CPU 收缩速度及数值结果.
# 输入: 文件开头的 contract_path, 按 peram, gamma, vvv 的顺序排列操作数.
#       也可以在命令行用第一个参数覆盖, 例如 python compare_contract.py "eufv,gkhl,ce,ik->aw".
# 输出: 终端打印 opt_einsum (optimize="dp") 和 pyqcd (默认 optimize) 各 n_run 次收缩的耗时,
#       后三次耗时的中位数加速比, 以及 compare_data 的总比较结果.

from pyqcd.einsum import contract as pyqcd_contract
from pyqcd.einsum.test import print_tune, run_tune
from opt_einsum import contract as opt_contract
import numpy as np
import time
import sys

sys.path.append("/public/home/huangcl/06_deuteron_gluon_pdf/98_tools")

from comparison_data_tools import compare_data  # noqa: E402  # type: ignore


peram_shape = (4, 4, 100, 100)
gamma_shape = (4, 4)
vvv_shape = (100, 100, 100)
n_run = 11

# 命令行给出收缩路径时覆盖文件中的默认值, 便于 bash 调用.
# if len(sys.argv) > 1:
#     contract_path = sys.argv[1]


# def init_pyqcd(equation: str, operands: list[np.ndarray]) -> None:
#     """
#     调优当前方程对应的 pyqcd 收缩设置.

#     参数:
#     -----
#     equation: einsum 收缩路径.
#     operands: 与收缩路径对应的随机复数张量.
#     """

#     tune_result = run_tune(
#         equation=equation,
#         shapes=[operand.shape for operand in operands],
#         device="cpu",
#         dtype="complex64",
#         backend="numpy",
#         rounds=5,
#         warmup=0,
#         timeout=15.0,
#     )
#     print_tune(tune_result)


def gen_operands(equation: str) -> list[np.ndarray]:
    """
    按收缩路径生成随机复数张量, 操作数顺序为先 peram, 再 gamma, 最后 vvv.

    参数:
    -----
    equation: einsum 收缩路径.

    输出:
    -----
    operands: 与收缩路径对应的随机复数张量列表.
    """

    terms = equation.split("->", 1)[0].split(",")
    n_peram = sum(len(term) == 4 for term in terms)
    n_gamma = sum(len(term) == 2 for term in terms)
    n_vvv = len(terms) - n_peram - n_gamma
    rng = np.random.default_rng()

    # para: each peram: sink_spin, source_spin, sink_eigen, source_eigen
    peram = [
        (rng.random(peram_shape) + 1j * rng.random(peram_shape)).astype(np.complex64)
        for _ in range(n_peram)
    ]
    # para: each gamma: sink_spin, source_spin
    gamma = [
        (rng.random(gamma_shape) + 1j * rng.random(gamma_shape)).astype(np.complex64)
        for _ in range(n_gamma)
    ]
    # para: each vvv: sink_eigen, middle_eigen, source_eigen
    vvv = [
        (rng.random(vvv_shape) + 1j * rng.random(vvv_shape)).astype(np.complex64)
        for _ in range(n_vvv)
    ]
    return [*peram, *gamma, *vvv]


def check_path(contract_path):
    operands = gen_operands(contract_path)
    print(f"path:\n    \"{contract_path}\"\n")

    print(f"\nopt_einsum")
    opt_times = []
    for _ in range(n_run):
        started = time.perf_counter()
        opt_result = opt_contract(
            contract_path, *operands, optimize="dp")
        opt_time = time.perf_counter() - started
        opt_times.append(opt_time)

        print(f"{opt_time:.2f} s")

    print(f"\npyqcd")
    # init_pyqcd(contract_path, operands)
    pyqcd_times = []
    for _ in range(n_run):
        started = time.perf_counter()
        pyqcd_result = pyqcd_contract(contract_path, *operands)
        pyqcd_time = time.perf_counter() - started
        pyqcd_times.append(pyqcd_time)

        print(f"{pyqcd_time:.2f} s")

    compare_data([(pyqcd_result, opt_result, "contract")], show_details=True)
    opt_median = float(np.median(opt_times[1:]))
    pyqcd_median = float(np.median(pyqcd_times[1:]))
    print(f"speedup (opt_einsum / pyqcd): {opt_median / pyqcd_median:.2f}x")
    print(f"----------")
    return opt_median, pyqcd_median
