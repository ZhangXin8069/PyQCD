#!/public/home/huangcl/.venv3_10/bin/python
# 功能: 用同一组随机复数张量比较 pyqcd 和 opt_einsum 的 CPU 收缩速度及数值结果.
# 输入: 文件开头的 contract_path, 按 peram, gamma, vvv 的顺序排列操作数.
#       也可以在命令行用第一个参数覆盖, 例如 python compare_contract.py "eufv,gkhl,ce,ik->aw".
# 输出: 终端打印 opt_einsum (optimize="dp") 和 pyqcd (默认 optimize) 各 n_run 次收缩的耗时,
#       后三次耗时的中位数加速比, 以及 compare_data 的总比较结果.

from opt_einsum import contract as opt_contract
import numpy as np
import sys

sys.path.append("/public/home/huangcl/06_deuteron_gluon_pdf/98_tools")

from comparison_data_tools import compare_data  # noqa: E402  # type: ignore


peram_shape = (4, 4, 100, 100)
gamma_shape = (4, 4)
vvv_shape = (100, 100, 100)
n_run = 5


def gen_operands(equation: str, dtype: type,
                 rng: np.random.Generator) -> list[np.ndarray]:
    """
    按收缩路径生成随机复数张量, 操作数顺序为先 peram, 再 gamma, 最后 vvv.

    参数:
    -----
    equation: einsum 收缩路径.
    dtype: 随机张量的数据精度.
    rng: 随机数生成器.

    输出:
    -----
    operands: 与收缩路径对应的随机复数张量列表.
    """

    terms = equation.split("->", 1)[0].split(",")
    n_peram = sum(len(term) == 4 for term in terms)
    n_gamma = sum(len(term) == 2 for term in terms)
    n_vvv = len(terms) - n_peram - n_gamma

    # para: each peram: sink_spin, source_spin, sink_eigen, source_eigen
    peram = [
        (rng.random(peram_shape) + 1j * rng.random(peram_shape)).astype(dtype)
        for _ in range(n_peram)
    ]
    # para: each gamma: sink_spin, source_spin
    gamma = [
        (rng.random(gamma_shape) + 1j * rng.random(gamma_shape)).astype(dtype)
        for _ in range(n_gamma)
    ]
    # para: each vvv: sink_eigen, middle_eigen, source_eigen
    vvv = [
        (rng.random(vvv_shape) + 1j * rng.random(vvv_shape)).astype(dtype)
        for _ in range(n_vvv)
    ]
    return [*peram, *gamma, *vvv]


def check_precision(contract_path: str) -> None:
    """分别以 complex64 和 complex128 计算并比较每轮结果."""

    print(f'path:\n    "{contract_path}"\n')
    rng = np.random.default_rng()

    for run in range(n_run):
        print(f"run {run + 1}/{n_run}")
        operands128 = gen_operands(contract_path, np.complex128, rng)
        operands64 = [operand.astype(np.complex64) for operand in operands128]

        result64 = opt_contract(contract_path, *operands64, optimize="dp")
        result128 = opt_contract(contract_path, *operands128, optimize="dp")
        compare_data([(result128, result64, "precision")])
        print("----------")
