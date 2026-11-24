import cupy as cp
import numpy as np
import time
from lqcddb_new import run_wick_analysis, set_backend, get_backend

set_backend('cupy')
backend = get_backend()
dtype = backend.complex128


shape_2pt = [[4,4,100,100],[4,4,100,100],[4,4,100,100],[4,4,100,100],[4,4,100,100],[4,4],[4,4],[4,4],[4,4],[1, 100,100],[1, 100,100,100],[1, 100,100],[1, 100,100,100]]
arr = [backend.asarray(x, dtype = dtype) for x in [backend.random.random((x)) + 1j * backend.random.random((x)) for x in shape_2pt]]

wick_2pt = run_wick_analysis(
    operator_groups = [(["|", "c^d", "gamma_M", "d", "|", "|", "u", "u", "C * gamma_B", "d", "|"], ["|", "d^d", "gamma_M", "c", "|", "|", "u^d", "C * gamma_B", "d^d", "u^d", "|"])],
    Cpt = '2pt',
    Vindex = ['N', 'N', 'M', 'M']
)

shape_3pt = [[4,4,100,100],[4,4,100,100],[4,4,100,100],[4,4,100,100],[4,4,100,100],[4,4],[4,4],[4,4],[4,4],[2, 100,100,100],[2, 100,100],[2, 100,100],[2, 100,100,100]]
arr = [backend.asarray(x, dtype = dtype) for x in [backend.random.random((x)) + 1j * backend.random.random((x)) for x in shape_3pt]]

wick_3pt = run_wick_analysis(
    operator_groups = [(["|", "d", "u", "C * gamma_B", "d", "|"], ["|", "d^d", "gamma_M", "u", "|", "|", "u^d", "C * gamma_B", "d^d", "d^d", "|"], ["|", "u^d", "gamma_M", "d", "|"])],
    Cpt = '3pt',
    Vindex = ['N', 'N', 'M', 'M']
)

shape_4pt = [[4,4,100,100],[4,4,100,100],[4,4,100,100],[4,4,100,100],[4,4,100,100],[4,4,100,100], [4,4],[4, 4,4],[4, 4,4],[4,4],[4,4], [1, 100,100,100],[9, 100,100],[9, 100,100],[1, 100,100],[1, 100,100,100]]
arr = [backend.asarray(x, dtype = dtype) for x in [backend.random.random((x)) + 1j * backend.random.random((x)) for x in shape_4pt]]

wick_4pt = run_wick_analysis(
    operator_groups = [(["|", "u", "u", "C * gamma_B", "d", "|"], ["|", "d^d", "gamma_M", "u", "|", "|", "u^d", "C * gamma_B", "d^d", "d^d", "|"], ["|", "u^d", "gamma_M", "d", "|", "|", "u^d", "gamma_M", "d", "|"])],
    Cpt = '3pt',
    Vindex = ['K', 'N', 'N', 'M', 'M'],
    Gindex = ['', 'G', 'G', '', ''],
)

# print(wick)
wick_indx = wick_4pt[2][2]['result_indx']
# print(wick_indx[0])
# exit()
# print(wick_indx[0])
# exit()
# print('opt_einsum contract')
# from opt_einsum import contract
# for i in range(20):
#     if backend.__name__ == 'cupy':
#         backend.cuda.runtime.deviceSynchronize()
#     A = time.perf_counter()
#     for wick_index in wick_indx:
#         optr = contract(*wick_index, *arr, optimize = 'dp')
#     if backend.__name__ == 'cupy':
#         backend.cuda.runtime.deviceSynchronize()
#     B = time.perf_counter()
#     if i > 10:
#         print(B - A)
        
# print('lqcddb cached_contract')
# from lqcddb import cached_contract as contract
# for i in range(10):
#     if backend.__name__ == 'cupy':
#         backend.cuda.runtime.deviceSynchronize()
#     A = time.perf_counter()
#     for wick_index in wick_indx:
#         lccr = contract(*wick_index, *arr, optimize = ['dp', 'auto', 'greedy'])
#     if backend.__name__ == 'cupy':
#         backend.cuda.runtime.deviceSynchronize()
#     B = time.perf_counter()
#     if i > 2:
#         print(B - A)

print('lqcddb_new cached_contract')
from lqcddb_new import cached_contract as contract
for i in range(20):
    if backend.__name__ == 'cupy':
        backend.cuda.runtime.deviceSynchronize()
    A = time.perf_counter()
    for wick_index in wick_indx:
        lccr = contract(*wick_index, *arr)
    if backend.__name__ == 'cupy':
        backend.cuda.runtime.deviceSynchronize()
    B = time.perf_counter()
    if i > 10:
        print(B - A)

print('pyqcd')
from pyqcd.einsum import contract
for i in range(20):
    if backend.__name__ == 'cupy':
        backend.cuda.runtime.deviceSynchronize()
    A = time.perf_counter()
    for wick_index in wick_indx:
        pycr = contract(*wick_index, *arr)
    if backend.__name__ == 'cupy':
        backend.cuda.runtime.deviceSynchronize()
    B = time.perf_counter()
    if i > 10:
        print(B - A)

print('lqcddb')
from lqcddb.base import cached_contract as contract
for i in range(20):
    if backend.__name__ == 'cupy':
        backend.cuda.runtime.deviceSynchronize()
    A = time.perf_counter()
    for wick_index in wick_indx:
        cacr = contract(*wick_index, *arr, optimize = ['dp', 'auto', 'greedy'])
    if backend.__name__ == 'cupy':
        backend.cuda.runtime.deviceSynchronize()
    B = time.perf_counter()
    if i > 10:
        print(B - A)


