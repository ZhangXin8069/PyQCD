import statistics
import sys
import time
from functools import partial

import numpy as np
import opt_einsum as oe
import pyqcd
import pyqcd.einsum as pe

# docs-skip: 15 个用例的全尺寸基准（单次 0.4–2.7 s），文档门禁不跑
DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cpu"     # "cpu" 或 "gpu"
ROUNDS = 3                                               # 交错计时轮数

if DEVICE == "gpu":
    import cupy as cp

    def to_device(array):
        """把 numpy 数组搬到显存。"""
        return cp.asarray(array)

    def synchronise():
        """GPU 计时前必须同步，否则只量到 kernel 下发的时间。"""
        cp.cuda.Device(0).synchronize()

    def release_memory():
        """每个用例后清空显存池，避免上一次的残留把下一个用例挤爆。"""
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    device_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
else:
    def to_device(array):
        return array

    def synchronise():
        return None

    def release_memory():
        return None

    device_name = "cpu"

# 先打印版本与路径：site-packages 里的旧版会把每个数字都变成"PyQCD 很慢"
print("pyqcd", pyqcd.__version__, pyqcd.__file__, "|", device_name)

rng = np.random.default_rng(0)


def random_array(*shape):
    """实数随机数组（float64）。"""
    return to_device(rng.standard_normal(shape))


def random_complex_array(*shape):
    """复数随机数组（complex128）：实部与虚部独立采样。"""
    real_part = rng.standard_normal(shape)
    imag_part = rng.standard_normal(shape)
    return to_device((real_part + 1j * imag_part).astype(np.complex128))


def one_call(function, repeat=1):
    """调用 repeat 次并返回最快的一次；repeat>1 用来摊掉固定开销。"""
    synchronise()
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        synchronise()
        best = min(best, time.perf_counter() - start)
    return best


# 15 个结构类别，形状与上面两张表完全一致。
# 每行 = (标签, 方程, 造数据的函数, 各操作数的形状)
CASES = [
    ("matmul 2D (real)", "ij,jk->ik", random_array,
     [(4096, 4096), (4096, 4096)]),
    ("matmul 2D (complex)", "ij,jk->ik", random_complex_array,
     [(2048, 2048), (2048, 2048)]),
    ("element-wise", "ij,ij->ij", random_array,
     [(12000, 12000), (12000, 12000)]),
    ("outer product", "i,j->ij", random_array,
     [(8000,), (8000,)]),
    ("full reduction", "ij,ij->", random_array,
     [(12000, 12000), (12000, 12000)]),
    ("partial reduction", "ij->i", random_array,
     [(16000, 16000)]),
    ("chain 3 ops", "ij,jk,kl->il", random_array,
     [(3072, 3072), (3072, 3072), (3072, 3072)]),
    ("chain 4 ops", "ij,jk,kl,lm->im", random_array,
     [(2048, 2048), (2048, 2048), (2048, 2048), (2048, 2048)]),
    ("hyper-edge", "ij,jk,ik->i", random_array,
     [(2048, 2048), (2048, 2048), (2048, 2048)]),
    ("batched GEMM", "bij,bjk->bik", random_array,
     [(16, 1024, 1024), (16, 1024, 1024)]),
    ("ellipsis", "...ij,...jk->...ik", random_array,
     [(256, 512, 512), (256, 512, 512)]),
    ("batch x matrix", "abc,cd->abd", random_array,
     [(8, 1024, 512), (512, 1024)]),
    ("many small batches", "bij,bjk->bik", random_array,
     [(4096, 64, 64), (4096, 64, 64)]),
    ("nucleon 2pt (kernel)", "gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M",
     random_complex_array,
     [(4, 4, 112, 112), (4, 4, 112, 112), (4, 4, 112, 112), (4, 4), (4, 4),
      (4, 112, 112, 112), (4, 112, 112, 112), (4, 4)]),
    ("gamma sandwich (kernel)", "ab,...bce,cd->...dae", random_complex_array,
     [(4, 4), (8192, 4, 4, 1024), (4, 4)]),
]

print(f"{'case':26s} {'pyqcd(s)':>9s} {'opt(s)':>9s} {'ratio':>6s}")
for label, equation, builder, shapes in CASES:
    operands = [builder(*shape) for shape in shapes]

    # 1) 先验正确性：与 opt_einsum 逐元素对照，快而不对的数字没有意义
    reference = oe.contract(equation, *operands, optimize="dp")
    got = pe.contract(equation, *operands)
    ours_value = cp.asnumpy(got) if DEVICE == "gpu" else np.asarray(got)
    their_value = cp.asnumpy(reference) if DEVICE == "gpu" else np.asarray(reference)
    scale = max(float(np.max(np.abs(their_value))), 1e-30)
    assert np.max(np.abs(ours_value - their_value), initial=0.0) <= 1e-10 * scale, label
    del reference, got, ours_value, their_value
    release_memory()

    # 2) 先探一次耗时，据此决定内层重复次数，让每个样本至少约 0.3 s
    ours_call = partial(pe.contract, equation, *operands)
    theirs_call = partial(oe.contract, equation, *operands, optimize="dp")
    probe = one_call(ours_call)
    inner = max(1, min(9, int(0.3 / max(probe, 1e-6))))

    # 3) 交错计时：每轮交换先后顺序，机器漂移在比值里互相抵消
    ours_times, theirs_times = [], []
    for round_index in range(ROUNDS):
        if round_index % 2 == 0:
            ours_times.append(one_call(ours_call, inner))
            theirs_times.append(one_call(theirs_call, inner))
        else:
            theirs_times.append(one_call(theirs_call, inner))
            ours_times.append(one_call(ours_call, inner))
        release_memory()

    # 4) 取中位数再相除，避免被单次抖动带偏
    median_ours = statistics.median(ours_times)
    median_theirs = statistics.median(theirs_times)
    print(f"{label:26s} {median_ours:9.3f} {median_theirs:9.3f} "
          f"{median_theirs / median_ours:6.2f}")
    del operands
    release_memory()