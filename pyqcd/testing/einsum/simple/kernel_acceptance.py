"""专用 kernel 验收：正确性 + 与 opt_einsum / 现有通用路径的三方性能对照。

判定规则（用户口径）：
    kernel 慢于 opt_einsum  → **BUG**（不合格）
    kernel 慢于现有通用路径但快于 opt → OK-WITH-CAVEAT（可用但不该接线）
    kernel 快于现有通用路径 → PASS

kernel 源码若仍为 `.wip` 后缀，视为未完成，直接 SKIP。

用法：
    python -m pyqcd.testing.einsum.simple.kernel_acceptance
    python -m pyqcd.testing.einsum.simple.kernel_acceptance --nvec 50
    python -m pyqcd.testing.einsum.simple.kernel_acceptance --only kupl_nucleon
"""

from __future__ import annotations

import argparse
import ctypes
import os
import statistics
import subprocess
import tempfile
import time

import numpy as np

from . import _common

REPO = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
QCD_COMPLEX128 = 0

KERNELS = {
    'kupl_nucleon': {
        'source': 'cpp/kupl/einsum/src/nucleon2pt.cpp',
        'include': 'cpp/kupl/einsum/include',
        'symbol': 'qcd_kupl_nucleon_2pt',
        'compiler': 'g++',
        'flags': ['-O3', '-shared', '-fPIC', '-std=c++17', '-fopenmp',
                  '-march=native'],
    },
    'cuda_nucleon': {
        'source': 'cpp/cuda/einsum/src/nucleon2pt.cu',
        'include': 'cpp/cuda/einsum/include',
        'symbol': 'qcd_cuda_nucleon_2pt',
        'compiler': 'nvcc',
        'flags': ['-O3', '-shared', '-std=c++17', '-arch=sm_89',
                  '-Xcompiler', '-fPIC'],
        'cuda': True,
    },
    'cuda_3pt': {
        'source': 'cpp/cuda/einsum/src/pattern_3pt.cu',
        'include': 'cpp/cuda/einsum/include',
        'symbol': 'qcd_cuda_pattern_3pt',
        'compiler': 'nvcc',
        'flags': ['-O3', '-shared', '-std=c++17', '-arch=sm_89',
                  '-Xcompiler', '-fPIC'],
        'cuda': True,
    },
    'kupl_3pt': {
        'source': 'cpp/kupl/einsum/src/pattern_3pt.cpp',
        'include': 'cpp/kupl/einsum/include',
        'symbol': 'qcd_kupl_pattern_3pt',
        'compiler': 'g++',
        'flags': ['-O3', '-shared', '-fPIC', '-std=c++17', '-fopenmp',
                  '-march=native'],
    },
}


def _prebuilt_library(spec):
    """优先复用仓库已构建的库：它才是生产实际加载的二进制。

    用 nvcc 现编的独立 .so 缺少仓库构建脚本里的
    ``-fvisibility=hidden`` 与 ``-lcublas``/``-lcudart`` 链接参数，
    实测性能与生产库可差 1.8x，用它做验收会得出错误结论。
    """
    if not spec.get('cuda'):
        return None
    candidate = os.path.join(REPO, 'cpp', 'cuda', 'einsum', 'build',
                             'libpyqcdeinsum_cuda.so')
    if not os.path.isfile(candidate):
        return None
    try:
        import subprocess

        defined = subprocess.run(['nm', '--defined-only', candidate],
                                 capture_output=True, text=True).stdout
    except OSError:
        return None
    return candidate if spec['symbol'] in defined else None


def _build(spec, outdir):
    prebuilt = _prebuilt_library(spec)
    if prebuilt is not None:
        return prebuilt, ''
    source = os.path.join(REPO, spec['source'])
    wip = source + '.wip'
    if not os.path.isfile(source):
        if os.path.isfile(wip):
            return None, f'仍为 .wip（未完成）: {os.path.basename(wip)}'
        return None, f'源文件不存在: {spec["source"]}'
    target = os.path.join(outdir, f'lib{os.path.basename(source)}.so')
    command = [spec['compiler'], *spec['flags'],
               f'-I{os.path.join(REPO, spec["include"])}', source, '-o', target]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        return None, f'编译失败: {result.stderr.strip().splitlines()[:1]}'
    return target, ''


def _random(shape, rng):
    return np.ascontiguousarray(
        (rng.standard_normal(shape)
         + 1j * rng.standard_normal(shape)).astype(np.complex128))


def _median(fn, iterations=10, warmup=2):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def _nucleon_case(nvec, rng):
    eq = 'gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M'
    mom = 1
    peram = _random((4, 4, nvec, nvec), rng)
    gamma = _random((4, 4), rng)
    vvv1 = _random((mom, nvec, nvec, nvec), rng)
    vvv2 = _random((mom, nvec, nvec, nvec), rng)
    projector = np.ascontiguousarray(np.eye(4).astype(np.complex128))
    operands = [peram, peram, peram, gamma, gamma, vvv1, vvv2, projector]
    kupl_dims = (nvec, mom, 4, 0, 0, 0, 0, 0)
    # CUDA 契约：dims=[N,mx,my,mode,permA,permB,permC,permV1,permV2,matflags]，恒等=123
    cuda_dims = (nvec, mom, mom, 0, 123, 123, 123, 123, 123, 0)
    return eq, operands, kupl_dims, cuda_dims, 0


def _three_pt_case(nvec, rng):
    eq = ('agbh,eqfr,ospt,cmdn,ikjl,ce,Gmo,gi,kq,'
          'Nbdf,NLnp,Mhj,Mlrt,ax,sy->MNGLxy')
    mom, link = 1, 2
    perams = [_random((4, 4, nvec, nvec), rng) for _ in range(5)]
    operands = perams + [
        _random((4, 4), rng), _random((4, 4, 4), rng),
        _random((4, 4), rng), _random((4, 4), rng),
        _random((mom, nvec, nvec, nvec), rng),
        _random((mom, link, nvec, nvec), rng),
        _random((mom, nvec, nvec), rng),
        _random((mom, nvec, nvec, nvec), rng),
        np.ascontiguousarray(_random((4, 4), rng)[:, :2]),
        np.ascontiguousarray(_random((4, 4), rng)[:, :2]),
    ]
    # KUPL: (nvec, mom_src, mom_sink, link, dirac, proj, -, -)
    kupl_dims = (nvec, mom, mom, link, 4, 2, 0, 0)
    # CUDA: (nev, mom, link, spin, proj, -, -, -)
    cuda_dims = (nvec, mom, link, 4, 2, 0, 0, 0)
    return eq, operands, kupl_dims, cuda_dims, 0


CASES = {
    'kupl_nucleon': _nucleon_case,
    'cuda_nucleon': _nucleon_case,
    'kupl_3pt': _three_pt_case,
    'cuda_3pt': _three_pt_case,
}

# CUDA kernel 的最后一个参数是 stream 句柄。
CUDA_STREAM_ARG = True


def evaluate(name, spec, nvec, iterations=10):
    oracle = _common.oracle_module()
    import pyqcd.einsum as target

    outdir = tempfile.mkdtemp(prefix='einsum-kernel-')
    library_path, problem = _build(spec, outdir)
    if library_path is None:
        print(f'SKIP {name}: {problem}')
        return None
    library = ctypes.CDLL(library_path)
    builder = CASES[name]
    rng = np.random.default_rng(20261108)
    eq, operands, kupl_dims, cuda_dims, variant = builder(nvec, rng)
    kernel_operands = operands
    if spec.get('cuda') and name.endswith('nucleon'):
        # CUDA 契约顺序：{P1,P2,P3,G1,G2,pr,V1,V2}（投影在 V 之前）；
        # reference 仍用 einsum 顺序计算。
        peram_a, peram_b, peram_c, g1, g2, v1, v2, pr = operands
        kernel_operands = [peram_a, peram_b, peram_c, g1, g2, pr, v1, v2]
    dims = (ctypes.c_int64 * 10)(*(cuda_dims if spec.get('cuda') else kupl_dims))

    function = getattr(library, spec['symbol'])
    function.restype = ctypes.c_int
    cuda = bool(spec.get('cuda'))
    if cuda and name.endswith('nucleon'):
        function.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p),
                             ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64),
                             ctypes.c_void_p]
    elif cuda:
        function.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p),
                             ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64),
                             ctypes.c_int, ctypes.c_void_p]
    else:
        function.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p),
                             ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64),
                             ctypes.c_int]

    def invoke():
        if cuda:
            cupy.cuda.Stream.null.synchronize()
            if name.endswith('nucleon'):
                status_value = function(QCD_COMPLEX128, pointers,
                                        output_pointer, dims, None)
            else:
                status_value = function(QCD_COMPLEX128, pointers,
                                        output_pointer, dims, variant, None)
            cupy.cuda.Stream.null.synchronize()
            output[...] = cupy.asnumpy(output_device)
            return status_value
        return function(QCD_COMPLEX128, pointers, output_pointer, dims,
                        variant)

    reference = oracle.contract(eq, *operands, optimize='dp')
    if cuda:
        # CUDA kernel 契约要求**设备**指针（见 include/nucleon2pt.h）。
        import cupy

        device_operands = [cupy.asarray(operand) for operand in kernel_operands]
        output_device = cupy.zeros(reference.shape, dtype=cupy.complex128)
        pointers = (ctypes.c_void_p * len(device_operands))(
            *[int(operand.data.ptr) for operand in device_operands])
        output_pointer = int(output_device.data.ptr)
        output = np.zeros(reference.shape, dtype=np.complex128)
    else:
        device_operands = None
        output = np.zeros(reference.shape, dtype=np.complex128)
        output_pointer = output.ctypes.data
        pointers = (ctypes.c_void_p * len(kernel_operands))(
            *[operand.ctypes.data for operand in kernel_operands])
    status = invoke()
    error = _common.relative_error(output, reference)

    kernel_seconds = _median(invoke, iterations)
    oracle_seconds = _median(
        lambda: oracle.contract(eq, *operands, optimize='dp'), iterations)
    # 必须显式 specialize=False：默认路径一旦接线就会走 kernel，
    # 否则这里测到的"generic"其实是 kernel 自己，对照失去意义。
    generic_seconds = _median(
        lambda: target.contract(eq, *operands, optimize='dp',
                                specialize=False), iterations)

    if status != 0 or error > _common.RELATIVE_TOLERANCE:
        verdict = 'BUG（数值/返回码不合格）'
    elif kernel_seconds >= oracle_seconds:
        verdict = 'BUG（慢于 opt_einsum）'
    elif kernel_seconds >= generic_seconds:
        verdict = 'OK-WITH-CAVEAT（快于 opt，慢于通用路径，不应接线）'
    else:
        verdict = 'PASS（快于通用路径与 opt_einsum）'

    print(f'\n[{name}] nvec={nvec} status={status} rel={error:.2e}')
    print(f'  kernel  = {kernel_seconds * 1e3:9.2f} ms')
    print(f'  generic = {generic_seconds * 1e3:9.2f} ms '
          f'(kernel/generic = {generic_seconds / kernel_seconds:.2f}x)')
    print(f'  opt     = {oracle_seconds * 1e3:9.2f} ms '
          f'(kernel/opt = {oracle_seconds / kernel_seconds:.2f}x)')
    print(f'  → {verdict}')
    return verdict


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nvec', type=int, default=50)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--only', default=None, help='逗号列表')
    args = parser.parse_args(argv)

    names = (args.only.split(',') if args.only else list(KERNELS))
    verdicts = {}
    for name in names:
        spec = KERNELS.get(name)
        if spec is None:
            print(f'SKIP {name}: 未知 kernel')
            continue
        verdicts[name] = evaluate(name, spec, args.nvec, args.iterations)

    bugs = [name for name, verdict in verdicts.items()
            if verdict and verdict.startswith('BUG')]
    print(f'\n结论：{len(bugs)} 个不合格' + (f'：{bugs}' if bugs else ''))
    return 1 if bugs else 0


if __name__ == '__main__':
    raise SystemExit(main())
