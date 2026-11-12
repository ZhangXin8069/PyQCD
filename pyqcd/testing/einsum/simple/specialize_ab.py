"""专用路径 A/B 对照：同一方程 ``specialize=True`` vs ``specialize=False``。

用法：
    python -m pyqcd.testing.einsum.simple.specialize_ab
    python -m pyqcd.testing.einsum.simple.specialize_ab --profile T1 --backend cuda \
        --equations E4,E5
    python -m pyqcd.testing.einsum.simple.specialize_ab --iters 20 --warmup 5 \
        --order offon
    python -m pyqcd.testing.einsum.simple.specialize_ab --list      # 已注册形态表

口径：同 profile、同 dtype、同设备、同 planner（``--optimize``，两侧各自规划一次），
每个变体先清空 ``pyqcd.einsum`` 的计划/编译上下文缓存，再预热 ``--warmup`` 次、
取 ``--iters`` 次中位耗时，只计稳态执行。

``specialize=True`` 命中专用路径时，库自身会打印 ``[pyqcd.einsum] 专用路径: ...``。
本工具把这些行以 ``[库自身输出]`` 前缀原样转发——那是**库**的日志，不是本工具的调试输出。

输出列：
    opt_ms     opt_einsum 的同方程中位耗时（倍率参照）
    on/off_ms  specialize=True / False 的中位耗时
    x_opt      opt_ms / on_ms、opt_ms / off_ms（>1 表示我们更快）
    gain       off_ms / on_ms（专用路径收益，>1 表示专用路径更快）
    rel        与 opt_einsum 结果的逐元素相对误差（均应 <= 1e-12）

注意：``spatial=1`` 下 E4/E5 通常在 <100 ms 量级，双方都低于 100 ms 的行标
``too_short``，**只作诊断**，不构成生产性能结论。
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import statistics

from . import _common

MIN_MEANINGFUL_SECONDS = 0.1
VARIANTS = (('on', True), ('off', False))


def subject_equations():
    """默认方程集：``_common`` 方程表中命中特化路径的那些。"""
    try:
        from pyqcd.einsum import _specialize
    except ImportError:  # pragma: no cover - 没有专用路径时退回全部方程
        return _common.equation_names()

    return [name for name in _common.equation_names()
            if _specialize.match(_common.EQUATION_BY_NAME[name]['eq']) is not None]


@contextlib.contextmanager
def library_output():
    """转发库自身的 stdout 提示并标明来源（本工具不是这些行的作者）。"""
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            yield
    finally:
        for line in buffer.getvalue().splitlines():
            print(f'[库自身输出] {line}', flush=True)


def reset_target_state():
    """丢弃已编译的计划与原生上下文，让下一个变体按当前开关重新编译。

    ``_NATIVE_CONTEXT_CACHE`` 的键不含 ``PYQCD_EINSUM_DISABLE_FUSED``；不清空的话
    ``specialize=False`` 会复用 ``specialize=True`` 编译出的（已匹配融合 kernel 的）
    上下文，A/B 就失去意义。
    """
    try:
        import pyqcd.einsum._api as api
    except ImportError:  # pragma: no cover - 私有模块改名时不影响计时
        return
    for name in ('_CONTRACT_PROGRAM_CACHE', '_NATIVE_CONTEXT_CACHE'):
        cache = getattr(api, name, None)
        if cache is not None and hasattr(cache, 'clear'):
            cache.clear()
    for name in ('_LAST_CONTRACT_SIGNATURE', '_LAST_CONTRACT_PROGRAM'):
        if hasattr(api, name):
            setattr(api, name, None)


def _reset_announcements():
    """允许下一轮重新打印专用路径提示（默认每形态每进程只打印一次）。"""
    try:
        from pyqcd.einsum import _specialize
    except ImportError:  # pragma: no cover
        return
    announced = getattr(_specialize, '_announced', None)
    if hasattr(announced, 'clear'):
        announced.clear()


def _cell(value, width, digits=3):
    if value is None:
        return f'{"-":>{width}s}'
    if value == float('inf'):
        return f'{"inf":>{width}s}'
    return f'{value:{width}.{digits}f}'


def _sci(value, width=9):
    if value is None:
        return f'{"-":>{width}s}'
    return f'{value:{width}.2e}'


def print_registry(limit=10):
    """打印新版特化路径（语料拓扑，唯一真源是 ``_specialize.describe()``）。"""
    from pyqcd.einsum import _specialize

    rows = _specialize.describe()
    print(f'{"name":22s} {"kind":18s} subscripts')
    print('-' * 110)
    for name, kind, subscripts in rows[:limit]:
        print(f'{name:22s} {kind:18s} {subscripts}')
    print(f'... 共 {len(rows)} 条（只列前 {min(limit, len(rows))} 条）')


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--profile', default=None,
                        help=f'profile 名（默认 {_common.DEFAULT_PROFILE}）')
    parser.add_argument('--backend', default='cpu', choices=('cpu', 'cuda'))
    parser.add_argument('--equations', default=None,
                        help='逗号列表；默认取方程表中命中专用路径的方程')
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--optimize', default='dp')
    parser.add_argument('--order', default='onoff', choices=('onoff', 'offon'),
                        help='两个变体的测量先后（默认 onoff）')
    parser.add_argument('--list', action='store_true',
                        help='只打印已注册形态表后退出')
    args = parser.parse_args(argv)

    if args.list:
        print_registry()
        return 0

    backend = args.backend
    if backend == 'cuda' and not _common.cuda_available():
        print('SKIP: CUDA 不可用（cupy 或设备缺失）')
        return 0
    os.environ.setdefault(
        'PYQCD_EINSUM_NATIVE', 'cuda' if backend == 'cuda' else 'kupl')

    names = (args.equations.split(',') if args.equations
             else subject_equations())
    order = VARIANTS if args.order == 'onoff' else VARIANTS[::-1]
    cfg = _common.profile_config(args.profile)
    oracle = _common.oracle_module()
    import pyqcd.einsum as target

    backend_name = 'cupy' if backend == 'cuda' else 'numpy'
    profile = args.profile or _common.DEFAULT_PROFILE
    print(f'profile={profile} backend={backend} dtype={_common.DTYPE} '
          f'iters={args.iters} warmup={args.warmup} '
          f'optimize={args.optimize} order={args.order} '
          f'nvec={cfg["nvec"]} spatial={cfg["spatial"]}')
    print('提示：以 [库自身输出] 开头的行由 pyqcd.einsum 自行打印'
          '（`[pyqcd.einsum] 专用路径: ...`），不是本工具的输出。')
    header = (f'{"eq":4s} {"opt_ms":>10s} {"on_ms":>10s} {"off_ms":>10s} '
              f'{"x_opt.on":>9s} {"x_opt.off":>9s} {"gain":>7s} '
              f'{"rel.on":>9s} {"rel.off":>9s}  note')
    print(header)
    print('-' * len(header))

    rows = []
    problems = []
    for name in names:
        meta = _common.EQUATION_BY_NAME.get(name)
        if meta is None:
            print(f'{name:4s} {_cell(None, 10)} {_cell(None, 10)} {_cell(None, 10)} '
                  f'{_cell(None, 9)} {_cell(None, 9)} {_cell(None, 7)} '
                  f'{_cell(None, 9)} {_cell(None, 9)}  未知方程')
            problems.append(f'{name}: 未知方程')
            continue

        arrays = _common.operands(name, cfg)
        host = _common.to_backend(arrays, backend)

        reference, reference_error = None, ''
        try:
            reference = _common.to_host(
                oracle.contract(meta['eq'], *host, optimize=args.optimize,
                                backend=backend_name), backend)
        except Exception as exc:  # noqa: BLE001 - oracle 失败不应中断 A/B
            reference_error = f'{type(exc).__name__}: {exc}'

        timings, errors, rel_errors = {}, {}, {}
        for label, flag in order:
            call = lambda flag=flag: target.contract(  # noqa: B023 - 显式绑定
                meta['eq'], *host, optimize=args.optimize,
                backend=backend_name, specialize=flag)
            try:
                with library_output():
                    value = call()
                    # 关键：先让本变体完成一次调用（决定融合开关的进程状态），
                    # 再丢弃已编译上下文，使预热在**正确的开关状态**下重新编译。
                    reset_target_state()
                    timings[label] = _common.median_seconds(
                        call, args.iters, args.warmup, backend)
                if reference is not None:
                    rel_errors[label] = _common.relative_error(
                        _common.to_host(value, backend), reference)
            except Exception as exc:  # noqa: BLE001 - 单侧失败只降级该侧
                errors[label] = f'{type(exc).__name__}: {exc}'
            _reset_announcements()

        opt_seconds = None
        if reference is not None:
            try:
                opt_seconds = _common.median_seconds(
                    lambda: oracle.contract(meta['eq'], *host,
                                            optimize=args.optimize,
                                            backend=backend_name),
                    args.iters, args.warmup, backend)
            except Exception as exc:  # noqa: BLE001
                reference_error = f'{type(exc).__name__}: {exc}'

        on_seconds, off_seconds = timings.get('on'), timings.get('off')
        x_on = (opt_seconds / on_seconds
                if opt_seconds and on_seconds else None)
        x_off = (opt_seconds / off_seconds
                 if opt_seconds and off_seconds else None)
        gain = (off_seconds / on_seconds
                if on_seconds and off_seconds else None)

        notes = []
        if errors and len(errors) == 2:
            notes.append('FAIL(两侧都无法执行) ' + '; '.join(
                f'{label}={message}' for label, message in errors.items()))
        elif errors:
            label, message = next(iter(errors.items()))
            notes.append(f'FAIL({label} 侧失败) {message}')
        if reference_error:
            notes.append(f'oracle 失败 {reference_error}')
        short = [value for value in (opt_seconds, on_seconds, off_seconds)
                 if value is not None]
        if short and min(short) < MIN_MEANINGFUL_SECONDS:
            notes.append('too_short(仅诊断)')
        for label, value in rel_errors.items():
            if value > _common.RELATIVE_TOLERANCE:
                notes.append(f'NUMERIC-FAIL({label}) rel={value:.2e}')
        problems.extend(f'{name}: {item}' for item in notes
                        if not item.startswith('too_short'))
        note = '; '.join(notes)

        print(f'{name:4s} {_cell(opt_seconds * 1e3 if opt_seconds else None, 10)} '
              f'{_cell(on_seconds * 1e3 if on_seconds else None, 10)} '
              f'{_cell(off_seconds * 1e3 if off_seconds else None, 10)} '
              f'{_cell(x_on, 9)} {_cell(x_off, 9)} {_cell(gain, 7)} '
              f'{_sci(rel_errors.get("on"), 9)} '
              f'{_sci(rel_errors.get("off"), 9)}  {note}')
        rows.append({'eq': name, 'opt': opt_seconds, 'on': on_seconds,
                     'off': off_seconds, 'gain': gain})
        del arrays, host, reference

    print()
    print('说明：本表只含本次实测（profile=%s, backend=%s, iters=%d, warmup=%d）；'
          '不得跨 profile/后端外推。' % (profile, backend, args.iters, args.warmup))
    if args.backend == 'cuda':
        print('注意：CUDA 侧 specialize=False 通过 PYQCD_EINSUM_DISABLE_FUSED 让 '
              'initialize_context 跳过 match_sandwich/match_e4_fused；'
              '两个变体各自重新编译。')
    else:
        print('注意：CPU（KUPL）不读 PYQCD_EINSUM_DISABLE_FUSED（该变量只在 '
              'cpp/cuda/einsum/src/apply_contract.cu 的 initialize_context 中读取）。'
              'CPU 上 specialize=False 的差异来自 Python 侧：contract() 的编译程序缓存'
              '要求 specialize is not False（_api.py 的 cache_enabled），'
              '关闭后每次调用都重新规划。')
    gains = [row['gain'] for row in rows
             if row['gain'] is not None and row['eq'] not in
             {problem.split(':')[0] for problem in problems}]
    if gains:
        print('专用路径收益 gain=off/on：最小 %.3f、中位 %.3f、最大 %.3f（%d 条方程）'
              % (min(gains), statistics.median(gains), max(gains), len(gains)))
    if problems:
        print('\n失败项：')
        for problem in problems:
            print(f'  {problem}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
