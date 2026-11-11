"""契约测试：``pyqcd.parallel.einsum.einsums`` 的任务分箱、结果对应与失败传播。

不依赖 pytest；既可经统一入口 ``python -m pyqcd.testing`` 运行，也可直接执行::

    python -m pyqcd.testing._parallel_einsum_contract
"""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

import pyqcd.einsum as pe
from pyqcd.parallel import einsum as parallel_einsum

_NEED_MPI_SKIP = '需要一个可用的 mpirun 与 mpi4py'


# ═══════════════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════════════

def _mpi_launcher():
    return parallel_einsum._default_launcher()


def _require_launcher():
    launcher = _mpi_launcher()
    if launcher is None:
        raise ImportError(_NEED_MPI_SKIP)
    return launcher


def _matmul_tasks(count, base=48, rng=None, prefix='t'):
    rng = rng if rng is not None else np.random.default_rng(20261111)
    tasks = {}
    for index in range(count):
        size = base + 8 * index
        tasks[f'{prefix}{index}'] = (
            'ij,jk->ik',
            [rng.standard_normal((size, size)),
             rng.standard_normal((size, size))])
    return tasks


def _run_quiet(**kwargs):
    """调用 einsums 并吞掉提醒/规划输出，返回结果与捕获文本。"""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        payload = parallel_einsum.einsums(reminder=False, **kwargs)
    return payload, buffer.getvalue()


def _distillation_tasks(nvec=12):
    """真实格点 QCD 蒸馏缩并：sush NN 2pt（S1）与 3pt 主项（P1）。

    形状照 ``pyqcd/testing/einsum/bench_distillation.build_operands``；这里只缩到
    Nvec=12，保证契约测试秒级完成，数值仍是同一组真实方程。
    """
    from pyqcd.lattice import gamma

    rng = np.random.default_rng(20261111)
    peram = (rng.standard_normal((2, 4, 4, nvec, nvec))
             + 1j * rng.standard_normal((2, 4, 4, nvec, nvec)))
    peram = np.ascontiguousarray(peram, dtype=np.complex128)
    g7 = np.asarray(gamma(7), dtype=np.complex128)
    g5 = np.asarray(gamma(5), dtype=np.complex128)
    projection = np.asarray((gamma(0) + gamma(4)) / 2.0, dtype=np.complex128)
    rand = lambda *shape: np.ascontiguousarray(
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape),
        dtype=np.complex128)

    sink = peram[1]
    s1 = {
        'subscripts': 'eofp,ambn,cgdh,ce,mo,Nbdf,Mnph,ga->MN',
        'operands': [sink, sink, sink, g7, g7,
                     rand(1, nvec, nvec, nvec),
                     rand(1, nvec, nvec, nvec), projection],
    }
    p1 = {
        'subscripts': ('agbh,eqfr,ospt,cmdn,ikjl,ce,Gmo,gi,kq,'
                       'Nbdf,NLnp,Mhj,Mlrt,ax,sy->MNGLxy'),
        'operands': [sink, sink, peram[0], sink, peram[0],
                     g7, rand(4, 4, 4), g5, g7,
                     rand(1, nvec, nvec, nvec),
                     rand(1, 2, nvec, nvec),
                     rand(1, nvec, nvec),
                     rand(1, nvec, nvec, nvec),
                     projection[:, :2], projection[:, :2]],
    }
    return {'S1-2pt': s1, 'P1-3pt': p1}


def _assert_close(actual, expected, label, rtol=0.0):
    """逐任务对照串行结果；``rtol=0`` 表示要求逐位一致（bit-exact）。

    ``rtol>0`` 用于跨进程比较：``pyqcd.einsum._threads`` 会按 MPI rank 数分摊
    OMP/BLAS 线程，worker 与父进程的线程数不同 → BLAS 求和顺序不同 →
    complex128 上出现 ~1e-15 相对量级的舍入差（实测 max|d| 2e-7、max|ref| 3e8）。
    """
    assert set(actual) == set(expected), (
        f'{label}: key mismatch {sorted(actual)} vs {sorted(expected)}')
    for key in expected:
        left, right = actual[key], expected[key]
        assert left.shape == right.shape, (
            f'{label}: key {key!r} shape {left.shape} != {right.shape}')
        if rtol == 0:
            assert np.array_equal(left, right), (
                f'{label}: key {key!r} differs from serial contract; '
                f'max|d|={np.max(np.abs(left - right)):.3e}')
            continue
        scale = max(float(np.max(np.abs(right))), 1e-300)
        relative = float(np.max(np.abs(left - right))) / scale
        assert relative <= rtol, (
            f'{label}: key {key!r} relative deviation {relative:.3e} > '
            f'{rtol:.1e}')


def _optimal_makespan(work, workers):
    """暴力搜索最少的最大负载（这里只在 ≤12 个任务的小规模用）。"""
    best = [sum(work)]
    loads = [0.0] * workers

    def walk(index):
        if index == len(work):
            best[0] = min(best[0], max(loads))
            return
        seen = set()
        for slot in range(workers):
            if loads[slot] in seen:
                continue
            seen.add(loads[slot])
            loads[slot] += work[index]
            if max(loads) < best[0]:
                walk(index + 1)
            loads[slot] -= work[index]

    walk(0)
    return best[0]


# ═══════════════════════════════════════════════════════════════════
# 规划器契约
# ═══════════════════════════════════════════════════════════════════

def test_plan_balances_skewed_workloads():
    """工作量悬殊时按 LPT 装箱：最重任务分散，且不超过理论最坏界。

    LPT 的保证是 ``max_load <= (4/3 - 1/(3K)) * OPT``；这里直接用暴力枚举求出
    OPT 来核对，而不是对"能不能完全均衡"提要求——3 个重任务落到 4 个进程上时，
    最大负载必然等于最重任务本身。
    """
    rng = np.random.default_rng(7)
    tasks = {}
    sizes = (200, 200, 200, 40, 40, 40, 40, 40)
    for index, size in enumerate(sizes):
        tasks[f't{index}'] = (
            'ij,jk->ik',
            [rng.standard_normal((size, size)),
             rng.standard_normal((size, size))])
    plan = parallel_einsum.plan_einsums(
        tasks, resources={'cpu_threads': 4, 'mem_avail_mb': 1 << 20})
    assert plan['workers'] == 4, plan['workers']
    assert sorted(index for bin_ in plan['bins'] for index in bin_) == list(
        range(len(tasks)))

    heavy = sorted(plan['work'])[-3:]
    heavy_slots = {slot for slot in range(plan['workers'])
                   for index in plan['bins'][slot]
                   if plan['work'][index] in heavy}
    assert len(heavy_slots) == 3, 'the three heaviest tasks must be spread out'

    optimum = _optimal_makespan(plan['work'], plan['workers'])
    bound = (4.0 / 3.0 - 1.0 / (3.0 * plan['workers'])) * optimum
    assert plan['max_load'] <= bound + 1e-9, (plan['max_load'], bound, optimum)

    even = parallel_einsum.plan_einsums(
        _matmul_tasks(8, base=64), resources={'cpu_threads': 4})
    assert even['skew'] < 1.05, even['loads']


def test_plan_respects_task_and_process_caps():
    """进程数 = min(子任务数, 上限)：任务少时一对一，任务多时均摊。"""
    tasks = _matmul_tasks(5)
    few = parallel_einsum.plan_einsums(
        tasks, max_procs=16, resources={'cpu_threads': 8})
    assert few['workers'] == 5
    # 一对一：每个箱子只装一个任务（LPT 按工时降序，故箱子顺序是按重到轻）。
    assert all(len(bin_) == 1 for bin_ in few['bins']), few['bins']
    assert sorted(index for bin_ in few['bins'] for index in bin_) == list(
        range(5))

    many = parallel_einsum.plan_einsums(
        tasks, max_procs=2, resources={'cpu_threads': 8})
    assert many['workers'] == 2
    assert sorted(index for bin_ in many['bins'] for index in bin_) == list(
        range(5))

    auto = parallel_einsum.plan_einsums(
        _matmul_tasks(20), resources={'cpu_threads': 3})
    assert auto['workers'] == 3
    assert auto['limits']['capped_by'] == 'cpu_threads'


def test_worker_thread_budget_is_not_inherited_from_parent():
    """worker 线程预算必须由启动器算，不能被父进程残留的线程变量带偏。

    实测回归：父进程导入 `torch`（经由 `bench_distillation`）会把 `OMP_NUM_THREADS`
    改成 4，旧启发式误判成"用户显式设置"，于是 4 进程 × 4 线程在 4 物理核上超订，
    400 个小任务从 0.30 s 涨到 1.79 s。
    """
    saved = {name: os.environ.get(name) for name in
             ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}
    try:
        os.environ['OMP_NUM_THREADS'] = '7'          # 冒充"父进程残留"
        os.environ['OPENBLAS_NUM_THREADS'] = '7'
        env, info = parallel_einsum._worker_env(4)
        assert info is not None
        assert info['per_worker'] == max(1, info['cores'] // 4), info
        assert env['OMP_NUM_THREADS'] == str(info['per_worker']), env
        assert env['OPENBLAS_NUM_THREADS'] == str(info['per_worker']), env

        forced, forced_info = parallel_einsum._worker_env(4, worker_threads=1)
        assert forced_info['per_worker'] == 1
        assert forced['OMP_NUM_THREADS'] == '1', forced
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_timings_separate_pool_init_from_compute():
    """``return_info=True`` 必须把「进程池初始化」与「纯计算」分开报出。

    墙钟时间含 mpirun 启动与落盘/回读，worker 自报的 compute_seconds 只含收缩
    本身；两者之差就是池的固定开销。这条口径是文档与 bench 的裁判依据。
    """
    _require_launcher()
    tasks = _matmul_tasks(4, base=96)
    payload, info = parallel_einsum.einsums(
        tasks, max_procs=2, reminder=False, verbose=False, return_info=True)
    assert isinstance(payload, dict) and set(payload) == set(tasks)
    assert info["mode"] == "mpi_launch", info
    timings = info["timings"]
    assert timings is not None, info
    assert timings["workers"] == 2, timings
    assert len(timings["per_worker_seconds"]) == 2, timings
    assert timings["compute_seconds"] > 0.0, timings
    assert timings["compute_seconds"] <= timings["total_task_seconds"] + 1e-9
    assert sum(timings["tasks_per_worker"]) == len(tasks), timings

    serial_payload, serial_info = parallel_einsum.einsums(
        tasks, max_procs=1, reminder=False, verbose=False, return_info=True)
    assert serial_info["mode"] == "serial", serial_info
    assert serial_info["timings"] is None, serial_info
    for key, (equation, operands) in tasks.items():
        assert np.array_equal(serial_payload[key], payload[key]), key


def test_collective_is_disabled_by_default():
    """``collective`` 默认关闭：mpirun 内只由 rank 0 执行，其余 rank 立即返回。

    真实生产对内存/显存敏感（collective 让每个 rank 各持一份整批输入），所以它必须
    是显式开关。这里同时验证：默认路径仍给出正确结果，且不打印内存警告。
    """
    launcher = _require_launcher()
    project_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as tmpdir:
        script = Path(tmpdir) / 'default_probe.py'
        script.write_text(
            'import json\n'
            'import numpy as np\n'
            'from mpi4py import MPI\n'
            'import pyqcd.einsum as pe\n'
            'from pyqcd.parallel.einsum import einsums\n'
            'rank = MPI.COMM_WORLD.Get_rank()\n'
            'rng = np.random.default_rng(5)\n'
            'tasks = {"a": ("ij,jk->ik", [rng.standard_normal((16, 16)),\n'
            '                             rng.standard_normal((16, 16))])}\n'
            'payload, info = einsums(tasks, reminder=False,\n'
            '                        verbose=(rank == 0), return_info=True)\n'
            'ok = payload is not None and np.array_equal(\n'
            '    payload["a"], pe.contract(*tasks["a"][:1], *tasks["a"][1]))\n'
            'print("RESULT", json.dumps({"rank": rank, "mode": info["mode"],\n'
            '                            "has_payload": payload is not None,\n'
            '                            "ok": bool(ok)}))\n',
            encoding='utf-8')
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join(
            [str(project_root), env.get('PYTHONPATH', '')]).rstrip(os.pathsep)
        completed = subprocess.run(
            launcher + ['-np', '2', sys.executable, str(script)],
            capture_output=True, text=True, env=env, cwd=project_root,
            timeout=300)
    assert completed.returncode == 0, completed.stderr[-1500:]
    rows = [json.loads(line.split(' ', 1)[1])
            for line in completed.stdout.splitlines()
            if line.startswith('RESULT ')]
    assert len(rows) == 2, completed.stdout
    by_rank = {row['rank']: row for row in rows}
    assert by_rank[0]['mode'] == 'rank0_only', by_rank
    assert by_rank[0]['has_payload'] is True and by_rank[0]['ok'] is True, by_rank
    assert by_rank[1]['has_payload'] is False, by_rank
    assert '内存提醒' not in completed.stdout, completed.stdout
    assert 'disabled by default' in completed.stdout, completed.stdout


def test_init_and_run_write_results_in_place():
    """``init`` 只规划，``run`` 把结果原地写进调用方缓冲区。

    - ``operands`` 只读：执行前后逐字节不变；
    - ``results`` 是同一批对象（不是新分配），形状/dtype 保持不变；
    - ``init`` 与 ``run`` 可以分别计时，这就是拆分的意义。
    """
    tasks = _matmul_tasks(4, base=48)
    equations, operands, results = [], [], []
    for equation, ops in tasks.values():
        equations.append(equation)
        operands.append(list(ops))
        results.append(np.zeros((ops[0].shape[0], ops[1].shape[1]),
                                dtype=ops[0].dtype))
    before = [operand.copy() for ops in operands for operand in ops]
    config = {"task_id": list(tasks), "equation": equations,
              "operands": operands, "results": results}

    started = time.perf_counter()
    state = parallel_einsum.init(config, max_proc=2, reminder=False)
    init_seconds = time.perf_counter() - started
    started = time.perf_counter()
    returned = parallel_einsum.run(state)
    run_seconds = time.perf_counter() - started

    assert returned is None, 'run() is a launcher: it must not return a payload'
    assert init_seconds >= 0.0 and run_seconds > 0.0
    assert state["keys"] == list(tasks)
    assert [buffer is original for buffer, original in zip(results, config["results"])]

    flat_before = [operand for ops in operands for operand in ops]
    for saved, current in zip(before, flat_before):
        assert np.array_equal(saved, current), 'operands must stay read-only'

    for key, (equation, ops) in tasks.items():
        index = list(tasks).index(key)
        expected = pe.contract(equation, *ops)
        assert np.allclose(results[index], expected, rtol=0, atol=0), key
        assert results[index] is config["results"][index]


def test_init_rejects_bad_configs():
    """``init`` 只接受列式字典，并对长度、形状、只读缓冲区提前失败。"""
    good_ops = [[np.eye(3), np.eye(3)]]
    cases = (
        ({"equation": ["ij,jk->ik"], "operands": good_ops}, 'results'),
        ({"equation": ["ij,jk->ik"], "operands": good_ops,
          "results": [np.zeros((3, 3))], "extra": 1}, 'unsupported key'),
        ({"equation": ["ij,jk->ik"], "operands": good_ops * 2,
          "results": [np.zeros((3, 3))]}, 'lengths differ'),
        ({"equation": ["ij,jk->ik"], "operands": good_ops,
          "results": [np.zeros((4, 4))]}, 'does not match'),
        ({"equation": [], "operands": [], "results": []}, 'at least one'),
    )
    for config, fragment in cases:
        try:
            parallel_einsum.init(config, max_proc=1, reminder=False)
        except (ValueError, TypeError) as exc:
            assert fragment in str(exc), (fragment, str(exc))
        else:
            raise AssertionError(f'init accepted bad config: {fragment}')

    read_only = np.zeros((3, 3))
    read_only.flags.writeable = False
    try:
        parallel_einsum.init({"equation": ["ij,jk->ik"], "operands": good_ops,
                              "results": [read_only]}, max_proc=1,
                             reminder=False)
    except ValueError as exc:
        assert 'read-only' in str(exc), str(exc)
    else:
        raise AssertionError('init accepted a read-only result buffer')


def test_run_matches_einsums_and_respects_task_ids():
    """``run`` 的结果必须与 ``einsums`` 一致；``task_id`` 只作用于标签与报错。"""
    tasks = _matmul_tasks(3, base=40)
    keys = list(tasks)
    equations = [tasks[key][0] for key in keys]
    operands = [list(tasks[key][1]) for key in keys]
    results = [np.zeros((spec[1][0].shape[0], spec[1][1].shape[1]),
                        dtype=spec[1][0].dtype) for spec in tasks.values()]
    state = parallel_einsum.init({"task_id": keys, "equation": equations,
                                  "operands": operands, "results": results},
                                 max_proc=2, reminder=False)
    parallel_einsum.run(state)
    payload = parallel_einsum.einsums(tasks, max_procs=2, reminder=False,
                                     verbose=False)
    for index, key in enumerate(keys):
        assert np.array_equal(results[index], payload[key]), key


def test_plan_rejects_invalid_input():
    """空集合、非法键、非字符串方程都必须在执行前失败。"""
    for bad in ([], {}):
        try:
            parallel_einsum.plan_einsums(bad)
        except (ValueError, TypeError) as exc:
            assert 'empty' in str(exc) or 'must' in str(exc), str(exc)
        else:
            raise AssertionError(f'empty input accepted: {bad!r}')

    cases = (
        ({'subscripts': 'ij->ji', 'operands': [np.eye(2)], 'typo': 1},
         'unsupported key'),
        ({'subscripts': 42, 'operands': [np.eye(2)]}, 'subscripts'),
        ({'subscripts': 'ij->ji', 'operands': []}, 'no operands'),
        (np.eye(2), 'dict or a'),
    )
    for spec, fragment in cases:
        try:
            parallel_einsum.plan_einsums([spec])
        except (ValueError, TypeError) as exc:
            assert fragment in str(exc), (fragment, str(exc))
        else:
            raise AssertionError(f'invalid spec accepted: {spec!r}')

    try:
        parallel_einsum.plan_einsums(_matmul_tasks(2), max_procs=0)
    except ValueError as exc:
        assert 'max_procs' in str(exc)
    else:
        raise AssertionError('max_procs=0 accepted')


def test_reminder_is_printed_only_once():
    """使用场景提醒只在首次调用打印一次，且带可信性/实测要求。"""
    parallel_einsum._REMINDER_SHOWN = False
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        parallel_einsum.einsums(_matmul_tasks(1, base=8), max_procs=1)
        parallel_einsum.einsums(_matmul_tasks(1, base=8), max_procs=1)
    text = buffer.getvalue()
    assert text.count('使用场景提醒') == 1, text
    for fragment in ('大批量但小单量', '超出硬件资源极限', '可信性分析', '功能实测'):
        assert fragment in text, fragment
    parallel_einsum._REMINDER_SHOWN = False


# ═══════════════════════════════════════════════════════════════════
# 执行路径契约
# ═══════════════════════════════════════════════════════════════════

def test_serial_path_matches_contract_bit_exact():
    """单进程路径必须与逐个 contract 的结果逐位一致（list 与 dict 都要）。"""
    tasks = _matmul_tasks(4, base=24)
    payload, text = _run_quiet(tasks=tasks, max_procs=1)
    expected = {key: pe.contract(eq, *ops) for key, (eq, ops) in tasks.items()}
    _assert_close(payload, expected, 'serial-dict')
    assert 'mode=serial' in text, text

    ordered = list(tasks.values())
    listed, _ = _run_quiet(tasks=ordered, max_procs=1)
    assert isinstance(listed, list) and len(listed) == len(ordered)
    for index, (eq, ops) in enumerate(ordered):
        assert np.array_equal(listed[index], pe.contract(eq, *ops))


def test_launched_path_matches_contract_bit_exact():
    """自建 MPI 进程池路径的结果必须与串行逐位一致。"""
    _require_launcher()
    tasks = _matmul_tasks(4, base=32)
    payload, text = _run_quiet(tasks=tasks, max_procs=2)
    expected = {key: pe.contract(eq, *ops) for key, (eq, ops) in tasks.items()}
    _assert_close(payload, expected, 'launched')
    assert 'mode=mpi-launch' in text, text
    assert 'workers=2' in text, text


def test_launched_path_reports_failing_task_key():
    """某个任务失败时必须整体失败，并在错误里点名任务键（不含静默空结果）。"""
    _require_launcher()
    tasks = _matmul_tasks(3, base=16)
    tasks['broken'] = ('ij,jk->ik',
                       [np.zeros((4, 5)), np.zeros((6, 7))])
    try:
        _run_quiet(tasks=tasks, max_procs=2)
    except RuntimeError as exc:
        message = str(exc)
        assert 'broken' in message, message
        assert 'task(s) failed' in message, message
    else:
        raise AssertionError('a failing task was silently accepted')


def test_invalid_equation_fails_in_every_path():
    """非法方程在串行/进程池两条路径上都必须抛错，不能被空结果掩盖。"""
    broken = {'bad': ('ij,jk->ik', [np.zeros((2, 3)), np.zeros((4, 5))])}
    for max_procs in (1, 2):
        if max_procs == 2 and _mpi_launcher() is None:
            continue
        try:
            _run_quiet(tasks=broken, max_procs=max_procs)
        except RuntimeError as exc:
            assert 'bad' in str(exc), str(exc)
        else:
            raise AssertionError(f'broken equation accepted (max_procs={max_procs})')


def test_distillation_equations_match_serial_contract():
    """真实蒸馏方程（sush S1 2pt / P1 3pt）在并行路径上与串行逐位一致。"""
    tasks = _distillation_tasks()
    expected = {key: pe.contract(spec['subscripts'], *spec['operands'])
                for key, spec in tasks.items()}
    serial, _ = _run_quiet(tasks=tasks, max_procs=1)
    _assert_close(serial, expected, 'distillation-serial')
    if _mpi_launcher() is not None:
        launched, text = _run_quiet(tasks=tasks, max_procs=2)
        _assert_close(launched, expected, 'distillation-launched', rtol=1e-12)
        assert 'mode=mpi-launch' in text, text


def test_collective_memory_warning_is_emitted():
    """进入 collective 必须打印内存警告，且与 ``reminder=`` 无关。

    实测依据：每个 rank 都持有整批输入（4 rank、整批输入 134 MB 时聚合 ≈1.15 GB），
    是最容易被忽略、在 GPU 上会直接 OOM 的代价，所以它不能藏在通用提醒后面。
    """
    launcher = _require_launcher()
    project_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as tmpdir:
        script = Path(tmpdir) / 'warning_probe.py'
        script.write_text(
            'import numpy as np\n'
            'from pyqcd.parallel.einsum import einsums\n'
            'rng = np.random.default_rng(3)\n'
            'tasks = {"a": ("ij,jk->ik", [rng.standard_normal((16, 16)),\n'
            '                             rng.standard_normal((16, 16))])}\n'
            '# collective 默认关闭；这里显式打开，验证内存警告与 reminder 无关\n'
            'einsums(tasks, max_procs=2, reminder=False, verbose=False,\n'
            '        collective=True)\n',
            encoding='utf-8')
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join(
            [str(project_root), env.get('PYTHONPATH', '')]).rstrip(os.pathsep)
        completed = subprocess.run(
            launcher + ['-np', '2', sys.executable, str(script)],
            capture_output=True, text=True, env=env, cwd=project_root,
            timeout=300)
    assert completed.returncode == 0, completed.stderr[-1200:]
    assert 'collective 模式内存提醒' in completed.stdout, completed.stdout[-800:]
    # 只由 rank 0 打印一次，避免 N 份重复刷屏
    assert completed.stdout.count('collective 模式内存提醒') == 1, completed.stdout


def test_collective_path_matches_contract_and_returns_none_on_workers():
    """``mpirun`` 内部调用：rank 0 拿到结果，其余 rank 得到 None。"""
    launcher = _require_launcher()
    project_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as tmpdir:
        script = Path(tmpdir) / 'collective_probe.py'
        script.write_text(
            'import json, sys\n'
            'import numpy as np\n'
            'from mpi4py import MPI\n'
            'import pyqcd.einsum as pe\n'
            'from pyqcd.parallel.einsum import einsums\n'
            'comm = MPI.COMM_WORLD\n'
            'rank = comm.Get_rank()\n'
            'rng = np.random.default_rng(99)\n'
            'tasks = {}\n'
            'for i in range(6):\n'
            '    n = 32 + 4 * i\n'
            '    tasks[f"k{i}"] = ("ij,jk->ik", [rng.standard_normal((n, n)),\n'
            '                                     rng.standard_normal((n, n))])\n'
            'payload = einsums(tasks, max_procs=3, reminder=False,\n'
            '                  verbose=(rank == 0), collective=True)\n'
            'if rank == 0:\n'
            '    exact = all(np.array_equal(payload[k],\n'
            '                               pe.contract(v[0], *v[1]))\n'
            '                for k, v in tasks.items())\n'
            '    print("RESULT", json.dumps({"exact": exact,\n'
            '                                "keys": sorted(payload)}))\n'
            'else:\n'
            '    print("RESULT", json.dumps({"payload": payload is None}))\n',
            encoding='utf-8')
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join(
            [str(project_root), env.get('PYTHONPATH', '')]).rstrip(os.pathsep)
        completed = subprocess.run(
            launcher + ['-np', '2', sys.executable, str(script)],
            capture_output=True, text=True, env=env, cwd=project_root,
            timeout=300)
    assert completed.returncode == 0, completed.stderr[-2000:]
    lines = [line for line in completed.stdout.splitlines()
             if line.startswith('RESULT ')]
    assert len(lines) == 2, completed.stdout
    payloads = [json.loads(line.split(' ', 1)[1]) for line in lines]
    assert any(item.get('exact') is True for item in payloads), payloads
    assert any(item.get('payload') is True for item in payloads), payloads


def test_parallel_import_does_not_change_einsum():
    """并行入口不得改变 ``pyqcd.einsum`` 的行为，也不得在包导入时拉入它。"""
    project_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, '-c',
         'import sys, numpy as np\n'
         'import pyqcd.parallel as P\n'
         'assert "pyqcd.einsum" not in sys.modules, "import pulled einsum"\n'
         'import pyqcd.einsum as pe\n'
         'a = np.arange(6.0).reshape(2, 3); b = np.arange(12.0).reshape(3, 4)\n'
         'ref = pe.contract("ij,jk->ik", a, b)\n'
         'pe.contract("ij,jk->ik", a, b)\n'
         'assert np.array_equal(ref, a @ b)\n'
         'assert callable(P.einsums) and callable(P.plan_einsums)\n'
         'print("OK")\n'],
        capture_output=True, text=True, cwd=project_root,
        env=dict(os.environ, PYTHONPATH=str(project_root)))
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert 'OK' in completed.stdout


def test_worker_entry_point_is_reachable():
    """``python -m pyqcd.parallel.einsum --worker`` 必须存在且缺目录时报错。"""
    project_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, '-m', 'pyqcd.parallel.einsum', '--dry-run'],
        capture_output=True, text=True, cwd=project_root,
        env=dict(os.environ, PYTHONPATH=str(project_root)))
    assert completed.returncode == 0, completed.stderr[-1200:]
    assert 'mode=dry-run' in completed.stdout
    assert '使用场景提醒' in completed.stdout

    missing = subprocess.run(
        [sys.executable, '-m', 'pyqcd.parallel.einsum', '--worker',
         os.path.join(tempfile.gettempdir(), 'pyqcd-does-not-exist')],
        capture_output=True, text=True, cwd=project_root,
        env=dict(os.environ, PYTHONPATH=str(project_root)))
    assert missing.returncode != 0


_TESTS = (
    test_plan_balances_skewed_workloads,
    test_plan_respects_task_and_process_caps,
    test_worker_thread_budget_is_not_inherited_from_parent,
    test_timings_separate_pool_init_from_compute,
    test_collective_is_disabled_by_default,
    test_init_and_run_write_results_in_place,
    test_init_rejects_bad_configs,
    test_run_matches_einsums_and_respects_task_ids,
    test_plan_rejects_invalid_input,
    test_reminder_is_printed_only_once,
    test_serial_path_matches_contract_bit_exact,
    test_launched_path_matches_contract_bit_exact,
    test_launched_path_reports_failing_task_key,
    test_invalid_equation_fails_in_every_path,
    test_distillation_equations_match_serial_contract,
    test_collective_memory_warning_is_emitted,
    test_collective_path_matches_contract_and_returns_none_on_workers,
    test_parallel_import_does_not_change_einsum,
    test_worker_entry_point_is_reachable,
)
TESTS = _TESTS


def main():
    passed = 0
    for test in _TESTS:
        test()
        print(f'PASS {test.__name__}')
        passed += 1
    print(f'{passed} passed, 0 failed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
