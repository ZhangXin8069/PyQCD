"""
MPI Parallel Execution for pyqcd
================================

Task-parallel execution of the pipeline across MPI processes, following
the user's resource formula:

    N * a  = n * b        (N processes x per-task VRAM = total usable VRAM)
    X      = m / N        (number of batches; each batch submits X tasks)
    Y      = N / n        (integer; Y processes per GPU)

where
- ``a`` : VRAM needed by one meta-task (smallest independent unit)
- ``m`` : total number of meta-tasks
- ``b`` : usable VRAM per GPU (default 80% of card capacity)
- ``n`` : number of GPUs

Meta-tasks are the per-configuration compute steps (vertex / 2pt / ope /
3pt / 4pt); each writes its own HDF5 files under ``data/conf<id>/`` so
ranks never share mutable state. Analysis / plotting / report steps stay
serial on rank 0 (light CPU work, as agreed).

Meta-task scheduling: round-robin static partitioning (rank r handles
conf_ids[r::N]); vertices are computed in a first wave (all ranks),
then a barrier, then 2pt/3pt/4pt/ope (they read the vertex HDF5 files).
Memory is released automatically after each meta-task
(``del`` + ``torch.cuda.empty_cache`` + ``gc.collect``).
"""

import gc
import math
import os
import time

from ._resources import detect_resources

# ═══════════════════════════════════════════════════════════════════
# MPI context
# ═══════════════════════════════════════════════════════════════════


def get_mpi_context():
    """Return (comm, rank, size) if running under MPI, else (None, 0, 1)."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        return comm, comm.Get_rank(), comm.Get_size()
    except Exception:
        return None, 0, 1


# ═══════════════════════════════════════════════════════════════════
# Parallel plan (user formula)
# ═══════════════════════════════════════════════════════════════════

def plan_parallel(m, a_mem_mb, resources=None, n_gpu=None, force_y=None,
                  verbose=True):
    """Plan the MPI process count from the user formula N*a = n*b.

    Parameters
    ----------
    m : int
        Number of meta-tasks (e.g. len(conf_ids) x compute steps).
    a_mem_mb : float
        VRAM needed by one meta-task (MB). If None/0, defaults to one
        process per GPU (Y=1).
    resources : dict, optional
        Output of ``detect_resources()``. Auto-detected if None.
    n_gpu : int, optional
        Override GPU count.
    force_y : int, optional
        Force Y (processes per GPU); N = Y*n (still capped by cpu/mem).

    Returns
    -------
    dict with keys: n_gpu, b_mb, gpu_total_mb, a_mb, m, N, Y, X,
    per_rank_vram_mb, cpu_ok, mem_ok, notes.
    """
    res = resources or detect_resources()
    n = n_gpu if n_gpu is not None else res['n_gpu']
    if n < 1:
        # no GPU: fall back to CPU process count (1 per 2 threads, mem-capped)
        N = max(1, min(res['cpu_threads'] // 2 or 1, m))
        return {
            'n_gpu': 0, 'b_mb': 0.0, 'gpu_total_mb': 0.0,
            'a_mb': a_mem_mb or 0.0, 'm': m, 'N': N, 'Y': 0,
            'X': math.ceil(m / N) if N else 0,
            'per_rank_vram_mb': 0.0, 'cpu_ok': True, 'mem_ok': True,
            'notes': 'no GPU detected; CPU-only plan',
        }

    b = res['gpu_usable_mb']          # 80% of card
    gpu_total = n * b
    if a_mem_mb:
        N_vram = max(1, int(gpu_total // a_mem_mb))     # N*a = n*b
    else:
        N_vram = n                                     # Y=1 default
    if force_y is not None:
        N_vram = force_y * n
    # Y = N/n must be an integer: round down to a multiple of n
    Y = max(1, N_vram // n)
    N = Y * n
    # caps: cpu threads, available RAM (~3x task VRAM per process), m
    cpu_ok = N <= max(1, res['cpu_threads'])
    mem = res['mem_avail_mb'] or res['mem_total_mb']
    mem_needed = N * max(1.0, a_mem_mb) * 3.0 if a_mem_mb else N * 2048.0
    mem_ok = mem is None or mem_needed <= mem
    if not cpu_ok:
        N = max(1, min(res['cpu_threads'], N))
        Y = N // n
        if Y < 1:
            N = n
            Y = 1
    if not mem_ok and a_mem_mb:
        N_from_mem = max(1, int(mem // (3.0 * a_mem_mb)))
        N = min(N, max(n, N_from_mem // n * n))
        Y = N // n
    if N > m:
        N = max(1, (m // n) * n) if m >= n else m
        Y = N // n
        if Y < 1:
            N = max(1, m)
            Y = 1

    return {
        'n_gpu': n, 'b_mb': b, 'gpu_total_mb': gpu_total,
        'a_mb': a_mem_mb or 0.0, 'm': m, 'N': N, 'Y': Y,
        'X': math.ceil(m / N) if N else 0,
        'per_rank_vram_mb': gpu_total / N if N else 0.0,
        'cpu_ok': N <= max(1, res['cpu_threads']),
        'mem_ok': True,
        'notes': 'N*a=n*b formula' if a_mem_mb else 'default Y=1 per GPU',
    }


def format_plan(plan):
    """Human-readable plan summary (used in reports / logs)."""
    if plan['n_gpu'] == 0:
        return (f"no-GPU plan: N={plan['N']} processes (CPU), "
                f"m={plan['m']}, X={plan['X']} batches")
    return (f"GPU plan: n={plan['n_gpu']} GPUs, b={plan['b_mb']:.0f} MB each "
            f"(80% of {plan['gpu_total_mb']/plan['n_gpu']/0.8:.0f} MB), "
            f"total={plan['gpu_total_mb']:.0f} MB; "
            f"a={plan['a_mb']:.0f} MB/task, m={plan['m']} tasks → "
            f"mpirun -np {plan['N']} (Y={plan['Y']} proc/GPU), "
            f"X={plan['X']} batches; per-rank VRAM {plan['per_rank_vram_mb']:.0f} MB")


# ═══════════════════════════════════════════════════════════════════
# Meta-task execution
# ═══════════════════════════════════════════════════════════════════

def run_meta_task(step, conf_id, config, run_dir, logger):
    """Run one meta-task (step, conf) and release memory afterwards."""
    from ..pipeline._steps import (
        compute_vertices_for_config, compute_2pt_for_config,
        compute_3pt_for_config, compute_4pt_for_config,
        compute_ope_for_config, _load_vertices_one,
        free_gpu_memory, _timer,
    )
    t0 = __import__('time').perf_counter()
    if step == 'vertex':
        compute_vertices_for_config(conf_id, run_dir, logger,
                                    config.get('precision', 'complex64'))
    elif step == '2pt':
        verts = _load_vertices_one(run_dir, conf_id)
        compute_2pt_for_config(conf_id, run_dir, logger, verts,
                               config.get('precision', 'complex64'),
                               config.get('channels', ('pp', 'pn', 'pion')))
        del verts
    elif step == '3pt':
        verts = _load_vertices_one(run_dir, conf_id)
        compute_3pt_for_config(conf_id, run_dir, logger, verts,
                               config.get('precision', 'complex64'),
                               config.get('t_sep', 12))
        del verts
    elif step == '4pt':
        verts = _load_vertices_one(run_dir, conf_id)
        compute_4pt_for_config(conf_id, run_dir, logger, verts,
                               config.get('precision', 'complex64'))
        del verts
    elif step == 'ope':
        # 注意：必须显式设置 GPU 后端（step_ope 内部会做，但并行元任务直接调用
        # compute_ope_for_config，需在此补齐 set_backend，否则退回 numpy/CPU 后端，
        # OPE 将极其缓慢且不利用 GPU）。
        from ..tools import set_backend
        set_backend(config.get('backend', 'cupy'),
                    device=config.get('device'))
        compute_ope_for_config(conf_id, run_dir, logger,
                               config.get('precision', 'complex64'))
    else:
        raise ValueError(f"no meta-task handler for step '{step}'")
    # auto memory release
    import torch
    torch.cuda.empty_cache()
    del torch
    gc.collect()
    return time.perf_counter() - t0


# ═══════════════════════════════════════════════════════════════════
# Parallel pipeline driver
# ═══════════════════════════════════════════════════════════════════

def run_parallel_pipeline(steps=('env', 'vertex', '2pt', 'ope', '3pt', '4pt',
                                 'analysis', 'plots', 'report'),
                          conf_ids=None, run_dir=None, logger=print,
                          precision='complex64', backend='torch',
                          device=None, a_mem_mb=None, resources=None,
                          plan=None, channels=('pp', 'pn', 'pion'),
                          n_gpu=None, force_y=None, dry_run=False):
    """Run the pipeline with MPI task parallelism.

    Serial behaviour (no mpirun / MPI size 1) is fully supported:
    falls back to the standard ``run_pipeline``.

    Parallel behaviour (mpirun -np N):
    - compute steps (vertex/2pt/ope/3pt/4pt) are split across ranks
      round-robin per configuration;
    - analysis / plots / report run once on rank 0 after a barrier;
    - the plan (N, Y, X) follows the user formula N*a = n*b.

    Returns (result_dict, plan_dict). ``dry_run=True`` only prints the
    plan (no computation).
    """
    from ..pipeline import run_pipeline as _serial_run

    conf_ids = list(conf_ids or [6250])
    res = resources or detect_resources()
    if plan is None:
        plan = plan_parallel(len(conf_ids), a_mem_mb, resources=res,
                             n_gpu=n_gpu, force_y=force_y, verbose=True)
    if logger is not None:
        logger(f"[parallel] plan: {format_plan(plan)}")

    comm, rank, size = get_mpi_context()
    if dry_run:
        return None, plan
    if size <= 1:
        return _serial_run(steps=steps, conf_ids=conf_ids, run_dir=run_dir,
                           logger=logger, precision=precision,
                           channels=channels, backend=backend,
                           device=device), plan

    # ── MPI parallel execution ─────────────────────────────────────
    from ..pipeline._steps import (
        run_pipeline as _run, dump_config_snapshot,
        _info, _timer,
    )
    from ..pipeline._config import PRECISION

    n = plan['n_gpu']
    my_device = None
    if backend == 'torch' and n >= 1:
        my_device = f'cuda:{rank % n}'
        import torch
        torch.cuda.set_device(my_device)
        device = my_device
    if logger is not None:
        logger(f"[rank {rank}] backend={backend} device={device}")

    config = {
        'precision': precision, 'channels': tuple(channels),
        'conf_ids': conf_ids, 'backend': backend, 'device': device,
    }
    os.makedirs(os.path.join(run_dir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'analysis'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'plots'), exist_ok=True)
    if rank == 0:
        dump_config_snapshot(config, os.path.join(run_dir, 'run_config.json'),
                             logger)

    timing = {}
    parallel_steps = ('vertex', '2pt', '3pt', '4pt', 'ope')
    for step in steps:
        if step not in parallel_steps:
            continue
        t0 = time.perf_counter()
        # round-robin: rank r handles confs[r::size]
        my_confs = conf_ids[rank::size]
        if logger is not None:
            logger(f"[rank {rank}] {step}: {len(my_confs)} confs "
                   f"→ {my_confs}")
        for cid in my_confs:
            _timer(f"[rank {rank}] {step} conf={cid}", logger,
                   run_meta_task, step, cid, config, run_dir, logger)
        from ..pipeline._steps import free_gpu_memory
        free_gpu_memory()
        comm.Barrier()   # vertices of all confs ready before next step
        timing[f'{step}'] = round(__import__('time').perf_counter() - t0, 1)
        if rank == 0 and logger is not None:
            logger(f"STEP {step} done (parallel) in {timing[step]}s")

    # analysis / plots / report: rank 0 only (light CPU work)
    meff_res, ratio_conn, summary, env = None, None, None, None
    if rank == 0:
        from ..pipeline._steps import (
            step_analysis, step_plots, step_report, run_pipeline as _rp,
        )
        for step in steps:
            if step == 'analysis':
                analysis = step_analysis(config, run_dir, logger)
                meff_res = analysis['meff']
                ratio_conn = analysis['connected_ratio']
            elif step == 'plots':
                meff_res = step_plots(config, run_dir, logger,
                                      meff_res, ratio_conn)
            elif step == 'report':
                summary = step_report(config, run_dir, logger,
                                      meff_res, timing, env)
    comm.Barrier()
    if rank == 0 and logger is not None:
        logger(f"[parallel] pipeline done: {plan['N']} ranks, "
               f"{len(conf_ids)} configs, {format_plan(plan)}")
    return {'run_dir': run_dir, 'timing': timing, 'summary': summary,
            'meff': meff_res, 'ratio_conn': ratio_conn}, plan


def main():
    """CLI entry: python -m pyqcd.parallel [--dry-run] [--confs a,b] ..."""
    import argparse
    ap = argparse.ArgumentParser(description='pyqcd MPI parallel pipeline')
    ap.add_argument('--steps', nargs='+',
                    default=['env', 'vertex', '2pt', 'ope', '3pt', '4pt',
                             'analysis', 'plots', 'report'])
    ap.add_argument('--confs', type=str, default=None,
                    help='comma-separated conf ids (default 6250)')
    ap.add_argument('--run-dir', default=None)
    ap.add_argument('--backend', default='torch', choices=['torch', 'cupy',
                                                           'numpy'])
    ap.add_argument('--device', default=None)
    ap.add_argument('--n-gpu', type=int, default=None,
                   help='强制使用的 GPU 数量（默认按 detect_resources 自动探测）；'
                        '例如 2 表示用 2 卡并行，逐 rank 绑定 cuda:{rank%%n}')
    ap.add_argument('--a-mem-mb', type=float, default=None,
                   help='VRAM per meta-task in MB (plan formula)')
    ap.add_argument('--precision', default='complex64',
                    choices=['complex64', 'complex128'])
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    confs = [int(x) for x in args.confs.split(',')] if args.confs else [6250]
    res = detect_resources()
    plan = plan_parallel(len(confs), args.a_mem_mb, resources=res,
                         n_gpu=args.n_gpu)
    print(format_plan(plan))
    if args.dry_run:
        return
    run_parallel_pipeline(
        steps=tuple(args.steps), conf_ids=confs, run_dir=args.run_dir,
        precision=args.precision, backend=args.backend,
        device=args.device, a_mem_mb=args.a_mem_mb, plan=plan,
        n_gpu=args.n_gpu)


if __name__ == '__main__':
    main()
