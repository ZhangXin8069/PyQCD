"""Report measured log timers separately from unbenchmarked optimization ideas."""
import csv
from collections import defaultdict
import json
import os
import re
import statistics

from resume_common import (HERE, PRODUCTION, atomic_json, load_plan, now, read_json,
                           require_allocation)


def summary(values):
    ordered = sorted(values)
    if not ordered:
        return dict(n=0)
    return dict(n=len(ordered), minimum=min(ordered), median=statistics.median(ordered),
                mean=statistics.mean(ordered), maximum=max(ordered),
                p10=ordered[round(.1 * (len(ordered) - 1))],
                p90=ordered[round(.9 * (len(ordered) - 1))])


def main():
    require_allocation()
    plan = load_plan()
    manifest = read_json(PRODUCTION / 'manifest.json')
    pattern = re.compile(r'PZ0_CASE_PASS\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([0-9.eE+\-]+)')
    cases, memory, log_files = {}, [], []
    duplicate_case_timers = 0
    for path in sorted((PRODUCTION / 'logs').glob('gpu_3926911_*.out')):
        log_files.append(str(path))
        with path.open(errors='replace') as stream:
            for line in stream:
                for match in pattern.finditer(line):
                    key = tuple(map(int, match.groups()[:4]))
                    if key in cases:
                        duplicate_case_timers += 1
                    cases[key] = float(match.group(5))
                if 'GPU_MEMORY ' in line:
                    try:
                        record, _ = json.JSONDecoder().raw_decode(line.split('GPU_MEMORY ', 1)[1].lstrip())
                        memory.append(record)
                    except (ValueError, TypeError):
                        pass
    by_flow, by_task = defaultdict(list), defaultdict(list)
    with (HERE / 'case_timings.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('task', 'conf', 'tau24', 'matched_tau64', 't_current', 'tsep', 'dt', 'contract_case_seconds'))
        for (task, tj, sep, dt), seconds in sorted(cases.items()):
            row = manifest['tasks'][task]
            by_flow[row['tau64']].append(seconds)
            by_task[task].append(seconds)
            writer.writerow((task, row['conf'], row['flow_time_a2'], row['tau64'], tj, sep, dt, seconds))
    basis_by_flow = defaultdict(list)
    basis_count = 0
    for path in (PRODUCTION / 'basis_cache').glob('*/*/basis/tau*/done.json'):
        meta = read_json(path)
        if meta.get('manifest_sha256') != plan['manifest_sha256']:
            continue
        tau64 = min(plan['matched_tau64'], key=lambda t: abs(t * (.0897 / .105)**2 - meta['flow_time']))
        basis_by_flow[tau64].append(float(meta['seconds']))
        basis_count += 1
    report = dict(status='MEASURED_LOG_TIMERS_AND_SOURCE_AUDIT_ONLY', generated=now(),
                  job=os.environ['SLURM_JOB_ID'], manifest_sha256=plan['manifest_sha256'],
                  original_gpu_job='3926911', source_logs=log_files, unique_case_timers=len(cases),
                  duplicate_timers_collapsed=duplicate_case_timers,
                  contract_case_seconds=summary(list(cases.values())),
                  contract_case_seconds_by_matched_tau64={str(t): summary(v) for t, v in sorted(by_flow.items())},
                  contract_case_seconds_by_task={str(t): summary(v) for t, v in sorted(by_task.items())},
                  basis_original_time_measurements=basis_count,
                  basis_seconds_per_original_time_by_matched_tau64={str(t): summary(v) for t, v in sorted(basis_by_flow.items())},
                  memory_stage_snapshots=memory,
                  numerical_kernel_changed=False,
                  limitations=['contract_case includes perambulator loading, VVV construction, contraction and saving, but excludes prior C2-oracle and full basis/vertex construction',
                               'case times vary with window, rank, I/O and concurrent allocations; no measured kernel-level speedup',
                               'basis seconds are per original time, not per conf-flow wall time; completed items are a censored sample',
                               'GPU memory stage snapshots are not peak-memory measurements'])
    atomic_json(HERE / 'performance_summary.json', report)
    lines = [
        '# L24×72 取消后恢复与性能审查', '',
        f'统计时间：{report["generated"]}；统计运行于 CPU job `{report["job"]}`。', '',
        '恢复使用原冻结内核，未改 B 权重、原始时间窗、APBC、Wick 项、流参数或数组精度。',
        '保留 basis；不持久保存 vertex/VVV。优化建议尚未成为本次生产代码。', '',
        '## 日志实测', '',
        f'旧 GPU 数组 `3926911` 中读到 {len(cases)} 条去重后的 case 计时。',
        '`contract_case` 计时包含该函数内的 perambulator/VVV/收缩/存盘，',
        '不包含此前批量 C2 oracle、完整 basis 与 generalized vertex 阶段。', '',
        '| 对应 L32×64 的 tau | L24×72 tau | case 数 | 中位数/s | P10/s | P90/s |',
        '| --- | --- | --- | --- | --- | --- |',
    ]
    for tau64, values in sorted(by_flow.items()):
        stat = summary(values)
        lines.append(f'| {tau64:.1f} | {tau64 * (.0897 / .105)**2:.9f} | {stat["n"]} | '
                     f'{stat["median"]:.2f} | {stat["p10"]:.2f} | {stat["p90"]:.2f} |')
    lines.extend([
        '', '不同窗口、组态、I/O 与并发状态会影响耗时，不能据此单独归因于某个内核。', '',
        '## 已由源码确认的重复开销，以及下一步优化顺序', '',
        '1. `gpu_worker.c2_oracle` 与 `contract_case` 重复构造外部 VVV；同一 current 的多个 case 还会重复使用相同 source/sink 时间。',
        '   候选：有界、进程内的 VVV 复用，key 必须包含 conf、source/sink 时间、动量、NE、输入身份和 sink/source 共轭约定。不恢复磁盘 VVV 缓存。',
        '2. 每个 case 新建 `Contractor("optimal")`；同 shape 的路径缓存随对象丢失。',
        '   候选：进程级复用 Contractor，但先测路径缓存增长和峰值显存，保持数学方程与 dtype 不变。',
        '3. `c3_reduced` 在原始右时间×动量循环中重复外部旋量投影。',
        '   候选：显式提取不变量；逐块比较六个 Wick 项、u/d、全部通道、±P/±z，再比较完整 C2/C3。不可丢掉 current 的 4×4 spin。',
        '4. 约 22 MB manifest 在部分内循环重复 SHA；可统一到已有的带 stat-change 拒绝机制的进程身份缓存。',
        '   身份校验不能取消；这是 I/O 改进候选，未证明是主要瓶颈。',
        '5. basis 当前 `FLOW_MODE_BATCH=16`，每批重复 coupled gauge/fermion flow。',
        '   32/64 batch 与共享多 flow checkpoint 属于第二阶段优化，必须测峰值显存、APBC、时间窗和直接/重启一致性。',
        '   七个独立目标的步数和为 700，最大为 160；700/160 只是步数比，不是实际加速倍数。',
        '6. `free_all_blocks()` 在内循环调用，可能增加 allocator 开销；不能在缺少峰值测量时直接删掉以免 OOM。', '',
        '## 不能为提速改变的内容', '',
        'B 的同原始时间二阶表、跨时间一阶乘积表；权重只施加一次；4D 流和 APBC；',
        '端点与 Wilson line 的同一 thin-link 历史；七个匹配物理半径；',
        '有界右时间 H 聚合；全部六个 Wick 项、复数值与 current spin；',
        'receipt/hash/投影完成门，以及已提交源码不可变性。', '',
        '## 推荐的优化回归（尚未提交/执行）', '',
        '建立隔离 successor，固定 conf=10000，选最小和最大 flow、tJ=0/71、T=7/12，',
        '覆盖接触点、平台中点和跨边界 source/sink。先分阶段计时并记录显存峰值；',
        '与原结果逐块对照 C2、H、六个 Wick 项、u/d 和三个通道，',
        '使用原 manifest 的代数误差门，再报告实测 wall time 和加速比。',
        '回归通过前不替换本次恢复生产内核。', '',
        '详细数据：`case_timings.csv` 与 `performance_summary.json`。', '',
    ])
    (HERE / 'PERFORMANCE_REPORT.md').write_text('\n'.join(lines))
    print('PERFORMANCE_REPORT_COMPLETE', len(cases), summary(list(cases.values())), flush=True)


if __name__ == '__main__':
    main()
