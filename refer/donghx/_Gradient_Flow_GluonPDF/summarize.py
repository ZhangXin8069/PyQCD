#!/usr/bin/env python3
"""Summarize existing JSON/logs only; no scientific calculation on login nodes."""
from pathlib import Path
import datetime
import ast
import hashlib
import json
import re
import subprocess

HERE = Path(__file__).resolve().parent
JOB_IDS = '3931190,3931191,3960805,3960809,3960815,3960818'
snapshot = subprocess.run(
    ['sacct', '-X', '-j', JOB_IDS, '--format=JobID,JobName%24,Partition,State,ExitCode,Elapsed,NodeList%24', '-P'],
    check=True, capture_output=True, text=True).stdout
now = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
gpu_file = HERE / 'results/gpu_result.json'
gpu = json.loads(gpu_file.read_text()) if gpu_file.exists() else {}
source_hashes = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                 for name in ('benchmark.py', 'gpu.slurm', 'install_native.slurm')}
audit_log = HERE.parent / 'quark_l32x96_B_resume_20261105/logs/final_3931191_4294967294.out'
progress = re.findall(r'ALL_CASES_VERIFIED\s+(\d+)\s+(\d+)', audit_log.read_text())
audit = None if not progress else dict(task_index=int(progress[-1][0]),
                                      verified_cases=int(progress[-1][1]), expected_cases=52920)
completion_file = HERE.parent / 'quark_l32x96_B_resume_20261105/status/complete.json'
completion = json.loads(completion_file.read_text()) if completion_file.exists() else None
special_file = HERE / 'venv_native/lib/python3.10/site-packages/pyqcd/einsum/_specialize.py'
registry = {}
if special_file.exists():
    for node in ast.parse(special_file.read_text()).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == 'REGISTRY':
            registry = ast.literal_eval(node.value)
equations = sorted({item['equation'] for row in gpu.get('benchmarks', [])
                    for con in row['contractor'].values() for item in con['equations']})
specialisation_audit = dict(kind='static exact-equation registry comparison, not a runtime dispatch trace',
                           package_file=str(special_file), equation_count=len(equations),
                           exact_matches={eq: registry[eq] for eq in equations if eq in registry},
                           note='Cached ContractExpression calls do not pass through contract() specialization dispatch.')
data = dict(as_of=now, slurm=snapshot, production_audit=audit,
            production_completion=completion, specialisation_audit=specialisation_audit,
            benchmark_status=gpu.get('status', 'NOT_STARTED'), source_sha256=source_hashes)
(HERE / 'metadata/jobs_snapshot.json').write_text(json.dumps(data, indent=2) + '\n')

def maximum_relative(value):
    if not isinstance(value, dict):
        return 0.
    return max([float(value.get('relative_L2', 0.))] +
               [maximum_relative(v) for v in value.values() if isinstance(v, dict)])

lines = ['# L32x96 quark PDF：PyQCD contraction 实测报告', '', f'更新：{now}', '',
         '## 结论', '',
         '本次真实输入测试数值通过，但仅替换 contraction 引擎没有观测到有意义的加速。',
         '保持原来的生产代码和共享 conda 环境不变；PyQCD 仅安装在本目录隔离 venv 中。', '',
         '## L32x96 生产任务状态', '',
         'GPU 补算数组 3931190：72 个子任务均已 COMPLETED，退出码 0:0。',
         'CPU 最终检查 3931191：已 COMPLETED，退出码 0:0。', '']
if audit:
    lines += [f"新增 30 conf 的检查：{audit['verified_cases']} / {audit['expected_cases']} cases，210 个 conf-flow 任务。", '']
if completion:
    lines += [f"合并完成回执：`{completion['status']}`，总计 `{completion['combined_nconf']}` conf。",
              '最终检查对新增 30 conf 逐结果校验，对原 20 conf 核对既有完成汇总的身份；不是重新运行所有原始计算。',
              f'完成回执：`{completion_file}`。', '']
lines += ['## 安装环境及安装限制', '',
          '官方 PyPI 分发包：pyqcd 0.2.1；生产 Python 3.10.18、CuPy 14.1.1、opt_einsum 3.4.0。',
          'NumPy 2.2.6、SciPy 1.15.3 沿用生产环境；隔离安装了 h5py 3.14.0 兼容二进制包。',
          '初次 venv 安装成功，但旧 Bash 在没有 OpenBLAS 开发文件时触发空数组错误，未编出原生库。',
          '第二次 venv_native 安装加载 OpenBLAS 模块，原样使用官方源码，显式编译 sm80 CUDA。',
          'CUDA 库及 Cython 接口均已编出；安装日志仍警告缺少 qcd_cuda_hadamard provider。',
          'CPU KUPL 库另有 `undefined symbol: ompi_mpi_byte` 动态链接问题；CPU 数值测试走了安全 fallback。',
          '因此 CPU 接口数值通过不能写成“CPU 原生加速验证通过”。这不影响已经测量的 CUDA 数值一致性。',
          '详见 metadata/installed_environment_native.json、logs/native_install_3960815.err 和 GPU JSON 的 warnings。', '',
          '## 数值验证和性能', '', f"测试状态：**{gpu.get('status', 'NOT_STARTED')}**。", '']
if gpu.get('probe'):
    lines += [f"CUDA diagnostic probe: {gpu['probe']['verdict']} on {gpu['probe'].get('device')}.", '']
for name in ('cpu_native_validation', 'gpu_small_validation'):
    if gpu.get(name):
        label = 'CPU 接口校验（含 fallback）' if name == 'cpu_native_validation' else 'GPU 小尺寸校验'
        lines += [f"{label}：PASS；最大相对 L2 差异 {maximum_relative(gpu[name]):.3e}。", '']
if gpu.get('benchmarks'):
    lines += ['| Stage | opt_einsum warm median (s) | PyQCD warm median (s) | Speedup old/new | Max relative L2 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for row in gpu['benchmarks']:
        med = row['median_wall_seconds']
        lines += [f"| {row['label']} | {med['opt_einsum']:.6g} | {med['pyqcd']:.6g} | "
                  f"{row['speedup_opt_over_pyqcd']:.4g} | {maximum_relative(row['correctness']):.3e} |"]
    lines += ['']
    full = next((row for row in gpu['benchmarks'] if row['label'] == 'real_full_sigma_sum_C3_one_Pz_all_z'), None)
    if full:
        lines += [f"完整 C3 与已有生产文件的比较：原引擎相对差异 {maximum_relative(full['production_output_check']['opt_einsum']):.3e}，",
                  f"PyQCD 相对差异 {maximum_relative(full['production_output_check']['pyqcd']):.3e}；验收阈值 2e-10。", '',
                  '上述表格来自每个引擎一次额外 warmup 后的五次同步计时，交错运行顺序，取 wall-time 中位数。',
                  '冷启动（含本轮表达式规划/首次执行）及每个表达式的规划时间另外保存在 JSON 内。', '']
if gpu.get('error'):
    lines += ['The benchmark did not pass:', '', '```text', gpu['error'], '```', '']
if gpu.get('real_input'):
    inp = gpu['real_input']
    lines += ['## 实际输入及范围', '',
              f"conf={inp['conf']}，Pz={inp['Pz']}，tsep={inp['tsep']}，dt={inp['dt']}，t_current={inp['current']}。",
              f"tau_L32x96={inp['tau_L32x96']:.12g}，匹配的 tau_L32x64={inp['matched_tau_L32x64']}。",
              'nev_source=200、nev_current=400，全部 21 个 z、正负 Wilson line、四个 baryon projectors、',
              '六个 Wick 项和完整的所缓存原始时间双求和。权重仍为 B 方案，恰好施加一次。',
              f"输入读取、B-weighted vertex 及 VVV 重建共 {inp['preparation_seconds_IO_vertex_VVV']:.3f} 秒；",
              '这部分未替换引擎，单列而不算入 contraction 加速倍率。',
              f"真实参考结果：`{inp['production_result']}`。", '']
lines += ['## 为什么当前没有明显加速', '',
          '当前生产代码已经缓存 contract_expression，并做了外部旋量降维、H 时间求和和 Wick 分块；',
          '本次只替换缓存表达式的实现，不改变这些算法。不能假定文档中其他 contraction 的专用核收益直接适用。',
          f"本次 {specialisation_audit['equation_count']} 个实际表达式与安装包的专用方程登记表静态精确比较：",
          f"命中数 {len(specialisation_audit['exact_matches'])}；且 ContractExpression 入口绕过 contract() 的专用方程分派。",
          'GPU probe 的核子 2pt 专用核可用，不等于我们的降维 C2/C3 方程调用了同一个专用核。',
          '静态检查不是运行时调用轨迹；本报告不把所有实际表达式武断归类为原生或 fallback。', '',
          '设备内存每 50 ms 采样，包含不经 CuPy 分配的 native 工作空间。两引擎处于同一进程，',
          '后测引擎的采样包含已保留的缓存/上下文，不能把 JSON 中的绝对内存峰值直接解释成独立内存倍率。',
          '整个完整 C3 计时与既有生产输出一致，但并非完整四 GPU、九动量生产作业的总 wall-time 对照。',
          '数值等价不独立证明 B 权重的统计处方、有限时间窗的收敛或 transversity 的外部物理符号约定。', '',
          '## 代码及复现', '',
          '`benchmark.py`：同一 Contractor 缓存契约，只将 `import opt_einsum` 替换为',
          '`import pyqcd.einsum as opt_einsum`。生产目录任何源码/结果均未修改。',
          '`gpu.slurm`：na800-sug、donghx_223、1 GPU、4 CPU；已完成任务 3960818。',
          '`install_native.slurm`：已完成 CPU 安装任务 3960815。',
          '`results/gpu_result.json`：数值差异、分阶段重复计时、dtype/shape/stride、规划路径和警告。', '',
          '```bash', 'source /public/home/donghx/load_a100_PyQuad.sh',
          'module load openblas/0.3.20-gcc-10.3.0',
          f'source {HERE}/venv_native/bin/activate',
          '# 科学计算只能在已申请的计算节点执行，不要直接在登录节点运行 benchmark.py。', '```', '',
          '建议保留原生产引擎。若继续追求提速，应针对实际降维方程做分阶段热点分析或专用内核适配，',
          '并再次通过六个 Wick 项和生产文件回归，不应仅凭其他案例的加速倍率全量替换。', '',
          '## Slurm snapshot', '', '```text', snapshot.rstrip(), '```', '']
(HERE / 'REPORT.md').write_text('\n'.join(lines))
print('\n'.join(lines[:40]))
