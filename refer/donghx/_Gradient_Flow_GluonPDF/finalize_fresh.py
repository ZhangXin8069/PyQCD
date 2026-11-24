"""Combine warm execution and production-style fresh-planning measurements."""
import csv
import hashlib
import json
import os
from pathlib import Path

HERE=Path(__file__).resolve().parent

def main():
    assert 'SLURM_JOB_ID' in os.environ
    fresh=json.loads((HERE/'results/fresh_summary.json').read_text())
    warm=json.loads((HERE/'results/summary.json').read_text())
    assert fresh['status']==warm['status']=='PASS'
    max_error=0.
    rows=[]
    for c in fresh['cases']:
        assert c['status']=='PASS'
        for mode,samples in c['samples'].items():
            assert len(samples)==3
            for sample in samples:
                max_error=max(max_error,sample['correctness']['C2']['relative_L2'],sample['correctness']['Wick']['relative_L2'],
                    *(v['relative_L2'] for v in sample['correctness']['each_wick']))
                rows.append(dict(flow_time_a2=c['flow_time_a2'],mode=mode,repeat=sample['repeat'],
                    c2_oracle_seconds=sample['c2_oracle_seconds'],c3_case_seconds=sample['c3_case_seconds'],sum_seconds=sample['sum_seconds']))
    assert max_error<2e-10
    with (HERE/'fresh_timings.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Quark gradient flow 显式 dp 测试：生产 case 的实际收益','',
        '已完成两类A100测试：缓存后的收缩执行，以及按生产方式为每个case重新创建Contractor。',
        '第二类最接近当前驱动：C2与C3各自有独立的新Contractor，GPU内核提前预热，计时包含路径搜索和表达式创建。','',
        f"原始热执行测试任务：{warm['environment']['job']}；生产式新case测试任务：{fresh['job']}。",
        '均在na100-ins/gpu025，1张A100-SXM4-80GB，reservation=donghx_250。',
        '软件环境：CuPy 14.1.1、NumPy 2.2.6、opt_einsum 3.4.0。',
        '样本是L24x72/conf30000的最小/最大flow，全部7个动量、16个z、6个Wick项，NE150/NV650；各3轮，顺序交替。','',
        '## 每个case重新规划：最接近现有生产方式','',
        '单位秒，为三轮中位数。C2列包含降维C2、完整旋量C2校验及误差检查；C3为完整sigma/动量循环。','',
        '|tau/a²|当前optimal C2|显式dp C2|当前optimal C3|显式dp C3|C2+C3合计加速|合计耗时降低|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for c in fresh['cases']:
        a=c['medians']['current_optimal'];b=c['medians']['explicit_dp'];s=c['speedups']['sum_seconds']
        lines.append(f"|{c['flow_time_a2']:.8f}|{a['c2_oracle_seconds']:.6f}|{b['c2_oracle_seconds']:.6f}|{a['c3_case_seconds']:.6f}|{b['c3_case_seconds']:.6f}|{s:.4f}×|{100*(1-1/s):.2f}%|")
    lines.extend(['','## 路径缓存后的执行','',
        '这里两种实现都复用相同形状的规划；说明纯数值执行的差别。','',
        '|tau/a²|optimal C2 reduced|dp C2 reduced|optimal C3|dp C3|C3耗时降低|',
        '|---|---:|---:|---:|---:|---:|'])
    for i in range(len(fresh['cases'])):
        c=json.loads((HERE/'results'/f'case{i}.json').read_text());a=c['modes']['current_optimal']['warm_median_seconds'];b=c['modes']['explicit_dp']['warm_median_seconds']
        lines.append(f"|{c['flow_time_a2']:.8f}|{a['c2_reduced']:.6f}|{b['c2_reduced']:.6f}|{a['c3_case']:.6f}|{b['c3_case']:.6f}|{100*(1-b['c3_case']/a['c3_case']):.2f}%|")
    lines.extend(['','## 正确性和解释','',
        f'所有生产式新case重复均通过已保存生产结果检查，C2、逐个六Wick项和总Wick的最大相对L2差为{max_error:.3e}，阈值2e-10。',
        '原热执行任务还检查了三个通道投影；CPU独立对照覆盖完整旋量C2、六Wick项、未因子化时间求和C3及显式路径传递/缓存。',
        '当前生产本来就是opt_einsum optimal，使用contract_expression；不是从greedy升级。',
        '本次L24尺寸下，dp与optimal的若干路径不同，但它们的估计FLOPs和最大中间张量相同，纯执行收益较小。',
        'dp最明确的收益在于更快找到路径。当前生产每个case新建Contractor，会反复支付规划开销，因此生产式case收益比热执行明显。',
        '规划实际时长、路径、方程profile和原始计时均保存。dp规划约0.03秒，而optimal约1秒，具体按各样本JSON读取。','',
        '## 使用边界','',
        '上述加速是输入准备完成后的C2/C3收缩部分；不包括输入I/O、VVV、vertex、flow/basis计算，不能当作整conf端到端加速。',
        '只对L24x72做了真实GPU测试。三ensemble共享相同收缩源码，但L32外部/流场维度与时间窗不同，需分别实测后再给加速比。',
        '现有生产源码和任务均保持原版本。本次提供独立候选ExplicitContractor，后续切换应建立新的版本和manifest。','',
        '复核入口：results/fresh_summary.json、results/fresh_case*.json、fresh_timings.csv；缓存后的完整对照见REPORT.zh-CN.md。',''])
    (HERE/'REPORT.production.zh-CN.md').write_text('\n'.join(lines))
    files=sorted(p for p in HERE.rglob('*') if p.is_file() and p.name not in ('delivery.json',) and 'logs' not in p.parts)
    delivery=dict(status='COMPLETE_BENCHMARK_REPORTS',job=os.environ['SLURM_JOB_ID'],
        gpu_jobs=[warm['environment']['job'],fresh['job']],production_modified=False,
        correctness_max_relative_L2=max_error,
        files_sha256={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (HERE/'delivery.json').write_text(json.dumps(delivery,indent=2)+'\n')
    print('FINAL_REPORT_PASS',max_error)
    print((HERE/'REPORT.production.zh-CN.md').read_text())

if __name__=='__main__':main()
