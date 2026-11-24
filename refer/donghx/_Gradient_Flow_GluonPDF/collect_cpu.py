"""Publish the joint handoff only after both independently verified analyses."""
from datetime import datetime
from common import ROOT, allocation, atomic_json, sha256
import json
import os


def main():
    allocation();records={}
    for key in ('L32x64','L32x96'):
        path=ROOT/key/'complete.json';value=json.loads(path.read_text())
        assert value['status'].startswith('COMPLETE') and value['nboot']==200
        records[key]=dict(path=str(path),sha256=sha256(path),nconf=value['nconf'],
            nboot=value['nboot'],flows=value['flows'],pz=value['pz'],fit_Tmin=value['fit_Tmin'],
            selected_available=value['selected_available'],point_count=value['point_count'],
            quality_no_fallback_points=value['quality_no_fallback_points'],
            diagnostic_fallback_points=value['diagnostic_fallback_points'],
            exact_symmetry_zero_points=value['exact_symmetry_zero_points'],
            no_finite_candidate_points=value['no_finite_candidate_points'])
    atomic_json(ROOT/'complete.json',dict(status='COMPLETE_TWO_ENSEMBLE_B_BOOT200_BARE_ANALYSIS',
        timestamp=datetime.now().astimezone().isoformat(),job=os.environ['SLURM_JOB_ID'],ensembles=records,
        scope='finite-flow connected B bare matrix elements; no normalization or matching',
        bootstrap_pairing='paired within each ensemble only; no cross-ensemble replica correlation'))
    lines=['# 两个 ensemble 的 B 方案：200-bootstrap 裸矩阵元', '',
        '## 已完成的范围', '',
        '| Ensemble | Nconf | Nboot | Pz | Tmin | 有限结果 | 无补选质量点 | 含诊断补选 |',
        '|---|---:|---:|---|---:|---:|---:|---:|']
    for key,d in records.items():
        lines.append(f'| {key} | {d["nconf"]} | 200 | {d["pz"]} | {d["fit_Tmin"]} | '
            f'{d["selected_available"]}/{d["point_count"]} | {d["quality_no_fallback_points"]} | {d["diagnostic_fallback_points"]} |')
    lines += ['', '固定模型为 $R(T,t)=M$；$Ta>0.6$ fm；$Q\ge10^{-3}$、$\chi^2/\mathrm{dof}\le10^3$。',
        '每个 bootstrap 独立执行 cut 内 AIC 窗口加权，再等权混合可用 cut。无通过候选时的最小统计误差补选单独标记。',
        '图中实心点为不依赖补选的质量结果；空心点为诊断补选/质量不完整；叉号为 odd z0 代数零（未拟合）。',
        '不能将有限覆盖率或空心点升级为可靠的平台信号。', '', '## 图片和数据入口', '']
    for key in records:
        lines += [f'### {key}', '',f'- [验收与质量汇总]({key}/RESULTS.md)',
            f'- [所有逐点结果]({key}/bare_matrix_progress.csv)',
            f'- [按通道/味/动量分组的质量覆盖率]({key}/coverage_by_channel_flavor_momentum.csv)',
            f'- `./{key}/ratios/`：200-bootstrap 复数 ratio。',
            f'- `./{key}/fits/`：裸矩阵元、200 样本、质量/补选和全部候选。', '']
        for channel in ('unpolarized','helicity','transversity'):
            for flavor in ('u','d','u_minus_d'):
                lines.append(f'- [{channel} {flavor}]({key}/plots/bare_{channel}_{flavor}_allflow_boot200.pdf)')
        lines += ['']
    lines += ['每份 PDF 两页：第 1 页完整 y 范围；第 2 页实部 [-1,2]、虚部 [-2,1] 的固定视窗。',
        '所有 z/a=0..20 均保存；不同 flow 的 x 坐标略错开，误差棒不连线。', '',
        'L32x64 保留 tau=0.1..0.8 全部已算 flow（其中 tau=0.1 不满足此前 a64<rf 的半径窗口，',
        '本次只做裸拟合，未以此删点）。L32x96 为与 L32x64 tau=0.2..0.8 相同物理流半径的七点。',
        'L32x64 的 Pz=3 仅有原前20conf，未混入50conf结果。两个 ensemble 的 bootstrap 行不能逐行相减作配对差。', '',
        '本次未做局域归一化、ringed Zchi、线减除、gradient-flow matching、flow-time/连续极限外推或 LaMET。',
        'connected u/d 非完整 singlet PDF；transversity 相对符号仍待外部物理定义确认。', '',
        f'生成时间：{datetime.now().astimezone().isoformat()}。']
    (ROOT/'DELIVERY.md').write_text('\n'.join(lines)+'\n')
    print('TWO_ENSEMBLE_BOOT200_BARE_DELIVERY_COMPLETE',flush=True)


if __name__=='__main__':
    main()
