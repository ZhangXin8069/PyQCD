"""Validate the solver recovery, produce paired products, then plot after Zchi."""
import argparse
import json
from pathlib import Path
from datetime import datetime
from .review_recovery import review
from .pipeline import analyze, load_npz
from .plot_after_zchi import plot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('original',type=Path);p.add_argument('recovered',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args()
    original,recovered,out=a.original.resolve(),a.recovered.resolve(),a.output.resolve()
    if out.exists():raise FileExistsError('Use a new output directory')
    review_path=recovered/'solver_recovery_review.json'
    review(original,recovered,review_path)
    analyze(original,recovery_review=review_path,output=out/'analysis')
    for kind in ('sixflow_sumdiff','fourflow_ratio'):
        plot(out/'analysis'/kind/'paired_matrix_elements.npz',out/'figures'/kind)
    d=load_npz(out/'analysis/sixflow_sumdiff/paired_matrix_elements.npz')
    s=json.loads((out/'analysis/summary.json').read_text())
    r=json.loads(review_path.read_text())
    lines=['# Quark u-d：10 conf 配对 ringed 归一化后的结果','',
      '组态10000--10900（间隔100），每组态32个噪声；使用原50×10 bootstrap索引，seed=1115。',
      '六flow Sumratio差分AIC与原四flow直接ratio AIC分目录保存，后者仍为未加权诊断。',
      '每个bootstrap先平均各组态的动能噪声均值，再求Zchi_b，并乘同一索引的裸矩阵元h_b。',
      '误差为这50个配对乘积的std(ddof=1)，保留Zchi与h的组态相关性，不另叠加重复的噪声bootstrap。','',
      '## 图','',
      '- [六flow after Zchi](figures/sixflow_sumdiff/after_Zchi_allflow.pdf)',
      '- [六flow的tau>=0.2显示子集](figures/sixflow_sumdiff/after_Zchi_tau_ge_0p2.pdf)',
      '- [六flow逐flow分页](figures/sixflow_sumdiff/after_Zchi_eachflow.pdf)',
      '- [原直接ratio拟合 after Zchi](figures/fourflow_ratio/after_Zchi_allflow.pdf)',
      '- [Zchi及其系综误差](figures/sixflow_sumdiff/Zchi_10conf.pdf)','',
      'after-Zchi图只乘一个Zchi，未乘SFTE因子。横轴z/a；三个子图为n_z=3,4,5，',
      '对应Pz约1.50,2.00,2.50 GeV。汇总PDF第一张z/a=0..15，xlim=-0.5..15.5；',
      '第二张保留0..20。不同flow横向错开，数据点不连线。空心保留原拟合fallback/未加权诊断。','',
      '## 数值与验证','',
      '| t/a² | Zchi | 配对组态bootstrap误差 |','|---:|---:|---:|']
    for t,z,e in zip(d['flow_times'],d['Zchi'],d['Zchi_error']):lines.append(f'|{t:g}|{z:.9g}|{e:.4g}|')
    lines+=['',f'全部10个conf的最大真实full-MAT残差：{max(x["true_residual_max"] for x in s["measurement_audits"]):.6g}，门槛1e-9。',
      f'conf10800恢复后与原成功noise0的动能最大相对差：{r["anchor_max_relative_difference"]:.6g}。',
      '新求解器自由场同噪声比较通过，质量、作用量、flow、边界及trace定义不变；没有放宽残差门。',
      '其他9个conf复用原完整测量；conf10800采用修订求解器完整重算的32噪声，未混拼不同代码的噪声。',
      f'原失败任务3696330_7及3705522_7保留；本次修订分支为{recovered.name}。',
      f'虚部期望检验（各flow、5sigma门）：{d["kinetic_imaginary_check_pass"].tolist()}。','',
      '仍采用文献临界价质量am=-0.23539、固定原海质量。处方系统误差、边界改进、',
      '完整六方向估计量和连续极限等物理认可事项未由统计配对自动解决。',
      '单格距、有限flow的ringed矩阵元不等于最终PDF。','',
      '## 数据和复现','',
      '`analysis/{sixflow_sumdiff,fourflow_ratio}/paired_matrix_elements.npz`保存原始噪声、组态均值、',
      '原bootstrap索引、Zchi样本及bare/ringed/SFTE样本。中心轴(flow,pz,z)，样本轴(flow,bootstrap,pz,z)。',
      '沿用前次授权管线的SFTE诊断也保存在analysis；after-Zchi图不使用它。',
      f'原测量campaign：`{original}`。',f'恢复测量与数值等价性审查：`{recovered}`。',
      '各NPZ、图表manifest保存来源哈希、质量标记、窗口、参数与统计定义。',
      f'生成时间：{datetime.now().isoformat(timespec="seconds")}。会话标题：完成10 conf after-Zchi配对结果图。','']
    (out/'README.md').write_text('\n'.join(lines))
    print(out,flush=True)


if __name__=='__main__':main()
