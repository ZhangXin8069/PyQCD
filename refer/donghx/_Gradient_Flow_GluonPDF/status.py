"""Metadata/Slurm-only progress report; safe on login, no array payload reads."""
from collections import Counter
import csv
from datetime import datetime
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent


def main():
    plan=json.loads((ROOT/'input_plan.json').read_text());record=json.loads((ROOT/'submission.json').read_text())
    rows=[];summary=[]
    for key,m in plan['ensembles'].items():
        nb=nf=0
        for i,f in enumerate(m['flows']):
            name='tau'+format(f['tau'],'.8f').replace('.','p')
            p=ROOT/key/'status'/f'ratio_{name}.json'
            passed=p.exists() and json.loads(p.read_text()).get('status')=='PASS'
            nb+=int(passed)
            fits=0
            for j in range(i*9,(i+1)*9):
                q=ROOT/key/'status'/f'fit_{j}.json'
                fits+=int(q.exists() and json.loads(q.read_text()).get('status')=='PASS')
            nf+=fits
            rows.append(dict(ensemble=key,nconf=len(m['confs']),nboot=200,tau64=f['tau64'],tau_a2=f['tau'],
                tf_fm2=f['tau64']*.0897**2,rf_fm=(8*f['tau64'])**.5*.0897,
                bootstrap_complete=passed,bare_fits_complete=fits,bare_fits_expected=9))
        summary.append(dict(ensemble=key,nconf=len(m['confs']),nboot=200,boot_complete=nb,boot_expected=len(m['flows']),
            fits_complete=nf,fits_expected=len(m['flows'])*9))
    numeric=sum((ROOT/'00_audit'/f'numeric_{i}.json').exists() for i in range(9))
    plot_audit=Path(record.get('plot_repair_root',str(ROOT)))/'00_audit'
    plots=sum((plot_audit/f'plots_{i}.json').exists() for i in range(9))
    ids=','.join(v['jobid'] for v in record['jobs'].values())
    command=['sacct','-X','-j',ids,'--format=JobIDRaw,JobName,State,ExitCode,Elapsed,NodeList','-n','-P']
    result=subprocess.run(command,check=False,text=True,capture_output=True,timeout=30)
    slurm=result.stdout if result.returncode==0 else 'SLURM_QUERY_UNAVAILABLE: '+result.stderr
    snapshot=dict(timestamp=datetime.now().astimezone().isoformat(),summary=summary,
        numerical_channel_flavor_complete=numeric,plots_channel_flavor_complete=plots,
        final_complete=(ROOT/'complete.json').exists(),slurm_records=slurm)
    (ROOT/'00_audit/progress_snapshot.json').write_text(json.dumps(snapshot,indent=2,ensure_ascii=False)+'\n')
    with (ROOT/'00_audit/progress.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# 三 ensemble CPU 计算状态','',snapshot['timestamp'],'',
        '|ensemble|conf|bootstrap|flow bootstrap 已完成|裸拟合已完成|','|---|---:|---:|---:|---:|']
    for r in summary:lines.append(f'|{r["ensemble"]}|{r["nconf"]}|200|{r["boot_complete"]}/{r["boot_expected"]}|{r["fits_complete"]}/{r["fits_expected"]}|')
    lines+=['',f'全阶段数值任务：{numeric}/9；图片任务：{plots}/9；最终 complete.json：{snapshot["final_complete"]}。','',
        '新生产数组运行并不等于200 bootstrap或连续极限已完成。完整验收以最终complete.json为准。','',
        '## 已提交作业','', '|阶段|JobID|','|---|---|']
    lines += [f'|{k}|{v["jobid"]}|' for k,v in record['jobs'].items()]
    lines += ['', '## Slurm 快照','', '```text',slurm.rstrip(),'```','',
        '再次运行 `python -B status.py` 更新元数据进度，不读取任何科学数组。','',
        '方案/约定见README.md；账户提交上限处理见SUBMISSION_NOTES.md；尚未闭合的系统误差见00_audit/KNOWN_SYSTEMATICS.md。']
    (ROOT/'RUN_STATUS.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:12]));print('numeric',numeric,'plots',plots,'final',snapshot['final_complete'])


if __name__=='__main__':main()
