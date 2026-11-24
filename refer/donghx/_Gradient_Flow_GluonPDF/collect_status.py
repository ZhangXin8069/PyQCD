"""Read-only Slurm/text/JSON status collection; never load scientific arrays."""
import collections
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import re
import signal
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
stamp = datetime.now().astimezone().isoformat()
job_ids = ['3929748','3931036','4001960','4001961','4014488','4014500','4014510','4014511','4014512','4014513','4016165','4016174','4016199','4016204','4016208','4002249','4002250','4002251','4002252','4002253','4002254','4002255','4002256','4002257','4002258','4002259','4002260','4000769','4000770','3993639','3993641','3993642','4001299','4001300']
commands = {
    'squeue': ['squeue','-u','donghx','-o','%.24i %.32j %.14P %.11T %.12M %.36R'],
    'sacct_recent_na100_ins': ['sacct','-X','-S','2026-11-16','-u','donghx','-r','na100-ins','-o','JobIDRaw,JobID%25,JobName%35,Partition,State,ExitCode,Elapsed,Start,End,NodeList%24','--parsable2'],
    'sacct_selected': ['sacct','-X','-j',','.join(job_ids),'-o','JobIDRaw,JobID%25,JobName%35,Partition,State,ExitCode,Elapsed,Start,End,NodeList%24','--parsable2'],
    'sinfo': ['sinfo','-N','-n','gpu025,gpu026,gpu027,gpu029','-o','%N %P %t %E'],
    'reservations': ['scontrol','-o','show','reservation'],
}
raw={}
for name,command in commands.items():
    result=subprocess.run(command,capture_output=True,text=True,timeout=40)
    (OUT/(name+'.txt')).write_text(result.stdout)
    (OUT/(name+'.stderr.txt')).write_text(result.stderr)
    raw[name]={'command':command,'returncode':result.returncode,'file':str(OUT/(name+'.txt'))}
    if result.returncode:
        raise RuntimeError((name,result.returncode,result.stderr))

def read(p):
    return json.loads(p.read_text())

def brief_audit(p):
    if not p.exists():
        return {'exists':False,'path':str(p)}
    d=read(p)
    out={k:v for k,v in d.items() if not isinstance(v,(dict,list))}
    out.update(exists=True,path=str(p))
    if 'task_results' in d:
        out['task_result_counts']=dict(collections.Counter(x.get('status') for x in d['task_results']))
    if isinstance(d.get('per_ensemble'),dict):
        out['per_ensemble']=d['per_ensemble']
    if isinstance(d.get('ensembles'),dict):
        out['ensembles']={e:{k:v for k,v in row.items() if not isinstance(v,(dict,list))} for e,row in d['ensembles'].items() if isinstance(row,dict)}
    return out

campaigns={}
new_name='quark_l24x72_B_conf121to200_matched_radius_20261116'
for name in ['quark_l24x72_B_50conf_matched_radius_20261104_v2','quark_l24x72_B_conf51to100_matched_radius_20261105','quark_l24x72_B_conf101to120_matched_radius_20261114',new_name]:
    base=ROOT/name
    manifest=read(base/'manifest.json')
    digest=hashlib.sha256((base/'manifest.json').read_bytes()).hexdigest()
    valid={}
    invalid=[]
    for marker in (base/'status').glob('task*.json'):
        if not re.fullmatch(r'task\d+\.json',marker.name):
            continue
        idx=int(marker.stem[4:])
        d=read(marker)
        if (d.get('status')=='PASS' and d.get('task')==idx and
            d.get('case_count')==252 and d.get('manifest_sha256')==digest):
            valid[idx]=d
        else:
            invalid.append(marker.name)
    conf_counts=collections.Counter(str(manifest['tasks'][i]['conf']) for i in valid)
    item=dict(expected_confs=len(manifest['confs']),expected_tasks=len(manifest['tasks']),
              valid_PASS_task_markers=len(valid),invalid_markers=invalid,
              fully_completed_confs=sorted((k for k,v in conf_counts.items() if v==7),key=int),
              partially_completed_confs={k:v for k,v in conf_counts.items() if v!=7},
              case_count_in_completed_tasks=252*len(valid),
              manifest_sha256=digest,
              final_audit=brief_audit(base/'summary/complete.json'))
    campaigns[name]=item
    if name==new_name:
        fields=['task','conf','flow_time_a2','status','worker']
        with (OUT/'remaining_conf_flow_tasks.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader()
            for idx,task in enumerate(manifest['tasks']):
                if idx not in valid:
                    writer.writerow(dict(task=idx,conf=task['conf'],flow_time_a2=task['flow_time_a2'],status='NO_COMPLETE_PASS_MARKER',worker=idx%24))
        item['remaining_tasks']=len(manifest['tasks'])-len(valid)
        item['per_flow_completed_tasks']=dict(collections.Counter(str(manifest['tasks'][i]['flow_time_a2']) for i in valid))
        item['verification_scope']='Checked task marker status, task identity, case_count and manifest hash; no new numerical/hash audit of scientific output arrays in this status check.'
        tails=[]
        for logfile in sorted((base/'logs').glob('gpu*.err')):
            text=logfile.read_text(errors='replace')
            tails.append({'path':str(logfile),'bytes':logfile.stat().st_size,'last_lines':text.splitlines()[-8:]})
        (OUT/'production_stderr_tails.json').write_text(json.dumps(tails,ensure_ascii=False,indent=2)+'\n')

aux={}
for name in ['gluon_flow_three_ensemble_matched_rf_gpu025_20261114','gluon_flow_L24x72_gpu025_tau06_12_20261114','gluon_flow_gpu025_dense_tau_20261113_v2','hyp4d10_landau_wilson_100conf_20261114']:
    aux[name]=brief_audit(ROOT/name/'completion_audit.json')
aux['quark_51to100_saved_outputs_only']=brief_audit(ROOT/'quark_l24x72_status_20261116/saved_outputs_complete.json')
bench=ROOT/'quark_c3c2_explicit_dp_benchmark_20261116'
aux['explicit_dp_benchmark']={
    'warm_status':read(bench/'results/summary.json')['status'],
    'fresh_status':read(bench/'results/fresh_summary.json')['status'],
    'delivery_status':read(bench/'delivery.json')['status'],
    'production_modified':read(bench/'delivery.json')['production_modified'],
}
rows=list(csv.DictReader(io.StringIO((OUT/'sacct_recent_na100_ins.txt').read_text()),delimiter='|'))
latest=[x for x in rows if x['JobName']=='q24_B200_flow']
summary=dict(generated=stamp,raw_commands=raw,campaigns=campaigns,auxiliary=aux,
             latest_production_scheduler_state_counts=dict(collections.Counter(x['State'] for x in latest)),
             latest_production_exit_code_counts=dict(collections.Counter(x['ExitCode'] for x in latest)),
             signal7_name=signal.Signals(7).name,
             failure_window='2026-11-17T09:43:29+08:00 to 2026-11-17T09:43:51+08:00',
             root_cause='UNCONFIRMED; 6 active workers signal 7 and 18 later launches exit 1, clustered on three nodes. No explicit numerical exception found in latest quark logs.',
             node_state_caveat='maint is also explained by existing year-long MAINT reservations (including donghx_250); this alone is not evidence of new maintenance today.',
             actions='Status inspection only; no production code changes, cancellation or resubmission.')
(OUT/'status_snapshot.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'generated':stamp,'latest_production':campaigns[new_name],
                  'state_counts':summary['latest_production_scheduler_state_counts'],
                  'exit_codes':summary['latest_production_exit_code_counts'],
                  'snapshot':str(OUT/'status_snapshot.json')},ensure_ascii=False,indent=2))
