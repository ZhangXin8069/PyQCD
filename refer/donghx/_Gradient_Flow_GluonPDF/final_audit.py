"""Audit retry payloads and build a unique configuration selection using frozen original audits."""
from pathlib import Path
from datetime import datetime
import hashlib
import importlib.util
import json
import os
import traceback

ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 p=Path(p);tmp=p.with_name(p.name+'.tmp')
 tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)

def main():
 assert os.environ.get('SLURM_JOB_ID'), 'CPU Slurm allocation required'
 m=read(ROOT/'retry_manifest.json')
 reports=[]
 for ens in m['ensembles']:
  errors=[];audits=[];selection={};retry_complete=set();expected=set(ens['retry_conf_ids'])
  assert sha(ens['original_audit'])==ens['original_audit_sha256']
  old=read(ens['original_audit'])
  original_ids=set()
  for oldroot in old['roots']:
   r=Path(oldroot['root']);lock=oldroot['source_lock_sha256']
   assert sha(r/'source_lock.json')==lock
   for name,digest in read(r/'source_lock.json')['sha256'].items():assert sha(r/name)==digest
   original_ids.update(x['conf_id'] for x in read(r/'request.json')['items'])
   # Original payloads were independently audited; recheck their completion/result identities.
   for entry in oldroot['complete']:
    rp=Path(entry['run']);marker=read(rp/'complete.json')
    assert not (rp/'failed.json').exists()
    assert marker['source_lock_sha256']==lock
    assert marker['result_sha256']==sha(rp/'result.json')
    assert entry['conf_id'] not in selection
    selection[entry['conf_id']]=dict(**entry,source='original_validated',source_lock_sha256=lock,
        result_sha256=marker['result_sha256'])
  for rootrec in ens['roots']:
   r=Path(rootrec['root'])
   assert sha(r/'source_lock.json')==rootrec['source_lock_sha256']
   try:
    spec=importlib.util.spec_from_file_location('payload_audit',r/'audit_results.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    audit=module.audit(r);audits.append(audit)
    errors.extend(audit['payload_audit_errors'])
    for entry in audit['complete']:
     conf=entry['conf_id'];assert conf in expected and conf not in selection
     marker=read(Path(entry['run'])/'complete.json')
     selection[conf]=dict(**entry,source='retry_fixed_alpha_validated',source_lock_sha256=audit['source_lock_sha256'],
         result_sha256=marker['result_sha256'])
     retry_complete.add(conf)
   except Exception:
    errors.append(dict(root=str(r),traceback=traceback.format_exc()))
  missing=sorted(original_ids-set(selection),key=int)
  assert set(selection)<=original_ids
  report=dict(ensemble=ens['ensemble'],original_requested=ens['original_requested'],
      original_complete=ens['original_complete'],retry_requested=len(expected),retry_complete=len(retry_complete),
      configurations_complete=len(selection),configurations_incomplete=len(missing),missing_conf_ids=missing,
      payload_audit_errors=errors,status='AUDIT_ERROR' if errors else 'PARTIAL_VALIDATED' if missing else 'PASS_ALL_COMPLETE',
      original_audit=ens['original_audit'],original_audit_sha256=ens['original_audit_sha256'],
      original_payload_verification='Reused frozen full-payload audit; source locks and current complete/result hashes rechecked.',
      retry_payload_verification='Full independent start0, precision, residual, NPZ, loop and Slurm audit.',
      selection=[selection[k] for k in sorted(selection,key=int)],retry_audits=audits,
      solver_policy_mixed=True,ensemble_linear_divergence_fit_performed=False)
  assert len(selection)+len(missing)==ens['original_requested']
  write(ROOT/(ens['ensemble']+'_merged_audit.json'),report);reports.append(report)
  print(ens['ensemble'],report['status'],len(selection),'/',ens['original_requested'],flush=True)
 result=dict(checked_at=datetime.now().astimezone().isoformat(),audit_job=os.environ['SLURM_JOB_ID'],
      status='PASS_ALL_COMPLETE' if all(r['status']=='PASS_ALL_COMPLETE' for r in reports) else 'INCOMPLETE_OR_AUDIT_ERROR',
      retry_requested=37,retry_complete=sum(r['retry_complete'] for r in reports),
      ensembles=[{k:v for k,v in r.items() if k not in ['selection','retry_audits']} for r in reports])
 write(ROOT/'completion_audit.json',result)
 return 0 if result['status']=='PASS_ALL_COMPLETE' else 2

if __name__=='__main__':raise SystemExit(main())
