#!/usr/bin/env python3
"""Audit and submit missing spatial (x/y) flowed-gluon OPE artifacts.

The audit is deliberately immutable: it writes a dated plan directory containing
the manifest hash, script hashes, inventory and a frozen task list.  ``submit``
only consumes that plan and never mutates existing output artifacts.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, shutil, subprocess, sys
from datetime import datetime
from pathlib import Path
import numpy as np

TAUS = (0.1,0.2,0.3,0.4,0.5,0.6,1.0,1.4,1.8,2.2,2.6,3.0,3.4,3.8)
DIRECTIONS = (0,1,2)

def tag(t): return f"tau{t:.3f}".replace('.', 'p')
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()
def base(root, conf, tau, eps, zdir):
    return root/f"conf{conf}"/tag(tau)/f"conf{conf}_{tag(tau)}_eps{eps:.3f}_zdir{zdir}"
def pair_state(b, conf=None, tau=None, zdir=None, z_count=25, nt=96, deep=False):
    n,j=Path(str(b)+'.npy'),Path(str(b)+'.json')
    if not n.is_file() and not j.is_file(): return 'missing'
    if not n.is_file() or not j.is_file(): return 'partial'
    # Every production task runs validate_output_v5 before success.  Presence
    # is therefore enough for the fast inventory; --deep rechecks headers and
    # metadata, which is intentionally slow on the shared filesystem.
    if not deep:
        return 'complete'
    try:
        with j.open() as f: meta=json.load(f)
        expected=(2,2,4,6,z_count,nt)
        with n.open('rb') as nf:
            version=np.lib.format.read_magic(nf)
            if version == (1,0): shape,fortran,dtype=np.lib.format.read_array_header_1_0(nf)
            elif version == (2,0): shape,fortran,dtype=np.lib.format.read_array_header_2_0(nf)
            else: return 'invalid'
        if tuple(shape) != expected or np.dtype(dtype) != np.dtype('complex128') or fortran: return 'invalid'
        if conf is not None and str(meta.get('conf_id')) != str(conf): return 'invalid'
        if tau is not None and abs(float(meta.get('flow',{}).get('tau_t_over_a2',-1))-tau)>1e-12: return 'invalid'
        # v5 metadata stores the Wilson-line direction inside ``operator``;
        # older drafts used a top-level z_direction key.  Accept only an
        # explicit match (never silently accept a missing direction).
        saved_zdir = meta.get('operator', {}).get('z_dir', meta.get('z_direction'))
        if zdir is not None and str(saved_zdir) != str(zdir): return 'invalid'
    except Exception: return 'invalid'
    return 'complete'

def audit(a):
    manifest=a.manifest.resolve(); root=a.output_root.resolve()
    rows=list(csv.DictReader(manifest.open(), delimiter='\t'))
    if not rows: raise SystemExit('empty manifest')
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    plan=(a.plan_root.resolve()/f'resume_spatial_v5_{stamp}')
    plan.mkdir(parents=True, exist_ok=False)
    frozen_manifest = plan/'manifest.tsv'
    shutil.copy2(manifest, frozen_manifest)
    tasks=[]; inventory=[]; partial=[]
    for r in rows:
        conf=r['conf_id']
        for tau in TAUS:
            for zdir in DIRECTIONS:
                st=pair_state(base(root,conf,tau,a.epsilon,zdir),conf,tau,zdir,
                              deep=(a.deep or conf == '1000'))
                inventory.append({'conf_id':conf,'tau':tau,'zdir':zdir,'state':st})
                if zdir in (0,1) and st=='missing':
                    tasks.append({'task_index':r['task_index'],'conf_id':conf,'gauge_path':r['gauge_path'],'tau':tau,'zdir':zdir})
                elif zdir in (0,1) and st=='partial': partial.append({'conf_id':conf,'tau':tau,'zdir':zdir})
    task_path=plan/'tasks.tsv'
    with task_path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['task_index','conf_id','gauge_path','tau','zdir'],delimiter='\t'); w.writeheader(); w.writerows(tasks)
    report={'schema':'gradient_flow_gluon_ope_resume_plan_v1','created':stamp,'manifest':str(manifest),'frozen_manifest':str(frozen_manifest),'output_root':str(root),'epsilon':a.epsilon,'fine_epsilon':a.fine_epsilon,'taus':list(TAUS),'directions':list(DIRECTIONS),'task_count':len(tasks),'partial_count':len(partial),'inventory':inventory,'partial':partial}
    (plan/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    hash_paths=[manifest,frozen_manifest,Path(__file__),Path(__file__).with_name('run_one_v5.py'),Path(__file__).with_name('validate_output_v5.py'),Path(__file__).with_name('compare_epsilon.py')]
    binary=(a.repo.resolve()/'build'/'flowed_gluon_ope_v5')
    if binary.is_file(): hash_paths.append(binary)
    hashes={str(p):sha(p) for p in hash_paths}
    (plan/'hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(f'PLAN={plan} tasks={len(tasks)} partial={len(partial)}')

def sbatch(cmd):
    out=subprocess.check_output(cmd,text=True).strip(); return out.split(';')[0]
def submit(a):
    plan=a.plan.resolve(); report=json.loads((plan/'report.json').read_text()); hashes=json.loads((plan/'hashes.json').read_text())
    for p,h in hashes.items():
        if not Path(p).is_file() or sha(p)!=h: raise SystemExit(f'plan hash mismatch: {p}')
    tasks=plan/'tasks.tsv'; repo=a.repo.resolve(); root=Path(report['output_root']); manifest=Path(report.get('frozen_manifest',report['manifest']))
    if int(report.get('task_count', 0)) == 0:
        print('plan has no missing tasks; nothing to submit')
        return
    # Best-effort duplicate guard (clusters with restricted squeue may deny access).
    try:
        q=subprocess.run(['squeue','-h','-o','%i %j'],text=True,capture_output=True,check=False)
        active=[x for x in q.stdout.splitlines() if 'gf_gOPE' in x]
        if active and not a.allow_existing: raise SystemExit('existing gf_gOPE jobs found; use --allow-existing')
    except FileNotFoundError: pass
    gate_jobs=[]; eps=float(report['epsilon']); fine=float(report['fine_epsilon'])
    for tau in TAUS:
        # gates are direction-qualified; create missing conf1000 pilot coarse/fine first.
        deps=[]
        for zdir in (0,1):
            cb,fb=base(root,'1000',tau,eps,zdir),base(root,'1000',tau,fine,zdir)
            pilot=[]
            for eb in (eps,fine):
                b=base(root,'1000',tau,eb,zdir)
                if pair_state(b,'1000',tau,zdir)=='missing':
                    jid=sbatch(['sbatch','--parsable','--array=0-0',f'--export=ALL,REPO_DIR={repo},MANIFEST={manifest},OUTPUT_ROOT={root},FLOW_TAU={tau},FLOW_EPSILON={eb},Z_DIR={zdir},Z_COUNT={a.z_count}',str(repo/'slurm/v5/flowed_ope_cpu_v5.slurm')]); pilot.append(jid)
                elif pair_state(b,'1000',tau,zdir)!='complete': raise SystemExit(f'partial/invalid pilot artifact: {b}')
            gate=plan/f'gate_conf1000_{tag(tau)}_zdir{zdir}.json'
            gate_cmd=['sbatch','--parsable']
            if pilot: gate_cmd.append(f'--dependency=afterok:{":".join(pilot)}')
            gate_cmd += [f'--export=ALL,REPO_DIR={repo},COARSE_BASE={cb},FINE_BASE={fb},GATE_REPORT={gate}',str(repo/'slurm/v5/compare_epsilon_cpu_v5.slurm')]
            gj=sbatch(gate_cmd)
            deps.append(gj)
        gate_jobs.extend(deps)
    # 32 static workers; each reads the immutable frozen task list and round-robins rows.
    n=int(a.workers)
    if n<1: raise SystemExit('workers must be positive')
    receipt=repo/'runs'/f"resume_spatial_v5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"; receipt.parent.mkdir(exist_ok=True)
    with receipt.open('w') as f:
        f.write(f'plan={plan}\ntasks={report["task_count"]}\nworkers={n}\n')
        for wid in range(n):
            dep=':'.join(gate_jobs)
            worker_cmd=['sbatch','--parsable']
            if dep: worker_cmd.append(f'--dependency=afterok:{dep}')
            worker_cmd += [f'--export=ALL,REPO_DIR={repo},MANIFEST={manifest},PLAN_DIR={plan},TASKS={tasks},OUTPUT_ROOT={root},WORKER_ID={wid},WORKERS={n},Z_COUNT={a.z_count},FLOW_EPSILON={eps}',str(repo/'slurm/v5/flowed_ope_batch_resume_v5.slurm')]
            jid=sbatch(worker_cmd)
            f.write(f'worker={wid} job={jid}\n'); f.flush(); print(f'SUBMITTED worker={wid} job={jid}')
    print(f'RECEIPT={receipt}')

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True)
    q=sub.add_parser('audit'); q.add_argument('--manifest',type=Path,default=Path('manifest.tsv')); q.add_argument('--output-root',type=Path,default=Path('output_v5')); q.add_argument('--plan-root',type=Path,default=Path('runs')); q.add_argument('--repo',type=Path,default=Path('.')); q.add_argument('--epsilon',type=float,default=.01); q.add_argument('--fine-epsilon',type=float,default=.005); q.add_argument('--deep',action='store_true',help='validate every existing JSON/NPY header (slow on NFS)')
    s=sub.add_parser('submit'); s.add_argument('--plan',type=Path,required=True); s.add_argument('--repo',type=Path,default=Path('.')); s.add_argument('--workers',type=int,default=32); s.add_argument('--z-count',type=int,default=25); s.add_argument('--allow-existing',action='store_true'); s.add_argument('--epsilon',type=float,default=.01); s.add_argument('--fine-epsilon',type=float,default=.005)
    a=p.parse_args(); audit(a) if a.cmd=='audit' else submit(a)
if __name__=='__main__': main()
