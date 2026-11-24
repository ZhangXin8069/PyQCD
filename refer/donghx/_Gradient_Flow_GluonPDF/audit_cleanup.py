"""Metadata-only, bounded cache inventory. Never reads lattice array payloads.

Deletion is a separate explicit-manifest operation; no wildcard/recursive delete.
Scientific outputs, basis, source, reference packages and active production are
excluded. This program is stdlib-only and suitable for a login-node audit.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time
from datetime import datetime

ROOT=Path('/public/group/lqcd/donghx/Gradient_Flow_GluonPDF')
HERE=ROOT/'workspace_cleanup_20261109'
FORBIDDEN={'.git','.agents','.codex','.aws','.ssh'}
SCIENCE={'data','raw','output','output_v5','output_L32x64_v1','outputs','results','Result',
    'C3_C2_schemeB','C3_schemeB_Pz3456_tsep7to12','C3_schemeB','C3_C2','C3','C2',
    'basis','basis_cache','flowed_basis','flowed_gauge','gauge_cache','flow_cache',
    'perambulators','peram','eigenvectors','eigensystem','vvv_cache','VVV','vertex',
    'vertex_schemeB','operators','packed','ratios','bootstrap','fits','logs','log',
    'plots','_plots','figures','figs','reference_data','references','reference',
    'reference_3ens','tools','lhapdf_comparison_20261025','_archive'}
CACHE_NAMES={'__pycache__','.pytest_cache','.mypy_cache','.ruff_cache','matplotlib_cache',
    'mplconfig','cupy_cache','quda_cache','worker_cache'}
ACTIVE_PREFIXES=('quark_l24x72_','quark_l32x96_B_')
SYNTHETIC_NAMES={'synthetic','integration','smoke','scratch','tmp','test_output','test_results'}
NOW=time.time()

def now():return datetime.now().astimezone().isoformat()

def dump(path,obj):
    temp=path.with_name('.'+path.name+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    os.replace(temp,path)

def table(path,rows,fields):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()

def meta(path):
    st=path.lstat()
    return dict(path=str(path.relative_to(ROOT)),size=st.st_size,allocated_bytes=st.st_blocks*512,
        mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns,inode=st.st_ino,device=st.st_dev,
        nlink=st.st_nlink,kind='symlink' if stat.S_ISLNK(st.st_mode) else
        'file' if stat.S_ISREG(st.st_mode) else 'directory' if stat.S_ISDIR(st.st_mode) else 'other')

def checked(relative):
    rel=Path(relative)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts or rel.parts[0] in FORBIDDEN:
        raise ValueError(f'Unsafe target {relative}')
    out=ROOT/rel
    for path in (out,*out.parents):
        if path==ROOT:break
        if path.is_symlink():raise ValueError(f'Symlink target/parent {path}')
    return out

def queue_snapshot():
    cmd=['squeue','-u','donghx','-h','-o','%i|%P|%j|%T|%M|%Z|%R']
    proc=subprocess.run(cmd,text=True,capture_output=True,timeout=25)
    if proc.returncode:raise RuntimeError(proc.stderr)
    jobs=[]
    for line in proc.stdout.splitlines():
        fields=line.split('|',6)
        if len(fields)!=7:raise RuntimeError('Unexpected squeue format')
        jobs.append(dict(zip(('id','partition','name','state','elapsed','workdir','node_or_reason'),fields)))
    return dict(timestamp=now(),command=cmd,jobs=jobs)

def active_conflict(relative, snapshot):
    """Protect known active families, job-specific paths and non-root workdirs."""
    rel=Path(relative)
    if rel.parts[0].startswith(ACTIVE_PREFIXES):
        return 'active production family'
    path=ROOT/rel
    tokens=set(re.findall(r'\d+', str(rel)))
    for job in snapshot['jobs']:
        if job['id'].split('_')[0] in tokens:
            return 'active Slurm job identifier in path'
        workdir=Path(job['workdir'])
        if workdir!=ROOT and workdir.is_relative_to(ROOT):
            if path.is_relative_to(workdir) or workdir.is_relative_to(path):
                return 'active Slurm working directory'
    return None

def walk_small_roots():
    """Discover cache roots, prune payloads BEFORE descending, depth<=7."""
    roots=[];notes=[];lockfiles=[];seen=0
    tops=sorted(ROOT.iterdir())
    for top in tops:
        if not top.is_dir() or top.is_symlink() or top.name in FORBIDDEN or top.name in SCIENCE or top==HERE:
            continue
        if top.name.startswith(ACTIVE_PREFIXES):
            notes.append(dict(path=top.name,reason='active/pending GPU production prefix; completely preserved'))
            continue
        for directory,dirs,files in os.walk(top,followlinks=False):
            base=Path(directory);seen+=1
            if seen%1000==0:print('AUDIT_DIRS',seen,str(base.relative_to(ROOT)),flush=True)
            depth=len(base.relative_to(top).parts)
            if base.name in CACHE_NAMES or base.name.startswith('scratch_'):
                roots.append(base);dirs[:]=[];continue
            if base.name in SYNTHETIC_NAMES:
                notes.append(dict(path=str(base.relative_to(ROOT)),reason='test/synthetic output; keep unless independently whitelisted'))
                dirs[:]=[];continue
            if depth>=7:
                dirs[:]=[];continue
            dirs[:]=[d for d in dirs if d not in SCIENCE and d not in FORBIDDEN and
                     not (base/d).is_symlink() and not d.startswith(('output','C3','C2','results','ratio_','ratios_','bootstrap_','Nev_','flowed_'))]
            for name in files:
                if 'lock' in name.lower() and name.endswith('.json'):
                    p=base/name
                    if not p.is_symlink() and p.stat().st_size<8*1024**2:lockfiles.append(p)
    return roots,notes,lockfiles,seen

def list_cache(path):
    for directory,dirs,files in os.walk(path,followlinks=False):
        base=Path(directory)
        dirs[:]=[d for d in dirs if not (base/d).is_symlink() and d not in FORBIDDEN]
        for name in files:
            p=base/name
            if not p.is_symlink():yield p

def plan():
    if (HERE/'cleanup_receipt.json').exists():raise RuntimeError('Preserve the executed plan; start a dated new audit')
    snapshot=queue_snapshot();dump(HERE/'active_jobs_before.json',snapshot)
    roots,notes,locks,seen=walk_small_roots()
    protected=set();lockstats=[]
    def collect(obj):
        if isinstance(obj,dict):
            for k,v in obj.items():collect(k);collect(v)
        elif isinstance(obj,list):
            for x in obj:collect(x)
        elif isinstance(obj,str) and obj.startswith(str(ROOT)+'/'):
            protected.add(obj)
    for p in locks:
        try:
            collect(json.loads(p.read_text()))
            lockstats.append(dict(path=str(p),sha256=sha(p)))
        except (OSError,json.JSONDecodeError) as error:notes.append(dict(path=str(p),reason=str(error)))
    active_ids={j['id'].split('_')[0] for j in snapshot['jobs']}
    candidates=[];held=[];inventory=[]
    for root in roots:
        size=allocated=count=0;selected=0
        category=root.name
        for path in list_cache(root):
            m=meta(path);count+=1;size+=m['size'];allocated+=m['allocated_bytes']
            why=None;rel=path.relative_to(root);parts=rel.parts
            if any(str(x) in active_ids for x in parts):why='active job identity in cache path'
            elif str(path) in protected:why='referenced by a frozen lock; preserve'
            elif m['kind']!='file' or m['nlink']!=1:why='not an unlinked regular single-link file'
            elif NOW-path.stat().st_mtime<86400:why='modified within 24h; preserve in this pass'
            elif category=='__pycache__' and path.suffix!='.pyc':why='unexpected Python cache payload'
            elif category in {'worker_cache'}:
                # Only regenerate accelerator kernel/tuning caches; no peram/basis arrays.
                if not any(x in {'quda','cupy','cupy_cache','quda_cache','__pycache__'} for x in parts):
                    why='worker payload is not proven kernel/tuning cache; preserve'
                elif path.suffix.lower() not in {'.tsv','.json','.cubin','.ptx','.cu','.cuh','.pyc','.pickle'}:
                    why='unknown worker-cache extension'
            elif category.startswith('scratch_'):
                why='old scratch may contain incomplete physics checkpoint; preserve pending targeted review'
            elif category in {'cupy_cache','quda_cache'} and path.suffix.lower() not in {'.tsv','.json','.cubin','.ptx','.cu','.cuh','.pickle'}:
                why='unknown accelerator-cache extension'
            elif category in {'matplotlib_cache','mplconfig'} and not path.name.startswith('fontlist-'):
                why='not a known Matplotlib font cache'
            if why:held.append(dict(**m,reason=why));continue
            m.update(category=category,reason='regenerable non-scientific cache',sha256=sha(path))
            candidates.append(m);selected+=1
        inventory.append(dict(path=str(root.relative_to(ROOT)),category=category,files=count,
                              size=size,allocated_bytes=allocated,delete_files=selected))
        print('CACHE',str(root.relative_to(ROOT)),count,size,'selected',selected,flush=True)
    # Latest analysis and postprocessor source locks must remain byte-identical.
    key=ROOT/'quark_B_three_ensemble_100_40_30_boot200_continuum_20261108'
    extra=[key/'source_lock.json',key/'complete.json',key/'config.json',
        key/'11_lambda_fourier_lightcone_20261108/source_lock.json',
        key/'11_lambda_fourier_lightcone_20261108/complete.json',
        key/'11_lambda_fourier_lightcone_20261108/00_audit/delivery_v1.json']
    anchors=[dict(path=str(p),sha256=sha(p)) for p in extra if p.is_file()]
    records=dict(generated=now(),root=str(ROOT),dirs_scanned=seen,cache_roots=inventory,
        delete_files=candidates,held_files=held,preserved_notes=notes,frozen_locks=lockstats,
        protected_products=anchors,protected_referenced_paths=len(protected),
        allocated_bytes=sum(m['allocated_bytes'] for m in candidates),
        logical_bytes=sum(m['size'] for m in candidates),
        scope='No raw arrays, source, results, basis, gauge, vertex, VVV, or active production selected')
    dump(HERE/'cleanup_plan.json',records)
    table(HERE/'cache_inventory.csv',inventory,['path','category','files','size','allocated_bytes','delete_files'])
    table(HERE/'deletions.csv',candidates,['path','category','reason','size','allocated_bytes','sha256'])
    table(HERE/'held_files.csv',held,['path','reason','size','allocated_bytes'])
    dump(HERE/'summary.json',dict(generated=now(),candidates=len(candidates),
        allocated_bytes=records['allocated_bytes'],logical_bytes=records['logical_bytes'],
        cache_roots=len(roots),held_files=len(held),frozen_locks=len(lockstats)))
    print('PLAN_SUMMARY',json.dumps(json.loads((HERE/'summary.json').read_text())),flush=True)

def apply():
    data=json.loads((HERE/'cleanup_plan.json').read_text())
    if data['root']!=str(ROOT):raise RuntimeError('Wrong workspace in plan')
    if (HERE/'cleanup_receipt.json').exists():raise RuntimeError('Do not overwrite a cleanup receipt')
    snapshot=queue_snapshot();dump(HERE/'active_jobs_predelete.json',snapshot)
    for record in data['frozen_locks']+data['protected_products']:
        assert sha(Path(record['path']))==record['sha256'],record['path']
    for record in data['delete_files']:
        conflict=active_conflict(record['path'],snapshot)
        if conflict:raise RuntimeError((record['path'],conflict))
        path=checked(record['path']);m=meta(path)
        for k in ('kind','inode','device','size','mtime_ns','ctime_ns','nlink'):assert m[k]==record[k],(path,k)
        assert sha(path)==record['sha256'],path
    for record in data.get('remove_empty_directories',[]):
        conflict=active_conflict(record['path'],snapshot)
        if conflict:raise RuntimeError((record['path'],conflict))
        m=meta(checked(record['path']))
        for k in ('kind','inode','device','mtime_ns'):assert m[k]==record[k],(record['path'],k)
    receipt=dict(started=now(),plan_sha256=sha(HERE/'cleanup_plan.json'),deleted=[],
        removed_empty_directories=[],failures=[])
    dump(HERE/'cleanup_receipt.json',receipt)
    for record in data['delete_files']:
        try:
            path=checked(record['path'])
            assert meta(path)['inode']==record['inode']
            path.unlink()
            receipt['deleted'].append(record)
        except Exception as error:
            receipt['failures'].append(dict(path=record['path'],error=repr(error)))
            dump(HERE/'cleanup_receipt.json',receipt)
            raise
        if len(receipt['deleted'])%100==0:dump(HERE/'cleanup_receipt.json',receipt)
    for record in data.get('remove_empty_directories',[]):
        try:
            path=checked(record['path']);m=meta(path)
            assert m['kind']=='directory' and m['inode']==record['inode'] and m['device']==record['device']
            # rmdir refuses nonempty directories; never recursively remove payloads.
            path.rmdir()
            receipt['removed_empty_directories'].append(record)
        except Exception as error:
            receipt['failures'].append(dict(path=record['path'],error=repr(error)))
            dump(HERE/'cleanup_receipt.json',receipt)
            raise
    receipt.update(finished=now(),deleted_count=len(receipt['deleted']),
        logical_bytes=sum(r['size'] for r in receipt['deleted']),
        allocated_bytes=sum(r['allocated_bytes'] for r in receipt['deleted']),
        recovery='No backup copy of caches or synthetic outputs; regenerate with preserved code/input. No scientific data removed.')
    dump(HERE/'cleanup_receipt.json',receipt)
    verify()

def verify():
    plan=json.loads((HERE/'cleanup_plan.json').read_text());receipt=json.loads((HERE/'cleanup_receipt.json').read_text())
    failures=[]
    for r in receipt['deleted']:
        if (ROOT/r['path']).exists():failures.append(dict(path=r['path'],error='cache recreated/still present'))
    for r in receipt.get('removed_empty_directories',[]):
        if (ROOT/r['path']).exists():failures.append(dict(path=r['path'],error='empty directory recreated/still present'))
    for r in plan['frozen_locks']+plan['protected_products']:
        if sha(Path(r['path']))!=r['sha256']:failures.append(dict(path=r['path'],error='protected hash changed'))
    dump(HERE/'verification.json',dict(timestamp=now(),status='passed' if not failures else 'failed',
        removed_files=len(receipt['deleted']),protected_locks=len(plan['frozen_locks']),
        protected_product_anchors=len(plan['protected_products']),failures=failures,
        removed_empty_directories=len(receipt.get('removed_empty_directories',[])),
        no_active_jobs_cancelled=True,no_physics_source_or_results_deleted=True))
    if failures:raise RuntimeError(failures)
    dump(HERE/'active_jobs_after.json',queue_snapshot())
    print('VERIFIED',receipt['deleted_count'],'files;',receipt['allocated_bytes'],'allocated bytes removed',flush=True)

def augment():
    """Add only individually reviewed synthetic outputs and old empty scratch."""
    if (HERE/'cleanup_receipt.json').exists():raise RuntimeError('Cleanup already started')
    data=json.loads((HERE/'cleanup_plan.json').read_text())
    if 'reviewed_additions' in data:raise RuntimeError('Plan already augmented')
    base=ROOT/'gluon_two_ensemble_workflow_20261102'
    evidence=base/'followup_tests_complete.json'
    report=json.loads(evidence.read_text())
    if report.get('synthetic_only_not_physics') is not True:
        raise RuntimeError('Missing synthetic-only evidence')
    if report.get('status')!='PASS_CONVERSION_LINEAR_PLOT_SMOKE':
        raise RuntimeError('Synthetic check did not pass')
    roots=[base/'gluon_plots_smoke_uiwsio6a',base/'gluon_plots_smoke_v3_e4lwbz6p']
    if report['scratch']!=str(roots[1]):raise RuntimeError('Unexpected test output path')
    complete=json.loads((base/'complete.json').read_text())
    if complete.get('status')!='PASS_WORKFLOW_DIAGNOSTICS_ONLY':
        raise RuntimeError('Subsequent production workflow has not completed')
    snapshot=queue_snapshot();dump(HERE/'active_jobs_review.json',snapshot)
    protected=set()
    def collect(obj):
        if isinstance(obj,dict):
            for k,v in obj.items():collect(k);collect(v)
        elif isinstance(obj,list):
            for v in obj:collect(v)
        elif isinstance(obj,str) and obj.startswith(str(ROOT)+'/'):protected.add(obj)
    for record in data['frozen_locks']:
        path=Path(record['path'])
        if sha(path)!=record['sha256']:raise RuntimeError('Frozen lock changed')
        collect(json.loads(path.read_text()))
    directories=[];added=[]
    for root in roots:
        checked(str(root.relative_to(ROOT)))
        for directory,dirs,files in os.walk(root,followlinks=False):
            path=Path(directory);dm=meta(path)
            if active_conflict(dm['path'],snapshot):raise RuntimeError('Active synthetic path')
            if any((path/name).is_symlink() for name in dirs+files):
                raise RuntimeError('Unexpected symlink in synthetic output')
            directories.append(dm)
            for name in files:
                file=path/name;m=meta(file)
                if (m['kind']!='file' or m['nlink']!=1 or NOW-file.stat().st_mtime<86400
                        or file.suffix.lower() not in {'.pdf','.npz','.png'}):
                    raise RuntimeError(('Unexpected synthetic payload',m['path']))
                if str(file) in protected:raise RuntimeError('Synthetic output is a locked input')
                m.update(category='synthetic_plot_output',
                    reason='Reviewed RNG-only isolated plot test; retained source and PASS/FAIL evidence',
                    sha256=sha(file))
                added.append(m)
    for root in data['cache_roots']:
        if root['category'].startswith('scratch_') and root['files']==0:
            path=checked(root['path'])
            if not list(path.iterdir()) and NOW-path.stat().st_mtime>=86400:
                if active_conflict(root['path'],snapshot):raise RuntimeError('Active scratch directory')
                directories.append(meta(path))
    for path in [evidence,base/'complete.json',base/'code/smoke_followup_cpu.py',
                 base/'code/smoke_followup_v3_cpu.py']:
        data['protected_products'].append(dict(path=str(path),sha256=sha(path)))
    data['delete_files'].extend(added)
    data['remove_empty_directories']=sorted(directories,key=lambda r:len(Path(r['path']).parts),reverse=True)
    data['reviewed_additions']=dict(timestamp=now(),synthetic_roots=[str(p.relative_to(ROOT)) for p in roots],
        synthetic_files=len(added),synthetic_evidence=str(evidence.relative_to(ROOT)),
        source_review='smoke_followup[_v3]_cpu.py uses seeded RNG; no real-data input stored in these roots',
        retained_real_data_smoke=['flow_zero_lambda_fourier_quasi_matching_20260927/smoke',
            'quark_l32x64_B_06_lambda_fourier_lamet_v2_20261025/smoke'])
    data['allocated_bytes']=sum(m['allocated_bytes'] for m in data['delete_files'])
    data['logical_bytes']=sum(m['size'] for m in data['delete_files'])
    data['scope']='Only regenerable non-science caches, reviewed RNG-only test outputs, and old empty scratch; no raw/science/source/basis/active production'
    dump(HERE/'cleanup_plan.json',data)
    table(HERE/'deletions.csv',data['delete_files'],['path','category','reason','size','allocated_bytes','sha256'])
    table(HERE/'empty_directories.csv',directories,['path','kind','inode','device','mtime_ns'])
    summary=dict(generated=now(),candidates=len(data['delete_files']),allocated_bytes=data['allocated_bytes'],
        logical_bytes=data['logical_bytes'],cache_roots=len(data['cache_roots']),held_files=len(data['held_files']),
        frozen_locks=len(data['frozen_locks']),synthetic_files=len(added),empty_directories=len(directories))
    dump(HERE/'summary.json',summary)
    print('REVIEWED_PLAN',json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['plan','augment','apply','verify']);a=p.parse_args()
    {'plan':plan,'augment':augment,'apply':apply,'verify':verify}[a.mode]()
