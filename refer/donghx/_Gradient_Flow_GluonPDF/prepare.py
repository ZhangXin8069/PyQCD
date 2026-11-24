"""Freeze the original main-ensemble scopes and isolated five-flow GPU production."""
from pathlib import Path
from datetime import datetime
import ast,csv,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def identity(p):
 p=Path(p).resolve();s=p.stat();return dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,inode=s.st_ino,device=s.st_dev)
req=dict(created_at=datetime.now().astimezone().isoformat(),taus=[.7,.8,.9,1.1,1.2],epsilon=.01,fine_epsilon=.005,regression_taus=[.6,1.,1.4],
 scope='All original main gluon-OPE configurations for L32x64 and L32x96; no L24x72 gluon main campaign inferred.',
 resources=dict(partition='na100-ins',node='gpu025',reservation='donghx_250',gpus_per_task=1,cpus_per_task=4,memory='48G',
      time='1-00:00:00',array_concurrency_per_ensemble=4,configurations_per_array_task=8,gpu_memory_preflight_bytes=16*1024**3),
 gates=dict(cpu_gpu_relative_l2=1e-4,cpu_gpu_plaquette_abs=1e-5,epsilon_relative_l2=.005,epsilon_plaquette_abs=5e-5),
 physics=dict(flow='4D Wilson QUDA RK3, raw thin links, no presmear, independent raw copy at each tau',
      flow_precision='complex128',local_algebra='complex64',spatial_accumulation='complex128',output='complex128',
      clover='F=-i(Q-Qdagger)/8; both legacy_untraced and traceless',zdir=2,zcount=25,orientations=['+z','-z','even_sum','odd_difference'],
      spatial_normalization='full spatial sum per timeslice; no extra volume/Nc/g/a factors',
      raw_combined_slots=dict(unpolarized='TU-SU',helicity='TH+SH'),note_helicity='(TH-SH)_odd saved separately',
      phase_or_matching_applied=False,gauge_fixing_applied=False),ensembles={})
for label,manifest,nconf,nt,a,outroot in [('L32x64','manifest_L32x64.tsv',1700,64,.0897,'output_L32x64_v1'),('L32x96','manifest.tsv',887,96,.0775,'output_v5')]:
 with (BASE/manifest).open() as f:rows=list(csv.DictReader(f,delimiter='\t'))
 assert len(rows)==nconf and len({r['conf_id'] for r in rows})==nconf and rows[0]['conf_id']=='1000'
 items=[]
 for r in rows:
  stat=identity(r['gauge_path']);assert stat['bytes']>=int(r['lime_payload_offset'])+int(r['lime_payload_bytes'])
  assert int(r['lime_payload_bytes'])==32**3*nt*4*9*16
  items.append(dict(conf_id=r['conf_id'],gauge_path=r['gauge_path'],identity=stat,lime_payload_offset=int(r['lime_payload_offset']),lime_payload_bytes=int(r['lime_payload_bytes'])))
 refs={}
 for tau in req['regression_taus']:
  tag=('tau%.3f'%tau).replace('.','p');base=BASE/outroot/'conf1000'/tag/('conf1000_'+tag+'_eps0.010_zdir2')
  refs[str(tau)]=dict(npy=str(base)+'.npy',npy_sha256=sha(str(base)+'.npy'),json=str(base)+'.json',json_sha256=sha(str(base)+'.json'))
 batches=[list(range(i,min(i+8,nconf))) for i in range(1,nconf,8)]
 assert [j for b in batches for j in b]==list(range(1,nconf))
 req['ensembles'][label]=dict(name=label,lattice_xyzt=[32,32,32,nt],a_fm=a,original_manifest=str(BASE/manifest),
      original_manifest_sha256=sha(BASE/manifest),references=refs,items=items,batches=batches,
      pilot_conf_ids=['1000'],production_configuration_count=nconf-1)
write(ROOT/'request.json',req)
# Scripts inherit the previously validated A100 environment and isolated caches.
env='''set -eo pipefail
source /public/home/donghx/load_a100_PyQuad.sh
set -u
export LC_ALL=C LANG=C PYTHONDONTWRITEBYTECODE=1 MPI4PY_RC_THREAD_LEVEL=serialized
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_opal_warn_on_missing_libcuda=0 UCX_TLS=sm,self UCX_MEMTYPE_CACHE=n
cd ROOT_PATH
export QUDA_RESOURCE_PATH="$PWD/cache/job_${SLURM_JOB_ID}/quda"
export CUPY_CACHE_DIR="$PWD/cache/job_${SLURM_JOB_ID}/cupy"
export QUDA_PROFILE_OUTPUT_BASE="$QUDA_RESOURCE_PATH/profile"
mkdir -p "$QUDA_RESOURCE_PATH" "$CUPY_CACHE_DIR"
'''.replace('ROOT_PATH',str(ROOT))
for label,ens in req['ensembles'].items():
 for stage in ['pilot','production']:
  head='''#!/usr/bin/env bash
#SBATCH --partition=na100-ins
#SBATCH --nodelist=gpu025
#SBATCH --reservation=donghx_250
#SBATCH --job-name=JOB_NAME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=ROOT_PATH/logs/JOB_TAG_%A_%a.out
#SBATCH --error=ROOT_PATH/logs/JOB_TAG_%A_%a.err
'''.replace('JOB_NAME','gfG_'+label+'_'+stage).replace('JOB_TAG',label+'_'+stage).replace('ROOT_PATH',str(ROOT))
  if stage=='production':head+='#SBATCH --array=0-%d%%4\n'%(len(ens['batches'])-1)
  body=env+'exec python -B -u '+('code/worker.py --ensemble '+label+' --task 0 --pilot\n' if stage=='pilot' else 'code/batch.py --ensemble '+label+' --batch "${SLURM_ARRAY_TASK_ID:?array required}"\n')
  (ROOT/(label+'_'+stage+'_gpu.slurm')).write_text(head+body)
 cpu='''#!/usr/bin/env bash
#SBATCH --partition=math,cpu6248R,cpueicc,a192c1t,i72c512g
#SBATCH --job-name=JOB_NAME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=ROOT_PATH/logs/JOB_NAME_%j.out
#SBATCH --error=ROOT_PATH/logs/JOB_NAME_%j.err
set -euo pipefail
export LC_ALL=C LANG=C PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd ROOT_PATH
exec /public/home/donghx/AnaConda/bin/python3 -B -u code/pilot_gate.py --ensemble ENSEMBLE
'''.replace('JOB_NAME','gfG_'+label+'_gate').replace('ROOT_PATH',str(ROOT)).replace('ENSEMBLE',label)
 (ROOT/(label+'_gate_cpu.slurm')).write_text(cpu)
final=cpu[:cpu.index('exec /public/home')]+ 'exec /public/home/donghx/AnaConda/bin/python3 -B -u code/final_audit.py\n'
final=final.replace('gfG_L32x96_gate','gfG_dense_final').replace('--time=00:30:00','--time=02:00:00')
(ROOT/'final_audit_cpu.slurm').write_text(final)
write(ROOT/'coverage_preflight.json',dict(status='PASS_STATIC_SCOPE_AND_IDENTITIES',created_at=datetime.now().astimezone().isoformat(),
      configurations=2587,flow_points=12935,taus=req['taus'],existing_main_tau_grid=[.1,.2,.3,.4,.5,.6,1.,1.4,1.8,2.2,2.6,3.,3.4,3.8],
      no_existing_main_tau_overlap=True,pilot_counts={k:1 for k in req['ensembles']},
      production_counts={k:v['production_configuration_count'] for k,v in req['ensembles'].items()},
      batches={k:len(v['batches']) for k,v in req['ensembles'].items()},
      old_small_campaign_tau08_not_reused='20-conf special campaign does not establish full-statistics production identity'))
for p in ROOT.rglob('*.py'):ast.parse(p.read_text())
for p in ROOT.glob('*.slurm'):subprocess.run(['bash','-n',str(p)],check=True)
files=[ROOT/'request.json',ROOT/'coverage_preflight.json',*sorted((ROOT/'code').glob('*.py')),*sorted((ROOT/'code').glob('*.cu')),*sorted(ROOT.glob('*_gpu.slurm')),*sorted(ROOT.glob('*_cpu.slurm'))]
files=[p for p in files if p.name!='diagnostic_gpu.slurm']
write(ROOT/'source_lock.json',dict(sha256={str(p.relative_to(ROOT)):sha(p) for p in files}))
print('PREPARED',[(k,len(v['items']),len(v['batches'])) for k,v in req['ensembles'].items()], '12935 new flow products',flush=True)
