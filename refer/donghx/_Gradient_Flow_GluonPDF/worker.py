"""One original HYP gauge start, no additional smearing/flow, one allocated GPU."""
import argparse, json, os, socket, time, traceback
from pathlib import Path
import core as c
import segments as s
ROOT=Path(__file__).resolve().parent

def check_sources():
 for name,digest in json.loads((ROOT/'source_lock.json').read_text())['sha256'].items():
  assert c.sha(ROOT/name)==digest, ('source changed',name)

def check_input(item):
 assert c.identity(item['gauge'])==item['identity'], ('input changed',item['gauge'])
 for path,digest in [('header','header_sha256'),('checksum_xml','checksum_sha256')]:
  assert c.sha(item[path])==item[digest]

def run(task,out,started):
 import numpy as np
 import cupy as cp
 import pyquda
 from pyquda.field import LatticeInfo,LatticeGauge
 from pyquda.enum_quda import QudaPrecision,QudaReconstructType
 from pyquda_io.mpi_utils import readMPIFile
 from pyquda_io.io_utils import checksumSciDAC
 from pyquda_utils.io import readILDGBinGauge
 check_sources()
 req=json.loads((ROOT/'request.json').read_text()); item=req['items'][task]; ens=req['ensembles'][item['ensemble']]
 assert req['selection']==dict(starts=[0],name='original_HYP',random_starts=False,best_of=False,gradient_flow=False)
 assert socket.gethostname().split('.')[0] in req['resources']['allowed_nodes']
 assert os.environ['SLURM_JOB_PARTITION']=='na100-ins'
 assert os.environ.get('SLURM_JOB_RESERVATION')=='donghx_250'
 check_input(item)
 c.np,c.cp=np,cp; c.LatticeInfo,c.LatticeGauge=LatticeInfo,LatticeGauge
 c.QudaPrecision,c.QudaReconstructType=QudaPrecision,QudaReconstructType
 c.CONTRACT=dict(req['contract'],line_nmax=ens['line_nmax'])
 assert not cp.cuda.runtime.is_hip and cp.cuda.runtime.getDeviceCount()==1
 assert pyquda.__version__=='0.10.54'
 pyquda.init(grid_size=[1,1,1,1],backend='cupy'); assert pyquda.getMPISize()==1
 props=cp.cuda.runtime.getDeviceProperties(0); assert 'A100' in str(props['name'])
 c.write_json(out/'environment.json',dict(job=os.environ['SLURM_JOB_ID'],array_job=os.environ.get('SLURM_ARRAY_JOB_ID'),host=socket.gethostname(),partition=os.environ['SLURM_JOB_PARTITION'],reservation=os.environ.get('SLURM_JOB_RESERVATION'),pyquda=pyquda.__version__,cupy=cp.__version__,gpu_name=str(props['name']),task=task))
 if item['selection_index']==0:
  c.write_json(out/'tiny_contract.json',c.tiny_contract())
 cp.get_default_memory_pool().free_all_blocks()
 dims=ens['lattice_xyzt']; x,y,z,t=dims
 # Same layout as installed pyquda_io.ildg.readGauge, directly on unpacked record.
 binary=readMPIFile(item['gauge'],'>c16',0,(t,z,y,x,4,3,3),(3,2,1,0))
 actual=tuple(int(v) for v in checksumSciDAC(dims,binary.reshape(t*z*y*x,36)))
 expected=tuple(int(item['checksum'][k],16) for k in ['suma','sumb'])
 c.write_json(out/'input_checksum.json',dict(expected=list(expected),actual=list(actual),passed=actual==expected))
 assert actual==expected, ('SciDAC checksum mismatch',actual,expected)
 lex=binary.transpose(4,0,1,2,3,5,6).astype('<c16'); del binary
 li=LatticeInfo(dims); raw=c.prepare(LatticeGauge(li,li.evenodd(lex,True))); del lex
 if task in req['pilot_tasks']:
  independent=c.prepare(readILDGBinGauge(item['gauge'],'>c16',dims))
  assert np.array_equal(raw.data,independent.data)
  c.write_json(out/'reader_crosscheck.json',dict(status='PASS',comparison='bitwise equal to installed readILDGBinGauge >c16',reunitarization=False))
  del independent
 raw_hash=s.data_hash(raw)
 initial,loops,_=c.measure(raw,out,'hyp_unfixed')
 gauge=c.prepare(raw.copy()); assert s.data_hash(gauge)==raw_hash
 c.write_json(out/'hyp_provenance.json',dict(ensemble=item['ensemble'],hyp_root=ens['hyp_root'],hyp_label=ens['hyp_label'],hyp_alpha_parameters=ens['hyp_alpha_parameters'],lattice_xyzt=dims,a_fm=ens['a_fm'],input_checksum_verified=True))
 c.write_json(out/'start_verified.json',dict(start_index=0,name='original_HYP',input=item,raw_data_sha256=raw_hash,start_data_sha256=s.data_hash(gauge),random_transform_applied=False,gradient_flow=False,extra_smearing=False,reunitarization=False,a_fm=ens['a_fm'],source_lock_sha256=c.sha(ROOT/'source_lock.json')))
 deadline=started+c.CONTRACT['soft_deadline_seconds']
 primary=s.phase(gauge,'FFT',c.CONTRACT['theta_primary'],c.CONTRACT['primary_max_steps'],out,'primary',deadline)
 tight=s.phase(gauge,'FFT',c.CONTRACT['theta_tight'],c.CONTRACT['tight_max_steps'],out,'tight',deadline) if primary['converged'] else None
 passed=primary['converged'] and tight is not None and tight['converged']
 label='final_converged' if passed else 'final_unconverged'
 final,_,_=c.measure(gauge,out,label,loops)
 assert final['gauge_functional']>=initial['gauge_functional']-1e-12
 assert s.data_hash(raw)==raw_hash, 'raw input copy mutated'
 check_input(item); check_sources()
 result=dict(status='PASS_HYP_LANDAU_START0' if passed else 'INCOMPLETE',task=task,ensemble=item['ensemble'],conf_id=item['conf_id'],start_index=0,primary=primary,tight=tight,measurement=final,source_lock_sha256=c.sha(ROOT/'source_lock.json'),elapsed_seconds=time.monotonic()-started,representations=req['representations'],ensemble_linear_divergence_coefficient=None)
 c.write_json(out/'result.json',result)
 if not passed: raise RuntimeError('Strict Landau residual not achieved; not a completed measurement')
 assert final['theta']<=c.CONTRACT['theta_tight']
 c.write_json(out/'complete.json',dict(status=result['status'],task=task,result_sha256=c.sha(out/'result.json'),source_lock_sha256=c.sha(ROOT/'source_lock.json')))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task',type=int,required=True); args=p.parse_args()
 assert os.environ.get('SLURM_JOB_ID')
 req=json.loads((ROOT/'request.json').read_text()); assert 0<=args.task<len(req['items'])
 out=ROOT/'runs'/('task%03d'%args.task)/('job'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True,exist_ok=False)
 try:run(args.task,out,time.monotonic())
 except BaseException as e:
  c.write_json(out/'failed.json',dict(status='FAILED',error=repr(e),traceback=traceback.format_exc()));raise
