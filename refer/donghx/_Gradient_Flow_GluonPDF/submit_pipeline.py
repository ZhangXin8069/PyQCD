"""Submit an immutable successor after the already running full input probe."""
import argparse
import ast
from datetime import datetime
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--probe-job',required=True)
    args=parser.parse_args()
    assert args.probe_job.isdecimal()
    target=HERE/'submission.json'
    assert not target.exists() and not (HERE/'submission_attempt.json').exists(), 'Do not duplicate a partial or existing submission'
    for path in HERE.rglob('*.py'):
        ast.parse(path.read_text(),filename=str(path))
    for path in HERE.glob('*.slurm'):
        subprocess.run(['bash','-n',str(path)],check=True)
    receipt=dict(created=datetime.now().astimezone().isoformat(),status='SUBMITTING_WITH_REQUIRED_RUNTIME_GATES',
        input_probe_job=args.probe_job,conf_count=80,configuration_indices=[121,200],
        conf_first='34000',conf_last='49800',configuration_spacing=200,conf_flow_tasks=560,
        expected_case_count=141120,combined_configuration_target=200,
        predecessor_root=str(HERE.parent/'quark_l24x72_B_conf101to120_matched_radius_20261114'),
        resource=dict(partition='na100-ins',reservation='donghx_250',exclude=['gpu025','gpu028'],
            eligible_nodes=['gpu026','gpu027','gpu029'],gpus_per_production_job=4,ntasks=4,
            cpus_per_task=2,mem='240G',time='3-00:00:00',array='0-23',workers=24,
            packing='one conf-flow per allocation; requeue same array element on success only'),jobs={})
    def save():
        temp=target.with_suffix('.tmp')
        temp.write_text(json.dumps(receipt,indent=2)+'\n')
        temp.replace(target)
    (HERE/'submission_attempt.json').write_text(json.dumps(dict(started=receipt['created'],probe_job=args.probe_job),indent=2)+'\n')
    save()
    def submit(label,script,dependency,extra=(),arguments=()):
        command=['sbatch','--parsable','--dependency=afterok:'+dependency,'--kill-on-invalid-dep=yes',*extra,str(HERE/script),*arguments]
        result=subprocess.run(command,cwd=HERE,text=True,capture_output=True)
        if result.returncode:
            receipt['status']='PARTIAL_SUBMISSION_DO_NOT_RESUBMIT_BLINDLY'
            receipt['failure']=dict(label=label,command=command,stderr=result.stderr,stdout=result.stdout)
            save()
            raise RuntimeError(receipt['failure'])
        job=result.stdout.strip().split(';')[0]
        assert job.isdecimal(),(result.stdout,result.stderr)
        receipt['jobs'][label]=dict(job_id=job,command=command,dependency=dependency,stderr=result.stderr)
        save()
        print(label,job,flush=True)
        return job
    freeze=submit('freeze_verified_source_and_campaign','freeze_cpu.slurm',args.probe_job)
    gate=submit('cpu_inputs_and_algebra','gate_cpu.slurm',freeze)
    epsilon=submit('gpu_zero_free_epsilon','epsilon_na100_ins.slurm',gate)
    smoke=submit('first_last_flow_real_C3_C2_gate','run_na100_ins.slurm',epsilon,extra=('--job-name=q24_B200_smoke',),arguments=('smoke',))
    production=submit('conf121to200_7flow_production','run_na100_ins.slurm',smoke,extra=('--array=0-23',))
    submit('full_cpu_audit','audit_cpu.slurm',production)
    receipt['status']='SUBMITTED_INPUT_PROBE_AND_RUNTIME_VALIDATION_PENDING'
    receipt['updated']=datetime.now().astimezone().isoformat()
    save()

if __name__=='__main__':
    main()
