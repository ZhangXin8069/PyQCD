"""Verify the 20-conf successor has unchanged scientific kernels."""
import ast
import csv
import json
from common import HERE,CONFS,rows,sha256
PREDECESSOR=HERE.parent/'quark_l24x72_B_conf51to100_matched_radius_20261105'
UNCHANGED=('audit_cpu.py','backend_sample.py','dispatch_tasks.py','epsilon_gate.py',
    'flow_safety.py','generate_basis.py','gpu_validation.py','gpu_worker.py','launch.py',
    'test_cpu.py','test_retry_cpu.py')

def normalized_common(path):
    tree=ast.parse(path.read_text())
    tree.body=[node for node in tree.body if not (
        isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str))
        and not (isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CONFS' for t in node.targets))]
    return ast.dump(tree,include_attributes=False)

def verify_expansion():
    previous=json.loads((PREDECESSOR/'request.json').read_text())
    for name,digest in previous['files_sha256'].items():
        assert sha256(PREDECESSOR/name)==digest,('changed predecessor',name)
    for name,digest in previous['upstream_sha256'].items():
        assert sha256(name)==digest,('changed upstream',name)
    prior=previous['all_confs']
    assert prior==list(map(str,range(10000,30000,200)))
    assert CONFS==tuple(map(str,range(30000,34000,200))) and not set(CONFS)&set(prior)
    paths=list(UNCHANGED)+[str(p.relative_to(HERE)) for p in sorted((HERE/'reference').glob('*.py'))]
    identical={}
    for name in paths:
        digest=sha256(HERE/name)
        assert digest==sha256(PREDECESSOR/name),('changed numerical kernel',name)
        identical[name]=digest
    assert normalized_common(HERE/'common.py')==normalized_common(PREDECESSOR/'common.py')
    expected=(PREDECESSOR/'prepare_cpu.py').read_text().replace('350','140').replace(
        'combined_configuration_target=100','combined_configuration_target=120').replace(
        'conf_count=50','conf_count=20').replace("'CPU_GATE_PASS',50,140","'CPU_GATE_PASS',20,140")
    assert (HERE/'prepare_cpu.py').read_text()==expected
    with (PREDECESSOR/'flow_schedule.csv').open(newline='') as stream:
        old_flows=list(csv.DictReader(stream))
    assert len(old_flows)==7
    for new,old in zip(rows()[:7],old_flows):
        for key in ('tau64','flow_time_a2','tf_fm2','radius_fm','epsilon','nstep','window64','window'):
            assert float(new[key])==float(old[key]),key
    return dict(status='PASS_SOURCE_AND_CAMPAIGN_IDENTITY',predecessor_root=str(PREDECESSOR),
        predecessor_request_sha256=sha256(PREDECESSOR/'request.json'),
        unchanged_numerical_files_sha256=identical,previous_confs=prior,new_confs=list(CONFS),
        combined_confs=prior+list(CONFS),combined_configuration_target=120,numerical_kernel_changed=False,
        common_change='CONFS=30000..33800 step200 and docstring only',
        prepare_change='task/conf counts and combined target only',
        prior_completion='GPU workers exited 0; final audit failed due missing perambulator21600; not asserted complete')

if __name__=='__main__':
    print(json.dumps(verify_expansion(),indent=2))
