"""Publish completed ensemble outputs after fixing duplicate-key summary assembly."""
import os,json,hashlib
from pathlib import Path
from datetime import datetime
import analyze as a
ROOT=Path(__file__).resolve().parent

def main():
 assert os.environ.get('SLURM_JOB_ID')
 for n,h in a.read(ROOT/'source_lock.json')['sha256'].items():assert a.sha(ROOT/n)==h,n
 cfg=a.read(ROOT/'config.json');summaries=[a.read(ROOT/e['name']/'report.json') for e in cfg['ensembles']]
 for e,s in zip(cfg['ensembles'],summaries):assert s['ensemble']==e['name'] and s['Ncfg']==e['Ncfg']
 examples=[dict(r) for s in summaries for r in s['examples']]
 a.table(ROOT/'comparison_examples.csv',examples)
 a.write(ROOT/'report.json',dict(status='PASS_NUMERICAL_ANALYSIS',created=datetime.now().astimezone().isoformat(),job=os.environ['SLURM_JOB_ID'],ensembles=summaries,
  physical_result_status='finite_flow_diagnostic_only; no final universal d',uncertainty='paired iid configuration jackknife; autocorrelation and continuum/flow extrapolation not controlled'))
 figs=list(ROOT.glob('*/figures/*.pdf'));assert len(figs)==12
 assert all(p.stat().st_size>5000 for p in figs)
 a.write(ROOT/'complete.json',dict(status='PASS_NUMERICAL_ANALYSIS',job=os.environ['SLURM_JOB_ID'],report_sha256=a.sha(ROOT/'report.json'),source_lock_sha256=a.sha(ROOT/'source_lock.json'),
  artifacts={str(p.relative_to(ROOT)):a.sha(p) for p in [ROOT/'comparison_examples.csv',ROOT/'report.json']+figs},visual_review='pending',
  publication_fix='duplicate ensemble key in final summary only; original completed fits and samples preserved'))
 # Keep the original failed job record as history rather than an active failure marker.
 if (ROOT/'failed.json').exists():(ROOT/'failed.json').rename(ROOT/'failed_3991934_summary_assembly.json')
 print('PUBLISHED_COMPLETE',flush=True)
if __name__=='__main__':main()
