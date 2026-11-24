from pathlib import Path
import json,os,socket
root=Path(__file__).resolve().parent
req=json.loads((root/'request.json').read_text());out={}
for label,ens in req['ensembles'].items():
 item=ens['items'][0];p=Path(item['gauge_path']).resolve();s=p.stat()
 actual=dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,inode=s.st_ino,device=s.st_dev)
 out[label]=dict(expected=item['identity'],actual=actual,differences={k:dict(expected=item['identity'][k],actual=actual[k]) for k in actual if actual[k]!=item['identity'][k]})
(root/'validation'/('source_identity_'+os.environ['SLURM_JOB_ID']+'.json')).write_text(json.dumps(dict(host=socket.gethostname(),ensembles=out),indent=2)+'\n')
print(json.dumps(out,indent=2))
