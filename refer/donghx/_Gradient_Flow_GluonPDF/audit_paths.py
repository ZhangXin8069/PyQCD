"""Record contraction planner estimates; these are not measured GPU timings."""
import json
from pathlib import Path
import sys
from opt_einsum import contract_path
sys.path.insert(0,str(Path(__file__).resolve().parent/'reference'))
from flowed_perambulator_factorization import RIGHT_FACTOR_SPECS
from spin2_contractions import SPECS

def main():
    ns,nc,nt,nmom=200,400,9,3
    base=dict(p=(4,4,ns,ns),b=(nt,4,4,nc,ns),g=(4,4),q=(4,4,4),
              vs=(nmom,ns,ns,ns),vr=(nmom,ns,ns,ns))
    reduced=dict(base,pl=(2,4,ns,ns),pr=(4,2,ns,ns),pb=(2,2,ns,ns),
                 br=(nt,4,2,nc,ns),q=(4,2,2))
    results=[]
    for old,new in zip(RIGHT_FACTOR_SPECS,SPECS):
        row={'name':old.name}
        for label,eq,keys,shapes in [('reference',old.equation,old.operands,base),
                                   ('spin2',new[3],new[4],reduced)]:
            path,info=contract_path(eq,*(shapes[k] for k in keys),shapes=True,optimize='optimal')
            row[label]=dict(equation=eq,optimizer_FLOPs=float(info.opt_cost),
                largest_intermediate_GiB=float(info.largest_intermediate)*16/1024**3,path=path)
        results.append(row)
    out=Path(__file__).resolve().parent/'planner_estimates.json'
    out.write_text(json.dumps(dict(disclaimer='Planner estimates only; no bandwidth, layout-copy or execution-time claim',
                                   records=results),indent=2)+'\n')
    for r in results:print(r['name'],'FLOP ratio',r['reference']['optimizer_FLOPs']/r['spin2']['optimizer_FLOPs'],
                          'GiB',r['reference']['largest_intermediate_GiB'],r['spin2']['largest_intermediate_GiB'])

if __name__=='__main__':main()
