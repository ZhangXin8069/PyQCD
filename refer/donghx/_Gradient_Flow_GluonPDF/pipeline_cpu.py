"""One CPU array element handles one flow's ratios and nine channel/flavor fits."""
import argparse
import subprocess
import sys
from common import HERE,FLOWS,allocation,manifest,atomic_json,sha256


def main():
    allocation();manifest()
    parser=argparse.ArgumentParser();parser.add_argument('--flow-index',type=int,required=True)
    args=parser.parse_args()
    if not 0<=args.flow_index<len(FLOWS):
        raise ValueError(args.flow_index)
    subprocess.run([sys.executable,'-B','-u',str(HERE/'ratios_cpu.py'),'--flow-index',str(args.flow_index)],check=True)
    for task in range(args.flow_index*9,(args.flow_index+1)*9):
        subprocess.run([sys.executable,'-B','-u',str(HERE/'fit_cpu.py'),'--task',str(task)],check=True)
    atomic_json(HERE/'status'/f'flow_pipeline{args.flow_index}.json',dict(status='PASS',
        flow_index=args.flow_index,tau=FLOWS[args.flow_index][0],fit_tasks=9,
        input_manifest_sha256=sha256(HERE/'input_manifest.json')))
    print('L96_B_FLOW_RATIO_AND_NINE_FITS_PASS',args.flow_index,flush=True)


if __name__=='__main__':
    main()
