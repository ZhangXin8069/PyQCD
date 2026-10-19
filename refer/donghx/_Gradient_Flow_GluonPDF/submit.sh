#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
N=$(awk 'NF && $1 !~ /^#/' "$HERE/missing_manifest.txt"|wc -l)
((N>0))||{ echo "No missing tasks";exit 0; }
if awk '$3+0==1.0{bad=1}END{exit !bad}' "$HERE/missing_manifest.txt";then echo "Refusing tf=1.0" >&2;exit 9;fi
mkdir -p "$HERE/logs"
JOBID=$(sbatch --parsable --array=0-$((N-1)) "$HERE/run_missing_no_tf1.slurm")
printf '%s\n' "$JOBID">"$HERE/jobid.txt"
printf 'Submitted %s tasks as job %s\n' "$N" "$JOBID"

