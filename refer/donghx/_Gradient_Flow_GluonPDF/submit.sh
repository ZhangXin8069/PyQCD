#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "$0")"
exec 9>submit.lock
flock -n 9
test ! -e submission.tsv
test -s source_lock.json
printf 'ensemble\tphase\tjob_id\tarray_or_dependency\n' > submission.tsv
for entry in 'L24x72 15' 'L32x64 26' 'L32x96 13'; do
    read -r ensemble last <<< "$entry"
    pilot=$(sbatch --parsable --job-name="gfrxyz_${ensemble}_pilot" --mem=96G --time=01:00:00 gpu.slurm pilot "$ensemble")
    pilot=${pilot%%;*}
    [[ "$pilot" =~ ^[0-9]+$ ]]
    printf '%s\tpilot\t%s\t-\n' "$ensemble" "$pilot" | tee -a submission.tsv
    production=$(sbatch --parsable --job-name="gfrxyz_${ensemble}_prod" --array="0-${last}%8" \
        --dependency="afterok:$pilot" --kill-on-invalid-dep=yes gpu.slurm production "$ensemble")
    production=${production%%;*}
    [[ "$production" =~ ^[0-9]+$ ]]
    printf '%s\tproduction\t%s\t0-%s%%8 afterok:%s\n' "$ensemble" "$production" "$last" "$pilot" | tee -a submission.tsv
    audit=$(sbatch --parsable --job-name="gfrxyz_${ensemble}_audit" --dependency="afterany:$production" audit.slurm "$ensemble")
    audit=${audit%%;*}
    [[ "$audit" =~ ^[0-9]+$ ]]
    printf '%s\taudit\t%s\tafterany:%s\n' "$ensemble" "$audit" "$production" | tee -a submission.tsv
done
