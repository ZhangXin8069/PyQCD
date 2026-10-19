#!/bin/bash
source /public/home/sush/envs/C12P12

module purge
module load hpcx-ompi/2.18.1
module load cuda/12.6.1-none-none
# for i in {5000..7000..40}
# do
# python 0-+2.py measure \
#     --gauge /public/group/lqcd/configurations/CLOVER/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64_cfg_${i}.lime \
#     --cfg-id ${i} \
#     --ensemble E32P29 \
#     --out /public/home/sush/distillation/glueball/result/op_${i}.npz &
# done

python 0-+2.py analyze \
    --manifest /public/home/sush/distillation/glueball/result/analysis.txt \
    --t0 0 --td 1 \
    --states 2 \
    --at-fm 0.0897 \
    --out spectrum
