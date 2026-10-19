#!/bin/bash

#SBATCH --job-name=light
#SBATCH --output=test_creat_ENV.out
#SBATCH --array=0
##SBATCH --partition=dgx2,nv100-ins,nv100-sug,dgx1
##SBATCH --partition=na800-pcie

#SBATCH --partition=gpu-debug
##SBATCH --partition=h20-nettr
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -n 1

#SBATCH --nodes=1

#SBATCH --error=test_pion_3pt_error
#SBATCH --output=test_pion_3pt_output

##SBATCH --begin=now+7hour
##SBATCH --time="2-00:00:00"
#SBATCH --time="30:00"
##SBATCH -w gpu047
##SBATCH -x gpu001,gpu018,gpu025
#SBATCH -x gpu054

Lx=24
Ly=24 
Lz=24
Lt=72
Ne=650

module purge


source /public/home/sush/envs/sm80

for i in {0..0..1}
do

conf_gap_2=200

source_shape=100

conf_stare=10000
conf_gap=200

conf_id=$[${conf_stare}+${conf_gap}*(${SLURM_ARRAY_TASK_ID})+${i}*${conf_gap_2}]

conf_path=/public/group/lqcd/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_

eigen_source=/nexdata/project/lqcd/sush/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72/${conf_id}/
output_peram_path=/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/

exe_peram=/public/home/sush/distillation/pion_form_factor/contraction.block.cupy.mpi.gen.C24P29.py
output_path=/public/home/sush/distillation/pion_form_factor/output/contraction.block.cupy.mpi.gen.C24P29.${conf_id}.log

dir_file=/public/home/sush/distillation/pion_form_factor/file/contraction.block.cupy.mpi.gen.C24P29.${conf_id}
mkdir ${dir_file}

if [ -d ${dir_file} ]; then
    mpirun -n 1 python -u ${exe_peram} ${Lx} ${Ly} ${Lz} ${Lt} ${Ne} ${conf_id} ${conf_path} ${eigen_source} ${output_peram_path} ${source_shape} ${i} > ${output_path} 2>&1

    if [ $? -eq 0 ]; then
        echo "Python 程序执行成功" >> ${output_path} 2>&1
        rm -r ${dir_file}
    else
        echo "Python 程序执行失败" >> ${output_path} 2>&1
    fi

else
    echo "calculate complete" >> ${output_path} 2>&1
fi

# mpirun -n 2 python -u ${exe_peram}  ${conf_id} > ./output/output_${conf_id}_peram.2.log 2>&1

done