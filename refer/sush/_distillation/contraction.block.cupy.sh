#!/bin/bash

#SBATCH --job-name=pff
#SBATCH --output=test_creat_ENV.out
#SBATCH --array=0#-9#-19
##SBATCH --partition=dgx1,dgx2,nv100-ins,nv100-sug#,na100-40g,na800-pcie
##SBATCH --partition=na800-pcie

#SBATCH --partition=gpu-debug

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -n 1

#SBATCH --nodes=1

#SBATCH --error=test_pion_3pt_error
#SBATCH --output=test_pion_3pt_output

##SBATCH --begin=now+7hour
##SBATCH --time="3-00:00:00"
#SBATCH --time="30:00"

##SBATCH -x gpu001,gpu018,gpu025

source /public/home/sush/env.sh
source /public/home/sush/envs/sm80

chack_data=0 # ==1 then find the mission which is not been sbatched and to caculate it
use_array=1 # ==1 then use the array ==0 use the for ==1 then don't do anything
test_contract=1 # ==1 then sbatch test.py else ==0 then sbatch the complete .py
test_mpi=0

conf_stare=4000
gap=100
Ncnfg=1

corr_2pt=1
corr_3pt=1
part_of_3pt=0

X=32
T=96

Px=0
Py=0
Pz=0

load_type=np_load # np_load will use the np.load function like the file of share work and readin will use the readin function
contract_type=curr # curr will contract at the curr place, sink will contract at the sink place

conf_type="beta6.20_mu-0.2770_ms-0.2400_L24x72"
peram_dir=/public/group/lqcd/perambulators/${conf_type}/light

run_dir=/public/home/sush/distillation/pion_form_factor
output_path=${run_dir}/output/Px${Px}Py${Py}Pz${Pz}

if [ -e ${output_path} ];then
    echo input output path exist
else
    echo mkdir
    mkdir -p ${run_dir}/output/Px${Px}Py${Py}Pz${Pz} 
fi

conf=$[${conf_stare}+${gap}*${SLURM_ARRAY_TASK_ID}]

output_file=${output_path}/output_${conf}_${X}x${T}_mpi.contraction.block.cupy.log
echo "$conf job starts at" `date` > ${output_file}

exe=${run_dir}/contraction.block.cupy.py

dir_file=${run_dir}/file/pion_contraction_block_cupy_${conf}
mkdir ${dir_file}

if [ -d ${dir_file} ]; then
    mpirun -n 1 python -u $exe ${conf}  >> ${output_file} 2>&1 

    if [ $? -eq 0 ]; then
        echo "Python 程序执行成功" >> ${output_path} 2>&1
        rm -r ${dir_file}
    else
        echo "Python 程序执行失败" >> ${output_path} 2>&1
    fi

else
    echo "calculate complete" >> ${output_path} 2>&1
fi

echo "$conf job ends at" `date` >> ${output_file}
