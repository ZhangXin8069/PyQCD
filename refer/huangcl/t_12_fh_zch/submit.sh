#!/bin/bash

# gpu partition: nv100-ins,nv100-sug,dgx2,na100-ins,na100-sug,gpu-debug,na100-40g,na800-sug,na800-pcie,h20-nettr
# cpu partition: cpu6248R,cpueicc,i72c512g,a192c1t,cpu-short
# if using gpu partition, you need to set how many gpu to be use in a task. Otherwise, commenting it.

#SBATCH --job-name=fit_FH
#SBATCH --partition=cpueicc,i72c512g,a192c1t,cpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# 限制单线程，匹配 --cpus-per-task=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ===== 用户配置：修改此处即可 =====
conf_short="L24x72"
P=6
# ==================================

# 从 conf_short 提取第一个数字作为 z 的数量: L24x72 → 24, L32x96 → 32
[[ "$conf_short" =~ [0-9]+ ]] && Nz="${BASH_REMATCH[0]}"

code="2_fit_FH"
log="_${code}.log"

. /public/home/huangcl/act_venv.sh

echo "job starts at $(date)" >"${log}"
echo "conf_short = ${conf_short}, P = ${P}" >>"${log}"
echo "Nz = ${Nz}" >>"${log}"

# for 循环顺序执行, 每个 z 独立启动 Python 进程
# 这样 lsqfit 每次重新初始化, 不会因为长时间运行而变慢
for (( z=0; z<Nz; z++ )); do
    echo "=== fitting z=${z} / $((Nz-1)) ===" >>"${log}"
    python -u "${code}.py" -c "${conf_short}" -p "${P}" -z "${z}" -u >>"${log}" 2>&1
    echo "=== z=${z} done ===" >>"${log}"
done

echo "job ends at $(date)" >>"${log}"
