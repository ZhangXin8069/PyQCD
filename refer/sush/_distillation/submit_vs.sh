#!/bin/bash

# 提交一个测试方案。用法：bash submit_vs.sh N100_all 24
set -u

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "用法：bash submit_vs.sh N100_all [提交数量]"
    exit 1
fi

vs_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
scheme="$1"
request_number="${2:-24}"
run_dir="${vs_dir}/${scheme}"
safe_limit=195

if [ ! -d "${run_dir}" ] || [ ! -f "${run_dir}/settings.yaml" ] || [ ! -f "${run_dir}/todo.txt" ]; then
    echo "找不到 ${scheme} 的 settings.yaml 或 todo.txt。"
    echo "请先运行：bash ${vs_dir}/prepare_vs.sh"
    exit 1
fi
if ! [[ "${request_number}" =~ ^[0-9]+$ ]] || [ "${request_number}" -eq 0 ]; then
    echo "提交数量必须是正整数。"
    exit 1
fi

job_prefix="$(awk '$1 == "job_prefix:" {print $2; exit}' "${run_dir}/settings.yaml")"
mpi_ranks="$(awk '$1 == "mpi_ranks:" {print $2; exit}' "${run_dir}/settings.yaml")"
mem_request="$(awk '$1 == "mem_request:" {print $2; exit}' "${run_dir}/settings.yaml")"
mem_request="${mem_request:-400G}"
if [ -z "${job_prefix}" ] || ! [[ "${mpi_ranks}" =~ ^[0-9]+$ ]] || [ "${mpi_ranks}" -eq 0 ]; then
    echo "settings.yaml 中的 job_prefix 或 mpi_ranks 无效。"
    exit 1
fi

queue_names="$(timeout 10 squeue -u "$(id -un)" -h -o '%j')"
queue_code=$?
if [ "${queue_code}" -ne 0 ]; then
    echo "squeue 查询失败，停止提交。"
    exit 1
fi

total_active="$(printf '%s\n' "${queue_names}" | awk 'NF {n++} END {print n+0}')"
free_slots=$((safe_limit - total_active))
echo "方案：${scheme}"
echo "当前全部队列任务：${total_active}"
echo "安全上限：${safe_limit}"
echo "本方案每个作业：${mpi_ranks} rank / ${mpi_ranks} GPU"
echo "可用安全名额：${free_slots}"

if [ "${free_slots}" -le 0 ]; then
    echo "没有可用队列名额。"
    exit 0
fi

mkdir -p "${run_dir}/log" "${run_dir}/status"
submitted=0
while IFS= read -r conf || [ -n "${conf}" ]; do
    [ -z "${conf}" ] && continue
    [[ "${conf}" == \#* ]] && continue
    [[ "${conf}" =~ ^[0-9]+$ ]] || continue

    input_file="${run_dir}/input/input_${conf}.yaml"
    result_file="${run_dir}/results/corr_DbarN_conf${conf}.npz"
    if [ ! -f "${input_file}" ]; then
        echo "跳过 ${conf}：没有输入文件"
        continue
    fi
    if [ -f "${result_file}" ]; then
        echo "跳过 ${conf}：结果已存在"
        continue
    fi
    if [ -f "${run_dir}/status/failed_${conf}.txt" ]; then
        echo "跳过 ${conf}：有失败标记"
        continue
    fi

    job_name="${conf}_${job_prefix}"
    if printf '%s\n' "${queue_names}" | grep -Fxq "${job_name}"; then
        echo "跳过 ${conf}：队列中已经存在"
        continue
    fi

    output="$(sbatch --parsable \
        --job-name="${job_name}" \
        --output="${run_dir}/log/${conf}_%j.out" \
        --error="${run_dir}/log/${conf}_%j.out" \
        --partition=dgx1 \
        --account=test \
        --nodelist=gpu030 \
        --nodes=1 \
        --ntasks-per-node="${mpi_ranks}" \
        --gres="gpu:v100:${mpi_ranks}" \
        --cpus-per-task=1 \
        --mem="${mem_request}" \
        --time=2-00:00:00 \
        "${vs_dir}/job.sh" "${run_dir}" "${conf}")"
    if [ "$?" -ne 0 ] || [ -z "${output}" ]; then
        echo "组态 ${conf} 提交失败"
        continue
    fi

    job_id="${output%%;*}"
    echo "${job_id}" > "${run_dir}/status/submitted_${conf}.txt"
    echo "已提交 ${conf}，Job ID：${job_id}，GPU：${mpi_ranks}"
    queue_names="${queue_names}"$'\n'"${job_name}"
    submitted=$((submitted + 1))

    if [ "${submitted}" -ge "${request_number}" ] || [ "${submitted}" -ge "${free_slots}" ]; then
        break
    fi
done < "${run_dir}/todo.txt"

echo "本次成功提交：${submitted}"
