#!/usr/bin/env bash
# test7 正式运行入口：bash run.sh ≡ bash ./run-local.sh --server
# （nohup 后台正式跑；日志 run-server-<TS>.log，tail -f 实时调控，kill 停止）
set -uo pipefail
WORK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$WORK/run-local.sh" --server "$@"
