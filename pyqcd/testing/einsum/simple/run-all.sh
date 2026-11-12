#!/usr/bin/env bash
# 一键跑全部 simple 检查；任一失败即非零退出。
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../../.." && pwd)
PY=${PY:-$(command -v python3)}
STATUS=0

run() {
    printf '\n===== %s =====\n' "$*"
    "$@" || STATUS=1
}

cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

run "$PY" -m pyqcd.testing.einsum.simple.check_compat
run "$PY" -m pyqcd.testing.einsum.simple.check_memory
run "$PY" -m pyqcd.testing.einsum.simple.check_equations
run env PYQCD_EINSUM_NATIVE=kupl "$PY" -m pyqcd.testing.einsum.simple.check_native --backend kupl
run env PYQCD_EINSUM_NATIVE=cuda "$PY" -m pyqcd.testing.einsum.simple.check_native --backend cuda

printf '\n===== simple suite exit=%s =====\n' "$STATUS"
exit "$STATUS"
