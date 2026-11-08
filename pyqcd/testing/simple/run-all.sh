#!/usr/bin/env bash
# 依次运行 pyqcd/testing/simple 下的全部单功能对照测试；任一失败即非零退出。
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)
PY=${PY:-$(command -v python3)}
STATUS=0

run() {
    printf '\n===== %s =====\n' "$*"
    "$@" || STATUS=1
}

cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

run "$PY" -m pyqcd.testing.simple.check_contract
run "$PY" -m pyqcd.testing.simple.check_contract_path
run "$PY" -m pyqcd.testing.simple.check_contract_expression
run "$PY" -m pyqcd.testing.simple.check_shared_intermediates
run "$PY" -m pyqcd.testing.simple.check_helpers
run "$PY" -m pyqcd.testing.simple.check_paths
run "$PY" -m pyqcd.testing.simple.check_blas
run "$PY" -m pyqcd.testing.simple.check_parser
run "$PY" -m pyqcd.testing.simple.check_backends
run "$PY" -m pyqcd.testing.simple.check_specialize

printf '\n===== pyqcd/testing/simple exit=%s =====\n' "$STATUS"
exit "$STATUS"
