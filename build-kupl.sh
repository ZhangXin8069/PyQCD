#!/usr/bin/env bash
# Build the KUPL einsum backend: cpp/kupl/einsum.
#
# "kupl" is the stable backend identifier (never "cpu") so a future Kunpeng
# port can reuse the same name, flags and environment contract.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
# shellcheck source=/dev/null
source "$ROOT/env.sh"

printf '=== PyQCD einsum build (kupl) ===\n'
bash "$ROOT/cpp/kupl/einsum/make.sh" "$@"
printf '=== PyQCD einsum build (kupl): SUCCESS ===\n'
