#!/usr/bin/env bash
set -euo pipefail

GLOBAL_AUDIT=${FORM_GLOBAL_AUDIT:-/root/configure/skills/form/scripts/form-audit.sh}
REPO_ROOT=$(git rev-parse --show-toplevel)

if [[ ! -x "$GLOBAL_AUDIT" ]]; then
  printf 'form-audit(pyqcd): missing global audit: %s\n' "$GLOBAL_AUDIT" >&2
  exit 2
fi

RAW_AUDIT=$("$GLOBAL_AUDIT" --root "$REPO_ROOT" --quiet)
EXPECTED=0
declare -a UNEXPECTED=()

while IFS= read -r line; do
  case "$line" in
    ERROR$'\t'TRACKED_DATA_CONTENT$'\t'data/*$'\t'*)
      EXPECTED=$((EXPECTED + 1))
      ;;
    WARNING$'\t'LOG_EXTENSION$'\t'logs/AGENTS.md$'\t'* | \
    WARNING$'\t'LOG_EXTENSION$'\t'logs/*/AGENTS.md$'\t'*)
      EXPECTED=$((EXPECTED + 1))
      ;;
    WARNING$'\t'DOC_EXTENSION$'\t'docs/.gitignore$'\t'* | \
    WARNING$'\t'DOC_EXTENSION$'\t'docs/_tree_pyqcd.txt$'\t'*)
      EXPECTED=$((EXPECTED + 1))
      ;;
    WARNING$'\t'FILE_UPPERCASE$'\t'pyqcd/testing/regression/baselines/docker_support/download_beta6.20_mu-0.2770_ms-0.2400_L24x72.sh$'\t'* | \
    WARNING$'\t'FILE_UPPERCASE$'\t'pyqcd/testing/regression/baselines/docker_support/download_beta6.20_mu-0.2770_ms-0.2400_L24x72-v20260801.sh$'\t'* | \
    WARNING$'\t'FILE_UPPERCASE$'\t'pyqcd/testing/regression/baselines/docker_support/pack_beta6.20_mu-0.2770_ms-0.2400_L24x72.sh$'\t'* | \
    WARNING$'\t'FILE_UPPERCASE$'\t'pyqcd/testing/regression/baselines/docker_support/pack_beta6.20_mu-0.2770_ms-0.2400_L24x72-v20260801.sh$'\t'*)
      EXPECTED=$((EXPECTED + 1))
      ;;
    '')
      ;;
    *)
      UNEXPECTED+=("$line")
      ;;
  esac
done <<<"$RAW_AUDIT"

if ((${#UNEXPECTED[@]})); then
  printf 'FAIL form-audit(pyqcd): unexpected findings=%d\n' \
    "${#UNEXPECTED[@]}" >&2
  printf '%s\n' "${UNEXPECTED[@]}" >&2
  exit 1
fi

printf 'PASS form-audit(pyqcd): registered exceptions=%d unexpected=0\n' \
  "$EXPECTED" >&2
