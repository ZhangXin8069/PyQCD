#!/usr/bin/env bash
set -euo pipefail

GLOBAL_AUDIT=${FORM_GLOBAL_AUDIT:-/root/configure/skills/form/scripts/form-audit.sh}
REPO_ROOT=$(git rev-parse --show-toplevel)

if [[ ! -x "$GLOBAL_AUDIT" ]]; then
  printf 'form-audit(pyqcd): missing global audit: %s\n' "$GLOBAL_AUDIT" >&2
  exit 2
fi

_registered_reference_doc() {
  local path=$1
  grep -Fxq "$path" <<'EOF'
refer/git-rep/EasyDistillation/docs/analy_distillation_20260817.pdf
refer/git-rep/EasyDistillation/docs/analy_distillation_20260817.tex
refer/git-rep/EasyDistillation/docs/pure_distillation_20260815.pdf
refer/git-rep/EasyDistillation/docs/pure_distillation_20260815.tex
refer/git-rep/EasyDistillation/docs/pure_distillation_20260817.pdf
refer/git-rep/EasyDistillation/docs/pure_distillation_20260817.tex
refer/git-rep/EasyDistillation/docs/report_distillation_20260828.tex
refer/git-rep/LQCD_Master/docs/analy_qcd_master_20260817.pdf
refer/git-rep/LQCD_Master/docs/analy_qcd_master_20260817.tex
refer/git-rep/LQCD_Master/docs/pure_qcd_master_20260817.pdf
refer/git-rep/LQCD_Master/docs/pure_qcd_master_20260817.tex
refer/git-rep/LQCD_Master/docs/pure_qcd_wick_codegen_20260815.pdf
refer/git-rep/LQCD_Master/docs/pure_qcd_wick_codegen_20260815.tex
refer/git-rep/PyQUDA/docs/analy_pyquda_20260817.pdf
refer/git-rep/PyQUDA/docs/analy_pyquda_20260817.tex
refer/git-rep/PyQUDA/docs/pure_pyquda_20260817.pdf
refer/git-rep/PyQUDA/docs/pure_pyquda_20260817.tex
refer/git-rep/PyQUDA/docs/pure_pyquda_core_20260815.pdf
refer/git-rep/PyQUDA/docs/pure_pyquda_core_20260815.tex
refer/git-rep/lamet-agent/docs/analy_lamet_20260817.pdf
refer/git-rep/lamet-agent/docs/analy_lamet_20260817.tex
refer/git-rep/lamet-agent/docs/pure_core_20260815.pdf
refer/git-rep/lamet-agent/docs/pure_core_20260815.tex
refer/git-rep/lamet-agent/docs/pure_lamet_20260817.pdf
refer/git-rep/lamet-agent/docs/pure_lamet_20260817.tex
EOF
}

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
    WARNING$'\t'TAG_NAME_IN_PATH$'\t'docs/from-logs/*$'\t'* | \
    WARNING$'\t'TAG_NAME_IN_PATH$'\t'logs/*$'\t'* | \
    WARNING$'\t'TAG_NAME_IN_PATH$'\t'data/from-logs/*$'\t'* | \
    WARNING$'\t'TAG_NAME_IN_PATH$'\t'pyqcd/testing/tmd/artifacts/*/docs/analy_test9_*.pdf$'\t'* | \
    WARNING$'\t'TAG_NAME_IN_PATH$'\t'pyqcd/testing/tmd/artifacts/*/docs/analy_test9_*.tex$'\t'*)
      EXPECTED=$((EXPECTED + 1))
      ;;
    WARNING$'\t'REFERENCE_DOC_MISPLACED$'\t'*)
      ref_path=${line#WARNING$'\t'REFERENCE_DOC_MISPLACED$'\t'}
      ref_path=${ref_path%%$'\t'*}
      if _registered_reference_doc "$ref_path"; then
        EXPECTED=$((EXPECTED + 1))
      else
        UNEXPECTED+=("$line")
      fi
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
