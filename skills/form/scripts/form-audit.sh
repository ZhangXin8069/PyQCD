#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)

# 全局实现的默认位置随安装用户而变（这里是 $HOME/configure，别处可能是
# /root/configure），因此按候选顺序解析，而不是写死一条绝对路径。
GLOBAL_AUDIT=${FORM_GLOBAL_AUDIT:-}
SEARCHED=()
if [[ -z "$GLOBAL_AUDIT" ]]; then
  for candidate in \
      "${XDG_CONFIG_HOME:-$HOME/.config}/configure/skills/form/scripts/form-audit.sh" \
      "$HOME/configure/skills/form/scripts/form-audit.sh" \
      "/root/configure/skills/form/scripts/form-audit.sh" \
      "/usr/local/share/configure/skills/form/scripts/form-audit.sh"; do
    SEARCHED+=("$candidate")
    if [[ -x "$candidate" ]]; then
      GLOBAL_AUDIT=$candidate
      break
    fi
  done
fi

if [[ -z "$GLOBAL_AUDIT" || ! -x "$GLOBAL_AUDIT" ]]; then
  printf 'form-audit(pyqcd): missing global audit implementation.\n' >&2
  printf '  set FORM_GLOBAL_AUDIT=/path/to/form-audit.sh, or install "configure" so one of:\n' >&2
  for candidate in "${SEARCHED[@]}"; do printf '    %s\n' "$candidate" >&2; done
  exit 2
fi

RAW_AUDIT=$("$GLOBAL_AUDIT" --root "$REPO_ROOT" --quiet)
EXPECTED=0
declare -a UNEXPECTED=()

# 已登记例外：
#  * docs/en/*、docs/zh/* 的 DOC_EXTENSION —— 双语 Read the Docs 站点源文件
#    （conf.py / index.rst / requirements.txt / .readthedocs.yaml）就是文档本体；
#  * cpp/**/tests/* 的 TEST_LOCATION —— C++ 单测与源码树同层，属于 C++ 构建约定。
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
    WARNING$'\t'DOC_EXTENSION$'\t'docs/_tree_pyqcd.txt$'\t'* | \
    WARNING$'\t'DOC_EXTENSION$'\t'docs/en/*$'\t'* | \
    WARNING$'\t'DOC_EXTENSION$'\t'docs/zh/*$'\t'*)
      EXPECTED=$((EXPECTED + 1))
      ;;
    WARNING$'\t'TEST_LOCATION$'\t'cpp/*/tests/*$'\t'* | \
    WARNING$'\t'TEST_LOCATION$'\t'cpp/*/*/tests/*$'\t'* | \
    WARNING$'\t'TEST_LOCATION$'\t'cpp/*/*/*/tests/*$'\t'*)
      EXPECTED=$((EXPECTED + 1))
      ;;
    WARNING$'\t'TAG_NAME_IN_PATH$'\t'docs/from-logs/*$'\t'* | \
    WARNING$'\t'TAG_NAME_IN_PATH$'\t'logs/*$'\t'* | \
    WARNING$'\t'TAG_NAME_IN_PATH$'\t'data/from-logs/*$'\t'* | \
    WARNING$'\t'TAG_NAME_IN_PATH$'\t'pyqcd/testing/tmd/artifacts/*/docs/analy_test9_*.pdf$'\t'* | \
    WARNING$'\t'TAG_NAME_IN_PATH$'\t'pyqcd/testing/tmd/artifacts/*/docs/analy_test9_*.tex$'\t'*)
      EXPECTED=$((EXPECTED + 1))
      ;;
    WARNING$'\t'FILE_UPPERCASE$'\t'pyqcd/testing/regression/baselines/docker_support/download_beta6.20_mu-0.2770_ms-0.2400_L24x72.sh$'\t'* | \
    WARNING$'\t'FILE_UPPERCASE$'\t'pyqcd/testing/regression/baselines/docker_support/download_beta6.20_mu-0.2770_ms-0.2400_L24x72-v20260801.sh$'\t'* | \
    WARNING$'\t'FILE_UPPERCASE$'\t'pyqcd/testing/regression/baselines/docker_support/pack_beta6.20_mu-0.2770_ms-0.2400_L24x72.sh$'\t'* | \
    WARNING$'\t'FILE_UPPERCASE$'\t'pyqcd/testing/regression/baselines/docker_support/pack_beta6.20_mu-0.2770_ms-0.2400_L24x72-v20260801.sh$'\t'* | \
    WARNING$'\t'TAG_NAME_IN_PATH$'\t'docs/absorb_dev20_gluon_tmd_audit_20261022.md$'\t'*)
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
