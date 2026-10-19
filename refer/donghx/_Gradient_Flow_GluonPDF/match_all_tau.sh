#!/usr/bin/env bash
set -euo pipefail

# Match all available per-configuration OPE files.  Original flowed-bare
# products are preserved.
ROOT="${1:-output_v5}"
OUT="${2:-matched_quasi_v1}"
MU="${MU_GEV:-2.0}"
AS="${ALPHA_S:-0.25}"
A="${A_FM:-0.0775}"
for tau_dir in "$ROOT"/conf*/tau*; do
    [[ -d "$tau_dir" ]] || continue
    tau_tag=$(basename "$tau_dir")
    tau=${tau_tag#tau}; tau=${tau//p/.}
    python matching/gluon_flow_to_quasi.py \
      --input "$tau_dir" \
      --output "$OUT"/$(basename "$(dirname "$tau_dir")")/"$tau_tag" \
      --tau "$tau" --a-fm "$A" --mu "$MU" --alpha-s "$AS"
done
