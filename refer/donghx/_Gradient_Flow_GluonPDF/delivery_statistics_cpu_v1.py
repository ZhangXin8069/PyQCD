"""Aggregate existing quality ledgers on a CPU allocation; no new physics fit."""

import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent


def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Quality aggregation requires a CPU allocation")
    source = ROOT / "00_audit/interpretation_v3"
    inputs = [source / "bare_replica_fallback.csv", source / "summary.json"]
    bare = load_csv(inputs[0])
    prior = json.loads(inputs[1].read_text())
    aggregates = []
    for ensemble in ("L32x64", "L32x96", "L24x72"):
        rows = [r for r in bare if r["ensemble"] == ensemble]
        n = sum(int(r["nonconstraint"]) for r in rows)
        entries = sum(int(r["entries"]) for r in rows)
        available = sum(int(r["available"]) for r in rows)
        center = sum(int(r["central_fallback"]) for r in rows)
        any_row = sum(int(r["any_replica_fallback"]) for r in rows)
        fraction = sum(
            int(r["nonconstraint"]) * float(r["mean_bootstrap_fallback_fraction"])
            for r in rows
        ) / n
        aggregates.append(dict(
            ensemble=ensemble, selected=available, entries=entries,
            nonconstraint_entries=n, central_fallback=center,
            central_fallback_fraction=center / n,
            any_replica_fallback=any_row,
            any_replica_fallback_fraction=any_row / n,
            mean_replica_fallback_fraction=fraction,
            flow_count=len(set(r["tau"] for r in rows)),
        ))
    flow = []
    for policy in ("allflow_display_only", "D_loose"):
        rows = [r for r in prior["flow"] if r["policy"] == policy]
        flow.append(dict(
            policy=policy,
            finite_nonlocal=sum(r["finite_nonlocal"] for r in rows),
            central_quality_nonlocal=sum(r["central_quality_nonlocal"] for r in rows),
            strict_upstream_quality_nonlocal=sum(
                r["strict_upstream_quality_nonlocal"] for r in rows
            ),
        ))
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    result = dict(
        timestamp=timestamp, job=os.environ["SLURM_JOB_ID"], nboot=200,
        bare=aggregates, continuum_flow=flow,
        numerical_summary_sha256=digest(inputs[1]),
        input_sha256={str(p): digest(p) for p in inputs},
        script_sha256=digest(Path(__file__)), physics_validated=False,
        note="All fractions exclude exact odd-z0 constraints; fallback means a cut/window "
             "needed the registered rejected finite candidate, not a fabricated value. "
             "An any-replica flag is NOT a statement that all 200 replicas failed.",
    )
    out = ROOT / "00_audit/delivery_statistics_v1"
    out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# 本轮质量总表", "", timestamp, "",
        "本表仅汇总已完成的 CPU 数值诊断，不重拟合、不放宽质量门。", "",
        "## 裸拟合", "",
        "回退比例分母排除严格的 odd-z0 代数零。L32x64 的 8 个 flow 包含额外 tau=.1 裸诊断，"
        "其余两套为 7 个共同物理 flow。selected 覆盖包含代数零。", "",
        "|ensemble|selected覆盖|非约束点数|中心回退点数/比例|任一bootstrap行回退点数/比例|平均bootstrap回退比例|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in aggregates:
        lines.append(
            f'|{r["ensemble"]}|{r["selected"]}/{r["entries"]}|{r["nonconstraint_entries"]}|'
            f'{r["central_fallback"]} / {r["central_fallback_fraction"]:.2%}|'
            f'{r["any_replica_fallback"]} / {r["any_replica_fallback_fraction"]:.2%}|'
            f'{r["mean_replica_fallback_fraction"]:.2%}|'
        )
    lines += [
        "", "- 中心回退：原始 ensemble 中心值至少一个 cut 需回退。",
        "- 任一行回退：200 行中至少一行至少一个 cut 需回退；不等于 200 行全回退。",
        "- 平均bootstrap回退比例：全部非约束点、全部200行上的回退标记占比。",
        "- strict_quality_nonconstraint：继承每个贡献输入及分母的严格质量，且无任一行回退，"
        "同时满足当前步骤拟合门；它比仅中心 Q/chi2 通过严格得多。", "",
        "## 连续极限及小流时间诊断", "",
        f'固定物理 tf 下 a² 主支：{prior["continuum_primary_finite_nonlocal"]} 个有限非局域拟合，'
        f'其中 {prior["continuum_primary_central_quality_nonlocal"]} 个中心 Q>=.05、chi2/dof<=3；'
        f'{prior["continuum_primary_strict_upstream_quality_nonlocal"]} 个继承完整严格上游质量。', "",
        "|连续后flow窗口|有限非局域点数|中心Q/chi2通过|完整严格上游质量通过|",
        "|---|---:|---:|---:|",
    ]
    for r in flow:
        lines.append(
            f'|{r["policy"]}|{r["finite_nonlocal"]}|{r["central_quality_nonlocal"]}|'
            f'{r["strict_upstream_quality_nonlocal"]}|'
        )
    lines += [
        "", "这些数量跨 channel/flavor/P/z/component/flow 汇总，格点之间相关，不能当独立检验计数。",
        "D_loose 是宽松尺度窗而非强层级保证；短距离缺失保持 NaN，不用 allflow 替代为质量通过。", "",
        f'连续分支短距离 m0：{prior["primary_m0_fits_executed"]}/9 组执行，'
        f'{prior["primary_m0_quality_pass"]}/9 组通过 Q>=.05 且 chi2/dof<=3。',
        "具体值及误差见 ../interpretation_v3/m0_primary.csv。统计误差小不能弥补模型失配，"
        "不建议将这些 m0 当成已验收的 MSbar 修正。", "",
        "本轮尚未闭合质量/体积/O(a)/激发态/流窗/NP-PT方案等系统误差；不能称最终物理 PDF。", "",
        "生成日期：2026-11-08（Asia/Shanghai）。会话：三ensemble、100/40/30 conf、200 bootstrap 与探索性连续极限。",
    ]
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("DELIVERY_QUALITY_SUMMARY_COMPLETE", str(out), flush=True)


if __name__ == "__main__":
    main()
