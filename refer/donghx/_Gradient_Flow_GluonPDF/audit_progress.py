#!/usr/bin/env python3
"""Read-only progress inventory; header/metadata checks, not a payload revalidation."""
import csv
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
TAUS = [(0.05, 2), (.1, 2), (.2, 3), (.3, 3), (.4, 4), (.5, 4), (.8, 6), (1., 7)]
CONF96 = list(range(10000, 11000, 100))+[11000, 11100]+list(range(12300, 14100, 100))
STAMP = datetime.now().astimezone().isoformat()
ROWS, DETAILS, WARNINGS = [], [], []
HEADER_CACHE = {}


def load(path):
    return json.loads(Path(path).read_text())


def safe_load(path):
    try:
        return load(path)
    except (OSError, ValueError) as exc:
        return {"_error": str(exc)}


def tag(tau, window):
    return f"thin_tf{tau:.8f}_eps0.01000000_tw{window}_Nv400_dz21".replace(".", "p")


def csv_write(path, records):
    fields = list(dict.fromkeys(key for row in records for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def row(ensemble, observable, campaign, scheme, tau, target, complete, **extra):
    r = dict(snapshot_time=STAMP, ensemble=ensemble, observable=observable,
             campaign=campaign, weight_scheme=scheme, tau_tf_over_a2=tau,
             a_fm=.0775 if ensemble == "L32x96" else .0897,
             epsilon=.01 if isinstance(tau, (float, int)) else "",
             nstep=round(tau/.01) if isinstance(tau, (float, int)) else "",
             target_nconf=target, complete_nconf=complete,
             completion_percent=round(100*complete/target, 3) if target else "")
    r.update(extra)
    ROWS.append(r)


def raw_header(path, nt):
    """Read ZIP inventory and embedded NPY headers only, never C3 payload."""
    key = (str(path.resolve()), nt)
    if key in HEADER_CACHE:
        return HEADER_CACHE[key]
    expected = {"corr_2pt.npy": (4, 3, nt, nt),
                "corr_3pt_u.npy": (2, 4, 3, 21, 4, 4, nt, nt),
                "corr_3pt_d.npy": (2, 4, 3, 21, 4, 4, nt, nt)}
    try:
        with zipfile.ZipFile(path) as archive:
            for name, shape_expected in expected.items():
                with archive.open(name) as stream:
                    version = np.lib.format.read_magic(stream)
                    shape, _, dtype = np.lib.format._read_array_header(stream, version)
                if shape != shape_expected or dtype != np.dtype("complex128"):
                    raise ValueError(f"{name}: shape/dtype {shape}/{dtype}")
                if archive.getinfo(name).file_size < int(np.prod(shape))*16:
                    raise ValueError(f"{name}: truncated member")
        result = "PASS_HEADERS_ONLY"
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, EOFError) as exc:
        result = f"INVALID: {exc}"
    HEADER_CACHE[key] = result
    return result


def raw_conf(roots, conf, tau, window, nt, tseps):
    candidates = []
    for root in roots:
        base = root/tag(tau, window)/"Nev_200"/str(conf)
        if not base.is_dir():
            continue
        timing = safe_load(base/"timing_summary.json")
        ok = []
        issues = []
        for tsep in tseps:
            file = base/f"corr_projected_tsep{tsep}_dz0-20_Nev200.npz"
            if not file.is_file():
                issues.append(f"missing_tsep{tsep}")
            else:
                check = raw_header(file, nt)
                if check == "PASS_HEADERS_ONLY":
                    ok.append(tsep)
                else:
                    issues.append(f"tsep{tsep}:{check}")
        metadata_ok = str(timing.get("conf")) == str(conf) and timing.get("flow_tag") == tag(tau, window)
        if not metadata_ok:
            issues.append("timing_identity_missing_or_wrong")
        candidates.append(dict(path=str(base), found_tseps=ok,
             complete=len(ok) == len(tseps) and metadata_ok,
             issues=issues, timing_schema=timing.get("schema", "")))
    if not candidates:
        return dict(path="", found_tseps=[], complete=False, issues=["no_current_product_directory"])
    return max(candidates, key=lambda r:(r["complete"], len(r["found_tseps"])))


def quark_inventory():
    canonical96 = Path("/public/group/lqcd/donghx/Gradient_Flow/Result/beta6.41_mu-0.2295_ms-0.2050_L32x96/thin_link")
    roots96 = [canonical96, ROOT/"quark_flow_products",
        ROOT/"quark_spin2_production_20260922/production_tf0p5_pilot_z",
        ROOT/"quark_spin2_expansion20_20260922/results_tf0p5_pilot_z",
        ROOT/"quark_spin2_fill_flows_20260923/results_remaining",
        ROOT/"quark_spin2_expansion30_20260925/results_all6"]
    root64 = ROOT/"quark_l32x64_flow_20261007/results"
    final64 = ROOT/"quark_l32x64_expand30_20261012/validation/final_audit.json"
    accepted64 = {(int(r["conf"]), float(r["tau_a2"])) for r in load(final64)["complete"]}
    for ensemble, confs, roots, nt, tseps in [
        ("L32x96", CONF96, roots96, 96, list(range(6, 11))),
        ("L32x64", list(range(10000, 11200, 40)), [root64], 64, list(range(5, 11)))]:
        for tau, window in TAUS:
            complete, partial, issues, complete_ids = 0, 0, [], []
            for conf in confs:
                result = raw_conf(roots, conf, tau, window, nt, tseps)
                if ensemble == "L32x64" and (conf,tau) not in accepted64:
                    result["complete"] = False
                    result["issues"].append("not_in_prior_final_validation")
                status = "COMPLETE_HEADER_METADATA_CHECKED" if result["complete"] else "PARTIAL" if result["found_tseps"] else "MISSING"
                complete += int(result["complete"])
                partial += int(not result["complete"] and bool(result["found_tseps"]))
                if result["complete"]:
                    complete_ids.append(str(conf))
                else:
                    issues.append(f"{conf}:"+";".join(result["issues"]))
                DETAILS.append(dict(snapshot_time=STAMP,ensemble=ensemble,
                    observable="quark_C2_C3_raw",campaign="legacy_A_full_source_current",tau_tf_over_a2=tau,
                    conf_id=conf,status=status,found_tseps=";".join(map(str,result["found_tseps"])),
                    data_path=result["path"],notes=";".join(result["issues"])))
            row(ensemble,"quark_C2_C3_raw","legacy_A_full_source_current","A_legacy_unified_table",tau,len(confs),complete,
                status="COMPLETE" if complete == len(confs) else "INCOMPLETE",
                partial_nconf=partial,missing_nconf=len(confs)-complete-partial,
                temporal_windows=str(window),pz_integer="+3;+4;+5",tsep_a=";".join(map(str,tseps)),
                source_current_coverage=f"{nt} sources and {nt} currents; common tseps only",
                channel_flavor="open_current_spin_4x4;u;d;u-d_by_analysis",
                completed_conf_ids=";".join(complete_ids),data_path=";".join(map(str,roots)),
                evidence_path=str(final64) if ensemble=="L32x64" else "timing_summary.json + embedded corr_2pt/corr_3pt NPY headers",
                validation_level="NPZ_inventory+shape_dtype_headers+metadata; no new full_payload_finite_or_CRC_scan",
                notes="原A方案；C2/C3产物可供后续构造ratio，不代表新统计量拟合或matching完成。"+("; "+" | ".join(issues) if issues else ""))
            print(f"{ensemble} legacy tau={tau:g} {complete}/{len(confs)}",flush=True)
    # Old extra-flow submissions are not completion evidence; inspect actual products.
    for tau, window in ((.6,5),(1.1,7),(1.2,8),(1.5,9)):
        checked = [raw_conf(roots96, c, tau, window, 96, list(range(6,11))) for c in CONF96[:10]]
        done = sum(c["complete"] for c in checked)
        row("L32x96","quark_C2_C3_raw","historical_extra_flow_check","A_legacy_unified_table",tau,10,done,
            status="COMPLETE" if done==10 else "CURRENT_PRODUCTS_NOT_CONFIRMED" if done==0 else "PARTIAL",
            temporal_windows=str(window),pz_integer="+3;+4;+5",tsep_a="6;7;8;9;10",
            data_path=";".join(map(str,roots96)),validation_level="known_output_roots_header_metadata_check",
            notes="仅检查历史额外flow产物；目录缺失不证明从未计算，不能把旧提交日志当作完成。窗口取历史提交设置，其他存档路径待确认。")
    # B diagnostic is a different estimator/coverage from the legacy production.
    b = ROOT/"quark_l32x64_P4_B_20conf_20261015"
    audit = load(b/"summary/complete.json")
    assert audit["status"] == "COMPLETE_20CONF_FINITE_WINDOW_DIAGNOSTIC" and audit["case_count"] == 1440
    cases = audit["cases"]
    for tau in (.2,.5,.8):
        report = load(b/f"summary/tau{tau:g}/report.json")
        selected = [c for c in cases if abs(c["flow_time_a2"]-tau)<1e-12]
        counts = Counter(c["conf"] for c in selected)
        complete = sum(v==24 for v in counts.values())
        row("L32x64","quark_C2_C3_and_paired_window_ratio","B_20conf_selected_current_midpoint","B_same_time_second_cross_time_product",tau,20,complete,
            status="COMPLETE_FINITE_WINDOW_DIAGNOSTIC",target_cases=480,complete_cases=len(selected),
            temporal_windows=";".join(map(str,report["windows"])),pz_integer="-6;-5;-4;-3;+3;+4;+5;+6",tsep_a="7;8;9;10;11;12",
            source_current_coverage="selected currents 0;16;32;63; midpoint insertion dt=floor(tsep/2)",
            channel_flavor="open_current_spin_4x4;u;d;paired_u-d",
            completed_conf_ids=";".join(sorted(counts,key=int)),data_path=str(b/"C3_schemeB_Pz3456_tsep7to12"),
            evidence_path=str(b/"summary/complete.json"),validation_level="existing_final_CPU_full_validation_1440_cases",
            related_jobs="3853935;3853936;3853937",notes="不能与A方案合并；有限嵌套时间窗测试，不是全current/all-insertion或精确全时间积分证明。")
    extra40 = ROOT/"quark_l32x64_expand40_20261013"
    if not (extra40/"manifest.json").is_file() and not (extra40/"submission.json").is_file():
        row("L32x64","quark_C2_C3_raw","extension40_prepared_only","A_legacy_unified_table","all_8_planned_flows",10,0,
            status="PREPARED_NO_SUBMISSION_RECEIPT",data_path=str(extra40),
            validation_level="no_frozen_manifest_or_submission_receipt",notes="已有代码不代表已提交/完成；目前正式原生产目标仍为30conf，不能报40conf。")


def npy_header(path, shape=None, dtype=None):
    try:
        with Path(path).open("rb") as handle:
            version = np.lib.format.read_magic(handle)
            got_shape, _, got_dtype = np.lib.format._read_array_header(handle,version)
            offset = handle.tell()
        return (shape is None or tuple(got_shape)==tuple(shape)) and (dtype is None or got_dtype==np.dtype(dtype)) and Path(path).stat().st_size >= offset+int(np.prod(got_shape))*got_dtype.itemsize
    except (OSError, ValueError, EOFError):
        return False


def loops_inventory():
    root = ROOT/"line_mass_L32x64_allconf_B_20261015/runs/allconf_B_2026-10-15T205426.224554_0800"
    manifest = load(root/"manifest.json")
    sources = manifest["data_sources"]
    records = list((root/"submission_workers_v1").glob("worker[0-9]*.json"))
    completed, active, failed = set(), set(), set()
    worker_states = Counter()
    for path in records:
        r = safe_load(path)
        worker_states[r.get("state","UNKNOWN")] += 1
        completed.update(c["conf_id"] for c in r.get("completed",[]))
        failed.update(c["conf_id"] for c in r.get("failed",[]))
        if r.get("state")=="RUNNING":
            done_ids = {c["task"] for c in r.get("completed",[])+r.get("failed",[])}
            remaining = [t for t in r["tasks"] if t not in done_ids]
            if remaining:
                active.add(manifest["tasks"][remaining[0]]["conf_id"])
    checkpoint = {t:{"Polyakov_loop":set(),"Wilson_loop":set()} for t in manifest["flow"]["taus"]}
    full = set()
    for i, source in enumerate(sources):
        path, conf = Path(source["path"]), str(source["conf_id"])
        commit = safe_load(path/"complete.json")
        is_full = commit.get("status")=="complete" and str(commit.get("conf_id"))==conf
        ptaus, wtaus = [], []
        for tau in checkpoint:
            base = path/("tau%.3f"%tau).replace(".","p")
            meta = safe_load(base/"metadata.json")
            identity = meta.get("status")=="measured_unrenormalized_polyakov_loop" and str(meta.get("conf_id"))==conf and abs(meta.get("tau_t_over_a2",-99)-tau)<1e-10
            pok = identity and npy_header(base/"p3.npy",(32,32,32),"complex128") and npy_header(base/"p8.npy",(32,32,32),"float64")
            if pok:
                checkpoint[tau]["Polyakov_loop"].add(conf)
                ptaus.append(tau)
            closed = safe_load(base/"closed_loops.json")
            wok = identity and closed.get("schema")=="closed_loops_v1" and closed.get("status")=="measured_bare_single_configuration"
            wok = wok and npy_header(base/"wilson_fundamental.npy",(3,13,17),"complex128") and npy_header(base/"wilson_adjoint.npy",(3,13,17),"float64")
            if wok:
                checkpoint[tau]["Wilson_loop"].add(conf)
                wtaus.append(tau)
        if is_full and len(ptaus)==len(checkpoint) and len(wtaus)==len(checkpoint) and (conf in completed or source["kind"] != "new_allconf_measurement"):
            full.add(conf)
        DETAILS.append(dict(snapshot_time=STAMP,ensemble="L32x64",observable="Wilson_and_Polyakov_loop",
            campaign="all1700",conf_id=conf,status="COMPLETE_FULL_GRID" if conf in full else "RUNNING" if conf in active else "FAILED" if conf in failed else "PARTIAL_CHECKPOINTS" if ptaus or wtaus else "NOT_STARTED",
            completed_polyakov_taus=";".join(map(str,ptaus)),completed_wilson_taus=";".join(map(str,wtaus)),data_path=str(path)))
        if (i+1)%200==0:
            print(f"loop progress checked {i+1}/{len(sources)}",flush=True)
    for tau, obs in checkpoint.items():
        for name, ids in obs.items():
            row("L32x64",name,"all1700","not_applicable_pure_gauge",tau,len(sources),len(ids),
                status="COMPLETE" if len(ids)==len(sources) else "IN_PROGRESS",
                full_grid_validated_nconf=len(full),running_nconf=len(active),failed_nconf=len(failed),
                missing_checkpoint_nconf=len(sources)-len(ids),data_path=str(root/"data"),
                evidence_path=str(root/"submission_workers_v1"),related_jobs="3853828;3853829",
                validation_level="checkpoint_JSON_identity+NPY_shape_dtype_size; full_grid_worker_validation_receipts",
                notes=("单flow完成数包含运行中组态已提交的checkpoint；full_grid列才是全部23flow已完成。纯规范量不使用B权重；B仅为后续C3上下文。"+
                    ("Fundamental/adjoint;XT,YT,ZT;R=0..12;T=0..16" if name=="Wilson_loop" else "Fundamental complex/adjoint signed; full Nt=64 winding; spatial pairs also measured")))
    old = ROOT/"polyakov_L32x96/runs/production100_20261011_113518"
    m = load(old/"manifest.json")
    audit = load(old/"validation/final_audit.json")
    assert audit["status"]=="PASS" and audit["validated_conf_count"]==100
    counts = Counter()
    for t in m["tasks"]:
        path, conf = old/t["output"], str(t["conf_id"])
        commit = safe_load(path/"complete.json")
        valid = commit.get("status")=="complete" and str(commit.get("conf_id"))==conf
        good = []
        for tau in m["taus"]:
            base = path/("tau%.3f"%tau).replace(".","p")
            meta = safe_load(base/"metadata.json")
            if valid and str(meta.get("conf_id"))==conf and abs(meta.get("tau_t_over_a2",-99)-tau)<1e-10 and npy_header(base/"p3.npy",(32,32,32),"complex128") and npy_header(base/"p8.npy",(32,32,32),"float64"):
                counts[tau] += 1
                good.append(tau)
        DETAILS.append(dict(snapshot_time=STAMP,ensemble="L32x96",observable="Polyakov_loop",campaign="production100",conf_id=conf,
            status="COMPLETE_FULL_GRID" if len(good)==len(m["taus"]) else "INCOMPLETE",completed_polyakov_taus=";".join(map(str,good)),data_path=str(path)))
    for tau in m["taus"]:
        row("L32x96","Polyakov_loop","production100","not_applicable_pure_gauge",tau,100,counts[tau],
            status="COMPLETE" if counts[tau]==100 else "INCOMPLETE",full_grid_validated_nconf=100,
            data_path=str(old/"data"),evidence_path=str(old/"validation/final_audit.json"),related_jobs="3833115;3833116;3853988;3853995;3854005",
            validation_level="2026-10-15 full_payload_revalidation_PASS + current_headers_metadata",
            notes="Nt=96;伴随圈信号不足，d_A未被可靠确定；测量完成不代表线质量拟合有效。")
    row("L32x96","Wilson_loop","not_in_Polyakov_production100","not_applicable_pure_gauge","not_measured_in_this_campaign",None,None,
        status="NOT_SUBMITTED_IN_AUDITED_CAMPAIGN",data_path=str(old),
        validation_level="production_source_and_manifest_scope",
        notes="该100conf批次只测单Polyakov，不含矩形Wilson loop；未将其他目录/HYP历史结果计入thin+flow生产。")
    return dict(worker_states=dict(worker_states),full_grid_complete_conf=len(full),new_conf_worker_completed=len(completed),active_conf=len(active),failed_conf=len(failed),target_conf=len(sources))


def gluon_ratios():
    base = ROOT/"ratio_bootstrap_latest_v1"
    report = load(base/"collection_validation.json")
    for record in report["records"]:
        path = Path(record["path"])
        valid = path.is_file() and path.stat().st_size==record["size_bytes"] and path.with_suffix(".npz.done.json").is_file()
        row("L32x96","gluon_disconnected_C3_C2_ratio","common406_shared_bootstrap","not_applicable_disconnected",record["tau_t_over_a2"],report["nconf"],report["nconf"] if valid else 0,
            status="COMPLETE_EXISTING_COLLECTION_AUDIT" if valid else "CURRENT_PRODUCT_MISSING_OR_CHANGED",
            pz_integer="+3;+4;+5",tsep_a="5..15",channel_flavor="unpolarized;helicity_TH;helicity_TH_plus_SH",
            data_path=str(path),evidence_path=str(base/"collection_validation.json"),
            validation_level="existing_collection_SHA_and_shared_index_validation + current_size/receipt; no rehash",
            notes=f"与quark C2/C3分开；Nboot={report['nboot']}；只是有限flow裸ratio，不代表重整化/matching。")
    row("L32x64","gluon_disconnected_C3_C2_ratio","thin_flow_ratio_not_confirmed","not_applicable_disconnected","pending_confirmation",None,None,
        status="PENDING_CONFIRMATION_NOT_COUNTED",data_path=str(ROOT/"build_L32x64_c3c2.py"),validation_level="source_audit_only",
        notes="本地build_L32x64_c3c2.py声明输入为HYP OPE，不能把其产物计成thin-link gradient-flow C3/C2；已有flow OPE另算，不等于ratio完成。")


def main():
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Run inventory on a CPU compute node")
    if (OUT/"complete.json").exists():
        raise RuntimeError("Snapshot already complete: create a versioned successor")
    quark_inventory()
    loops = loops_inventory()
    gluon_ratios()
    slurm = {}
    for key, command in {
        "queue": ["squeue","-r","-u","donghx","-h","-o","%F|%K|%j|%P|%T|%R"],
        "cpu_array": ["sacct","-X","-j","3853828","-n","-P","-o","JobID,State,ExitCode"]
    }.items():
        p = subprocess.run(command, text=True,capture_output=True)
        slurm[key] = dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    csv_write(OUT/"gradient_flow_progress.csv",ROWS)
    csv_write(OUT/"configuration_progress.csv",DETAILS)
    summary = dict(status="COMPLETE_PROGRESS_SNAPSHOT",snapshot_time=STAMP,finished_time=datetime.now().astimezone().isoformat(),
        job=os.environ["SLURM_JOB_ID"],host=os.uname().nodename,summary_csv_rows=len(ROWS),configuration_csv_rows=len(DETAILS),
        cpu_loops=loops,quark_rows=[r for r in ROWS if r["observable"].startswith("quark")],warnings=WARNINGS,
        validation_scope="read-only metadata, existing validators and NPY headers; no new numerical contractions/full payload scans",
        raw_C2_C3_is_not_already_assembled_or_fitted_ratio=True,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),slurm=slurm)
    (OUT/"complete.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in ("quark_rows","slurm")},indent=2,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
