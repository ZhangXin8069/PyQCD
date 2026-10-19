#!/usr/bin/env python3
"""Finish the progress snapshot from successful header blocks and immutable receipts.

The initial job completed all sixteen raw-C2/C3 header blocks before being
stopped during its unnecessarily exhaustive loop header rescan. This successor
uses those successful counts and full-grid per-conf validation receipts, not
an existence-only or synthetic numerical validation.
"""
from collections import Counter
import importlib.util
import os
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("initial_inventory",HERE/"audit_progress.py")
a=importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def quark_from_header_log():
    log=HERE/"logs/audit_3854552.out"
    matches=re.findall(r"(L32x\d+) legacy tau=([\d.]+) (\d+)/(\d+)",log.read_text())
    counts={(ensemble,float(tau)):(int(n),int(total)) for ensemble,tau,n,total in matches}
    assert len(counts)==16, "All original header blocks must have finished before reuse"
    for (ensemble,tau),(n,total) in counts.items():
        if ensemble=="L32x64":
            path=a.ROOT/"quark_l32x64_flow_20261007/results"
            tseps="5;6;7;8;9;10"
        else:
            path=Path("/public/group/lqcd/donghx/Gradient_Flow/Result/beta6.41_mu-0.2295_ms-0.2050_L32x96/thin_link")
            tseps="6;7;8;9;10"
        a.row(ensemble,"quark_C2_C3_raw","legacy_A_full_source_current","A_legacy_unified_table",tau,total,n,
            status="COMPLETE" if n==total else "INCOMPLETE",missing_or_partial_nconf=total-n,
            temporal_windows=dict(a.TAUS)[tau],pz_integer="+3;+4;+5",tsep_a=tseps,
            source_current_coverage="all sources/currents; declared common tseps",
            channel_flavor="open_current_spin_4x4;u;d;u-d_by_analysis",data_path=str(path),
            evidence_path=str(log),validation_level="all_conf_NPZ_array_headers_shape_dtype_size+timing_identity_checked_on_CPU_3854552",
            notes="原A方案；继承初次CPU审计已成功结束的逐flow文件头检查块，非仅按目录计数；不代表新统计量ratio/拟合/matching完成。")
    # The finalizer rechecks the actual historical tag widths before publishing.
    for tau,window in ((.6,5),(1.1,8),(1.2,9),(1.5,10)):
        a.row("L32x96","quark_C2_C3_raw","historical_extra_flow_check","A_legacy_unified_table",tau,10,0,
            status="PENDING_FINALIZER",temporal_windows=window,data_path="known L32x96 raw output roots",
            validation_level="historical_extra_flow_tag_recheck_in_finalizer")
    b=a.ROOT/"quark_l32x64_P4_B_20conf_20261015"
    complete=a.load(b/"summary/complete.json")
    assert complete["status"]=="COMPLETE_20CONF_FINITE_WINDOW_DIAGNOSTIC" and complete["case_count"]==1440
    for tau in (.2,.5,.8):
        report=a.load(b/f"summary/tau{tau:g}/report.json")
        selected=[c for c in complete["cases"] if abs(c["flow_time_a2"]-tau)<1e-12]
        n=sum(v==24 for v in Counter(c["conf"] for c in selected).values())
        a.row("L32x64","quark_C2_C3_and_paired_window_ratio","B_20conf_selected_current_midpoint","B_same_time_second_cross_time_product",tau,20,n,
            status="COMPLETE_FINITE_WINDOW_DIAGNOSTIC",target_cases=480,complete_cases=len(selected),
            temporal_windows=";".join(map(str,report["windows"])),pz_integer="-6;-5;-4;-3;+3;+4;+5;+6",tsep_a="7;8;9;10;11;12",
            source_current_coverage="currents 0;16;32;63;midpoint dt=floor(tsep/2)",channel_flavor="open_current_spin_4x4;u;d;paired_u-d",
            data_path=str(b/"C3_schemeB_Pz3456_tsep7to12"),evidence_path=str(b/"summary/complete.json"),
            validation_level="existing_final_CPU_full_payload_algebra_validation_1440_cases",
            related_jobs="3853935;3853936;3853937",notes="与A分开；有限嵌套时间窗，不是全current/all-insertion或精确全时间积分证明。")
    extra=a.ROOT/"quark_l32x64_expand40_20261013"
    if not (extra/"manifest.json").exists() and not (extra/"submission.json").exists():
        a.row("L32x64","quark_C2_C3_raw","extension40_prepared_only","A_legacy_unified_table","all_8_planned_flows",10,0,
            status="PREPARED_NO_SUBMISSION_RECEIPT",data_path=str(extra),validation_level="no_frozen_manifest_or_submission_receipt",
            notes="已有代码不代表已提交/完成；已确认原生产为30conf。")


def fast_loops():
    root=a.ROOT/"line_mass_L32x64_allconf_B_20261015/runs/allconf_B_2026-10-15T205426.224554_0800"
    m=a.load(root/"manifest.json")
    completed,active,failed=set(),set(),set()
    states=Counter()
    for file in (root/"submission_workers_v1").glob("worker[0-9]*.json"):
        r=a.safe_load(file)
        states[r.get("state","UNKNOWN")]+=1
        completed.update(x["conf_id"] for x in r.get("completed",[]))
        failed.update(x["conf_id"] for x in r.get("failed",[]))
        done={x["task"] for x in r.get("completed",[])+r.get("failed",[])}
        remaining=[t for t in r.get("tasks",[]) if t not in done]
        if r.get("state")=="RUNNING" and remaining:
            active.add(m["tasks"][remaining[0]]["conf_id"])
    taus=m["flow"]["taus"]
    counts={t:{"Polyakov_loop":0,"Wilson_loop":0} for t in taus}
    # A single directory inventory avoids probing 23 non-existent paths per not-started conf.
    data=root/"data/epsilon0.010"
    known={p.name for p in data.iterdir() if p.is_dir()}
    full=set()
    for i,s in enumerate(m["data_sources"]):
        p,conf=Path(s["path"]),str(s["conf_id"])
        reused=s["kind"]!="new_allconf_measurement"
        if not reused and p.name not in known:
            goodp,goodw=[],[]
        else:
            commit=a.safe_load(p/"complete.json")
            all_done=commit.get("status")=="complete" and str(commit.get("conf_id"))==conf and (conf in completed or reused)
            all_done=all_done and len(commit.get("taus",[]))==len(taus) and all(abs(x-y)<1e-10 for x,y in zip(commit.get("taus",[]),taus))
            if all_done:
                full.add(conf)
                goodp,goodw=taus,taus
            else:
                goodp,goodw=[],[]
                for directory in p.glob("tau*p*"):
                    match=re.fullmatch(r"tau(\d+)p(\d+)",directory.name)
                    if not match:
                        continue
                    value=float(match[1]+"."+match[2])
                    tau=next((t for t in taus if abs(t-value)<1e-10),None)
                    if tau is None:
                        continue
                    meta=a.safe_load(directory/"metadata.json")
                    identity=meta.get("status")=="measured_unrenormalized_polyakov_loop" and str(meta.get("conf_id"))==conf and abs(meta.get("tau_t_over_a2",-99)-tau)<1e-10
                    if identity and a.npy_header(directory/"p3.npy",(32,32,32),"complex128") and a.npy_header(directory/"p8.npy",(32,32,32),"float64"):
                        goodp.append(tau)
                    closed=a.safe_load(directory/"closed_loops.json")
                    if identity and closed.get("status")=="measured_bare_single_configuration" and a.npy_header(directory/"wilson_fundamental.npy",(3,13,17),"complex128") and a.npy_header(directory/"wilson_adjoint.npy",(3,13,17),"float64"):
                        goodw.append(tau)
        for tau in goodp:
            counts[tau]["Polyakov_loop"]+=1
        for tau in goodw:
            counts[tau]["Wilson_loop"]+=1
        a.DETAILS.append(dict(snapshot_time=a.STAMP,ensemble="L32x64",observable="Wilson_and_Polyakov_loop",campaign="all1700",conf_id=conf,
            status="COMPLETE_FULL_GRID" if conf in full else "RUNNING" if conf in active else "FAILED" if conf in failed else "PARTIAL_CHECKPOINTS" if goodp or goodw else "NOT_STARTED",
            completed_polyakov_taus=";".join(map(str,goodp)),completed_wilson_taus=";".join(map(str,goodw)),data_path=str(p)))
        if (i+1)%200==0:
            print(f"fast loop receipt checked {i+1}/{len(m['data_sources'])}",flush=True)
    for tau,values in counts.items():
        for obs,n in values.items():
            a.row("L32x64",obs,"all1700","not_applicable_pure_gauge",tau,len(m["data_sources"]),n,
                status="IN_PROGRESS" if n<len(m["data_sources"]) else "COMPLETE",full_grid_validated_nconf=len(full),running_nconf=len(active),failed_nconf=len(failed),
                data_path=str(root/"data"),evidence_path=str(root/"submission_workers_v1"),related_jobs="3853828;3853829",
                validation_level="complete_grid_commit+existing_worker_per_conf_validation; partial_checkpoint_JSON+NPY_headers",
                notes="全flow完成按验收回执计；单flow数量另含运行中已提交检查点；纯规范量无A/B权重；统计汇总3853829等待全部测量。")
    p96=a.ROOT/"polyakov_L32x96/runs/production100_20261011_113518"
    m96=a.load(p96/"manifest.json")
    validation=a.load(a.ROOT/"polyakov_L32x96/analysis/line_mass_test_20261015/statistics/revalidation.json")
    assert validation["status"]=="PASS" and validation["nconf"]==100 and validation["validated_pairs"]==2100
    ids=[]
    for task in m96["tasks"]:
        path=p96/task["output"]
        commit=a.safe_load(path/"complete.json")
        if commit.get("status")=="complete" and str(commit.get("conf_id"))==str(task["conf_id"]):
            ids.append(str(task["conf_id"]))
        a.DETAILS.append(dict(snapshot_time=a.STAMP,ensemble="L32x96",observable="Polyakov_loop",campaign="production100",conf_id=task["conf_id"],
            status="COMPLETE_FULL_GRID" if str(task["conf_id"]) in ids else "INCOMPLETE",data_path=str(path)))
    for tau in m96["taus"]:
        a.row("L32x96","Polyakov_loop","production100","not_applicable_pure_gauge",tau,100,len(ids),status="COMPLETE" if len(ids)==100 else "INCOMPLETE",
            full_grid_validated_nconf=len(ids),data_path=str(p96/"data"),evidence_path=str(a.ROOT/"polyakov_L32x96/analysis/line_mass_test_20261015/statistics/revalidation.json"),
            validation_level="2026-10-15_full_payload_revalidation_PASS+current_complete_commits",notes="Nt=96,2100检查点含tau0；信号不足，d_A尚未可靠确定。")
    a.row("L32x96","Wilson_loop","not_in_Polyakov_production100","not_applicable_pure_gauge","not_measured_in_this_campaign",None,None,
        status="NOT_SUBMITTED_IN_AUDITED_CAMPAIGN",data_path=str(p96),validation_level="production_manifest_source_scope",
        notes="该100conf批次只测单Polyakov，不含矩形Wilson loop；未混入HYP历史结果。")
    return dict(worker_states=dict(states),full_grid_complete_conf=len(full),new_conf_worker_completed=len(completed),active_conf=len(active),failed_conf=len(failed),target_conf=len(m["data_sources"]))


if __name__=="__main__":
    a.quark_inventory=quark_from_header_log
    a.loops_inventory=fast_loops
    a.main()
    # Same CPU allocation finishes the two CSV products; no duplicate child submission.
    spec2=importlib.util.spec_from_file_location("finalize_inventory",HERE/"finalize_progress.py")
    final=importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(final)
    final.main()
