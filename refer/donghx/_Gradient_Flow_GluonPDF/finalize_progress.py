#!/usr/bin/env python3
"""Add ratio products and verify historical extra-flow tags without changing production."""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from datetime import datetime
import re
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("progress_inventory", HERE/"audit_progress.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
ROOT = audit.ROOT


def main():
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("CPU allocation required")
    completion = json.loads((HERE/"complete.json").read_text())
    with (HERE/"gradient_flow_progress.csv").open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    # Replace supplementary rows on reruns instead of duplicating them.
    replaced_campaigns={"common406_allops_bootstrap1000", "legacy_A_20conf_100boot_analysis_subset"}
    rows=[r for r in rows if r["campaign"] not in replaced_campaigns]
    audit.ROWS.clear()
    # Historical submission scripts specify these widths; do not guess them.
    canonical = Path("/public/group/lqcd/donghx/Gradient_Flow/Result/beta6.41_mu-0.2295_ms-0.2050_L32x96/thin_link")
    roots=[canonical,ROOT/"quark_flow_products",ROOT/"quark_spin2_production_20260922/production_tf0p5_pilot_z",
        ROOT/"quark_spin2_expansion20_20260922/results_tf0p5_pilot_z",ROOT/"quark_spin2_fill_flows_20260923/results_remaining",
        ROOT/"quark_spin2_expansion30_20260925/results_all6"]
    for tau,window in ((.6,5),(1.1,8),(1.2,9),(1.5,10)):
        records=[audit.raw_conf(roots,c,tau,window,96,list(range(6,11))) for c in audit.CONF96[:10]]
        done=sum(r["complete"] for r in records)
        r=next(r for r in rows if r["campaign"]=="historical_extra_flow_check" and abs(float(r["tau_tf_over_a2"])-tau)<1e-10)
        r.update(complete_nconf=done,completion_percent=10*done,temporal_windows=window,
            status="COMPLETE" if done==10 else "CURRENT_PRODUCTS_NOT_CONFIRMED" if done==0 else "PARTIAL",
            notes="按历史实际提交width复查当前产物。目录缺失不证明从未计算；未将缓存或旧提交日志计为完成。",
            evidence_path="submit_quark_flow_0811_all10.sh; /public/home/donghx/Gradient_Flow/Blending_L32x96/submit_quark_flow_101215_all10.sh")
    # Current 1000-bootstrap all-operator disconnected product, distinct from older 1500 replicas.
    base=ROOT/"ratio_bootstrap_all_operators_v1/results_v1"
    report=audit.load(base/"collection_validation.json")
    assert report["status"]=="complete_and_validated"
    for p in report["products"]:
        path=Path(p["product"])
        valid=path.is_file() and path.stat().st_size>0 and zipfile.is_zipfile(path)
        audit.row("L32x96","gluon_disconnected_C3_C2_ratio","common406_allops_bootstrap1000","not_applicable_disconnected",p["flow_tau"],406,406 if valid else 0,
            status="COMPLETE_EXISTING_COLLECTION_AUDIT" if valid else "CURRENT_PRODUCT_MISSING",
            pz_integer="+3;+4;+5",tsep_a="5..15",channel_flavor=";".join(p["operator_labels"]),
            data_path=str(path),evidence_path=str(base/"collection_validation.json"),
            validation_level="existing_collection_SHA_shared_index_audit+current_NPZ_inventory; no_new_full_payload_scan",
            notes="Nboot=1000,seed=1115;与旧1500bootstrap并存，不重复增加组态数。")
    # Existing connected-quark ratios on the 20-conf analysis subset.
    base=ROOT/"quark_bare_ratio_20_ud_20260927"
    manifest=audit.load(base/"input_manifest.json")
    expected=[("ratio_original.npy",(2,3,21,5,11)),("ratio_boot.npy",(100,2,3,21,5,11))]
    filename_pattern=re.compile(
        r"^ratio_tau(?P<tau>[0-9]+p[0-9]+)_"
        r"(?P<channel>unpolarized|helicity|transversity)_"
        r"(?P<flavor>u_minus_d|u|d)_Nboot100_seed1115\.npz$")
    ratio_files=[]
    for path in sorted((base/"ratios").glob("ratio_tau*.npz")):
        match=filename_pattern.fullmatch(path.name)
        if match:
            ratio_files.append((float(match["tau"].replace("p",".")),
                                (match["channel"],match["flavor"]),path))
    expected_channels_flavors={(c,f) for c in ("unpolarized","helicity","transversity")
                               for f in ("u","d","u_minus_d")}
    for tau in manifest["taus"]:
        # Parse the numeric tag: production uses 0p10, not 0p1.
        files=[(channel_flavor,path) for value,channel_flavor,path in ratio_files
               if abs(value-float(tau))<1e-10]
        valid=set()
        errors=[]
        for channel_flavor,path in files:
            try:
                with zipfile.ZipFile(path) as z:
                    for name,shape in expected:
                        with z.open(name) as stream:
                            version=np.lib.format.read_magic(stream)
                            got_shape,_,_=np.lib.format._read_array_header(stream,version)
                        if got_shape!=shape:
                            raise ValueError((path,name,got_shape))
                valid.add(channel_flavor)
            except (OSError,ValueError,KeyError,zipfile.BadZipFile,EOFError) as exc:
                errors.append(f"{path.name}: {exc}")
        complete=valid==expected_channels_flavors and not manifest.get("missing")
        audit.row("L32x96","quark_C3_C2_ratio_bootstrap","legacy_A_20conf_100boot_analysis_subset","A_legacy_unified_table",tau,20,20 if complete else 0,
            status="COMPLETE_EXISTING_ANALYSIS_PRODUCTS" if complete else "INCOMPLETE_ANALYSIS_PRODUCTS",
            pz_integer="+3;+4;+5",tsep_a="6;7;8;9;10",target_cases=9,complete_cases=len(valid),
            channel_flavor="unpolarized;helicity;transversity x u;d;u-d",
            data_path=str(base/"ratios"),evidence_path=str(base/"input_manifest.json"),
            validation_level="input_manifest_missing_empty + current_ratio_NPZ_headers; no_new_full_payload_scan",
            notes="Nboot=100,seed=1115;这是旧A方案20conf子集，不是最新30conf或B方案的重采样。transversity外部相对符号仍待确认。",
            validation_errors="; ".join(errors))
    rows.extend(audit.ROWS)
    def normalized_tau(value):
        try:
            return float(value)
        except (TypeError,ValueError):
            return str(value)
    keys=[(r["ensemble"],r["observable"],r["campaign"],r["weight_scheme"],
           normalized_tau(r["tau_tf_over_a2"])) for r in rows]
    assert len(keys)==len(set(keys)), "Duplicate progress entries"
    for r in rows:
        if r["complete_nconf"]!="" and r["target_nconf"]!="":
            assert 0<=int(r["complete_nconf"])<=int(r["target_nconf"]), r
    audit.csv_write(HERE/"gradient_flow_progress.csv",rows)
    completion.update(finalized_time=datetime.now().astimezone().isoformat(),finalizer_job=os.environ["SLURM_JOB_ID"],summary_csv_rows=len(rows),
        historical_extra_flow_widths={"0.6":5,"1.1":8,"1.2":9,"1.5":10},
        finalizer_source="finalize_progress.py; no production source edited",
        quark_rows=[r for r in rows if r["observable"].startswith("quark_")],
        csv_validation="PASS_unique_ensemble_observable_campaign_scheme_tau_and_completion_bounds",
        supplementary_audit_source_sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
            for name in ("audit_progress.py","fast_snapshot.py","finalize_progress.py")})
    (HERE/"complete.json").write_text(json.dumps(completion,indent=2,ensure_ascii=False)+"\n")
    print(f"FINALIZED rows={len(rows)}",flush=True)
    for r in rows:
        if r["observable"]=="quark_C2_C3_raw" or r["observable"]=="quark_C3_C2_ratio_bootstrap":
            print(r["ensemble"],r["campaign"],r["tau_tf_over_a2"],r["complete_nconf"],r["target_nconf"],r["status"],flush=True)


if __name__=="__main__":
    main()
