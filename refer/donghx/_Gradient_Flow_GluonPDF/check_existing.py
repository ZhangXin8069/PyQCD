"""CPU-only audit of the running 50-conf predecessor; isolate known bad artifacts.

No production source is changed. Only case directories proven invalid in failed
worker 58 (tasks 116/117) may be moved to a recoverable quarantine. All active
worker output is left untouched. Other failures close the expansion gate.
"""
import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'quark_l32x64_B_expand50_P456_20261026'
sys.path.insert(0, str(PREVIOUS))
import common as old
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quarantine-known-bad', action='store_true')
    args = parser.parse_args()
    old.require_allocation()
    manifest = old.load_manifest()
    assert manifest['confs'] == list(map(str, range(10800, 12000, 40)))
    assert manifest['momenta'] == [4, -4, 5, -5, 6, -6]
    assert manifest['flows'] == [.1, .2, .3, .4, .5, .6, .7, .8]
    assert manifest['weight_scheme'] == old.SCHEME
    completed, unexpected, repair, c2_reference = [], [], [], {}
    cases_read = 0
    for row in manifest['tasks']:
        path = PREVIOUS/'status'/f'task{row["task"]}.json'
        if not path.exists():
            continue  # active/pending output is intentionally not sampled
        marker = json.loads(path.read_text())
        assert marker['manifest_sha256'] == old.manifest_hash()
        assert marker['status'] == 'PASS' and marker['case_count'] == 252
        assert marker['momenta'] == list(old.MOMENTA)
        for tj, sep, dt in [(0, 7, 1), (16, 9, 4), (32, 10, 5), (63, 12, 11)]:
            try:
                _, _, c2, _, _ = old.read_case(row, tj, sep, dt, verify_hash=True)
                key = row['conf'], tj, sep, dt
                if key in c2_reference:
                    np.testing.assert_allclose(c2, c2_reference[key], rtol=2e-10, atol=0)
                else:
                    c2_reference[key] = c2
                cases_read += 1
            except Exception as exc:
                unexpected.append(dict(task=row['task'], case=[tj, sep, dt],
                                       error=repr(exc)))
        completed.append(row['task'])
    # The only previously observed failed worker must not be live while repaired.
    running = subprocess.check_output(
        ['squeue', '-h', '-j', '3871769_58', '-o', '%T'], text=True).strip()
    if running:
        raise RuntimeError(('repair worker is live; refuse to touch output', running))
    for task in (116, 117):
        row = manifest['tasks'][task]
        old.check_inputs(row)
        for tj in old.CURRENTS:
            for sep in old.TSEPS:
                for dt in range(sep+1):
                    directory = old.output_dir(row, tj, sep, dt)
                    if not directory.exists():
                        continue
                    if not (directory/'receipt.json').exists():
                        # No commit marker: preserve partial output and let the
                        # existing atomic writer replace it on the resume path.
                        continue
                    try:
                        old.read_case(row, tj, sep, dt, verify_hash=True)
                        cases_read += 1
                    except Exception as exc:
                        repair.append(dict(task=task, conf=row['conf'],
                            flow_time_a2=row['flow_time_a2'], case=[tj, sep, dt],
                            directory=str(directory), error=repr(exc)))
    report = dict(created=datetime.now().astimezone().isoformat(),
        job=os.environ['SLURM_JOB_ID'], previous_root=str(PREVIOUS),
        previous_manifest_sha256=old.manifest_hash(),
        completed_conf_flow_markers=len(completed),
        complete_all8_configuration_count=sum(
            all(8*k+j in completed for j in range(8)) for k in range(30)),
        representative_cases_verified=cases_read, representative_scope=
            'four committed cases per completed conf-flow; all committed cases in failed pair',
        unexpected_sample_failures=unexpected, known_failed_pair_bad_cases=repair,
        no_full_predecessor_audit_claimed=True,
        repair_tasks=[116, 117], quarantine_applied=False)
    old.atomic_json(HERE/'status/existing_audit.json', report)
    if unexpected or len(repair) > 4:
        raise RuntimeError('broader integrity issue; expansion blocked; inspect existing_audit.json')
    if args.quarantine_known_bad:
        for item in repair:
            source = Path(item['directory'])
            relative = source.relative_to(PREVIOUS/'C3_C2_schemeB')
            target = HERE/'quarantine_previous'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise RuntimeError(('quarantine destination already exists', target))
            source.rename(target)
            item['quarantine'] = str(target)
        report['quarantine_applied'] = True
    report['status'] = 'PASS_WITH_ISOLATED_REPAIR' if repair else 'PASS'
    old.atomic_json(HERE/'status/existing_audit.json', report)
    print('PREVIOUS_CAMPAIGN_AUDIT', json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
