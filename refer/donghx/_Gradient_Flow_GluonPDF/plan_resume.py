"""Freeze only the missing original C24 conf-flow tasks for a Slurm array."""
from collections import Counter
import csv
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent / 'quark_l24x72_B_conf121to200_matched_radius_20261116'
sys.path.insert(0, str(CAMPAIGN))
from common import json, load_manifest, sha256


def main():
    manifest = load_manifest()
    digest = sha256(CAMPAIGN / 'manifest.json')
    for name in ('cpu_gate', 'epsilon_gate', 'smoke_gate'):
        gate = json.loads((CAMPAIGN / 'status' / f'{name}.json').read_text())
        assert gate['status'] == 'PASS' and gate['manifest_sha256'] == digest, name
    pending = []
    for row in manifest['tasks']:
        task = row['task']
        marker = CAMPAIGN / 'status' / f'task{task}.json'
        if not marker.exists():
            pending.append(row)
            continue
        info = json.loads(marker.read_text())
        assert info['status'] == 'PASS' and info['task'] == task and info['case_count'] == 252, marker
        assert info['manifest_sha256'] == digest and str(info['conf']) == str(row['conf']), marker
    assert len(manifest['tasks']) == 560 and len(pending) == 40
    assert Counter(row['task'] % 24 for row in pending) == {22: 20, 23: 20}
    assert len({row['conf'] for row in pending}) == 23
    with (HERE / 'missing_tasks.csv').open('x', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(('array_index', 'task', 'conf', 'flow_time_a2', 'old_worker', 'manifest_sha256'))
        for index, row in enumerate(pending):
            writer.writerow((index, row['task'], row['conf'], row['flow_time_a2'], row['task'] % 24, digest))
    print('RESUME_PLAN_PASS', len(pending), 'missing tasks across 23 configurations;', len(manifest['tasks'])-len(pending), 'valid receipts')


if __name__ == '__main__':
    main()
