"""Contract for the generated TMD production report."""
from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class TmdFullReportContract(unittest.TestCase):
    def test_report_generator_emits_evidence_and_no_placeholders(self):
        from pyqcd.testing.tmd.build_full_report import build_report

        with tempfile.TemporaryDirectory() as root:
            run_dir = Path(root) / 'run'
            analysis = run_dir / 'analysis'
            analysis.mkdir(parents=True)
            workers = run_dir / 'workers'
            workers.mkdir()
            for tag, backend in (('a', 'cupy'), ('b', 'torch')):
                (workers / f'worker_{tag}.json').write_text(
                    json.dumps({
                        'status': 'complete',
                        'backend': backend,
                        'device': 0,
                        'conf_ids': [6250, 6450],
                        'started_utc': '2026-11-01T00:00:00+00:00',
                        'finished_utc': '2026-11-01T06:00:00+00:00',
                    }),
                    encoding='utf-8',
                )
            (workers / 'system_resources.jsonl').write_text(
                json.dumps({
                    'cpu_percent': 50.0,
                    'memory_percent': 40.0,
                    'disk_percent': 75.0,
                    'gpu': [{
                        'index': 0,
                        'name': 'GPU',
                        'memory_percent': 30.0,
                        'utilization_percent': 60.0,
                    }],
                }) + '\n',
                encoding='utf-8',
            )
            (analysis / 'analysis_summary.json').write_text(
                json.dumps({
                    'status': 'diagnostic_only',
                    'physical_acceptance': False,
                    'conf_ids': [6250, 6450],
                    'tau_values': [1, 2, 3],
                    'n_ratio_1d': 12,
                    'n_ratio_2d': 6,
                    'n_pdf': 6,
                }),
                encoding='utf-8',
            )
            tex, summary = build_report(
                run_dir, Path(root) / 'report.tex',
                Path(root) / 'summary.md')
            source = tex.read_text(encoding='utf-8')
            text = summary.read_text(encoding='utf-8')
            build = Path(root) / 'build'
            build.mkdir()
            for _ in range(2):
                completed = subprocess.run(
                    [
                        'xelatex', '-interaction=nonstopmode',
                        '-halt-on-error', '-file-line-error',
                        '-output-directory', str(build), str(tex),
                    ],
                    capture_output=True, text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stdout)
                for diagnostic in (
                        'Overfull', 'Float too large', 'Missing character'):
                    self.assertNotIn(diagnostic, completed.stdout)
        self.assertIn(r'\documentclass[UTF8,aspectratio=169,11pt]{ctexbeamer}',
                      source)
        self.assertIn('diagnostic\\_only', source)
        self.assertIn('physical\\_acceptance=false', source)
        self.assertIn('soft/rapidity', source)
        self.assertIn(r'\dimexpr\paperwidth-1.1cm\relax', source)
        self.assertIn('准 PDF/TMD-PDF', text)
        for forbidden in ('TODO', 'TBD', '<待填>'):
            self.assertNotIn(forbidden, source)
            self.assertNotIn(forbidden, text)


if __name__ == '__main__':
    unittest.main()
