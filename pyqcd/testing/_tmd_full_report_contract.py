"""Contract for the generated TMD production report."""
from __future__ import annotations

import base64
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


PNG_1X1 = base64.b64decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=')
REPORT_IMAGES = (
    'analysis/flow/flow_history.png',
    'analysis/spectrum/effective_mass_nopol_P000.png',
    'analysis/spectrum/effective_mass_nopol_P200.png',
    'analysis/spectrum/dispersion_diagnostic.png',
    'analysis/spectrum/ratio_3pt_proton_P0.png',
    'analysis/spectrum/direction_momentum_comparison.png',
    'analysis/ratio_1d/ratio_v2_tau1_x_P002_ratio.png',
    'analysis/ratio_1d/ratio_v2_tau1_y_P020_ratio.png',
    'analysis/ratio_1d/ratio_v2_tau1_z_P200_ratio.png',
    'analysis/ratio_2d/staircase_sensitivity.png',
    'analysis/plateaus_2d/c0_tau1_order01_heatmap.png',
    'analysis/plateaus_2d/c0_tau3_order01_heatmap.png',
    'analysis/plateaus/c0_tau1_z_P200_windows.png',
    'analysis/plateaus/c0_tau2_z_P200_windows.png',
    'analysis/plateaus/c0_tau3_z_P200_windows.png',
    'analysis/plateaus/sumratio_quality.png',
    'analysis/flow_windows/flow_window_coverage.png',
    'analysis/flow_windows/flow_time_paired_difference.png',
    'analysis/pdf/quasi_tau1_P200.png',
    'analysis/pdf/quasi_tau3_P200.png',
    'analysis/pdf/quasi_tau3_P400.png',
    'analysis/pdf/hR_tau1_P200.png',
    'analysis/pdf/hR_tau3_P200.png',
    'analysis/pdf/hR_tau3_P400.png',
    'analysis/pdf/cs_kernel_tau1.png',
    'analysis/pdf/cs_kernel_tau3.png',
    'analysis/error_budget.png',
)


class TmdFullReportContract(unittest.TestCase):
    def test_markdown_summary_includes_spectrum_and_plateau_tables(self):
        from pyqcd.testing.tmd.build_full_report import _summary_markdown

        text = _summary_markdown(
            {
                'conf_ids': [6250, 6450], 'tau_values': [1, 2, 3],
                'n_ratio_1d': 1, 'n_ratio_2d': 1, 'n_pdf': 1,
            },
            workers=[{'status': 'complete'}],
            resources={
                'cpu_max': 50, 'memory_max': 40, 'disk_max': 75,
                'system_gpu_util_max': 100,
            },
            manifest={'file_count': 1, 'total_bytes': 1},
            status='diagnostic_only',
            spectrum={
                'nopol_P200': {
                    'energy_GeV': 1.2, 'energy_error_GeV': 0.1,
                    'plateau_window': [5, 11], 'energy_status': 'diagnostic',
                },
            },
            plateau={
                'tau1_z_P200': {
                    'primary_identifiable_fraction': 0.75,
                    'sumratio_quality_fraction': 0.5,
                    'primary_window': [7, 12],
                },
            },
        )
        self.assertIn('## 谱学平台', text)
        self.assertIn('nopol_P200', text)
        self.assertIn('1.2', text)
        self.assertIn('## 平台质量', text)
        self.assertIn('tau1_z_P200', text)
        self.assertIn('0.75', text)
        self.assertIn('系统级 GPU 遥测瞬时利用率峰值 100%', text.replace('\\%', '%'))
        self.assertIn('不能认证全局瞬时占用始终不超过 80%', text)

    def test_figure_missing_is_fail_closed(self):
        from pyqcd.testing.tmd.build_full_report import _figure

        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):
                _figure(Path(root) / 'missing.png', 'missing')

    def test_report_requires_all_twelve_configurations(self):
        from pyqcd.testing.tmd.build_full_report import _validate_report_summary

        conf_ids = [
            6250, 6450, 6650, 6850, 7050, 7250,
            7450, 7650, 7850, 8050, 8250, 8450,
        ]
        summary = {
            'status': 'diagnostic_only',
            'physical_acceptance': False,
            'conf_ids': conf_ids,
            'tau_values': [1, 2, 3],
        }
        _validate_report_summary(summary)
        summary['conf_ids'] = summary['conf_ids'][:-1]
        with self.assertRaisesRegex(ValueError, '12'):
            _validate_report_summary(summary)

    def test_report_accepts_explicit_extension_conf_set(self):
        from pyqcd.testing.tmd.build_full_report import _validate_report_summary

        conf_ids = [
            4150, 6250, 6450, 6650, 6850, 7050,
            7250, 7450, 7650, 7850, 8050, 8250, 8450,
        ]
        summary = {
            'status': 'diagnostic_only',
            'physical_acceptance': False,
            'conf_ids': conf_ids,
            'tau_values': [1, 2, 3],
        }
        self.assertEqual(
            _validate_report_summary(summary, conf_ids)[0], tuple(conf_ids))
        summary['conf_ids'] = summary['conf_ids'][:-1]
        with self.assertRaisesRegex(ValueError, '13'):
            _validate_report_summary(summary, conf_ids)

    def test_report_generator_emits_evidence_and_no_placeholders(self):
        from pyqcd.testing.tmd.build_full_report import build_report

        with tempfile.TemporaryDirectory() as root:
            run_dir = Path(root) / 'run'
            analysis = run_dir / 'analysis'
            analysis.mkdir(parents=True)
            workers = run_dir / 'workers'
            workers.mkdir()
            for tag, backend in (('a', 'cupy'),):
                (workers / f'worker_{tag}.json').write_text(
                    json.dumps({
                        'status': 'complete',
                        'backend': backend,
                        'device': 0,
                        'conf_ids': [
                            6250, 6450, 6650, 6850, 7050, 7250,
                            7450, 7650, 7850, 8050, 8250, 8450,
                        ],
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
                    'conf_ids': [
                        6250, 6450, 6650, 6850, 7050, 7250,
                        7450, 7650, 7850, 8050, 8250, 8450,
                    ],
                    'tau_values': [1, 2, 3],
                    'n_ratio_1d': 12,
                    'n_ratio_2d': 6,
                    'n_pdf': 6,
                }),
                encoding='utf-8',
            )
            for relative in (
                    'spectrum/spectrum_summary.json',
                    'plateaus/plateau_summary.json',
                    'flow_windows/flow_window_summary.json',
                    'error_budget.json',
                    'ratio_2d/staircase_summary.json',
                    'pdf/pdf_summary.json'):
                path = analysis / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{}\n', encoding='utf-8')
            (analysis / 'artifact_manifest.json').write_text(
                json.dumps({
                    'file_count': len(REPORT_IMAGES),
                    'total_bytes': len(REPORT_IMAGES) * len(PNG_1X1),
                    'entries': [],
                }),
                encoding='utf-8',
            )
            for relative in REPORT_IMAGES:
                image = run_dir / relative
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(PNG_1X1)
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
