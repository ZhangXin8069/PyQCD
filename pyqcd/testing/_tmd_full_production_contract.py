"""Contracts for the resumable full TMD production driver."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


class TmdFullProductionContract(unittest.TestCase):
    def test_gpu_duty_cycle_uses_eighty_percent_target(self):
        from pyqcd.testing.tmd import run_full_production as production

        with patch.object(production.time, 'sleep') as sleep:
            seconds = production._throttle_backend_work(
                10.0,
                {'backend': 'torch', 'gpu_duty_cycle': 0.8},
            )
        self.assertAlmostEqual(seconds, 2.5)
        sleep.assert_called_once_with(2.5)

    def test_resource_guard_uses_hard_eighty_percent_limit(self):
        from pyqcd.testing.tmd.run_full_production import (
            ResourceLimitError,
            validate_resource_snapshot,
        )

        accepted = {
            'cpu_percent': 79.9,
            'memory_percent': 79.9,
            'disk_percent': 79.9,
            'gpu_memory_percent': 79.9,
        }
        self.assertEqual(
            validate_resource_snapshot(accepted, max_fraction=0.8),
            accepted,
        )
        for key in accepted:
            rejected = dict(accepted)
            rejected[key] = 80.1
            with self.subTest(key=key), self.assertRaisesRegex(
                    ResourceLimitError, key):
                validate_resource_snapshot(rejected, max_fraction=0.8)

    def test_disk_limit_can_be_relaxed_without_relaxing_other_resources(self):
        from pyqcd.testing.tmd.run_full_production import (
            ResourceLimitError,
            validate_resource_snapshot,
        )

        accepted = {
            'cpu_percent': 79.9,
            'memory_percent': 79.9,
            'disk_percent': 94.9,
            'gpu_memory_percent': 79.9,
        }
        self.assertEqual(
            validate_resource_snapshot(
                accepted,
                max_fraction=0.8,
                max_disk_fraction=0.95,
            ),
            accepted,
        )

        high_disk = dict(accepted, disk_percent=95.1)
        with self.assertRaisesRegex(ResourceLimitError, 'disk_percent'):
            validate_resource_snapshot(
                high_disk,
                max_fraction=0.8,
                max_disk_fraction=0.95,
            )

        high_cpu = dict(accepted, cpu_percent=80.1)
        with self.assertRaisesRegex(ResourceLimitError, 'cpu_percent'):
            validate_resource_snapshot(
                high_cpu,
                max_fraction=0.8,
                max_disk_fraction=0.95,
            )

        with self.assertRaisesRegex(ValueError, 'max_disk_fraction'):
            validate_resource_snapshot(
                accepted,
                max_fraction=0.8,
                max_disk_fraction=0.951,
            )

    def test_disk_reserve_uses_the_independent_limit(self):
        from types import SimpleNamespace
        from pyqcd.testing.tmd import run_full_production as production
        from pyqcd.testing.tmd.run_full_production import ResourceLimitError

        usage = SimpleNamespace(total=100, used=90, free=10)
        project_root = Path(__file__).resolve().parents[2]
        with patch.object(production.shutil, 'disk_usage', return_value=usage):
            production._assert_disk_reserve(project_root, 0.95, 5)
            with self.assertRaisesRegex(
                    ResourceLimitError, '磁盘预留失败'):
                production._assert_disk_reserve(project_root, 0.95, 6)

    def test_worker_skips_completed_stages_and_checkpoints_failures(self):
        from pyqcd.testing.tmd.run_full_production import run_worker

        calls = []

        def executor(stage, conf_id, _context):
            calls.append((stage, conf_id))
            return {
                'stage': stage,
                'conf_id': conf_id,
                'seconds': 0.01,
                'artifacts': ['/bin/true'],
            }

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            first = run_worker(
                root,
                [10, 20],
                backend='numpy',
                device='cpu',
                stages=('flow',),
                stage_executor=executor,
                resource_checker=lambda: {
                    'cpu_percent': 1.0,
                    'memory_percent': 1.0,
                    'disk_percent': 1.0,
                    'gpu_memory_percent': None,
                },
                logger=lambda *_args: None,
            )
            self.assertEqual(calls, [('flow', 10), ('flow', 20)])
            self.assertEqual(first['status'], 'complete')
            checkpoint = json.loads(
                (root / 'workers' / 'cpu.json').read_text(encoding='utf-8'))
            self.assertEqual(
                checkpoint['configurations']['10']['stages']['flow']['status'],
                'complete',
            )

            calls.clear()
            resumed = run_worker(
                root,
                [10, 20],
                backend='numpy',
                device='cpu',
                stages=('flow',),
                stage_executor=executor,
                resource_checker=lambda: {
                    'cpu_percent': 1.0,
                    'memory_percent': 1.0,
                    'disk_percent': 1.0,
                    'gpu_memory_percent': None,
                },
                logger=lambda *_args: None,
            )
            self.assertEqual(calls, [])
            self.assertEqual(resumed['status'], 'complete')

            def failing_executor(stage, conf_id, _context):
                raise RuntimeError(f'{stage} failed for {conf_id}')

            failure_root = root / 'failure'
            with self.assertRaisesRegex(RuntimeError, 'ope failed for 30'):
                run_worker(
                    failure_root,
                    [30],
                    backend='numpy',
                    device='cpu',
                    stages=('ope',),
                    stage_executor=failing_executor,
                    resource_checker=lambda: {
                        'cpu_percent': 1.0,
                        'memory_percent': 1.0,
                        'disk_percent': 1.0,
                        'gpu_memory_percent': None,
                    },
                    logger=lambda *_args: None,
                )
            failed = json.loads(
                (failure_root / 'workers' / 'cpu.json').read_text(
                    encoding='utf-8'))
            self.assertEqual(failed['status'], 'failed')
            self.assertEqual(
                failed['configurations']['30']['stages']['ope']['status'],
                'failed',
            )

    def test_production_defaults_cover_requested_physical_grid(self):
        from pyqcd.testing.tmd import run_full_production as production

        self.assertEqual(
            production.CONF_IDS,
            (6250, 6450, 6650, 6850, 7050, 7250,
             7450, 7650, 7850, 8050, 8250, 8450),
        )
        self.assertEqual(production.FLOW_TIMES, (1.0, 2.0, 3.0))
        self.assertEqual(production.Z_LIST, tuple(range(13)))
        self.assertEqual(production.B_LIST, tuple(range(7)))
        self.assertEqual(production.B_X_LIST, tuple(range(7)))
        self.assertEqual(production.B_Y_LIST, tuple(range(3)))
        self.assertEqual(production.MOMENTA_AXES[0], (0, 0, 0))
        self.assertIn((2, 0, 0), production.MOMENTA_AXES)
        self.assertIn((0, 2, 0), production.MOMENTA_AXES)
        self.assertIn((0, 0, 2), production.MOMENTA_AXES)
        self.assertIn((4, 0, 0), production.MOMENTA_AXES)

    def test_cli_exposes_independent_disk_limit(self):
        from pyqcd.testing.tmd.run_full_production import parse_args

        args = parse_args([
            '--run-dir', '/tmp/tmd-run',
            '--max-resource-fraction', '0.8',
            '--max-disk-fraction', '0.95',
        ])
        self.assertEqual(args.max_resource_fraction, 0.8)
        self.assertEqual(args.max_disk_fraction, 0.95)

    def test_worker_checkpoint_rejects_changed_request_contract(self):
        from pyqcd.testing.tmd.run_full_production import run_worker

        resources = lambda: {
            'cpu_percent': 1.0,
            'memory_percent': 1.0,
            'disk_percent': 1.0,
            'gpu_memory_percent': None,
        }
        with tempfile.TemporaryDirectory() as root:
            run_worker(
                root,
                [10],
                backend='numpy',
                device='cpu',
                stages=('flow',),
                b_list=(0, 1),
                stage_executor=lambda *_args: {},
                resource_checker=resources,
                logger=lambda *_args: None,
            )
            with self.assertRaisesRegex(ValueError, 'request contract'):
                run_worker(
                    root,
                    [10],
                    backend='numpy',
                    device='cpu',
                    stages=('flow',),
                    b_list=(0, 1, 2),
                    stage_executor=lambda *_args: {},
                    resource_checker=resources,
                    logger=lambda *_args: None,
                )

    def test_resumed_worker_refreshes_active_process_owner(self):
        from pyqcd.testing.tmd.run_full_production import run_worker

        resources = lambda: {
            'cpu_percent': 1.0,
            'memory_percent': 1.0,
            'disk_percent': 1.0,
            'gpu_memory_percent': None,
        }
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                run_worker(
                    root,
                    [10],
                    backend='numpy',
                    device='cpu',
                    stages=('flow',),
                    max_disk_fraction=0.95,
                    stage_executor=lambda *_args: (_ for _ in ()).throw(
                        RuntimeError('interrupted')),
                    resource_checker=resources,
                    logger=lambda *_args: None,
                )
            checkpoint = root / 'workers' / 'cpu.json'
            payload = json.loads(checkpoint.read_text(encoding='utf-8'))
            payload['pid'] = None
            payload['hostname'] = None
            checkpoint.write_text(
                json.dumps(payload), encoding='utf-8')

            observed = {}

            def executor(stage, conf_id, _context):
                active = json.loads(checkpoint.read_text(encoding='utf-8'))
                observed.update(active)
                return {'stage': stage, 'conf_id': conf_id}

            run_worker(
                root,
                [10],
                backend='numpy',
                device='cpu',
                stages=('flow',),
                max_disk_fraction=0.95,
                stage_executor=executor,
                resource_checker=resources,
                logger=lambda *_args: None,
            )
            self.assertEqual(observed['status'], 'running')
            self.assertEqual(observed['pid'], __import__('os').getpid())
            self.assertTrue(observed['hostname'])
            self.assertEqual(observed['max_disk_fraction'], 0.95)
            self.assertEqual(
                observed['request_contract']['max_disk_fraction'], 0.95)

    def test_flow_history_uses_requested_flow_times(self):
        from pyqcd.testing.tmd import run_full_production as production

        snapshots = iter([0.0, 1.0, 2.0, 3.0])

        def fake_action(_gauge):
            import numpy as np
            return np.asarray(float(next(snapshots)))

        class FakeGauge:
            shape = (2, 2, 2, 2, 4, 3, 3)
            dtype = 'complex128'

        with patch.object(production, '_read_raw_gauge', return_value=FakeGauge()), \
                patch.object(production, 'wilson_flow_step',
                             side_effect=lambda gauge, _eps: gauge), \
                patch.object(production, 'flow_action_density',
                             side_effect=fake_action):
            result = production.compute_flow_history(
                6250, tau_values=(1.0, 2.0, 3.0), eps=1.0,
                backend='numpy', precision='complex128',
            )
        self.assertEqual(result['flow_time'].tolist(), [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(result['tau_checkpoints'].tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(
            result['checkpoint_action'].tolist(), [1.0, 2.0, 3.0])

    def test_flow_history_npz_publish_is_atomic(self):
        from pyqcd.testing.tmd.run_full_production import _save_flow_history

        with tempfile.TemporaryDirectory() as root:
            path = _save_flow_history(
                root, 6250, {'flow_time': np.arange(2)})
            self.assertTrue(path.is_file())
            self.assertFalse(list(path.parent.glob('*.tmp.npz')))


if __name__ == '__main__':
    unittest.main()
