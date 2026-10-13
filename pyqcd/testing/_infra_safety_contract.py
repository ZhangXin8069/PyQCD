"""Infrastructure safety contracts for eigcompression, ASCII I/O, and MPI."""

import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pyqcd.tools import read_data_ascii, set_backend, write_data_ascii
from pyqcd.vertex._eigcompress import (
    check_orthonormal,
    compress_matrix_V4,
)
from pyqcd.testing import _mpi_failure_contract as mpi_contract


class AsciiCounterSafetyContract(unittest.TestCase):
    """The external time counter is data, not an ignorable label."""

    def setUp(self):
        set_backend('numpy')
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.data = (
            np.arange(6.0).reshape(2, 3, 1)
            + 1j * (100.0 + np.arange(6.0).reshape(2, 3, 1))
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_valid_round_trip_is_unchanged(self):
        path = self.root / 'valid.dat'
        write_data_ascii(
            self.data, T=3, L=4, filename=str(path), is_complex=True,
        )

        restored, meta = read_data_ascii(str(path))

        self.assertEqual(meta['nsamples'], 2)
        self.assertEqual(meta['T'], 3)
        self.assertTrue(meta['is_complex'])
        np.testing.assert_array_equal(
            restored[..., 0] + 1j * restored[..., 1],
            self.data[..., 0])

    def test_corrupt_time_counter_is_rejected(self):
        path = self.root / 'bad-counter.dat'
        write_data_ascii(
            self.data, T=3, L=4, filename=str(path), is_complex=True,
        )
        lines = path.read_text().splitlines()
        fields = lines[1].split()
        fields[0] = '1'
        lines[1] = ' '.join(fields)
        path.write_text('\n'.join(lines) + '\n')

        with self.assertRaisesRegex(ValueError, 'counter'):
            read_data_ascii(str(path))


class EigcompressSchemeSafetyContract(unittest.TestCase):
    """Legacy noise modes must not claim orthonormal compression semantics."""

    def setUp(self):
        set_backend('numpy')
        self.vectors = np.eye(
            8, dtype=np.complex128).reshape(8, 2, 2, 2, 1)

    def _compress(self, random_type, check=None):
        return compress_matrix_V4(
            self.vectors, 2, 2, Ctype='B', seed=1,
            random_type=random_type, check=check,
        )

    def test_literal_ZN_placeholder_fails_closed(self):
        with self.assertRaisesRegex(ValueError, r"Z_N.*Z_2.*Z_8"):
            self._compress('Z_N')

    def test_legacy_scheme_enum_rejects_out_of_range_values(self):
        for scheme in ('Z_0', 'Z_1', 'Z_9', 'Z_foo'):
            with self.subTest(scheme=scheme):
                with self.assertRaisesRegex(ValueError, r"Z_2.*Z_8"):
                    self._compress(scheme)

    def test_legacy_z4_output_is_not_marked_orthonormal(self):
        result = self._compress('Z_4')

        self.assertFalse(check_orthonormal(result))
        with self.assertRaises(RuntimeError):
            self._compress('Z_4', check=True)


class MpiContractTimingSafetyContract(unittest.TestCase):
    """Timeout and contract failure must remain distinct outcomes."""

    def test_timeout_is_reported_as_timeout(self):
        expired = subprocess.TimeoutExpired(
            cmd='mpirun', timeout=0.01,
            output='partial stdout', stderr='partial stderr',
        )
        with patch.object(
                mpi_contract.subprocess, 'run', side_effect=expired):
            with self.assertRaisesRegex(
                    AssertionError, r'^MPI plan-log TIMEOUT'):
                mpi_contract._run_scenario_process(
                    'plan-log', timeout=0.01)

    def test_nonzero_exit_is_reported_as_contract_failure(self):
        failed = subprocess.CompletedProcess(
            args=['mpirun'], returncode=9,
            stdout='PASS plan-log: misleading partial success',
            stderr='contract stderr',
        )
        with patch.object(
                mpi_contract.subprocess, 'run', return_value=failed):
            with self.assertRaisesRegex(
                    AssertionError,
                    r'^MPI plan-log CONTRACT FAILURE rc=9'):
                mpi_contract._run_scenario_process('plan-log')

    @unittest.skipIf(importlib.util.find_spec('mpi4py') is None,
                     'mpi4py unavailable')
    @unittest.skipIf(shutil.which('mpirun') is None, 'mpirun unavailable')
    def test_setup_scenario_completes_within_bounded_wait(self):
        completed, output = mpi_contract._run_scenario_process(
            'setup', timeout=15.0)

        self.assertEqual(completed.returncode, 0)
        self.assertIn('PASS setup:', output)


if __name__ == '__main__':
    unittest.main(verbosity=2)
