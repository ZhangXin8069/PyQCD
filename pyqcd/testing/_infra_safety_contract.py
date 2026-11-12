"""Infrastructure safety contracts for eigcompression and ASCII I/O."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from pyqcd.tools import read_data_ascii, set_backend, write_data_ascii
from pyqcd.vertex._eigcompress import (
    check_orthonormal,
    compress_matrix_V4,
)


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


if __name__ == '__main__':
    unittest.main(verbosity=2)
