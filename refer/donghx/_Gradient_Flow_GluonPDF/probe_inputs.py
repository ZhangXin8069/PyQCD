"""Read scientific file headers on CPU, before freezing the production bundle."""
import json
import os
import struct
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
ENSEMBLE = 'beta6.20_mu-0.2770_ms-0.2400_L24x72'
TAG = 'e50_s[100, 200, 300, 500, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n500_V1_stout_smear_20_0.12'


def header(path):
    st = path.stat()
    with path.open('rb') as stream:
        version = np.lib.format.read_magic(stream)
        shape, order, dtype = (np.lib.format.read_array_header_1_0(stream) if version == (1, 0)
                               else np.lib.format.read_array_header_2_0(stream))
        offset = stream.tell()
    count = 1
    for n in shape:
        count *= n
    assert not order and not dtype.hasobject and st.st_size == offset + count * dtype.itemsize
    return dict(path=str(path), shape=list(shape), dtype=str(dtype), bytes=st.st_size)


def main():
    assert 'SLURM_JOB_ID' in os.environ
    confs = list(map(str, range(10000, 20000, 200)))
    result = dict(job=os.environ['SLURM_JOB_ID'], status='PASS', configurations=[])
    for conf in confs:
        try:
            eigen = Path('/nexdata/project/lqcd/sush/eigensystem') / ENSEMBLE / conf
            peram = Path('/nexdata/project/lqcd/sush/perambulators') / ENSEMBLE / 'light' / conf
            gauge = Path('/public/group/lqcd/configurations/CLOVER') / ENSEMBLE / f'{ENSEMBLE}_cfg_{conf}.lime'
            # Production uses all required endpoint times; the probe checks every time.
            files = [header(eigen / f'{conf}_t{t:03d}_{TAG}.npy') for t in range(72)]
            files += [header(peram / f't{t:03d}_{TAG}.npy') for t in range(72)]
            with gauge.open('rb') as stream:
                while True:
                    raw = stream.read(144)
                    assert len(raw) == 144
                    magic, version, flags, size, kind = struct.unpack('>IHHQ128s', raw)
                    assert magic == 0x456789AB
                    if kind.split(b'\0', 1)[0] == b'ildg-binary-data':
                        assert size == 72 * 24**3 * 4 * 3 * 3 * 2 * 8
                        assert stream.tell() + size <= gauge.stat().st_size
                        break
                    stream.seek((size + 7) // 8 * 8, 1)
            item = dict(conf=conf, status='PASS', eigen_headers=files[:72], peram_headers=files[72:],
                        gauge=str(gauge), payload_bytes=size)
        except Exception as error:
            item = dict(conf=conf, status='FAIL', error=repr(error))
            result['status'] = 'FAIL'
        result['configurations'].append(item)
        print(conf, item['status'], item.get('error', ''), flush=True)
    out = ROOT / 'status/input_probe.json'
    with out.open('w') as stream:
        json.dump(result, stream, indent=2)
    if result['status'] != 'PASS':
        raise RuntimeError('Missing/incompatible inputs; inspect status/input_probe.json')
    print('HEADER_PROBE_PASS', len(confs), flush=True)


if __name__ == '__main__':
    main()
